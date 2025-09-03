# src/reviewer.py
import os
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from collections import defaultdict

# --- 프로젝트 공통 유틸리티 및 모듈 임포트 ---
from utils import (
    setup_logging,
    load_config,
    OUTPUT_DIR,
    KST
)
from notifier import send_discord_message, is_valid_webhook

# --- 로깅 설정 ---
setup_logging()
logger = logging.getLogger("Reviewer")

# --- 상수 정의 ---
DB_PATH = OUTPUT_DIR / "trading_log.db"
REVIEW_LOG_PATH = OUTPUT_DIR / "review_log.json"
CONFIG_PATH = Path("/app/config/config.json")  # 설정 파일 경로 명시
ANALYSIS_CSV_PATH = OUTPUT_DIR / "completed_trades_analysis.csv"

# --- 자동 튜닝을 위한 안전장치 및 상수 정의 ---
MIN_STOP_LOSS_PCT = 0.015   # 손절 하한(1.5%)
MAX_TAKE_PROFIT_PCT = 0.50  # 익절 상한(50%)
CONSECUTIVE_FAILURES_THRESHOLD = 3  # 연속 부진 임계


# ─────────────────────────────────────────────────────────────────────────
# Webhook 로딩: config.json(notifier.discord_webhook) → ENV(DISCORD_WEBHOOK_URL)
# ─────────────────────────────────────────────────────────────────────────
def _load_webhook_url() -> str:
    try:
        cfg = load_config()
        webhook = (
            cfg.get("notifier", {}).get("discord_webhook")
            or os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        )
        return webhook or ""
    except Exception as e:
        logger.warning(f"웹훅 URL 로드 중 예외(무시): {e}")
        return ""


WEBHOOK_URL = _load_webhook_url()


# ─────────────────────────────────────────────────────────────────────────
# 리뷰 로그 I/O
# ─────────────────────────────────────────────────────────────────────────
def get_last_review_info() -> dict:
    """마지막 리뷰 기록(ID, 날짜, 연속 실패 횟수)을 로드합니다."""
    if not REVIEW_LOG_PATH.exists():
        return {"last_trade_id": 0, "consecutive_failures": 0}
    try:
        with open(REVIEW_LOG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "consecutive_failures" not in data:
                data["consecutive_failures"] = 0
            if "last_trade_id" not in data:
                data["last_trade_id"] = 0
            return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"리뷰 로그 파싱 실패(초기화): {e}")
        return {"last_trade_id": 0, "consecutive_failures": 0}


def update_review_log(last_trade_id: int, consecutive_failures: int):
    """리뷰 로그를 최신 정보로 업데이트합니다."""
    log_data = {
        "last_trade_id": int(last_trade_id),
        "last_review_date": datetime.now(KST).isoformat(),
        "consecutive_failures": int(consecutive_failures)
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    logger.info(
        f"[리뷰 로그 업데이트] last_trade_id={log_data['last_trade_id']}, "
        f"consecutive_failures={log_data['consecutive_failures']} "
        f"({log_data['last_review_date']})"
    )


# ─────────────────────────────────────────────────────────────────────────
# DB 로드: 미검토 거래 가져오기(일반화, 안전한 정규화/검증 포함)
# trades 스키마 가정 컬럼(가급적):
#   id, timestamp, ticker, name?, side('buy'|'sell'), qty, price,
#   pnl_amount?, sell_reason?, gpt_score?, gpt_analysis?, strategy_name?, parent_trade_id?
# ─────────────────────────────────────────────────────────────────────────
def fetch_new_trades(last_trade_id: int) -> pd.DataFrame:
    if not DB_PATH.exists():
        logger.warning(f"거래 DB 미존재: {DB_PATH}")
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT * FROM trades WHERE id > ? ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn, params=[int(last_trade_id)], parse_dates=['timestamp'])

    if df.empty:
        logger.info("새로운 거래가 없습니다.")
        return df

    # 타입 정규화
    for col in ['qty', 'price', 'pnl_amount', 'id', 'parent_trade_id', 'gpt_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # side/ticker/name 정규화
    if 'side' in df.columns:
        df['side'] = df['side'].astype(str).str.lower().str.strip()
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.strip()
    if 'name' in df.columns:
        df['name'] = df['name'].astype(str).str.strip()

    # 필수 필드 체크
    required_cols = {'id', 'timestamp', 'ticker', 'side', 'qty', 'price'}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"필수 컬럼 누락: {missing}")
        return pd.DataFrame()

    # 결측/이상치 제거
    before = len(df)
    df = df.dropna(subset=list(required_cols))
    df = df[(df['qty'] > 0) & (df['price'] > 0)]
    after = len(df)
    if after < before:
        logger.info(f"정규화에서 {before - after}건 필터링됨(결측/음수 제거).")

    return df


# ─────────────────────────────────────────────────────────────────────────
# FIFO 매칭: 매도별 실현손익 재계산
# 반환 컬럼: sell_id, profit
# ─────────────────────────────────────────────────────────────────────────
def match_trades_fifo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['sell_id', 'profit'])

    buys = df[df['side'] == 'buy'].sort_values('timestamp').copy()
    sells = df[df['side'] == 'sell'].sort_values('timestamp').copy()

    if buys.empty or sells.empty:
        logger.info("매수/매도 데이터가 부족하여 매칭 불가.")
        return pd.DataFrame(columns=['sell_id', 'profit'])

    buys['qty_remaining'] = buys['qty']
    completed = []

    for _, sell in sells.iterrows():
        sell_qty_needed = float(sell['qty'])
        potential_buys = buys[
            (buys['ticker'] == sell['ticker']) &
            (buys['timestamp'] < sell['timestamp']) &
            (buys['qty_remaining'] > 0)
        ].sort_values('timestamp')

        for buy_idx, buy in potential_buys.iterrows():
            if sell_qty_needed <= 0:
                break
            match_qty = float(min(sell_qty_needed, buy['qty_remaining']))
            profit = (float(sell['price']) - float(buy['price'])) * match_qty

            completed.append({
                'sell_id': int(sell['id']),
                'profit': float(profit),
                # 아래 값들은 CSV/분석 확장을 위한 힌트
                'sell_t': sell['timestamp'],
                'sell_price': float(sell['price']),
                'sell_qty': float(sell['qty']),
                'sell_ticker': sell['ticker'],
                'sell_name': str(sell['name']) if 'name' in sell else None,
                'sell_reason': str(sell['sell_reason']) if 'sell_reason' in sell else None,
                'buy_id': int(buy['id']),
                'buy_t': buy['timestamp'],
                'buy_price': float(buy['price']),
                'buy_qty': float(match_qty),
                'buy_strategy': str(buy['strategy_name']) if 'strategy_name' in buy else None,
                'buy_gpt_score': float(buy['gpt_score']) if 'gpt_score' in buy and pd.notna(buy['gpt_score']) else None,
                'buy_gpt_analysis': buy['gpt_analysis'] if 'gpt_analysis' in buy else None,
            })

            sell_qty_needed -= match_qty
            buys.at[buy_idx, 'qty_remaining'] = float(buy['qty_remaining'] - match_qty)

    if not completed:
        return pd.DataFrame(columns=['sell_id', 'profit'])

    return pd.DataFrame(completed)


# ─────────────────────────────────────────────────────────────────────────
# 성과 지표 계산 (FIFO 결과 기준)
# ─────────────────────────────────────────────────────────────────────────
def analyze_performance(completed_df: pd.DataFrame) -> Dict:
    """
    반환:
      - num_completed_trades: 체결 완료(매도 기준) 거래 수
      - win_rate: 승률
      - profit_factor: 이익합 / 손실합(절댓값) (손실이 0이면 inf)
      - last_trade_id: 마지막 매도 트랜잭션 ID
      - total_pnl: 전체 실현손익 합계
    """
    if completed_df.empty:
        return {}

    trade_pnl = completed_df.groupby('sell_id')['profit'].sum()
    total_trades = int(len(trade_pnl))
    if total_trades == 0:
        return {}

    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]

    win_trades = int((trade_pnl > 0).sum())
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0

    sum_pos = float(wins.sum()) if not wins.empty else 0.0
    sum_neg_abs = float(-losses.sum()) if not losses.empty else 0.0
    profit_factor = (sum_pos / sum_neg_abs) if sum_neg_abs > 0 else float('inf')

    metrics: Dict = {
        "num_completed_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "last_trade_id": int(completed_df['sell_id'].max()),
        "total_pnl": float(trade_pnl.sum()),
    }

    logger.info(
        f"[성과] trades={metrics['num_completed_trades']} | "
        f"win_rate={metrics['win_rate']:.2%} | "
        f"PF={metrics['profit_factor']:.2f} | "
        f"total_pnl={metrics['total_pnl']:.0f} | "
        f"last_sell_id={metrics['last_trade_id']}"
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────
# 자동 튜닝 (서킷 브레이커 포함) + 디스코드 알림
# ─────────────────────────────────────────────────────────────────────────
def tune_parameters(performance_metrics: dict, last_review_info: dict) -> int:
    """
    성과 지표에 따라 config.json의 전략 파라미터를 동적으로 조정합니다.
    성과 부진 정의: win_rate < 0.5 또는 profit_factor < 1.5
    """
    config = load_config()
    strategy_params = config.get("strategy_params", {}) or {}

    win_rate = float(performance_metrics.get('win_rate', 0.0))
    profit_factor = float(performance_metrics.get('profit_factor', 0.0))

    config_changed = False
    is_poor_performance = (win_rate < 0.5) or (profit_factor < 1.5)
    consecutive_failures = int(last_review_info.get("consecutive_failures", 0))

    if is_poor_performance:
        consecutive_failures += 1
        logger.warning(f"성과 부진 감지. 연속 실패 횟수: {consecutive_failures}")

        if consecutive_failures >= CONSECUTIVE_FAILURES_THRESHOLD:
            msg = (
                f"🚨 **[경고] {consecutive_failures}회 연속으로 성과가 부진합니다.**\n"
                f"자동 파라미터 튜닝을 중단합니다. 전략/리스크 규칙의 근본 점검이 필요합니다."
            )
            logger.critical(msg)
            if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
                send_discord_message(content=msg)
            return consecutive_failures
    else:
        consecutive_failures = 0  # 성과가 좋으면 실패 횟수 초기화

    # 승률 저하 시: 손절 강화(5% 강화, Floor 적용)
    if win_rate < 0.5:
        original = float(strategy_params.get('stop_loss_pct', 0.05))
        new_value = max(round(original * 0.95, 4), MIN_STOP_LOSS_PCT)
        if not np.isclose(new_value, original):
            strategy_params['stop_loss_pct'] = new_value
            logger.info(f"승률 저하({win_rate:.2%}) -> 손절 강화: {original:.3f} -> {new_value:.3f}")
            config_changed = True

    # 손익비 저하 시: 익절 상향(5% 상향, Ceiling 적용)
    if profit_factor < 1.5:
        original = float(strategy_params.get('take_profit_pct', 0.10))
        new_value = min(round(original * 1.05, 4), MAX_TAKE_PROFIT_PCT)
        if not np.isclose(new_value, original):
            strategy_params['take_profit_pct'] = new_value
            logger.info(f"손익비 저하({profit_factor:.2f}) -> 익절 상향: {original:.3f} -> {new_value:.3f}")
            config_changed = True

    if config_changed:
        config['strategy_params'] = strategy_params
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"전략 파라미터가 수정되어 '{CONFIG_PATH}'에 저장되었습니다. {strategy_params}")

        if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
            summary = (
                f"[Reviewer] 자동 튜닝 적용\n"
                f"- win_rate: {win_rate:.2%}, PF: {profit_factor:.2f}\n"
                f"- stop_loss_pct: {strategy_params.get('stop_loss_pct')}\n"
                f"- take_profit_pct: {strategy_params.get('take_profit_pct')}"
            )
            send_discord_message(content=summary)

    return consecutive_failures


# ─────────────────────────────────────────────────────────────────────────
# CSV 내보내기 (가능한 필드 최대한 채움: GPT 점수도 추출 시도)
# ─────────────────────────────────────────────────────────────────────────
def _safe_get_gpt_scores(gpt_json) -> Dict:
    out = {"FinScore": None, "TechScore": None, "SectorScore": None}
    try:
        data = json.loads(str(gpt_json))
        stock_info = data.get('stock_info', {})
        out["FinScore"] = stock_info.get('FinScore', data.get('FinScore'))
        out["TechScore"] = stock_info.get('TechScore', data.get('TechScore'))
        out["SectorScore"] = stock_info.get('SectorScore', data.get('SectorScore'))
    except Exception:
        pass
    return out


def build_completed_trades_detail(raw_df: pd.DataFrame, fifo_df: pd.DataFrame) -> pd.DataFrame:
    """
    raw_df: fetch_new_trades로 읽은 원본 거래
    fifo_df: match_trades_fifo 결과(매도별 매칭 조각)
    반환: 매도건(sell_id) 단위의 요약 레코드 집합(가능한 필드 채움)
    """
    if fifo_df.empty:
        return pd.DataFrame()

    # sell_id 단위 집계
    groups = []
    for sell_id, g in fifo_df.groupby('sell_id'):
        # sell 행(원본 DF에서)
        sell_rows = raw_df[raw_df['id'] == sell_id]
        if sell_rows.empty:
            # sell 레코드가 안 보이면 스킵
            continue
        sell = sell_rows.iloc[0]

        # 매칭된 buy들의 가중 평균 매입가, 최초 buy 시각
        total_qty = float(g['buy_qty'].sum()) if 'buy_qty' in g.columns else float(sell.get('qty', 0))
        if total_qty <= 0:
            total_qty = float(sell.get('qty', 0))

        if total_qty > 0 and 'buy_price' in g.columns and 'buy_qty' in g.columns:
            avg_buy_price = float((g['buy_price'] * g['buy_qty']).sum() / g['buy_qty'].sum())
        else:
            avg_buy_price = float('nan')

        first_buy_t = pd.to_datetime(g['buy_t'].min()) if 'buy_t' in g.columns else None
        sell_t = pd.to_datetime(sell['timestamp'])
        holding_days = (sell_t - first_buy_t).days if (first_buy_t is not None and pd.notna(first_buy_t)) else None

        # 실현손익 합계(해당 sell_id)
        pnl_amount = float(g['profit'].sum())
        sell_qty = float(sell.get('qty', np.nan))
        sell_price = float(sell.get('price', np.nan))
        denom = (avg_buy_price * sell_qty) if (sell_qty and avg_buy_price and not np.isnan(avg_buy_price)) else np.nan
        pnl_rate_pct = (pnl_amount / denom) * 100 if denom and denom != 0 and not np.isnan(denom) else np.nan

        # GPT 점수 추출(가능 시)
        gpt_score = float('nan')
        fin = tech = sector = None
        try:
            # 매칭된 buy 중 gpt_score/gpt_analysis가 있는 첫 항목 사용
            cand = g.dropna(subset=['buy_gpt_analysis']) if 'buy_gpt_analysis' in g.columns else pd.DataFrame()
            if not cand.empty:
                scores = _safe_get_gpt_scores(cand.iloc[0]['buy_gpt_analysis'])
                fin, tech, sector = scores["FinScore"], scores["TechScore"], scores["SectorScore"]
            if 'buy_gpt_score' in g.columns and pd.notna(g['buy_gpt_score']).any():
                gpt_score = float(g['buy_gpt_score'].dropna().iloc[0])
        except Exception:
            pass

        groups.append({
            'sell_id': int(sell_id),
            'ticker': str(sell.get('ticker', '')),
            'name': str(sell.get('name', '')) if 'name' in sell else '',
            'buy_timestamp': first_buy_t,
            'sell_timestamp': sell_t,
            'holding_days': int(holding_days) if holding_days is not None else None,
            'buy_price': float(round(avg_buy_price, 4)) if not np.isnan(avg_buy_price) else None,
            'sell_price': float(round(sell_price, 4)) if not np.isnan(sell_price) else None,
            'sell_qty': float(round(sell_qty, 4)) if not np.isnan(sell_qty) else None,
            'pnl_amount': float(round(pnl_amount, 4)),
            'pnl_rate_pct': float(round(pnl_rate_pct, 4)) if not (pnl_rate_pct is None or np.isnan(pnl_rate_pct)) else None,
            'sell_reason': str(sell.get('sell_reason', '')) if 'sell_reason' in sell else '',
            'buy_strategy': str(sell.get('strategy_name', '')) if 'strategy_name' in sell else '',
            'gpt_score': float(round(gpt_score, 4)) if not (gpt_score is None or np.isnan(gpt_score)) else None,
            'FinScore': fin,
            'TechScore': tech,
            'SectorScore': sector,
        })

    if not groups:
        return pd.DataFrame()

    out = pd.DataFrame(groups).sort_values('sell_timestamp')
    return out


def export_analysis_to_csv(detail_df: pd.DataFrame):
    """분석 데이터를 CSV 파일로 저장합니다."""
    if detail_df is None or detail_df.empty:
        logger.info("CSV 내보내기: 저장할 완료 거래가 없습니다.")
        return

    # 저장 컬럼 우선순위
    final_cols = [
        'sell_id', 'ticker', 'name',
        'buy_timestamp', 'sell_timestamp', 'holding_days',
        'buy_price', 'sell_price', 'sell_qty',
        'pnl_amount', 'pnl_rate_pct',
        'sell_reason', 'buy_strategy',
        'gpt_score', 'FinScore', 'TechScore', 'SectorScore'
    ]
    use_cols = [c for c in final_cols if c in detail_df.columns]
    df_final = detail_df[use_cols].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(ANALYSIS_CSV_PATH, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 분석 리포트({len(df_final)}건)를 '{ANALYSIS_CSV_PATH}' 파일로 저장했습니다.")


# ─────────────────────────────────────────────────────────────────────────
# 파이프라인
# ─────────────────────────────────────────────────────────────────────────
def run_reviewer(min_trades_for_review: int = 5, export_csv: bool = False):
    """
    성과 분석 및 튜닝 전체 파이프라인 실행
    - 최근 미검토 거래를 읽고, FIFO로 체결 쌍을 매칭
    - 최소 N건 이상이면 성과 계산 및 자동 튜닝
    - 리뷰 로그(last_trade_id, consecutive_failures) 갱신
    - 옵션에 따라 CSV 저장
    """
    logger.info("=" * 60)
    logger.info("▶ 매매 성과 분석(Reviewer)을 시작합니다.")

    last_info = get_last_review_info()
    last_id = int(last_info.get('last_trade_id', 0))
    logger.info(f"마지막 리뷰 ID: {last_id}")

    raw = fetch_new_trades(last_id)
    if raw.empty:
        logger.info("새로운 거래가 없어 종료합니다.")
        return

    # 최소 거래 수(전체 raw 기준) 체크: 너무 적으면 튜닝/CSV 미수행
    if len(raw) < 2:
        logger.info("새로운 거래 기록이 부족하여 분석을 건너뜁니다.")
        update_review_log(int(raw['id'].max()), int(last_info.get("consecutive_failures", 0)))
        return

    fifo = match_trades_fifo(raw)

    if fifo.empty:
        logger.info("완료된(매수-매도 매칭된) 거래가 없어 튜닝을 건너뜁니다.")
        update_review_log(int(raw['id'].max()), int(last_info.get("consecutive_failures", 0)))
        return

    # 매도건(독립 sell_id) 기준으로 최소 분석건 체크
    completed_sells = int(fifo['sell_id'].nunique())
    if completed_sells < int(min_trades_for_review):
        logger.info(
            f"완료된 거래가 {completed_sells}건으로, 분석 최소 기준({min_trades_for_review}건)에 미달. 튜닝/CSV 건너뜀."
        )
        update_review_log(int(raw['id'].max()), int(last_info.get("consecutive_failures", 0)))
        return

    # 성과 계산
    metrics = analyze_performance(fifo)
    if metrics:
        consecutive_failures = tune_parameters(metrics, last_info)
        update_review_log(metrics['last_trade_id'], consecutive_failures)
        logger.info("✅ 성과 분석 및 자동 튜닝 완료.")
    else:
        logger.info("성과 지표 생성 실패. 튜닝을 건너뜁니다.")
        update_review_log(int(raw['id'].max()), int(last_info.get("consecutive_failures", 0)))

    # CSV 저장 옵션
    if export_csv:
        detail = build_completed_trades_detail(raw, fifo)
        export_analysis_to_csv(detail)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Performance Reviewer (FIFO Matching + Auto Tuner + CSV)")
    parser.add_argument(
        "--min-trades",
        type=int,
        default=5,
        help="자동 튜닝 및 CSV 저장을 실행하기 위한 최소 '완료(매도)' 거래 수 (기본값: 5)"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="분석 결과를 completed_trades_analysis.csv 파일로 저장합니다."
    )
    args = parser.parse_args()

    run_reviewer(min_trades_for_review=args.min_trades, export_csv=args.export_csv)
