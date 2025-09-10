# src/reviewer.py
import os
import json
import logging
import sqlite3
import time
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

# --- 프로젝트 공통 유틸리티 및 모듈 임포트 ---
from utils import (
    setup_logging,
    load_config,
    OUTPUT_DIR,
    KST,
    find_latest_file,
    get_account_snapshot_cached,  # ← 최신 스냅샷 utils 경유
    _to_int_krw,                  # named import ok (not via *)
)
from notifier import send_discord_message, is_valid_webhook

# --- 로깅 설정 ---
setup_logging()
logger = logging.getLogger("Reviewer")

# --- 상수 정의 ---
DB_PATH = OUTPUT_DIR / "trading_log.db"
REVIEW_LOG_PATH = OUTPUT_DIR / "review_log.json"
ROTATION_LOG_PATH = OUTPUT_DIR / "rotation_log.json"
TRADE_CONTEXT_LOG_PATH = OUTPUT_DIR / "trade_context_log.json"  # 거래 컨텍스트 로그
CONFIG_PATH = Path("/app/config/config.json")  # 설정 파일 경로 명시
ANALYSIS_CSV_PATH = OUTPUT_DIR / "completed_trades_analysis.csv"

# --- 자동 튜닝을 위한 안전장치 및 상수 정의 ---
MIN_STOP_LOSS_PCT = 0.015   # 손절 하한(1.5%)
MAX_TAKE_PROFIT_PCT = 0.50  # 익절 상한(50%)
CONSECUTIVE_FAILURES_THRESHOLD = 3  # 연속 부진 임계

# --- 사이클 컨텍스트 (scheduler가 env로 주입) ---
RUN_ID = os.getenv("RUN_ID", "")
RUN_STARTED_AT = float(os.getenv("RUN_STARTED_AT", "0") or 0)
RUN_SUCCESS = os.getenv("RUN_SUCCESS", "").lower() == "true"


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
def append_json(file_path: Path, record: dict):
    """JSON 파일에 레코드를 리스트 형태로 추가합니다."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]  # 기존 dict 를 list로 변환
        else:
            data = []
        data.append(record)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to append to JSON file {file_path}: {e}")

def record_trade_context(ctx: dict):
    """
    거래 당시의 근거/지표를 함께 저장하여 사후 성과 분석에 활용
    """
    rec = {
        "ts": datetime.now().isoformat(),
        "ticker": ctx["ticker"],
        "action": ctx["action"],  # BUY/SELL/SKIP/ROTATION
        "score_total": ctx.get("score_total"),
        "score_components": ctx.get("score_components"),
        "vol_kki": ctx.get("vol_kki"),
        "pos_52w": ctx.get("pos_52w"),
        "exclude_reasons": ctx.get("exclude_reasons", []),
        "gpt_rationale": ctx.get("gpt_rationale",""),
        "reason_code": ctx.get("reason_code",""),
        "rotation": ctx.get("rotation", None)  # {from_ticker,to_ticker,delta}
    }
    append_json(TRADE_CONTEXT_LOG_PATH, rec)

def get_last_review_info() -> dict:
    """마지막 리뷰 기록(ID, 날짜, 연속 실패 횟수)을 로드합니다."""
    if not REVIEW_LOG_PATH.exists():
        return {"last_trade_id": 0, "consecutive_failures": 0}
    try:
        with open(REVIEW_LOG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"last_trade_id": 0, "consecutive_failures": 0}
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
# DB 로드: 미검토 거래 가져오기
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
      - profit_factor: 이익합 / 손실합(절댓값)
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
# 자동 튜닝 (서킷 브레이커 포함)
# ─────────────────────────────────────────────────────────────────────────
def tune_parameters(performance_metrics: dict, last_review_info: dict, *, apply_changes: bool = False) -> int:
    """
    성과 지표에 따라 config.json의 전략 파라미터를 동적으로 조정합니다.
    성과 부진 정의: win_rate < 0.5 또는 profit_factor < 1.5
    - apply_changes=False(기본): 드라이런(변경사항 로그만 출력)
    - apply_changes=True: 실제 파일에 반영
    반환값: 업데이트된 consecutive_failures
    """
    config = load_config()
    strategy_params = config.get("strategy_params", {}) or {}

    win_rate = float(performance_metrics.get('win_rate', 0.0))
    profit_factor = float(performance_metrics.get('profit_factor', 0.0))

    config_changed = False
    planned_changes = {}
    is_poor_performance = (win_rate < 0.5) or (profit_factor < 1.5)
    consecutive_failures = int(last_review_info.get("consecutive_failures", 0))

    if is_poor_performance:
        consecutive_failures += 1
        logger.warning(f"성과 부진 감지. 연속 실패 횟수: {consecutive_failures}")
        if consecutive_failures >= CONSECUTIVE_FAILURES_THRESHOLD:
            logger.critical(
                f"[경고] {consecutive_failures}회 연속으로 성과가 부진합니다. "
                f"자동 파라미터 튜닝을 중단합니다. 전략/리스크 규칙 점검 필요."
            )
            return consecutive_failures
    else:
        consecutive_failures = 0

    # 승률 저하 시: 손절 강화(5% 강화, Floor 적용)
    if win_rate < 0.5:
        original = float(strategy_params.get('stop_loss_pct', 0.05))
        new_value = max(round(original * 0.95, 4), MIN_STOP_LOSS_PCT)
        if not np.isclose(new_value, original):
            planned_changes['stop_loss_pct'] = (original, new_value)
            if apply_changes:
                strategy_params['stop_loss_pct'] = new_value
                config_changed = True

    # 손익비 저하 시: 익절 상향(5% 상향, Ceiling 적용)
    if profit_factor < 1.5:
        original = float(strategy_params.get('take_profit_pct', 0.10))
        new_value = min(round(original * 1.05, 4), MAX_TAKE_PROFIT_PCT)
        if not np.isclose(new_value, original):
            planned_changes['take_profit_pct'] = (original, new_value)
            if apply_changes:
                strategy_params['take_profit_pct'] = new_value
                config_changed = True

    # 변경 계획 로그
    if planned_changes:
        msg_lines = ["[튜닝 계획]"]
        for k, (old, new) in planned_changes.items():
            msg_lines.append(f" - {k}: {old:.4f} → {new:.4f} ({'적용' if apply_changes else '드라이런'})")
        logger.info("\n".join(msg_lines))
    else:
        logger.info("튜닝 변경 계획 없음.")

    if apply_changes and config_changed:
        config['strategy_params'] = strategy_params
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"전략 파라미터가 수정되어 '{CONFIG_PATH}'에 저장되었습니다. {strategy_params}")

    return consecutive_failures


# ─────────────────────────────────────────────────────────────────────────
# CSV 내보내기 (가능한 필드 최대한 채움)
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
    if fifo_df.empty:
        return pd.DataFrame()

    groups = []
    for sell_id, g in fifo_df.groupby('sell_id'):
        sell_rows = raw_df[raw_df['id'] == sell_id]
        if sell_rows.empty:
            continue
        sell = sell_rows.iloc[0]

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

        pnl_amount = float(g['profit'].sum())
        sell_qty = float(sell.get('qty', np.nan))
        sell_price = float(sell.get('price', np.nan))
        denom = (avg_buy_price * sell_qty) if (sell_qty and avg_buy_price and not np.isnan(avg_buy_price)) else np.nan
        pnl_rate_pct = (pnl_amount / denom) * 100 if denom and denom != 0 and not np.isnan(denom) else np.nan

        gpt_score = float('nan')
        fin = tech = sector = None
        try:
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
    if detail_df is None or detail_df.empty:
        logger.info("CSV 내보내기: 저장할 완료 거래가 없습니다.")
        return

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
# 사이클 요약(요청 사양) — 단 1회 전송
# ─────────────────────────────────────────────────────────────────────────
def send_cycle_summary():
    try:
        start_dt = pd.to_datetime(RUN_STARTED_AT, unit='s') if RUN_STARTED_AT else None
        with sqlite3.connect(DB_PATH) as conn:
            if start_dt is not None:
                q = "SELECT side, COUNT(*) c FROM trades WHERE timestamp >= ? GROUP BY side"
                cnt = pd.read_sql_query(q, conn, params=[start_dt.isoformat()])
            else:
                q = "SELECT side, COUNT(*) c FROM trades GROUP BY side"
                cnt = pd.read_sql_query(q, conn)
        buy_n = int(cnt.loc[cnt['side'] == 'buy', 'c'].sum()) if not cnt.empty else 0
        sell_n = int(cnt.loc[cnt['side'] == 'sell', 'c'].sum()) if not cnt.empty else 0

        # 보류: 최신 gpt_trades 파일
        hold_n = 0
        p = find_latest_file("gpt_trades_*.json")
        if p:
            plans = json.loads(Path(p).read_text(encoding='utf-8'))
            hold_n = sum(1 for x in plans if x.get("결정") == "보류")

        # 총평가 Δ (2회 샘플) — utils 캐시 리더 사용
        s1, _, sp1, _ = get_account_snapshot_cached()
        time.sleep(2)  # ← 파일 갱신 지연 대비, 1초 → 2초로 확대
        s2, _, sp2, _ = get_account_snapshot_cached()
        def tv(s): return _to_int_krw(s.get('tot_evlu_amt', 0)) if s else 0
        delta = tv(s2) - tv(s1) if s1 and s2 else None

        color = 3066993 if RUN_SUCCESS else 16711680
        title = f"파이프라인 사이클 요약 (run_id={RUN_ID or 'N/A'})"
        fields = [
            {"name": "성공여부", "value": "성공" if RUN_SUCCESS else "실패", "inline": True},
            {"name": "소요시간", "value": (f"{(time.time() - RUN_STARTED_AT):.0f}s" if RUN_STARTED_AT else "N/A"), "inline": True},
            {"name": "매수/매도/보류", "value": f"{buy_n} / {sell_n} / {hold_n}", "inline": False},
            {"name": "총평가 Δ", "value": (f"{delta:+,}원" if delta is not None else "N/A"), "inline": True}
        ]
        embed = {"title": title, "color": color, "fields": fields, "footer": {"text": "AI Trading Bot"}}
        if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
            send_discord_message(embeds=[embed])
        logger.info(f"사이클 요약 전송 완료: run_id={RUN_ID}, 성공={RUN_SUCCESS}, 매수/매도/보류={buy_n}/{sell_n}/{hold_n}, Δ={delta}")
    except Exception as e:
        logger.warning(f"사이클 요약 카드 전송 실패: {e}")


# ─────────────────────────────────────────────────────────────────────────
# 파이프라인
# ─────────────────────────────────────────────────────────────────────────
def run_reviewer(min_trades_for_review: int = 5, export_csv: bool = False, *, tune_parameters_enabled: bool = False):
    """
    성과 분석 및 튜닝 전체 파이프라인 실행 (DB→FIFO→성과지표→튜닝)
    - tune_parameters_enabled=False(기본): 드라이런(설정 변경 없이 리포트/로그만)
    - tune_parameters_enabled=True : 실제 설정 파일에 전략 파라미터 반영
    (사이클 종료 시 요약 1회 전송)
    """
    logger.info("=" * 60)
    logger.info("▶ 매매 성과 분석(Reviewer)을 시작합니다. "
                f"[드라이런={'아니오' if tune_parameters_enabled else '예'}]")

    # ── 분석 수행(중간 디스코드 알림 없음)
    last_info = get_last_review_info()
    last_id = int(last_info.get('last_trade_id', 0))
    raw = fetch_new_trades(last_id)

    if not raw.empty:
        # 최소 레코드 체크
        if len(raw) >= 2:
            fifo = match_trades_fifo(raw)
            if not fifo.empty and fifo['sell_id'].nunique() >= int(min_trades_for_review):
                metrics = analyze_performance(fifo)
                if metrics:
                    consecutive_failures = tune_parameters(
                        metrics, last_info, apply_changes=bool(tune_parameters_enabled)
                    )
                    update_review_log(metrics['last_trade_id'], consecutive_failures)
                else:
                    update_review_log(int(raw['id'].max()), int(last_info.get("consecutive_failures", 0)))
            else:
                update_review_log(int(raw['id'].max()), int(last_info.get("consecutive_failures", 0)))
        else:
            update_review_log(int(raw['id'].max()), int(last_info.get("consecutive_failures", 0)))

        if export_csv:
            try:
                fifo = match_trades_fifo(raw)
                if not fifo.empty:
                    detail = build_completed_trades_detail(raw, fifo)
                    export_analysis_to_csv(detail)
            except Exception as e:
                logger.warning(f"CSV 내보내기 중 예외(무시): {e}")

    # ── 사이클 요약(단 1회 전송)
    send_cycle_summary()

    logger.info("✅ Reviewer 종료(사이클 요약 1회 전송).")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trading Performance Reviewer (FIFO Matching + Auto Tuner + CSV + Cycle Summary)"
    )
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
    # 드라이런 기본값 유지, --tune 주면 실제 적용
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tune",
        action="store_true",
        help="자동 튜닝 결과를 config.json에 실제로 반영합니다(기본: 드라이런)."
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="설정 변경 없이 리포트만 생성합니다(기본 동작)."
    )
    args = parser.parse_args()

    run_reviewer(
        min_trades_for_review=args.min_trades,
        export_csv=args.export_csv,
        tune_parameters_enabled=bool(args.tune)
    )
