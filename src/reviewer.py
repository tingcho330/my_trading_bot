# src/reviewer.py
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
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
CONFIG_PATH = Path("/app/config/config.json") # 설정 파일 경로 명시

# --- [개선 1] 자동 튜닝을 위한 안전장치 및 상수 정의 ---
MIN_STOP_LOSS_PCT = 0.015  # 최소 손절율 (1.5%) - 이 값 밑으로 내려가지 않음
MAX_TAKE_PROFIT_PCT = 0.50 # 최대 익절율 (50%) - 이 값 위로 올라가지 않음
CONSECUTIVE_FAILURES_THRESHOLD = 3 # 연속 실패 임계값 (3회 연속 성과 나쁘면 튜닝 중단)

def get_last_review_info() -> dict:
    """마지막 리뷰 기록(ID, 날짜, 연속 실패 횟수)을 로드합니다."""
    if not REVIEW_LOG_PATH.exists():
        return {"last_trade_id": 0, "consecutive_failures": 0}
    try:
        with open(REVIEW_LOG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 호환성을 위해 키가 없는 경우 기본값 0 설정
            if "consecutive_failures" not in data:
                data["consecutive_failures"] = 0
            return data
    except (json.JSONDecodeError, IOError):
        return {"last_trade_id": 0, "consecutive_failures": 0}

def update_review_log(last_trade_id: int, consecutive_failures: int):
    """리뷰 로그를 최신 정보로 업데이트합니다."""
    log_data = {
        "last_trade_id": last_trade_id,
        "last_review_date": datetime.now(KST).isoformat(),
        "consecutive_failures": consecutive_failures
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

def fetch_new_trades(last_trade_id: int) -> pd.DataFrame:
    """DB에서 마지막 리뷰 이후의 모든 거래 기록을 가져옵니다."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        query = f"SELECT * FROM trades WHERE id > {last_trade_id} ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
    
    for col in ['qty', 'price', 'pnl_amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def match_trades_fifo(df: pd.DataFrame) -> pd.DataFrame:
    """FIFO 원칙에 따라 매수-매도 거래를 매칭하고 수익률을 계산합니다."""
    buys = df[df['side'] == 'buy'].sort_values('timestamp').copy()
    sells = df[df['side'] == 'sell'].sort_values('timestamp').copy()
    
    buys['qty_remaining'] = buys['qty']
    completed_trades = []

    for _, sell in sells.iterrows():
        sell_qty_needed = sell['qty']
        potential_buys = buys[
            (buys['ticker'] == sell['ticker']) &
            (buys['timestamp'] < sell['timestamp']) &
            (buys['qty_remaining'] > 0)
        ].sort_values('timestamp')

        for buy_idx, buy in potential_buys.iterrows():
            if sell_qty_needed <= 0: break
            
            match_qty = min(sell_qty_needed, buy['qty_remaining'])
            profit = (sell['price'] - buy['price']) * match_qty
            
            completed_trades.append({
                'sell_id': sell['id'],
                'profit': profit,
            })
            
            sell_qty_needed -= match_qty
            buys.loc[buy_idx, 'qty_remaining'] -= match_qty

    return pd.DataFrame(completed_trades)

def analyze_performance(completed_df: pd.DataFrame) -> Dict:
    """성과 지표를 계산합니다."""
    if completed_df.empty: return {}

    trade_pnl = completed_df.groupby('sell_id')['profit'].sum()
    total_trades = len(trade_pnl)
    win_trades = (trade_pnl > 0).sum()
    
    return {
        "num_completed_trades": total_trades,
        "win_rate": win_trades / total_trades if total_trades > 0 else 0,
        "profit_factor": abs(trade_pnl[trade_pnl > 0].sum() / trade_pnl[trade_pnl <= 0].sum()) if (trade_pnl <= 0).any() and trade_pnl[trade_pnl > 0].any() else float('inf'),
        "last_trade_id": int(completed_df['sell_id'].max())
    }

def tune_parameters(performance_metrics: dict, last_review_info: dict) -> int:
    """성과 지표에 따라 config.json의 전략 파라미터를 동적으로 조정합니다."""
    config = load_config()
    strategy_params = config.get("strategy_params", {})
    
    win_rate = performance_metrics['win_rate']
    profit_factor = performance_metrics['profit_factor']
    
    config_changed = False
    is_poor_performance = False

    # 튜닝 조건 확인
    if win_rate < 0.5 or profit_factor < 1.5:
        is_poor_performance = True

    consecutive_failures = last_review_info.get("consecutive_failures", 0)

    if is_poor_performance:
        consecutive_failures += 1
        logger.warning(f"성과 부진 감지. 연속 실패 횟수: {consecutive_failures}")
        
        # [개선 2] 서킷 브레이커: 연속 실패 횟수가 임계값을 넘으면 튜닝을 중단하고 경고
        if consecutive_failures >= CONSECUTIVE_FAILURES_THRESHOLD:
            msg = (f"🚨 **[경고] {consecutive_failures}회 연속으로 성과가 부진합니다.**\n"
                   f"자동 파라미터 튜닝을 중단합니다. 종목 선정 로직 등 근본적인 전략 점검이 필요합니다.")
            logger.critical(msg)
            if is_valid_webhook(WEBHOOK_URL):
                send_discord_message(content=msg)
            return consecutive_failures # 변경 없이 실패 횟수만 반환

    else:
        consecutive_failures = 0 # 성과가 좋으면 실패 횟수 초기화

    # 승률이 낮을 때: 손절 기준 강화
    if win_rate < 0.5:
        original = strategy_params.get('stop_loss_pct', 0.05)
        new_value = original * 0.95 # 5%씩 감소
        # [개선 3] 최소 손절율(Stop-Loss Floor) 적용
        new_value = max(round(new_value, 4), MIN_STOP_LOSS_PCT)
        
        if new_value != original:
            strategy_params['stop_loss_pct'] = new_value
            logger.info(f"승률 저하({win_rate:.2%}) -> 손절 기준 강화: {original:.3f} -> {new_value:.3f}")
            config_changed = True

    # 손익비가 낮을 때: 익절 기준 상향
    if profit_factor < 1.5:
        original = strategy_params.get('take_profit_pct', 0.10)
        new_value = original * 1.05 # 5%씩 증가
        # [개선 4] 최대 익절율(Take-Profit Ceiling) 적용
        new_value = min(round(new_value, 4), MAX_TAKE_PROFIT_PCT)

        if new_value != original:
            strategy_params['take_profit_pct'] = new_value
            logger.info(f"손익비 저하({profit_factor:.2f}) -> 익절 기준 상향: {original:.3f} -> {new_value:.3f}")
            config_changed = True

    if config_changed:
        config['strategy_params'] = strategy_params
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"전략 파라미터가 수정되어 '{CONFIG_PATH}'에 저장되었습니다.")
        
    return consecutive_failures


def run_reviewer(min_trades_for_review: int = 5):
    """성과 분석 및 튜닝 전체 파이프라인 실행"""
    logger.info("="*50)
    logger.info("매매 성과 분석(Reviewer)을 시작합니다.")
    
    last_info = get_last_review_info()
    df = fetch_new_trades(last_info['last_trade_id'])
    
    if len(df) < 2:
        logger.info("새로운 거래 기록이 부족하여 분석을 건너뜁니다.")
        return

    completed_trades_df = match_trades_fifo(df)
    
    if completed_trades_df.empty or completed_trades_df['sell_id'].nunique() < min_trades_for_review:
        logger.info(f"완료된 거래가 {completed_trades_df['sell_id'].nunique()}건으로, 분석 최소 기준({min_trades_for_review}건)에 미치지 못해 튜닝을 건너뜁니다.")
        # 리뷰할 데이터는 없지만, 다음 번에 중복으로 읽지 않도록 마지막 거래 ID는 업데이트
        if not df.empty:
            update_review_log(int(df['id'].max()), last_info.get("consecutive_failures", 0))
        return
        
    metrics = analyze_performance(completed_trades_df)
    
    if metrics:
        consecutive_failures = tune_parameters(metrics, last_info)
        update_review_log(metrics['last_trade_id'], consecutive_failures)
        logger.info("성과 분석 및 자동 튜닝이 완료되었습니다.")
    else:
        logger.info("성과 분석 지표를 생성할 수 없어 튜닝을 건너뜁니다.")


if __name__ == "__main__":
    run_reviewer()