# src/recorder.py

import os
import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional
from contextlib import contextmanager

# --- DB 경로 및 초기화 ------------------------------------------------
DB_PATH = "/app/output/trading_log.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
@contextmanager
def get_db_connection(db_path: Optional[str] = None):
    """
    안전한 DB 커넥션 컨텍스트 매니저.
    자동 commit/rollback/close.
    """
    final_db_path = db_path or DB_PATH
    conn = None
    try:
        conn = sqlite3.connect(final_db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        logger.debug(f"DB Connection ↗ {final_db_path}")
        yield conn
    except sqlite3.Error as e:
        logger.error(f"DB 트랜잭션 실패: {e}", exc_info=True)
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
            logger.debug(f"DB Connection ↘ {final_db_path}")

# --------------------------------------------------------------------
def _get_existing_columns(cursor: sqlite3.Cursor, table_name: str) -> Set[str]:
    """테이블에 존재하는 컬럼명 집합 반환"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}

# --------------------------------------------------------------------
def _migrate_db_schema(conn: sqlite3.Connection):
    """
    ① trades 테이블이 없으면 생성
    ② 필수 컬럼이 빠져 있으면 ALTER TABLE 로 추가
    """
    cursor = conn.cursor()

    # 1) 테이블 최초 생성 (필요 시)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            side            TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            name            TEXT,
            qty             INTEGER NOT NULL,
            price           REAL NOT NULL,
            pnl_amount      REAL,
            parent_trade_id INTEGER,
            strategy_name   TEXT,
            trade_status    TEXT,
            strategy_details TEXT,
            gpt_summary     TEXT,
            gpt_analysis    TEXT
        )
    """
    )

    # 2) 컬럼 누락 시 추가
    REQUIRED_COLS = {
        "trades": {
            "gpt_summary": "TEXT",
            "gpt_analysis": "TEXT",
            "strategy_details": "TEXT",
            "trade_status": "TEXT",
            "pnl_amount": "REAL",
        }
    }

    for tbl, col_map in REQUIRED_COLS.items():
        existing = _get_existing_columns(cursor, tbl)
        for col, coltype in col_map.items():
            if col not in existing:
                logger.info(f"마이그레이션: {tbl}.{col} 컬럼 추가")
                cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {coltype}")

    conn.commit()

# --------------------------------------------------------------------
def initialize_db(db_path: Optional[str] = None):
    """DB 초기화 및 스키마 검증/마이그레이션"""
    try:
        with get_db_connection(db_path) as conn:
            _migrate_db_schema(conn)
        logger.info("✅ DB 스키마 확인 & 마이그레이션 완료")
    except Exception as e:
        logger.critical(f"DB 초기화 실패: {e}", exc_info=True)
        raise

# --------------------------------------------------------------------
# 데이터 기록 및 조회
# --------------------------------------------------------------------
def record_trade(trade_data: dict, db_path: Optional[str] = None):
    """거래 데이터를 trades 테이블에 INSERT"""
    if not trade_data:
        return

    gpt_analysis_data = trade_data.get("gpt_analysis")
    gpt_analysis_json = (
        json.dumps(gpt_analysis_data, ensure_ascii=False)
        if isinstance(gpt_analysis_data, dict)
        else gpt_analysis_data
    )

    sql = """
        INSERT INTO trades (
            timestamp, side, ticker, name, qty, price,
            pnl_amount, parent_trade_id, strategy_name, trade_status,
            strategy_details, gpt_summary, gpt_analysis
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """

    params = (
        datetime.now().isoformat(),
        trade_data.get("side"),
        trade_data.get("ticker"),
        trade_data.get("name"),
        int(trade_data.get("qty", 0)),
        float(trade_data.get("price", 0.0)),
        trade_data.get("pnl_amount"),
        trade_data.get("parent_trade_id"),
        trade_data.get("strategy_name"),
        trade_data.get("trade_status"),
        json.dumps(trade_data.get("strategy_details"), ensure_ascii=False)
        if trade_data.get("strategy_details")
        else None,
        trade_data.get("gpt_summary"),
        gpt_analysis_json,
    )

    try:
        with get_db_connection(db_path) as conn:
            conn.execute(sql, params)
            conn.commit()
        logger.info(
            f"✅ 거래 기록: {trade_data.get('side').upper()} {trade_data.get('name')} {trade_data.get('qty')}주"
        )
    except Exception as e:
        logger.error(f"DB 거래 기록 실패: {e}", exc_info=True)


def fetch_active_trades(db_path: Optional[str] = None) -> List[Dict]:
    """trade_status='active' 인 거래 반환"""
    active_trades = []
    try:
        with get_db_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE trade_status = 'active'"
            ).fetchall()
            for row in rows:
                trade = dict(row)
                if trade.get("strategy_details"):
                    trade["strategy_details"] = json.loads(trade["strategy_details"])
                active_trades.append(trade)
    except Exception as e:
        logger.error(f"Active 거래 조회 실패: {e}")
    return active_trades


def update_trade_status(
    trade_id: int,
    new_status: str,
    new_details: Dict = None,
    db_path: Optional[str] = None,
):
    """거래 상태(예: active→completed) 업데이트"""
    details_json = json.dumps(new_details, ensure_ascii=False) if new_details else None
    sql = "UPDATE trades SET trade_status = ?, strategy_details = ? WHERE id = ?"
    try:
        with get_db_connection(db_path) as conn:
            conn.execute(sql, (new_status, details_json, trade_id))
            conn.commit()
        logger.info(f"거래 ID {trade_id} → 상태 '{new_status}' 업데이트")
    except Exception as e:
        logger.error(f"거래 상태 업데이트 실패 (ID={trade_id}): {e}")


def fetch_trades_by_tickers(
    tickers: List[str], db_path: Optional[str] = None
) -> Dict[str, Dict]:
    """티커별 마지막 매수 거래 정보를 dict 로 반환"""
    if not tickers:
        return {}

    trades_map = {}
    placeholders = ",".join("?" for _ in tickers)
    query = f"""
        SELECT * FROM trades
        WHERE id IN (
            SELECT MAX(id) FROM trades
            WHERE side = 'buy' AND ticker IN ({placeholders})
            GROUP BY ticker
        )
    """
    try:
        with get_db_connection(db_path) as conn:
            rows = conn.execute(query, tickers).fetchall()
            for row in rows:
                trades_map[row["ticker"]] = dict(row)
    except Exception as e:
        logger.error(f"티커별 거래 조회 실패: {e}")

    return trades_map

if __name__ == '__main__':
    # 데이터베이스 초기화 (테이블 생성 및 마이그레이션)
    initialize_db()

    # --- 테스트용 샘플 데이터 ---
    sample_buy_trade = {
        "side": "buy",
        "ticker": "005930",
        "name": "삼성전자",
        "qty": 10,
        "price": 70000,
        "strategy_name": "TrendFollowingStrategy",
        "trade_status": "active",
        "gpt_summary": "강력한 기술적 지표와 긍정적인 뉴스 흐름에 기반한 매수 결정",
        "gpt_analysis": {
            "Overall Score": 0.85,
            "FinScore": 0.7,
            "TechScore": 0.9,
            "NewsSentiment": "positive"
        }
    }

    sample_sell_trade = {
        "side": "sell",
        "ticker": "000660",
        "name": "SK하이닉스",
        "qty": 5,
        "price": 150000,
        "pnl_amount": 50000, # 5만원 수익
        "parent_trade_id": 1, # 매수 거래 ID
        "strategy_name": "RsiReversalStrategy",
        "trade_status": "completed",
        "strategy_details": {"RSI": 75, "reason": "RSI 과매수 구간 진입"},
    }
    
    # 1. 매수 기록 테스트
    record_trade(sample_buy_trade)

    # 2. 현재 active 상태인 거래 조회
    print("\n--- Active Trades ---")
    active = fetch_active_trades()
    print(json.dumps(active, indent=2, ensure_ascii=False))

    # 3. 매도 기록 테스트
    record_trade(sample_sell_trade)

    # 4. 티커로 거래 조회 테스트
    print("\n--- Fetch by Tickers ---")
    trades = fetch_trades_by_tickers(["005930"])
    print(json.dumps(trades, indent=2, ensure_ascii=False))
    
    # 5. 거래 상태 업데이트 테스트
    if active:
        update_trade_status(active[0]['id'], "completed")
        print("\n--- Active Trades After Update ---")
        active_after_update = fetch_active_trades()
        print(json.dumps(active_after_update, indent=2, ensure_ascii=False))