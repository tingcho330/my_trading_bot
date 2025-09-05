# src/recorder.py
import os
import sqlite3
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Optional
from contextlib import contextmanager

# ── 공통 유틸 ──
from utils import setup_logging, OUTPUT_DIR, KST

# ── 디스코드 노티파이어 ──
from notifier import (
    DiscordLogHandler,
    WEBHOOK_URL,
    is_valid_webhook,
    send_discord_message,
    create_trade_embed,  # 있으면 사용, 없으면 텍스트만 전송
)

# ───────────────── 로깅/경로 초기화 ─────────────────
setup_logging()
logger = logging.getLogger("recorder")

# 루트 로거에 디스코드 에러 핸들러 장착(중복 방지)
_root = logging.getLogger()
if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

# ───────────────── 알림 쿨다운 ─────────────────
_last_sent: Dict[str, float] = {}
def _notify(msg: str, key: str, cooldown_sec: int = 120):
    if not (WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL)):
        return
    now = time.time()
    if key not in _last_sent or now - _last_sent[key] >= cooldown_sec:
        _last_sent[key] = now
        try:
            send_discord_message(content=msg)
        except Exception:
            # 알림 실패는 기능에 영향 없도록 무시
            pass

def _notify_embed_safe(embed: Dict, key: str, cooldown_sec: int = 120):
    if not (WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL)):
        return
    now = time.time()
    if key not in _last_sent or now - _last_sent[key] >= cooldown_sec:
        _last_sent[key] = now
        try:
            send_discord_message(embeds=[embed])
        except Exception:
            pass

# ───────────────── DB 경로 ─────────────────
DB_PATH = str((OUTPUT_DIR / "trading_log.db").resolve())
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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
        # 심각 오류 알림(쿨다운)
        _notify(f" DB 트랜잭션 실패: {str(e)[:900]}", key="recorder_db_tx_fail", cooldown_sec=300)
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
            gpt_analysis    TEXT,
            gpt_score       REAL,
            sell_reason     TEXT,
            reason_code     TEXT,
            levels_source   TEXT
        )
        """
    )

    # 2) 컬럼 누락 시 추가 (안전 보강)
    REQUIRED_COLS = {
        "trades": {
            "gpt_summary": "TEXT",
            "gpt_analysis": "TEXT",
            "strategy_details": "TEXT",
            "trade_status": "TEXT",
            "pnl_amount": "REAL",
            "gpt_score": "REAL",
            "sell_reason": "TEXT",
            "reason_code": "TEXT",
            "levels_source": "TEXT",
            "parent_trade_id": "INTEGER",
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
        _notify(f" DB 초기화 실패: {str(e)[:900]}", key="recorder_db_init_fail", cooldown_sec=600)
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
        if isinstance(gpt_analysis_data, (dict, list))
        else gpt_analysis_data
    )

    # gpt_analysis에서 score 추출(가능 시)
    gpt_score = None
    if isinstance(gpt_analysis_data, dict):
        stock_info = gpt_analysis_data.get("stock_info", {})
        if "Score" in stock_info:
            gpt_score = stock_info.get("Score")
        elif "Overall Score" in gpt_analysis_data:
            gpt_score = gpt_analysis_data.get("Overall Score")

    sql = """
        INSERT INTO trades (
            timestamp, side, ticker, name, qty, price,
            pnl_amount, parent_trade_id, strategy_name, trade_status,
            strategy_details, gpt_summary, gpt_analysis,
            gpt_score, sell_reason, reason_code, levels_source
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """

    ts = datetime.now(KST).isoformat()
    params = (
        ts,
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
        if trade_data.get("strategy_details") is not None
        else None,
        trade_data.get("gpt_summary"),
        gpt_analysis_json,
        gpt_score,
        trade_data.get("sell_reason"),
        trade_data.get("reason_code"),
        trade_data.get("levels_source"),
    )

    try:
        with get_db_connection(db_path) as conn:
            conn.execute(sql, params)
            conn.commit()
        # 로그 & (선택) 간단 알림
        s = (trade_data.get("side") or "").upper()
        name = trade_data.get("name")
        qty = trade_data.get("qty")
        logger.info(f"✅ 거래 기록: {s} {name} {qty}주 (ts={ts})")

        # 임베드 알림(가능 시)
        try:
            embed = create_trade_embed({
                "side": s,
                "name": name,
                "ticker": trade_data.get("ticker"),
                "qty": trade_data.get("qty"),
                "price": trade_data.get("price"),
                "trade_status": trade_data.get("trade_status"),
                "reason_code": trade_data.get("reason_code"),
                "strategy_details": trade_data.get("strategy_details"),
            })
            _notify_embed_safe(embed, key=f"recorder_insert_{s}_{trade_data.get('ticker','')}", cooldown_sec=60)
        except Exception:
            # 템플릿 실패 → 텍스트로 최소 알림(쿨다운)
            _notify(
                f" 거래 기록: {s} {name} x{qty} @ {trade_data.get('price')}",
                key=f"recorder_insert_text_{trade_data.get('ticker','')}",
                cooldown_sec=60
            )

    except Exception as e:
        logger.error(f"DB 거래 기록 실패: {e}", exc_info=True)
        _notify(f" DB 거래 기록 실패: {str(e)[:900]}", key="recorder_record_fail", cooldown_sec=300)

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
                    try:
                        trade["strategy_details"] = json.loads(trade["strategy_details"])
                    except Exception:
                        pass
                active_trades.append(trade)
    except Exception as e:
        logger.error(f"Active 거래 조회 실패: {e}")
        _notify(f" Active 거래 조회 실패: {str(e)[:900]}", key="recorder_fetch_active_fail", cooldown_sec=300)
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

        _notify(f" 거래 상태 변경: id={trade_id}, status={new_status}",
                key=f"recorder_status_{trade_id}_{new_status}", cooldown_sec=120)

    except Exception as e:
        logger.error(f"거래 상태 업데이트 실패 (ID={trade_id}): {e}")
        _notify(f" 거래 상태 업데이트 실패 (ID={trade_id}): {str(e)[:900]}",
                key="recorder_update_status_fail", cooldown_sec=300)

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
        _notify(f" 티커별 거래 조회 실패: {str(e)[:900]}", key="recorder_fetch_tickers_fail", cooldown_sec=300)

    return trades_map

# ───────────────── 단독 실행 테스트 ─────────────────
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
            "NewsSentiment": "positive",
            "stock_info": {"Score": 0.82}
        },
        "reason_code": None,
        "levels_source": None,
    }

    sample_sell_trade = {
        "side": "sell",
        "ticker": "000660",
        "name": "SK하이닉스",
        "qty": 5,
        "price": 150000,
        "pnl_amount": 50000,               # 5만원 수익
        "parent_trade_id": 1,              # 매수 거래 ID
        "strategy_name": "RsiReversalStrategy",
        "trade_status": "completed",
        "strategy_details": {"RSI": 75, "reason": "RSI 과매수 구간 진입"},
        "sell_reason": "RSI 과매수 구간 진입",
        "reason_code": "RSI_OVERBOUGHT",
        "levels_source": "atr_swing",
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
