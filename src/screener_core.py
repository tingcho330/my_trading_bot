# src/screener_core.py
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock as pykrx

logger = logging.getLogger("screener_core")

# ─────────── 내부 유틸 ───────────
def _yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

def _round_px(x: float) -> int:
    return int(round(float(x)))

# ─────────── 시세 조회 ───────────
def get_historical_prices(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """티커의 과거 OHLCV 조회(pykrx 우선, 실패 시 FDR 백업). 컬럼: Open/High/Low/Close/Volume"""
    try:
        df = pykrx.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if df is not None and not df.empty:
            return df.rename(columns={
                '시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close', '거래량': 'Volume'
            })
    except Exception as e:
        logger.debug("%s: pykrx 시세 조회 실패(%s). fdr로 전환.", ticker, e)
    try:
        df = fdr.DataReader(ticker, start=start_date, end=end_date)
        if df is not None and not df.empty:
            return df  # FDR은 이미 Open/High/Low/Close/Volume 형태
    except Exception as e:
        logger.debug("%s: fdr 시세 조회도 실패(%s).", ticker, e)
    return None

def _safe_fetch_prices(ticker: str, start: str, end: str, retries: int = 3) -> Optional[pd.DataFrame]:
    for _ in range(retries):
        df = get_historical_prices(ticker, start, end)
        if df is not None and not df.empty:
            return df
    # 마지막으로 한 번 더 시도(fdr)
    try:
        df = fdr.DataReader(ticker, start=start, end=end)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None

# ─────────── RSI ───────────
def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    if pd.isna(loss.iloc[-1]) or loss.iloc[-1] == 0:
        return 50.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return float(100 - (100 / (1 + rs)))

# ─────────── 손절/목표가 (ATR·스윙) ───────────
def _compute_levels(ticker: str, entry_price: float, date_str: str, risk_params: dict) -> Dict:
    """
    ATR·스윙 저점 기반 손절/목표가 산출, 실패 시 퍼센트 백업.
    returns: {"손절가": int, "목표가": int, "source": "atr_swing"|"percent_backup"}
    """
    def _percent_backup() -> Dict:
        stop_px = entry_price * (1.0 - stop_pct)
        risk = max(1e-6, entry_price - stop_px)
        tgt_px = entry_price + rr * risk
        return {"손절가": _round_px(stop_px), "목표가": _round_px(tgt_px), "source": "percent_backup"}

    try:
        atr_period = int(risk_params.get("atr_period", 14))
        atr_k_stop = float(risk_params.get("atr_k_stop", 1.5))
        swing_lookback = int(risk_params.get("swing_lookback", 20))
        rr = float(risk_params.get("reward_risk", 2.0))
        stop_pct = float(risk_params.get("stop_pct", 0.03))

        end_dt = datetime.strptime(date_str, "%Y%m%d")
        start_dt = end_dt - timedelta(days=max(atr_period * 6, 180))
        df = _safe_fetch_prices(ticker, _yyyymmdd(start_dt), _yyyymmdd(end_dt))
        if df is None or len(df) < max(atr_period + 5, swing_lookback + 5):
            return _percent_backup()

        # 컬럼 정규화
        cols = {c.lower(): c for c in df.columns}
        high = df[cols.get("high", "High")].astype(float)
        low = df[cols.get("low", "Low")].astype(float)
        close = df[cols.get("close", "Close")].astype(float)

        prev_close = close.shift(1)
        tr = (high - low).abs().to_frame("TR")
        tr["H-PC"] = (high - prev_close).abs()
        tr["L-PC"] = (low - prev_close).abs()
        TR = tr.max(axis=1)
        ATR = TR.rolling(window=atr_period, min_periods=atr_period).mean()
        if ATR.dropna().empty:
            return _percent_backup()
        atr = float(ATR.dropna().iloc[-1])

        # 스윙 저점
        swing_low = float(low.tail(swing_lookback).min())

        # 손절/목표
        stop_atr = entry_price - atr_k_stop * atr
        stop_px = max(stop_atr, swing_low)
        if stop_px >= entry_price:  # 역전 방지 → 퍼센트 백업
            stop_px = entry_price * (1.0 - stop_pct)
        risk = max(1e-6, entry_price - stop_px)
        tgt_px = entry_price + rr * risk

        return {"손절가": _round_px(stop_px), "목표가": _round_px(tgt_px), "source": "atr_swing"}
    except Exception:
        # 파라미터/데이터 이슈 시 안전 백업
        rr = float(risk_params.get("reward_risk", 2.0))
        stop_pct = float(risk_params.get("stop_pct", 0.03))
        stop_px = entry_price * (1.0 - stop_pct)
        risk = max(1e-6, entry_price - stop_px)
        tgt_px = entry_price + rr * risk
        return {"손절가": _round_px(stop_px), "목표가": _round_px(tgt_px), "source": "percent_backup"}
