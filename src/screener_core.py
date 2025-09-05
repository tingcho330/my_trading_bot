# src/screener_core.py
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock as pykrx

logger = logging.getLogger("screener_core")

# ─────────── 내부 유틸 ───────────
def _yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

# ── KRX 호가단위(간략 규칙) ──
def _krx_tick_size(price: float) -> int:
    """단순화된 코스피/코스닥 호가단위. 필요 시 정밀 규칙으로 교체."""
    p = float(price)
    if p < 1000: return 1
    if p < 5000: return 5
    if p < 10000: return 10
    if p < 50000: return 50
    if p < 100000: return 100
    if p < 500000: return 500
    return 1000

def _round_to_tick(px: float, use_tick: bool = True) -> int:
    if not use_tick:
        return int(round(float(px)))
    tick = _krx_tick_size(px)
    return int(round(float(px) / tick) * tick)

def _is_finite_number(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

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
            # FDR은 이미 Open/High/Low/Close/Volume 형태
            need_cols = {"Open","High","Low","Close","Volume"}
            missing = need_cols.difference(df.columns)
            if missing:
                return None
            return df
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
            need_cols = {"Open","High","Low","Close","Volume"}
            if need_cols.issubset(df.columns):
                return df
    except Exception:
        pass
    return None

# ─────────── RSI ───────────
def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    if loss.empty or pd.isna(loss.iloc[-1]) or float(loss.iloc[-1]) == 0.0:
        return 50.0
    rs = float(gain.iloc[-1]) / float(loss.iloc[-1])
    if not _is_finite_number(rs):
        return 50.0
    return float(100 - (100 / (1 + rs)))

# ─────────── ATR (일관성/안전) ───────────
def calculate_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """DataFrame으로 ATR(Average True Range)을 계산합니다. 유효 수치만 반환."""
    if df is None or len(df) < period + 1:
        return None
    try:
        # 컬럼 정규화
        cols = {c.lower(): c for c in df.columns}
        high = pd.to_numeric(df[cols.get("high", "High")], errors="coerce")
        low = pd.to_numeric(df[cols.get("low", "Low")], errors="coerce")
        close = pd.to_numeric(df[cols.get("close", "Close")], errors="coerce")
        # 결측 최소화
        if high.dropna().empty or low.dropna().empty or close.dropna().empty:
            return None

        prev_close = close.shift(1)
        tr = pd.DataFrame({
            "HL": (high - low).abs(),
            "HPC": (high - prev_close).abs(),
            "LPC": (low - prev_close).abs()
        })
        TR = tr.max(axis=1)

        ATR = TR.rolling(window=period, min_periods=period).mean()
        if ATR is None or ATR.dropna().empty:
            return None
        val = float(ATR.dropna().iloc[-1])
        return val if _is_finite_number(val) and val > 0 else None
    except Exception as e:
        logger.debug("calculate_atr 실패: %s", e)
        return None

# ─────────── 손절/목표가 (ATR·스윙, 백업 포함) ───────────
def _compute_levels(ticker: str, entry_price: float, date_str: str, risk_params: dict) -> Dict:
    """
    ATR·스윙 저점 기반 손절/목표가 산출, 실패 시 퍼센트 백업.
    returns: {"손절가": int, "목표가": int, "source": "atr_swing"|"percent_backup"}
    - 원 단위 정수 보장 + (옵션) KRX 호가단위 반영
    """
    def _percent_backup() -> Dict:
        stop_pct = float(risk_params.get("stop_pct", 0.03))
        rr = float(risk_params.get("reward_risk", 2.0))
        use_tick = bool(risk_params.get("use_tick_rounding", True))

        stop_px = entry_price * (1.0 - stop_pct)
        risk = max(1e-6, entry_price - stop_px)
        tgt_px = entry_price + rr * risk

        return {
            "손절가": _round_to_tick(stop_px, use_tick),
            "목표가": _round_to_tick(tgt_px, use_tick),
            "source": "percent_backup",
        }

    try:
        atr_period = int(risk_params.get("atr_period", 14))
        atr_k_stop = float(risk_params.get("atr_k_stop", 1.5))
        swing_lookback = int(risk_params.get("swing_lookback", 20))
        rr = float(risk_params.get("reward_risk", 2.0))
        stop_pct = float(risk_params.get("stop_pct", 0.03))
        use_tick = bool(risk_params.get("use_tick_rounding", True))

        end_dt = datetime.strptime(date_str, "%Y%m%d")
        start_dt = end_dt - timedelta(days=max(atr_period * 6, 180))
        df = _safe_fetch_prices(ticker, _yyyymmdd(start_dt), _yyyymmdd(end_dt))
        if df is None or len(df) < max(atr_period + 5, swing_lookback + 5):
            return _percent_backup()

        # ATR 계산
        atr = calculate_atr(df, period=atr_period)
        if not _is_finite_number(atr) or atr is None:
            return _percent_backup()

        # 스윙 저점
        low_col = [c for c in df.columns if c.lower() == "low"]
        if not low_col:
            return _percent_backup()
        swing_low = float(pd.to_numeric(df[low_col[0]], errors="coerce").tail(swing_lookback).min())
        if not _is_finite_number(swing_low):
            return _percent_backup()

        # 손절/목표 계산
        stop_atr = float(entry_price) - atr_k_stop * float(atr)
        stop_px = max(stop_atr, swing_low)

        # 역전 방지 → 퍼센트 백업
        if stop_px >= float(entry_price):
            stop_px = float(entry_price) * (1.0 - stop_pct)

        risk_amt = max(1e-6, float(entry_price) - float(stop_px))
        tgt_px = float(entry_price) + rr * risk_amt

        return {
            "손절가": _round_to_tick(stop_px, use_tick),
            "목표가": _round_to_tick(tgt_px, use_tick),
            "source": "atr_swing",
        }
    except Exception as e:
        logger.debug("_compute_levels 예외: %s", e)
        # 파라미터/데이터 이슈 시 안전 백업
        return _percent_backup()
