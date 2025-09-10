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

# ─────────── 공통: OHLCV 스키마 보정 ───────────
_REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df

def _ensure_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 컬럼 표준화: Open/High/Low/Close/Volume 보장 (없으면 생성: NaN/0 적절히)
    - dtype 강제: O/H/L/C → float, Volume → int (가능한 경우)
    - 결측은 드랍/보간하지 않고 그대로 둠
    """
    if df is None:
        return pd.DataFrame(columns=_REQUIRED_COLS)

    d = df.copy()
    # 소문자로 통일 후 매핑
    lower_map = {str(c).strip().lower(): c for c in d.columns}
    rename_map = {}
    if "시가" in lower_map:
        rename_map[lower_map["시가"]] = "Open"
    if "고가" in lower_map:
        rename_map[lower_map["고가"]] = "High"
    if "저가" in lower_map:
        rename_map[lower_map["저가"]] = "Low"
    if "종가" in lower_map:
        rename_map[lower_map["종가"]] = "Close"
    if "거래량" in lower_map:
        rename_map[lower_map["거래량"]] = "Volume"

    # FDR 형식(Open/High/Low/Close/Volume)은 그대로 사용
    # 위 매핑과 충돌 없도록 존재하는 표준 컬럼은 건드리지 않음
    d = d.rename(columns=rename_map)

    # 필요 컬럼 생성(없으면 NaN/0)
    for col in _REQUIRED_COLS:
        if col not in d.columns:
            d[col] = np.nan if col != "Volume" else 0

    # dtype 강제
    for col in ["Open", "High", "Low", "Close"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    # Volume은 정수형으로 가능하면 변환, 안 되면 float → 마지막에 NaN을 0으로
    d["Volume"] = pd.to_numeric(d["Volume"], errors="coerce").fillna(0)
    try:
        # 큰 값에서도 안전하게 int64
        d["Volume"] = d["Volume"].astype("int64")
    except Exception:
        # 불가능하면 float 유지
        pass

    # 인덱스 정규화
    d = _ensure_datetime_index(d)

    # 정렬 및 필요한 컬럼만 우선 배치(나머지는 보존)
    other_cols = [c for c in d.columns if c not in _REQUIRED_COLS]
    d = d[_REQUIRED_COLS + other_cols]
    return d

# ─────────── 시세 조회 ───────────
def get_historical_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    티커의 과거 OHLCV 조회(pykrx 우선, 실패/빈 → FDR 폴백).
    - 항상 DataFrame 반환 (실패 시 빈 DF)
    - 컬럼 보장: Open/High/Low/Close/Volume
    - dtype 보장: O/H/L/C=float, Volume=int(가능 시)
    - DEBUG 로그에 길이 출력
    """
    df_out = pd.DataFrame(columns=_REQUIRED_COLS)

    # 1차: pykrx
    try:
        df = pykrx.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if df is not None and not df.empty:
            df = df.rename(columns={
                "시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"
            })
            df_out = _ensure_ohlcv_schema(df)
            logger.debug("get_historical_prices(%s, pykrx) len=%d", ticker, len(df_out))
            return df_out
    except Exception as e:
        logger.debug("%s: pykrx 조회 실패(%s). fdr로 폴백 시도.", ticker, e)

    # 2차: FDR 폴백
    try:
        df = fdr.DataReader(ticker, start=start_date, end=end_date)
        if df is not None and not df.empty:
            df_out = _ensure_ohlcv_schema(df)
            logger.debug("get_historical_prices(%s, fdr) len=%d", ticker, len(df_out))
            return df_out
    except Exception as e:
        logger.debug("%s: fdr 조회 실패(%s). 빈 DF 반환.", ticker, e)

    # 모두 실패 → 빈 DF (None 금지)
    logger.debug("get_historical_prices(%s) len=0 (both sources failed)", ticker)
    return df_out

def _safe_fetch_prices(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """
    내부용 보조: 여러 번 재시도 후에도 실패하면 빈 DF 반환.
    호출부는 길이/스키마가 보장된 DF를 항상 받는다.
    """
    for _ in range(retries):
        df = get_historical_prices(ticker, start, end)
        if df is not None and not df.empty:
            return df
    # 마지막으로 한 번 더 시도(FDR 직접)
    try:
        df = fdr.DataReader(ticker, start=start, end=end)
        if df is not None and not df.empty:
            return _ensure_ohlcv_schema(df)
    except Exception:
        pass
    return pd.DataFrame(columns=_REQUIRED_COLS)

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
        # 컬럼 정규화(대소문자 혼용 방지)
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
