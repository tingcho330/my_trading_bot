# src/screener.py
import os
import json
import logging
import argparse
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock as pykrx

# ───────────────── utils ─────────────────
from utils import (
    setup_logging,
    load_config,
    OUTPUT_DIR,
    CACHE_DIR,
    cache_load,
    cache_save,
    find_latest_file,
    is_market_open_day,
)
from api.kis_auth import KIS

# ───────────────── notifier ─────────────────
from notifier import (
    DiscordLogHandler,
    WEBHOOK_URL,
    is_valid_webhook,
    send_discord_message,
)

# ───────────────── 기본 설정/로깅 ─────────────────
setup_logging()
logger = logging.getLogger("screener")
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# 루트 로거에 디스코드 에러 핸들러 부착(중복 방지)
_root = logging.getLogger()
if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

# ── 간단 쿨다운(스팸 방지) ──
_last_sent = {}
def _notify(content: str, key: str, cooldown_sec: int = 120):
    now = time.time()
    if key not in _last_sent or now - _last_sent[key] >= cooldown_sec:
        _last_sent[key] = now
        send_discord_message(content=content)

@contextmanager
def stage(name: str, notify_key: Optional[str] = None):
    t0 = time.perf_counter()
    logger.info("▶ %s 시작", name)
    if notify_key:
        _notify(f"▶ {name} 시작", key=f"{notify_key}_start", cooldown_sec=60)
    try:
        yield
    finally:
        secs = time.perf_counter() - t0
        logger.info("⏱ %s 완료 (%.2fs)", name, secs)
        if notify_key:
            _notify(f"⏱ {name} 완료 ({secs:.1f}s)", key=f"{notify_key}_done", cooldown_sec=60)

# ───────────────── 유틸 함수 (로컬) ─────────────────
def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _describe_series(name: str, s: pd.Series):
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    if s_num.empty:
        logger.info("[%s] 값 없음", name)
        return
    qs = s_num.quantile([0.5, 0.75, 0.9, 0.95]).to_dict()
    logger.info("[%s] 중앙값=%s, P75=%s, P90=%s, P95=%s, 최대=%s",
                name,
                f"{int(qs.get(0.5, 0)):,}", f"{int(qs.get(0.75, 0)):,}",
                f"{int(qs.get(0.9, 0)):,}", f"{int(qs.get(0.95, 0)):,}",
                f"{int(s_num.max()):,}")

# ─────────── 손절/목표가 계산 로직 ───────────
def _yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

def _round_px(x: float) -> int:
    return int(round(float(x)))

def get_historical_prices(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """티커의 과거 OHLCV 조회(pykrx 우선, 실패 시 FDR 백업)"""
    try:
        df = pykrx.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if df is not None and not df.empty:
            return df.rename(columns={
                '시가':'Open','고가':'High','저가':'Low','종가':'Close','거래량':'Volume'
            })
    except Exception as e:
        logger.debug("%s: pykrx 시세 조회 실패(%s). fdr로 전환.", ticker, e)
    try:
        df = fdr.DataReader(ticker, start=start_date, end=end_date)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.debug("%s: fdr 시세 조회도 실패(%s).", ticker, e)
    return None

def _safe_fetch_prices(ticker: str, start: str, end: str, retries: int = 3):
    for _ in range(retries):
        df = get_historical_prices(ticker, start, end)
        if df is not None and not df.empty:
            return df
        time.sleep(0.5)
    try:
        df = fdr.DataReader(ticker, start=start, end=end)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None

def _compute_levels(ticker: str, entry_price: float, date_str: str, risk_params: dict) -> Dict:
    """ATR·스윙 저점 기반 손절/목표가 산출, 실패 시 퍼센트 백업."""
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

        swing_low = float(low.tail(swing_lookback).min())
        stop_atr = entry_price - atr_k_stop * atr
        stop_px = max(stop_atr, swing_low)
        if stop_px >= entry_price:
            stop_px = entry_price * (1.0 - stop_pct)
        risk = max(1e-6, entry_price - stop_px)
        tgt_px = entry_price + rr * risk

        return {"손절가": _round_px(stop_px), "목표가": _round_px(tgt_px), "source": "atr_swing"}
    except Exception:
        return _percent_backup()

# ─────────── 데이터/지표 함수 ───────────
def get_stock_listing(market: str = "KOSPI") -> pd.DataFrame:
    logger.info("종목 목록 조회(FDR): market=%s", market)
    df = fdr.StockListing(market)
    if 'Code' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'Code'})
    df = df.set_index('Code')
    df = _norm_code_index(df)
    df = df.rename(columns={"MarketCap": "Marcap", "종목명": "Name"}, errors="ignore")
    return df

def _norm_code_index(obj: pd.DataFrame) -> pd.DataFrame:
    if obj is None or obj.empty:
        return obj
    try:
        idx = obj.index.astype(str).str.replace(r"[^0-9]", "", regex=True).str.zfill(6)
        obj = obj.copy()
        obj.index = idx
    except Exception:
        pass
    return obj

def get_fundamentals(date_str: str, market: str = "KOSPI") -> pd.DataFrame:
    logger.info("펀더멘털 조회(PYKRX): date=%s market=%s", date_str, market)
    try:
        df = pykrx.get_market_fundamental_by_ticker(date_str, market=market)
        return _norm_code_index(df)
    except Exception as e:
        logger.error("펀더멘털 조회 실패 (%s, %s): %s", date_str, market, e)
        return pd.DataFrame()

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    if pd.isna(loss.iloc[-1]) or loss.iloc[-1] == 0:
        return 50.0
    return 100 - (100 / (1 + (gain.iloc[-1] / loss.iloc[-1])))

def get_market_trend(date_str: str) -> str:
    current_date = datetime.strptime(date_str, "%Y%m%d")
    start = (current_date - timedelta(days=60)).strftime("%Y-%m-%d")
    end = current_date.strftime("%Y-%m-%d")
    try:
        df = fdr.DataReader("KS11", start, end)
        if len(df) < 20:
            return "Sideways"
        df["MA5"], df["MA20"] = df["Close"].rolling(5).mean(), df["Close"].rolling(20).mean()
        latest = df.iloc[-1]
        if pd.isna(latest["MA5"]) or pd.isna(latest["MA20"]):
            return "Sideways"
        return "Bull" if latest["MA5"] > latest["MA20"] else "Bear"
    except Exception as e:
        logger.error(f"시장 추세 분석 실패: {e}. 'Sideways'로 대체.")
        return "Sideways"

def analyze_ma20_trend(df: pd.DataFrame) -> bool:
    if len(df) < 21:
        return False
    ma20 = df['Close'].rolling(window=20).mean()
    if pd.isna(ma20.iloc[-1]) or pd.isna(ma20.iloc[-2]):
        return False
    return ma20.iloc[-1] > ma20.iloc[-2]

def analyze_accumulation_volume(df: pd.DataFrame, period: int = 20) -> bool:
    if len(df) < period:
        return False
    recent_df = df.tail(period)
    up_days = recent_df[recent_df['Close'] > recent_df['Open']]
    down_days = recent_df[recent_df['Close'] <= recent_df['Open']]
    if len(up_days) < 3 or len(down_days) < 3:
        return False
    avg_vol_up = up_days['Volume'].mean()
    avg_vol_down = down_days['Volume'].mean()
    return avg_vol_up > avg_vol_down * 1.5

def detect_higher_lows(df: pd.DataFrame, period: int = 10) -> bool:
    if len(df) < period:
        return False
    recent_lows = df['Low'].tail(period)
    x = np.arange(len(recent_lows))
    slope, _ = np.polyfit(x, recent_lows, 1)
    return slope > 0

def detect_consolidation(df: pd.DataFrame, prior_trend_period: int = 60, consolidation_period: int = 15) -> bool:
    if len(df) < prior_trend_period + consolidation_period:
        return False
    start_price = df['Close'].iloc[-(prior_trend_period + consolidation_period)]
    peak_price_before_consolidation = df['Close'].iloc[-consolidation_period]
    if (peak_price_before_consolidation - start_price) / start_price < 0.3:
        return False
    cons_df = df.tail(consolidation_period)
    max_high = cons_df['High'].max()
    min_low = cons_df['Low'].min()
    return (max_high - min_low) / min_low < 0.15

def detect_yey_pattern(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    d2, d1, d0 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    is_yang2 = d2['Close'] > d2['Open']
    is_eum1 = d1['Close'] < d1['Open']
    is_yang0 = d0['Close'] > d0['Open']
    is_reversal = d0['Close'] > d2['Close']
    return is_yang2 and is_eum1 and is_yang0 and is_reversal

def _normalize_sector_name(x: Optional[str]) -> str:
    if not x or str(x).strip().upper() in {"", "NAN", "NA", "N/A"}:
        return "N/A"
    s = str(x).strip()
    mapping = {
        "보험": "금융", "증권": "금융", "은행": "금융",
        "IT 서비스": "IT서비스", "정보기술": "IT서비스",
        "반도체": "전기전자", "전자": "전기전자",
        "건설": "건설", "조선": "제조", "기계": "제조", "화학": "화학",
        "유통": "유통", "통신": "통신", "의료정밀": "의료정밀", "의약품": "의약품",
    }
    if s in mapping:
        return mapping[s]
    for k, v in mapping.items():
        if k in s:
            return v
    return s

def _extract_sector_from_kis_df(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in ["sect_kr_nm", "bstp_kor_isnm", "bstp_kor_isnm_nm", "induty_kor_isnm"]:
        if col in df.columns:
            val = str(df[col].iloc[0]).strip()
            if val and val.upper() not in {"N/A", "NONE"}:
                return val
    code_cols = ["bstp_cls_code", "std_idst_clsf_cd"]
    for col in code_cols:
        if col in df.columns:
            code = str(df[col].iloc[0]).strip()
            code_map = {"01": "제조", "10": "금융", "15": "IT서비스"}
            if code in code_map:
                return code_map[code]
    return None

# ─────────── KIS 호출 & 섹터 보강 ───────────
def _get_kis_sector_map(codes: List[str], kis: KIS, cache_key: Optional[str] = None, workers: int = 4) -> Dict[str, str]:
    if cache_key:
        cached = cache_load('kis_sector_map', cache_key)
        if isinstance(cached, dict) and cached:
            logger.info("kis 섹터맵 캐시 사용: kis_sector_map_%s.pkl", cache_key)
            return cached

    def _fetch_one(code: str) -> Tuple[str, str]:
        try:
            df = kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=str(code).zfill(6))
            if df is not None and not df.empty:
                sec = _extract_sector_from_kis_df(df)
                return (str(code).zfill(6), _normalize_sector_name(sec) if sec else "N/A")
            return (str(code).zfill(6), "N/A")
        except Exception as e:
            logger.debug("KIS 섹터 조회 실패(%s): %s", code, e)
            return (str(code).zfill(6), "N/A")

    sectors: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_fetch_one, c): c for c in codes}
        total = len(codes)
        for i, fut in enumerate(as_completed(futs), start=1):
            k, v = fut.result()
            sectors[k] = v
            if i % 20 == 0 or i == total:
                logger.info("  >> KIS(inquire_price) 섹터 조회 진행: %d/%d (%.1f%%)", i, total, i * 100.0 / total)
    if cache_key:
        cache_save('kis_sector_map', cache_key, sectors)
    return sectors

def _enrich_sector_with_kis_api(df_base: pd.DataFrame, kis: KIS, workers: int, cache_key: Optional[str] = None) -> pd.DataFrame:
    if df_base is None or df_base.empty:
        out = df_base.copy(); out['Sector'] = out.get('Sector', "N/A"); return out
    out = df_base.copy()
    if "Sector" not in out.columns: out["Sector"] = np.nan
    out["Sector"] = out["Sector"].astype("object")
    target_idx = out.index[out["Sector"].isna() | out["Sector"].eq("N/A")]
    if len(target_idx) == 0: logger.info("KIS 보강 대상 없음."); return out
    logger.info("KIS(inquire_price) 섹터 보강 시작 (대상 %d종목)", len(target_idx))
    ck = cache_key or datetime.now().strftime("%Y%m%d")
    kis_map = _get_kis_sector_map([str(x).zfill(6) for x in target_idx.tolist()], kis, ck, workers)
    out.loc[target_idx, "Sector"] = out.loc[target_idx].index.to_series().map(kis_map).values
    out["Sector"] = out["Sector"].map(_normalize_sector_name).fillna("N/A").astype("object")
    logger.info("✅ KIS(inquire_price) 섹터 정보 보강 완료.")
    return out

def _enrich_sector_with_fdr_krx(df_base: pd.DataFrame, market: str = "KOSPI") -> pd.DataFrame:
    out = df_base.copy()
    try:
        dfs = []
        try:
            df_mkt = fdr.StockListing(market)
            if "Code" in df_mkt.columns: df_mkt = df_mkt.set_index("Code")
            df_mkt = _norm_code_index(df_mkt).rename(columns={"종목명": "Name"}, errors="ignore")
            dfs.append(df_mkt)
        except Exception as e:
            logger.debug("FDR %s listing 실패: %s", market, e)
        try:
            df_krx = fdr.StockListing("KRX")
            if "Code" in df_krx.columns: df_krx = df_krx.set_index("Code")
            df_krx = _norm_code_index(df_krx).rename(columns={"종목명": "Name"}, errors="ignore")
            dfs.append(df_krx)
        except Exception as e:
            logger.debug("FDR KRX listing 실패: %s", e)
        if not dfs:
            if "Sector" not in out.columns: out["Sector"] = "N/A"
            out["Sector"] = out["Sector"].map(_normalize_sector_name).fillna("N/A")
            return out
        base = pd.concat(dfs, axis=0)[~pd.concat(dfs, axis=0).index.duplicated(keep="first")]
        cand_cols = [c for c in ["Sector", "Industry"] if c in base.columns]
        if not cand_cols:
            out["Sector"] = out.get("Sector", "N/A").fillna("N/A").map(_normalize_sector_name)
            return out
        joined = out.join(base[cand_cols], how="left")
        if "Sector" in joined.columns and "Industry" in joined.columns:
            joined["Sector"] = joined["Sector"].fillna(joined["Industry"])
        elif "Sector" not in joined.columns and "Industry" in joined.columns:
            joined["Sector"] = joined["Industry"]
        joined["Sector"] = joined["Sector"].map(_normalize_sector_name).fillna("N/A").astype("object")
        return joined.drop(columns=[c for c in ["Industry"] if c in joined.columns])
    except Exception as e:
        logger.debug("FDR 섹터 보강 실패: %s", e)
        if "Sector" not in out.columns: out["Sector"] = "N/A"
        out["Sector"] = out["Sector"].map(_normalize_sector_name).fillna("N/A").astype("object")
        return out

def _log_sector_summary(df: pd.DataFrame, label: str):
    if "Sector" not in df.columns:
        logger.info("섹터 요약(%s): Sector 컬럼 없음", label)
        return
    sec = df["Sector"].fillna("N/A")
    vc = sec.value_counts()
    na = int(vc.get("N/A", 0))
    tot = int(len(df))
    ratio = (na / tot * 100) if tot > 0 else 0.0
    logger.info("섹터 요약(%s): 고유=%d, N/A=%d (%.1f%%), TOP5=%s",
                label, len(vc), na, ratio, vc.head(5).to_dict())

def _get_pykrx_ticker_sector_map(date_str: str) -> Dict[str, str]:
    cached = cache_load("pykrx_sector_map", date_str)
    if cached is not None:
        logger.info("pykrx 섹터맵 캐시 사용: pykrx_sector_map_%s.pkl", date_str)
        return cached
    logger.info("pykrx를 이용한 티커-섹터 정보 매핑 시작...")
    ticker_sector_map: Dict[str, str] = {}
    try:
        kospi_sectors = pykrx.get_index_ticker_list(market='KOSPI')
        for sector_code in kospi_sectors:
            sector_name = pykrx.get_index_ticker_name(sector_code)
            if str(sector_name).startswith("코스피"): continue
            constituent_tickers = pykrx.get_index_portfolio_deposit_file(sector_code, date=date_str)
            for ticker in constituent_tickers:
                ticker_sector_map[str(ticker).zfill(6)] = _normalize_sector_name(sector_name)
        logger.info("✅ %d개 종목의 섹터 정보 매핑 완료.", len(ticker_sector_map))
    except Exception as e:
        logger.error("티커-섹터 정보 매핑 중 오류 발생: %s", e)
    cache_save("pykrx_sector_map", date_str, ticker_sector_map)
    return ticker_sector_map

def _calculate_sector_trends(date_str: str) -> Dict[str, float]:
    cached = cache_load("sector_trends", date_str)
    if cached is not None:
        logger.info("섹터 트렌드 캐시 사용: sector_trends_%s.pkl", date_str)
        return cached
    logger.info("KOSPI 업종별 트렌드 분석 시작...")
    sector_trends: Dict[str, float] = {}
    try:
        sector_tickers = pykrx.get_index_ticker_list(market='KOSPI')
        end_date = datetime.strptime(date_str, "%Y%m%d")
        start_date = (end_date - timedelta(days=60)).strftime("%Y%m%d")
        for ticker in sector_tickers:
            sector_name = pykrx.get_index_ticker_name(ticker)
            if str(sector_name).startswith("코스피"): continue
            df_index = pykrx.get_index_ohlcv_by_date(start_date, date_str, ticker)
            if df_index is None or len(df_index) < 20: continue
            close = df_index['종가']
            ma5 = close.rolling(window=5).mean().iloc[-1]
            ma20 = close.rolling(window=20).mean().iloc[-1]
            score = 0.5 if (pd.isna(ma5) or pd.isna(ma20)) else (1.0 if ma5 > ma20 else 0.0)
            sector_trends[_normalize_sector_name(sector_name)] = float(score)
        logger.info("✅ %d개 업종 트렌드 분석 완료.", len(sector_trends))
    except Exception as e:
        logger.error("업종 트렌드 분석 중 오류 발생: %s. 빈 데이터를 반환합니다.", e)
    cache_save("sector_trends", date_str, sector_trends)
    return sector_trends

def _apply_sector_source_order(df_base: pd.DataFrame, order: List[str], kis: KIS, workers: int, date_str: str, market: str) -> pd.DataFrame:
    df = df_base.copy()
    if "Sector" not in df.columns: df["Sector"] = np.nan
    df["Sector"] = df["Sector"].astype("object")
    if "SectorSource" not in df.columns: df["SectorSource"] = pd.Series(index=df.index, dtype="object")
    order_norm = [s.strip().lower() for s in order if s and s.strip().lower() in {"pykrx", "kis", "fdr"}] or ["pykrx", "kis", "fdr"]
    logger.info("섹터 소스 우선순위: %s", order_norm)
    for src in order_norm:
        missing_idx = df.index[df["Sector"].isna() | df["Sector"].eq("N/A")]
        if len(missing_idx) == 0: break
        if src == "pykrx":
            with stage("섹터 매핑(pykrx)"):
                mapping = _get_pykrx_ticker_sector_map(date_str)
                if mapping:
                    filled = df.loc[missing_idx].index.to_series().map(mapping)
                    df.loc[missing_idx, "Sector"] = filled
                    df.loc[missing_idx[filled.notna().values], "SectorSource"] = "pykrx"
            _log_sector_summary(df, "pykrx 매핑 후")
        elif src == "kis":
            logger.info("섹터 보강(KIS) 대상: %d 종목", len(missing_idx))
            kis_df = _enrich_sector_with_kis_api(df.loc[missing_idx].copy(), kis, workers, cache_key=date_str)
            kis_df["Sector"] = kis_df["Sector"].astype("object")
            df.loc[missing_idx, "Sector"] = kis_df.loc[missing_idx, "Sector"]
            df.loc[missing_idx, "SectorSource"] = np.where(kis_df.loc[missing_idx, "Sector"].notna(), "kis", df.loc[missing_idx, "SectorSource"])
            _log_sector_summary(df, "KIS 보강 후")
        elif src == "fdr":
            logger.info("섹터 보강(FDR) 대상: %d 종목", len(missing_idx))
            fdr_df = _enrich_sector_with_fdr_krx(df.loc[missing_idx].copy(), market=market)
            fdr_df["Sector"] = fdr_df["Sector"].astype("object")
            df.loc[missing_idx, "Sector"] = fdr_df.loc[missing_idx, "Sector"]
            df.loc[missing_idx, "SectorSource"] = np.where(fdr_df.loc[missing_idx, "Sector"].notna(), "fdr", df.loc[missing_idx, "SectorSource"])
            _log_sector_summary(df, "FDR 보강 후")
    df["Sector"] = df["Sector"].map(_normalize_sector_name).fillna("N/A").astype("object")
    _log_sector_summary(df, f"섹터 최종({','.join(order_norm)})")
    return df

def _resolve_business_date(date_str: str, market: str) -> str:
    dt = datetime.strptime(date_str, "%Y%m%d")
    for _ in range(20):
        d = dt.strftime("%Y%m%d")
        try:
            ohlcv = pykrx.get_market_ohlcv_by_ticker(d, market=market)
            if ohlcv is not None and not ohlcv.empty:
                f = get_fundamentals(d, market)
                if f is not None and not f.empty and pd.to_numeric(f["PBR"], errors='coerce').gt(0).sum() > 50:
                    if d != date_str: logger.info("비거래일/데이터 부족 감지 → 기준일 보정: %s → %s", date_str, d)
                    return d
        except Exception:
            pass
        dt -= timedelta(days=1)
    logger.warning("직전 거래일 탐지 실패. 원래 날짜를 사용합니다: %s", date_str)
    return date_str

def _get_trading_value_5d_avg(date_str: str, market: str) -> pd.Series:
    amounts = []
    dt = datetime.strptime(date_str, "%Y%m%d")
    found, tried = 0, 0
    while found < 5 and tried < 25:
        d = dt.strftime("%Y%m%d")
        try:
            df = pykrx.get_market_ohlcv_by_ticker(d, market=market)
            if df is not None and not df.empty and "거래대금" in df.columns:
                amounts.append(_norm_code_index(df)["거래대금"].rename(d).astype("float64"))
                found += 1
        except Exception:
            pass
        dt -= timedelta(days=1)
        tried += 1
    if not amounts:
        logger.warning("거래대금 5D 계산 실패. 빈 Series 반환.")
        return pd.Series(dtype="float64", name="Amount5D")
    return pd.concat(amounts, axis=1).mean(axis=1).rename("Amount5D")

def _get_market_regime_score(date_str: str, market: str) -> float:
    index_code = {"KOSPI": "1001", "KOSDAQ": "2001"}.get(market.upper(), "1001")
    start = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    try:
        df = pykrx.get_index_ohlcv_by_date(start, date_str, index_code)
        close = df['종가']
        if len(close) < 200: return 0.5
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        rsi = calculate_rsi(close)
        if any(pd.isna(x) for x in [ma50, ma200, rsi]): return 0.5
        score = ((1 if close.iloc[-1] > ma50 else 0) + (1 if ma50 > ma200 else 0) + (max(0, 1 - abs(rsi - 50) / 50))) / 3.0
        return float(score)
    except Exception:
        return 0.5

def _get_market_regime_components(date_str: str, market: str) -> Dict[str, float]:
    idx = {"KOSPI": "1001", "KOSDAQ": "2001"}.get(market.upper(), "1001")
    start = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    try:
        df = pykrx.get_index_ohlcv_by_date(start, date_str, idx)
        close = df['종가']
        if len(close) < 200: return {"above_ma50": 0.0, "ma50_gt_ma200": 0.0, "rsi_term": 0.5}
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        rsi = calculate_rsi(close)
        return {"above_ma50": 1.0 if close.iloc[-1] > ma50 else 0.0, "ma50_gt_ma200": 1.0 if ma50 > ma200 else 0.0, "rsi_term": max(0.0, 1 - abs(rsi - 50) / 50)}
    except Exception:
        return {"above_ma50": 0.5, "ma50_gt_ma200": 0.5, "rsi_term": 0.5}

def _filter_initial_stocks(date_str: str, cfg: Dict[str, Any], market: str, risk: Dict[str, Any], debug: bool) -> Tuple[pd.DataFrame, str]:
    logger.info("1차 필터링 시작...")
    fixed_date = _resolve_business_date(date_str, market)
    df_all = get_stock_listing(market)
    fundamentals = get_fundamentals(fixed_date, market)
    if fundamentals is None or fundamentals.empty:
        alt_date = (datetime.strptime(fixed_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        logger.warning("펀더멘털 결과가 비었습니다. 하루 전(%s)으로 재시도합니다.", alt_date)
        fundamentals = get_fundamentals(alt_date, market)
        if fundamentals is None or fundamentals.empty:
            logger.error("재시도에도 펀더멘털이 비어 있습니다. 필터링을 중단합니다.")
            return pd.DataFrame(), fixed_date
        else:
            fixed_date = alt_date
    amt5 = _get_trading_value_5d_avg(fixed_date, market)
    df_pre = df_all[["Name", "Marcap"]].join(amt5, how="left").join(fundamentals[["PER", "PBR"]], how="left")
    df_pre["Marcap"] = pd.to_numeric(df_pre["Marcap"], errors="coerce").fillna(0)
    if debug:
        (OUTPUT_DIR / "debug").mkdir(exist_ok=True, parents=True)
        df_pre.to_csv(OUTPUT_DIR / f"debug/debug_joined_{market}_{fixed_date}.csv")
    _describe_series("Marcap", df_pre["Marcap"])
    _describe_series("Amount5D", df_pre["Amount5D"])
    min_mc = float(cfg.get("min_market_cap", 0))
    min_amt = float(cfg.get("min_trading_value_5d_avg", 0))
    mask_mc = df_pre["Marcap"] >= min_mc
    amt_num = pd.to_numeric(df_pre["Amount5D"], errors='coerce').fillna(0)
    mask_amt = amt_num >= min_amt
    n0 = len(df_pre); n1 = int(mask_mc.sum()); n2 = int((mask_mc & mask_amt).sum())
    logger.info("단계별 생존 수: 시작=%d → Marcap(≥%s)=%d → +Amount5D(≥%s)=%d", n0, f"{int(min_mc):,}", n1, f"{int(min_amt):,}", n2)
    logger.info("탈락 사유: Marcap 미달=%d, Amount5D 미달(마켓캡 통과 중)=%d", int((~mask_mc).sum()), int((mask_mc & ~mask_amt).sum()))
    df_filtered = df_pre[mask_mc & mask_amt].copy()
    bl = {str(x).zfill(6) for x in risk.get("blacklist_tickers", []) if x}
    wl = {str(x).zfill(6) for x in risk.get("whitelist_tickers", []) if x}
    if wl: before = len(df_filtered); df_filtered = df_filtered[df_filtered.index.isin(wl)]; logger.info("화이트리스트 적용: %d → %d", before, len(df_filtered))
    if bl: before = len(df_filtered); df_filtered = df_filtered[~df_filtered.index.isin(bl)]; logger.info("블랙리스트 적용: %d → %d", before, len(df_filtered))
    logger.info("✅ 1차 필터링 완료: %d → %d 종목 (시장=%s, 기준일=%s, min_mc=%s, min_amt5D=%s)", len(df_pre), len(df_filtered), market, fixed_date, f"{int(min_mc):,}", f"{int(min_amt):,}")
    return df_filtered, fixed_date

def _calculate_scores_for_ticker(code: str, date_str: str, fin_info: pd.Series, cfg: Dict[str, Any], market_score: float, sector_trends: Dict[str, float]) -> Optional[Dict[str, Any]]:
    try:
        start_dt_str = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
        df_price = get_historical_prices(code, start_dt_str, date_str)
        if df_price is None or len(df_price) < 200: return None
        close = df_price["Close"].iloc[-1]; ma50 = df_price["Close"].rolling(50).mean().iloc[-1]; ma200 = df_price["Close"].rolling(200).mean().iloc[-1]; rsi = calculate_rsi(df_price["Close"])
        if any(pd.isna(x) for x in [ma50, ma200, rsi]): return None
        tech_score = ((1 if close > ma50 else 0) + (1 if ma50 > ma200 else 0) + (max(0, 1 - abs(rsi - 50) / 50))) / 3
        per_val = pd.to_numeric(fin_info.get("PER"), errors='coerce'); pbr_val = pd.to_numeric(fin_info.get("PBR"), errors='coerce')
        per_term = max(0, min(1, (50 - per_val) / 50)) if pd.notna(per_val) and per_val > 0 else 0
        pbr_term = max(0, min(1, (5 - pbr_val) / 5)) if pd.notna(pbr_val) and pbr_val > 0 else 0
        fin_score = 0.5 * (per_term + pbr_term)
        sector_name = str(fin_info.get("Sector", "N/A")); sector_score = float(sector_trends.get(sector_name, 0.5))
        ma20_up = analyze_ma20_trend(df_price); accum_vol = analyze_accumulation_volume(df_price); hl_trend = detect_higher_lows(df_price); consd = detect_consolidation(df_price); yey = detect_yey_pattern(df_price)
        pattern_flags = [ma20_up, accum_vol, hl_trend, consd, yey]; pattern_score = float(np.mean(pattern_flags)) if pattern_flags else 0.0
        fin_w = float(cfg.get('fin_weight', 0.5)); tech_w = float(cfg.get('tech_weight', 0.5)); mkt_w = float(cfg.get('mkt_weight', 0.0)); sector_w = float(cfg.get('sector_weight', 0.0)); pattern_w = float(cfg.get('pattern_weight', 0.0))
        total_score = (fin_score * fin_w + tech_score * tech_w + market_score * mkt_w + sector_score * sector_w + pattern_score * pattern_w)
        total_score = float(np.clip(total_score, 0.0, 1.0))
        return {"Ticker": code, "Price": int(round(float(close))), "Score": round(float(total_score), 4), "FinScore": round(float(fin_score), 4), "TechScore": round(float(tech_score), 4), "MktScore": round(float(market_score), 4), "SectorScore": round(float(sector_score), 4), "PatternScore": round(float(pattern_score), 4), "MA20Up": bool(ma20_up), "AccumVol": bool(accum_vol), "HigherLows": bool(hl_trend), "Consolidation": bool(consd), "YEY": bool(yey), "PER": round(float(per_val), 2) if pd.notna(per_val) else None, "PBR": round(float(pbr_val), 2) if pd.notna(pbr_val) else None, "RSI": round(float(rsi), 2), "MA50": round(float(ma50), 2), "MA200": round(float(ma200), 2)}
    except Exception as ex:
        logger.debug("[%s] 스코어 계산 예외: %s", code, ex)
        return None

def diversify_by_sector(df_sorted: pd.DataFrame, top_n: int, sector_cap: float) -> pd.DataFrame:
    if top_n <= 0 or df_sorted.empty: return df_sorted.iloc[0:0]
    if sector_cap <= 0: return df_sorted.head(top_n)
    max_per_sector = max(1, int(np.ceil(top_n * float(sector_cap))))
    sector_series = df_sorted["Sector"] if "Sector" in df_sorted.columns else pd.Series(["N/A"] * len(df_sorted), index=df_sorted.index)
    counts: Dict[str, int] = {}; selected_idx: List[Any] = []
    for idx, sec in zip(df_sorted.index, sector_series):
        c = counts.get(sec, 0)
        if c < max_per_sector:
            selected_idx.append(idx)
            counts[sec] = c + 1
        if len(selected_idx) >= top_n: break
    if len(selected_idx) < top_n:
        for idx in df_sorted.index:
            if idx not in selected_idx:
                selected_idx.append(idx)
                if len(selected_idx) >= top_n: break
    return df_sorted.loc[selected_idx]

# ─────────── 메인 실행 ───────────
def run_screener(date_str: str, market: str, config_path: Optional[str], workers: int, debug: bool):
    start_msg = f"📊 스크리너 시작 (date={date_str}, market={market}, workers={workers}, debug={debug})"
    logger.info(start_msg)
    _notify(start_msg, key="screener_start", cooldown_sec=60)

    if debug:
        logger.setLevel(logging.DEBUG)

    ensure_output_dir()

    # config 로드/병합
    settings = load_config()
    if config_path and Path(config_path).expanduser().is_file():
        try:
            with open(Path(config_path).expanduser(), "r", encoding="utf-8") as f:
                cli_cfg = json.load(f)
            settings.update(cli_cfg or {})
            logger.info("CLI config 병합 완료: %s", str(Path(config_path).expanduser()))
        except Exception as e:
            logger.warning("CLI config 병합 실패(%s): %s", config_path, e)

    if not settings:
        msg = "설정 로딩 실패로 종료합니다."
        logger.error(msg)
        _notify(f"❌ {msg}", key="screener_config_err", cooldown_sec=60)
        return

    # KIS 인스턴스
    broker_config = settings.get("kis_broker", {})
    trading_env = settings.get("trading_environment", "mock")
    kis = KIS(broker_config, env=trading_env)
    if not getattr(kis, "auth_token", None):
        msg = "KIS API 인증 실패로 종료합니다."
        logger.error(msg)
        _notify(f"❌ {msg}", key="screener_kis_auth_fail", cooldown_sec=60)
        return
    logger.info("'%s' 모드로 KIS API 인증 완료.", trading_env)

    # 개장일 안내(옵션)
    try:
        open_day = is_market_open_day()
        logger.info("오늘 한국 시장 개장일 여부: %s", "개장" if open_day else "휴장")
    except Exception as e:
        logger.warning("시장 개장일 확인 실패: %s", e)

    screener_params = settings.get("screener_params", {})
    risk_params     = settings.get("risk_params", {})

    with stage("1차 필터링", notify_key="screener_stage1"):
        df_filtered, fixed_date = _filter_initial_stocks(date_str, screener_params, market, risk_params, debug)
        if df_filtered.empty:
            msg = "❌ 1차 필터링 결과, 대상 종목이 없습니다."
            logger.warning(msg)
            _notify(msg, key="screener_no_candidates_stage1", cooldown_sec=60)
            return

    with stage("섹터 보강", notify_key="screener_sector"):
        order = screener_params.get("sector_source_priority", ["pykrx", "kis", "fdr"])
        df_filtered = _apply_sector_source_order(df_filtered, order, kis, workers, fixed_date, market)

    with stage("시장 레짐 계산", notify_key="screener_regime"):
        regime = _get_market_regime_score(fixed_date, market)
        market_score = 0.7 * regime + 0.3 * 0.5
        comps = _get_market_regime_components(fixed_date, market)
        market_trend = get_market_trend(fixed_date)
        logger.info("시장 레짐 스코어 (가중치 적용): %.3f", market_score)
        logger.info("레짐 구성요소: above_ma50=%.2f, ma50>ma200=%.2f, rsi_term=%.2f",
                    comps["above_ma50"], comps["ma50_gt_ma200"], comps["rsi_term"])
        logger.info("시장 단기 추세(60D MA5/MA20): %s", market_trend)

    with stage("섹터 트렌드 계산", notify_key="screener_sector_trend"):
        sector_trends = _calculate_sector_trends(fixed_date)

    with stage("상세 분석(스코어링)", notify_key="screener_scoring"):
        results = []
        total = len(df_filtered)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _calculate_scores_for_ticker, code, fixed_date, row,
                    screener_params, market_score, sector_trends
                ): code
                for code, row in df_filtered.iterrows()
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                if i % 50 == 0 or i == total:
                    logger.info("  >> 상세 분석 진행률: %d/%d (%.1f%%)", i, total, i * 100.0 / total)
                res = fut.result()
                if res:
                    results.append(res)
        if not results:
            msg = "❌ 2차 스크리닝 결과, 최종 후보가 없습니다."
            logger.warning(msg)
            _notify(msg, key="screener_no_candidates_stage2", cooldown_sec=60)
            return

    with stage("정렬/다양화/손절·목표가 계산/저장", notify_key="screener_finalize"):
        df_scores = pd.DataFrame(results).set_index("Ticker")
        df_scores = df_scores.drop(columns=["Sector"], errors="ignore")

        df_filtered_no_dup = df_filtered.drop(columns=['PER', 'PBR'], errors='ignore')
        df_final = df_filtered_no_dup.join(df_scores, how="inner").reset_index().rename(columns={"index": "Ticker"})
        df_sorted = df_final.sort_values("Score", ascending=False)

        top_n = min(int(screener_params.get("top_n", 10)), int(risk_params.get("max_positions", 10)))
        sector_cap = float(screener_params.get("sector_cap", 0.3))

        final_candidates = diversify_by_sector(df_sorted.set_index('Ticker'), top_n, sector_cap).reset_index()

        levels_data = []
        for _, row in final_candidates.iterrows():
            levels = _compute_levels(row["Ticker"], row["Price"], fixed_date, risk_params)
            levels_data.append(levels)
        df_levels = pd.DataFrame(levels_data, index=final_candidates.index)
        final_candidates = pd.concat([final_candidates, df_levels], axis=1)

        cols = ["Ticker","Name","Sector","Price","손절가", "목표가", "source", "MA50","MA200","Score",
                "FinScore","TechScore","MktScore","SectorScore","PatternScore",
                "MA20Up","AccumVol","HigherLows","Consolidation","YEY",
                "PER","PBR","RSI","Marcap","Amount5D","SectorSource"]
        keep = [c for c in cols if c in final_candidates.columns]
        final_candidates = final_candidates[keep + [c for c in final_candidates.columns if c not in keep]]

        with pd.option_context('display.width', 240):
            print("\n--- ⭐ 최종 스크리닝 결과 ⭐ ---")
            print(final_candidates.to_string(index=False))

        full_json  = OUTPUT_DIR / f"screener_full_{fixed_date}_{market}.json"
        final_json = OUTPUT_DIR / f"screener_results_{fixed_date}_{market}.json"
        df_sorted.to_json(full_json, orient='records', indent=2, force_ascii=False)
        final_candidates.to_json(final_json, orient='records', indent=2, force_ascii=False)
        logger.info("전체 랭킹 저장: %s", full_json)
        logger.info("✅ 스크리닝 완료. %d개 후보 저장: %s", len(final_candidates), final_json)

        # 디스코드 요약 알림(상위 5개)
        try:
            top5 = final_candidates.head(5)[["Ticker","Name","Sector","Price","목표가","손절가","Score"]]
            lines = ["Top5:"]
            for _, r in top5.iterrows():
                lines.append(f"- {r['Name']}({str(r['Ticker']).zfill(6)}), Sec:{r['Sector']}, Px:{int(r['Price']):,}, TP:{int(r['목표가']):,}, SL:{int(r['손절가']):,}, S:{r['Score']:.3f}")
            _notify("✅ 스크리너 완료\n" + "\n".join(lines), key="screener_done", cooldown_sec=60)
        except Exception:
            _notify("✅ 스크리너 완료 (요약 구성 실패)", key="screener_done", cooldown_sec=60)

# ─────────── CLI ───────────
def parse_args():
    parser = argparse.ArgumentParser(description="KOSPI/KOSDAQ/KONEX 스크리너")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--market", default=os.getenv("MARKET", "KOSPI"), choices=["KOSPI", "KOSDAQ", "KONEX"])
    parser.add_argument("--config", help="추가/오버레이할 config.json 파일 경로")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "4")))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_screener(args.date, args.market, args.config, max(1, args.workers), args.debug)
