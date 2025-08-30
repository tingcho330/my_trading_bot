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

# KIS API 모듈
import kis_auth as ka
from domestic_stock import domestic_stock_functions as ds

# ───────────────────────────── 기본 설정 ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")  # 지수표기 방지

@contextmanager
def stage(name: str):
    t0 = time.perf_counter()
    logger.info("▶ %s 시작", name)
    try:
        yield
    finally:
        logger.info("⏱ %s 완료 (%.2fs)", name, time.perf_counter() - t0)

# ───────────────────────────── 경로/상수 ─────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CWD = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "output"

# ─────────────────────────── 유틸리티/로딩 ───────────────────────────
def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _first_existing_path(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p.is_file():
                return p
        except Exception:
            pass
    return None

def _build_config_candidates(cli_path: Optional[str]) -> List[Path]:
    env_path = os.getenv("CONFIG_PATH", "").strip() or None
    candidates: List[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates += [CWD / "config" / "config.json", CWD / "config.json"]
    candidates += [BASE_DIR / ".." / "config" / "config.json",
                   PROJECT_ROOT / "config" / "config.json",
                   PROJECT_ROOT / "config.json"]
    candidates += [Path("/config/config.json"), Path("/app/config/config.json")]
    normed = []
    for p in candidates:
        try:
            normed.append(p.resolve())
        except Exception:
            normed.append(p)
    return normed

def load_settings(config_path: Optional[str]) -> Dict[str, Any]:
    cands = _build_config_candidates(config_path)
    logger.info("CONFIG 탐색 후보(상위 10개): %s", [str(p) for p in cands[:10]])
    found = _first_existing_path(cands)
    if not found:
        logger.error("설정 파일을 찾지 못했습니다.")
        return {}
    try:
        with open(found, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        logger.info("설정 파일 로딩 성공: %s", str(found))
        return cfg
    except Exception as e:
        logger.error("설정 파일 로딩 실패(%s): %s", str(found), e)
        return {}

def _fsize(p: Path) -> str:
    try:
        return f"{p.stat().st_size/1024:.1f} KB"
    except Exception:
        return "?"

# ─────────────────────────── 공통 보조 함수 ───────────────────────────
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

# ─────────────── 데이터 제공 및 기술 지표 계산 ───────────────
def get_stock_listing(market: str = "KOSPI") -> pd.DataFrame:
    logger.info("종목 목록 조회(FDR): market=%s", market)
    df = fdr.StockListing(market)
    if 'Code' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'Code'})
    df = df.set_index('Code')
    df = _norm_code_index(df)
    df = df.rename(columns={"MarketCap": "Marcap", "종목명": "Name"}, errors="ignore")
    return df

# ────────────────────── 펀더멘털/시세/지표 ──────────────────────
def get_fundamentals(date_str: str, market: str = "KOSPI") -> pd.DataFrame:
    """pykrx로 특정 기준일의 펀더멘털(PER, PBR 등)을 티커별 조회"""
    logger.info("펀더멘털 조회(PYKRX): date=%s market=%s", date_str, market)
    try:
        df = pykrx.get_market_fundamental_by_ticker(date_str, market=market)
        return _norm_code_index(df)
    except Exception as e:
        logger.error("펀더멘털 조회 실패 (%s, %s): %s", date_str, market, e)
        return pd.DataFrame()

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

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    if pd.isna(loss.iloc[-1]) or loss.iloc[-1] == 0:
        return 50.0
    return 100 - (100 / (1 + (gain.iloc[-1] / loss.iloc[-1])))

# === 요청하신 보조 함수들 (시장/패턴) ===
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
    consolidation_df = df.tail(consolidation_period)
    max_high = consolidation_df['High'].max()
    min_low = consolidation_df['Low'].min()
    if (max_high - min_low) / min_low < 0.15:
        return True
    return False

def detect_yey_pattern(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    d2, d1, d0 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    is_yang2 = d2['Close'] > d2['Open']
    is_eum1 = d1['Close'] < d1['Open']
    is_yang0 = d0['Close'] > d0['Open']
    is_reversal = d0['Close'] > d2['Close']
    return is_yang2 and is_eum1 and is_yang0 and is_reversal

# ─────────────────────────── 섹터 관련 (KIS/FDR/pykrx) ───────────────────────────
def _enrich_sector_with_kis_api(df_base: pd.DataFrame, workers: int) -> pd.DataFrame:
    logger.info("KIS API를 통해 섹터 정보 조회 시작...")
    sectors = {}
    total = len(df_base)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_code = {
            executor.submit(ds.search_stock_info, prdt_type_cd="300", pdno=code): code
            for code in df_base.index
        }
        for i, future in enumerate(as_completed(future_to_code), start=1):
            code = future_to_code[future]
            if i % 10 == 0 or i == total:
                 logger.info("  >> 섹터 조회 진행률: %d/%d (%.1f%%)", i, total, i * 100.0 / total)
            try:
                result_df = future.result()
                if result_df is not None and not result_df.empty and 'sect_kr_nm' in result_df.columns:
                    sector_name = str(result_df['sect_kr_nm'].iloc[0]).strip()
                    sectors[code] = sector_name if sector_name else "N/A"
                else:
                    sectors[code] = "N/A"
            except Exception as e:
                logger.debug("%s: 섹터 정보 조회 실패 - %s", code, e)
                sectors[code] = "N/A"
    out = df_base.copy()
    out['Sector'] = out.index.map(sectors).fillna("N/A")
    logger.info("✅ KIS API 섹터 정보 조회 완료.")
    return out

def _enrich_sector_with_fdr_krx(df_base: pd.DataFrame) -> pd.DataFrame:
    try:
        krx = fdr.StockListing("KRX")
        if "Code" in krx.columns:
            krx = krx.set_index("Code")
        krx = _norm_code_index(krx)
        krx = krx.rename(columns={"종목명": "Name"}, errors="ignore")
        cols = [c for c in ["Sector", "Industry"] if c in krx.columns]
        out = df_base.copy()
        if cols:
            out = out.join(krx[cols], how="left")
            if "Sector" not in out.columns:
                out["Sector"] = out.get("Industry", "N/A")
            out["Sector"] = out["Sector"].fillna(out.get("Industry")).fillna("N/A")
        else:
            out["Sector"] = out.get("Sector", "N/A").fillna("N/A")
        return out
    except Exception as e:
        logger.debug("FDR KRX 섹터 보강 실패: %s", e)
        out = df_base.copy()
        if "Sector" not in out.columns:
            out["Sector"] = "N/A"
        out["Sector"] = out["Sector"].fillna("N/A")
        return out

def _log_sector_summary(df: pd.DataFrame, label: str):
    if "Sector" not in df.columns:
        logger.info("섹터 요약(%s): Sector 컬럼 없음", label)
        return
    sec = df["Sector"].fillna("N/A")
    vc = sec.value_counts()
    na = int(vc.get("N/A", 0))
    tot = int(len(df))
    ratio = (na / tot * 100) if tot else 0.0
    logger.info("섹터 요약(%s): 고유=%d, N/A=%d (%.1f%%), TOP5=%s",
                label, len(vc), na, ratio, vc.head(5).to_dict())

# === pykrx 기반 섹터 트렌드 / 섹터 매핑 ===
def _calculate_sector_trends(date_str: str) -> Dict[str, float]:
    logger.info("KOSPI 업종별 트렌드 분석 시작...")
    sector_trends: Dict[str, float] = {}
    try:
        sector_tickers = pykrx.get_index_ticker_list(market='KOSPI')
        end_date = datetime.strptime(date_str, "%Y%m%d")
        start_date = (end_date - timedelta(days=60)).strftime("%Y%m%d")
        for ticker in sector_tickers:
            sector_name = pykrx.get_index_ticker_name(ticker)
            if str(sector_name).startswith("코스피"):
                continue
            df_index = pykrx.get_index_ohlcv_by_date(start_date, date_str, ticker)
            if df_index is None or len(df_index) < 20:
                continue
            close = df_index['종가']
            ma5 = close.rolling(window=5).mean().iloc[-1]
            ma20 = close.rolling(window=20).mean().iloc[-1]
            if pd.isna(ma5) or pd.isna(ma20):
                sector_trends[sector_name] = 0.5
            elif ma5 > ma20:
                sector_trends[sector_name] = 1.0
            else:
                sector_trends[sector_name] = 0.0
        logger.info("✅ %d개 업종 트렌드 분석 완료.", len(sector_trends))
    except Exception as e:
        logger.error("업종 트렌드 분석 중 오류 발생: %s. 빈 데이터를 반환합니다.", e)
    return sector_trends

def _get_pykrx_ticker_sector_map(date_str: str) -> Dict[str, str]:
    logger.info("pykrx를 이용한 티커-섹터 정보 매핑 시작...")
    ticker_sector_map: Dict[str, str] = {}
    try:
        kospi_sectors = pykrx.get_index_ticker_list(market='KOSPI')
        for sector_code in kospi_sectors:
            sector_name = pykrx.get_index_ticker_name(sector_code)
            if str(sector_name).startswith("코스피"):
                continue
            constituent_tickers = pykrx.get_index_portfolio_deposit_file(sector_code, date=date_str)
            for ticker in constituent_tickers:
                ticker_sector_map[str(ticker).zfill(6)] = sector_name
        logger.info("✅ %d개 종목의 섹터 정보 매핑 완료.", len(ticker_sector_map))
    except Exception as e:
        logger.error("티커-섹터 정보 매핑 중 오류 발생: %s", e)
    return ticker_sector_map

def _enrich_sector(df_base: pd.DataFrame, workers: int, date_str: str) -> pd.DataFrame:
    df = df_base.copy()
    if "Sector" not in df.columns:
        df["Sector"] = np.nan

    # 1) pykrx 섹터 매핑
    with stage("섹터 매핑(pykrx)"):
        mapping = _get_pykrx_ticker_sector_map(date_str)
        if mapping:
            df["Sector"] = df["Sector"].fillna(df.index.to_series().map(mapping))
    _log_sector_summary(df, "pykrx 매핑 후")

    # 2) KIS API 보강(누락분만)
    missing_idx = df.index[df["Sector"].isna() | df["Sector"].eq("N/A")]
    if len(missing_idx) > 0:
        logger.info("섹터 보강(KIS) 대상: %d 종목", len(missing_idx))
        kis_df = _enrich_sector_with_kis_api(df.loc[missing_idx].copy(), workers)
        df.loc[missing_idx, "Sector"] = kis_df.loc[missing_idx, "Sector"]
    _log_sector_summary(df, "KIS 보강 후")

    # 3) FDR(KRX) 최종 보완
    missing_idx = df.index[df["Sector"].isna() | df["Sector"].eq("N/A")]
    if len(missing_idx) > 0:
        logger.info("섹터 보강(FDR KRX) 대상: %d 종목", len(missing_idx))
        fdr_df = _enrich_sector_with_fdr_krx(df.loc[missing_idx].copy())
        df.loc[missing_idx, "Sector"] = fdr_df.loc[missing_idx, "Sector"]

    df["Sector"] = df["Sector"].fillna("N/A")
    _log_sector_summary(df, "섹터 최종")
    return df

# ────────────────────── 비거래일 보정 & 5D 거래대금 ──────────────────────
def _resolve_business_date(date_str: str, market: str) -> str:
    dt = datetime.strptime(date_str, "%Y%m%d")
    for _ in range(20):
        d = dt.strftime("%Y%m%d")
        try:
            ohlcv = pykrx.get_market_ohlcv_by_ticker(d, market=market)
            if ohlcv is not None and not ohlcv.empty:
                f = get_fundamentals(d, market)
                if f is not None and not f.empty and pd.to_numeric(f["PBR"], errors='coerce').gt(0).sum() > 50:
                    if d != date_str:
                        logger.info("비거래일/데이터 부족 감지 → 기준일 보정: %s → %s", date_str, d)
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

# ────────────────────── 시장 레짐 스코어 ──────────────────────
def _get_market_regime_score(date_str: str, market: str) -> float:
    index_code = {"KOSPI": "1001", "KOSDAQ": "2001"}.get(market.upper(), "1001")
    start = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    try:
        df = pykrx.get_index_ohlcv_by_date(start, date_str, index_code)
        close = df['종가']
        if len(close) < 200:
            return 0.5
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        rsi = calculate_rsi(close)
        if any(pd.isna(x) for x in [ma50, ma200, rsi]):
            return 0.5
        score = ((1 if close.iloc[-1] > ma50 else 0) +
                 (1 if ma50 > ma200 else 0) +
                 (max(0, 1 - abs(rsi - 50) / 50))) / 3.0
        return float(score)
    except Exception:
        return 0.5

def _get_market_regime_components(date_str: str, market: str) -> Dict[str, float]:
    idx = {"KOSPI": "1001", "KOSDAQ": "2001"}.get(market.upper(), "1001")
    start = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    try:
        df = pykrx.get_index_ohlcv_by_date(start, date_str, idx)
        close = df['종가']
        if len(close) < 200:
            return {"above_ma50": 0.0, "ma50_gt_ma200": 0.0, "rsi_term": 0.5}
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        rsi = calculate_rsi(close)
        return {
            "above_ma50": 1.0 if close.iloc[-1] > ma50 else 0.0,
            "ma50_gt_ma200": 1.0 if ma50 > ma200 else 0.0,
            "rsi_term": max(0.0, 1 - abs(rsi - 50) / 50)
        }
    except Exception:
        return {"above_ma50": 0.5, "ma50_gt_ma200": 0.5, "rsi_term": 0.5}

# ─────────────────────────── 1차 필터 ───────────────────────────
def _filter_initial_stocks(date_str: str, cfg: Dict[str, Any], market: str, risk: Dict[str, Any], debug: bool) -> Tuple[pd.DataFrame, str]:
    logger.info("1차 필터링 시작...")
    fixed_date = _resolve_business_date(date_str, market)

    df_all = get_stock_listing(market)
    fundamentals = get_fundamentals(fixed_date, market)

    # 🔁 펀더멘털 비어있을 때 하루 전으로 재시도
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

    mask_mc  = df_pre["Marcap"] >= min_mc
    amt_num  = pd.to_numeric(df_pre["Amount5D"], errors='coerce').fillna(0)
    mask_amt = amt_num >= min_amt

    n0 = len(df_pre)
    n1 = int(mask_mc.sum())
    n2 = int((mask_mc & mask_amt).sum())
    logger.info("단계별 생존 수: 시작=%d → Marcap(≥%s)=%d → +Amount5D(≥%s)=%d",
                n0, f"{int(min_mc):,}", n1, f"{int(min_amt):,}", n2)
    logger.info("탈락 사유: Marcap 미달=%d, Amount5D 미달(마켓캡 통과 중)=%d",
                int((~mask_mc).sum()),
                int((mask_mc & ~mask_amt).sum()))

    df_filtered = df_pre[mask_mc & mask_amt].copy()

    bl = {str(x).zfill(6) for x in risk.get("blacklist_tickers", []) if x}
    wl = {str(x).zfill(6) for x in risk.get("whitelist_tickers", []) if x}
    if wl:
        before = len(df_filtered)
        df_filtered = df_filtered[df_filtered.index.isin(wl)]
        logger.info("화이트리스트 적용: %d → %d", before, len(df_filtered))
    if bl:
        before = len(df_filtered)
        df_filtered = df_filtered[~df_filtered.index.isin(bl)]
        logger.info("블랙리스트 적용: %d → %d", before, len(df_filtered))

    logger.info("✅ 1차 필터링 완료: %d → %d 종목 (시장=%s, 기준일=%s, min_mc=%s, min_amt5D=%s)",
                len(df_pre), len(df_filtered), market, fixed_date, f"{int(min_mc):,}", f"{int(min_amt):,}")
    return df_filtered, fixed_date

# ─────────────────────────── 스코어 계산 ───────────────────────────
def _calculate_scores_for_ticker(
    code: str,
    date_str: str,
    fin_info: pd.Series,
    cfg: Dict[str, Any],
    market_score: float,
    sector_trends: Dict[str, float]
) -> Optional[Dict[str, Any]]:
    try:
        start_dt_str = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
        df_price = get_historical_prices(code, start_dt_str, date_str)
        if df_price is None or len(df_price) < 200:
            return None

        close = df_price["Close"].iloc[-1]
        ma50  = df_price["Close"].rolling(50).mean().iloc[-1]
        ma200 = df_price["Close"].rolling(200).mean().iloc[-1]
        rsi   = calculate_rsi(df_price["Close"])
        if any(pd.isna(x) for x in [ma50, ma200, rsi]):
            return None

        tech_score = ((1 if close > ma50 else 0) +
                      (1 if ma50 > ma200 else 0) +
                      (max(0, 1 - abs(rsi - 50) / 50))) / 3

        per_val = pd.to_numeric(fin_info.get("PER"), errors='coerce')
        pbr_val = pd.to_numeric(fin_info.get("PBR"), errors='coerce')
        per_term = max(0, min(1, (50 - per_val) / 50)) if pd.notna(per_val) and per_val > 0 else 0
        pbr_term = max(0, min(1, (5  - pbr_val) / 5 )) if pd.notna(pbr_val) and pbr_val > 0 else 0
        fin_score = 0.5 * (per_term + pbr_term)

        # 섹터 점수(트렌드)
        sector_name = str(fin_info.get("Sector", "N/A"))
        sector_score = float(sector_trends.get(sector_name, 0.5))

        # 패턴/마이크로트렌드 신호 (옵션)
        ma20_up   = analyze_ma20_trend(df_price)
        accum_vol = analyze_accumulation_volume(df_price)
        hl_trend  = detect_higher_lows(df_price)
        consd     = detect_consolidation(df_price)
        yey       = detect_yey_pattern(df_price)
        pattern_flags = [ma20_up, accum_vol, hl_trend, consd, yey]
        pattern_score = float(np.mean(pattern_flags)) if pattern_flags else 0.0

        fin_w     = float(cfg.get('fin_weight', 0.5))
        tech_w    = float(cfg.get('tech_weight', 0.5))
        mkt_w     = float(cfg.get('mkt_weight', 0.0))
        sector_w  = float(cfg.get('sector_weight', 0.0))
        pattern_w = float(cfg.get('pattern_weight', 0.0))  # 기본 0.0 (점수 비반영)

        total_score = (
            fin_score    * fin_w +
            tech_score   * tech_w +
            market_score * mkt_w +
            sector_score * sector_w +
            pattern_score * pattern_w
        )

        return {
            "Ticker": code,
            "Price": int(round(float(close))),
            "Score": round(float(total_score), 4),
            "FinScore": round(float(fin_score), 4),
            "TechScore": round(float(tech_score), 4),
            "MktScore": round(float(market_score), 4),
            "SectorScore": round(float(sector_score), 4),
            "PatternScore": round(float(pattern_score), 4),
            # 패턴 플래그 (디버깅/참고용)
            "MA20Up": bool(ma20_up),
            "AccumVol": bool(accum_vol),
            "HigherLows": bool(hl_trend),
            "Consolidation": bool(consd),
            "YEY": bool(yey),
            "PER": round(float(per_val), 2) if pd.notna(per_val) else None,
            "PBR": round(float(pbr_val), 2) if pd.notna(pbr_val) else None,
            "RSI": round(float(rsi), 2)
        }
    except Exception as ex:
        logger.debug("[%s] 스코어 계산 예외: %s", code, ex)
        return None

# ─────────────────────────── 섹터 다양화 ───────────────────────────
def diversify_by_sector(df_sorted: pd.DataFrame, top_n: int, sector_weight: float) -> pd.DataFrame:
    if top_n <= 0 or df_sorted.empty:
        return df_sorted.iloc[0:0]
    if sector_weight <= 0:
        return df_sorted.head(top_n)

    max_per_sector = max(1, int(np.ceil(top_n * float(sector_weight))))

    if "Sector" in df_sorted.columns:
        sector_series = df_sorted["Sector"]
    else:
        sector_series = pd.Series(["N/A"] * len(df_sorted), index=df_sorted.index)

    counts: Dict[str, int] = {}
    selected_idx: List[Any] = []

    for idx, sec in zip(df_sorted.index, sector_series):
        c = counts.get(sec, 0)
        if c < max_per_sector:
            selected_idx.append(idx)
            counts[sec] = c + 1
        if len(selected_idx) >= top_n:
            break

    if len(selected_idx) < top_n:
        for idx in df_sorted.index:
            if idx not in selected_idx:
                selected_idx.append(idx)
                if len(selected_idx) >= top_n:
                    break

    return df_sorted.loc[selected_idx]

# ─────────────────────────── 메인 ───────────────────────────
def run_screener(date_str: str, market: str, config_path: Optional[str], workers: int, debug: bool):
    logger.info("▶ KIS API 사용 스크리닝 시작 (기준일: %s, 시장: %s)", date_str, market)
    if debug:
        logger.setLevel(logging.DEBUG)

    ensure_output_dir()
    settings = load_settings(config_path)
    if not settings:
        logger.error("설정 로딩 실패로 종료합니다.")
        return

    # 인증 (토큰 재사용은 kis_auth 내부 처리)
    ka.auth(svr='prod')

    screener_params = settings.get("screener_params", {})
    risk_params = settings.get("risk_params", {})

    logger.info(
        "가중치: fin=%.2f, tech=%.2f, mkt=%.2f, sector(점수)=%.2f, pattern=%.2f & 섹터상한(cap)=%.2f | top_n=%s, max_positions=%s",
        float(screener_params.get("fin_weight", 0)),
        float(screener_params.get("tech_weight", 0)),
        float(screener_params.get("mkt_weight", 0)),
        float(screener_params.get("sector_weight", 0)),
        float(screener_params.get("pattern_weight", 0.0)),
        float(screener_params.get("sector_weight", 0)),  # cap에 동일 비율 사용(필요시 분리 가능)
        screener_params.get("top_n"),
        risk_params.get("max_positions"),
    )

    t0 = time.perf_counter()

    with stage("1차 필터링"):
        df_filtered, fixed_date = _filter_initial_stocks(date_str, screener_params, market, risk_params, debug)
        if df_filtered.empty:
            logger.warning("❌ 1차 필터링 결과, 대상 종목이 없습니다.")
            return

    with stage("섹터 보강"):
        df_filtered = _enrich_sector(df_filtered, workers, fixed_date)

    with stage("시장 레짐 계산"):
        regime = _get_market_regime_score(fixed_date, market)
        market_score = 0.7 * regime + 0.3 * 0.5
        comps = _get_market_regime_components(fixed_date, market)
        market_trend = get_market_trend(fixed_date)
        logger.info("시장 레짐 스코어 (가중치 적용): %.3f", market_score)
        logger.info("레짐 구성요소: above_ma50=%.2f, ma50>ma200=%.2f, rsi_term=%.2f",
                    comps["above_ma50"], comps["ma50_gt_ma200"], comps["rsi_term"])
        logger.info("시장 단기 추세(60D MA5/MA20): %s", market_trend)

    with stage("섹터 트렌드 계산"):
        sector_trends = _calculate_sector_trends(fixed_date)

    with stage("상세 분석(스코어링)"):
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
            logger.warning("❌ 2차 스크리닝 결과, 최종 후보가 없습니다.")
            return

    with stage("결과 정렬/다양화/저장"):
        df_scores = pd.DataFrame(results).set_index("Ticker")
        # 충돌 방지: 스코어 DF에는 Sector 컬럼을 넣지 않음(이미 필터 DF에 존재)
        df_scores = df_scores.drop(columns=["Sector"], errors="ignore")

        # df_filtered에 이미 PER/PBR가 있으므로 중복 방지
        df_filtered_no_dup = df_filtered.drop(columns=['PER', 'PBR'], errors='ignore')

        df_final = df_filtered_no_dup.join(df_scores, how="inner")
        df_final = df_final.reset_index().rename(columns={"index": "Ticker"})
        df_sorted = df_final.sort_values("Score", ascending=False)

        top_n = min(int(screener_params.get("top_n", 10)), int(risk_params.get("max_positions", 10)))
        final_candidates = diversify_by_sector(df_sorted, top_n, float(screener_params.get("sector_weight", 0.0)))
        final_candidates = final_candidates.head(top_n)

        # 출력 컬럼 정렬
        cols = ["Ticker","Name","Sector","Price","Score",
                "FinScore","TechScore","MktScore","SectorScore","PatternScore",
                "MA20Up","AccumVol","HigherLows","Consolidation","YEY",
                "PER","PBR","RSI","Marcap","Amount5D"]
        keep = [c for c in cols if c in df_sorted.columns]
        df_sorted = df_sorted[keep + [c for c in df_sorted.columns if c not in keep]]
        final_candidates = final_candidates[keep + [c for c in final_candidates.columns if c not in keep]]

        # 보기 좋은 출력(정수형 표기)
        to_show = final_candidates.copy()
        for c in ["Price", "Marcap", "Amount5D"]:
            if c in to_show.columns:
                to_show[c] = pd.to_numeric(to_show[c], errors="coerce").round(0).astype("Int64")

        with pd.option_context('display.width', 240):
            print("\n--- ⭐ 최종 스크리닝 결과 ⭐ ---")
            print(to_show.to_string(index=False))

        # 저장 (전체/최종)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        full_json  = OUTPUT_DIR / f"screener_full_{fixed_date}_{market}.json"
        full_csv   = OUTPUT_DIR / f"screener_full_{fixed_date}_{market}.csv"
        final_json = OUTPUT_DIR / f"screener_results_{fixed_date}_{market}.json"
        final_csv  = OUTPUT_DIR / f"screener_results_{fixed_date}_{market}.csv"

        df_sorted.to_json(full_json, orient='records', indent=2, force_ascii=False)
        final_candidates.to_json(final_json, orient='records', indent=2, force_ascii=False)
        try:
            df_sorted.to_csv(full_csv, index=False)
            final_candidates.to_csv(final_csv, index=False)
        except Exception:
            pass

        logger.info("전체 랭킹 저장: %s (%s)", full_json, _fsize(full_json))
        logger.info("✅ 스크리닝 완료. %d개 후보 저장: %s (%s)",
                    len(final_candidates), final_json, _fsize(final_json))

    logger.info("총 소요 시간: %.2fs", time.perf_counter() - t0)

# ─────────────────────────────── CLI ─────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="KOSPI/KOSDAQ/KONEX 스크리너")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--market", default=os.getenv("MARKET", "KOSPI"), choices=["KOSPI", "KOSDAQ", "KONEX"])
    parser.add_argument("--config")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "4")))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

# ───────────────────────────── 실행 블록 ─────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    run_screener(args.date, args.market, args.config, max(1, args.workers), args.debug)
