# src/screener.py

import os
import json
import logging
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock as pykrx

# ───────────────────────────── 기본 설정 ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ───────────────────────────── 경로/상수 ─────────────────────────────
BASE_DIR = Path(__file__).resolve().parent          # 예: /app/src
PROJECT_ROOT = BASE_DIR.parent                      # 예: /app
CWD = Path.cwd()                                    # 실제 실행 디렉토리
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
    """CLI > ENV > CWD > 프로젝트 루트 > 도커 관례 경로 순으로 후보 생성"""
    env_path = os.getenv("CONFIG_PATH", "").strip() or None

    candidates: List[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    if env_path:
        candidates.append(Path(env_path).expanduser())

    # CWD 기준
    candidates += [CWD / "config" / "config.json", CWD / "config.json"]
    # src 기준과 프로젝트 루트 기준
    candidates += [BASE_DIR / ".." / "config" / "config.json",
                   PROJECT_ROOT / "config" / "config.json",
                   PROJECT_ROOT / "config.json"]
    # 도커 볼륨 관례 경로
    candidates += [Path("/config/config.json"), Path("/app/config/config.json")]

    # 정규화
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
        logger.error("설정 파일을 찾지 못했습니다. 전체 후보 개수: %d", len(cands))
        logger.info("디버그: BASE_DIR=%s, PROJECT_ROOT=%s, CWD=%s",
                    str(BASE_DIR), str(PROJECT_ROOT), str(CWD))
        return {}
    try:
        with open(found, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        logger.info("설정 파일 로딩 성공: %s", str(found))
        return cfg
    except Exception as e:
        logger.error("설정 파일 로딩 실패(%s): %s", str(found), e)
        return {}

# ─────────────────────────── 공통 보조 함수 ───────────────────────────
def _norm_code_index(obj: pd.DataFrame) -> pd.DataFrame:
    """인덱스(티커 코드)를 6자리 zero-pad 문자열로 표준화."""
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
    s_num = pd.to_numeric(s, errors="coerce")
    s_num = s_num[np.isfinite(s_num)]
    if s_num.empty:
        logger.info("[%s] 값 없음", name)
        return
    qs = s_num.quantile([0.5, 0.75, 0.9, 0.95]).to_dict()
    logger.info("[%s] 중앙값=%s, P75=%s, P90=%s, P95=%s, 최대=%s",
                name,
                f"{int(qs.get(0.5, 0)):,}",
                f"{int(qs.get(0.75, 0)):,}",
                f"{int(qs.get(0.9, 0)):,}",
                f"{int(qs.get(0.95, 0)):,}",
                f"{int(s_num.max()):,}")

# ─────────────── 데이터 제공 및 기술 지표 계산 (DataProvider) ───────────────
def get_stock_listing(market: str = "KOSPI") -> pd.DataFrame:
    """FinanceDataReader로 시장 종목 목록."""
    logger.info("종목 목록 조회(FDR): market=%s", market)
    df = fdr.StockListing(market)
    if 'Code' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns:
            df.rename(columns={'index': 'Code'}, inplace=True)
    df = df.set_index('Code')
    df = _norm_code_index(df)
    if not {"Name", "Marcap"}.issubset(df.columns):
        df = df.rename(columns={"MarketCap": "Marcap", "종목명": "Name"}, errors="ignore")
    return df

def _enrich_sector_with_fdr_krx(df_base: pd.DataFrame) -> pd.DataFrame:
    """FDR KRX 전체 목록에서 Sector/Industry 보강."""
    try:
        krx = fdr.StockListing("KRX")
        if "Code" in krx.columns:
            krx = krx.set_index("Code")
        krx = _norm_code_index(krx)
        krx = krx.rename(columns={"종목명": "Name"}, errors="ignore")
        cols = [c for c in ["Sector", "Industry"] if c in krx.columns]
        if cols:
            df_base = df_base.join(krx[cols], how="left")
            if "Sector" not in df_base.columns:
                df_base["Sector"] = df_base.get("Industry", "N/A")
            df_base["Sector"] = df_base["Sector"].fillna(df_base.get("Industry")).fillna("N/A")
    except Exception:
        pass
    if "Sector" not in df_base.columns:
        df_base["Sector"] = "N/A"
    df_base["Sector"] = df_base["Sector"].fillna("N/A")
    return df_base

def get_fundamentals(date_str: str, market: str = "KOSPI") -> pd.DataFrame:
    """pykrx로 펀더멘털 조회."""
    logger.info("펀더멘털 조회(PYKRX): date=%s market=%s", date_str, market)
    df = pykrx.get_market_fundamental_by_ticker(date_str, market=market)
    return _norm_code_index(df)

def get_historical_prices(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """pykrx→fdr 순서로 과거시세 조회 폴백."""
    try:
        df = pykrx.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if df is not None and not df.empty:
            df = df.rename(columns={'시가':'Open', '고가':'High', '저가':'Low', '종가':'Close', '거래량':'Volume'})
            return df
    except Exception as e:
        logger.debug("%s: pykrx 시세 조회 실패(%s). fdr로 전환.", ticker, e)
    try:
        df = fdr.DataReader(ticker, start=start_date, end=end_date)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.debug("%s: fdr 시세 조회 실패(%s).", ticker, e)
    return None

# ────────────────────── 비거래일 보정 & 5D 거래대금 ──────────────────────
def _is_business_day(date_str: str, market: str) -> bool:
    try:
        df = pykrx.get_market_ohlcv_by_ticker(date_str, market=market)
        return df is not None and not df.empty
    except Exception:
        return False

def _has_meaningful_fundamentals(date_str: str, market: str) -> bool:
    """PBR>0 종목이 일정 수 이상이면 의미있는 펀더멘털로 간주."""
    try:
        f = get_fundamentals(date_str, market)
        if f is None or f.empty or "PBR" not in f.columns:
            return False
        pbr_num = pd.to_numeric(f["PBR"], errors="coerce")
        return (pbr_num > 0).sum() >= 50
    except Exception:
        return False

def _resolve_business_date(date_str: str, market: str) -> str:
    """요청 날짜가 비거래일이거나 펀더멘털이 무의미하면 직전 거래일을 반환."""
    dt = datetime.strptime(date_str, "%Y%m%d")
    for _ in range(20):
        d = dt.strftime("%Y%m%d")
        if _is_business_day(d, market) and _has_meaningful_fundamentals(d, market):
            if d != date_str:
                logger.info("비거래일/무의미 데이터 감지 → 기준일 보정: %s → %s", date_str, d)
            return d
        dt -= timedelta(days=1)
    logger.warning("직전 거래일 탐지 실패. 원래 날짜를 사용합니다: %s", date_str)
    return date_str

def _get_trading_value_5d_avg(date_str: str, market: str) -> pd.Series:
    """
    최근 5거래일(기준일 포함 뒤로) 각 티커의 거래대금 평균을 반환(index=ticker, name='Amount5D').
    """
    amounts = []
    dt = datetime.strptime(date_str, "%Y%m%d")
    found, tried = 0, 0
    while found < 5 and tried < 25:
        d = dt.strftime("%Y%m%d")
        try:
            df = pykrx.get_market_ohlcv_by_ticker(d, market=market)
            if df is not None and not df.empty and ("거래대금" in df.columns):
                df = _norm_code_index(df)
                s = df["거래대금"].rename(d).astype("float64")
                amounts.append(s)
                found += 1
        except Exception:
            pass
        dt -= timedelta(days=1)
        tried += 1
    if not amounts:
        logger.warning("거래대금 5D 계산 실패(데이터 없음). 임시로 빈 Series 반환.")
        return pd.Series(dtype="float64", name="Amount5D")
    df_amt = pd.concat(amounts, axis=1)  # index=ticker
    return df_amt.mean(axis=1).rename("Amount5D")

# ────────────────────── 시장 레짐 스코어 ──────────────────────
def _get_market_regime_score(date_str: str, market: str) -> float:
    """
    시장 레짐 스코어 0~1.
    KOSPI: '1001', KOSDAQ: '2001', KONEX는 임시로 KOSPI 코드 사용.
    """
    index_code = "1001"
    if market.upper() == "KOSDAQ":
        index_code = "2001"
    elif market.upper() == "KOSPI":
        index_code = "1001"
    start = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
    try:
        df = pykrx.get_index_ohlcv_by_date(start, date_str, index_code)
        if df is None or df.empty:
            return 0.5
        df = df.rename(columns={'종가': 'Close', '고가': 'High', '저가': 'Low'})
        close = df["Close"]
        if len(close) < 200:
            return 0.5
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        rsi = calculate_rsi(close)
        if any(pd.isna(x) for x in [ma50, ma200, rsi]):
            return 0.5
        tech_like = ((1 if close.iloc[-1] > ma50 else 0) +
                     (1 if ma50 > ma200 else 0) +
                     (max(0, 1 - abs(rsi - 50) / 50))) / 3.0
        return float(tech_like)
    except Exception:
        return 0.5

# ─────────────────────────── 스크리너 핵심 로직 ───────────────────────────
def _filter_initial_stocks(
    date_str: str, cfg: Dict[str, Any], market: str, risk: Dict[str, Any], debug: bool=False
) -> Tuple[pd.DataFrame, str]:
    """
    기본 조건(시총, 5D 거래대금) 1차 필터. PER/PBR는 필터에서 제외(스코어에서 반영).
    반환값: (필터링된 DataFrame, 보정된 기준일 fixed_date)
    """
    logger.info("1차 필터링 시작...")
    try:
        # 1) 기준일 보정
        fixed_date = _resolve_business_date(date_str, market)

        # 2) 기본 표본 + 섹터 보강
        df_all = get_stock_listing(market)                      # index=Code
        df_all = _enrich_sector_with_fdr_krx(df_all)
        fundamentals = get_fundamentals(fixed_date, market)    # index=Code
        amt5 = _get_trading_value_5d_avg(fixed_date, market)   # Series(index=Code)

        # 3) 조인 및 타입 정리
        df_pre = df_all[["Name", "Marcap", "Sector"]].copy()
        df_pre["Marcap"] = pd.to_numeric(df_pre["Marcap"], errors="coerce").fillna(0)
        df_pre = df_pre.join(amt5, how="left")
        df_pre = df_pre.join(fundamentals[["PER", "PBR"]], how="left")

        if debug:
            try:
                df_all.to_csv(OUTPUT_DIR / f"debug_listing_{market}.csv")
                fundamentals.to_csv(OUTPUT_DIR / f"debug_fundamentals_{market}_{fixed_date}.csv")
                df_pre.to_csv(OUTPUT_DIR / f"debug_joined_{market}_{fixed_date}.csv")
                logger.info("디버그 CSV 저장 완료: output/debug_*.csv")
            except Exception as e:
                logger.debug("디버그 CSV 저장 실패: %s", e)

        # 4) 데이터 품질 진단
        per_num = pd.to_numeric(df_pre.get("PER"), errors="coerce")
        pbr_num = pd.to_numeric(df_pre.get("PBR"), errors="coerce")
        logger.info("표본 크기: listing=%d, fundamentals=%d, amt5=%d",
                    len(df_all), len(fundamentals), amt5.shape[0])
        logger.info("결측치: Amount5D NaN=%d, PER NaN=%d, PBR NaN=%d",
                    df_pre["Amount5D"].isna().sum(),
                    per_num.isna().sum() if per_num is not None else -1,
                    pbr_num.isna().sum() if pbr_num is not None else -1)
        if per_num is not None and pbr_num is not None:
            logger.info("양수 건수: PER>0=%d, PBR>0=%d",
                        (per_num > 0).sum(), (pbr_num > 0).sum())
        _describe_series("Marcap", df_pre["Marcap"])
        _describe_series("Amount5D", df_pre["Amount5D"])

        # 5) 임계값 적용(마켓캡/거래대금만)
        min_mc = float(cfg.get("min_market_cap", 0))
        min_amt = float(cfg.get("min_trading_value_5d_avg", 0))
        mask_mc  = df_pre["Marcap"] >= min_mc
        mask_amt = pd.to_numeric(df_pre["Amount5D"], errors="coerce").fillna(0) >= min_amt
        df_filtered = df_pre[mask_mc & mask_amt].copy()

        # 6) 화이트/블랙 리스트 적용
        bl = set(str(x).zfill(6) for x in risk.get("blacklist_tickers", []) if x)
        wl = set(str(x).zfill(6) for x in risk.get("whitelist_tickers", []) if x)
        if wl:
            before = len(df_filtered)
            df_filtered = df_filtered[df_filtered.index.isin(wl)]
            logger.info("화이트리스트 적용: %d → %d", before, len(df_filtered))
        if bl:
            before = len(df_filtered)
            df_filtered = df_filtered[~df_filtered.index.isin(bl)]
            logger.info("블랙리스트 적용: %d → %d", before, len(df_filtered))

        logger.info(
            "✅ 1차 필터링 완료: %d → %d 종목 (시장=%s, 기준일=%s, min_mc=%s, min_amt5D=%s)",
            len(df_pre), len(df_filtered), market, fixed_date, f"{int(min_mc):,}", f"{int(min_amt):,}"
        )
        return df_filtered, fixed_date
    except Exception as e:
        logger.error("기본 필터링 중 오류: %s", e, exc_info=True)
        return pd.DataFrame(), date_str

# ─────────────────────────── 스코어 계산 ───────────────────────────
def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    if pd.isna(loss.iloc[-1]) or loss.iloc[-1] == 0:
        return 50.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    df_cap = df.copy()
    df_cap.columns = [col.capitalize() for col in df_cap.columns]
    high_low = df_cap['High'] - df_cap['Low']
    high_close = np.abs(df_cap['High'] - df_cap['Close'].shift())
    low_close = np.abs(df_cap['Low'] - df_cap['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean().iloc[-1]

def _safe_fin_term_from_per(per: float) -> float:
    if per is None or not np.isfinite(per) or per <= 0:
        return 0.0
    return max(0.0, min(1.0, (50.0 - per) / 50.0))

def _safe_fin_term_from_pbr(pbr: float) -> float:
    if pbr is None or not np.isfinite(pbr) or pbr <= 0:
        return 0.0
    return max(0.0, min(1.0, (5.0 - pbr) / 5.0))

def _calculate_scores_for_ticker(
    code: str,
    date_str: str,
    fin_info: pd.Series,
    cfg: Dict[str, Any],
    market_score: float
) -> Optional[Dict[str, Any]]:
    try:
        start_dt_str = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
        df_price = get_historical_prices(code, start_dt_str, date_str)
        if df_price is None or df_price.empty or len(df_price) < 200:
            return None

        close = float(df_price["Close"].iloc[-1])
        ma50 = float(df_price["Close"].rolling(50).mean().iloc[-1])
        ma200 = float(df_price["Close"].rolling(200).mean().iloc[-1])
        rsi = float(calculate_rsi(df_price["Close"]))
        if any(pd.isna(x) for x in [ma50, ma200, rsi]):
            return None

        tech_score = ((1 if close > ma50 else 0) +
                      (1 if ma50 > ma200 else 0) +
                      (max(0, 1 - abs(rsi - 50) / 50))) / 3

        per_val = float(fin_info.get("PER", np.nan)) if pd.notna(fin_info.get("PER", np.nan)) else np.nan
        pbr_val = float(fin_info.get("PBR", np.nan)) if pd.notna(fin_info.get("PBR", np.nan)) else np.nan
        per_term = _safe_fin_term_from_per(per_val)
        pbr_term = _safe_fin_term_from_pbr(pbr_val)
        fin_score = 0.5 * (per_term + pbr_term)

        fin_w  = float(cfg.get('fin_weight', 0.5))
        tech_w = float(cfg.get('tech_weight', 0.5))
        mkt_w  = float(cfg.get('mkt_weight', 0.0))
        total_score = fin_score * fin_w + tech_score * tech_w + market_score * mkt_w

        return {
            "Ticker": code,
            "Name": fin_info.get("Name", ""),
            "Sector": fin_info.get("Sector", "N/A"),
            "Price": int(round(close)),
            "Score": round(float(total_score), 4),
            "FinScore": round(float(fin_score), 4),
            "TechScore": round(float(tech_score), 4),
            "MktScore": round(float(market_score), 4),
            "PER": round(per_val, 2) if np.isfinite(per_val) else None,
            "PBR": round(pbr_val, 2) if np.isfinite(pbr_val) else None,
            "RSI": round(rsi, 2)
        }
    except Exception as ex:
        logger.debug("[%s] 스코어 계산 예외: %s", code, ex)
        return None

# ─────────────────────────── 섹터 다양화 ───────────────────────────
def diversify_by_sector(df_sorted: pd.DataFrame, top_n: int, sector_weight: float) -> pd.DataFrame:
    """
    섹터별 상한을 적용해 상위 랭킹에서 편중을 줄인다.
    섹터당 최대 = ceil(top_n * sector_weight). sector_weight<=0이면 상한 미적용.
    반환: df_sorted에서 순서를 유지한 채 상위 top_n 행 선택.
    """
    if top_n <= 0 or df_sorted.empty:
        return df_sorted.iloc[0:0]
    if sector_weight <= 0:
        return df_sorted.head(top_n)

    max_per_sector = max(1, int(np.ceil(top_n * float(sector_weight))))

    # 섹터 시리즈 확보(없으면 N/A)
    if "Sector" in df_sorted.columns:
        sector_series = df_sorted["Sector"]
    else:
        sector_series = pd.Series(["N/A"] * len(df_sorted), index=df_sorted.index)

    counts: Dict[str, int] = {}
    selected_idx: List[Any] = []

    # 정렬된 DataFrame의 인덱스 순회하며 상한 적용
    for idx, sec in zip(df_sorted.index, sector_series):
        c = counts.get(sec, 0)
        if c < max_per_sector:
            selected_idx.append(idx)
            counts[sec] = c + 1
        if len(selected_idx) >= top_n:
            break

    # 섹터 상한 때문에 부족하면 남는 것 채우기(중복 없이)
    if len(selected_idx) < top_n:
        for idx in df_sorted.index:
            if idx not in selected_idx:
                selected_idx.append(idx)
                if len(selected_idx) >= top_n:
                    break

    return df_sorted.loc[selected_idx]

# ─────────────────────────── 메인 ───────────────────────────
def run_screener(date_str: str, market: str, config_path: Optional[str], workers: int, debug: bool):
    logger.info("▶ KIS API 비사용 스크리닝 시작 (기준일: %s, 시장: %s)", date_str, market)
    logger.info("환경 요약: CONFIG_PATH(env)=%s, CLI=%s, WORKERS=%d",
                os.getenv("CONFIG_PATH", None), config_path, workers)
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("DEBUG 모드 활성화")

    ensure_output_dir()
    settings = load_settings(config_path)
    if not settings:
        logger.error("설정 로딩 실패로 종료합니다.")
        return

    screener_params = settings.get("screener_params", {})
    risk_params = settings.get("risk_params", {})

    # 가중치/제약 로그
    logger.info("가중치 요약: fin=%.2f, tech=%.2f, mkt=%.2f, sector_cap=%.2f",
                float(screener_params.get('fin_weight', 0)),
                float(screener_params.get('tech_weight', 0)),
                float(screener_params.get('mkt_weight', 0)),
                float(screener_params.get('sector_weight', 0)))

    # top_n은 max_positions와 동기화 (상한)
    top_n_conf = int(screener_params.get("top_n", 10))
    max_positions = int(risk_params.get("max_positions", top_n_conf))
    top_n = min(top_n_conf, max_positions)

    # 1차 필터(블랙/화이트리스트 포함), 보정일 획득
    df_filtered, fixed_date = _filter_initial_stocks(
        date_str, screener_params, market, risk_params, debug=debug
    )
    if df_filtered.empty:
        logger.warning("❌ 1차 필터링 결과, 대상 종목이 없습니다.")
        return

    # 시장 레짐 스코어(보정된 기준일 기반) 계산
    regime = _get_market_regime_score(fixed_date, market)
    # (선택) 스무딩: 70% 레짐 + 30% 중립(0.5)
    market_score = 0.7 * regime + 0.3 * 0.5
    logger.info("시장 레짐 스코어: %.3f", market_score)

    # 상세 분석
    results: List[Dict[str, Any]] = []
    total = len(df_filtered)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_calculate_scores_for_ticker, code, fixed_date, row, screener_params, market_score): code
            for code, row in df_filtered.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            if i == 1 or i % 50 == 0 or i == total:
                logger.info("  >> 상세 분석 진행률: %d/%d (%.1f%%)", i, total, i * 100.0 / max(1, total))
            res = fut.result()
            if res:
                results.append(res)

    if not results:
        logger.warning("❌ 2차 스크리닝 결과, 최종 후보가 없습니다.")
        return

    df_final = pd.DataFrame(results)

    # 섹터/마켓캡/거래대금 보강
    if "Sector" not in df_final.columns and "Sector" in df_filtered.columns:
        df_final = df_final.join(df_filtered["Sector"], on="Ticker", how="left")

    # df_filtered의 Marcap/Amount5D 매핑
    if "Marcap" in df_filtered.columns:
        marcap_map = df_filtered["Marcap"]
        df_final["Marcap"] = df_final["Ticker"].map(marcap_map)
    if "Amount5D" in df_filtered.columns:
        amt5_map = df_filtered["Amount5D"]
        df_final["Amount5D"] = df_final["Ticker"].map(amt5_map)

    # 전체 랭킹 정렬
    df_sorted = df_final.sort_values("Score", ascending=False)

    # 섹터 다양화 적용 (옵션)
    sector_weight = float(screener_params.get("sector_weight", 0.0))
    final_candidates = diversify_by_sector(df_sorted, top_n, sector_weight)
    final_candidates = final_candidates.head(top_n)

    # 출력 컬럼 정렬
    keep_cols = ["Ticker","Name","Sector","Price",
                 "Score","FinScore","TechScore","MktScore",
                 "PER","PBR","RSI","Marcap","Amount5D"]
    ordered_cols = [c for c in keep_cols if c in df_sorted.columns] + \
                   [c for c in df_sorted.columns if c not in keep_cols]
    df_sorted = df_sorted[ordered_cols]
    final_candidates = final_candidates[ordered_cols]

    print("\n--- ⭐ 최종 스크리닝 결과 ⭐ ---")
    print(final_candidates.to_string(index=False))

    # 저장 (보정된 기준일 사용) - 전체/최종 모두 저장
    full_json = OUTPUT_DIR / f"screener_full_{fixed_date}_{market}.json"
    full_csv  = OUTPUT_DIR / f"screener_full_{fixed_date}_{market}.csv"
    final_json = OUTPUT_DIR / f"screener_results_{fixed_date}_{market}.json"
    final_csv  = OUTPUT_DIR / f"screener_results_{fixed_date}_{market}.csv"

    df_sorted.to_json(full_json, orient='records', indent=2, force_ascii=False)
    final_candidates.to_json(final_json, orient='records', indent=2, force_ascii=False)
    try:
        df_sorted.to_csv(full_csv, index=False)
        final_candidates.to_csv(final_csv, index=False)
    except Exception:
        pass

    logger.info("전체 랭킹 저장: %s", full_json)
    logger.info("✅ 스크리닝 완료. %d개 후보 저장: %s", len(final_candidates), final_json)

# ─────────────────────────────── CLI ───────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="KOSPI/KOSDAQ/KONEX 스크리너")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y%m%d"),
                        help="기준일(YYYYMMDD). 기본=오늘")
    parser.add_argument("--market", type=str, default=os.getenv("MARKET", "KOSPI"),
                        choices=["KOSPI", "KOSDAQ", "KONEX"],
                        help="시장 선택(KOSPI/KOSDAQ/KONEX)")
    parser.add_argument("--config", type=str, default=None,
                        help="config.json 경로(미지정 시 ENV CONFIG_PATH 또는 다중 후보 탐색)")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "4")),
                        help="동시 작업자 수")
    parser.add_argument("--debug", action="store_true", help="디버그 모드: 상세 로그 및 중간 CSV 저장")
    return parser.parse_args()

# ───────────────────────────── 실행 블록 ─────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    run_screener(
        date_str=args.date,
        market=args.market,
        config_path=args.config,
        workers=max(1, args.workers),
        debug=bool(args.debug),
    )
