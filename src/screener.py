# src/screener.py

import os
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List, Any
from pprint import pprint
from pathlib import Path

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock as pykrx

# --- 기본 설정 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- 경로 설정 ---
# 이 스크립트 파일(screener.py)을 기준으로 프로젝트의 루트 디렉토리를 찾습니다.
# 컨테이너의 작업 디렉토리(WORKDIR)는 /app 입니다.
# docker-compose.yml 설정에 따라, 로컬의 config 폴더는 컨테이너의 /app/config 에 마운트됩니다.

# [수정된 부분]
# 현재 실행되는 스크립트의 디렉토리 경로 (/app)
BASE_DIR = Path(__file__).resolve().parent
# 올바른 설정 파일 경로 (/app/../config/config.json -> /config/config.json 이 아닌)
# /app 폴더를 기준으로 config 폴더를 찾도록 수정합니다.
CONFIG_PATH = BASE_DIR.parent / "config" / "config.json"
OUTPUT_DIR = BASE_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True) # output 폴더가 없으면 생성
WORKERS = 4  # 동시 작업자 수 (PC 사양에 따라 조절)


# ────────────────── 데이터 제공 및 기술 지표 계산 (DataProvider 통합) ──────────────────

def get_stock_listing(market: str = "KOSPI") -> pd.DataFrame:
    """fdr을 이용해 전체 종목 목록을 가져옵니다."""
    logger.info(f"데이터 소스(fdr)로부터 {market} 전체 종목 목록을 가져옵니다...")
    df = fdr.StockListing(market)
    if 'Code' not in df.columns:
        df.reset_index(inplace=True)
    return df.set_index('Code')

def get_fundamentals(date_str: str, market: str = "KOSPI") -> pd.DataFrame:
    """pykrx를 이용해 펀더멘털 정보를 가져옵니다."""
    logger.info(f"데이터 소스(pykrx)로부터 {date_str} 기준 펀더멘털 정보를 가져옵니다...")
    return pykrx.get_market_fundamental_by_ticker(date_str, market=market)

def get_historical_prices(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """pykrx 또는 fdr을 이용해 과거 시세를 조회합니다."""
    logger.debug(f"{ticker}: 과거 시세 조회를 시작합니다...")
    try:
        df = pykrx.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if not df.empty:
            df.rename(columns={'시가':'Open', '고가':'High', '저가':'Low', '종가':'Close', '거래량':'Volume'}, inplace=True)
            return df
    except Exception as e:
        logger.warning(f"{ticker}: pykrx 시세 조회 실패 ({e}). fdr로 전환합니다.")
    try:
        df = fdr.DataReader(ticker, start=start_date, end=end_date)
        if not df.empty:
            return df
    except Exception as e:
        logger.error(f"{ticker}: fdr 시세 조회도 최종 실패했습니다 ({e}).")
    return None

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    if pd.isna(loss.iloc[-1]) or loss.iloc[-1] == 0: return 50.0
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

# ────────────────── 스크리너 핵심 로직 ──────────────────

def load_settings() -> Dict:
    """config.json 파일에서 스크리너 설정을 불러옵니다."""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"설정 파일 로딩 실패: {e}")
        return {}

def _filter_initial_stocks(date_str: str, cfg: Dict) -> pd.DataFrame:
    """기본 조건(시총, 거래대금, PER/PBR)으로 1차 필터링을 수행합니다."""
    logger.info("1차 기본 필터링 시작...")
    try:
        df_all = get_stock_listing("KOSPI")
        fundamentals = get_fundamentals(date_str, "KOSPI")
        
        df_pre = df_all[["Name", "Marcap", "Amount"]].copy()
        df_pre = df_pre.join(fundamentals[["PER", "PBR"]], how="left")
        df_pre["Amount"] = pd.to_numeric(df_pre["Amount"], errors="coerce").fillna(0)
        
        df_pre = df_pre[
            (df_pre["Marcap"] >= cfg.get("min_market_cap", 0)) &
            (df_pre["Amount"] >= cfg.get("min_trading_value_5d_avg", 0))
        ]
        df_filtered = df_pre.dropna(subset=["PER", "PBR"])
        df_filtered = df_filtered[(df_filtered["PER"] > 0) & (df_filtered["PBR"] > 0)].copy()
        logger.info(f"✅ 1차 필터링 완료: {len(df_all)}개 -> {len(df_filtered)}개 종목")
        return df_filtered
    except Exception as e:
        logger.error(f"기본 필터링 중 오류: {e}", exc_info=True)
        return pd.DataFrame()

def _calculate_scores_for_ticker(code: str, date_str: str, fin_info: pd.Series, cfg: Dict) -> Optional[Dict]:
    """개별 종목의 상세 분석 및 점수 계산을 수행합니다."""
    try:
        start_dt_str = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=365)).strftime("%Y%m%d")
        df_price = get_historical_prices(code, start_dt_str, date_str)
        
        if df_price is None or df_price.empty or len(df_price) < 200:
            return None

        close = df_price["Close"].iloc[-1]
        ma50 = df_price["Close"].rolling(50).mean().iloc[-1]
        ma200 = df_price["Close"].rolling(200).mean().iloc[-1]
        rsi = calculate_rsi(df_price["Close"])

        if any(pd.isna(x) for x in [ma50, ma200, rsi]):
            return None

        tech_score = ((1 if close > ma50 else 0) + (1 if ma50 > ma200 else 0) + (max(0, 1 - abs(rsi - 50) / 50))) / 3
        
        per, pbr = fin_info.get("PER", 999), fin_info.get("PBR", 999)
        fin_score = (max(0, (50 - per)) / 50 + max(0, (5 - pbr)) / 5) / 2

        total_score = (fin_score * cfg.get('fin_weight', 0.5)) + (tech_score * cfg.get('tech_weight', 0.5))
        
        return {
            "Ticker": code, "Name": fin_info["Name"], "Price": int(close), 
            "Score": round(total_score, 4),
            "FinScore": round(fin_score, 4), "TechScore": round(tech_score, 4),
            "PER": per, "PBR": pbr, "RSI": round(rsi, 2)
        }
    except Exception as ex:
        logger.error(f"[{code}] 스코어 계산 중 예외: {ex}")
        return None

def run_screener(date_str: str):
    """스크리너 메인 실행 함수"""
    logger.info(f"▶ KIS API 비사용 스크리닝 시작 (기준일: {date_str})...")
    
    settings = load_settings()
    if not settings: return
    
    screener_params = settings.get("screener_params", {})
    
    df_filtered = _filter_initial_stocks(date_str, screener_params)
    if df_filtered.empty:
        logger.warning("❌ 1차 필터링 결과, 대상 종목이 없습니다.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_calculate_scores_for_ticker, code, date_str, row, screener_params): code
            for code, row in df_filtered.iterrows()
        }
        for i, fut in enumerate(as_completed(futures)):
            logger.info(f"  >> 상세 분석 진행률: {i+1}/{len(df_filtered)}")
            res = fut.result()
            if res:
                results.append(res)
    
    if not results:
        logger.warning("❌ 2차 스크리닝 결과, 최종 후보가 없습니다.")
        return

    df_final = pd.DataFrame(results)
    df_sorted = df_final.sort_values("Score", ascending=False).reset_index(drop=True)
    
    top_n = screener_params.get("top_n", 10)
    final_candidates = df_sorted.head(top_n)

    print("\n--- ⭐ 최종 스크리닝 결과 ⭐ ---")
    print(final_candidates.to_string(index=False))

    final_path = OUTPUT_DIR / f"screener_results_{date_str}.json"
    final_candidates.to_json(final_path, orient='records', indent=2, force_ascii=False)
    logger.info(f"✅ 스크리닝 완료. {len(final_candidates)}개 후보 저장: {final_path}")

# ────────────────── 실행 블록 ──────────────────
if __name__ == '__main__':
    today_str = datetime.now().strftime("%Y%m%d")
    run_screener(date_str=today_str)