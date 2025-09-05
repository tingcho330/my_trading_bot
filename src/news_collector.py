# src/news_collector.py

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import re
import os
from typing import Dict, List, Tuple, Optional
import difflib

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───────────────── 공통 유틸 (KST 로깅/경로) ─────────────────
from utils import (
    setup_logging,
    OUTPUT_DIR,
    find_latest_file,
)

# ───────────────── 로깅 설정 ─────────────────
setup_logging()
logger = logging.getLogger("news_collector")

# ───────────────── .env 로딩 (고정 경로 + 폴백) ─────────────────
def load_env_with_fallback() -> str:
    candidates = [
        Path("/app/config/.env"),
        Path(__file__).resolve().parents[1] / "config" / ".env",
        Path(__file__).resolve().parent / "config" / ".env",
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / "config" / ".env",
        Path.cwd() / ".env",
    ]
    loaded = ""
    for p in candidates:
        try:
            if p.is_file():
                if load_dotenv(dotenv_path=p, override=False):
                    loaded = str(p)
                    break
        except Exception:
            continue
    if not loaded:
        try:
            found = find_dotenv(usecwd=True)
            if found:
                load_dotenv(found, override=False)
                loaded = found
        except Exception:
            pass
    logger.info(f".env loaded from: {loaded if loaded else 'None'}")
    return loaded

_ = load_env_with_fallback()
NAVER_ID = os.getenv("NAVER_CLIENT_ID", "").strip()
NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET", "").strip()
DEDUPE_THRESHOLD = float(os.getenv("NEWS_TITLE_SIM_THRESHOLD", "0.85"))

if not (NAVER_ID and NAVER_SECRET):
    logger.warning("NAVER API 키가 비었습니다. (/app/config/.env 확인) 예: NAVER_CLIENT_ID=xxx, NAVER_CLIENT_SECRET=yyy")

# ─────────────────────── 유틸 함수 ───────────────────────
def _clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")
    return " ".join(text.split())

def _normalize_title(text: str) -> str:
    if not text:
        return ""
    t = _clean_text(text)
    t = re.sub(r"[\(\[\{（\[｛].*?[\)\]\}）\]｝]", " ", t)
    t = re.sub(r"\s*[-–—]\s*[^-–—]{0,20}$", " ", t)
    t = " ".join(t.split())
    return t

def _title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def _dedupe_items_by_title(items: List[Dict], threshold: float = DEDUPE_THRESHOLD) -> List[Dict]:
    selected: List[Dict] = []
    norm_titles: List[str] = []
    for it in items:
        title = _normalize_title(it.get("title", "") or "")
        if not title:
            selected.append(it)
            norm_titles.append(title)
            continue
        dup = any(prev and _title_similarity(title, prev) >= threshold for prev in norm_titles)
        if not dup:
            selected.append(it)
            norm_titles.append(title)
    return selected

# ───────────────── NAVER API / 스크레이핑 ─────────────────
def _fetch_naver_news_api(keyword: str, num_articles: int) -> List[Dict]:
    if not (NAVER_ID and NAVER_SECRET):
        return []
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": NAVER_ID, "X-Naver-Client-Secret": NAVER_SECRET}
    params = {"query": keyword, "display": num_articles, "sort": "date"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("items", []) if isinstance(data, dict) else []

def _scrape_article_content(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, "html.parser")

        selectors = [
            "#articleBodyContents",
            "div.news_end",
            "#articletxt",
            ".article_body",
            "[itemprop='articleBody']",
            "#article-view-content-div",
            "article",
        ]
        container = next((soup.select_one(s) for s in selectors if soup.select_one(s)), None)
        if not container:
            return _clean_text(soup.get_text())

        for unwanted_tag in container.select("script, style, .ad, .aside, .related-news, figure, .promotion"):
            unwanted_tag.decompose()

        return _clean_text(container.get_text())
    except Exception as e:
        logger.warning(f"본문 스크레이핑 실패({url}): {e}")
        return "본문을 가져오지 못했습니다."

def _parse_pubdate(pub: str) -> datetime:
    dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z")
    return dt.astimezone(timezone.utc)

def _normalize_stock_dict(d: Dict) -> Dict:
    out = dict(d)
    if not out.get("Ticker"):
        if out.get("Code"):
            out["Ticker"] = str(out["Code"]).zfill(6)
        elif out.get("ticker"):
            out["Ticker"] = str(out["ticker"]).zfill(6)
    if not out.get("Name"):
        for k in ["Name", "종목명", "name"]:
            if d.get(k):
                out["Name"] = str(d[k])
                break
    return out

# ───────────────── 뉴스 수집 코어 ─────────────────
def _fetch_news_for_single_stock(
    stock: Dict, cutoff_utc: datetime, num_articles: int
) -> Tuple[str, str]:
    """
    반환: (ticker, text)
    - 데이터 부족 시 text는 '[NO_NEWS] ...' 형식으로 반환
    - 오류 시 text는 '[ERROR] ...' 형식
    """
    stock = _normalize_stock_dict(stock)
    name, ticker = stock.get("Name"), stock.get("Ticker")
    if not (name and ticker):
        return (str(ticker) if ticker else ""), "[NO_NEWS] 종목 정보 누락"

    try:
        # 1) API 호출
        items = _fetch_naver_news_api(f'"{name}"', 100)
        if not items:
            return str(ticker), "[NO_NEWS] 뉴스 API 호출 결과 없음"

        # 2) 기간 필터
        recent: List[Dict] = []
        for it in items:
            try:
                pub = _parse_pubdate(it["pubDate"])
                if pub >= cutoff_utc:
                    recent.append(it)
            except Exception:
                continue
        if not recent:
            return str(ticker), "[NO_NEWS] 최근 기간 내 뉴스 없음"

        # 3) 중복 제거 (제목 유사도)
        recent = _dedupe_items_by_title(recent, threshold=DEDUPE_THRESHOLD)

        # 4) 스크래핑 및 포맷
        articles = []
        for it in recent[:num_articles]:
            raw_link = it.get("link") or ""
            link = raw_link if "naver.com" in raw_link else it.get("originallink") or raw_link
            title = _clean_text((it.get("title") or "").strip())
            desc = it.get("description")

            content = _scrape_article_content(link) if link else "원문 링크 없음"
            parts = [f"제목: {title}"]
            if desc:
                parts.append(f"요약: {_clean_text(desc)}")
            parts.extend([f"링크: {link or 'N/A'}", f"본문: {content}"])
            articles.append("\n".join(parts))

        if not articles:
            return str(ticker), "[NO_NEWS] 기사 본문 수집 실패"

        return str(ticker), "\n\n---\n\n".join(articles)

    except Exception as e:
        logger.error(f"'{name}'({ticker}) 뉴스 수집 오류: {e}")
        return str(ticker), "[ERROR] 뉴스 수집 중 오류 발생"

def fetch_news_for_stocks(
    stocks: List[Dict],
    num_articles_per_stock: int = 5,
    days: int = 90,
    max_workers: Optional[int] = None,
) -> Dict[str, str]:
    """
    항상 각 티커 키를 포함한 dict를 반환.
    - 기본값은 '[NO_NEWS] 데이터 부족(수집 0건)'으로 채우고, 성공 시 덮어씀.
    (저장 직전에 구조화 포맷으로 통일함)
    """
    if not stocks:
        return {}

    cutoff_utc = datetime.now(timezone.utc) - timedelta(days=days)
    if max_workers is None:
        max_workers = max(1, min(10, len(stocks)))

    # 기본값: 모두 NO_NEWS로 초기화
    base_map: Dict[str, str] = {}
    norm_list: List[Dict] = []
    for stock in stocks:
        s = _normalize_stock_dict(stock)
        t = str(s.get("Ticker", "")).zfill(6)
        if t and s.get("Name"):
            base_map[t] = "[NO_NEWS] 데이터 부족(수집 0건)"
            norm_list.append({"Name": s["Name"], "Ticker": t})

    news_cache: Dict[str, str] = dict(base_map)

    logger.info(f"{len(norm_list)}개 종목 뉴스 병렬 수집 시작... (최근 {days}일)")
    if not norm_list:
        return news_cache

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_fetch_news_for_single_stock, stock, cutoff_utc, num_articles_per_stock): str(stock.get("Name", ""))
            for stock in norm_list
        }
        for fut in as_completed(futures):
            stock_name = futures[fut]
            try:
                ticker, text = fut.result()
                if ticker:
                    news_cache[ticker] = text or "[NO_NEWS] 빈 본문"
            except Exception as e:
                logger.error(f"'{stock_name}' 처리 중 예외: {e}", exc_info=True)

    logger.info(f"✅ 뉴스 수집 완료 (총 {len(news_cache)}종목)")
    return news_cache

# ───────────────── 저장 전 포맷 통일 유틸 ─────────────────
def _to_structured_news_map(news_data: Dict[str, object], all_tickers: List[str]) -> Dict[str, Dict[str, str]]:
    """
    다양한 형태의 news_data 값을 저장 전 통일:
      - dict(status,text) → 그대로
      - str 시작이 "[NO_NEWS]" → {"status":"NO_NEWS","text":<원문 또는 빈 문자열>}
      - 그 외 str → {"status":"OK","text":<원문>}
      - None/빈값/키없음 → {"status":"NO_NEWS","text":""}
    """
    out: Dict[str, Dict[str, str]] = {}
    for t in all_tickers:
        v = news_data.get(t, None)
        if isinstance(v, dict) and "status" in v and "text" in v:
            out[t] = {"status": str(v.get("status")), "text": str(v.get("text") or "")}
        elif isinstance(v, str):
            s = v.strip()
            if s.startswith("[NO_NEWS]"):
                out[t] = {"status": "NO_NEWS", "text": s}
            elif s.startswith("[ERROR]"):
                out[t] = {"status": "ERROR", "text": s}
            else:
                out[t] = {"status": "OK", "text": s}
        else:
            out[t] = {"status": "NO_NEWS", "text": ""}
    return out

def run_news_collection_from_results_file(
    results_file: Path, num_articles_per_stock: int = 5, days: int = 90
) -> None:
    t0 = time.perf_counter()

    if not results_file.exists():
        logger.error(f"결과 파일({results_file})이 존재하지 않습니다.")
        return

    stem = results_file.stem  # e.g., "screener_candidates_YYYYMMDD_KOSPI"
    parts = stem.split("_")
    if len(parts) >= 4:
        fixed_date, market = parts[-2], parts[-1]
    else:
        fixed_date, market = "unknown", "UNKNOWN"

    logger.info(f"결과 파일 로드 → {results_file}")
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            screened_stocks = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"결과 파일 읽기 오류: {e}")
        return

    if not isinstance(screened_stocks, list) or not screened_stocks:
        logger.info("종목이 없어 뉴스 수집 종료.")
        return

    stocks_for_news: List[Dict] = []
    for s in screened_stocks:
        norm = _normalize_stock_dict(s)
        if norm.get("Ticker") and norm.get("Name"):
            stocks_for_news.append({"Name": norm["Name"], "Ticker": str(norm["Ticker"]).zfill(6)})

    if not stocks_for_news:
        logger.warning("Ticker/Name이 유효한 종목이 없습니다.")
        return

    news_data = fetch_news_for_stocks(
        stocks_for_news, num_articles_per_stock=num_articles_per_stock, days=days
    )

    # 파일 저장: 비어 있어도 NO_NEWS 기본값으로 저장되도록 보장
    if not news_data:
        news_data = {s["Ticker"]: "[NO_NEWS] 데이터 부족(수집 0건)" for s in stocks_for_news}

    out_file = results_file.parent / f"collected_news_{fixed_date}_{market}.json"
    logger.info(f"저장 → {out_file}")

    # ✅ 저장 직전: 모든 티커에 대해 구조화 포맷 보장 + 빈 값은 NO_NEWS 채움
    all_tickers = [s['Ticker'] for s in stocks_for_news]
    news_data_struct = _to_structured_news_map(news_data, all_tickers)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(news_data_struct, f, ensure_ascii=False, indent=2)

    logger.info(f"완료 (소요 {time.perf_counter() - t0:.2f}초)")

# ───────────────────────────── CLI ─────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="News Collector (compat with screener outputs)")
    parser.add_argument("--file", type=str, help="스크리너 결과 JSON 경로")
    parser.add_argument("--articles", type=int, default=5, help="종목별 기사 수 (기본 5)")
    parser.add_argument("--days", type=int, default=90, help="최근 N일만 수집 (기본 90)")
    args = parser.parse_args()

    if args.file:
        run_news_collection_from_results_file(
            Path(args.file), num_articles_per_stock=args.articles, days=args.days
        )
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        latest = (
            find_latest_file("screener_candidates_*.json")
            or find_latest_file("screener_candidates_full_*.json")
            or find_latest_file("screener_rank_*.json")
            or find_latest_file("screener_rank_full_*.json")
        )
        if latest is None:
            logger.error("output/ 폴더에 screener_candidates_*.json 또는 screener_rank_*.json 파일이 없습니다.")
        else:
            logger.info(f"자동 선택: {latest.name}")
            run_news_collection_from_results_file(
                latest, num_articles_per_stock=args.articles, days=args.days
            )
