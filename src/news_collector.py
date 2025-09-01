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
    """
    /app/config/.env 우선 → 파일 기준 후보 → CWD 후보 → find_dotenv 순으로 탐색.
    로드 성공 시 경로 문자열을 반환, 없으면 빈 문자열 반환.
    """
    candidates = [
        Path("/app/config/.env"),                                    # 절대 경로 우선
        Path(__file__).resolve().parents[1] / "config" / ".env",     # .../src → /app/config/.env
        Path(__file__).resolve().parent / "config" / ".env",         # 현재 폴더 하위 config/.env
        Path(__file__).resolve().parent / ".env",                    # 현재 폴더 .env
        Path.cwd() / "config" / ".env",                              # CWD/config/.env
        Path.cwd() / ".env",                                         # CWD/.env
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

# .env 로드 및 키 읽기
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
    """제목 정규화: 태그/엔티티 제거, 괄호 내용 제거, 접미 매체명 제거, 공백 정리."""
    if not text:
        return ""
    t = _clean_text(text)
    t = re.sub(r"[\(\[\{（\[｛].*?[\)\]\}）\]｝]", " ", t)  # 괄호류 내용 제거
    t = re.sub(r"\s*[-–—]\s*[^-–—]{0,20}$", " ", t)       # 끝의 ' - 매체명' 제거 시도
    t = " ".join(t.split())
    return t

def _title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def _dedupe_items_by_title(items: List[Dict], threshold: float = DEDUPE_THRESHOLD) -> List[Dict]:
    """제목 유사도로 중복 기사 제거(선입 우선). threshold 이상이면 중복으로 간주."""
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
    """NAVER 키가 없으면 빈 리스트 반환(예외 X). 호출부는 빈 결과를 적절히 처리."""
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
    # 예: 'Fri, 30 Aug 2025 12:34:56 +0900'
    dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z")
    return dt.astimezone(timezone.utc)

def _normalize_stock_dict(d: Dict) -> Dict:
    """screener 결과 키 정규화: Ticker/Name 보정"""
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
    stock = _normalize_stock_dict(stock)
    name, ticker = stock.get("Name"), stock.get("Ticker")
    if not (name and ticker):
        return (str(ticker) if ticker else ""), "종목 정보 누락"

    try:
        # 1) API 호출
        items = _fetch_naver_news_api(f'"{name}"', 100)
        if not items:
            return str(ticker), "뉴스 API 호출 결과 없음"

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
            return str(ticker), "최근 기간 내 뉴스가 없습니다."

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

        return str(ticker), "\n\n---\n\n".join(articles)

    except Exception as e:
        logger.error(f"'{name}'({ticker}) 뉴스 수집 오류: {e}")
        return str(ticker), "뉴스 수집 중 오류가 발생했습니다."

def fetch_news_for_stocks(
    stocks: List[Dict],
    num_articles_per_stock: int = 5,
    days: int = 90,
    max_workers: Optional[int] = None,
) -> Dict[str, str]:
    if not stocks:
        return {}

    cutoff_utc = datetime.now(timezone.utc) - timedelta(days=days)
    news_cache: Dict[str, str] = {}
    if max_workers is None:
        max_workers = max(1, min(10, len(stocks)))

    logger.info(f"📰 {len(stocks)}개 종목 뉴스 병렬 수집 시작... (최근 {days}일)")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_fetch_news_for_single_stock, stock, cutoff_utc, num_articles_per_stock): str(_normalize_stock_dict(stock).get("Name", ""))
            for stock in stocks
        }
        for fut in as_completed(futures):
            stock_name = futures[fut]
            try:
                ticker, text = fut.result()
                if ticker:
                    news_cache[ticker] = text
            except Exception as e:
                logger.error(f"'{stock_name}' 처리 중 예외: {e}", exc_info=True)

    logger.info(f"✅ {len(news_cache)}/{len(stocks)}개 종목 뉴스 완료")
    return news_cache

def run_news_collection_from_results_file(
    results_file: Path, num_articles_per_stock: int = 5, days: int = 90
) -> None:
    t0 = time.perf_counter()

    if not results_file.exists():
        logger.error(f"결과 파일({results_file})이 존재하지 않습니다.")
        return

    stem = results_file.stem  # e.g., "screener_results_YYYYMMDD_KOSPI"
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
    if not news_data:
        logger.warning("수집된 뉴스 없음.")
        return

    out_file = results_file.parent / f"collected_news_{fixed_date}_{market}.json"
    logger.info(f"저장 → {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(news_data, f, ensure_ascii=False, indent=2)

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
        latest = find_latest_file("screener_results_*.json")
        if latest is None:
            logger.error("output/ 폴더에 screener_results_*.json 파일이 없습니다.")
        else:
            logger.info(f"자동 선택: {latest.name}")
            run_news_collection_from_results_file(
                latest, num_articles_per_stock=args.articles, days=args.days
            )
