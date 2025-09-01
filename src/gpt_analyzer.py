# src/gpt_analyzer.py

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv, find_dotenv

# ────────────── 공통 유틸 사용 ──────────────
from utils import (
    setup_logging,
    OUTPUT_DIR,
    load_config,
    find_latest_file,
)

# ────────────── 로깅 ──────────────
setup_logging()
logger = logging.getLogger("gpt_analyzer")

# ────────────── .env 로딩 ──────────────
def _load_env() -> None:
    candidates = [
        Path("/app/config/.env"),
        Path(__file__).resolve().parents[1] / "config" / ".env",
        Path.cwd() / "config" / ".env",
        Path.cwd() / ".env",
    ]
    loaded = ""
    for p in candidates:
        try:
            if p.is_file():
                load_dotenv(dotenv_path=p, override=False)
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

_load_env()

# ────────────── 외부 의존(리포 내부) ──────────────
try:
    from src.screener import get_market_trend
except Exception:
    def get_market_trend(date_str: str) -> str:
        return "Sideways"

# ────────────── Config 및 OpenAI ──────────────
_config: Dict[str, Any] = load_config() or {}
_gpt_params = _config.get("gpt_params", {})
OPENAI_MODEL = _gpt_params.get("openai_model", "gpt-4o-mini")
_strategy_weights = _gpt_params.get("strategy_weights", {})

client = None
try:
    from openai import OpenAI
    _api_key = os.getenv("OPENAI_API_KEY", "")
    if _api_key:
        client = OpenAI(api_key=_api_key)
        _OPENAI_AVAILABLE = True
    else:
        _OPENAI_AVAILABLE = False
except Exception:
    _OPENAI_AVAILABLE = False

# ────────────── 프롬프트 템플릿 ──────────────
INITIAL_FILTER_PROMPT_TMPL = (
    "You are a fast-paced hedge fund analyst. Your task is to perform a quick initial screening. "
    "Based on the provided score and news, decide if this stock is worth a deeper look for a short-term swing trade. "
    "Be decisive. Your entire response must be a single, valid JSON object.\n\n"
    "**Input Data:**\n"
    "- Stock Name: {name}\n"
    "- Quantitative Score: {score}\n"
    "- Recent News Snippet (up to 1500 chars):\n"
    "{news_text}\n\n"
    "**Decision Guidelines:**\n"
    "1. If news is clearly negative (scandal, lawsuit, poor earnings), lean towards \"보류\".\n"
    "2. If the quantitative score is very low (e.g., under 60), be skeptical unless the news provides a very strong catalyst.\n"
    "3. Look for a combination of a good score AND positive news.\n\n"
    "**--- Required JSON Output (in Korean) ---**\n"
    "{{\"decision\": \"<매수 고려 or 보류>\", \"reason\": \"<The core reason for your decision in a single, concise Korean sentence.>\"}}"
)

TACTICAL_PLAN_PROMPT_TMPL = (
    "You are a Chief Investment Strategist creating a final, actionable trade plan. Analyze all provided data meticulously. "
    "Your entire response must be a single, valid JSON object without any other text or explanations. "
    "Pay close attention to the Score Breakdown; a high Sector score in a weak Market can indicate a market-leading theme. "
    "Also, use the Pattern Analysis to strengthen your conviction.\n\n"
    "**Market & Stock Data:**\n"
    "- Overall Market Trend: {market_trend}\n"
    "- Company: {name} ({ticker})\n"
    "- Sector: {stock_sector}\n"
    "- Price: {price:,.0f} KRW\n\n"
    "**Scoring Profile:**\n"
    "- Overall Score: {score}\n"
    "- Score Breakdown: Fin={fin_score:.4f}, Tech={tech_score:.4f}, Market={mkt_score:.4f}, Sector={sector_score:.4f}\n"
    "- PatternScore: {pattern_score:.4f}\n\n"
    "- Financials: PER={per}, PBR={pbr}\n"
    "- Technicals:\n"
    "  • RSI: {rsi:.2f}\n"
    "  • MA50: {ma50:,.0f}\n"
    "  • MA200: {ma200:,.0f}\n\n"
    "**Chart & Volume Pattern Analysis:**\n"
    "- Is MA20 Rising?: {is_ma20_rising}\n"
    "- Accumulation Volume Detected?: {is_accumulation_volume}\n"
    "- Higher Lows Pattern?: {has_higher_lows}\n"
    "- Post-Surge Consolidation?: {is_consolidating}\n"
    "- Yang-Eum-Yang Candle Pattern?: {has_yey_pattern}\n\n"
    "- Recent News:\n"
    "{news_text}\n\n"
    "**Available High-Level Strategies (Choose ONE):**\n"
    "1. `RsiReversalStrategy`\n"
    "2. `TrendFollowingStrategy`\n"
    "3. `AdvancedTechnicalStrategy`\n"
    "4. `DynamicAtrStrategy`\n"
    "5. `BaseStrategy`\n\n"
    "**--- Required JSON Object Structure ---**\n"
    "{{\n"
    "  \"결정\": \"<매수 or 보류>\",\n"
    "  \"분석\": \"<**[중요]** '매수' 또는 '보류' 결정에 대한 핵심적인 종합 분석. **특히 Score Breakdown(특히 Sector 점수)과 Pattern Analysis 결과를 반드시 분석에 반영할 것.** 결정을 '보류'할 경우, 그 이유를 여기에 명확히 서술할 것.>\",\n"
    "  \"전략_클래스\": \"<위 5가지 전략 중 가장 적합한 것 하나를 선택>\",\n"
    "  \"매매전술\": \"<'집중 투자 전략' 또는 '분할 매수 전략' 중 선택>\",\n"
    "  \"parameters\": {{\"installments\": []}}\n"
    "}}"
)

# ────────────── 헬퍼들 ──────────────
def _read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _detect_files(fixed_date: str, market: str):
    screener_file = OUTPUT_DIR / f"screener_results_{fixed_date}_{market}.json"
    news_file     = OUTPUT_DIR / f"collected_news_{fixed_date}_{market}.json"
    return screener_file, news_file

def _normalize_candidates(cands: List[Dict]) -> List[Dict]:
    out = []
    for c in cands:
        item = dict(c)
        if not item.get("Ticker"):
            if item.get("Code"):
                item["Ticker"] = str(item["Code"]).zfill(6)
        if not item.get("Name"):
            for k in ["Name", "종목명", "name"]:
                if c.get(k):
                    item["Name"] = str(c[k])
                    break

        def _f(key, default=0.0):
            try:
                return float(item.get(key, default))
            except Exception:
                return float(default)

        item["Score"]        = _f("Score")
        item["FinScore"]     = _f("FinScore")
        item["TechScore"]    = _f("TechScore")
        item["MktScore"]     = _f("MktScore")
        item["SectorScore"]  = _f("SectorScore")
        item["PatternScore"] = _f("PatternScore")
        item["RSI"]          = _f("RSI")
        item["Price"]        = _f("Price", 0.0)
        item["PER"]          = _f("PER", 0.0) if item.get("PER") is not None else None
        item["PBR"]          = _f("PBR", 0.0) if item.get("PBR") is not None else None

        for b in ["MA20Up","AccumVol","HigherLows","Consolidation","YEY"]:
            item[b] = bool(item.get(b, False))

        out.append(item)
    return out

def _strip_to_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if candidates:
        return max(candidates, key=len)
    return text

def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None

def _call_openai_json(system_prompt: str, user_prompt: str) -> Optional[dict]:
    if not _OPENAI_AVAILABLE or client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content or ""
        return _safe_json_loads(_strip_to_json(txt))
    except Exception as e:
        logger.warning(f"OpenAI 호출 실패: {e}")
        return None

def _pretty_print_plans(plans: List[Dict]) -> None:
    if not plans:
        print("\n--- 🙅 생성된 매수 계획 없음 ---")
        return
    print("\n=== ✨ 생성된 매수 계획 ===")
    for i, plan in enumerate(plans, 1):
        stock = plan.get("stock_info", {})
        name = stock.get("Name", "N/A"); ticker = stock.get("Ticker", "N/A")
        strategy = plan.get("전략_클래스", "N/A"); tactic = plan.get("매매전술", "N/A")
        decision = plan.get("결정", "N/A"); reason = (plan.get("분석", "") or "")[:300]
        stop_px = stock.get("손절가"); tgt_px = stock.get("목표가"); source = stock.get("source")
        print(f"\n[{i}] {name} ({ticker}) - {decision}")
        print(f" - 전략: {strategy}")
        print(f" - 전술: {tactic}")
        if stop_px and tgt_px and decision == "매수":
            print(f" - 손절/목표: {int(round(stop_px)):,} / {int(round(tgt_px)):,}  (source={source})")
        print(f" - 근거: {reason}...")

def _apply_strategy_weights(selected: str, c: Dict, market_trend: str, weights: Dict[str, float]) -> str:
    rsi = float(c.get("RSI", 50.0))
    ma20 = bool(c.get("MA20Up", False))
    hl = bool(c.get("HigherLows", False))
    cons = bool(c.get("Consolidation", False))
    psc = float(c.get("PatternScore", 0.0))
    tech = float(c.get("TechScore", 0.5))

    tf = 0.5 + (0.2 if ma20 else 0.0) + (0.2 if hl else 0.0) + (0.1 if market_trend in ("Bull","Sideways") else 0.0)
    rr = 0.5 + (0.25 if rsi < 40 else 0.0) + (0.15 if not ma20 else 0.0) + (0.1 if not cons else 0.0)
    at = 0.4 + min(0.6, psc)
    da = 0.4 + min(0.5, tech)
    bs = 0.5

    ctx = {
        "TrendFollowingStrategy": tf,
        "RsiReversalStrategy": rr,
        "AdvancedTechnicalStrategy": at,
        "DynamicAtrStrategy": da,
        "BaseStrategy": bs,
    }
    scored = {k: ctx.get(k, 0.0) * float(weights.get(k, 0.5)) for k in ctx.keys()}
    return max(scored.items(), key=lambda kv: (kv[1], 1.0 if kv[0] == selected else 0.0))[0]

def _initial_filter_gpt(name: str, score: float, news_text: str) -> Optional[dict]:
    sys = "You are a fast, no-nonsense investment analyst. Always reply with a single JSON object only."
    user = INITIAL_FILTER_PROMPT_TMPL.format(name=name, score=score, news_text=news_text[:1500])
    return _call_openai_json(sys, user)

def _tactical_plan_gpt(market_trend: str, c: Dict, news_text: str) -> Optional[dict]:
    name = c.get("Name", "N/A")
    ticker = str(c.get("Ticker", "N/A"))
    sector = c.get("Sector", "N/A")
    price = float(c.get("Price", 0.0))
    ma50 = float(c.get("MA50", 0.0))
    ma200 = float(c.get("MA200", 0.0))
    user = TACTICAL_PLAN_PROMPT_TMPL.format(
        market_trend=market_trend,
        name=name, ticker=ticker, stock_sector=sector, price=price,
        score=float(c.get("Score", 0.0)),
        fin_score=float(c.get("FinScore", 0.0)),
        tech_score=float(c.get("TechScore", 0.0)),
        mkt_score=float(c.get("MktScore", 0.0)),
        sector_score=float(c.get("SectorScore", 0.0)),
        pattern_score=float(c.get("PatternScore", 0.0)),
        per=("null" if c.get("PER") is None else c.get("PER")),
        pbr=("null" if c.get("PBR") is None else c.get("PBR")),
        rsi=float(c.get("RSI", 0.0)),
        ma50=ma50, ma200=ma200,
        is_ma20_rising=bool(c.get("MA20Up", False)),
        is_accumulation_volume=bool(c.get("AccumVol", False)),
        has_higher_lows=bool(c.get("HigherLows", False)),
        is_consolidating=bool(c.get("Consolidation", False)),
        has_yey_pattern=bool(c.get("YEY", False)),
        news_text=news_text[:1500],
    )
    sys = "You are a Chief Investment Strategist. Output must be a single JSON object ONLY."
    return _call_openai_json(sys, user)

def _heuristic_plan(c: Dict, news_text: str, market_trend: str) -> Dict:
    score = float(c.get("Score", 0.0))
    decision = "매수" if score >= 0.65 else "보류"
    base_strategy = "TrendFollowingStrategy" if market_trend == "Bull" else ("RsiReversalStrategy" if market_trend == "Bear" else "BaseStrategy")
    tactic = "분할 매수 전략" if decision == "매수" else "집중 투자 전략"
    reason = f"휴리스틱: Score={score:.3f}, 시장={market_trend}, 뉴스길이={len(news_text)}"
    return {"결정": decision, "분석": reason, "전략_클래스": base_strategy, "매매전술": tactic, "parameters": {"installments": []}}

def analyze_candidates_and_create_plans(
    candidates: List[Dict],
    news_cache: Dict[str, str],
    market_trend: str,
    available_slots: int = 3,
) -> List[Dict]:

    cand_sorted = sorted(candidates, key=lambda x: float(x.get("Score", 0.0)), reverse=True)
    results: List[Dict] = []

    for c in cand_sorted:
        if len(results) >= max(1, int(available_slots)):
            break

        name = c.get("Name", "N/A")
        ticker = str(c.get("Ticker", "N/A"))
        score = float(c.get("Score", 0.0))
        news = (news_cache.get(ticker, "") or "")[:1500]

        passed = True
        if _OPENAI_AVAILABLE:
            js = _initial_filter_gpt(name=name, score=score, news_text=news)
            if js and isinstance(js, dict) and js.get("decision") and "보류" in js["decision"]:
                passed = False
            logger.info(f"[Initial] {name}({ticker}) → {js.get('decision') if js else '실패'}")
        elif score < 0.6 and len(news) < 200:
            passed = False

        if not passed:
            continue

        plan_js = _tactical_plan_gpt(market_trend=market_trend, c=c, news_text=news) if _OPENAI_AVAILABLE else _heuristic_plan(c, news, market_trend)
        if not (plan_js and isinstance(plan_js, dict) and plan_js.get("결정")):
            plan_js = _heuristic_plan(c, news, market_trend)

        sel = plan_js.get("전략_클래스", "BaseStrategy")
        best = _apply_strategy_weights(selected=sel, c=c, market_trend=market_trend, weights=_strategy_weights)
        if best != sel:
            plan_js["전략_클래스"] = best

        stock_info = {k: v for k, v in c.items()}
        merged = {"rank": len(results) + 1, "stock_info": stock_info, **plan_js}
        results.append(merged)

    return results

def run_pipeline(
    fixed_date: Optional[str] = None,
    market: str = "KOSPI",
    available_slots: int = 3
) -> Optional[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not fixed_date:
        latest = find_latest_file("screener_results_*_*.json")
        if not latest:
            logger.error("screener_results_*.json 파일을 찾지 못했습니다.")
            return None
        parts = latest.stem.split("_")
        fixed_date, market = (parts[-2], parts[-1]) if len(parts) >= 4 else (None, market)
        if not fixed_date:
            logger.error(f"파일명에서 날짜/시장 추출 실패: {latest.name}")
            return None

    screener_file, news_file = _detect_files(fixed_date, market)
    if not screener_file.exists():
        logger.error(f"스크리너 결과 없음: {screener_file}")
        return None
    if not news_file.exists():
        logger.error(f"뉴스 결과 없음: {news_file} (먼저 news_collector 실행 필요)")
        return None

    logger.info(f"로드: {screener_file.name}, {news_file.name}")
    candidates: List[Dict] = _normalize_candidates(_read_json(screener_file))
    news_cache: Dict[str, str] = _read_json(news_file)
    market_trend = get_market_trend(fixed_date)
    logger.info(f"시장 추세: {market_trend}")

    plans = analyze_candidates_and_create_plans(candidates, news_cache, market_trend, available_slots)
    _pretty_print_plans(plans)

    out_path = OUTPUT_DIR / f"gpt_trades_{fixed_date}_{market}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plans or [], f, ensure_ascii=False, indent=2)
    logger.info(f"저장 완료 → {out_path}")
    return out_path

# ────────────── CLI ──────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Screener + News + GPT Analyzer Pipeline")
    parser.add_argument("--date", help="YYYYMMDD (미지정 시 최신 파일 자동 탐색)")
    parser.add_argument("--market", default="KOSPI", choices=["KOSPI", "KONEX", "KOSDAQ"])
    parser.add_argument("--slots", type=int, default=3, help="생성할 최대 매수 계획 개수")
    args = parser.parse_args()

    run_pipeline(fixed_date=args.date, market=args.market, available_slots=args.slots)
