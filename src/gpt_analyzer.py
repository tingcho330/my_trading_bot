# src/gpt_analyzer.py
import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv, find_dotenv

# ────────────── 공통 유틸 ──────────────
from utils import (
    setup_logging,
    OUTPUT_DIR,
    load_config,
    find_latest_file,
    # ▼ 예산 컨텍스트를 위해 추가
    get_account_snapshot_cached,
    extract_cash_from_summary,
)

# ────────────── notifier 연동 ──────────────
from notifier import (
    DiscordLogHandler,
    WEBHOOK_URL,
    is_valid_webhook,
    send_discord_message,
)

# ────────────── 로깅 ──────────────
setup_logging()
logger = logging.getLogger("gpt_analyzer")

# 루트 로거에 디스코드 에러 핸들러 장착(중복 방지)
_root = logging.getLogger()
if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

# ── 간단 쿨다운(스팸 방지) ──
_last_sent: Dict[str, float] = {}
def _notify(content: str, key: str, cooldown_sec: int = 120):
    """
    경량 텍스트 알림(최소화). 실패는 무시한다.
    """
    try:
        now = time.time()
        if key not in _last_sent or now - _last_sent[key] >= cooldown_sec:
            _last_sent[key] = now
            if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
                send_discord_message(content=content)
    except Exception:
        pass

def _notify_embed(title: str, description: str, fields: Optional[List[Dict[str, Any]]] = None):
    """
    최종 한 번만 쓰는 임베드 알림(스케줄러 요약과 중복 최소화).
    """
    if not (WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL)):
        return
    try:
        embed = {
            "title": title,
            "description": description,
            "type": "rich",
        }
        if fields:
            embed["fields"] = fields
        send_discord_message(content="", embeds=[embed])
    except Exception as e:
        logger.warning("알림(임베드) 전송 실패: %s", e)

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
_gpt_params = _config.get("gpt_params", {}) or {}
_trading_params = _config.get("trading_params", {}) or {}

OPENAI_MODEL = _gpt_params.get("openai_model", "gpt-4o-mini")
_strategy_weights = _gpt_params.get("strategy_weights", {})

# ▼ 예산 가드 설정
_BUDGET_GUARD_ENABLED: bool = bool(_gpt_params.get("budget_guard", False))
_MAX_ENTRY_PRICE_RATIO: float = float(_gpt_params.get("max_entry_price_ratio", 0.95))
_CASH_BUFFER_RATIO: float = float(_trading_params.get("cash_buffer_ratio", 0.0))

client = None
try:
    from openai import OpenAI
    _api_key = os.getenv("OPENAI_API_KEY", "")
    if _api_key:
        client = OpenAI(api_key=_api_key)
        _OPENAI_AVAILABLE = True
        logger.info("OpenAI 클라이언트 초기화 완료.")
    else:
        _OPENAI_AVAILABLE = False
        logger.warning("OPENAI_API_KEY 미설정. 휴리스틱 모드로 동작합니다.")
        _notify("ℹ️ OPENAI_API_KEY 미설정 → 휴리스틱 분석으로 진행", key="gpt_analyzer_no_api", cooldown_sec=600)
except Exception as e:
    _OPENAI_AVAILABLE = False
    logger.warning(f"OpenAI 초기화 실패: {e}")
    _notify(f"ℹ️ OpenAI 초기화 실패 → 휴리스틱 분석으로 진행\n```{str(e)[:400]}```", key="gpt_analyzer_openai_fail", cooldown_sec=600)

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
    "{news_text}\n"
    "{budget_guard_block}\n"  # ← 예산 가드 블록(조건부 삽입)
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
    preferred = OUTPUT_DIR / f"screener_candidates_{fixed_date}_{market}.json"
    if preferred.exists():
        return preferred, OUTPUT_DIR / f"collected_news_{fixed_date}_{market}.json"

    fallbacks = [
        OUTPUT_DIR / f"screener_candidates_full_{fixed_date}_{market}.json",
        OUTPUT_DIR / f"screener_rank_{fixed_date}_{market}.json",
        OUTPUT_DIR / f"screener_rank_full_{fixed_date}_{market}.json",
    ]
    for fb in fallbacks:
        if fb.exists():
            return fb, OUTPUT_DIR / f"collected_news_{fixed_date}_{market}.json"

    legacy = OUTPUT_DIR / f"screener_results_{fixed_date}_{market}.json"
    return legacy, OUTPUT_DIR / f"collected_news_{fixed_date}_{market}.json"

def _detect_latest_screener_file() -> Optional[Path]:
    patterns = [
        "screener_candidates_*.json",
        "screener_candidates_full_*.json",
        "screener_rank_*.json",
        "screener_rank_full_*.json",
        "screener_results_*_*.json",
    ]
    for pat in patterns:
        p = find_latest_file(pat)
        if p:
            return p
    return None

def _to_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

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
        item["Price"]        = _f("Price", 0.0)
        item["RSI"]          = _f("RSI", 50.0)
        item["ATR"]          = _to_float(item.get("ATR"), None)
        item["MA50"]         = _f("MA50", 0.0)
        item["MA200"]        = _f("MA200", 0.0)

        item["PER"]          = _to_float(item.get("PER"), None) if item.get("PER") is not None else None
        item["PBR"]          = _to_float(item.get("PBR"), None) if item.get("PBR") is not None else None

        for b in ["MA20Up","AccumVol","HigherLows","Consolidation","YEY"]:
            item[b] = bool(item.get(b, False))

        if item.get("Sector") is None and c.get("Industry"):
            item["Sector"] = c.get("Industry")
        item["Sector"] = item.get("Sector", "N/A")
        item["SectorSource"] = item.get("SectorSource", None)

        stop_price = item.get("stop_price", item.get("손절가"))
        target_price = item.get("target_price", item.get("목표가"))
        levels_source = item.get("levels_source", item.get("source"))
        item["stop_price"] = int(round(float(stop_price))) if stop_price is not None else None
        item["target_price"] = int(round(float(target_price))) if target_price is not None else None
        item["levels_source"] = levels_source if levels_source is not None else None

        if "daily_chart" in c:
            item["daily_chart"] = c["daily_chart"]
        if "investor_flow" in c:
            item["investor_flow"] = c["investor_flow"]

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
        _notify(f"⚠️ OpenAI 호출 실패: {str(e)[:400]}", key="gpt_analyzer_call_fail", cooldown_sec=300)
        return None

def _pretty_print_plans(plans: List[Dict]) -> None:
    if not plans:
        print("\n--- 매수 계획 없음 ---")
        return
    print("\n=== ✨ 생성된 매수 계획 ===")
    for i, plan in enumerate(plans, 1):
        stock = plan.get("stock_info", {})
        name = stock.get("Name", "N/A"); ticker = stock.get("Ticker", "N/A")
        strategy = plan.get("전략_클래스", "N/A"); tactic = plan.get("매매전술", "N/A")
        decision = plan.get("결정", "N/A"); reason = (plan.get("분석", "") or "")[:300]
        stop_px = stock.get("stop_price"); tgt_px = stock.get("target_price"); source = stock.get("levels_source")
        print(f"\n[{i}] {name} ({ticker}) - {decision}")
        print(f" - 전략: {strategy}")
        print(f" - 전술: {tactic}")
        if stop_px and tgt_px and decision == "매수":
            try:
                print(f" - 손절/목표: {int(round(float(stop_px))):,} / {int(round(float(tgt_px))):,}  (source={source})")
            except Exception:
                print(f" - 손절/목표: {stop_px} / {tgt_px}  (source={source})")
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

def _build_budget_guard_block(
    enabled: bool,
    usable_cash: Optional[int],
    ratio: float,
    buffer_ratio: float,
    price: float,
    score: float,
) -> str:
    """
    예산 가드 프롬프트 블록을 생성한다(조건부).
    """
    if not enabled or usable_cash is None:
        return ""
    max_entry = int(usable_cash * ratio)
    return (
        "\n**Budget Guard (예산 가드):**\n"
        f"- usable_cash = {usable_cash:,} KRW\n"
        f"- cash_buffer_ratio = {buffer_ratio:.2f}\n"
        f"- max_entry_price_ratio = {ratio:.2f}\n"
        f"- max_allowed_entry_price = {max_entry:,} KRW\n"
        f"- candidate_price = {int(price):,} KRW | candidate_score = {score:.4f}\n\n"
        "지시사항: price <= usable_cash * max_entry_price_ratio를 만족하는 종목만 추천. "
        "없으면 ‘추천 없음(LOW_FUNDS)’라고 답하라.\n"
    )

def _tactical_plan_gpt(
    market_trend: str,
    c: Dict,
    news_text: str,
    budget_ctx: Optional[Dict[str, Any]] = None
) -> Optional[dict]:
    name = c.get("Name", "N/A")
    ticker = str(c.get("Ticker", "N/A"))
    sector = c.get("Sector", "N/A")
    price = float(c.get("Price", 0.0))
    ma50 = float(c.get("MA50", 0.0))
    ma200 = float(c.get("MA200", 0.0))

    budget_block = _build_budget_guard_block(
        enabled=bool(budget_ctx and budget_ctx.get("enabled")),
        usable_cash=budget_ctx.get("usable_cash") if budget_ctx else None,
        ratio=float(budget_ctx.get("max_entry_price_ratio", 0.95)) if budget_ctx else 0.95,
        buffer_ratio=float(budget_ctx.get("cash_buffer_ratio", 0.0)) if budget_ctx else 0.0,
        price=price,
        score=float(c.get("Score", 0.0)),
    )

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
        news_text=(news_text or "")[:1500],
        budget_guard_block=budget_block,
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

def _compose_reason_suffix(c: Dict) -> str:
    sec = c.get("Sector", "N/A")
    sec_src = c.get("SectorSource", "N/A")
    rsi = c.get("RSI", None)
    psc = c.get("PatternScore", None)
    m20 = "▲" if c.get("MA20Up") else "─"
    lv_src = c.get("levels_source", "N/A")
    sp = c.get("stop_price", None)
    tp = c.get("target_price", None)
    parts = [
        f"섹터={sec}({sec_src})",
        f"RSI={rsi:.2f}" if isinstance(rsi, (int, float)) else "RSI=N/A",
        f"PatternScore={psc:.2f}" if isinstance(psc, (int, float)) else "PatternScore=N/A",
        f"MA20={m20}",
        f"레벨={lv_src}" + (f" [SL:{sp:,}/TP:{tp:,}]" if (isinstance(sp,(int,float)) and isinstance(tp,(int,float))) else ""),
    ]
    return " | " + " / ".join(parts)

def _get_usable_cash() -> Optional[int]:
    """
    최신 요약/밸런스 캐시에서 가용 현금을 읽어온다.
    """
    try:
        summary_dict, *_ = get_account_snapshot_cached(
            summary_pattern="summary_*.json",
            balance_pattern="balance_*.json",
            ttl_sec=None,
        )
        cash_map = extract_cash_from_summary(summary_dict) if summary_dict else {}
        usable = cash_map.get("available_cash")
        if isinstance(usable, (int, float)):
            return int(usable)
        return None
    except Exception as e:
        logger.debug(f"usable_cash 로드 실패: {e}")
        return None

def analyze_candidates_and_create_plans(
    candidates: List[Dict],
    news_cache: Dict[str, Any],
    market_trend: str,
    available_slots: int = 3,
    budget_ctx: Optional[Dict[str, Any]] = None,
) -> List[Dict]:

    cand_sorted = sorted(candidates, key=lambda x: float(x.get("Score", 0.0)), reverse=True)
    results: List[Dict] = []

    for c in cand_sorted:
        if len(results) >= max(1, int(available_slots)):
            break

        name = c.get("Name", "N/A")
        ticker = str(c.get("Ticker", "N/A"))
        score = float(c.get("Score", 0.0))

        raw_news = news_cache.get(ticker, "")
        news_status = None
        news_text = ""
        if isinstance(raw_news, dict):
            news_status = raw_news.get("status")
            news_text = (raw_news.get("text") or "")[:1500]
        else:
            news_text = (raw_news or "")[:1500]

        passed = True
        if news_status == "NO_NEWS":
            if score < 0.65:
                passed = False
            logger.info(f"[Initial] {name}({ticker}) → NO_NEWS 플래그 감지(Score={score:.3f})")

        if _OPENAI_AVAILABLE:
            js = _initial_filter_gpt(name=name, score=score, news_text=news_text)
            if js and isinstance(js, dict) and js.get("decision") and "보류" in js["decision"]:
                passed = False
            logger.info(f"[Initial] {name}({ticker}) → {js.get('decision') if js else '실패'} (passed={passed})")
        else:
            if score < 0.6 and len(news_text) < 200:
                passed = False

        if not passed:
            continue

        # ── 전술/플랜 생성(GPT 또는 휴리스틱) ──
        plan_js = (
            _tactical_plan_gpt(
                market_trend=market_trend,
                c=c,
                news_text=news_text,
                budget_ctx=budget_ctx if _BUDGET_GUARD_ENABLED else None
            )
            if _OPENAI_AVAILABLE else
            _heuristic_plan(c, news_text, market_trend)
        )
        if not (plan_js and isinstance(plan_js, dict) and plan_js.get("결정")):
            plan_js = _heuristic_plan(c, news_text, market_trend)

        sel = plan_js.get("전략_클래스", "BaseStrategy")
        best = _apply_strategy_weights(selected=sel, c=c, market_trend=market_trend, weights=_strategy_weights)
        if best != sel:
            plan_js["전략_클래스"] = best

        stock_info = {k: v for k, v in c.items()}
        src = stock_info.get("source") or stock_info.get("levels_source")
        sector = stock_info.get("Sector")
        rsi = stock_info.get("RSI"); ma50 = stock_info.get("MA50"); ma200 = stock_info.get("MA200")
        news_tag = "(NO_NEWS)" if news_status == "NO_NEWS" else "(NEWS_OK)"
        extra = f" [{news_tag} | levels_source={src} | sector={sector} | RSI={rsi}, MA50={ma50}, MA200={ma200}]"

        if "분석" in plan_js and isinstance(plan_js["분석"], str):
            plan_js["분석"] = (plan_js["분석"] or "") + extra
        else:
            plan_js["분석"] = extra
        plan_js["분석"] += _compose_reason_suffix(c)

        stock_info_min = {
            "Ticker": c.get("Ticker"),
            "Name": c.get("Name"),
            "Price": c.get("Price"),
            "ATR": c.get("ATR"),
            "RSI": c.get("RSI"),
            "MA50": c.get("MA50"),
            "MA200": c.get("MA200"),
            "Score": c.get("Score"),
            "FinScore": c.get("FinScore"),
            "TechScore": c.get("TechScore"),
            "MktScore": c.get("MktScore"),
            "SectorScore": c.get("SectorScore"),
            "PatternScore": c.get("PatternScore"),
            "MA20Up": c.get("MA20Up"),
            "AccumVol": c.get("AccumVol"),
            "HigherLows": c.get("HigherLows"),
            "Consolidation": c.get("Consolidation"),
            "YEY": c.get("YEY"),
            "PER": c.get("PER"),
            "PBR": c.get("PBR"),
            "Sector": c.get("Sector"),
            "SectorSource": c.get("SectorSource"),
            "stop_price": c.get("stop_price"),
            "target_price": c.get("target_price"),
            "levels_source": c.get("levels_source"),
            "daily_chart": c.get("daily_chart"),
            "investor_flow": c.get("investor_flow"),
        }

        merged = {"rank": len(results) + 1, "stock_info": stock_info_min, **plan_js}
        results.append(merged)

    return results

def run_pipeline(
    fixed_date: Optional[str] = None,
    market: str = "KOSPI",
    available_slots: int = 3
) -> Optional[Path]:
    start_msg = f"▶ GPT 분석 시작 (date={fixed_date or 'auto'}, market={market}, slots={available_slots})"
    logger.info(start_msg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not fixed_date:
        latest = _detect_latest_screener_file()
        if not latest:
            msg = "스크리너 결과 파일을 찾지 못했습니다. (candidates/rank 패턴 모두 실패)"
            logger.error(msg)
            _notify(f"❌ {msg}", key="gpt_analyzer_missing_screener", cooldown_sec=120)
            return None
        parts = latest.stem.split("_")
        fixed_date, market = (parts[-2], parts[-1]) if len(parts) >= 4 else (None, market)
        if not fixed_date:
            msg = f"파일명에서 날짜/시장 추출 실패: {latest.name}"
            logger.error(msg)
            _notify(f"❌ {msg}", key="gpt_analyzer_parse_fail", cooldown_sec=120)
            return None

    screener_file, news_file = _detect_files(fixed_date, market)
    if not screener_file.exists():
        msg = f"스크리너 결과 없음: {screener_file.name}"
        logger.error(msg)
        _notify(f"❌ {msg}", key="gpt_analyzer_no_screener", cooldown_sec=120)
        return None
    if not news_file.exists():
        msg = f"뉴스 결과 없음: {news_file.name} (먼저 news_collector 실행 필요)"
        logger.error(msg)
        _notify(f"❌ {msg}", key="gpt_analyzer_no_news", cooldown_sec=120)
        return None

    logger.info(f"로드: {screener_file.name}, {news_file.name}")
    candidates_raw: List[Dict] = _read_json(screener_file)
    candidates: List[Dict] = _normalize_candidates(candidates_raw)
    news_cache: Dict[str, Any] = _read_json(news_file)
    market_trend = get_market_trend(fixed_date)
    logger.info(f"시장 추세: {market_trend}")

    # ▼ 예산 가드용 컨텍스트 구성
    usable_cash = _get_usable_cash()
    if _BUDGET_GUARD_ENABLED and usable_cash is not None:
        logger.info(
            f"[BudgetGuard] enabled=True | usable_cash={usable_cash:,}, "
            f"max_entry_price_ratio={_MAX_ENTRY_PRICE_RATIO:.2f}, cash_buffer_ratio={_CASH_BUFFER_RATIO:.2f}"
        )
    budget_ctx = {
        "enabled": _BUDGET_GUARD_ENABLED,
        "usable_cash": usable_cash,
        "max_entry_price_ratio": _MAX_ENTRY_PRICE_RATIO,
        "cash_buffer_ratio": _CASH_BUFFER_RATIO,
    }

    plans = analyze_candidates_and_create_plans(
        candidates,
        news_cache,
        market_trend,
        available_slots,
        budget_ctx=budget_ctx,
    )
    _pretty_print_plans(plans)

    out_path = OUTPUT_DIR / f"gpt_trades_{fixed_date}_{market}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plans or [], f, ensure_ascii=False, indent=2)
    logger.info(f"저장 완료 → {out_path}")

    try:
        if plans:
            top = plans[:min(3, len(plans))]
            fields = []
            for i, p in enumerate(top, 1):
                s = p.get('stock_info', {})
                fields.append({
                    "name": f"[{i}] {s.get('Name','N/A')} ({str(s.get('Ticker','N/A')).zfill(6)})",
                    "value": f"{p.get('결정')} / {p.get('전략_클래스')} / SL:{s.get('stop_price')} TP:{s.get('target_price')} ({s.get('levels_source')})",
                    "inline": False
                })
            _notify_embed(
                title=f"✅ GPT 분석 완료: {len(plans)}건 생성",
                description=f"date={fixed_date}, market={market}",
                fields=fields
            )
        else:
            _notify(f"ℹ️ GPT 분석 완료: 생성된 계획 없음 (date={fixed_date}, {market})", key="gpt_analyzer_done_empty", cooldown_sec=60)
    except Exception as e:
        logger.warning("최종 요약 알림 실패: %s", e)

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
