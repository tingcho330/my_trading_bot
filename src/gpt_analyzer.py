# src/gpt_analyzer.py
"""
스크리너 결과(screener_results_*.json) + 뉴스 결과(collected_news_*.json)를 읽어
GPT를 사용해 1) Initial Filter → 2) Tactical Plan을 생성하고 출력/저장하는 실행 스크립트.
+ 리스크 파라미터(risk_params)를 사용해 (최종 '매수' 통과 종목에 한해) 손절가/목표가 산출.

개선사항:
1) 손절/목표 산출 경로 source(atr_swing / percent_backup) 콘솔/JSON 모두 표시
2) 손절/목표 표기 정수(원 단위)로 통일
3) strategy_params.weights 가중치를 사용해 전략 선택 보정
4) 사용 중인 리스크 파라미터(ATR/스윙/R:R/stop_pct) 로그 노출

- .env: /app/config/.env 우선 로드(폴백 탐색)
- OPENAI_API_KEY 없으면 휴리스틱 모드로 자동 전환
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time

from dotenv import load_dotenv, find_dotenv

# ───────────────── 로깅 ─────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gpt_analyzer")

# ───────────────── .env 로딩 ─────────────────
def _load_env() -> None:
    candidates = [
        Path("/app/config/.env"),
        Path(__file__).resolve().parents[1] / "config" / ".env",
        Path.cwd() / "config" / ".env",
        Path.cwd() / ".env",
    ]
    loaded = ""
    for p in candidates:
        if p.is_file():
            load_dotenv(dotenv_path=p, override=False)
            loaded = str(p)
            break
    if not loaded:
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)
            loaded = found
    logger.info(f".env loaded from: {loaded if loaded else 'None'}")

_load_env()

# ───────────────── 경로 규칙 ─────────────────
PROJECT_ROOT = Path.cwd()          # /app
OUTPUT_DIR   = PROJECT_ROOT / "output"

# ───────────────── 외부 의존(리포 내부) ─────────────────
try:
    from src.screener import get_market_trend, get_historical_prices
except Exception:
    def get_market_trend(date_str: str) -> str:
        return "Sideways"
    def get_historical_prices(ticker: str, start_date: str, end_date: str):
        return None

# FDR 백업용(옵션)
try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None

def _safe_fetch_prices(ticker: str, start: str, end: str, retries: int = 3):
    # 1) screener.get_historical_prices 재사용 (리트라이)
    for _ in range(retries):
        df = get_historical_prices(ticker, start, end)
        if df is not None and len(df) > 0:
            return df
        time.sleep(0.5)
    # 2) 최종 백업: FDR 직접 조회
    if fdr is not None:
        for _ in range(retries):
            try:
                df = fdr.DataReader(ticker, start=start, end=end)
                if df is not None and len(df) > 0:
                    return df
            except Exception:
                time.sleep(0.5)
    return None

# ───────────────── OpenAI SDK (>=1.0) ─────────────────
_OPENAI_AVAILABLE = False
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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

# ───────────────── 프롬프트 ─────────────────
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

# ───────────────── Config 로딩 ─────────────────
def _load_config() -> Dict[str, Any]:
    candidates = [
        Path("/app/config/config.json"),
        Path(__file__).resolve().parents[1] / "config" / "config.json",
        Path.cwd() / "config" / "config.json",
        Path.cwd() / "config.json",
    ]
    for p in candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"config 로딩 실패({p}): {e}")
                break
    return {}

def _risk_defaults(cfg: Dict[str, Any]) -> Dict[str, float]:
    rp = (cfg.get("risk_params") or {}) if isinstance(cfg, dict) else {}
    return {
        "atr_period":     float(rp.get("atr_period", 14)),
        "atr_k_stop":     float(rp.get("atr_k_stop", 1.5)),
        "swing_lookback": int(rp.get("swing_lookback", 20)),
        "reward_risk":    float(rp.get("reward_risk", 2.0)),
        "stop_pct":       float(rp.get("stop_pct", 0.03)),   # 백업 퍼센트 손절
    }

def _strategy_weights(cfg: Dict[str, Any]) -> Dict[str, float]:
    sp = (cfg.get("strategy_params") or {}).get("weights", {}) if isinstance(cfg, dict) else {}
    # 기본 가중치
    base = {
        "RsiReversalStrategy": 0.6,
        "TrendFollowingStrategy": 0.8,
        "AdvancedTechnicalStrategy": 0.6,
        "DynamicAtrStrategy": 0.7,
        "BaseStrategy": 0.5,
    }
    base.update({k: float(v) for k, v in sp.items() if isinstance(v, (int, float))})
    return base

# ───────────────── 유틸 ─────────────────
def _read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _latest_by_pattern(pattern: str) -> Optional[Path]:
    files = sorted(OUTPUT_DIR.glob(pattern))
    return files[-1] if files else None

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
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content or ""
        js = _safe_json_loads(_strip_to_json(txt))
        return js
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
        name  = stock.get("Name", "N/A")
        ticker= stock.get("Ticker", "N/A")
        strategy = plan.get("전략_클래스", "N/A")
        tactic   = plan.get("매매전술", "N/A")
        decision = plan.get("결정", "N/A")
        reason   = (plan.get("분석", "") or "")[:300]
        stop_px  = plan.get("손절가")
        tgt_px   = plan.get("목표가")
        source   = plan.get("source")
        print(f"\n[{i}] {name} ({ticker}) - {decision}")
        print(f" - 전략: {strategy}")
        print(f" - 전술: {tactic}")
        if stop_px and tgt_px and decision == "매수":
            print(f" - 손절/목표: {int(round(stop_px)):,} / {int(round(tgt_px)):,}  (source={source})")
        print(f" - 근거: {reason}...")

# ───────────────── 리스크/레벨 계산 ─────────────────
def _yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

def _round_px(x: float) -> int:
    # 가격은 정수(원 단위)로 통일
    return int(round(float(x)))

def _compute_levels(
    ticker: str,
    entry_price: float,
    date_str: str,
    atr_period: int = 14,
    atr_k_stop: float = 1.5,
    swing_lookback: int = 20,
    rr: float = 2.0,
    stop_pct: float = 0.03,
) -> Optional[Dict[str, float]]:
    """
    1순위: ATR 기반 & 최근 스윙저점 비교 후 더 보수적인(높은) 손절가 사용
    실패/데이터부족 시: 퍼센트 손절( stop_pct )로 백업 산출
    목표가: R:R 기반
    """
    try:
        def _percent_backup() -> Dict[str, float]:
            stop_px = entry_price * (1.0 - float(stop_pct))
            risk    = max(1e-6, entry_price - stop_px)
            tgt_px  = entry_price + rr * risk
            return {
                "ATR": None,
                "SwingLow": None,
                "손절가": _round_px(stop_px),   # 정수 통일
                "목표가": _round_px(tgt_px),   # 정수 통일
                "R_R": rr,
                "source": "percent_backup"
            }

        end_dt = datetime.strptime(date_str, "%Y%m%d")
        start_dt = end_dt - timedelta(days=max(atr_period*6, 180))  # 룩백 완충
        df = _safe_fetch_prices(ticker, _yyyymmdd(start_dt), _yyyymmdd(end_dt))

        if df is None:
            logger.warning(f"{ticker}: 시세 조회 실패 → 퍼센트 백업 사용(stop_pct={stop_pct})")
            return _percent_backup()
        if len(df) < max(atr_period + 5, swing_lookback + 5):
            logger.warning(f"{ticker}: 히스토리 부족 → 퍼센트 백업 사용(stop_pct={stop_pct})")
            return _percent_backup()

        cols = {c.lower(): c for c in df.columns}
        high = df[cols.get("high", "High")].astype(float)
        low  = df[cols.get("low",  "Low")].astype(float)
        close= df[cols.get("close","Close")].astype(float)

        prev_close = close.shift(1)
        tr = (high - low).abs().to_frame("TR")
        tr["H-PC"] = (high - prev_close).abs()
        tr["L-PC"] = (low - prev_close).abs()
        TR = tr.max(axis=1)
        ATR = TR.rolling(window=atr_period, min_periods=atr_period).mean()
        if ATR.dropna().empty:
            logger.warning(f"{ticker}: ATR 계산 불가 → 퍼센트 백업 사용(stop_pct={stop_pct})")
            return _percent_backup()
        atr = float(ATR.dropna().iloc[-1])

        swing_low = float(low.tail(swing_lookback).min())
        stop_atr  = entry_price - atr_k_stop * atr
        stop_swing= swing_low
        stop_px   = max(stop_atr, stop_swing)
        if stop_px >= entry_price:
            stop_px = entry_price * (1.0 - float(stop_pct))

        risk   = max(1e-6, entry_price - stop_px)
        tgt_px = entry_price + rr * risk

        return {
            "ATR": round(atr, 4),
            "SwingLow": _round_px(swing_low),  # 정수 통일
            "손절가": _round_px(stop_px),      # 정수 통일
            "목표가": _round_px(tgt_px),       # 정수 통일
            "R_R": rr,
            "source": "atr_swing"
        }
    except Exception as e:
        logger.warning(f"{ticker}: 레벨 계산 예외 → 퍼센트 백업 사용(stop_pct={stop_pct}) | {e}")
        stop_px = entry_price * (1.0 - float(stop_pct))
        risk    = max(1e-6, entry_price - stop_px)
        tgt_px  = entry_price + rr * risk
        return {
            "ATR": None,
            "SwingLow": None,
            "손절가": _round_px(stop_px),
            "목표가": _round_px(tgt_px),
            "R_R": rr,
            "source": "percent_backup"
        }

# ───────────────── 전략 가중치 적용 ─────────────────
def _score_strategies_with_context(c: Dict, market_trend: str) -> Dict[str, float]:
    """
    전략별 컨텍스트 점수(0~1)를 간단히 계산:
    - TrendFollowing: MA20Up/HigherLows + Market Bull/Sideways에서 유리
    - RsiReversal: RSI<40, MA20Up=False, Consolidation=False에서 유리
    - AdvancedTechnical: PatternScore 기반
    - DynamicAtr: Volatility(여기선 Proxy로 TechScore) 기반
    - BaseStrategy: 중립
    """
    rsi  = float(c.get("RSI", 50.0))
    ma20 = bool(c.get("MA20Up", False))
    hl   = bool(c.get("HigherLows", False))
    cons = bool(c.get("Consolidation", False))
    psc  = float(c.get("PatternScore", 0.0))
    tech = float(c.get("TechScore", 0.5))

    tf = 0.5 + (0.2 if ma20 else 0.0) + (0.2 if hl else 0.0) + (0.1 if market_trend in ("Bull","Sideways") else 0.0)
    rr = 0.5 + (0.25 if rsi < 40 else 0.0) + (0.15 if not ma20 else 0.0) + (0.1 if not cons else 0.0)
    at = 0.4 + min(0.6, psc)  # 패턴 점수 영향
    da = 0.4 + min(0.5, tech) # 기술점수 기반 변동성 대리
    bs = 0.5

    return {
        "TrendFollowingStrategy": max(0.0, min(1.0, tf)),
        "RsiReversalStrategy":    max(0.0, min(1.0, rr)),
        "AdvancedTechnicalStrategy": max(0.0, min(1.0, at)),
        "DynamicAtrStrategy":     max(0.0, min(1.0, da)),
        "BaseStrategy":           bs
    }

def _apply_strategy_weights(selected: str, c: Dict, market_trend: str, weights: Dict[str, float]) -> str:
    """
    GPT/휴리스틱이 제시한 selected 전략을 컨텍스트 점수×가중치로 보정하여 최종 선택.
    """
    ctx = _score_strategies_with_context(c, market_trend)
    # 가중치 곱
    scored = {k: ctx.get(k, 0.0) * float(weights.get(k, 0.5)) for k in ctx.keys()}
    # 기존 선택 전략에 약간의 우선권(동점시 유지)
    best = max(scored.items(), key=lambda kv: (kv[1], 1.0 if kv[0] == selected else 0.0))[0]
    return best

# ───────────────── GPT 분석 ─────────────────
def _initial_filter_gpt(name: str, score: float, news_text: str) -> Optional[dict]:
    sys = "You are a fast, no-nonsense investment analyst. Always reply with a single JSON object only."
    user = INITIAL_FILTER_PROMPT_TMPL.format(name=name, score=score, news_text=news_text[:1500])
    return _call_openai_json(sys, user)

def _tactical_plan_gpt(market_trend: str, c: Dict, news_text: str) -> Optional[dict]:
    name   = c.get("Name", "N/A")
    ticker = str(c.get("Ticker", "N/A"))
    sector = c.get("Sector", "N/A")
    price  = float(c.get("Price", 0.0))
    ma50   = float(c.get("MA50", 0.0))
    ma200  = float(c.get("MA200", 0.0))

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
        news_text=news_text[:1500]
    )
    sys = "You are a Chief Investment Strategist. Output must be a single JSON object ONLY."
    return _call_openai_json(sys, user)

# ───────────────── 휴리스틱 백업 ─────────────────
def _heuristic_plan(c: Dict, news_text: str, market_trend: str) -> Dict:
    score = float(c.get("Score", 0.0))
    decision = "매수" if score >= 0.65 else "보류"
    # 기본 전략(컨텍스트 기반) — 이후 가중치로 보정됨
    if market_trend == "Bull":
        base_strategy = "TrendFollowingStrategy"
    elif market_trend == "Bear":
        base_strategy = "RsiReversalStrategy"
    else:
        base_strategy = "BaseStrategy"
    tactic   = "분할 매수 전략" if decision == "매수" else "집중 투자 전략"
    reason   = f"휴리스틱: Score={score:.3f}, 시장={market_trend}, 뉴스길이={len(news_text)}"
    return {
        "결정": decision,
        "분석": reason,
        "전략_클래스": base_strategy,
        "매매전술": tactic,
        "parameters": {"installments": []}
    }

# ───────────────── 공개 함수 ─────────────────
def analyze_candidates_and_create_plans(
    candidates: List[Dict],
    news_cache: Dict[str, str],
    market_trend: str,
    available_slots: int = 3,
    fixed_date: Optional[str] = None,
    market: str = "KOSPI",
) -> List[Dict]:
    """
    파이프라인:
      1) Initial Filter (GPT/휴리스틱)로 예비 통과
      2) Tactical Plan (GPT/휴리스틱)으로 최종 '매수/보류' 결정
      3) '매수'로 최종 통과된 종목에 한해 손절/목표가 계산
      4) 전략 선택은 strategy_params.weights로 보정
    """
    cfg = _load_config()
    rp  = _risk_defaults(cfg)
    sw  = _strategy_weights(cfg)

    # (4) 사용 중인 리스크 파라미터 로그
    logger.info(
        "Risk Params: ATR(%s)  k(%.2f)  SwingLB(%s)  R:R(%.2f)  stop_pct(%.2f)",
        int(rp["atr_period"]), float(rp["atr_k_stop"]), int(rp["swing_lookback"]),
        float(rp["reward_risk"]), float(rp["stop_pct"])
    )

    cand_sorted = sorted(candidates, key=lambda x: float(x.get("Score", 0.0)), reverse=True)
    results: List[Dict] = []

    for c in cand_sorted:
        if len(results) >= max(1, int(available_slots)):
            break

        name   = c.get("Name", "N/A")
        ticker = str(c.get("Ticker", "N/A"))
        score  = float(c.get("Score", 0.0))
        news   = (news_cache.get(ticker, "") or "")[:1500]

        # 1) Initial Filter
        passed = True
        if _OPENAI_AVAILABLE:
            js = _initial_filter_gpt(name=name, score=score, news_text=news)
            if js and isinstance(js, dict) and js.get("decision"):
                decision_init = js.get("decision")
                if "보류" in decision_init:
                    passed = False
                logger.info(f"[Initial] {name}({ticker}) → {decision_init} / {js.get('reason','')}")
            else:
                logger.info(f"[Initial] {name}({ticker}) → GPT 응답 파싱 실패, 휴리스틱으로 진행")
        else:
            if score < 0.6 and len(news) < 200:
                passed = False

        if not passed:
            continue

        # 2) Tactical Plan: 최종 '매수/보류' 결정
        if _OPENAI_AVAILABLE:
            plan_js = _tactical_plan_gpt(market_trend=market_trend, c=c, news_text=news)
            if not (plan_js and isinstance(plan_js, dict) and plan_js.get("결정")):
                plan_js = _heuristic_plan(c, news, market_trend)
        else:
            plan_js = _heuristic_plan(c, news, market_trend)

        decision_final = str(plan_js.get("결정", "")).strip()

        # (3) 전략 가중치 적용: 선택 전략 보정
        sel = plan_js.get("전략_클래스", "BaseStrategy")
        best = _apply_strategy_weights(selected=sel, c=c, market_trend=market_trend, weights=sw)
        if best != sel:
            plan_js["전략_클래스"] = best  # 보정 반영

        merged = {
            "rank": len(results) + 1,
            "stock_info": {
                "Name": name, "Ticker": ticker, "Score": score,
                "Sector": c.get("Sector", "N/A"), "Price": c.get("Price", 0.0)
            },
            **plan_js
        }

        # 3) 최종 통과('매수') 종목에만 손절/목표가 계산
        if decision_final == "매수":
            entry_price = float(c.get("Price", 0.0)) or None
            if fixed_date and entry_price:
                lv = _compute_levels(
                    ticker=ticker,
                    entry_price=entry_price,
                    date_str=fixed_date,
                    atr_period=int(rp["atr_period"]),
                    atr_k_stop=float(rp["atr_k_stop"]),
                    swing_lookback=int(rp["swing_lookback"]),
                    rr=float(rp["reward_risk"]),
                    stop_pct=float(rp["stop_pct"]),
                )
                if lv:
                    merged.update(lv)  # lv에는 source 포함
            else:
                logger.warning(f"{ticker}: 레벨 계산 생략 (fixed_date 또는 Price 부재)")

        results.append(merged)

    return results

# ───────────────── 파이프라인(파일 로드/저장) ─────────────────
def run_pipeline(
    fixed_date: Optional[str] = None,
    market: str = "KOSPI",
    available_slots: int = 3
) -> Optional[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not fixed_date:
        latest = _latest_by_pattern("screener_results_*_*.json")
        if not latest:
            logger.error("screener_results_*.json 파일을 찾지 못했습니다.")
            return None
        parts = latest.stem.split("_")
        if len(parts) >= 4:
            fixed_date = parts[-2]
            market     = parts[-1]
        else:
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

    plans = analyze_candidates_and_create_plans(
        candidates=candidates,
        news_cache=news_cache,
        market_trend=market_trend,
        available_slots=available_slots,
        fixed_date=fixed_date,
        market=market,
    )

    _pretty_print_plans(plans)

    out_path = OUTPUT_DIR / f"gpt_trades_{fixed_date}_{market}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plans or [], f, ensure_ascii=False, indent=2)
    logger.info(f"저장 완료 → {out_path}")
    return out_path

# ───────────────── CLI ─────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Screener + News + GPT Analyzer 파이프라인 (최종 '매수'만 손절/목표 산출; 소스/가중치/표기 개선)")
    parser.add_argument("--date", help="YYYYMMDD (미지정 시 최신 파일 자동 탐색)")
    parser.add_argument("--market", default="KOSPI", choices=["KOSPI", "KONEX", "KOSDAQ"])
    parser.add_argument("--slots", type=int, default=3, help="생성할 최대 매수 계획 개수")
    args = parser.parse_args()

    run_pipeline(fixed_date=args.date, market=args.market, available_slots=args.slots)
