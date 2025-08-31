import json
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# KIS API 모듈 임포트
from api.kis_auth import KIS

# ───────────────── 로깅 설정 ─────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("trader")

# ───────────────── 경로 설정 ─────────────────
CONFIG_PATH = Path("/app/config/config.json")
OUTPUT_DIR = Path("/app/output")
COOLDOWN_FILE = OUTPUT_DIR / "cooldown.json"


# ───────────────── 설정 및 유틸리티 함수 ─────────────────

def load_settings() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tick_size(price: float) -> float:
    if price < 2000: return 1
    elif price < 5000: return 5
    elif price < 20000: return 10
    elif price < 50000: return 50
    elif price < 200000: return 100
    elif price < 500000: return 500
    else: return 1000


def find_latest_trade_plan(date_str: str, market: str) -> Optional[Path]:
    plan_file = OUTPUT_DIR / f"gpt_trades_{date_str}_{market}.json"
    if plan_file.exists():
        return plan_file
    files = sorted(OUTPUT_DIR.glob("gpt_trades_*.json"))
    return files[-1] if files else None


class Trader:
    def __init__(self, settings: dict):
        self.settings = settings
        self.env = settings.get("trading_environment", "vps")
        self.debug = bool(settings.get("debug", False) or os.environ.get("TRADER_DEBUG") == "1")
        self.is_real_trading = (self.env == "prod")
        self.risk_params = settings.get("risk_params", {})
        self.cooldown_list = self._load_cooldown_list()
        self.cooldown_period_days = self.risk_params.get("cooldown_period_days", 10)

        # 우선 현금 키(설정/환경변수). 미지정 시 합리적 기본값.
        self.cash_keys_priority: List[str] = settings.get("cash_keys_priority") or \
            [k.strip() for k in os.environ.get("KIS_CASH_KEYS", "").split(",") if k.strip()]
        if not self.cash_keys_priority:
            self.cash_keys_priority = [
                "ord_psbl_cash", "ord_psbl_amt", "ord_psbl",
                "dnca_avl_amt", "available_cash", "avail_cash",
                "주문가능현금", "출금가능금액", "가용현금", "가용예수금"
            ]

        try:
            self.kis = KIS(config={}, env=self.env)
            if not getattr(self.kis, "auth_token", None):
                raise ConnectionError("KIS API 인증에 실패했습니다 (토큰 없음).")
            logger.info(f"'{self.env}' 모드로 KIS API 인증 완료.")
        except Exception as e:
            logger.error(f"KIS API 초기화 중 오류 발생: {e}", exc_info=True)
            raise ConnectionError("KIS API 초기화에 실패했습니다.") from e

    # ───────────────── 내부 유틸 ─────────────────

    def _parse_krw(self, v) -> int:
        if v is None:
            return 0
        if isinstance(v, (int, float)):
            return max(0, int(v))
        if isinstance(v, str):
            s = v.replace(",", "").replace("+", "").replace("원", "").strip()
            try:
                return max(0, int(float(s))) if s else 0
            except Exception:
                return 0
        return 0

    def _load_cooldown_list(self) -> dict:
        if not COOLDOWN_FILE.exists():
            return {}
        try:
            with open(COOLDOWN_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            logger.warning("쿨다운 파일을 읽는 데 실패했습니다.")
            return {}

    def _save_cooldown_list(self):
        COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(COOLDOWN_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.cooldown_list, f, indent=4, ensure_ascii=False)

    def _add_to_cooldown(self, ticker: str):
        end_date = (datetime.now() + timedelta(days=self.cooldown_period_days)).isoformat()
        self.cooldown_list[ticker] = end_date
        self._save_cooldown_list()
        logger.info(f"{ticker}를 {end_date}까지 쿨다운 목록에 추가합니다.")

    def _is_in_cooldown(self, ticker: str) -> bool:
        if ticker not in self.cooldown_list:
            return False
        cooldown_end_date = datetime.fromisoformat(self.cooldown_list[ticker])
        if datetime.now() < cooldown_end_date:
            return True
        else:
            del self.cooldown_list[ticker]
            self._save_cooldown_list()
            return False

    # ───────────────── 디버그 도구 ─────────────────

    def _dump_debug_dfs(self, df_balance, df_summary, tag: str = "balance"):
        if not self.debug:
            return
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            if df_balance is not None:
                try:
                    logger.info(f"[DEBUG] df_balance.columns({len(df_balance.columns)}): {list(df_balance.columns)}")
                    logger.info(f"[DEBUG] df_balance.dtypes:\n{str(df_balance.dtypes)}")
                except Exception:
                    pass
                if not df_balance.empty:
                    logger.info(f"[DEBUG] df_balance.head(1): {df_balance.head(1).to_dict('records')[0]}")

            if df_summary is not None:
                try:
                    logger.info(f"[DEBUG] df_summary.columns({len(df_summary.columns)}): {list(df_summary.columns)}")
                    logger.info(f"[DEBUG] df_summary.dtypes:\n{str(df_summary.dtypes)}")
                except Exception:
                    pass
                if not df_summary.empty:
                    logger.info(f"[DEBUG] df_summary.head(1): {df_summary.head(1).to_dict('records')[0]}")

            if df_balance is not None:
                try: df_balance.to_csv(OUTPUT_DIR / f"kis_debug_{tag}_df_balance_{ts}.csv", index=False)
                except Exception: pass
                try:
                    with open(OUTPUT_DIR / f"kis_debug_{tag}_df_balance_{ts}.json", "w", encoding="utf-8") as f:
                        json.dump(df_balance.to_dict("records"), f, ensure_ascii=False, indent=2)
                except Exception: pass

            if df_summary is not None:
                try: df_summary.to_csv(OUTPUT_DIR / f"kis_debug_{tag}_df_summary_{ts}.csv", index=False)
                except Exception: pass
                try:
                    with open(OUTPUT_DIR / f"kis_debug_{tag}_df_summary_{ts}.json", "w", encoding="utf-8") as f:
                        json.dump(df_summary.to_dict("records"), f, ensure_ascii=False, indent=2)
                except Exception: pass

            logger.info(f"[DEBUG] KIS 응답 덤프 저장 완료: {OUTPUT_DIR}")
        except Exception as e:
            logger.warning(f"[DEBUG] 덤프 저장 중 오류: {e}")

    # ───────────────── 현금 후보 추출/선택 ─────────────────

    def _extract_cash_candidates(self, row: Dict[str, Any]) -> Dict[str, int]:
        candidates: Dict[str, int] = {}
        if not row or not isinstance(row, dict):
            return candidates

        def norm(s: Any) -> str:
            try:
                s = "" if s is None else str(s)
            except Exception:
                s = ""
            s = s.lower().strip()
            return re.sub(r"[^a-z0-9가-힣_]", "", s)

        # 블록: 수량/평가/매수/대출/손익/합계 등 비현금 키
        BLOCK_PATTERNS = [
            r"_?qty$", r"수량$", r"보유수$", r"잔고수량", r"order_qty", r"가능수량", r"주문가능수량",
            r"^pchs_?amt$", r"^evlu_?amt$", r"^evlu_?pfls_?amt$", r"^loan_?amt$",
            r"평가", r"매수금액", r"대출", r"손익", r"총액", r"합계"
        ]
        STRONG_PATTERNS = [
            r"^ord_?psbl_?(cash|amt)$",
            r"^dnca_?avl_?amt$",
            r"^available_?cash$", r"^avail_?cash$",
            r"출금.*가능.*(금액|현금)"
        ]
        WEAK_PATTERNS = [
            r"ord_?psbl.*(cash|amt)",
            r"(cash|현금)$", r"예수금", r"avl", r"avail"
        ]
        PREFERRED_KEYS = [
            "ord_psbl_cash", "ord_psbl_amt", "ord_psbl",
            "dnca_avl_amt", "available_cash", "avail_cash",
            "주문가능현금", "출금가능금액", "가용현금", "가용예수금"
        ]

        def is_blocked(ks: str) -> bool:
            ks_n = norm(ks)
            for pat in BLOCK_PATTERNS:
                if re.search(pat, ks_n):
                    return True
            return False

        # 1) 선호 키 정확일치
        for k in list(row.keys()):
            ks = str(k)
            if ks in PREFERRED_KEYS and not is_blocked(ks):
                candidates[ks] = self._parse_krw(row.get(k))

        # 2) 강한 패턴
        for k, v in row.items():
            ks = str(k)
            if is_blocked(ks): continue
            kn = norm(ks)
            if any(re.search(pat, kn) for pat in STRONG_PATTERNS):
                candidates[ks] = self._parse_krw(v)

        # 3) 일반 패턴
        for k, v in row.items():
            ks = str(k)
            if is_blocked(ks) or ks in candidates: continue
            kn = norm(ks)
            if any(re.search(pat, kn) for pat in WEAK_PATTERNS):
                candidates[ks] = self._parse_krw(v)

        return {k: int(v) for k, v in candidates.items() if isinstance(v, int)}

    def _score_cash_key(self, key: str, value: int) -> Tuple[int, str]:
        kn = re.sub(r"[^a-z0-9가-힣_]", "", str(key).lower())
        score = 0
        reason = []
        if re.search(r"_?qty$|수량$|잔고수량|^pchs_?amt$|^evlu_?amt$|^evlu_?pfls_?amt$|^loan_?amt$|평가|매수금액|대출|손익|총액|합계", kn):
            return -9999, "blocked"
        if re.search(r"^ord_?psbl_?(cash|amt)$", kn):
            score += 100; reason.append("ord_psbl_cash/amt")
        elif re.search(r"^dnca_?avl_?amt$", kn):
            score += 90; reason.append("dnca_avl_amt")
        elif re.search(r"^available_?cash$|^avail_?cash$", kn):
            score += 80; reason.append("available_cash")
        if re.search(r"ord_?psbl.*(cash|amt)", kn):
            score += 40; reason.append("ord_psbl + cash/amt")
        if re.search(r"(cash|현금)$", kn):
            score += 20; reason.append("cash/현금")
        if "예수금" in key or re.search(r"예수금", kn):
            score += 10; reason.append("예수금")
        if re.search(r"avl|avail", kn):
            score += 5; reason.append("avl/avail")
        if value == 0:
            score -= 200; reason.append("value=0")
        try:
            import math
            score += int(min(50, math.log10(max(1, value)) * 5))
            reason.append("value-size-bonus")
        except Exception:
            pass
        return score, "+".join(reason) if reason else "none"

    def _select_best_cash(self, candidates: Dict[str, int]) -> Tuple[str, int, List[Tuple[str, int, int, str]]]:
        ranked: List[Tuple[str, int, int, str]] = []
        for k, v in candidates.items():
            sc, why = self._score_cash_key(k, v)
            ranked.append((k, v, sc, why))
        ranked.sort(key=lambda x: (x[2], x[1]), reverse=True)
        if not ranked:
            return "", 0, []
        best_key, best_val, best_score, best_reason = ranked[0]
        lines = ["현금 후보 스코어링(상위→하위):"]
        for k, v, s, why in ranked:
            lines.append(f"- {k:24s} | {v:>15,} | score={s:>4d} | {why}")
        logger.info("\n".join(lines))
        return best_key, best_val, ranked

    # ───────────────── 우선키(설정) 기반 탐색 ─────────────────

    def _pick_from_priority(self, summary: Dict[str, Any], bal0: Dict[str, Any]) -> Tuple[str, int]:
        keys = [k for k in self.cash_keys_priority if k]
        if not keys:
            return "", 0

        def search_exact(d: Dict[str, Any]) -> Tuple[str, int]:
            for k in keys:
                if k in d:
                    v = self._parse_krw(d.get(k))
                    logger.info(f"[CASH] priority hit: exact key='{k}' source={'summary' if d is summary else 'balance'} value={v:,}")
                    if v > 0:
                        return k, v
            return "", 0

        k, v = search_exact(summary)
        if v > 0: return k, v
        k, v = search_exact(bal0)
        if v > 0: return f"balance.{k}", v

        def search_partial(d: Dict[str, Any]) -> Tuple[str, int]:
            d_keys = list(d.keys())
            for target in keys:
                tl = target.lower()
                for dk in d_keys:
                    if tl in str(dk).lower():
                        val = self._parse_krw(d.get(dk))
                        logger.info(f"[CASH] priority hit: partial key='{dk}' (target='{target}') source={'summary' if d is summary else 'balance'} value={val:,}")
                        if val > 0:
                            return str(dk), val
            return "", 0

        k, v = search_partial(summary)
        if v > 0: return k, v
        k, v = search_partial(bal0)
        if v > 0: return f"balance.{k}", v

        return "", 0

    # ───────────────── KIS 주문 안전 래퍼 ─────────────────

    def _order_cash_safe(self, **kwargs) -> Dict[str, Any]:
        try:
            df = self.kis.order_cash(**kwargs)
            rt_cd = None; msg1 = None
            try:
                if df is not None and not df.empty:
                    rec = df.to_dict('records')[0]
                    rt_cd = rec.get('rt_cd') or rec.get('rtcd') or rec.get('rtCode')
                    msg1 = rec.get('msg1') or rec.get('msg') or rec.get('message')
            except Exception:
                pass
            return {'ok': (rt_cd == '0'), 'rt_cd': rt_cd, 'msg1': msg1, 'df': df}
        except Exception as e:
            emsg = str(e)
            logger.error(f"[order_cash_safe] 예외: {emsg}", exc_info=True)
            hints = ("가능 원인: 장외/점검시간, 가격/수량 유효성, 계좌권한/환경(prod), 최소주문금액, "
                     "주문구분(ord_dv/ord_dvsn) 불일치, 종목 상태(매매정지/단일가) 등.")
            return {'ok': False, 'rt_cd': None, 'msg1': hints, 'error': emsg, 'df': None}

    # ───────────────── 계좌/주문 로직 ─────────────────

    def get_account_balance(self) -> dict:
        try:
            df_balance, df_summary = self.kis.inquire_balance(
                inqr_dvsn="02", afhr_flpr_yn="N", ofl_yn="", unpr_dvsn="01",
                fund_sttl_icld_yn="N", fncg_amt_auto_rdpt_yn="N", prcs_dvsn="00"
            )

            # 디버그 덤프
            self._dump_debug_dfs(df_balance, df_summary, tag="balance")

            # --- 요약 레코드 언랩 처리: {0: {...}} → {...} ---
            summary_raw = df_summary.to_dict('records')[0] if (df_summary is not None and not df_summary.empty) else {}
            summary = summary_raw
            if isinstance(summary_raw, dict) and len(summary_raw) == 1:
                only_key = next(iter(summary_raw))
                if isinstance(only_key, int) and isinstance(summary_raw[only_key], dict):
                    summary = summary_raw[only_key]  # ✅ 언랩
                    logger.info("[DEBUG] summary unwrapped from {0: {...}} form")

            if not isinstance(summary, dict):
                try: summary = dict(summary)
                except Exception: summary = {}

            bal0 = {}
            if df_balance is not None and not df_balance.empty:
                bal0 = df_balance.to_dict('records')[0]
                if not isinstance(bal0, dict):
                    try: bal0 = dict(bal0)
                    except Exception: bal0 = {}

            # 1) 설정 기반 우선키 탐색
            if self.cash_keys_priority:
                pk, pv = self._pick_from_priority(summary, bal0)
                if pv > 0:
                    logger.info(f"가용 예산 선택: key='{pk}', value={pv:,} 원 (source=priority)")
                    return {
                        "available_cash": pv,
                        "available_cash_key": pk,
                        "dnca_tot_amt": self._parse_krw(summary.get("dnca_tot_amt", 0)),
                        "holdings": df_balance.to_dict('records') if (df_balance is not None and not df_balance.empty) else []
                    }

            # 2) 자동 후보 추출 + 스코어링
            candidates: Dict[str, int] = {}
            candidates.update(self._extract_cash_candidates(summary))
            if bal0:
                bal_candidates = self._extract_cash_candidates(bal0)
                bal_candidates = {f"balance.{k}": v for k, v in bal_candidates.items()}
                candidates.update(bal_candidates)
            if "dnca_tot_amt" in summary and "dnca_tot_amt" not in candidates:
                candidates["dnca_tot_amt"] = self._parse_krw(summary.get("dnca_tot_amt"))

            if candidates:
                best_key, available_cash, _rank = self._select_best_cash(candidates)
                logger.info("현금 후보 필드(원시): " + ", ".join([f"{k}={v:,}" for k, v in candidates.items()]))
                logger.info(f"가용 예산 선택: key='{best_key}', value={available_cash:,} 원 (source=scored)")
                return {
                    "available_cash": available_cash,
                    "available_cash_key": best_key,
                    "dnca_tot_amt": self._parse_krw(summary.get("dnca_tot_amt", 0)),
                    "holdings": df_balance.to_dict('records') if (df_balance is not None and not df_balance.empty) else []
                }

            # 3) 완전 실패 → dnca_tot_amt로 폴백 + 강제 전체키/샘플 로깅
            logger.warning("현금 후보 필드를 찾지 못해 dnca_tot_amt로 대체합니다.")
            if summary:
                logger.info(f"[FALLBACK DEBUG] summary keys: {list(summary.keys())}")
                logger.info(f"[FALLBACK DEBUG] summary row0: {summary}")
            if bal0:
                logger.info(f"[FALLBACK DEBUG] balance keys: {list(bal0.keys())}")
                logger.info(f"[FALLBACK DEBUG] balance row0: {bal0}")

            available_cash = self._parse_krw(summary.get("dnca_tot_amt", 0))
            return {
                "available_cash": available_cash,
                "available_cash_key": "dnca_tot_amt",
                "dnca_tot_amt": available_cash,
                "holdings": df_balance.to_dict('records') if (df_balance is not None and not df_balance.empty) else []
            }

        except Exception as e:
            logger.error(f"계좌 잔고 조회 중 오류 발생: {e}", exc_info=True)
            return {}

    def run_sell_logic(self, account_balance: dict):
        holdings = account_balance.get("holdings", [])
        if not holdings:
            logger.info("매도할 보유 종목이 없습니다.")
            return
        logger.info(f"--------- 보유 종목 {len(holdings)}개에 대한 매도 로직 실행 ---------")

        for holding in holdings:
            ticker = holding.get("pdno")
            name = holding.get("prdt_name")
            stop_loss_price = holding.get("손절가")
            target_price = holding.get("목표가")

            if not all([ticker, name]): continue

            try:
                price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                if price_df.empty:
                    logger.warning(f"{name}({ticker}) 현재가 조회 실패. 매도 로직 건너뜁니다.")
                    continue
                current_price = float(price_df['stck_prpr'].iloc[0])
            except Exception as e:
                logger.error(f"{name}({ticker}) 현재가 조회 중 오류: {e}")
                continue

            sell_reason = None
            if stop_loss_price and current_price <= stop_loss_price:
                sell_reason = f"손절가({stop_loss_price:,.0f}원) 도달"
            elif target_price and current_price >= target_price:
                sell_reason = f"목표가({target_price:,.0f}원) 도달"

            if sell_reason:
                quantity = int(holding.get("hldg_qty", 0))
                if quantity == 0: continue
                logger.info(f"매도 결정: {name}({ticker}) {quantity}주. 사유: {sell_reason}")

                if self.is_real_trading:
                    try:
                        result = self._order_cash_safe(
                            ord_dv="01", pdno=ticker, ord_dvsn="01", ord_qty=quantity, ord_unpr=0
                        )
                        if result['ok']:
                            logger.info(f"매도 주문 성공: {result.get('msg1')}")
                            self._add_to_cooldown(ticker)
                        else:
                            logger.error(f"{name}({ticker}) 매도 주문 실패: {result.get('msg1')} | err={result.get('error')}")
                    except Exception as e:
                        logger.error(f"{name}({ticker}) 매도 주문 중 예외 발생: {e}", exc_info=True)
                else:
                    logger.info(f"[모의] {name}({ticker}) {quantity}주 시장가 매도 실행.")
                    self._add_to_cooldown(ticker)

    def run_buy_logic(self, account_balance: dict, trade_plans: list):
        cash_to_invest = int(account_balance.get("available_cash") or account_balance.get("dnca_tot_amt") or 0)
        cash_key = account_balance.get("available_cash_key", "unknown")
        logger.info(f"매수 로직 가용 예산: {cash_to_invest:,} 원 (key={cash_key})")

        holdings = account_balance.get("holdings", [])
        max_positions = self.risk_params.get("max_positions", 5)

        buy_plans = [p for p in trade_plans if p.get("결정") == "매수"]
        if not buy_plans:
            logger.info("매수할 종목이 없습니다.")
            return

        holding_tickers = {h.get("pdno") for h in holdings}
        new_buy_targets = []
        for plan in buy_plans:
            ticker = plan["stock_info"]["Ticker"]
            if ticker in holding_tickers:
                logger.info(f"{plan['stock_info']['Name']}({ticker})는 이미 보유 중이므로 건너뜁니다.")
                continue
            if self._is_in_cooldown(ticker):
                logger.info(f"{plan['stock_info']['Name']}({ticker})는 쿨다운 기간이므로 건너뜁니다.")
                continue
            new_buy_targets.append(plan)

        slots_to_fill = max_positions - len(holdings)
        if slots_to_fill <= 0:
            logger.info(f"매수 슬롯이 없습니다 (최대 보유: {max_positions}, 현재 보유: {len(holdings)}).")
            return

        targets_to_buy = new_buy_targets[:slots_to_fill]
        if not targets_to_buy:
            logger.info("신규로 매수할 종목이 없습니다.")
            return

        if cash_to_invest <= 0:
            logger.warning("가용 예산이 0원 이하이므로 매수 로직을 실행할 수 없습니다.")
            return

        budget_per_stock = cash_to_invest // len(targets_to_buy)
        logger.info(f"--------- 신규 매수 로직 실행 (대상: {len(targets_to_buy)}개) ---------")
        logger.info(f"총 가용 예산: {cash_to_invest:,.0f}원, 종목당 예산: {budget_per_stock:,.0f}원")

        for plan in targets_to_buy:
            stock_info = plan["stock_info"]
            ticker = stock_info["Ticker"]
            name = stock_info["Name"]

            try:
                price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                if price_df.empty:
                    logger.warning(f"{name}({ticker}) 현재가 조회 실패. 매수 로직 건너뜁니다.")
                    continue
                current_price = float(price_df['stck_prpr'].iloc[0])
            except Exception as e:
                logger.error(f"{name}({ticker}) 현재가 조회 중 오류: {e}")
                continue

            tick_size = get_tick_size(current_price)
            num_ticks = random.randint(1, 3)
            order_price = current_price + (tick_size * num_ticks)

            quantity = int(budget_per_stock // order_price)
            if quantity == 0:
                logger.warning(f"{name}({ticker}) 예산 부족으로 최소 1주도 매수할 수 없습니다.")
                continue

            logger.info(f"매수 준비: {name}({ticker}), 수량: {quantity}주, 지정가: {order_price:,.0f}원")

            if self.is_real_trading:
                try:
                    result = self._order_cash_safe(
                        ord_dv="02", pdno=ticker, ord_dvsn="00",
                        ord_qty=quantity, ord_unpr=int(order_price)
                    )
                    if result['ok']:
                        logger.info(f"매수 주문 성공: {result.get('msg1')}")
                    else:
                        logger.error(f"{name}({ticker}) 매수 주문 실패: {result.get('msg1')} | err={result.get('error')}")
                except Exception as e:
                    logger.error(f"{name}({ticker}) 매수 주문 중 예외 발생: {e}", exc_info=True)
            else:
                logger.info(f"[모의] {name}({ticker}) {quantity}주 @{order_price:,.0f}원 지정가 매수 실행.")

            time.sleep(0.5)


if __name__ == "__main__":
    try:
        settings_data = load_settings()
        trader = Trader(settings_data)

        account = trader.get_account_balance()
        if not account:
            logger.error("계좌 정보를 조회할 수 없어 프로그램을 종료합니다.")
            exit()

        trader.run_sell_logic(account)

        today_str = datetime.now().strftime("%Y%m%d")
        market_name = "KOSPI"
        latest_plan_file = find_latest_trade_plan(today_str, market_name)

        if latest_plan_file:
            logger.info(f"최신 분석 파일 사용: {latest_plan_file.name}")
            with open(latest_plan_file, "r", encoding="utf-8") as f:
                trade_plans = json.load(f)
            trader.run_buy_logic(account, trade_plans)
        else:
            logger.warning("오늘 날짜의 GPT 분석 파일을 찾을 수 없어 매수 로직을 건너뜁니다.")

    except Exception as e:
        logger.critical(f"트레이더 실행 중 심각한 오류 발생: {e}", exc_info=True)
