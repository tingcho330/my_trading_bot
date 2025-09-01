# src/trader.py
import json
import logging
import os
import random
import time
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# 공통 유틸리티 및 설정/전략 모듈 임포트
from utils import setup_logging, find_latest_file, OUTPUT_DIR
from api.kis_auth import KIS
from risk_manager import RiskManager
from settings import settings
from recorder import initialize_db, record_trade

# ── 디스코드 노티파이어 ──
from notifier import (
    DiscordLogHandler,
    WEBHOOK_URL,
    is_valid_webhook,
    send_discord_message,
    create_trade_embed,
)

# 로깅 설정
setup_logging()
logger = logging.getLogger("trader")

# 루트 로거에 디스코드 에러 핸들러 장착(중복 방지)
_root = logging.getLogger()
if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

# ── 간단 레이트 리밋(스팸 방지) ──
_last_sent_ts = defaultdict(float)
DEFAULT_COOLDOWN_SEC = 120  # 동일 키 알림 최소 간격(초)

def _can_send(key: str, cooldown: int = DEFAULT_COOLDOWN_SEC) -> bool:
    now = time.time()
    if now - _last_sent_ts[key] >= cooldown:
        _last_sent_ts[key] = now
        return True
    return False

def _notify_text(content: str, key: str = "trader_generic", cooldown: int = DEFAULT_COOLDOWN_SEC):
    if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL) and _can_send(key, cooldown):
        send_discord_message(content=content)

def _notify_embed(embed: Dict, key: str, cooldown: int = DEFAULT_COOLDOWN_SEC):
    if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL) and _can_send(key, cooldown):
        send_discord_message(embeds=[embed])

# 경로/옵션
COOLDOWN_FILE = OUTPUT_DIR / "cooldown.json"
ACCOUNT_SCRIPT_PATH = "/app/src/account.py"  # 스크립트 경로
REFRESH_ON_START = os.getenv("TRADER_REFRESH_ACCOUNT", "0") == "1"  # 시작 시 1회 강제 갱신 여부

def get_tick_size(price: float) -> float:
    """호가 단위를 반환합니다."""
    if price < 2000: return 1
    elif price < 5000: return 5
    elif price < 20000: return 10
    elif price < 50000: return 50
    elif price < 200000: return 100
    elif price < 500000: return 500
    else: return 1000

class Trader:
    def __init__(self, settings_obj):
        self.settings = settings_obj._config
        self.env = self.settings.get("trading_environment", "vps")
        self.is_real_trading = (self.env == "prod")
        self.risk_params = self.settings.get("risk_params", {})
        self.cooldown_list = self._load_cooldown_list()
        self.cooldown_period_days = self.risk_params.get("cooldown_period_days", 10)
        self.all_stock_data = self._load_all_stock_data()

        initialize_db()
        logger.info("거래 기록용 데이터베이스가 초기화되었습니다.")

        try:
            self.kis = KIS(config={}, env=self.env)
            if not getattr(self, "kis", None) or not getattr(self.kis, "auth_token", None):
                raise ConnectionError("KIS API 인증에 실패했습니다 (토큰 없음).")
            logger.info(f"'{self.env}' 모드로 KIS API 인증 완료.")
        except Exception as e:
            logger.error(f"KIS API 초기화 중 오류 발생: {e}", exc_info=True)
            raise ConnectionError("KIS API 초기화에 실패했습니다.") from e

        self.risk_manager = RiskManager(settings_obj)
        _notify_text(f"🤖 Trader 초기화 완료 (env={self.env}, real_trading={self.is_real_trading})",
                     key="trader_init", cooldown=60)

    # ───────── 파일/데이터 로드 ─────────
    def _load_all_stock_data(self) -> Dict[str, Dict]:
        full_screener_file = find_latest_file("screener_full_*.json")
        if not full_screener_file:
            logger.warning("screener_full.json 파일을 찾을 수 없어, 실시간 데이터 조회만 가능합니다.")
            _notify_text("ℹ️ 스크리너 전체 데이터 없음 -> 실시간 조회로 진행", key="trader_no_full_screener", cooldown=600)
            return {}
        try:
            with open(full_screener_file, 'r', encoding='utf-8') as f:
                all_stocks = json.load(f)
            return {str(stock.get('Ticker', '')).zfill(6): stock for stock in all_stocks}
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"{full_screener_file.name} 파일 로드 실패: {e}")
            _notify_text(f"⚠️ 스크리너 전체 데이터 로드 실패: {e}", key="trader_full_screener_load_err", cooldown=600)
            return {}

    # ───────── 계좌 동기화 ─────────
    def _update_account_info(self):
        logger.info("최신 계좌 정보 업데이트를 위해 account.py를 실행합니다...")
        try:
            subprocess.run(
                ["python", str(ACCOUNT_SCRIPT_PATH)],
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            logger.info("account.py 실행 완료.")
            _notify_text("📗 account.py 실행 완료(요약/잔고 갱신)", key="trader_account_update", cooldown=60)
        except FileNotFoundError:
            msg = f"스크립트를 찾을 수 없습니다: {ACCOUNT_SCRIPT_PATH}"
            logger.error(msg)
            _notify_text(f"❗ {msg}", key="trader_account_not_found", cooldown=300)
        except subprocess.CalledProcessError as e:
            msg = f"account.py 실행 중 오류 (Exit {e.returncode})"
            logger.error(f"{msg}:\n{e.stderr}")
            _notify_text(f"❗ {msg}\n```stderr\n{(e.stderr or '')[:1500]}\n```", key="trader_account_cpe", cooldown=300)
        except Exception as e:
            msg = f"계좌 정보 업데이트 중 예외: {e}"
            logger.error(msg)
            _notify_text(f"❗ {msg}", key="trader_account_exc", cooldown=300)

    # ───────── 유틸 ─────────
    def _parse_krw(self, v) -> int:
        if isinstance(v, (int, float)): return int(v)
        if isinstance(v, str):
            s = v.replace(",", "").strip()
            return int(s) if s.isdigit() else 0
        return 0

    def _load_cooldown_list(self) -> dict:
        if not COOLDOWN_FILE.exists(): return {}
        try:
            with open(COOLDOWN_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return {}

    def _save_cooldown_list(self):
        COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(COOLDOWN_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.cooldown_list, f, indent=2, ensure_ascii=False)

    def _add_to_cooldown(self, ticker: str, reason: str):
        end_date = (datetime.now() + timedelta(days=self.cooldown_period_days)).isoformat()
        self.cooldown_list[ticker] = end_date
        self._save_cooldown_list()
        logger.info(f"[{ticker}] {reason}으로 인해 쿨다운 목록에 추가. ({end_date}까지)")
        _notify_text(f"⛔ {ticker} 쿨다운 등록: {reason} (until {end_date[:19]})",
                     key=f"cooldown_{ticker}", cooldown=60)

    def _is_in_cooldown(self, ticker: str) -> bool:
        if ticker not in self.cooldown_list: return False
        cooldown_end_date = datetime.fromisoformat(self.cooldown_list[ticker])
        if datetime.now() < cooldown_end_date:
            return True
        else:
            del self.cooldown_list[ticker]
            self._save_cooldown_list()
            return False

    def _order_cash_safe(self, **kwargs) -> Dict[str, Any]:
        try:
            df = self.kis.order_cash(**kwargs)
            if df is None or df.empty:
                return {'ok': False, 'msg1': 'API 응답 없음'}
            res = df.to_dict('records')[0]
            rt_cd = res.get('rt_cd', '')
            ok = (rt_cd == '0')
            msg = res.get('msg1', '메시지 없음')
            return {'ok': ok, 'rt_cd': rt_cd, 'msg1': msg, 'df': df}
        except Exception as e:
            logger.error(f"주문 API 호출 중 예외 발생: {e}", exc_info=True)
            return {'ok': False, 'msg1': str(e), 'error': str(e)}

    # ───────── 파일에서 계좌 정보 읽기(참조값 양식 대응) ─────────
    def _pick_cash_from_summary_core(self, core: Dict[str, Any]) -> Tuple[int, Dict[str, int], str]:
        """
        참조값 양식에 맞춰 현금성 지표를 파싱하고, 사용 가용예산과 선택된 키명을 반환.
        우선순위: prvs_rcdl_excc_amt → nxdy_excc_amt → ord_psbl_cash → dnca_tot_amt
        """
        dnca = self._parse_krw(core.get("dnca_tot_amt", 0))
        nxdy = self._parse_krw(core.get("nxdy_excc_amt", 0))
        prvs = self._parse_krw(core.get("prvs_rcdl_excc_amt", 0))
        ord_psbl = self._parse_krw(core.get("ord_psbl_cash", 0))
        tot_ev = self._parse_krw(core.get("tot_evlu_amt", core.get("nass_amt", 0)))

        picked_key = None
        if prvs > 0:
            available = prvs; picked_key = "prvs_rcdl_excc_amt"
        elif nxdy > 0:
            available = nxdy; picked_key = "nxdy_excc_amt"
        elif ord_psbl > 0:
            available = ord_psbl; picked_key = "ord_psbl_cash"
        else:
            available = dnca; picked_key = "dnca_tot_amt"

        return available, {"dnca": dnca, "nxdy": nxdy, "prvs": prvs, "ord_psbl": ord_psbl, "tot_ev": tot_ev}, picked_key

    def get_account_info_from_files(self) -> Tuple[int, List[Dict]]:
        """
        balance/summary JSON 파일에서 보유/가용예산을 읽는다.
        ※ 여기서는 API 재조회(account.py 실행)를 하지 않는다.
        """
        # --- summary ---
        available_cash = 0
        dnca = nxdy = prvs = ord_psbl = tot_ev = 0
        picked = "n/a"

        summary_file = find_latest_file("summary_*.json")
        if not summary_file:
            logger.warning("summary.json 파일을 찾을 수 없어 주문 가능 금액을 0으로 설정합니다.")
            _notify_text("⚠️ summary 파일 없음 → 가용 현금 0 처리", key="trader_no_summary", cooldown=600)
        else:
            logger.info(f"가용 예산 조회를 위해 summary 파일 읽기: {summary_file}")
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_json = json.load(f)

                # 참조값 구조: {"data": [ {"0": {...}} ]}
                summary_data = summary_json.get("data", [])
                core_data = {}
                if summary_data and isinstance(summary_data, list) and isinstance(summary_data[0], dict):
                    core_data = summary_data[0].get("0", summary_data[0])

                if core_data and isinstance(core_data, dict):
                    available_cash, parts, picked = self._pick_cash_from_summary_core(core_data)
                    dnca, nxdy, prvs, ord_psbl, tot_ev = parts["dnca"], parts["nxdy"], parts["prvs"], parts["ord_psbl"], parts["tot_ev"]
                else:
                    logger.error("summary.json의 data[0]['0'] 구조를 찾지 못했습니다. 원본 키: %s", list(summary_json.keys()))

            except json.JSONDecodeError as e:
                logger.error(f"{summary_file.name} 파일 형식 오류: {e}")
            except Exception as e:
                logger.error(f"{summary_file.name} 파일 처리 중 오류: {e}", exc_info=True)

        # --- balance ---
        holdings: List[Dict] = []
        balance_file = find_latest_file("balance_*.json")
        if not balance_file:
            logger.warning("balance.json 파일을 찾을 수 없어 보유 종목이 없는 것으로 간주합니다.")
        else:
            try:
                with open(balance_file, 'r', encoding='utf-8') as f:
                    balance_json = json.load(f)

                holdings_data = balance_json.get("data", [])
                holdings = [h for h in holdings_data if self._parse_krw(h.get("hldg_qty", 0)) > 0]
                logger.info(f"보유 종목 로드 완료: 총 {len(holdings_data)}개 → 유효 보유 {len(holdings)} 종목 (from {balance_file.name})")
            except json.JSONDecodeError as e:
                logger.error(f"{balance_file.name} 파일 형식 오류: {e}")
            except Exception as e:
                logger.error(f"{balance_file.name} 파일 처리 중 오류: {e}")

        # 요약 로그(최종 확정 값만 1회 출력)
        logger.info(
            "💼 계좌 요약: D+2=%s원, 익일=%s원, 예수금=%s원, 총평가=%s원 → 사용 가용예산=%s원 (%s)",
            f"{prvs:,}", f"{nxdy:,}", f"{dnca:,}", f"{tot_ev:,}", f"{available_cash:,}", picked
        )
        return available_cash, holdings

    # ───────── 매도 로직 ─────────
    def run_sell_logic(self, holdings: List[Dict]):
        logger.info(f"--------- 보유 종목 {len(holdings)}개 매도 로직 실행 ---------")

        executed_sell = False
        if not holdings:
            logger.info("매도할 보유 종목이 없습니다.")
            return

        for holding in holdings:
            ticker = str(holding.get("pdno", "")).zfill(6)
            name = holding.get("prdt_name", "N/A")
            quantity = self._parse_krw(holding.get("hldg_qty", 0))

            if not ticker or quantity == 0:
                continue

            stock_info = self.all_stock_data.get(ticker, {})
            decision, reason = self.risk_manager.check_sell_condition(holding, stock_info)

            if decision == "SELL":
                logger.info(f"매도 결정: {name}({ticker}) {quantity}주. 사유: {reason}")
                executed_sell = True

                if self.is_real_trading:
                    # 시장가(01) 매도
                    result = self._order_cash_safe(
                        ord_dv="01", pdno=ticker, ord_dvsn="01", ord_qty=str(quantity), ord_unpr="0"
                    )
                    if result.get('ok'):
                        logger.info(f"매도 주문 성공: {result.get('msg1')}")
                        price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                        current_price = self._parse_krw(price_df['stck_prpr'].iloc[0]) if (price_df is not None and not price_df.empty) else 0
                        trade_data = {
                            "side": "sell", "ticker": ticker, "name": name,
                            "qty": quantity, "price": current_price,
                            "trade_status": "completed", "strategy_details": {"reason": reason}
                        }
                        record_trade(trade_data)
                        _notify_embed(create_trade_embed({
                            "side": "SELL", "name": name, "ticker": ticker, "qty": quantity,
                            "price": current_price, "trade_status": "completed",
                            "strategy_details": {"reason": reason}
                        }), key=f"trade_sell_{ticker}", cooldown=30)
                        self._add_to_cooldown(ticker, "매도 완료")
                    else:
                        err = result.get('msg1', 'Unknown error')
                        logger.error(f"{name}({ticker}) 매도 주문 실패: {err}")
                        trade_data = {
                            "side": "sell", "ticker": ticker, "name": name, "qty": quantity, "price": 0,
                            "trade_status": "failed", "strategy_details": {"error": err}
                        }
                        record_trade(trade_data)
                        _notify_embed(create_trade_embed({
                            "side": "SELL", "name": name, "ticker": ticker, "qty": quantity,
                            "price": 0, "trade_status": "failed",
                            "strategy_details": {"error": err}
                        }), key=f"trade_sell_fail_{ticker}", cooldown=30)
                else:
                    logger.info(f"[모의] {name}({ticker}) {quantity}주 시장가 매도 실행.")
                    _notify_text(f"🧪 [모의] SELL {name}({ticker}) x{quantity}", key=f"paper_sell_{ticker}", cooldown=30)
                    self._add_to_cooldown(ticker, "모의 매도 완료")

        if executed_sell:
            time.sleep(5)
            # ✅ 실제 체결 발생 시에만 최신 동기화
            self._update_account_info()

    # ───────── 매수 로직 ─────────
    def run_buy_logic(self, available_cash: int, holdings: List[Dict]):
        logger.info(f"--------- 신규 매수 로직 실행 (가용 예산: {available_cash:,} 원) ---------")

        if available_cash < 10000:
            logger.warning("가용 예산이 부족하여 매수 로직을 실행할 수 없습니다.")
            _notify_text("⚠️ 가용 예산 부족 → 매수 스킵", key="trader_cash_low", cooldown=300)
            return

        trade_plan_file = find_latest_file("gpt_trades_*.json")
        if not trade_plan_file:
            logger.warning("매수 계획 파일(gpt_trades_*.json)이 없어 매수를 건너뜁니다.")
            _notify_text("ℹ️ gpt_trades 파일 없음 → 매수 스킵", key="trader_no_gpt_trades", cooldown=600)
            return

        with open(trade_plan_file, 'r', encoding='utf-8') as f:
            trade_plans = json.load(f)

        buy_plans = [p for p in trade_plans if p.get("결정") == "매수"]
        if not buy_plans:
            logger.info("매수 결정이 내려진 종목이 없습니다.")
            _notify_text("ℹ️ 매수 결정된 종목 없음", key="trader_no_buy", cooldown=300)
            return

        holding_tickers = {str(h.get("pdno", "")).zfill(6) for h in holdings}
        new_targets = []
        for plan in buy_plans:
            stock_info = plan.get("stock_info", {})
            ticker = str(stock_info.get("Ticker", "")).zfill(6)
            name = stock_info.get("Name", "N/A")
            if ticker in holding_tickers:
                logger.info(f"[{name}({ticker})] 이미 보유 중이므로 매수 대상에서 제외합니다.")
                continue
            if self._is_in_cooldown(ticker):
                logger.info(f"[{name}({ticker})] 쿨다운 기간이므로 매수 대상에서 제외합니다.")
                continue
            new_targets.append(plan)

        max_positions = self.risk_params.get("max_positions", 5)
        slots_to_fill = max_positions - len(holdings)
        if slots_to_fill <= 0:
            logger.info(f"매수 슬롯이 없습니다 (최대: {max_positions}, 현재: {len(holdings)}).")
            _notify_text(f"ℹ️ 매수 슬롯 없음 (max={max_positions}, curr={len(holdings)})",
                         key="trader_no_slots", cooldown=300)
            return

        targets_to_buy = new_targets[:slots_to_fill]
        if not targets_to_buy:
            logger.info("신규로 매수할 최종 대상이 없습니다.")
            _notify_text("ℹ️ 신규 매수 대상 없음", key="trader_no_targets", cooldown=300)
            return

        remaining_cash = available_cash
        executed_buy = False
        logger.info(f"총 {len(targets_to_buy)}개 종목 신규 매수 시도. 유동적 예산 배분 적용.")
        _notify_text(f"🛒 신규 매수 시도 {len(targets_to_buy)}종목 (예산 {available_cash:,}원)",
                     key="trader_buy_start", cooldown=120)

        for i, plan in enumerate(targets_to_buy):
            stock_info = plan["stock_info"]
            ticker, name = str(stock_info["Ticker"]).zfill(6), stock_info["Name"]
            slots_left = len(targets_to_buy) - i
            budget_for_this_stock = remaining_cash // max(slots_left, 1)
            logger.info(f"  -> [{i+1}/{len(targets_to_buy)}] {name}({ticker}) 배분 예산: {budget_for_this_stock:,.0f}원")

            current_price = self._parse_krw(stock_info.get("Price", 0))
            if current_price == 0:
                price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                if price_df is not None and not price_df.empty:
                    current_price = self._parse_krw(price_df['stck_prpr'].iloc[0])
                else:
                    logger.warning(f"  -> [{name}({ticker})] 현재가 조회 실패. 매수를 건너뜁니다.")
                    _notify_text(f"⚠️ 가격 조회 실패 → 스킵: {name}({ticker})", key=f"trader_price_fail_{ticker}", cooldown=120)
                    continue

            tick_size = get_tick_size(current_price)
            order_price = current_price + (tick_size * random.randint(1, 3))
            quantity = int(budget_for_this_stock // order_price)
            if quantity == 0:
                logger.warning(f"  -> [{name}({ticker})] 예산 부족으로 매수 불가.")
                _notify_text(f"⚠️ 예산 부족 → 스킵: {name}({ticker})", key=f"trader_budget_low_{ticker}", cooldown=120)
                continue

            logger.info(f"  -> 매수 준비: {name}({ticker}), 수량: {quantity}주, 지정가: {order_price:,.0f}원")
            actual_spent = quantity * order_price
            executed_buy = True

            if self.is_real_trading:
                result = self._order_cash_safe(
                    ord_dv="02", pdno=ticker, ord_dvsn="00", ord_qty=str(quantity), ord_unpr=str(int(order_price))
                )
                if result.get('ok'):
                    logger.info(f"  -> 매수 주문 성공: {result.get('msg1')}")
                    trade_data = {
                        "side": "buy", "ticker": ticker, "name": name,
                        "qty": quantity, "price": order_price,
                        "trade_status": "active", "gpt_analysis": plan
                    }
                    record_trade(trade_data)
                    remaining_cash -= actual_spent
                    _notify_embed(create_trade_embed({
                        "side": "BUY", "name": name, "ticker": ticker, "qty": quantity,
                        "price": order_price, "trade_status": "completed"
                    }), key=f"trade_buy_{ticker}", cooldown=30)
                else:
                    err = result.get('msg1', 'Unknown error')
                    logger.error(f"  -> {name}({ticker}) 매수 주문 실패: {err}")
                    trade_data = {
                        "side": "buy", "ticker": ticker, "name": name, "qty": quantity,
                        "price": order_price, "trade_status": "failed",
                        "strategy_details": {"error": err}, "gpt_analysis": plan
                    }
                    record_trade(trade_data)
                    self._add_to_cooldown(ticker, "매수 주문 실패")
                    _notify_embed(create_trade_embed({
                        "side": "BUY", "name": name, "ticker": ticker, "qty": quantity,
                        "price": order_price, "trade_status": "failed",
                        "strategy_details": {"error": err}
                    }), key=f"trader_buy_fail_{ticker}", cooldown=30)
            else:
                logger.info(f"  -> [모의] {name}({ticker}) {quantity}주 @{order_price:,.0f}원 지정가 매수 실행.")
                remaining_cash -= actual_spent
                _notify_text(f"🧪 [모의] BUY {name}({ticker}) x{quantity} @ {order_price:,.0f}",
                             key=f"paper_buy_{ticker}", cooldown=30)

            logger.info(f"  -> 남은 예산: {remaining_cash:,.0f}원")
            time.sleep(0.3)

        if executed_buy:
            time.sleep(5)
            # ✅ 실제 주문 시에만 최신 동기화
            self._update_account_info()

# ───────── 실행 진입 ─────────
if __name__ == "__main__":
    try:
        trader = Trader(settings)

        # ✅ 시작 시 강제 갱신은 환경변수로만 허용
        if REFRESH_ON_START:
            trader._update_account_info()

        # 파일 기준으로 계좌/보유 로드
        _, initial_holdings = trader.get_account_info_from_files()
        valid_holdings = [h for h in initial_holdings if trader._parse_krw(h.get("hldg_qty", 0)) > 0]

        if valid_holdings:
            trader.run_sell_logic(valid_holdings)
        else:
            logger.info("보유 종목이 없어 매도 로직을 건너뜁니다.")

        cash_after_sell, holdings_after_sell = trader.get_account_info_from_files()
        trader.run_buy_logic(cash_after_sell, holdings_after_sell)

        logger.info("모든 트레이딩 로직 실행 완료.")
        _notify_text("✅ 트레이딩 로직 실행 완료", key="trader_done", cooldown=60)
    except Exception as e:
        logger.critical(f"트레이더 실행 중 심각한 오류 발생: {e}", exc_info=True)
        _notify_text(f"🛑 트레이더 치명적 오류: {str(e)[:1800]}", key="trader_fatal", cooldown=30)
