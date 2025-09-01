import json
import logging
import os
import random
import time
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

# 공통 유틸리티 및 설정/전략 모듈 임포트
from utils import setup_logging, find_latest_file, OUTPUT_DIR
from api.kis_auth import KIS
from risk_manager import RiskManager
from settings import settings
from recorder import initialize_db, record_trade

# 로깅 설정
setup_logging()
logger = logging.getLogger("trader")

# 경로 설정
COOLDOWN_FILE = OUTPUT_DIR / "cooldown.json"
ACCOUNT_SCRIPT_PATH = "/app/src/account.py" # 스크립트 경로 수정

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

    def _load_all_stock_data(self) -> Dict[str, Dict]:
        full_screener_file = find_latest_file("screener_full_*.json")
        if not full_screener_file:
            logger.warning("screener_full.json 파일을 찾을 수 없어, 실시간 데이터 조회만 가능합니다.")
            return {}
        try:
            with open(full_screener_file, 'r', encoding='utf-8') as f:
                all_stocks = json.load(f)
            return {stock['Ticker']: stock for stock in all_stocks}
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"{full_screener_file.name} 파일 로드 실패: {e}")
            return {}

    def _update_account_info(self):
        logger.info("최신 계좌 정보 업데이트를 위해 account.py를 실행합니다...")
        try:
            subprocess.run(
                ["python", str(ACCOUNT_SCRIPT_PATH)],
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            logger.info("account.py 실행 완료.")
        except FileNotFoundError:
            logger.error(f"스크립트를 찾을 수 없습니다: {ACCOUNT_SCRIPT_PATH}")
        except subprocess.CalledProcessError as e:
            logger.error(f"account.py 실행 중 오류 발생 (Exit Code: {e.returncode}):\n{e.stderr}")
        except Exception as e:
            logger.error(f"계좌 정보 업데이트 중 예외 발생: {e}")

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

    def get_account_info_from_files(self) -> Tuple[int, List[Dict]]:
        available_cash = 0
        summary_file = find_latest_file("summary_*.json")

        if not summary_file:
            logger.warning("summary.json 파일을 찾을 수 없어 주문 가능 금액을 0으로 설정합니다.")
        else:
            logger.info(f"가용 예산 조회를 위해 summary 파일 읽기: {summary_file}")
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_json = json.load(f)
                
                # 중첩된 JSON 구조 파싱: data -> list[0] -> "0"
                summary_data = summary_json.get("data", [])
                core_data = {}
                if summary_data and isinstance(summary_data, list) and isinstance(summary_data[0], dict):
                    core_data = summary_data[0].get("0", summary_data[0])

                if core_data and isinstance(core_data, dict):
                    # 우선순위에 따라 가용 현금 키 탐색
                    cash_keys_priority = ["prvs_rcdl_excc_amt", "nxdy_excc_amt", "ord_psbl_cash", "dnca_tot_amt"]
                    found_key, cash_str = None, "0"
                    
                    for key in cash_keys_priority:
                        if key in core_data and core_data[key]:
                            cash_str = core_data[key]
                            found_key = key
                            logger.info(f"가용 예산 키 '{found_key}' 발견. 값: '{cash_str}'")
                            break
                    
                    if not found_key:
                        logger.warning(f"우선순위 키를 찾지 못했습니다. 사용 가능한 키: {list(core_data.keys())}")

                    available_cash = self._parse_krw(cash_str)
                    logger.info(f"✅ 최종 파싱된 가용 예산: {available_cash:,} 원")
                else:
                    logger.error("summary.json 파일에서 유효한 데이터 구조(data -> [0] -> '0')를 찾지 못했습니다.")

            except json.JSONDecodeError as e:
                logger.error(f"{summary_file.name} 파일이 비어있거나 형식이 잘못되었습니다: {e}")
            except Exception as e:
                logger.error(f"{summary_file.name} 파일 처리 중 오류 발생: {e}", exc_info=True)

        holdings = []
        balance_file = find_latest_file("balance_*.json")
        if not balance_file:
            logger.warning("balance.json 파일을 찾을 수 없어 보유 종목이 없는 것으로 간주합니다.")
        else:
            try:
                with open(balance_file, 'r', encoding='utf-8') as f:
                    balance_json = json.load(f)
                
                holdings_data = balance_json.get("data", [])
                holdings = [h for h in holdings_data if self._parse_krw(h.get("hldg_qty", 0)) > 0]
                logger.info(f"보유 종목 로드 완료: 총 {len(holdings_data)}개 항목 중 유효 보유 {len(holdings)} 종목 (from {balance_file.name})")

            except json.JSONDecodeError as e:
                logger.error(f"{balance_file.name} 파일이 비어있거나 형식이 잘못되었습니다: {e}")
            except Exception as e:
                logger.error(f"{balance_file.name} 파일 처리 중 오류: {e}")
        
        return available_cash, holdings
    
    def run_sell_logic(self, holdings: List[Dict]):
        # ... (이전과 동일한 내용) ...
        logger.info(f"--------- 보유 종목 {len(holdings)}개 매도 로직 실행 ---------")
        
        executed_sell = False
        if not holdings:
            logger.info("매도할 보유 종목이 없습니다.")
            return

        for holding in holdings:
            ticker = holding.get("pdno")
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
                    result = self._order_cash_safe(ord_dv="01", pdno=ticker, ord_dvsn="01", ord_qty=str(quantity), ord_unpr="0")
                    if result.get('ok'):
                        logger.info(f"매도 주문 성공: {result.get('msg1')}")
                        current_price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                        current_price = self._parse_krw(current_price_df['stck_prpr'].iloc[0]) if not current_price_df.empty else 0
                        trade_data = {"side": "sell", "ticker": ticker, "name": name, "qty": quantity, "price": current_price, "trade_status": "completed", "strategy_details": {"reason": reason}}
                        record_trade(trade_data)
                        self._add_to_cooldown(ticker, "매도 완료")
                    else:
                        logger.error(f"{name}({ticker}) 매도 주문 실패: {result.get('msg1')}")
                        trade_data = {"side": "sell", "ticker": ticker, "name": name, "qty": quantity, "price": 0, "trade_status": "failed", "strategy_details": {"error": result.get('msg1', 'Unknown error')}}
                        record_trade(trade_data)
                else:
                    logger.info(f"[모의] {name}({ticker}) {quantity}주 시장가 매도 실행.")
                    self._add_to_cooldown(ticker, "모의 매도 완료")
        
        if executed_sell:
            time.sleep(5)
            self._update_account_info()

    def run_buy_logic(self, available_cash: int, holdings: List[Dict]):
        # ... (이전과 동일한 내용) ...
        logger.info(f"--------- 신규 매수 로직 실행 (가용 예산: {available_cash:,} 원) ---------")

        if available_cash < 10000:
            logger.warning("가용 예산이 부족하여 매수 로직을 실행할 수 없습니다.")
            return

        trade_plan_file = find_latest_file("gpt_trades_*.json")
        if not trade_plan_file:
            logger.warning("매수 계획 파일(gpt_trades_*.json)이 없어 매수를 건너뜁니다.")
            return
            
        with open(trade_plan_file, 'r', encoding='utf-8') as f:
            trade_plans = json.load(f)

        buy_plans = [p for p in trade_plans if p.get("결정") == "매수"]
        if not buy_plans:
            logger.info("매수 결정이 내려진 종목이 없습니다.")
            return

        holding_tickers = {h.get("pdno") for h in holdings}
        new_targets = []
        for plan in buy_plans:
            ticker = plan["stock_info"]["Ticker"]
            name = plan["stock_info"]["Name"]
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
            return

        targets_to_buy = new_targets[:slots_to_fill]
        if not targets_to_buy:
            logger.info("신규로 매수할 최종 대상이 없습니다.")
            return
            
        remaining_cash = available_cash
        executed_buy = False
        logger.info(f"총 {len(targets_to_buy)}개 종목 신규 매수 시도. 유동적 예산 배분 적용.")

        for i, plan in enumerate(targets_to_buy):
            stock_info = plan["stock_info"]
            ticker, name = stock_info["Ticker"], stock_info["Name"]
            slots_left = len(targets_to_buy) - i
            budget_for_this_stock = remaining_cash // slots_left
            logger.info(f"  -> [{i+1}/{len(targets_to_buy)}] {name}({ticker}) 배분 예산: {budget_for_this_stock:,.0f}원")
            current_price = self._parse_krw(stock_info.get("Price", 0))
            if current_price == 0:
                price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                if not price_df.empty:
                    current_price = self._parse_krw(price_df['stck_prpr'].iloc[0])
                else:
                    logger.warning(f"  -> [{name}({ticker})] 현재가 조회 실패. 매수를 건너뜁니다.")
                    continue
            
            tick_size = get_tick_size(current_price)
            order_price = current_price + (tick_size * random.randint(1, 3))
            quantity = int(budget_for_this_stock // order_price)
            if quantity == 0:
                logger.warning(f"  -> [{name}({ticker})] 예산 부족으로 매수 불가.")
                continue

            logger.info(f"  -> 매수 준비: {name}({ticker}), 수량: {quantity}주, 지정가: {order_price:,.0f}원")
            actual_spent = quantity * order_price
            executed_buy = True
            
            if self.is_real_trading:
                result = self._order_cash_safe(ord_dv="02", pdno=ticker, ord_dvsn="00", ord_qty=str(quantity), ord_unpr=str(int(order_price)))
                if result.get('ok'):
                    logger.info(f"  -> 매수 주문 성공: {result.get('msg1')}")
                    trade_data = {"side": "buy", "ticker": ticker, "name": name, "qty": quantity, "price": order_price, "trade_status": "active", "gpt_analysis": plan}
                    record_trade(trade_data)
                    remaining_cash -= actual_spent
                else:
                    logger.error(f"  -> {name}({ticker}) 매수 주문 실패: {result.get('msg1')}")
                    trade_data = {"side": "buy", "ticker": ticker, "name": name, "qty": quantity, "price": order_price, "trade_status": "failed", "strategy_details": {"error": result.get('msg1', 'Unknown error')}, "gpt_analysis": plan}
                    record_trade(trade_data)
                    self._add_to_cooldown(ticker, "매수 주문 실패")
            else:
                logger.info(f"  -> [모의] {name}({ticker}) {quantity}주 @{order_price:,.0f}원 지정가 매수 실행.")
                remaining_cash -= actual_spent
            
            logger.info(f"  -> 남은 예산: {remaining_cash:,.0f}원")
            time.sleep(0.5)

        if executed_buy:
            time.sleep(5)
            self._update_account_info()

if __name__ == "__main__":
    try:
        trader = Trader(settings)
        trader._update_account_info()
        _, initial_holdings = trader.get_account_info_from_files()
        valid_holdings = [h for h in initial_holdings if trader._parse_krw(h.get("hldg_qty", 0)) > 0]

        if valid_holdings:
            trader.run_sell_logic(valid_holdings)
        else:
            logger.info("보유 종목이 없어 매도 로직을 건너뜁니다.")
            
        cash_after_sell, holdings_after_sell = trader.get_account_info_from_files()
        trader.run_buy_logic(cash_after_sell, holdings_after_sell)
        logger.info("모든 트레이딩 로직 실행 완료.")
    except Exception as e:
        logger.critical(f"트레이더 실행 중 심각한 오류 발생: {e}", exc_info=True)
