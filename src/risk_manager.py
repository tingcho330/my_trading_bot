import pandas as pd
import time
import os
import json
from pathlib import Path
from datetime import datetime, time as time_obj
from zoneinfo import ZoneInfo  # ZoneInfo 임포트

# KIS API 모듈 임포트
from api.kis_auth import KIS

# --- KST 시간대 정의 ---
KST = ZoneInfo("Asia/Seoul")

# ───────────────── 경로 설정 ─────────────────
CONFIG_PATH = Path("/app/config/config.json")
OUTPUT_DIR = Path("/app/output")

def load_settings() -> dict:
    """설정 파일(config.json)을 로드합니다."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def find_latest_trade_plan() -> Path | None:
    """output 디렉토리에서 가장 최신의 gpt_trades 파일을 찾습니다."""
    trade_files = list(OUTPUT_DIR.glob("gpt_trades_*.json"))
    if not trade_files:
        return None
    return max(trade_files, key=lambda p: p.stat().st_mtime)

def is_market_open_day() -> bool:
    """오늘이 한국 주식 시장 개장일(월-금)인지 확인합니다."""
    # KST 기준으로 오늘의 요일을 확인
    return datetime.now(KST).weekday() < 5 # 0:월, 1:화, 2:수, 3:목, 4:금

def is_market_open_time() -> bool:
    """현재 '시간'이 주식 시장 개장 시간(09:00 ~ 15:30)인지 확인합니다."""
    # KST 기준으로 현재 시간을 확인
    now_time = datetime.now(KST).time()
    return time_obj(9, 0) <= now_time <= time_obj(15, 30)


class RiskManager:
    def __init__(self, settings: dict):
        self.settings = settings
        self.env = settings.get("trading_environment", "vps")
        self.kis = KIS(config={}, env=self.env)
        self.risk_params = settings.get("risk_params", {})
        self.stop_loss_pct = self.risk_params.get("stop_loss_pct", -0.05)  # -5%
        self.take_profit_pct = self.risk_params.get("take_profit_pct", 0.10)  # +10%
        self.trade_plans = self._load_trade_plans()

    def _load_trade_plans(self) -> dict:
        """gpt_trades 분석 파일을 로드하여 Ticker를 key로 하는 딕셔너리를 생성합니다."""
        latest_plan_file = find_latest_trade_plan()
        if not latest_plan_file:
            print("GPT 분석 파일(gpt_trades_*.json)을 찾을 수 없습니다. 설정 파일의 % 기반으로 작동합니다.")
            return {}

        print(f"GPT 분석 파일 로드: {latest_plan_file.name}")
        with open(latest_plan_file, 'r', encoding='utf-8') as f:
            plans = json.load(f)

        plan_map = {}
        for plan in plans:
            stock_info = plan.get("stock_info", {})
            ticker = stock_info.get("Ticker")
            stop_loss = stock_info.get("손절가")
            take_profit = stock_info.get("목표가")
            if ticker and stop_loss is not None and take_profit is not None:
                plan_map[ticker] = {
                    "stop_loss": float(stop_loss),
                    "take_profit": float(take_profit)
                }
        return plan_map


    def get_realtime_price(self, ticker: str) -> float:
        """실시간 주가를 조회합니다."""
        try:
            price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
            if not price_df.empty:
                return float(price_df['stck_prpr'].iloc[0])
        except Exception as e:
            print(f"Error fetching realtime price for {ticker}: {e}")
        return 0.0

    def monitor_holdings(self):
        """보유 주식을 실시간으로 모니터링하고 리스크를 관리합니다."""
        print("보유 주식 실시간 모니터링을 시작합니다...")
        
        today_str = datetime.now(KST).strftime("%Y%m%d")
        balance_file = OUTPUT_DIR / f"balance_{today_str}.json"

        if not balance_file.exists():
            balance_files = sorted(list(OUTPUT_DIR.glob("balance_*.json")), reverse=True)
            if not balance_files:
                print(f"balance 파일을 찾을 수 없습니다. account.py를 먼저 실행해주세요.")
                return
            balance_file = balance_files[0]
            print(f"오늘 날짜의 파일이 없어 가장 최신 파일인 {balance_file.name}을 사용합니다.")


        with open(balance_file, 'r', encoding='utf-8') as f:
            loaded_json = json.load(f)
            if isinstance(loaded_json, dict):
                balance_data = loaded_json.get("data", [])
            elif isinstance(loaded_json, list):
                balance_data = loaded_json
            else:
                balance_data = []

        if not balance_data:
            print("보유 주식이 없습니다.")
            return

        holdings_df = pd.DataFrame(balance_data)
        
        while True:
            if is_market_open_day() and is_market_open_time():
                print(f"\n--- {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} | 보유 종목 리스크 점검 ---")
                for _, holding in holdings_df.iterrows():
                    ticker = holding["pdno"]
                    prdt_name = holding["prdt_name"]
                    pchs_avg_pric = float(holding["pchs_avg_pric"])
                    hldg_qty = int(holding["hldg_qty"])
                    
                    if hldg_qty == 0:
                        continue

                    current_price = self.get_realtime_price(ticker)
                    if current_price == 0.0:
                        continue
                    
                    profit_loss_pct = (current_price - pchs_avg_pric) / pchs_avg_pric

                    stop_loss_price = 0
                    take_profit_price = 0
                    price_source = ""

                    if ticker in self.trade_plans:
                        stop_loss_price = self.trade_plans[ticker]['stop_loss']
                        take_profit_price = self.trade_plans[ticker]['take_profit']
                        price_source = "GPT 분석"
                    else:
                        stop_loss_price = pchs_avg_pric * (1 + self.stop_loss_pct)
                        take_profit_price = pchs_avg_pric * (1 + self.take_profit_pct)
                        price_source = "설정(%)"

                    print(f"[{prdt_name}({ticker})] 현재가: {current_price:,.0f} | 수익률: {profit_loss_pct:.2%} | 목표가: {take_profit_price:,.0f} | 손절가: {stop_loss_price:,.0f} (기준: {price_source})")

                    if current_price <= stop_loss_price:
                        print(f"!!! 손절매 조건 충족: {prdt_name}({ticker}) !!!")
                        # TODO: 매도 주문 실행 로직 추가
                    elif current_price >= take_profit_price:
                        print(f"*** 이익실현 조건 충족: {prdt_name}({ticker}) ***")
                        # TODO: 매도 주문 실행 로직 추가
                
                time.sleep(60)
            
            else:
                now_str = datetime.now(KST).strftime('%H:%M:%S')
                print(f"장이 열리지 않았습니다. 대기합니다. (현재 시간: {now_str} KST)", end='\r')
                time.sleep(300)

if __name__ == "__main__":
    try:
        settings = load_settings()
        risk_manager = RiskManager(settings)
        risk_manager.monitor_holdings()
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")