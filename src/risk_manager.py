# src/risk_manager.py
import pandas as pd
import time
import os
import json
from pathlib import Path
from datetime import datetime, time as time_obj
from zoneinfo import ZoneInfo
from typing import Tuple, Dict, Any

# KIS API 모듈 임포트
from api.kis_auth import KIS
# 새로 추가된 모듈 임포트
from settings import Settings
from strategies import StrategyMixer

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

def find_latest_file(pattern: str) -> Path | None:
    """output 디렉토리에서 특정 패턴의 가장 최신 파일을 찾습니다."""
    try:
        files = sorted(OUTPUT_DIR.glob(pattern), key=os.path.getmtime)
        return files[-1] if files else None
    except (FileNotFoundError, IndexError):
        return None

def is_market_open_day() -> bool:
    """오늘이 한국 주식 시장 개장일(월-금)인지 확인합니다."""
    return datetime.now(KST).weekday() < 5

def is_market_open_time() -> bool:
    """현재 '시간'이 주식 시장 개장 시간(09:00 ~ 15:30)인지 확인합니다."""
    now_time = datetime.now(KST).time()
    return time_obj(9, 0) <= now_time <= time_obj(15, 30)


class RiskManager:
    def __init__(self, settings: Settings): # Settings 객체를 받도록 수정
        self.settings = settings
        self.env = settings._config.get("trading_environment", "vps")
        self.kis = KIS(config={}, env=self.env)
        self.risk_params = settings.risk_params
        
        # StrategyMixer 초기화
        self.strategy_mixer = StrategyMixer(settings)

    def get_realtime_price(self, ticker: str) -> float:
        """실시간 주가를 조회합니다."""
        try:
            price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
            if not price_df.empty and 'stck_prpr' in price_df.columns:
                return float(price_df['stck_prpr'].iloc[0])
        except Exception as e:
            print(f"실시간 가격 조회 오류 ({ticker}): {e}")
        return 0.0

    def check_sell_condition(self, holding: Dict[str, Any], stock_info: Dict[str, Any]) -> Tuple[str, str]:
        """
        보유 종목에 대한 매도 조건을 확인하고 결정과 이유를 반환합니다.
        StrategyMixer를 사용하도록 수정되었습니다.
        """
        ticker = holding.get("pdno")
        prdt_name = holding.get("prdt_name", "N/A")
        avg_price_str = holding.get("pchs_avg_pric", "0")
        
        try:
            avg_price = float(avg_price_str)
        except (ValueError, TypeError):
            avg_price = 0.0

        if not ticker or avg_price <= 0:
            return "HOLD", "종목 정보 부족"

        # 실시간 가격 조회 (stock_info에 없을 경우)
        if "Price" not in stock_info:
            current_price = self.get_realtime_price(ticker)
            if current_price == 0.0:
                return "HOLD", "현재가 조회 실패"
            stock_info["Price"] = current_price
        
        # StrategyMixer를 통해 매도 여부 결정
        should_sell, reason = self.strategy_mixer.decide_sell(holding, stock_info)
        
        current_price = stock_info.get("Price", 0)
        profit_loss_pct = (current_price - avg_price) / avg_price if avg_price > 0 else 0
        
        print(f"[{prdt_name}({ticker})] 현재가: {current_price:,.0f} | 수익률: {profit_loss_pct:.2%} | 매도판단: {should_sell} | 근거: {reason}")
        
        if should_sell:
            return "SELL", reason
        
        return "HOLD", "매도 조건 미충족"

    def monitor_holdings(self):
        """(독립 실행용) 보유 주식을 실시간으로 모니터링합니다."""
        print("보유 주식 실시간 모니터링을 시작합니다...")
        
        balance_file = find_latest_file("balance_*.json")
        if not balance_file:
            print("balance 파일을 찾을 수 없습니다. account.py를 먼저 실행해주세요.")
            return

        with open(balance_file, 'r', encoding='utf-8') as f:
            balance_data = json.load(f).get("data", [])
            
        if not balance_data:
            print("보유 주식이 없습니다.")
            return
            
        # screener_full...json 파일에서 모든 종목의 상세 정보를 로드
        full_screener_file = find_latest_file("screener_full_*.json")
        all_stock_data = {}
        if full_screener_file:
            with open(full_screener_file, 'r', encoding='utf-8') as f:
                all_stocks = json.load(f)
            all_stock_data = {stock['Ticker']: stock for stock in all_stocks}

        while True:
            if is_market_open_day() and is_market_open_time():
                print(f"\n--- {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} | 보유 종목 리스크 점검 ---")
                for holding in balance_data:
                    if int(holding.get("hldg_qty", 0)) > 0:
                        ticker = holding.get("pdno")
                        stock_info = all_stock_data.get(ticker, {})
                        decision, reason = self.check_sell_condition(holding, stock_info)
                        if decision == "SELL":
                            print(f"!!! 매도 신호: {holding.get('prdt_name')} - {reason} !!!")
                
                time.sleep(60)
            else:
                now_str = datetime.now(KST).strftime('%H:%M:%S')
                print(f"장이 열리지 않았습니다. 대기합니다. (현재 시간: {now_str} KST)", end='\r')
                time.sleep(300)

if __name__ == "__main__":
    try:
        # settings 인스턴스를 직접 사용하도록 수정
        from settings import settings
        risk_manager = RiskManager(settings)
        # monitor_holdings는 stock_info를 필요로 하므로, 단독 실행 시 제한이 있을 수 있습니다.
        # 이 기능을 사용하려면 screener_full...json 파일이 필요합니다.
        risk_manager.monitor_holdings() 
        print("RiskManager가 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")