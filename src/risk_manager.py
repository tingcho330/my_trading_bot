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
    def __init__(self, settings: dict):
        self.settings = settings
        self.env = settings.get("trading_environment", "vps")
        self.kis = KIS(config={}, env=self.env)
        self.risk_params = settings.get("risk_params", {})
        self.stop_loss_pct = self.risk_params.get("stop_loss_pct", -0.05)
        self.take_profit_pct = self.risk_params.get("take_profit_pct", 0.10)
        self.trade_plans = self._load_trade_plans()

    def _load_trade_plans(self) -> dict:
        """gpt_trades 분석 파일을 로드하여 Ticker를 key로 하는 딕셔너리를 생성합니다."""
        latest_plan_file = find_latest_file("gpt_trades_*.json")
        if not latest_plan_file:
            print("GPT 분석 파일(gpt_trades_*.json)을 찾을 수 없습니다.")
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
            if not price_df.empty and 'stck_prpr' in price_df.columns:
                return float(price_df['stck_prpr'].iloc[0])
        except Exception as e:
            print(f"실시간 가격 조회 오류 ({ticker}): {e}")
        return 0.0

    def check_sell_condition(self, holding: Dict[str, Any]) -> Tuple[str, str]:
        """
        보유 종목에 대한 매도 조건을 확인하고 결정과 이유를 반환합니다.
        
        반환: (결정, 이유) ex: ("SELL", "손절가 도달"), ("HOLD", "조건 미충족")
        """
        ticker = holding.get("pdno")
        prdt_name = holding.get("prdt_name", "N/A")
        pchs_avg_pric = float(holding.get("pchs_avg_pric", 0))

        if not ticker or pchs_avg_pric == 0:
            return "HOLD", "종목 정보 부족"

        current_price = self.get_realtime_price(ticker)
        if current_price == 0.0:
            return "HOLD", "현재가 조회 실패"

        stop_loss_price, take_profit_price, source = 0, 0, ""

        if ticker in self.trade_plans:
            stop_loss_price = self.trade_plans[ticker]['stop_loss']
            take_profit_price = self.trade_plans[ticker]['take_profit']
            source = "GPT 분석"
        else:
            stop_loss_price = pchs_avg_pric * (1 + self.stop_loss_pct)
            take_profit_price = pchs_avg_pric * (1 + self.take_profit_pct)
            source = "설정(%)"

        profit_loss_pct = (current_price - pchs_avg_pric) / pchs_avg_pric
        print(f"[{prdt_name}({ticker})] 현재가: {current_price:,.0f} | 수익률: {profit_loss_pct:.2%} | 목표가: {take_profit_price:,.0f} | 손절가: {stop_loss_price:,.0f} (기준: {source})")
        
        if current_price <= stop_loss_price:
            return "SELL", f"손절가({stop_loss_price:,.0f}원) 도달"
        elif current_price >= take_profit_price:
            return "SELL", f"이익실현가({take_profit_price:,.0f}원) 도달"
        
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
        
        while True:
            if is_market_open_day() and is_market_open_time():
                print(f"\n--- {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} | 보유 종목 리스크 점검 ---")
                for holding in balance_data:
                    if int(holding.get("hldg_qty", 0)) > 0:
                        decision, reason = self.check_sell_condition(holding)
                        if decision == "SELL":
                            print(f"!!! 매도 신호: {holding.get('prdt_name')} - {reason} !!!")
                
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