# src/strategies.py

import logging
from typing import Dict, List, Tuple, Union
import pandas as pd
from settings import Settings

logger = logging.getLogger(__name__)

class BaseStrategy:
    def __init__(self, settings: Settings):
        self.params = settings.strategy_params
    def decide_sell(self, holding: Dict, stock_info: Dict) -> bool:
        current_price = stock_info.get("Price")
        avg_price = holding.get("pchs_avg_pric", 0) # 'avg_price'를 'pchs_avg_pric'로 수정
        if isinstance(avg_price, str):
            avg_price = float(avg_price)
            
        if not current_price or avg_price <= 0: return False
        
        profit_pct = (current_price - avg_price) / avg_price
        
        # risk_params에서 stop_loss_pct, take_profit_pct를 가져오도록 수정
        stop_loss_pct = self.params.get("stop_loss_pct", 0.05) 
        take_profit_pct = self.params.get("take_profit_pct", 0.10)

        if profit_pct <= -stop_loss_pct: return True
        if profit_pct >= take_profit_pct: return True
        return False

class RsiReversalStrategy(BaseStrategy):
    def decide_sell(self, holding: Dict, stock_info: Dict) -> bool:
        return stock_info.get("RSI", 50) >= 70

class TrendFollowingStrategy(BaseStrategy):
    def decide_sell(self, holding: Dict, stock_info: Dict) -> bool:
        ma50, ma200 = stock_info.get("MA50"), stock_info.get("MA200")
        return ma50 is not None and ma200 is not None and ma50 < ma200

class AdvancedTechnicalStrategy(BaseStrategy):
    # 이 전략은 현재 프로젝트에서 daily_chart, investor_flow 데이터를 수집하지 않으므로
    # 실제 동작 시 제한적일 수 있습니다. 우선 기본 로직만 유지합니다.
    def decide_sell(self, holding: Dict, stock_info: Dict) -> bool:
        df_daily: pd.DataFrame = stock_info.get("daily_chart")
        if df_daily is None or len(df_daily) < 20: return False
        
        current_price = stock_info.get("Price")
        if current_price < df_daily["Close"].rolling(20).mean().iloc[-1]: return True
        
        # 거래량 조건
        if (df_daily["Volume"].iloc[-1] < df_daily["Volume"].rolling(20).mean().iloc[-1] * 0.4 and 
            df_daily["Close"].iloc[-1] < df_daily["Open"].iloc[-1]):
            return True
            
        # 수급 데이터 (현재 미수집)
        df_flow: pd.DataFrame = stock_info.get("investor_flow")
        if df_flow is not None and len(df_flow) >= 3:
            last3 = df_flow.tail(3)
            if ((last3.get("기관합계", 0) < 0) & (last3.get("외국인합계", 0) < 0)).all():
                return True
        return False

class DynamicAtrStrategy(BaseStrategy):
    # 이 전략은 현재 프로젝트에서 ATR을 계산하지 않으므로, 
    # 실제 동작을 위해서는 screener.py에 ATR 계산 로직 추가가 필요합니다.
    def decide_sell(self, holding: Dict, stock_info: Dict) -> bool:
        current_price = stock_info.get("Price")
        avg_price = holding.get("pchs_avg_pric", 0) # 'avg_price'를 'pchs_avg_pric'로 수정
        if isinstance(avg_price, str):
            avg_price = float(avg_price)
        
        atr = stock_info.get("ATR") # 스크리너 결과에 ATR이 포함되어야 함
        
        if not all([current_price, avg_price > 0, atr]): return False
        
        stop_loss_price, take_profit_price = avg_price - (atr * 2), avg_price + (atr * 4)
        if current_price <= stop_loss_price or current_price >= take_profit_price:
            return True
        return False

class StrategyMixer:
    def __init__(self, settings: Settings, strategies_to_use: Union[List[type], None] = None):
        self.settings = settings
        sp = settings.strategy_params
        self.weights = sp.get("weights", {})
        self.sell_threshold = sp.get("sell_threshold", 0.5)
        
        if strategies_to_use is None:
            strategies_to_use = [
                RsiReversalStrategy, 
                TrendFollowingStrategy, 
                # AdvancedTechnicalStrategy, # daily_chart, investor_flow 데이터 부재로 비활성화
                # DynamicAtrStrategy,      # ATR 데이터 부재로 비활성화
            ]
        
        self.strategies: List[BaseStrategy] = [cls(settings) for cls in strategies_to_use]

    def decide_sell(self, holding: Dict, stock_info: Dict) -> Tuple[bool, str]:
        # 1. 기본 손절/익절 조건 우선 확인
        if BaseStrategy(self.settings).decide_sell(holding, stock_info):
            current_price = stock_info.get("Price", 0)
            avg_price = holding.get("pchs_avg_pric", 1)
            if isinstance(avg_price, str):
                avg_price = float(avg_price)
            if avg_price == 0: avg_price = 1

            profit_pct = (current_price - avg_price) / avg_price
            reason = '고정 이익실현' if profit_pct > 0 else '고정 손절'
            return True, f"{reason} ({profit_pct:.2%})"
        
        # 2. 여러 전략의 가중합 계산
        weighted_sum, reasons = 0.0, []
        for strategy in self.strategies:
            if strategy.decide_sell(holding, stock_info):
                name = strategy.__class__.__name__
                weight = self.weights.get(name, 0.0)
                if weight > 0:
                    weighted_sum += weight
                    reasons.append(f"{name}(가중치:{weight})")
        
        if weighted_sum >= self.sell_threshold:
            return True, f"가중합({weighted_sum:.2f}) ≥ 임계값({self.sell_threshold}): [{', '.join(reasons)}]"
            
        return False, "-"