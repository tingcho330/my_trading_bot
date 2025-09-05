# src/strategies.py

import logging
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
from settings import Settings

logger = logging.getLogger(__name__)

def _to_float_safe(val, default: float = 0.0) -> float:
    """문자열/숫자를 안전하게 float 변환 (쉼표 제거 포함). 실패 시 default."""
    try:
        if val is None:
            return default
        if isinstance(val, str):
            return float(val.replace(",", ""))
        return float(val)
    except Exception:
        return default

def _to_float_or_none(val) -> Optional[float]:
    """실패 시 None을 반환(디폴트 0 금지)."""
    try:
        if val is None:
            return None
        if isinstance(val, str):
            val = val.replace(",", "")
        x = float(val)
        return x
    except Exception:
        return None


class BaseStrategy:
    def __init__(self, settings: Settings):
        self.params = settings.strategy_params

    def decide_sell(self, holding: Dict, stock_info: Dict) -> bool:
        current_price = _to_float_safe(stock_info.get("Price"))
        avg_price = _to_float_safe(holding.get("pchs_avg_pric"), 0.0)

        if current_price <= 0 or avg_price <= 0:
            return False

        profit_pct = (current_price - avg_price) / avg_price

        stop_loss_pct = self.params.get("stop_loss_pct", 0.05)
        take_profit_pct = self.params.get("take_profit_pct", 0.10)

        if profit_pct <= -stop_loss_pct:
            return True
        if profit_pct >= take_profit_pct:
            return True
        return False


class RsiReversalStrategy(BaseStrategy):
    def decide_sell(self, holding: Dict, stock_info: Dict) -> bool:
        rsi = _to_float_safe(stock_info.get("RSI"), 50.0)
        return rsi >= 70.0


class TrendFollowingStrategy(BaseStrategy):
    def decide_sell(self, holding: Dict, stock_info: Dict) -> bool:
        ma50 = _to_float_safe(stock_info.get("MA50"))
        ma200 = _to_float_safe(stock_info.get("MA200"))
        return ma50 > 0 and ma200 > 0 and ma50 < ma200


class AdvancedTechnicalStrategy(BaseStrategy):
    """screener.py가 daily_chart/investor_flow를 records 형태로 제공한다고 가정."""
    def decide_sell(self, holding: Dict, stock_info: Dict) -> bool:
        # --- daily_chart: 안전 변환 및 필수 컬럼 검사 ---
        daily_chart_data = stock_info.get("daily_chart")
        if not daily_chart_data:
            return False
        try:
            df_daily = pd.DataFrame(daily_chart_data)
        except Exception:
            return False

        needed_dc = {"Open", "High", "Low", "Close", "Volume"}
        if not needed_dc.issubset(df_daily.columns):
            return False
        if len(df_daily) < 20:
            return False

        # 타입 캐스팅(숫자화)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df_daily[col] = pd.to_numeric(df_daily[col], errors="coerce")
        if df_daily[["Close", "Open", "Volume"]].tail(1).isna().any(axis=None):
            return False

        current_price = _to_float_safe(stock_info.get("Price"))
        if current_price <= 0:
            return False

        # 기준1: 종가 < MA20
        ma20 = df_daily["Close"].rolling(20, min_periods=20).mean()
        if pd.notna(ma20.iloc[-1]) and current_price < ma20.iloc[-1]:
            return True

        # 기준2: 거래량 급감 + 음봉
        vol_ma20 = df_daily["Volume"].rolling(20, min_periods=20).mean()
        if pd.notna(vol_ma20.iloc[-1]):
            last = df_daily.iloc[-1]
            if (last["Volume"] < vol_ma20.iloc[-1] * 0.4) and (last["Close"] < last["Open"]):
                return True

        # --- investor_flow: 안전 변환 ---
        investor_flow_data = stock_info.get("investor_flow")
        if investor_flow_data:
            try:
                df_flow = pd.DataFrame(investor_flow_data)
                needed_if = {"기관합계", "외국인합계"}
                if needed_if.issubset(df_flow.columns):
                    df_flow["기관합계"] = pd.to_numeric(df_flow["기관합계"], errors="coerce")
                    df_flow["외국인합계"] = pd.to_numeric(df_flow["외국인합계"], errors="coerce")
                    last3 = df_flow.tail(3).dropna(subset=["기관합계", "외국인합계"])
                    if len(last3) >= 3:
                        if ((last3["기관합계"] < 0) & (last3["외국인합계"] < 0)).all():
                            return True
            except Exception:
                # 수급 데이터 문제시 신호 미발생으로 취급
                pass

        return False


class DynamicAtrStrategy(BaseStrategy):
    """ATR 기반 동적 손절/익절. screener 결과에 'ATR' 필요."""
    def decide_sell(self, holding: Dict, stock_info: Dict) -> bool:
        current_price = _to_float_safe(stock_info.get("Price"))
        avg_price = _to_float_safe(holding.get("pchs_avg_pric"), 0.0)
        atr = _to_float_or_none(stock_info.get("ATR"))  # ← None 허용

        # ATR 미존재/비양수 → 즉시 패스
        if current_price <= 0 or avg_price <= 0 or atr is None or atr <= 0:
            return False

        atr_k_stop = self.params.get("atr_k_stop", 2.0)
        atr_k_profit = self.params.get("atr_k_profit", 4.0)

        stop_loss_price = avg_price - (atr * atr_k_stop)
        take_profit_price = avg_price + (atr * atr_k_profit)

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
                AdvancedTechnicalStrategy,
                DynamicAtrStrategy,
            ]

        self.strategies: List[BaseStrategy] = [cls(settings) for cls in strategies_to_use]

    def decide_sell(self, holding: Dict, stock_info: Dict) -> Tuple[bool, str]:
        # 공통(고정 손절/익절) 먼저 체크
        if BaseStrategy(self.settings).decide_sell(holding, stock_info):
            current_price = _to_float_safe(stock_info.get("Price"), 0.0)
            avg_price = _to_float_safe(holding.get("pchs_avg_pric"), 1.0)
            if avg_price <= 0:
                avg_price = 1.0
            profit_pct = (current_price - avg_price) / avg_price
            reason = "고정 이익실현" if profit_pct > 0 else "고정 손절"
            return True, f"{reason} ({profit_pct:.2%})"

        # 가중 전략 합산
        weighted_sum, reasons = 0.0, []
        for strategy in self.strategies:
            try:
                if strategy.decide_sell(holding, stock_info):
                    name = strategy.__class__.__name__
                    weight = float(self.weights.get(name, 0.0))
                    if weight > 0:
                        weighted_sum += weight
                        reasons.append(f"{name}(가중치:{weight})")
            except Exception as e:
                logger.debug("Strategy %s 실행 중 오류: %s", strategy.__class__.__name__, e)

        if weighted_sum >= float(self.sell_threshold):
            return True, f"가중합({weighted_sum:.2f}) ≥ 임계값({self.sell_threshold}): [{', '.join(reasons)}]"

        return False, "-"
