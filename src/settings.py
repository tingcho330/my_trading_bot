# src/settings.py
import json
from pathlib import Path
from typing import Dict, Any

class Settings:
    def __init__(self, config_path: Path = Path("/app/config/config.json")):
        self._config = self._load_config(config_path)

        # ── 최상위 기본값 ─────────────────────────────────────────────
        self._config.setdefault("trading_environment", "vps")  # "prod"면 실매매

        # 섹션 존재 보장
        self._config.setdefault("strategy_params", {})
        self._config.setdefault("risk_params", {})
        self._config.setdefault("gpt_params", {})
        self._config.setdefault("notifications", {})
        self._config.setdefault("trading_params", {})
        self._config.setdefault("trading_guards", {})
        self._config.setdefault("screener_params", {})
        self._config.setdefault("reporting", {})

        # 섹션 핸들
        self.strategy_params: Dict[str, Any]   = self._config["strategy_params"]
        self.risk_params: Dict[str, Any]       = self._config["risk_params"]
        self.gpt_params: Dict[str, Any]        = self._config["gpt_params"]
        self.notifications: Dict[str, Any]     = self._config["notifications"]
        self.trading_params: Dict[str, Any]    = self._config["trading_params"]
        self.trading_guards: Dict[str, Any]    = self._config["trading_guards"]
        self.screener_params: Dict[str, Any]   = self._config["screener_params"]
        self.reporting: Dict[str, Any]         = self._config["reporting"]

        # ── 기본 전략 파라미터(기존) ─────────────────────────────────
        self.strategy_params.setdefault("atr_k_stop", 2.0)
        self.strategy_params.setdefault("atr_k_profit", 4.0)
        self.strategy_params.setdefault("sell_threshold", 1.0)
        self.strategy_params.setdefault("weights", {
            "RsiReversalStrategy": 0.5,
            "TrendFollowingStrategy": 0.8,
            "AdvancedTechnicalStrategy": 0.6,
            "DynamicAtrStrategy": 0.7
        })

        # ── 리스크 파라미터 ─────────────────────────────────────────
        self.risk_params.setdefault("atr_period", 14)
        self.risk_params.setdefault("cooldown_period_days", 10)
        self.risk_params.setdefault("max_positions", 4)
        self.risk_params.setdefault("cooldown_fail_threshold", 2)

        # ── 노티 기본값 ──────────────────────────────────────────────
        self.notifications.setdefault("discord_cooldown_sec", 60)
        self.notifications.setdefault("snapshot_change_threshold_pct", 1.0)

        # ── 트레이딩 파라미터(새로 추가된 섹션) ───────────────────────
        tp = self.trading_params
        tp.setdefault("buy_time_windows", ["09:05-14:50"])
        tp.setdefault("sell_time_windows", ["09:05-15:10"])
        tp.setdefault("allow_rebuy", False)
        tp.setdefault("max_positions", self.risk_params.get("max_positions", 4))
        tp.setdefault("max_legs_per_ticker", 1)
        tp.setdefault("per_ticker_max_weight", 1.0)   # trader.py 내부에서 안전 클램프
        tp.setdefault("min_order_cash", 0)            # 금액 기준 최소 주문
        tp.setdefault("rebuy_atr_k", 0.0)
        tp.setdefault("rebuy_rsi_ceiling", 100.0)
        tp.setdefault("min_cash_reserve", 0)
        tp.setdefault("cash_buffer_ratio", 0.0)       # 가용 현금 버퍼

        # 분할 매수 설정 (P1 대응)
        split = tp.setdefault("split_buy", {})
        split.setdefault("enabled", False)
        split.setdefault("slices", 3)                 # 분할 개수
        # weights 미설정 시 내부에서 균등 분배
        split.setdefault("weights", [])               # 예: [0.5, 0.3, 0.2]
        split.setdefault("ladder_ticks", [0, 1, 2])   # 각 슬라이스의 틱 가산
        split.setdefault("interval_sec", 0.6)         # 슬라이스 간 인터벌
        split.setdefault("jitter_sec", 0.15)          # 인터벌 지터
        split.setdefault("min_qty", 250)              # 분할 전환 최소 수량
        split.setdefault("min_cash_per_slice", tp.get("min_order_cash", 0))

        # ── 트레이딩 가드 ────────────────────────────────────────────
        tg = self.trading_guards
        tg.setdefault("skip_when_low_funds", False)
        tg.setdefault("min_total_cash_to_trade", 0)
        tg.setdefault("auto_shrink_slots", True)      # 현금에 따라 슬롯 자동 축소

        # ── 스크리너 동작 파라미터 ───────────────────────────────────
        sp = self.screener_params
        sp.setdefault("affordability_filter", False)  # 가용현금 기반 필터 강제 여부

        # ── 리포팅 ───────────────────────────────────────────────────
        rp = self.reporting
        rp.setdefault("coherent_summary", True)
        rp.setdefault("include_cash_breakdown", True)

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("config.json 최상위 구조는 객체여야 합니다.")
            return data

# 싱글턴처럼 사용할 수 있도록 인스턴스 생성
settings = Settings()
