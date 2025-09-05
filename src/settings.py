# src/settings.py
import json
from pathlib import Path
from typing import Dict, Any

class Settings:
    def __init__(self, config_path: Path = Path("/app/config/config.json")):
        self._config = self._load_config(config_path)

        # 주요 섹션 노출
        self.strategy_params: Dict[str, Any] = self._config.get("strategy_params", {}) or {}
        self.risk_params: Dict[str, Any] = self._config.get("risk_params", {}) or {}
        self.gpt_params: Dict[str, Any] = self._config.get("gpt_params", {}) or {}
        self.notifications: Dict[str, Any] = self._config.get("notifications", {}) or {}

        # 하위 기본값 보강(누락 시 안전 가드)
        self.strategy_params.setdefault("atr_k_stop", 2.0)
        self.strategy_params.setdefault("atr_k_profit", 4.0)
        self.strategy_params.setdefault("sell_threshold", 1.0)
        self.strategy_params.setdefault("weights", {
            "RsiReversalStrategy": 0.5,
            "TrendFollowingStrategy": 0.8,
            "AdvancedTechnicalStrategy": 0.6,
            "DynamicAtrStrategy": 0.7
        })

        self.risk_params.setdefault("atr_period", 14)
        self.risk_params.setdefault("cooldown_period_days", 10)
        self.risk_params.setdefault("max_positions", 4)

        self.notifications.setdefault("discord_cooldown_sec", 60)
        self.notifications.setdefault("snapshot_change_threshold_pct", 1.0)

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
