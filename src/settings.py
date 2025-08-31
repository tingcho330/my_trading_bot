# src/settings.py

import json
from pathlib import Path
from typing import Dict, Any

class Settings:
    def __init__(self, config_path: Path = Path("/app/config/config.json")):
        self._config = self._load_config(config_path)
        self.strategy_params = self._config.get("strategy_params", {})
        self.risk_params = self._config.get("risk_params", {})
        # 필요에 따라 다른 설정들을 속성으로 추가할 수 있습니다.

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

# 싱글턴처럼 사용할 수 있도록 인스턴스 생성
settings = Settings()