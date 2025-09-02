# src/utils.py
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

# ────────────────────────────────
# 공개 심볼 (다른 모듈에서 무엇을 가져갈지 명시)
# ────────────────────────────────
__all__ = [
    "KST",
    "OUTPUT_DIR",
    "CACHE_DIR",
    "CONFIG_PATH",
    "setup_logging",
    "load_config",
    "is_market_open_day",
    "find_latest_file",
    "cache_save",
    "cache_load",
    "load_account_files_with_retry",
    "extract_cash_from_summary",
]

# ────────────────────────────────
# KST 타임존 정의
# ────────────────────────────────
KST = ZoneInfo("Asia/Seoul")

# ────────────────────────────────
# 공통 경로
#   - 환경변수로 오버라이드 가능
# ────────────────────────────────
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output")).resolve()
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(OUTPUT_DIR / "cache"))).resolve()
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/app/config/config.json")).resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────
# 로깅 설정
#   - 모든 로그 타임스탬프를 KST 기준으로 출력
#   - 중복 포맷팅 방지
# ────────────────────────────────
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """통합 로깅 시스템 설정 (KST 타임스탬프). 여러 번 호출해도 안전."""
    root = logging.getLogger()
    root.setLevel(level)

    # 시간 포맷을 KST로 변환
    logging.Formatter.converter = lambda *args: datetime.now(KST).timetuple()
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # 기본 핸들러가 없으면 하나 추가
    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    else:
        # 모든 핸들러에 포맷 적용
        for h in root.handlers:
            h.setFormatter(fmt)
            h.setLevel(level)

    root.debug(
        "logging initialized (KST). OUTPUT_DIR=%s, CACHE_DIR=%s, CONFIG_PATH=%s",
        str(OUTPUT_DIR),
        str(CACHE_DIR),
        str(CONFIG_PATH),
    )
    return root

# ────────────────────────────────
# 설정 파일 로더
#   - /app/config/config.json 기본
#   - CONFIG_PATH 환경변수로 경로 재정의 가능
# ────────────────────────────────
def load_config() -> Optional[Dict[str, Any]]:
    """
    설정 파일을 로드합니다.
    반환: dict (성공) / None (실패)
    """
    logger = logging.getLogger(__name__)
    path = CONFIG_PATH
    try:
        if not path.exists():
            logger.error("설정 파일을 찾을 수 없습니다: %s", path)
            return None
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if not isinstance(cfg, dict):
                logger.error("설정 파일의 최상위 구조가 dict가 아닙니다: %s", path)
                return None
            logger.info("설정 로드 완료: %s", path)
            return cfg
    except (json.JSONDecodeError, OSError) as e:
        logger.error("설정 파일 읽기 실패(%s): %s", path, e)
        return None

# ────────────────────────────────
# 장 개장일 체크 (간단: 주말 제외)
#   - 필요 시 한국거래소 휴장일 캘린더로 확장
# ────────────────────────────────
def is_market_open_day() -> bool:
    """한국 거래소 기준 평일 여부(토/일 제외)."""
    today = datetime.now(KST).date()
    return today.weekday() < 5  # 0=월 ... 6=일

# ────────────────────────────────
# 최신 파일 찾기 (OUTPUT_DIR 기준)
# ────────────────────────────────
def find_latest_file(pattern: str) -> Optional[Path]:
    """OUTPUT_DIR 안에서 glob 패턴에 맞는 파일 중 최신 파일 경로 반환."""
    candidates = list(OUTPUT_DIR.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

# ────────────────────────────────
# 캐시 유틸리티 (pickle)
# ────────────────────────────────
def cache_save(prefix: str, key: str, data: Any) -> None:
    import pickle
    p = CACHE_DIR / f"{prefix}_{key}.pkl"
    try:
        with open(p, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logging.getLogger(__name__).warning("캐시 저장 실패(%s): %s", p, e)

def cache_load(prefix: str, key: str) -> Any:
    import pickle
    p = CACHE_DIR / f"{prefix}_{key}.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.getLogger(__name__).warning("캐시 로드 실패(%s): %s", p, e)
        return None

# ────────────────────────────────
# JSON 파일 로드 & 계좌/잔고 파싱
# ────────────────────────────────
def _read_json(p: Path) -> Optional[dict]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger(__name__).error("JSON 읽기 실패: %s (%s)", p.name, e)
        return None

def _parse_summary_payload(obj: dict) -> Dict[str, str]:
    """
    account.py 저장 포맷 예시:
    {"comments": {...}, "data": [ { "0": { ... } } ]}  또는  {"data": [ { ... } ]}
    """
    if not obj:
        return {}
    data = obj.get("data", [])
    if not data or not isinstance(data, list):
        return {}
    first = data[0]
    if isinstance(first, dict) and "0" in first and isinstance(first["0"], dict):
        return dict(first["0"])
    if isinstance(first, dict):
        return dict(first)
    return {}

def _parse_balance_payload(obj: dict) -> List[Dict]:
    """보유 종목 리스트 파싱"""
    if not obj:
        return []
    data = obj.get("data", [])
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []

def _to_int_krw(v) -> int:
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        s = v.replace(",", "").strip()
        if s.startswith("-") and s[1:].isdigit():
            return int(s)
        return int(s) if s.isdigit() else 0
    return 0

def load_account_files_with_retry(
    summary_pattern: str = "summary_*.json",
    balance_pattern: str = "balance_*.json",
    max_wait_sec: int = 5,
) -> Tuple[Dict[str, str], List[Dict], Optional[Path], Optional[Path]]:
    """
    최신 summary/balance 파일을 읽고 (summary_dict, balance_list, summary_path, balance_path) 반환.
    파일 생성 직후일 수 있어 최대 max_wait_sec 동안 재시도.
    """
    logger = logging.getLogger(__name__)
    deadline = time.time() + max_wait_sec
    summary_path: Optional[Path] = None
    balance_path: Optional[Path] = None
    parsed_summary: Dict[str, str] = {}
    parsed_balance: List[Dict] = []

    while time.time() < deadline:
        if summary_path is None:
            summary_path = find_latest_file(summary_pattern)
        if balance_path is None:
            balance_path = find_latest_file(balance_pattern)

        ok = True
        if summary_path and summary_path.exists():
            js = _read_json(summary_path)
            parsed_summary = _parse_summary_payload(js)
        else:
            ok = False

        if balance_path and balance_path.exists():
            jb = _read_json(balance_path)
            parsed_balance = _parse_balance_payload(jb)

        if ok:
            return parsed_summary, parsed_balance, summary_path, balance_path
        time.sleep(0.5)

    # 마지막 시도
    if summary_path and summary_path.exists() and not parsed_summary:
        js = _read_json(summary_path)
        parsed_summary = _parse_summary_payload(js)
    if balance_path and balance_path.exists() and not parsed_balance:
        jb = _read_json(balance_path)
        parsed_balance = _parse_balance_payload(jb)

    if not parsed_summary:
        logger.warning("요약 파일을 찾지 못했거나 파싱 실패 (pattern=%s)", summary_pattern)
    return parsed_summary, parsed_balance, summary_path, balance_path

def extract_cash_from_summary(summary_dict: Dict[str, str]) -> Dict[str, int]:
    """
    summary.json에서 현금 관련 키를 추출하고 주문 가능 금액을 계산합니다.
      - prvs_rcdl_excc_amt: D+2 예수금 (가장 보수적)
      - nxdy_excc_amt: D+1 예수금
      - dnca_tot_amt: 총 예수금
    """
    if not summary_dict:
        return {"available_cash": 0}

    cash_map = {k: _to_int_krw(v) for k, v in summary_dict.items() if isinstance(k, str) and "amt" in k}

    # 주문 가능 금액 우선순위: D+2 > D+1 > 총 예수금
    if cash_map.get("prvs_rcdl_excc_amt", 0) > 0:
        available = cash_map["prvs_rcdl_excc_amt"]
    elif cash_map.get("nxdy_excc_amt", 0) > 0:
        available = cash_map["nxdy_excc_amt"]
    else:
        available = cash_map.get("dnca_tot_amt", 0)

    cash_map["available_cash"] = available
    return cash_map

# ────────────────────────────────
# 모듈 로드 확인용 디버그 로그
# ────────────────────────────────
logging.getLogger(__name__).debug(
    "utils loaded. exports: %s",
    ", ".join(__all__),
)
