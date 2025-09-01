# src/utils.py
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from zoneinfo import ZoneInfo

# ────────────────────────────────
# KST 타임존 정의
# ────────────────────────────────
KST = ZoneInfo("Asia/Seoul")

# ────────────────────────────────
# 공통 경로
# ────────────────────────────────
OUTPUT_DIR = Path("/app/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────
# 로깅 설정
# ────────────────────────────────
def setup_logging(level=logging.INFO):
    """통합 로깅 시스템 설정"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    logger = logging.getLogger()
    logger.info("통합 로깅 시스템이 KST 기준으로 설정되었습니다.")
    return logger

# ────────────────────────────────
# 장 개장일 체크 (단순 예시)
# ────────────────────────────────
def is_market_open_day() -> bool:
    """한국 거래소 기준 평일 여부"""
    today = datetime.now(KST).date()
    # 간단히 주말만 제외
    return today.weekday() < 5

# ────────────────────────────────
# 최신 파일 찾기
# ────────────────────────────────
def find_latest_file(pattern: str) -> Optional[Path]:
    """OUTPUT_DIR 안에서 최신 파일 경로 반환"""
    candidates = list(OUTPUT_DIR.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

# ────────────────────────────────
# JSON 파서 및 계좌/잔고 파싱
# ────────────────────────────────
def _read_json(p: Path) -> Optional[dict]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"JSON 읽기 실패: {p.name} ({e})")
        return None

def _parse_summary_payload(obj: dict) -> Dict[str, str]:
    """
    account.py 저장 포맷:
    {"comments": {...}, "data": [ { "0": { ... } } ]}
    또는 {"data": [ { ... } ]}
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
        return int(s) if s.isdigit() else 0
    return 0

def load_account_files_with_retry(
    summary_pattern: str = "summary_*.json",
    balance_pattern: str = "balance_*.json",
    max_wait_sec: int = 5
) -> Tuple[Dict[str, str], List[Dict], Optional[Path], Optional[Path]]:
    """
    최신 summary/balance 파일을 읽고 (summary_dict, balance_list, summary_path, balance_path) 반환.
    파일이 막 생성된 직후일 수 있으므로 최대 max_wait_sec 동안 재시도.
    """
    logger = logging.getLogger(__name__)
    deadline = time.time() + max_wait_sec
    summary_path, balance_path = None, None
    parsed_summary, parsed_balance = {}, []

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
        else:
            ok = False

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

    return parsed_summary, parsed_balance, summary_path, balance_path

def extract_cash_from_summary(summary_dict: Dict[str, str]) -> Dict[str, int]:
    """
    summary.json에서 현금 관련 키를 추출
    우선순위: prvs_rcdl_excc_amt > nxdy_excc_amt > ord_psbl_cash > dnca_tot_amt > tot_evlu_amt
    """
    keys_priority = [
        "prvs_rcdl_excc_amt",
        "nxdy_excc_amt",
        "ord_psbl_cash",
        "dnca_tot_amt",
        "tot_evlu_amt",
    ]
    out = {}
    for k in keys_priority:
        if k in summary_dict:
            out[k] = _to_int_krw(summary_dict.get(k))
    return out
