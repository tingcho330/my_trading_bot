# src/utils.py
"""
공통 유틸리티 모듈
- 경로 및 KST 시간대 정의
- 통합 로깅 설정
- 설정 파일(config.json) 로드
- 파일 캐시(pickle) 관리
- 휴장일 확인
- output 디렉토리 내 최신 파일 탐색
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional, Any, List

# ───────────────── 경로 및 시간대 정의 ─────────────────

# KST 시간대 정의
KST = ZoneInfo("Asia/Seoul")

# 공통 경로 정의
APP_DIR = Path("/app")
SRC_DIR = APP_DIR / "src"
CONFIG_DIR = APP_DIR / "config"
OUTPUT_DIR = APP_DIR / "output"
CACHE_DIR = OUTPUT_DIR / "cache"
CONFIG_PATH = CONFIG_DIR / "config.json"


# ───────────────── 로깅 설정 ─────────────────

class KSTFormatter(logging.Formatter):
    """로그 타임스탬프를 KST로 변환하는 포매터"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, KST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec='milliseconds')

_is_logging_configured = False

def setup_logging():
    """모든 모듈에서 호출할 통합 로깅 설정 함수"""
    global _is_logging_configured
    if _is_logging_configured:
        return

    # 기본 로거 포맷 설정
    log_format = "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 핸들러 중복 추가 방지
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 스트림 핸들러 (콘솔 출력)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(KSTFormatter(log_format))
    root_logger.addHandler(stream_handler)
    
    _is_logging_configured = True
    logging.info("통합 로깅 시스템이 KST 기준으로 설정되었습니다.")


# ───────────────── 설정 및 파일 유틸리티 ─────────────────

def load_config() -> dict:
    """/app/config/config.json 설정 파일을 로드합니다."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"설정 파일({CONFIG_PATH})의 JSON 형식이 잘못되었습니다: {e}")
        raise
    except Exception as e:
        logging.error(f"설정 파일({CONFIG_PATH}) 로드 중 오류 발생: {e}")
        raise

def find_latest_file(pattern: str) -> Optional[Path]:
    """output 디렉토리에서 특정 패턴을 가진 가장 최신 파일을 찾습니다."""
    try:
        # output 디렉토리가 없으면 생성
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(OUTPUT_DIR.glob(pattern), key=os.path.getmtime)
        return files[-1] if files else None
    except (FileNotFoundError, IndexError):
        return None

# ───────────────── 캐시 관리 ─────────────────

def cache_path(name: str, date_str: str) -> Path:
    """캐시 파일의 경로를 반환합니다."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_{date_str}.pkl"

def cache_load(name: str, date_str: str) -> Optional[Any]:
    """파일에서 캐시를 로드합니다."""
    p = cache_path(name, date_str)
    if not p.is_file():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning(f"캐시 파일 로드 실패({p.name}): {e}")
        return None

def cache_save(name: str, date_str: str, obj: Any):
    """객체를 파일에 캐시로 저장합니다."""
    p = cache_path(name, date_str)
    try:
        with open(p, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        logging.warning(f"캐시 파일 저장 실패({p.name}): {e}")

# ───────────────── 시장 개장일 확인 ─────────────────
_kr_holidays = None

def is_market_open_day() -> bool:
    """오늘(KST)이 한국 주식 시장 개장일(월-금, 공휴일 제외)인지 확인합니다."""
    global _kr_holidays
    today = datetime.now(KST)

    if today.weekday() >= 5:  # 토요일(5), 일요일(6)
        return False

    if _kr_holidays is None:
        try:
            import holidays
            _kr_holidays = holidays.country_holidays("KR", years=today.year)
        except ImportError:
            logging.warning("'holidays' 라이브러리가 설치되지 않아 공휴일 확인을 건너뜁니다.")
            return True # 라이브러리가 없으면 평일 여부만 판단
        except Exception as e:
            logging.error(f"holidays 라이브러리 오류: {e}")
            return True # 오류 발생 시 개장일로 간주

    if today.date() in _kr_holidays:
        return False

    return True
