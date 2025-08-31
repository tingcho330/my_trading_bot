# src/scheduler.py

import schedule
import time
import subprocess
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import os

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Scheduler")

# --- KST 시간대 정의 ---
KST = ZoneInfo("Asia/Seoul")

# --- 실행할 스크립트 목록 (순서 중요) ---
# 1. health_check: API 서버 상태 사전 점검
# 2. screener -> news_collector -> gpt_analyzer: 매수 대상 종목 분석 및 선정
# 3. trader: 최종 매매 실행
# ※ account.py는 trader.py 내부에서 필요시 호출되므로, 여기서는 제외합니다.
PIPELINE_SCRIPTS = [
    "health_check.py",
    "screener.py",
    "news_collector.py",
    "gpt_analyzer.py",
    "trader.py"
]

# --- 실행 환경 변수 ---
MARKET = os.getenv("MARKET", "KOSPI")
SLOTS = os.getenv("SLOTS", "3")

# --- 재시도 관련 설정 ---
MAX_ATTEMPTS = 3 # 최대 재시도 횟수
INITIAL_BACKOFF_MINUTES = 2 # 초기 재시도 대기 시간 (분)

class PipelineRunFailedError(Exception):
    """파이프라인의 특정 스크립트 실행 실패 시 발생하는 예외"""
    pass

def is_market_open_day() -> bool:
    """오늘이 한국 주식 시장 개장일(월-금)인지 확인합니다."""
    # risk_manager.py의 함수와 동일한 로직
    return datetime.now(KST).weekday() < 5

def run_script(script_name: str) -> bool:
    """주어진 파이썬 스크립트를 실행하고 성공 여부를 반환합니다."""
    
    args = []
    if script_name == "screener.py":
        args = ["--market", MARKET]
    elif script_name == "gpt_analyzer.py":
        args = ["--market", MARKET, "--slots", SLOTS]
        
    command = ["python", f"/app/src/{script_name}"] + args
    
    logger.info(f"--- '{' '.join(command)}' 실행 시작 ---")
    
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding='utf-8'
        )
        # stdout의 마지막 10줄만 로깅하여 로그가 너무 길어지는 것을 방지
        stdout_tail = "\n".join(result.stdout.strip().splitlines()[-10:])
        logger.info(f"'{script_name}' 실행 성공.\n... (마지막 로그) ...\n{stdout_tail}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"'{script_name}' 실행 중 오류 발생 (Exit Code: {e.returncode}):")
        logger.error(f"--- STDERR ---\n{e.stderr}")
        return False
    except Exception as e:
        logger.critical(f"'{script_name}' 실행 중 예상치 못한 예외 발생: {e}")
        return False

def run_trading_pipeline():
    """안정성 기능이 추가된 전체 자동매매 파이프라인 실행 함수"""
    
    # --- 파이프라인 시작 전, 개장일인지 먼저 확인 ---
    if not is_market_open_day():
        logger.info("오늘은 주말(휴장일)이므로 자동매매 파이프라인을 실행하지 않습니다.")
        return

    kst_now = datetime.now(KST)
    logger.info(f"🚀 KST {kst_now.strftime('%Y-%m-%d %H:%M:%S')} - 자동매매 파이프라인을 시작합니다.")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info(f"--- 시도 {attempt}/{MAX_ATTEMPTS} ---")
            
            for script in PIPELINE_SCRIPTS:
                if not run_script(script):
                    # 스크립트 실행 실패 시 예외 발생
                    raise PipelineRunFailedError(f"'{script}' 실행에 실패했습니다.")
            
            # 모든 스크립트가 성공적으로 실행된 경우
            logger.info("✅ 파이프라인의 모든 단계가 성공적으로 완료되었습니다.")
            break  # 재시도 루프 종료

        except PipelineRunFailedError as e:
            logger.error(f"파이프라인 실행 중 오류 발생 (시도 {attempt}/{MAX_ATTEMPTS}): {e}")
            
            if attempt < MAX_ATTEMPTS:
                # Exponential Backoff 적용: 재시도 횟수가 늘어날수록 대기 시간 증가 (예: 2분 -> 4분 -> 8분)
                wait_time_minutes = INITIAL_BACKOFF_MINUTES * (2 ** (attempt - 1))
                logger.info(f"{wait_time_minutes}분 후 재시도합니다...")
                time.sleep(wait_time_minutes * 60)
            else:
                logger.critical("최대 재시도 횟수를 초과하여 파이프라인을 최종 중단합니다.")
                break # 루프 종료

    logger.info("파이프라인 한 사이클을 종료합니다.")


if __name__ == "__main__":
    # --- 스케줄 설정 ---
    # 매주 월요일~금요일, 한국 시간 기준 오전 8시에 파이프라인 실행
    schedule.every().monday.at("10:00", "Asia/Seoul").do(run_trading_pipeline)
    schedule.every().tuesday.at("10:00", "Asia/Seoul").do(run_trading_pipeline)
    schedule.every().wednesday.at("10:00", "Asia/Seoul").do(run_trading_pipeline)
    schedule.every().thursday.at("10:00", "Asia/Seoul").do(run_trading_pipeline)
    schedule.every().friday.at("10:00", "Asia/Seoul").do(run_trading_pipeline)

    logger.info("스케줄러가 시작되었습니다. 다음 작업 대기 중...")
    
    # # 즉시 테스트를 원할 경우 아래 코드의 주석을 해제하세요.
    # run_trading_pipeline()

    while True:
        schedule.run_pending()
        time.sleep(1)