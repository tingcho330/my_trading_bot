# src/scheduler.py
import schedule
import time
import subprocess
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict
from pathlib import Path

# 공통 유틸리티
from utils import setup_logging, is_market_open_day, KST

# 디스코드 노티파이어 연동
from notifier import (
    DiscordLogHandler,
    WEBHOOK_URL,
    is_valid_webhook,
    send_discord_message,
)

# ───────────────── 로깅 초기화 ─────────────────
setup_logging()
logger = logging.getLogger("Scheduler")

# 루트 로거에 디스코드 에러 핸들러 장착(중복 방지)
root_logger = logging.getLogger()
if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
    if not any(isinstance(h, DiscordLogHandler) for h in root_logger.handlers):
        root_logger.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

# ───────────────── 파이프라인 설정 ─────────────────
PIPELINE_SCRIPTS: List[str] = [
    "health_check.py",   # 실패 시 즉시 중단
    "screener.py",
    "news_collector.py",
    "gpt_analyzer.py",
    "trader.py",
]

MARKET = os.getenv("MARKET", "KOSPI")
SLOTS = os.getenv("SLOTS", "3")

MAX_ATTEMPTS = 3
INITIAL_BACKOFF_MINUTES = 2

# 서브프로세스 타임아웃/슬로우 경고 임계치(초)
SCRIPT_TIMEOUT_SEC = int(os.getenv("SCRIPT_TIMEOUT_SEC", "600"))
SLOW_STEP_SEC = int(os.getenv("SLOW_STEP_SEC", "90"))

# 스팸 방지 노티 쿨다운
_last_sent: Dict[str, float] = {}

def _notify(msg: str, key: str, cooldown_sec: int = 60):
    """디스코드 알림(쿨다운 + 1회 재시도)"""
    try:
        if not (WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL)):
            return
        now = time.time()
        if key not in _last_sent or now - _last_sent[key] >= cooldown_sec:
            _last_sent[key] = now
            try:
                send_discord_message(content=msg)
            except Exception:
                time.sleep(1.5)
                send_discord_message(content=msg)
    except Exception:
        pass

class PipelineRunFailedError(Exception):
    """파이프라인의 특정 스크립트 실행 실패 시 발생하는 예외"""
    pass

def _tail(text: str, n: int = 12) -> str:
    """로그 텍스트의 꼬리 n줄만 반환(디스코드 제한 방지용)"""
    if not text:
        return ""
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])

# ───────────────── 중복 실행 방지 락 ─────────────────
LOCK_PATH = Path("/tmp/trading_pipeline.lock")

def _acquire_lock() -> bool:
    try:
        if LOCK_PATH.exists():
            return False
        LOCK_PATH.write_text(str(os.getpid()))
        return True
    except Exception:
        return False

def _release_lock():
    try:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    except Exception:
        pass

# ───────────────── 유틸: 잡 리스트 출력 ─────────────────
def _list_jobs():
    try:
        local_tz = datetime.now().astimezone().tzinfo  # OS 로컬 tz
        for j in schedule.get_jobs():
            nr = j.next_run
            if nr:
                # schedule의 next_run은 naive 로컬 시간이다 → 로컬 tz 부여 후 변환
                nr_local = nr.replace(tzinfo=local_tz)
                nr_kst = nr_local.astimezone(KST)
                nr_utc = nr_local.astimezone(timezone.utc)
                logger.info(f"[JOB] {j} | next_run local={nr_local} | KST={nr_kst} | UTC={nr_utc}")
            else:
                logger.info(f"[JOB] {j} | next_run=None")
    except Exception:
        pass

def _startup_banner():
    try:
        local_tz = datetime.now().astimezone().tzinfo
        lines = []
        for j in schedule.get_jobs():
            nr = j.next_run
            if nr:
                nr_kst = nr.replace(tzinfo=local_tz).astimezone(KST)
                lines.append(f"- {j} → next_run KST {nr_kst}")
        if lines:
            _notify("🗓️ 스케줄 등록 완료:\n" + "\n".join(lines[:10]), key="startup_jobs", cooldown_sec=10)
    except Exception:
        pass

# ───────────────── 서브프로세스 실행 ─────────────────
def run_script(script_name: str) -> bool:
    """주어진 파이썬 스크립트를 실행하고 성공 여부를 반환합니다."""
    args = []
    if script_name == "screener.py":
        args = ["--market", MARKET]
    elif script_name == "gpt_analyzer.py":
        args = ["--market", MARKET, "--slots", SLOTS]

    command = ["python", f"/app/src/{script_name}"] + args
    cmd_str = " ".join(command)
    logger.info(f"--- '{cmd_str}' 실행 시작 ---")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            timeout=SCRIPT_TIMEOUT_SEC,
        )
        dur = time.perf_counter() - t0
        stdout_tail = _tail(result.stdout, 12)
        logger.info(f"'{script_name}' 실행 성공.\n... (마지막 로그) ...\n{stdout_tail}")
        logger.info(f"{script_name} duration: {dur:.1f}s")
        if dur > SLOW_STEP_SEC:
            logger.warning(f"{script_name} 실행이 느립니다({dur:.1f}s > {SLOW_STEP_SEC}s)")
            _notify(
                msg=f"🐢 **{script_name}** 느린 실행 {dur:.1f}s",
                key=f"slow:{script_name}",
                cooldown_sec=120,
            )

        _notify(
            msg=f"✅ **{script_name}** 완료\n```tail\n{stdout_tail[:1600]}\n```",
            key=f"done:{script_name}",
            cooldown_sec=30,
        )
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"'{script_name}' 실행 타임아웃({SCRIPT_TIMEOUT_SEC}s). 강제 실패 처리.")
        _notify(
            msg=f"⏱️ **{script_name}** 타임아웃({SCRIPT_TIMEOUT_SEC}s)",
            key=f"timeout:{script_name}",
            cooldown_sec=30,
        )
        return False

    except subprocess.CalledProcessError as e:
        dur = time.perf_counter() - t0
        stderr_tail = _tail(e.stderr, 80)  # 실패 시 tail 더 길게
        logger.error(f"'{script_name}' 실행 중 오류 발생 (Exit Code: {e.returncode}):")
        logger.error(f"--- STDERR ---\n{stderr_tail}")
        logger.info(f"{script_name} duration(before fail): {dur:.1f}s")

        _notify(
            msg=f"❌ **{script_name}** 실패 (exit={e.returncode})\n```stderr\n{stderr_tail[:1600]}\n```",
            key=f"fail:{script_name}",
            cooldown_sec=30,
        )
        return False

    except Exception as e:
        dur = time.perf_counter() - t0
        logger.critical(f"'{script_name}' 실행 중 예상치 못한 예외 발생: {e}", exc_info=True)
        logger.info(f"{script_name} duration(before exception): {dur:.1f}s")
        _notify(
            msg=f"⛔ **{script_name}** 예외 발생\n```\n{str(e)[:1800]}\n```",
            key=f"except:{script_name}",
            cooldown_sec=30,
        )
        return False

# ───────────────── 파이프라인 ─────────────────
def run_trading_pipeline():
    """안정성 기능이 추가된 전체 자동매매 파이프라인 실행 함수"""
    if not _acquire_lock():
        logger.warning("이미 파이프라인이 실행 중이므로 트리거를 무시합니다.")
        _notify("⛔ 다른 파이프라인 실행 중 → 이번 트리거 무시", key="lock_busy", cooldown_sec=30)
        return

    try:
        if not is_market_open_day():
            msg = "오늘은 휴장일이므로 자동매매 파이프라인을 실행하지 않습니다."
            logger.info(msg)
            _notify(msg=f"ℹ️ {msg}", key="holiday", cooldown_sec=600)
            return

        kst_now = datetime.now(KST)
        start_msg = f"🚀 KST {kst_now.strftime('%Y-%m-%d %H:%M:%S')} - 자동매매 파이프라인 시작 (MARKET={MARKET}, SLOTS={SLOTS})"
        logger.info(start_msg)
        _notify(msg=start_msg, key="pipeline_start", cooldown_sec=30)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                logger.info(f"--- 시도 {attempt}/{MAX_ATTEMPTS} ---")
                _notify(msg=f"🔁 파이프라인 시도 {attempt}/{MAX_ATTEMPTS} 시작", key=f"attempt:{attempt}", cooldown_sec=30)

                for script in PIPELINE_SCRIPTS:
                    ok = run_script(script)
                    # health_check 실패는 즉시 중단
                    if script == "health_check.py" and not ok:
                        raise PipelineRunFailedError("헬스체크 실패로 파이프라인 중단")
                    if not ok:
                        raise PipelineRunFailedError(f"'{script}' 실행에 실패했습니다.")

                success_msg = "✅ 파이프라인의 모든 단계가 성공적으로 완료되었습니다."
                logger.info(success_msg)
                _notify(msg=success_msg, key="pipeline_done", cooldown_sec=30)
                break

            except PipelineRunFailedError as e:
                logger.error(f"파이프라인 실행 중 오류 발생 (시도 {attempt}/{MAX_ATTEMPTS}): {e}")
                if attempt < MAX_ATTEMPTS:
                    wait_time_minutes = INITIAL_BACKOFF_MINUTES * (2 ** (attempt - 1))
                    info_msg = f"{wait_time_minutes}분 후 재시도합니다..."
                    logger.info(info_msg)
                    _notify(msg=f"⚠️ 재시도 대기: {info_msg}", key=f"retry_wait:{attempt}", cooldown_sec=30)
                    time.sleep(wait_time_minutes * 60)
                else:
                    critical_msg = "최대 재시도 횟수를 초과하여 파이프라인을 최종 중단합니다."
                    logger.critical(critical_msg)
                    _notify(msg=f"🛑 {critical_msg}", key="pipeline_stop", cooldown_sec=30)
                    break

        logger.info("파이프라인 한 사이클을 종료합니다.")
        _notify(msg="🔚 파이프라인 사이클 종료", key="pipeline_end", cooldown_sec=30)

    finally:
        _release_lock()

# ───────────────── 메인 ─────────────────
def _register_jobs():
    # ⚠️ schedule은 tz 인자를 지원하지 않는다. 컨테이너/호스트 TZ를 KST로 설정해 사용.
    schedule.every().monday.at("13:46").do(run_trading_pipeline)
    schedule.every().tuesday.at("10:00").do(run_trading_pipeline)
    schedule.every().wednesday.at("10:00").do(run_trading_pipeline)
    schedule.every().thursday.at("10:00").do(run_trading_pipeline)
    schedule.every().friday.at("10:00").do(run_trading_pipeline)

if __name__ == "__main__":
    _register_jobs()
    _list_jobs()
    _startup_banner()

    logger.info("스케줄러가 시작되었습니다. 다음 작업 대기 중...")
    # run_trading_pipeline()  # 즉시 테스트용

    while True:
        schedule.run_pending()
        time.sleep(1)
