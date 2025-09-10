# src/scheduler.py
import schedule
import time
import subprocess
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
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
    "news_collector.py",
    "gpt_analyzer.py",
    "trader.py",
    "reviewer.py",
    "cleanup_output.py",
]

MARKET = os.getenv("MARKET", "KOSPI")
SLOTS = os.getenv("SLOTS", "3")

MAX_ATTEMPTS = int(os.getenv("SCHED_MAX_ATTEMPTS", "3"))
INITIAL_BACKOFF_MINUTES = int(os.getenv("SCHED_INITIAL_BACKOFF_MINUTES", "2"))

# 서브프로세스 타임아웃/슬로우 경고 임계치(초)
SCRIPT_TIMEOUT_SEC = int(os.getenv("SCRIPT_TIMEOUT_SEC", "600"))
SLOW_STEP_SEC = int(os.getenv("SLOW_STEP_SEC", "90"))

# 스팸 방지 노티 쿨다운 (키별 마지막 전송시각)
_last_sent: Dict[str, float] = {}

def _notify(msg: str, key: str, cooldown_sec: int = 60):
    """디스코드 알림(쿨다운 + 1회 재시도). 전체 파이프라인 목표: 시작/요약/종료 2~3회만 송신."""
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
    """로그 텍스트의 꼬리 n줄만 반환(디스코드 제한 방지용, 내부 요약용)"""
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
            _notify("️ 스케줄 등록 완료:\n" + "\n".join(lines[:10]), key="startup_jobs", cooldown_sec=10)
    except Exception:
        pass

# ───────────────── 서브프로세스 실행(알림 없음·로그만) ─────────────────
def run_script(script_name: str, run_id: str) -> Tuple[bool, bool, float]:
    """주어진 파이썬 스크립트를 실행하고 (성공여부, 경고발생여부, 소요시간초)를 반환.
    - 알림은 보내지 않고, 로그에만 기록한다.
    - '느린 실행(>SLOW_STEP_SEC)'은 경고로 취급(warned=True)."""
    args: List[str] = []
    if script_name == "screener.py":
        args = ["--market", MARKET]
    elif script_name == "gpt_analyzer.py":
        args = ["--market", MARKET, "--slots", SLOTS]

    command = ["python", f"/app/src/{script_name}"] + args
    cmd_str = " ".join(command)

    # 하위 프로세스 환경 구성
    child_env = dict(os.environ)
    child_env["RUN_ID"] = os.environ.get("RUN_ID", run_id)
    child_env["RUN_STARTED_AT"] = os.environ.get("RUN_STARTED_AT", str(time.time()))
    child_env.setdefault("MARKET", MARKET)
    child_env.setdefault("SLOTS", SLOTS)

    logger.info(f"[{run_id}] ▶ STEP START: {script_name} | cmd='{cmd_str}'")
    t0 = time.perf_counter()
    warned = False
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            timeout=SCRIPT_TIMEOUT_SEC,
            env=child_env,
        )
        dur = time.perf_counter() - t0
        stdout_tail = _tail(result.stdout, 12)
        logger.info(f"[{run_id}] ✅ STEP OK: {script_name} | {dur:.1f}s")
        logger.debug(f"[{run_id}] --- {script_name} tail ---\n{stdout_tail}")

        if dur > SLOW_STEP_SEC:
            warned = True
            logger.warning(f"[{run_id}] ⚠️ SLOW STEP: {script_name} ({dur:.1f}s > {SLOW_STEP_SEC}s)")

        return True, warned, dur

    except subprocess.TimeoutExpired:
        dur = time.perf_counter() - t0
        logger.error(f"[{run_id}] ❌ STEP TIMEOUT: {script_name} ({SCRIPT_TIMEOUT_SEC}s) | {dur:.1f}s 경과")
        return False, warned, dur

    except subprocess.CalledProcessError as e:
        dur = time.perf_counter() - t0
        stderr_tail = _tail(e.stderr, 80)
        logger.error(f"[{run_id}] ❌ STEP FAIL: {script_name} (exit={e.returncode}) | {dur:.1f}s")
        logger.error(f"[{run_id}] --- STDERR tail ---\n{stderr_tail}")
        return False, warned, dur

    except Exception as e:
        dur = time.perf_counter() - t0
        logger.critical(f"[{run_id}] ⛔ STEP EXCEPTION: {script_name} | {dur:.1f}s | {e}", exc_info=True)
        return False, warned, dur

# ───────────────── 스크리너 전용 잡(개별 실행) ─────────────────
def run_screener_job():
    """개별 스크리너 실행 (휴장일/중복실행 가드 포함) — 알림은 1~2회로 최소화"""
    try:
        if not is_market_open_day():
            msg = "오늘은 휴장일이므로 screener 실행을 건너뜁니다."
            logger.info(msg)
            _notify(msg=f"ℹ️ {msg}", key="screener_holiday", cooldown_sec=600)
            return

        if not _acquire_lock():
            logger.warning("다른 파이프라인(또는 스크리너)이 실행 중 → 이번 스크리너 트리거 무시")
            _notify("⛔ 다른 작업 실행 중 → 스크리너 트리거 무시", key="screener_lock_busy", cooldown_sec=60)
            return

        run_id = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
        os.environ["RUN_ID"] = run_id
        os.environ["RUN_STARTED_AT"] = str(time.time())

        start = f"[{run_id}] KST {datetime.now(KST):%Y-%m-%d %H:%M:%S} - 스크리너 단독 실행 시작 (MARKET={MARKET})"
        logger.info(start)
        _notify(start, key=f"{run_id}:screener_start", cooldown_sec=30)

        ok, warned, dur = run_script("screener.py", run_id)
        status = "✅ 완료" if ok else "❌ 실패"
        warn_tag = " (⚠️ slow)" if (ok and warned) else ""
        _notify(f"[{run_id}] 스크리너 {status}{warn_tag} | {dur:.1f}s", key=f"{run_id}:screener_end", cooldown_sec=30)

    finally:
        _release_lock()

# ───────────────── 파이프라인 ─────────────────
def run_trading_pipeline():
    """알림 집약(시작/요약/종료) + SUCCESS_WITH_WARNINGS 상태 도입한 전체 파이프라인"""
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

        # 시작 컨텍스트
        run_id = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
        os.environ["RUN_ID"] = run_id
        os.environ["RUN_STARTED_AT"] = str(time.time())
        os.environ["RUN_SUCCESS"] = ""   # legacy
        os.environ["RUN_STATUS"] = ""    # SUCCESS | SUCCESS_WITH_WARNINGS | FAIL
        os.environ["RUN_WARNINGS"] = "0" # 경고 건수

        start_msg = f"[{run_id}] KST {datetime.now(KST):%Y-%m-%d %H:%M:%S} - 자동매매 파이프라인 시작 (MARKET={MARKET}, SLOTS={SLOTS})"
        logger.info(start_msg)
        _notify(msg=start_msg, key=f"{run_id}:pipeline_start", cooldown_sec=30)

        pipeline_ok = True
        warn_count_total = 0
        attempts_used = 0
        last_error: Optional[str] = None
        last_failed_step: Optional[str] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts_used = attempt
            try:
                logger.info(f"[{run_id}] --- 시도 {attempt}/{MAX_ATTEMPTS} ---")
                for script in PIPELINE_SCRIPTS:
                    ok, warned, dur = run_script(script, run_id)
                    if warned:
                        warn_count_total += 1
                    # health_check 실패는 즉시 중단
                    if script == "health_check.py" and not ok:
                        pipeline_ok = False
                        last_failed_step = script
                        raise PipelineRunFailedError("헬스체크 실패로 파이프라인 중단")
                    if not ok:
                        pipeline_ok = False
                        last_failed_step = script
                        raise PipelineRunFailedError(f"'{script}' 실행 실패")
                # 모두 성공했다면 반복 종료
                break

            except PipelineRunFailedError as e:
                last_error = str(e)
                logger.error(f"[{run_id}] 파이프라인 실행 중 오류 발생 (시도 {attempt}/{MAX_ATTEMPTS}): {e}")
                if attempt < MAX_ATTEMPTS:
                    # 재시도는 자체적으로 '경고'로 집계
                    warn_count_total += 1
                    wait_time_minutes = INITIAL_BACKOFF_MINUTES * (2 ** (attempt - 1))
                    logger.info(f"[{run_id}] {wait_time_minutes}분 후 재시도합니다...")
                    time.sleep(wait_time_minutes * 60)
                else:
                    logger.critical(f"[{run_id}] 최대 재시도 횟수 초과. 파이프라인 최종 중단.")
                    break

        # 종료/상태 결정
        os.environ["RUN_ENDED_AT"] = str(time.time())
        status: str
        if pipeline_ok:
            status = "SUCCESS_WITH_WARNINGS" if warn_count_total > 0 else "SUCCESS"
            os.environ["RUN_SUCCESS"] = "true"
        else:
            status = "FAIL"
            os.environ["RUN_SUCCESS"] = "false"
        os.environ["RUN_STATUS"] = status
        os.environ["RUN_WARNINGS"] = str(int(warn_count_total))

        # ── 요약(1회) ──
        elapsed = 0.0
        try:
            started_at = float(os.environ.get("RUN_STARTED_AT", "0") or 0.0)
            if started_at:
                elapsed = time.time() - started_at
        except Exception:
            pass

        # 상태 이모지
        status_emoji = "✅" if status == "SUCCESS" else ("⚠️" if status == "SUCCESS_WITH_WARNINGS" else "❌")
        summary_lines = [
            f"{status_emoji} 파이프라인 요약 (run_id={run_id})",
            f"- 상태: {status}",
            f"- 경고 건수: {warn_count_total}",
            f"- 시도/최대: {attempts_used}/{MAX_ATTEMPTS}",
            f"- 마지막 실패 단계: {last_failed_step or 'N/A'}",
            f"- 소요시간: {elapsed:.0f}s",
        ]
        if last_error and status == "FAIL":
            summary_lines.append(f"- 에러: {last_error[:300]}")
        _notify("\n".join(summary_lines), key=f"{run_id}:pipeline_summary", cooldown_sec=15)

        # ── 종료(1회) ──
        end_msg = f"[{run_id}] 파이프라인 사이클 종료 (status={status}, warnings={warn_count_total})"
        logger.info(end_msg)
        _notify(msg=end_msg, key=f"{run_id}:pipeline_end", cooldown_sec=30)

    finally:
        _release_lock()

# ───────────────── 메인 ─────────────────
def _register_jobs():
    # ⚠️ schedule은 tz 인자를 지원하지 않는다. 컨테이너/호스트 TZ를 KST로 설정해 사용.

    # 평일 09:05 → 장 시작 전 스크리너만 실행 (휴장일 검사 + 락)
    schedule.every().monday.at("10:05").do(run_screener_job)
    schedule.every().tuesday.at("10:05").do(run_screener_job)
    schedule.every().wednesday.at("10:05").do(run_screener_job)
    schedule.every().thursday.at("10:05").do(run_screener_job)
    schedule.every().friday.at("10:05").do(run_screener_job)

    # 평일 10:40 → 전체 파이프라인 실행
    schedule.every().monday.at("13:40").do(run_trading_pipeline)
    schedule.every().tuesday.at("13:40").do(run_trading_pipeline)
    schedule.every().wednesday.at("13:40").do(run_trading_pipeline)
    schedule.every().thursday.at("13:40").do(run_trading_pipeline)
    schedule.every().friday.at("13:40").do(run_trading_pipeline)

if __name__ == "__main__":
    _register_jobs()
    _list_jobs()
    _startup_banner()

    logger.info("스케줄러가 시작되었습니다. 다음 작업 대기 중...")
    #run_trading_pipeline()  # 즉시 테스트용

    while True:
        schedule.run_pending()
        time.sleep(1)
