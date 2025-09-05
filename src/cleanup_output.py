# /app/src/cleanup_output.py
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time
import re
from datetime import datetime, timezone

# 공통 로깅(KST 포맷)
from utils import setup_logging, OUTPUT_DIR, KST

setup_logging()
logger = logging.getLogger("cleanup")

# ───────────────── 설정 ─────────────────
# 보존 회차(최근 N회): 환경변수로 오버라이드 가능
RETAIN_CYCLES = int(os.getenv("RETAIN_CYCLES", "10"))
# 보존 개수(패턴별) – 파일명이 회차 개념이 약한 것들은 mtime 기준으로 정리
DEFAULT_KEEP = RETAIN_CYCLES

# 회차 단위로 함께 정리해야 하는 그룹(후보/랭킹, 로그)
GROUPED_PATTERNS_BY_CYCLE: Dict[str, List[str]] = {
    # 시장별 회차 보존
    "SCREENER": [
        "screener_candidates_*.json",
        "screener_candidates_full_*_*.json",
        "screener_rank_*.json",
        "screener_rank_full_*_*.json",
    ],
    # 로그(시장 구분 없이 회차 보존)
    "LOGS": [
        "pipeline_*.log",
        "step_*.log",
    ],
}

# 그 외 개별 보존(파일 수 기준)
PATTERNS_TO_CLEAN = {
    "collected_news_*.json": DEFAULT_KEEP,
    "gpt_trades_*.json": DEFAULT_KEEP,
    "summary_*.json": max(DEFAULT_KEEP * 2, 20),   # 계좌 스냅샷은 조금 더 여유 보존
    "balance_*.json": max(DEFAULT_KEEP * 2, 20),
    "review_log.json": 1,                          # 단일 파일(최신만 유지)
    "completed_trades_analysis.csv": 1,            # 최신만 유지
}

# 절대 보존(삭제 금지) 파일/경로
PROTECT_FILES = {
    "trading_log.db",
    "cooldown.json",
}
PROTECT_DIRS = {
    "cache",  # /output/cache
}

# 드라이런(미삭제) 모드: 1이면 목록/로그만 남기고 실제 삭제 안함
DRY_RUN = os.getenv("CLEANUP_DRY_RUN", "0") == "1"

# ───────────────── 파일명 메타 파서 ─────────────────
DATE_PATTERNS = [
    re.compile(r"(?P<date>\d{8})[-_.]?(?P<hms>\d{6})?"),  # 20250101 또는 20250101-093000
]
MARKET_PATTERN = re.compile(r"(KOSPI|KOSDAQ|KONEX|NYSE|NASDAQ|AMEX|SPX|NIKKEI|HKEX)", re.IGNORECASE)

def _extract_date_hms(name: str) -> Optional[str]:
    for pat in DATE_PATTERNS:
        m = pat.search(name)
        if m:
            d = m.group("date")
            h = m.group("hms")
            return f"{d}-{h}" if h else d
    return None

def _extract_market(name: str) -> Optional[str]:
    m = MARKET_PATTERN.search(name)
    return m.group(0).upper() if m else None

def _gather(pattern: str) -> List[Path]:
    try:
        return list(OUTPUT_DIR.glob(pattern))
    except Exception:
        return []

def _mtime_kst(p: Path) -> float:
    try:
        # 파일 mtime(UTC) → KST epoch(정렬용) 크게 의미는 없지만 일관성
        ts = p.stat().st_mtime
        return ts
    except Exception:
        return 0.0

def _is_protected(p: Path) -> bool:
    if p.name in PROTECT_FILES:
        return True
    for d in PROTECT_DIRS:
        if (OUTPUT_DIR / d) in p.parents:
            return True
    return False

def _delete_files(files: List[Path]) -> Tuple[int, int]:
    deleted = 0
    bytes_freed = 0
    for f in files:
        try:
            if _is_protected(f):
                logger.debug(f"Skip(protected): {f.name}")
                continue
            sz = f.stat().st_size if f.exists() else 0
            if DRY_RUN:
                logger.info(f"[DRY] delete {f.name} ({sz} bytes)")
                continue
            f.unlink(missing_ok=True)
            deleted += 1
            bytes_freed += sz
            logger.info(f"deleted: {f.name} ({sz} bytes)")
        except Exception as e:
            logger.warning(f"delete fail: {f.name} ({e})")
    return deleted, bytes_freed

# ───────────────── 회차-기반 정리 로직 ─────────────────
def _build_cycle_key_for_screener(p: Path) -> str:
    """
    시장별 회차 키: MARKET 또는 'GEN' (없으면 일반)
    키 형식: f"{market}:{date_hms}"
    """
    name = p.name
    market = _extract_market(name) or "GEN"
    date_hms = _extract_date_hms(name)
    if not date_hms:
        # 날짜가 없으면 mtime의 KST 날짜를 사용
        kst_dt = datetime.fromtimestamp(_mtime_kst(p), tz=timezone.utc).astimezone(KST)
        date_hms = kst_dt.strftime("%Y%m%d-%H%M%S")
    return f"{market}:{date_hms}"

def _build_cycle_key_for_logs(p: Path) -> str:
    """
    로그 회차 키: 시장 구분 없이 날짜(옵션 시각)
    키 형식: f"LOG:{date_hms}"
    """
    date_hms = _extract_date_hms(p.name)
    if not date_hms:
        kst_dt = datetime.fromtimestamp(_mtime_kst(p), tz=timezone.utc).astimezone(KST)
        date_hms = kst_dt.strftime("%Y%m%d-%H%M%S")
    return f"LOG:{date_hms}"

def _group_and_trim_cycles(patterns: List[str], build_key_fn, keep_n: int) -> Tuple[int, int]:
    """
    패턴 묶음에 속한 파일들을 회차 키로 그룹핑하고, 최신 keep_n 회차만 유지.
    반환: (삭제파일수, 해제용량 바이트)
    """
    # 1) 후보 수집
    files: List[Path] = []
    for pat in patterns:
        files.extend(_gather(pat))
    if not files:
        return (0, 0)

    # 2) 회차 키 부여 및 대표 타임스탬프(최신 파일 기준) 계산
    groups: Dict[str, Dict[str, any]] = {}
    for f in files:
        key = build_key_fn(f)
        info = groups.setdefault(key, {"files": [], "ts": 0.0})
        info["files"].append(f)
        info["ts"] = max(info["ts"], _mtime_kst(f))

    # 3) 회차 정렬(최신 우선) 후 보존/삭제 분리
    ordered = sorted(groups.items(), key=lambda kv: kv[1]["ts"], reverse=True)
    keep_keys = set(k for k, _ in ordered[:max(keep_n, 0)])
    delete_files: List[Path] = []
    for k, v in ordered:
        if k in keep_keys:
            continue
        delete_files.extend(v["files"])

    # 4) 삭제 실행
    if not delete_files:
        logger.info(f"[cycle-keep] patterns={patterns} | groups={len(groups)} keep={len(keep_keys)} (no deletion)")
        return (0, 0)

    logger.info(
        f"[cycle-trim] patterns={patterns} | total_groups={len(groups)} "
        f"→ keep={len(keep_keys)} delete_groups={len(groups)-len(keep_keys)} "
        f"delete_files={len(delete_files)}"
    )
    return _delete_files(delete_files)

# ───────────────── 개별 패턴 정리 ─────────────────
def _trim_by_count(pattern: str, keep_n: int) -> Tuple[int, int]:
    files = sorted(_gather(pattern), key=_mtime_kst, reverse=True)
    if not files:
        return (0, 0)
    to_delete = files[keep_n:] if keep_n > 0 else files
    if not to_delete:
        logger.info(f"[keep] {pattern}: {len(files)} files (no deletion)")
        return (0, 0)
    logger.info(f"[pattern] {pattern}: total={len(files)} → delete={len(to_delete)} keep={keep_n}")
    return _delete_files(to_delete)

# ───────────────── 메인 ─────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== cleanup start (OUTPUT_DIR={OUTPUT_DIR}, retain={RETAIN_CYCLES}, dry={DRY_RUN}) ===")
    total_deleted = 0
    total_bytes = 0

    # 1) 회차 기반 그룹 정리 — 스크리너(시장별)
    d, b = _group_and_trim_cycles(
        GROUPED_PATTERNS_BY_CYCLE["SCREENER"],
        build_key_fn=_build_cycle_key_for_screener,
        keep_n=RETAIN_CYCLES,
    )
    total_deleted += d
    total_bytes += b

    # 2) 회차 기반 그룹 정리 — 로그(시장 구분 없음)
    d, b = _group_and_trim_cycles(
        GROUPED_PATTERNS_BY_CYCLE["LOGS"],
        build_key_fn=_build_cycle_key_for_logs,
        keep_n=RETAIN_CYCLES,
    )
    total_deleted += d
    total_bytes += b

    # 3) 나머지 개별 패턴 정리 (mtime 기준)
    for pattern, keep_n in PATTERNS_TO_CLEAN.items():
        d, b = _trim_by_count(pattern, keep_n)
        total_deleted += d
        total_bytes += b

    # 4) OUTPUT_DIR 루트의 temp/old 정리
    extras = [
        p for p in OUTPUT_DIR.iterdir()
        if p.is_file() and p.suffix in {".tmp", ".old"} and not _is_protected(p)
    ]
    if extras:
        logger.info(f"[extras] delete temp/old files: {len(extras)}")
        d, b = _delete_files(extras)
        total_deleted += d
        total_bytes += b

    logger.info(f"=== cleanup done: deleted={total_deleted} files, freed={total_bytes/1024:.1f} KB ===")

if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    finally:
        logger.info(f"(took {time.time()-t0:.2f}s)")
