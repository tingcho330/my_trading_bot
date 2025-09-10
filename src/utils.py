# src/utils.py
import os
import json
import time as pytime   # ← 모듈 time 충돌 방지
import logging
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Iterable, Union
from datetime import datetime, time as dt_time  # ← datetime.time 별칭
from zoneinfo import ZoneInfo
import threading
import re
import pandas as pd


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
    "get_cfg",
    "compute_52w_position",
    "compute_kki_metrics",
    "count_consecutive_up",
    "is_newly_listed",
    "in_time_windows",
    "is_market_open_day",
    "find_latest_file",
    "cache_save",
    "cache_load",
    "load_account_files_with_retry",
    "extract_cash_from_summary",
    "_to_int_krw",  # ← 공개 심볼 추가
    # 추가: 계좌 스냅샷 캐시 프로바이더 & 호가 유틸
    "get_account_snapshot_cached",
    "get_tick_size",
    "round_to_tick",
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
# 로깅 설정 (KST)
# ────────────────────────────────
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level)

    logging.Formatter.converter = lambda *args: datetime.now(KST).timetuple()
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    else:
        for h in root.handlers:
            h.setFormatter(fmt)
            h.setLevel(level)

    # noisy 네트워크 로깅 차단
    for noisy in ("httpx", "httpcore", "urllib3"):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.WARNING)
        lg.propagate = False

    root.debug(
        "logging initialized (KST). OUTPUT_DIR=%s, CACHE_DIR=%s, CONFIG_PATH=%s",
        str(OUTPUT_DIR),
        str(CACHE_DIR),
        str(CONFIG_PATH),
    )
    return root

# ────────────────────────────────
# 설정 파일 및 데이터 분석 유틸리티
# ────────────────────────────────
def load_config(path: Path = CONFIG_PATH) -> Optional[Dict[str, Any]]:
    logger = logging.getLogger(__name__)
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

def get_cfg(path: Path = CONFIG_PATH) -> dict:
    """설정 로드 + 기본값 보정 + 유효성 검사"""
    logger = logging.getLogger(__name__)
    cfg = load_config(path)
    if cfg is None:
        logger.warning("get_cfg: 설정 로드에 실패하여 기본값으로 진행합니다.")
        cfg = {}  # Fallback to an empty dict
        
    s = cfg.get("screener_params", {})
    s.setdefault("max_market_cap", int(1e13))
    s.setdefault("vol_kki_weight", 0.10)
    s.setdefault("pos_52w_weight", 0.05)
    s.setdefault("exclude_newly_listed_days", 60)
    s.setdefault("exclude_consecutive_up_days", 3)
    cfg.setdefault("trading_guards", {"min_cash_to_trade": 120000, "auto_shrink_slots": True})
    cfg.setdefault("prompting", {"core_questions": True})
    cfg.setdefault("rotation", {"enabled": True, "delta_score_min": 0.10})
    return cfg

def compute_52w_position(series: pd.Series) -> float:
    """52주 범위에서 현재 위치(0~1). 데이터 결측/분모 0은 0 처리"""
    if series is None or series.empty:
        return 0.0
    high_52 = series[-252:].max()
    low_52 = series[-252:].min()
    last = series.iloc[-1]
    rng = max(1e-9, (high_52 - low_52))
    pos = (last - low_52) / rng
    return float(max(0.0, min(1.0, pos)))

def compute_kki_metrics(df: pd.DataFrame) -> float:
    """
    '끼' 점수(0~1): 60D 수익률 표준편차 정규화 + 1Y 상한가 빈도
    df: 반드시 'close','open','high','low' 포함, 일자 오름차순
    """
    if df is None or df.empty:
        return 0.0
    # ← 컬럼 케이스 보정(호출부 무관하게 동작)
    df = df.rename(columns=str.lower)
    if any(c not in df.columns for c in ["close", "open", "high", "low"]):
        return 0.0

    rets = df["close"].pct_change()
    vol = rets.rolling(60).std().iloc[-1]
    # z-score를 간단히 0~3 범위에 맵핑(로버스트 클립)
    if pd.isna(vol):
        vol_norm = 0.0
    else:
        vol_norm = min(3.0, max(0.0, (vol / (rets.std() + 1e-9)))) / 3.0
    # 1Y 상한가 빈도(보수적 근사: 1.29배)
    year = df.tail(252)
    prev_close = year["close"].shift(1)
    limit_hits = ((year["high"] >= prev_close * 1.29).fillna(False)).sum()
    limit_freq = min(1.0, limit_hits / 252.0)
    return float(max(0.0, min(1.0, 0.7 * vol_norm + 0.3 * limit_freq)))

def count_consecutive_up(df: pd.DataFrame, window: int = 3) -> int:
    """연속 양봉 수(마지막 날 기준)"""
    if df is None or df.empty:
        return 0
    up = (df["close"] > df["open"]).astype(int)
    cnt = 0
    for v in reversed(up.tolist()):
        if v == 1:
            cnt += 1
        else:
            break
    return cnt

def is_newly_listed(listing_date: datetime, today: datetime, limit_days: int) -> bool:
    if listing_date is None:
        return False
    return (today.date() - listing_date.date()).days < limit_days

#
# ────────────────────────────────
# 시간창 포함 여부 체크 (형식 검증 + 자정 교차 구간 지원)
# ────────────────────────────────
_WINDOW_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")

def _parse_hhmm(hh: str, mm: str) -> dt_time:
    """'HH','MM' 숫자 문자열을 datetime.time으로 안전 변환(범위 보정 포함)."""
    h = max(0, min(23, int(hh)))
    m = max(0, min(59, int(mm)))
    return dt_time(h, m)

def in_time_windows(
    now: datetime,
    windows: Optional[List[str]] = None,
    tz: Optional[ZoneInfo] = None,
) -> bool:
    """
    주어진 시각(now)이 'HH:MM-HH:MM' 형태의 구간 리스트 중 하나라도 포함되면 True
    - 잘못된 포맷은 무시하고 경고 로그를 남깁니다
    - 시작 > 종료 인 구간은 '자정 교차(cross-midnight)' 구간으로 해석합니다
      예) 23:50-00:10 → 23:50~24:00 또는 00:00~00:10
    - windows가 비어있거나 None이면 '제한 없음'으로 간주하여 True를 반환합니다
    """
    logger = logging.getLogger(__name__)

    # 타임존 보정: naive → 지정 tz로 로컬라이즈, aware → tz로 변환
    if tz is not None:
        if now.tzinfo is None:
            now = now.replace(tzinfo=tz)
        else:
            now = now.astimezone(tz)

    # 제한 구간이 없으면 통과 (기본 허용)
    if not windows:
        return True

    hm = now.time()
    for raw in windows:
        if not isinstance(raw, str):
            logger.warning("in_time_windows: 잘못된 항목(문자열 아님) 무시: %r", raw)
            continue
        m = _WINDOW_RE.match(raw)
        if not m:
            logger.warning("in_time_windows: 포맷 불일치 'HH:MM-HH:MM' 무시: %s", raw)
            continue
        s = _parse_hhmm(m.group(1), m.group(2))
        e = _parse_hhmm(m.group(3), m.group(4))

        if s <= e:
            # 일반 구간: s <= hm <= e
            if s <= hm <= e:
                return True
        else:
            # 자정 교차 구간: s..24:00 또는 00:00..e
            if hm >= s or hm <= e:
                return True
    return False

# ────────────────────────────────
# 장 개장일 체크 (간단: 주말 제외)
# ────────────────────────────────
def is_market_open_day() -> bool:
    today = datetime.now(KST).date()
    return today.weekday() < 5

# ────────────────────────────────
# 내부: 파일명에서 날짜/시장/런ID 추출
# ────────────────────────────────
_date_patterns = [
    re.compile(r"(?P<date>\d{8})[._-]?(?P<hms>\d{6})?"),  # 20250904 or 20250904-134000
    re.compile(r"(?P<date>\d{8})"),                       # 20250904
]
_market_pattern = re.compile(r"(KOSPI|KOSDAQ|KONEX|NYSE|NASDAQ|AMEX|SPX|NIKKEI|HKEX)", re.IGNORECASE)

def _extract_meta_from_name(name: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"date": None, "hms": None, "market": None}
    for pat in _date_patterns:
        m = pat.search(name)
        if m:
            meta["date"] = m.group("date")
            if "hms" in m.groupdict():
                meta["hms"] = m.group("hms")
            break
    mm = _market_pattern.search(name)
    if mm:
        meta["market"] = mm.group(0).upper()
    return meta

def _score_file(p: Path, prefer_date: bool = True) -> Tuple[int, int, float]:
    """
    스코어: (date_int, hms_int, mtime)
    - 날짜가 없으면 0 취급
    - prefer_date=True면 날짜/시간 우선, 동률이면 mtime
    """
    name = p.name
    meta = _extract_meta_from_name(name)
    try:
        date_int = int(meta["date"]) if meta["date"] else 0
    except Exception:
        date_int = 0
    try:
        hms_int = int(meta["hms"]) if meta["hms"] else 0
    except Exception:
        hms_int = 0
    try:
        mtime = p.stat().st_mtime
    except Exception:
        mtime = 0.0

    # 날짜 우선 정렬: (date, hms, mtime)
    return (date_int if prefer_date else 0, hms_int if prefer_date else 0, mtime)

def _iter_globs(patterns: Union[str, Iterable[str]]) -> List[Path]:
    pats: List[str] = []
    if isinstance(patterns, str):
        pats = [patterns]
    else:
        pats = [str(x) for x in patterns]
    seen: Dict[str, Path] = {}
    for pat in pats:
        for p in OUTPUT_DIR.glob(pat):
            seen[str(p)] = p
    return list(seen.values())

# ────────────────────────────────
# 최신 파일 찾기 (다중 패턴/마켓 필터/날짜 우선)
# ────────────────────────────────
def find_latest_file(
    patterns: Union[str, Iterable[str]],
    *,
    market: Optional[str] = None,
    prefer_date_over_mtime: bool = True,
) -> Optional[Path]:
    """
    OUTPUT_DIR에서 patterns에 매칭되는 파일 중 '최신' 하나를 반환합니다.
    - patterns: 문자열 패턴 또는 패턴 리스트/튜플
    - market: "KOSPI" 등 필터(대소문자 무시). 파일명에 시장명이 포함된 경우에만 필터 적용.
    - prefer_date_over_mtime: 파일명 날짜 우선, 동일/부재시 mtime 기준.
    """
    logger = logging.getLogger(__name__)
    candidates = _iter_globs(patterns)
    if not candidates:
        return None

    mkt = (market or os.getenv("MARKET", "")).upper().strip()
    filtered: List[Tuple[Tuple[int, int, float], Path]] = []

    for p in candidates:
        meta = _extract_meta_from_name(p.name)
        # 마켓 필터: 파일명에 마켓 문자열이 있는 경우에만 필터링
        if mkt and meta.get("market") and meta["market"] != mkt:
            continue
        score = _score_file(p, prefer_date=prefer_date_over_mtime)
        filtered.append((score, p))

    if not filtered:
        logger.debug(
            "find_latest_file: 후보 없음 → fallback(mtime). market_filter=%s, patterns=%s",
            mkt or "NONE", patterns,
        )
        try:
            return max(candidates, key=lambda x: x.stat().st_mtime)
        except Exception:
            return None

    # 날짜/시간/mtime 우선 순위 정렬
    latest = max(filtered, key=lambda t: t[0])[1]
    return latest

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
    deadline = pytime.time() + max_wait_sec
    summary_path: Optional[Path] = None
    balance_path: Optional[Path] = None
    parsed_summary: Dict[str, str] = {}
    parsed_balance: List[Dict] = []

    while pytime.time() < deadline:
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
        pytime.sleep(0.5)

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
# Account Snapshot Provider (파일 mtime/락 기반 캐시)
# ────────────────────────────────
_SNAPSHOT_CACHE_LOCK = threading.Lock()
_SNAPSHOT_CACHE: Dict[str, Any] = {
    "ts": 0.0,                  # 캐시 생성 시각 (epoch)
    "summary_path": None,       # 마지막 사용 summary 파일 경로
    "balance_path": None,       # 마지막 사용 balance 파일 경로
    "summary_mtime": 0.0,       # summary 파일 mtime
    "balance_mtime": 0.0,       # balance 파일 mtime
    "summary": {},              # 파싱된 summary
    "balance": [],              # 파싱된 balance
}
_SNAPSHOT_LOCKFILE = Path(os.getenv("ACCOUNT_SNAPSHOT_LOCK", "/tmp/account_snapshot.lock"))
_SNAPSHOT_TTL_SEC = int(os.getenv("ACCOUNT_SNAPSHOT_TTL_SEC", "90"))  # 기본 90초
_SNAPSHOT_WAIT_ON_LOCK_SEC = int(os.getenv("ACCOUNT_SNAPSHOT_WAIT_SEC", "5"))  # 락이 있으면 최대 대기

def _files_unchanged(summary_path: Optional[Path], balance_path: Optional[Path],
                     cached_summary_mtime: float, cached_balance_mtime: float) -> bool:
    try:
        sm_ok = (summary_path is None) or (summary_path.exists() and abs(summary_path.stat().st_mtime - cached_summary_mtime) < 1e-6)
        bl_ok = (balance_path is None) or (balance_path.exists() and abs(balance_path.stat().st_mtime - cached_balance_mtime) < 1e-6)
        return sm_ok and bl_ok
    except Exception:
        return False

def _touch_lockfile() -> None:
    try:
        _SNAPSHOT_LOCKFILE.write_text(str(pytime.time()))
    except Exception:
        pass

def _lock_is_recent(max_age_sec: int = 10) -> bool:
    try:
        if not _SNAPSHOT_LOCKFILE.exists():
            return False
        age = pytime.time() - _SNAPSHOT_LOCKFILE.stat().st_mtime
        return age <= max_age_sec
    except Exception:
        return False

def get_account_snapshot_cached(
    summary_pattern: str = "summary_*.json",
    balance_pattern: str = "balance_*.json",
    ttl_sec: Optional[int] = None,
) -> Tuple[Dict[str, int], List[Dict], Optional[Path], Optional[Path]]:
    """
    요약/잔고 파일을 읽어 캐시로 제공.
    - 캐시 TTL(기본 90초) 내에서는 메모리 캐시 반환
    - 캐시가 있어도 파일 mtime 변경 시 즉시 재로딩
    - 다른 프로세스가 동시에 갱신 중이면 lockfile 존재 시 잠깐 대기 후 캐시 재확인
    반환: (summary_dict, balance_list, summary_path, balance_path)
    """
    logger = logging.getLogger(__name__)
    ttl = int(ttl_sec if ttl_sec is not None else _SNAPSHOT_TTL_SEC)

    # 1) 락 파일이 최신이라면 잠깐 대기(중복 IO 억제)
    wait_deadline = pytime.time() + _SNAPSHOT_WAIT_ON_LOCK_SEC
    while _lock_is_recent() and pytime.time() < wait_deadline:
        pytime.sleep(0.2)

    with _SNAPSHOT_CACHE_LOCK:
        now = pytime.time()
        # 캐시가 유효하면 그대로 반환
        if (now - _SNAPSHOT_CACHE["ts"]) <= ttl:
            # 파일 변경 없는지 확인
            sp = _SNAPSHOT_CACHE["summary_path"]
            bp = _SNAPSHOT_CACHE["balance_path"]
            if _files_unchanged(sp, bp, _SNAPSHOT_CACHE["summary_mtime"], _SNAPSHOT_CACHE["balance_mtime"]):
                return (
                    dict(_SNAPSHOT_CACHE["summary"]),
                    list(_SNAPSHOT_CACHE["balance"]),
                    sp,
                    bp,
                )

        # 2) 재로딩 (락 생성 후 로드)
        _touch_lockfile()
        summary_dict, balance_list, summary_path, balance_path = load_account_files_with_retry(
            summary_pattern=summary_pattern,
            balance_pattern=balance_pattern,
            max_wait_sec=5,
        )

        # 3) 캐시에 저장
        try:
            sm_mtime = summary_path.stat().st_mtime if summary_path and summary_path.exists() else 0.0
            bl_mtime = balance_path.stat().st_mtime if balance_path and balance_path.exists() else 0.0
        except Exception:
            sm_mtime = bl_mtime = 0.0

        _SNAPSHOT_CACHE.update({
            "ts": now,
            "summary_path": summary_path,
            "balance_path": balance_path,
            "summary_mtime": sm_mtime,
            "balance_mtime": bl_mtime,
            "summary": summary_dict,
            "balance": balance_list,
        })

        # 4) 반환
        return summary_dict, balance_list, summary_path, balance_path

# ────────────────────────────────
# 호가 단위 유틸 (표준화)
# ────────────────────────────────
def get_tick_size(price: float) -> int:
    """
    KRX 일반 호가단위 (원 기준, 단순화 버전)
    """
    try:
        p = float(price)
    except Exception:
        p = 0.0
    if p < 2000: return 1
    elif p < 5000: return 5
    elif p < 20000: return 10
    elif p < 50000: return 50
    elif p < 200000: return 100
    elif p < 500000: return 500
    else: return 1000

def round_to_tick(price: float, mode: str = "nearest") -> int:
    """
    호가단위에 맞춰 반올림.
      - mode='nearest' (기본): 가장 가까운 호가
      - mode='down'         : 아래 호가
      - mode='up'           : 위 호가
    """
    try:
        p = float(price)
    except Exception:
        return 0
    tick = get_tick_size(p)
    if tick <= 0:
        return int(round(p))
    if mode == "down":
        return int((p // tick) * tick)
    if mode == "up":
        return int(((p + tick - 1) // tick) * tick)
    # nearest
    return int(round(p / tick) * tick)

# ────────────────────────────────
# 모듈 로드 확인용 디버그 로그
# ────────────────────────────────
logging.getLogger(__name__).debug(
    "utils loaded. exports: %s",
    ", ".join(__all__),
)
