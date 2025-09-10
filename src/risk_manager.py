# src/risk_manager.py
import os
import json
import logging
import subprocess
import time as pytime
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, Tuple, Optional, List

# ── 공통 유틸/노티파이어 ────────────────────────────────────────────────
from utils import (
    KST,
    OUTPUT_DIR,
    extract_cash_from_summary,
    setup_logging,          # ← 공통 로깅 초기화
    in_time_windows,        # ← 시간창 판별 (고정)
    get_account_snapshot_cached,  # ← 읽기는 utils 캐시 사용
)
from notifier import (
    DiscordLogHandler,
    WEBHOOK_URL,
    is_valid_webhook,
    send_discord_message,
)

# ── 계산 전용 코어 모듈 사용 ───────────────────────────────────────────
from screener_core import (
    _compute_levels,          # 손절/목표가 계산 (ATR/스윙 기반, 퍼센트 백업)
    get_historical_prices,    # 과거 시세 조회 (pykrx 우선, fdr 백업)
    calculate_rsi,            # RSI 계산
)

# ───────────────── 로깅 기본 설정 ─────────────────
setup_logging()
logger = logging.getLogger("RiskManager")
logger.setLevel(logging.INFO)

# 루트 로거에 디스코드 에러 핸들러 장착(중복 방지)
_root = logging.getLogger()
if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

ACCOUNT_SCRIPT_PATH = "/app/src/account.py"
TRADER_SCRIPT_PATH = "/app/src/trader.py"  # [NEW] 파이프라인 트리거 대상

# ── 장중 정의(평일 09:00~15:30) ────────────────────────────────────────
MARKET_START = dt_time(9, 0)
MARKET_END   = dt_time(15, 30)

def is_market_hours(now: Optional[datetime] = None) -> bool:
    """평일 09:00~15:30 (KST) 에만 True"""
    if now is None:
        now = datetime.now(KST)
    if now.weekday() > 4:  # 0=월 ~ 4=금
        return False
    now_t = now.time()
    return MARKET_START <= now_t <= MARKET_END

def next_market_open_kst(now: Optional[datetime] = None) -> datetime:
    """다음 장 시작(평일 09:00) 시각 계산"""
    if now is None:
        now = datetime.now(KST)

    # 이미 장중이면 지금 반환
    if is_market_hours(now):
        return now

    # 오늘 09:00 기준
    candidate = now.replace(hour=9, minute=0, second=0, microsecond=0)

    # 오늘 장이 끝났으면 익일 09:00
    if now.time() >= MARKET_END:
        candidate = candidate + timedelta(days=1)

    # 주말 건너뛰기
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)

    return candidate

def sleep_until_kst(when_dt: datetime):
    """지정한 KST 시각까지 대기. 15분 단위로 쪼개서 sleep."""
    while True:
        now = datetime.now(KST)
        remain = (when_dt - now).total_seconds()
        if remain <= 0:
            return
        pytime.sleep(min(remain, 900))  # 최대 15분 간격으로 슬립

# ── 알림/트리거 쿨다운 ─────────────────────────────────────────────────
_last_sent: Dict[str, float] = {}
_last_trigger: Dict[str, float] = {}  # [NEW] 파이프라인 트리거 쿨다운

def _notify(msg: str, key: str = "risk_manager", cooldown_sec: int = 300) -> None:
    """디스코드 알림(쿨다운 적용). 실패해도 파이프라인 저지하지 않음."""
    try:
        now = pytime.time()
        if key not in _last_sent or now - _last_sent[key] >= cooldown_sec:
            _last_sent[key] = now
            if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
                send_discord_message(content=msg)
    except Exception:
        pass

def _can_trigger(key: str, cooldown_sec: int) -> bool:
    """[NEW] 파이프라인 기동 쿨다운"""
    now = pytime.time()
    last = _last_trigger.get(key, 0.0)
    if now - last >= cooldown_sec:
        _last_trigger[key] = now
        return True
    return False

# ── 데이터 클래스: 규칙 파라미터 ─────────────────────────────────────────
@dataclass
class SellRules:
    """매도 판단 규칙 파라미터"""
    stop_loss_buffer: float = 0.0     # 손절가 대비 추가 버퍼(비율). 예: 0.003 -> 손절가*1.003
    take_profit_buffer: float = 0.0   # 목표가 대비 추가 버퍼(비율)
    rsi_take_profit: Optional[float] = 75.0  # RSI가 이 값 이상이면 이익실현 고려(None이면 비활성)
    max_holding_days: Optional[int] = None   # 보유일수 상한(None이면 비활성)
    # [NEW] 전일 종가 이탈 + 시간대 로직
    prev_close_break_sell: bool = False          # 전일 종가 하회 시 매도 규칙 활성화
    prev_close_buffer_pct: float = 0.003         # 전일 종가 대비 추가 버퍼(예: 0.003 => -0.3%)
    time_windows_for_sells: Optional[List[str]] = None  # 매도 전반 허용 시간대(예: ["09:05-14:50","15:00-15:20"])
    time_windows_for_take_profit: Optional[List[str]] = None  # 이익실현만 허용 시간대(없으면 전반 윈도우 사용)
    confirm_bars_for_break: int = 0              # [단순 버전] 확인봉 개수(일봉 기준, 0=미사용)

# ── 유틸 함수들 ────────────────────────────────────────────────────────
def _to_int(x) -> int:
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        s = x.replace(",", "").strip()
        try:
            return int(float(s))
        except Exception:
            return 0
    return 0

def _to_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace(",", "").strip()
        return float(s)
    except Exception:
        return default

def _percent_backup_levels(entry_price: float, risk_params: Dict) -> Dict[str, float]:
    """손절/목표가가 없을 때 즉시 산출하는 퍼센트 백업"""
    stop_pct = float(risk_params.get("stop_pct", 0.03))
    rr = float(risk_params.get("reward_risk", 2.0))
    stop_px = entry_price * (1.0 - stop_pct)
    risk = max(1e-6, entry_price - stop_px)
    tgt_px = entry_price + rr * risk
    return {
        "손절가": int(round(stop_px)),
        "목표가": int(round(tgt_px)),
        "source": "percent_backup",
    }

# ── RiskManager 본체 ───────────────────────────────────────────────────
class RiskManager:
    """
    - settings(settings.py의 settings 객체)를 받아 리스크 파라미터를 로드
    - check_sell_condition(holding, stock_info) 제공
    - 필요 시 계좌 스냅샷(account.py) 트리거하는 헬퍼 제공
    - [NEW] 보유 0일 때 트레이딩 파이프라인 자동 기동(조건부)
    """

    def __init__(self, settings_obj):
        self.settings_obj = settings_obj
        self.config = getattr(settings_obj, "_config", {}) or {}
        self.env = self.config.get("trading_environment", "prod")

        # risk_params에서 룰 추출
        rp = self.config.get("risk_params", {}) or {}
        self.rules = SellRules(
            stop_loss_buffer=float(rp.get("stop_loss_buffer", 0.0)),
            take_profit_buffer=float(rp.get("take_profit_buffer", 0.0)),
            rsi_take_profit=(float(rp["rsi_take_profit"]) if rp.get("rsi_take_profit") is not None else None),
            max_holding_days=(int(rp["max_holding_days"]) if rp.get("max_holding_days") is not None else None),
            # [NEW] 전일 종가 + 시간대
            prev_close_break_sell=bool(rp.get("prev_close_break_sell", False)),
            prev_close_buffer_pct=float(rp.get("prev_close_buffer_pct", 0.003)),
            time_windows_for_sells=rp.get("time_windows_for_sells") or rp.get("time_windows") or None,
            time_windows_for_take_profit=rp.get("time_windows_for_take_profit") or None,
            confirm_bars_for_break=int(rp.get("confirm_bars_for_break", 0)),
        )

        # [NEW] 자동 파이프라인 트리거 설정
        self.auto_trigger_when_empty: bool = bool(rp.get("auto_trigger_trader_when_empty", True))
        self.auto_trigger_cooldown_sec: int = int(rp.get("auto_trigger_cooldown_sec", 900))  # 15분
        self.min_cash_to_trigger: int = int(rp.get("min_cash_to_trigger", 100_000))
        self.buy_time_windows: List[str] = rp.get("buy_time_windows") or ["09:05-14:50"]

        logger.info(f"RiskManager 초기화 완료 (env={self.env})")
        # [NEW] 전일 종가 캐시(세션 단위)
        self._prev_close_cache: Dict[str, int] = {}

    # ── screener_core 호출로 실시간 지표/레벨 ───────────────────────────
    def compute_realtime_levels(self, ticker: str, entry_price: float) -> Dict:
        """
        손절가/목표가/RSI 계산(파일 참조 없이 함수 호출).
        - entry_price: 진입가가 없다면 현재가를 그대로 넣어도 됨
        인터페이스 보장: 항상 {'손절가','목표가','RSI','Price','source'} 포함.
        """
        t = str(ticker).zfill(6)
        ep = float(entry_price)
        risk_params = self.config.get("risk_params", {}) or {}

        # 기본 페이로드 + 퍼센트 백업(선적용, 이후 코어 계산 성공 시 덮어씀)
        out: Dict = {
            "Ticker": t,
            "Price": int(round(ep)),
            "RSI": 50.0,
            **_percent_backup_levels(ep, risk_params),
        }

        # 1) 손절/목표가 (성공 시만 덮어쓰기)
        try:
            date_str = datetime.now(KST).strftime("%Y%m%d")
            levels = _compute_levels(t, ep, date_str, risk_params)
            if isinstance(levels, dict):
                if "손절가" in levels and "목표가" in levels:
                    out["손절가"] = int(round(_to_float(levels["손절가"], out["손절가"])))
                    out["목표가"] = int(round(_to_float(levels["목표가"], out["목표가"])))
                out["source"] = str(levels.get("source") or out.get("source") or "core_levels")
        except Exception as e:
            logger.warning(f"[{t}] 손절/목표가 계산 실패: {e} (백업 사용)")

        # 2) RSI
        try:
            end_dt = datetime.now(KST)
            start_dt = end_dt - timedelta(days=365)
            df = get_historical_prices(t, start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
            if df is not None and not df.empty:
                close_col = "Close" if "Close" in df.columns else [c for c in df.columns if c.lower() == "close"][0]
                out["RSI"] = round(float(calculate_rsi(df[close_col])), 2)
        except Exception as e:
            logger.warning(f"[{t}] RSI 계산 실패: {e} (기본 50.0 사용)")

        # (선택) 미러링 필드
        out["levels_source"] = out.get("source")
        return out

    # ── [NEW] 전일 종가 조회(세션 캐시) ───────────────────────────────
    def _get_prev_close(self, ticker: str) -> Optional[int]:
        t = str(ticker).zfill(6)
        if t in self._prev_close_cache:
            return self._prev_close_cache[t]
        try:
            end_dt = datetime.now(KST)
            start_dt = end_dt - timedelta(days=10)  # 주말 포함 여유
            df = get_historical_prices(t, start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
            if df is None or len(df) < 2:
                return None
            close_col = "Close" if "Close" in df.columns else [c for c in df.columns if c.lower() == "close"][0]
            prev_close = float(df[close_col].iloc[-2])
            val = int(round(prev_close))
            self._prev_close_cache[t] = val
            return val
        except Exception as e:
            logger.debug(f"[{t}] 전일 종가 조회 실패: {e}")
            return None

    # ── 계좌 스냅샷 트리거/읽기 분리 ────────────────────────────────────
    def trigger_account_snapshot(self) -> bool:
        """
        account.py를 실행해 최신 summary/balance 파일을 생성만 합니다.
        읽기는 호출측(트레이더)에서 utils.get_account_snapshot_cached 사용 권장.
        """
        try:
            subprocess.run(
                ["python", str(ACCOUNT_SCRIPT_PATH)],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
            )
            logger.info("(RiskManager) account.py 자동 실행 완료")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"(RiskManager) account.py 실행 실패: exit={e.returncode}\n{e.stderr}")
        except FileNotFoundError:
            logger.error(f"(RiskManager) account.py 경로를 찾지 못했습니다: {ACCOUNT_SCRIPT_PATH}")
        except Exception as e:
            logger.error(f"(RiskManager) account.py 실행 중 예외: {e}")
        return False

    def refresh_account_snapshot(self) -> Tuple[Dict[str, int], List[Dict], Optional[str], Optional[str]]:
        """
        [호환 유지] 최신 스냅샷을 생성 → utils 캐시로 읽어 반환합니다.
        return: (cash_info_dict, holdings_list, summary_file, balance_file)
        """
        self.trigger_account_snapshot()
        summary_dict, balance_list, summary_path, balance_path = get_account_snapshot_cached(
            summary_pattern="summary_*.json",
            balance_pattern="balance_*.json",
            ttl_sec=5,  # 즉시 재로딩 유도(파일 mtime 변화 감지)
        )
        cash_map = extract_cash_from_summary(summary_dict)
        return (
            cash_map,
            balance_list,
            str(summary_path) if summary_path else None,
            str(balance_path) if balance_path else None,
        )

    # ── [NEW] 보유 0일 때 트레이딩 파이프라인 자동 기동 ────────────────
    def _should_trigger_trader(self, cash_map: Dict[str, int], holdings: List[Dict]) -> Tuple[bool, str]:
        """
        트레이더 자동 기동 조건 판단.
        - 보유수량 총합 0
        - 장중 & 매수 시간 창
        - available_cash(없으면 dnca_tot_amt) >= min_cash_to_trigger
        - 쿨다운 내 중복 트리거 방지
        """
        if not self.auto_trigger_when_empty:
            return False, "auto_trigger_trader_when_empty=False"

        # 보유수량 총합
        total_qty = sum(int(str(h.get("hldg_qty", 0)).replace(",", "")) for h in holdings)
        if total_qty > 0:
            return False, f"holdings_qty>0 ({total_qty})"

        now = datetime.now(KST)
        if not is_market_hours(now):
            return False, "장외"

        # 매수 시간 창
        if not in_time_windows(now, self.buy_time_windows):
            return False, f"매수 시간대 아님: {self.buy_time_windows}"

        available = int(cash_map.get("available_cash") or cash_map.get("dnca_tot_amt") or 0)
        if available < self.min_cash_to_trigger:
            return False, f"가용 현금 부족 {available:,} < {self.min_cash_to_trigger:,}"

        # 쿨다운
        if not _can_trigger("trigger_trader_when_empty", self.auto_trigger_cooldown_sec):
            return False, "쿨다운"

        return True, "OK"

    def _trigger_trader_pipeline_once(self) -> bool:
        """
        trader.py를 단발 실행. 성공/실패 여부 반환.
        """
        try:
            logger.info("[AUTO] 보유 0 & 조건 충족 → trader.py 자동 기동")
            res = subprocess.run(
                ["python", str(TRADER_SCRIPT_PATH)],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                timeout=1200,  # 20분 안전 타임아웃
            )
            head = (res.stdout or "")[-600:]
            logger.info("[AUTO] trader.py 완료. tail:\n%s", head)
            _notify("🤖 보유 0 → 트레이딩 파이프라인 자동 기동 완료", key="auto_trigger_trader_ok", cooldown_sec=300)
            return True
        except subprocess.CalledProcessError as e:
            logger.error("[AUTO] trader.py 실패: exit=%s\nstdout:\n%s\nstderr:\n%s",
                         e.returncode, (e.stdout or "")[-600:], (e.stderr or "")[-600:])
            _notify("❌ 보유 0 → 트레이딩 파이프라인 기동 실패", key="auto_trigger_trader_fail", cooldown_sec=300)
        except subprocess.TimeoutExpired:
            logger.error("[AUTO] trader.py 타임아웃")
            _notify("⏱️ 보유 0 → 트레이딩 파이프라인 타임아웃", key="auto_trigger_trader_fail", cooldown_sec=300)
        except FileNotFoundError:
            logger.error("[AUTO] trader.py 경로를 찾지 못함: %s", TRADER_SCRIPT_PATH)
            _notify("❗ trader.py를 찾지 못했습니다.", key="auto_trigger_trader_fail", cooldown_sec=300)
        except Exception as e:
            logger.error("[AUTO] trader.py 기동 중 예외: %s", e, exc_info=True)
            _notify(f"❗ trader.py 기동 중 예외: {e}", key="auto_trigger_trader_fail", cooldown_sec=300)
        return False

    # ── 매도 판단 로직 ────────────────────────────────────────────────
    def check_sell_condition(self, holding: Dict, stock_info: Dict) -> Tuple[str, str]:
        """
        보유 종목/스크리너 정보 기반 매도 판단.
        return: ("SELL" or "KEEP", reason)
        요구사항:
          - stop/target이 들어오면 최우선 사용, 없으면 퍼센트 백업 즉시 산출
          - RSI 미존재 시 50.0으로 대체하고 (지표부재) 표기
          - 사유에 levels_source/전략명 포함
        """
        ticker = str(holding.get("pdno", "")).zfill(6)
        name = holding.get("prdt_name", "N/A")
        qty = _to_int(holding.get("hldg_qty", 0))
        cur_price = _to_int(holding.get("prpr", 0))  # 현재가
        if qty <= 0 or cur_price <= 0:
            return "KEEP", f"{name}({ticker}) 수량/가격 정보 부족"

        # 입력 손절/목표 우선
        stop_px_in = _to_float(stock_info.get("손절가"), 0.0)
        take_px_in = _to_float(stock_info.get("목표가"), 0.0)
        levels_source = str(stock_info.get("source") or "").strip()

        # 없으면 퍼센트 백업 즉시 산출
        if stop_px_in <= 0 or take_px_in <= 0:
            entry_price = _to_float(holding.get("pchs_avg_pric"), 0.0) or float(cur_price)
            backup = _percent_backup_levels(entry_price, self.config.get("risk_params", {}) or {})
            stop_px = float(backup["손절가"]); take_px = float(backup["목표가"])
            levels_source = "percent_backup"
        else:
            stop_px, take_px = float(stop_px_in), float(take_px_in)
            if not levels_source:
                levels_source = "unknown"

        # 버퍼 적용
        stop_threshold = stop_px * (1.0 + self.rules.stop_loss_buffer) if (self.rules.stop_loss_buffer and stop_px > 0) else stop_px
        tp_threshold   = take_px * (1.0 - self.rules.take_profit_buffer) if (self.rules.take_profit_buffer and take_px > 0) else take_px

        # RSI 확보 (미존재 시 50.0 + 지표부재 표기)
        rsi_raw = stock_info.get("RSI")
        rsi_missing = (rsi_raw is None or str(rsi_raw).strip() == "")
        rsi = _to_float(rsi_raw, 50.0)
        rsi_note = " (지표부재)" if rsi_missing else ""

        # 공통: 시간대 허용 여부
        # in_time_windows(now, None or []) → True 특성에 맞춰 None/빈 리스트면 True로 처리
        now_kst = datetime.now(KST)
        sell_win_ok = True if not self.rules.time_windows_for_sells else in_time_windows(now_kst, self.rules.time_windows_for_sells)
        tp_base = self.rules.time_windows_for_take_profit or self.rules.time_windows_for_sells
        tp_win_ok = True if not tp_base else in_time_windows(now_kst, tp_base)

        # 1) 손절 전략 (시간대 필터 적용)
        if stop_threshold > 0 and cur_price <= stop_threshold and sell_win_ok:
            return (
                "SELL",
                f"손절가 도달({cur_price:,} ≤ {int(round(stop_threshold)):,}) | 전략=StopLoss, levels_source={levels_source} | win={self.rules.time_windows_for_sells or 'ALL'}"
            )
        # 2) 목표가 전략 (시간대 필터 적용: 없으면 전반 윈도우 사용)
        if tp_threshold > 0 and cur_price >= tp_threshold and tp_win_ok:
            return (
                "SELL",
                f"목표가 도달({cur_price:,} ≥ {int(round(tp_threshold)):,}) | 전략=TakeProfit, levels_source={levels_source} | win={self.rules.time_windows_for_take_profit or self.rules.time_windows_for_sells or 'ALL'}"
            )
        # 3) RSI 과열 전략
        if self.rules.rsi_take_profit is not None and rsi >= float(self.rules.rsi_take_profit):
            return (
                "SELL",
                f"RSI 과열({rsi:.1f}≥{float(self.rules.rsi_take_profit):.1f}{rsi_note}) | 전략=RSI_TP, levels_source={levels_source}"
            )
        # 3.5) 전일 종가 이탈 전략 (시간대 필터 + 버퍼)
        if self.rules.prev_close_break_sell and sell_win_ok:
            prev_close = self._get_prev_close(ticker)
            if prev_close and prev_close > 0:
                thresh = int(round(prev_close * (1.0 - float(self.rules.prev_close_buffer_pct))))
                if cur_price <= thresh:
                    confirm_note = f", confirm={self.rules.confirm_bars_for_break}D" if self.rules.confirm_bars_for_break > 0 else ""
                    return (
                        "SELL",
                        f"전일 종가 이탈({cur_price:,} ≤ {thresh:,}) | 전략=PrevCloseBreak{confirm_note}, levels_source={levels_source} | prev_close={prev_close:,} | win={self.rules.time_windows_for_sells or 'ALL'}"
                    )
        # 4) 보유일수 상한
        if self.rules.max_holding_days and stock_info.get("entry_date"):
            try:
                dt = datetime.fromisoformat(str(stock_info["entry_date"]))
                days = (datetime.now(KST) - dt.astimezone(KST)).days
                if days >= int(self.rules.max_holding_days):
                    return (
                        "SELL",
                        f"보유일수 초과({days}d ≥ {int(self.rules.max_holding_days)}d) | 전략=MaxHoldingDays, levels_source={levels_source}"
                    )
            except Exception:
                pass

        # 유지
        return (
            "KEEP",
            f"유지: {name}({ticker}) 현재가={cur_price:,}, 손절={int(round(stop_px)) if stop_px else 'N/A'}, "
            f"목표={int(round(take_px)) if take_px else 'N/A'}, RSI={rsi:.1f}{rsi_note}, levels_source={levels_source}"
            f", win_sell={self.rules.time_windows_for_sells or 'ALL'}"
        )

    # ── 상태 요약(디스코드/로그) ────────────────────────────────────────
    def summarize_account_state(self, cash_map: Dict[str, int], holdings: List[Dict]) -> str:
        d2 = cash_map.get("prvs_rcdl_excc_amt", 0)
        nx = cash_map.get("nxdy_excc_amt", 0)
        dn = cash_map.get("dnca_tot_amt", 0)
        total = cash_map.get("tot_evlu_amt", 0) or 0
        return (
            f"보유종목: {len([h for h in holdings if _to_int(h.get('hldg_qty', 0))>0])}개\n"
            f"D+2 출금가능: {d2:,}원\n"
            f"익일 출금가능: {nx:,}원\n"
            f"예수금: {dn:,}원\n"
            f"총평가(요약): {total:,}원"
        )

# ── 실행 루틴 ──────────────────────────────────────────────────────────
def _run_cycle(rm: RiskManager, *, notify_summary: bool = True) -> None:
    """리스크 체크 1회 사이클"""
    # 1) 계좌 스냅샷 갱신 및 요약 로그
    cash, holds, s_path, b_path = rm.refresh_account_snapshot()
    msg = rm.summarize_account_state(cash, holds)
    logger.info("\n" + msg + f"\nfiles: {b_path}, {s_path}")
    if notify_summary:
        _notify(" 계좌 요약\n" + msg, key="risk_summary", cooldown_sec=600)

    # [NEW] 보유 0이면 트레이딩 파이프라인 조건부 기동
    ok, why = rm._should_trigger_trader(cash, holds)
    if ok:
        rm._trigger_trader_pipeline_once()
    else:
        logger.info(f"[AUTO] 파이프라인 미기동: {why}")

    # 2) 각 보유 종목: 손절/목표가/RSI 즉시 계산 후 판단
    if holds:
        for h in holds:
            ticker = str(h.get("pdno", "")).zfill(6)
            cur_price = _to_float(h.get("prpr"), 0.0)
            if cur_price <= 0:
                logger.info(f"유지 판단: {h.get('prdt_name','N/A')}({ticker}) 현재가 정보 없음")
                continue

            stock_info = rm.compute_realtime_levels(ticker, cur_price)
            decision, reason = rm.check_sell_condition(h, stock_info)
            if decision == "SELL":
                log_msg = f" 매도 판단: {reason}"
                logger.warning(log_msg)
                _notify("⚠️" + log_msg, key=f"risk_sell_{ticker}", cooldown_sec=300)
            else:
                logger.info(f"유지 판단: {reason}")

# ── 단독 실행: 장외 대기 + 장중 주기 모니터링 루프 ─────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Risk Manager loop / one-shot")
    parser.add_argument("--interval", type=int, default=180, help="주기(초) - 기본 180초")
    parser.add_argument("--once", action="store_true", help="단발 실행(1사이클만 수행 후 종료)")
    parser.add_argument("--no-wait", action="store_true", help="장외 대기 없이 즉시 실행(스모크 테스트 용)")
    args = parser.parse_args()

    # settings 모듈이 없을 수도 있으므로 가짜 설정으로 구동 테스트
    class _DummySettings:
        _config = {
            "trading_environment": os.getenv("ENV", "prod"),
            "risk_params": {
                "stop_loss_buffer": 0.0,
                "take_profit_buffer": 0.0,
                "rsi_take_profit": 75,
                "max_holding_days": None,
                # 시간대/전일종가 규칙 샘플
                # "time_windows_for_sells": ["09:05-14:50"],
                # "time_windows_for_take_profit": ["09:05-15:20"],
                # "prev_close_break_sell": True,
                # "prev_close_buffer_pct": 0.003,
                # "confirm_bars_for_break": 0,
                # screener_core._compute_levels 에서 사용하는 키들(없어도 퍼센트 백업 경로 동작)
                "atr_period": 14,
                "atr_k_stop": 1.5,
                "swing_lookback": 20,
                "reward_risk": 2.0,
                "stop_pct": 0.03,

                # [AUTO TRIGGER DEFAULTS]
                "auto_trigger_trader_when_empty": True,
                "auto_trigger_cooldown_sec": 900,
                "min_cash_to_trigger": 100_000,
                "buy_time_windows": ["09:05-14:50"],
            }
        }

    rm = RiskManager(_DummySettings())

    if args.once:
        # 스모크/수동 확인용 단발 실행
        if not args.no_wait:
            now = datetime.now(KST)
            if not is_market_hours(now):
                nxt = next_market_open_kst(now)
                logger.info("장외입니다. 다음 장 시작까지 대기: %s", nxt.strftime("%Y-%m-%d %H:%M:%S %Z"))
                sleep_until_kst(nxt)
        _run_cycle(rm, notify_summary=True)
    else:
        # 주기 실행 루프
        logger.info("RiskManager 모니터링 루프 시작 (interval=%ds)", args.interval)
        while True:
            now = datetime.now(KST)

            if not args.no_wait and not is_market_hours(now):
                nxt = next_market_open_kst(now)
                logger.info("장외입니다. 다음 장 시작까지 대기: %s", nxt.strftime("%Y-%m-%d %H:%M:%S %Z"))
                sleep_until_kst(nxt)
                # 깨어나면 장중일 것

            # 안전: 장중 아니라면 한 번 더 확인
            if args.no_wait or is_market_hours():
                _run_cycle(rm, notify_summary=True)
            else:
                logger.info("아직 장외입니다. 재확인 대기.")

            logger.info("⏳ %d초 대기 후 다음 주기 실행", args.interval)
            pytime.sleep(max(5, int(args.interval)))
