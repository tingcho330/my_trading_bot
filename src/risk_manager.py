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
    load_account_files_with_retry,
    extract_cash_from_summary,
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

# ── 장중 정의(평일 09:00~15:30) ────────────────────────────────────────
MARKET_START = dt_time(9, 0)
MARKET_END   = dt_time(15, 30)

def is_market_hours() -> bool:
    """평일 09:00~15:30 (KST) 에만 True"""
    now = datetime.now(KST)
    if now.weekday() > 4:  # 0=월 ~ 4=금
        return False
    now_t = now.time()
    return MARKET_START <= now_t <= MARKET_END

# ── 데이터 클래스: 규칙 파라미터 ─────────────────────────────────────────
@dataclass
class SellRules:
    """매도 판단 규칙 파라미터"""
    stop_loss_buffer: float = 0.0     # 손절가 대비 추가 버퍼(비율). 예: 0.003 -> 손절가*1.003
    take_profit_buffer: float = 0.0   # 목표가 대비 추가 버퍼(비율)
    rsi_take_profit: Optional[float] = 75.0  # RSI가 이 값 이상이면 이익실현 고려(None이면 비활성)
    max_holding_days: Optional[int] = None   # 보유일수 상한(None이면 비활성)

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

def _notify(msg: str):
    try:
        if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
            send_discord_message(content=msg)
    except Exception:
        pass

# ── RiskManager 본체 ───────────────────────────────────────────────────
class RiskManager:
    """
    - settings(settings.py의 settings 객체)를 받아 리스크 파라미터를 로드
    - check_sell_condition(holding, stock_info) 제공
    - 필요 시 계좌 스냅샷(account.py) 트리거하는 헬퍼 제공
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
        )

        logger.info(f"🛡️ RiskManager 초기화 완료 (env={self.env})")

    # ── screener_core 호출로 실시간 지표/레벨 ───────────────────────────
    def compute_realtime_levels(self, ticker: str, entry_price: float) -> Dict:
        """
        손절가/목표가/RSI 계산(파일 참조 없이 함수 호출).
        - entry_price: 진입가가 없다면 현재가를 그대로 넣어도 됨
        """
        out: Dict = {"Ticker": str(ticker).zfill(6), "Price": int(round(float(entry_price)))}

        # 1) 손절/목표가
        try:
            date_str = datetime.now(KST).strftime("%Y%m%d")
            risk_params = self.config.get("risk_params", {}) or {}
            levels = _compute_levels(str(ticker).zfill(6), float(entry_price), date_str, risk_params)
            if isinstance(levels, dict):
                out.update({k: levels.get(k) for k in ("손절가", "목표가", "source") if k in levels})
        except Exception as e:
            logger.warning(f"[{ticker}] 손절/목표가 계산 실패: {e}")

        # 2) RSI
        try:
            end_dt = datetime.now(KST)
            start_dt = end_dt - timedelta(days=365)
            df = get_historical_prices(
                str(ticker).zfill(6),
                start_dt.strftime("%Y%m%d"),
                end_dt.strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                close_col = "Close" if "Close" in df.columns else [c for c in df.columns if c.lower() == "close"][0]
                rsi_val = float(calculate_rsi(df[close_col]))
                out["RSI"] = round(rsi_val, 2)
            else:
                out["RSI"] = 50.0
        except Exception as e:
            logger.warning(f"[{ticker}] RSI 계산 실패: {e}")
            out["RSI"] = 50.0

        return out

    # ── 계좌 스냅샷 로드/트리거 ────────────────────────────────────────
    def refresh_account_snapshot(self) -> Tuple[Dict[str, int], List[Dict], Optional[str], Optional[str]]:
        """
        account.py를 실행해 최신 summary/balance 생성 후 읽어온다.
        return: (cash_info_dict, holdings_list, summary_file, balance_file)
        """
        try:
            subprocess.run(
                ["python", str(ACCOUNT_SCRIPT_PATH)],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
            )
            logger.info("📗 (RiskManager) account.py 자동 실행 완료")
        except subprocess.CalledProcessError as e:
            logger.error(f"(RiskManager) account.py 실행 실패: exit={e.returncode}\n{e.stderr}")
        except FileNotFoundError:
            logger.error(f"(RiskManager) account.py 경로를 찾지 못했습니다: {ACCOUNT_SCRIPT_PATH}")
        except Exception as e:
            logger.error(f"(RiskManager) account.py 실행 중 예외: {e}")

        summary_dict, balance_list, summary_path, balance_path = load_account_files_with_retry(
            summary_pattern="summary_*.json",
            balance_pattern="balance_*.json",
            max_wait_sec=5,
        )
        cash_map = extract_cash_from_summary(summary_dict)
        return (
            cash_map,
            balance_list,
            str(summary_path) if summary_path else None,
            str(balance_path) if balance_path else None,
        )

    # ── 매도 판단 로직 ────────────────────────────────────────────────
    def check_sell_condition(self, holding: Dict, stock_info: Dict) -> Tuple[str, str]:
        """
        보유 종목/스크리너 정보 기반 매도 판단.
        return: ("SELL" or "HOLD", reason)
        """
        ticker = str(holding.get("pdno", "")).zfill(6)
        name = holding.get("prdt_name", "N/A")
        qty = _to_int(holding.get("hldg_qty", 0))
        cur_price = _to_int(holding.get("prpr", 0))  # 현재가
        if qty <= 0 or cur_price <= 0:
            return "HOLD", f"{name}({ticker}) 수량/가격 정보 부족"

        stop_px = _to_float(stock_info.get("손절가"), 0.0)
        take_px = _to_float(stock_info.get("목표가"), 0.0)
        rsi = _to_float(stock_info.get("RSI"), 50.0)

        # 버퍼 적용
        stop_threshold = stop_px * (1.0 + self.rules.stop_loss_buffer) if (self.rules.stop_loss_buffer and stop_px > 0) else stop_px
        tp_threshold   = take_px * (1.0 - self.rules.take_profit_buffer) if (self.rules.take_profit_buffer and take_px > 0) else take_px

        # 1) 손절
        if stop_threshold > 0 and cur_price <= stop_threshold:
            return "SELL", f"손절가 도달({cur_price:,} ≤ {int(stop_threshold):,})"
        # 2) 목표가
        if tp_threshold > 0 and cur_price >= tp_threshold:
            return "SELL", f"목표가 도달({cur_price:,} ≥ {int(tp_threshold):,})"
        # 3) RSI 과열
        if self.rules.rsi_take_profit is not None and rsi >= float(self.rules.rsi_take_profit):
            return "SELL", f"RSI 과열({rsi:.1f}≥{float(self.rules.rsi_take_profit):.1f})"
        # 4) 보유일수 상한
        if self.rules.max_holding_days and stock_info.get("entry_date"):
            try:
                dt = datetime.fromisoformat(str(stock_info["entry_date"]))
                days = (datetime.now(KST) - dt.astimezone(KST)).days
                if days >= int(self.rules.max_holding_days):
                    return "SELL", f"보유일수 초과({days}d ≥ {int(self.rules.max_holding_days)}d)"
            except Exception:
                pass

        return "HOLD", f"유지: {name}({ticker}) 현재가={cur_price:,}, 손절={int(stop_px) if stop_px else 'N/A'}, 목표={int(take_px) if take_px else 'N/A'}, RSI={rsi:.1f}"

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

# ── 단독 실행: 장중 3분 간격 모니터링 루프 ─────────────────────────────
if __name__ == "__main__":
    # 장중 가드: 평일 09:00~15:30 외에는 아예 실행하지 않음
    if not is_market_hours():
        logger.info(f"({datetime.now(KST):%Y-%m-%d %H:%M:%S}) 장중이 아니므로 RiskManager 실행을 종료합니다.")
        raise SystemExit(0)

    # settings 모듈이 없을 수도 있으므로 가짜 설정으로 구동 테스트
    class _DummySettings:
        _config = {
            "trading_environment": os.getenv("ENV", "prod"),
            "risk_params": {
                "stop_loss_buffer": 0.0,
                "take_profit_buffer": 0.0,
                "rsi_take_profit": 75,
                "max_holding_days": None,
                # screener_core._compute_levels 에서 사용하는 키들(없어도 퍼센트 백업 경로 동작)
                "atr_period": 14,
                "atr_k_stop": 1.5,
                "swing_lookback": 20,
                "reward_risk": 2.0,
                "stop_pct": 0.03,
            }
        }

    rm = RiskManager(_DummySettings())
    logger.info("📡 RiskManager 모니터링 루프 시작 (3분 간격)")

    while True:
        # 장 종료 시 루프 종료
        if not is_market_hours():
            logger.info(f"({datetime.now(KST):%Y-%m-%d %H:%M:%S}) 장 종료 → RiskManager 모니터링 루프 종료")
            break

        # 1) 계좌 스냅샷 갱신 및 요약 로그
        cash, holds, s_path, b_path = rm.refresh_account_snapshot()
        msg = rm.summarize_account_state(cash, holds)
        logger.info("\n" + msg + f"\nfiles: {b_path}, {s_path}")

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
                    log_msg = f"🚨 매도 판단: {reason}"
                    logger.warning(log_msg)
                    _notify(log_msg)
                else:
                    logger.info(f"유지 판단: {reason}")

        logger.info("⏳ 3분 대기 후 다음 주기 실행")
        pytime.sleep(180)  # 3분(180초) 간격
