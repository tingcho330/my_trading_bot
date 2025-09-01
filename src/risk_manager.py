# src/risk_manager.py
import os
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
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

logger = logging.getLogger("RiskManager")

# 루트 로거에 디스코드 에러 핸들러 장착(중복 방지)
_root = logging.getLogger()
if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

ACCOUNT_SCRIPT_PATH = "/app/src/account.py"

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
        return int(s) if s.isdigit() else 0
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
            rsi_take_profit=(
                float(rp["rsi_take_profit"]) if rp.get("rsi_take_profit") is not None else None
            ),
            max_holding_days=(
                int(rp["max_holding_days"]) if rp.get("max_holding_days") is not None else None
            ),
        )

        logger.info(f"🛡️ RiskManager 초기화 완료 (env={self.env})")

    # ── 계좌 스냅샷 로드/트리거 ────────────────────────────────────────
    def refresh_account_snapshot(self) -> Tuple[Dict[str, int], List[Dict], Optional[str], Optional[str]]:
        """
        account.py를 실행해 최신 summary/balance 생성 후 읽어온다.
        return: (cash_info_dict, holdings_list, summary_file, balance_file)
        """
        try:
            # account.py 실행 (성공/실패와 무관하게 읽기 재시도)
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

        # 최신 파일 읽기(재시도 포함)
        summary_dict, balance_list, summary_path, balance_path = load_account_files_with_retry(
            summary_pattern="summary_*.json",
            balance_pattern="balance_*.json",
            max_wait_sec=5,
        )
        # 현금 파싱
        cash_map = extract_cash_from_summary(summary_dict)
        # 파일명(문자열)만 반환
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
        # 필수 데이터 정리
        ticker = str(holding.get("pdno", "")).zfill(6)
        name = holding.get("prdt_name", "N/A")
        qty = _to_int(holding.get("hldg_qty", 0))
        cur_price = _to_int(holding.get("prpr", 0))  # 현재가
        if qty <= 0 or cur_price <= 0:
            return "HOLD", f"{name}({ticker}) 수량/가격 정보 부족"

        # 스크리너 정보 (손절가/목표가/RSI 등)
        stop_px = _to_float(stock_info.get("손절가"), 0.0)
        take_px = _to_float(stock_info.get("목표가"), 0.0)
        rsi = _to_float(stock_info.get("RSI"), 50.0)
        entry_px = _to_float(stock_info.get("Price"), cur_price)  # 진입가 없는 경우 현재가로 fallback

        # 버퍼 적용 손절/목표 기준
        if self.rules.stop_loss_buffer and stop_px > 0:
            stop_threshold = stop_px * (1.0 + self.rules.stop_loss_buffer)
        else:
            stop_threshold = stop_px

        if self.rules.take_profit_buffer and take_px > 0:
            tp_threshold = take_px * (1.0 - self.rules.take_profit_buffer)
        else:
            tp_threshold = take_px

        # 1) 손절 조건: 현재가 <= 손절가(버퍼 적용)
        if stop_threshold > 0 and cur_price <= stop_threshold:
            return "SELL", f"손절가 도달({cur_price} ≤ {int(stop_threshold)})"

        # 2) 목표가 도달: 현재가 ≥ 목표가(버퍼 적용)
        if tp_threshold > 0 and cur_price >= tp_threshold:
            return "SELL", f"목표가 도달({cur_price} ≥ {int(tp_threshold)})"

        # 3) RSI 이익실현
        if self.rules.rsi_take_profit is not None and rsi >= float(self.rules.rsi_take_profit):
            # 목표가가 없거나 멀다면 RSI 단독 기준으로도 정리
            return "SELL", f"RSI 과열({rsi:.1f}≥{float(self.rules.rsi_take_profit):.1f})"

        # 4) 보유일수 상한 검사(거래 기록 DB로부터 체결일을 알 수 있다면 적용)
        # trader.recorder에 parent_trade_id 등 기록 시 가져와 판단 가능.
        # 여기서는 stock_info에 가상의 'entry_date'가 있을 경우만 예시 적용.
        if self.rules.max_holding_days and stock_info.get("entry_date"):
            try:
                dt = datetime.fromisoformat(str(stock_info["entry_date"]))
                days = (datetime.now(KST) - dt.astimezone(KST)).days
                if days >= int(self.rules.max_holding_days):
                    return "SELL", f"보유일수 초과({days}d ≥ {int(self.rules.max_holding_days)}d)"
            except Exception:
                pass

        # (선택) 추세 역전 등 기술적 보조 룰을 추가하고 싶다면 여기에 확장
        # ex) 20일선 하향, 패턴 플래그 역진 등…

        return "HOLD", f"유지: {name}({ticker}) 현재가={cur_price:,}, 손절={int(stop_px) if stop_px else 'N/A'}, 목표={int(take_px) if take_px else 'N/A'}, RSI={rsi:.1f}"

    # ── 상태 요약(디스코드용) ──────────────────────────────────────────
    def summarize_account_state(self, cash_map: Dict[str, int], holdings: List[Dict]) -> str:
        """
        디스코드/로그용 간단 요약 문자열 생성
        """
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

# ── 단독 실행 테스트(선택) ─────────────────────────────────────────────
if __name__ == "__main__":
    # settings 모듈이 없을 수도 있으므로 가짜 설정으로 구동 테스트
    class _DummySettings:
        _config = {
            "trading_environment": os.getenv("ENV", "prod"),
            "risk_params": {
                "stop_loss_buffer": 0.0,
                "take_profit_buffer": 0.0,
                "rsi_take_profit": 75,
                "max_holding_days": None,
            }
        }

    rm = RiskManager(_DummySettings())

    # 계좌 스냅샷 갱신 및 요약 출력
    cash, holds, s_path, b_path = rm.refresh_account_snapshot()
    msg = rm.summarize_account_state(cash, holds)
    logger.info("\n" + msg + f"\nfiles: {b_path}, {s_path}")

    # 더미 매도 판단 테스트
    dummy_holding = {"pdno": "005930", "prdt_name": "삼성전자", "hldg_qty": "10", "prpr": "71000"}
    dummy_info = {"손절가": 68000, "목표가": 76000, "RSI": 73.2, "Price": 70000}
    decision, reason = rm.check_sell_condition(dummy_holding, dummy_info)
    logger.info(f"매도 판단 → {decision}: {reason}")
