# src/risk_manager.py
import json
import time as time_mod
import logging
from pathlib import Path
from datetime import datetime, time as dt_time
from typing import Tuple, Dict, Any, Optional
from collections import defaultdict

import pandas as pd

# ── 공통 유틸 (KST/로깅/경로/설정/최근파일/휴장일) ──
from utils import (
    setup_logging,
    KST,
    OUTPUT_DIR,
    load_config,
    find_latest_file,
    is_market_open_day,
)

# KIS API
from api.kis_auth import KIS

# 외부 모듈
from settings import Settings
from strategies import StrategyMixer

# ── 디스코드 노티파이어 ──
from notifier import (
    DiscordLogHandler,
    WEBHOOK_URL,
    is_valid_webhook,
    send_discord_message,
    create_trade_embed,
)

# ───────────────── 로깅 ─────────────────
setup_logging()
logger = logging.getLogger("risk_manager")

# 루트 로거에 디스코드 에러 핸들러 장착(중복 방지)
_root = logging.getLogger()
if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

# ── 간단 레이트 리밋(스팸 방지) ──
_last_sent_ts = defaultdict(float)
DEFAULT_COOLDOWN_SEC = 180  # 동일 키 알림 최소 간격


def _can_send(key: str, cooldown: int = DEFAULT_COOLDOWN_SEC) -> bool:
    now = time_mod.time()
    if now - _last_sent_ts[key] >= cooldown:
        _last_sent_ts[key] = now
        return True
    return False


def _notify_text(content: str, key: str = "generic", cooldown: int = DEFAULT_COOLDOWN_SEC):
    """텍스트 알림(쿨다운 적용)"""
    if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL) and _can_send(key, cooldown):
        send_discord_message(content=content)


def _notify_embed(embed: Dict, key: str, cooldown: int = DEFAULT_COOLDOWN_SEC):
    """임베드 알림(쿨다운 적용)"""
    if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL) and _can_send(key, cooldown):
        send_discord_message(embeds=[embed])


def is_market_open_time() -> bool:
    """현재 KST 시각이 정규장(09:00~15:30)인지 확인."""
    now_time = datetime.now(KST).time()
    return dt_time(9, 0) <= now_time <= dt_time(15, 30)


class RiskManager:
    def __init__(self, settings: Settings):
        """
        settings: Settings 객체 (전략/리스크 파라미터 포함)
        - KIS 인증은 settings._config.kis_broker 우선, 없으면 utils.load_config()에서 폴백
        """
        self.settings = settings
        self.env = settings._config.get("trading_environment", "mock")
        kis_cfg = settings._config.get("kis_broker") or load_config().get("kis_broker", {})

        self.kis = KIS(config=kis_cfg, env=self.env)
        if not getattr(self.kis, "auth_token", None):
            raise ConnectionError("KIS API 인증 실패")

        self.risk_params = settings.risk_params
        self.strategy_mixer = StrategyMixer(settings)

        logger.info("RiskManager 초기화 완료 (env=%s)", self.env)
        _notify_text(f"🛡️ RiskManager 초기화 완료 (env={self.env})", key="rm_init", cooldown=60)

    # ─────────── 데이터/호출 ───────────
    def get_realtime_price(self, ticker: str) -> float:
        """KIS 실시간 현재가 조회 (단일 종목). 실패 시 0.0 반환."""
        try:
            df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=str(ticker).zfill(6))
            if isinstance(df, pd.DataFrame) and not df.empty and "stck_prpr" in df.columns:
                return float(df["stck_prpr"].iloc[0])
        except Exception as e:
            logger.warning("실시간 가격 조회 오류 (%s): %s", ticker, e)
        return 0.0

    # ─────────── 의사결정 ───────────
    def check_sell_condition(self, holding: Dict[str, Any], stock_info: Dict[str, Any]) -> Tuple[str, str]:
        """
        보유 종목 매도 조건 판단.
        - StrategyMixer.decide_sell 사용
        - 반환: ("SELL"|"HOLD", 이유)
        """
        ticker = str(holding.get("pdno", "")).zfill(6)
        prdt_name = holding.get("prdt_name", "N/A")
        avg_price_str = holding.get("pchs_avg_pric", "0")

        try:
            avg_price = float(avg_price_str)
        except (ValueError, TypeError):
            avg_price = 0.0

        if not ticker or avg_price <= 0:
            return "HOLD", "종목 정보 부족"

        # 현재가 보강
        if "Price" not in stock_info:
            current_price = self.get_realtime_price(ticker)
            if current_price <= 0:
                return "HOLD", "현재가 조회 실패"
            stock_info["Price"] = current_price

        # 매도 판단
        should_sell, reason = self.strategy_mixer.decide_sell(holding, stock_info)

        current_price = float(stock_info.get("Price", 0) or 0)
        pl_pct = (current_price - avg_price) / avg_price if avg_price > 0 else 0.0

        logger.info(
            "[%s(%s)] 현재가: %s | 수익률: %.2f%% | 매도판단: %s | 근거: %s",
            prdt_name, ticker, f"{current_price:,.0f}", pl_pct * 100, should_sell, reason
        )

        if should_sell:
            return "SELL", reason
        return "HOLD", "매도 조건 미충족"

    # ─────────── 모니터링 루프 ───────────
    def monitor_holdings(self):
        """
        보유 주식 실시간 모니터링 루프.
        - /app/output/balance_YYYYMMDD.json 을 최신 1개 자동 선택
        - /app/output/screener_full_*.json 에서 보조 정보(TA/점수 등) 로드(있으면)
        """
        logger.info("보유 주식 실시간 모니터링 시작...")
        _notify_text("👀 보유 종목 실시간 모니터링 시작", key="rm_monitor_start", cooldown=60)

        balance_file: Optional[Path] = find_latest_file("balance_*.json")
        if not balance_file:
            msg = "balance 파일을 찾을 수 없습니다. 먼저 account.py를 실행하세요."
            logger.error(msg)
            _notify_text(f"❗ {msg}", key="rm_no_balance", cooldown=60)
            return

        try:
            with open(balance_file, "r", encoding="utf-8") as f:
                balance_payload = json.load(f)
            balance_data = balance_payload.get("data", [])
        except Exception as e:
            msg = f"balance 파일 로드 실패({balance_file}): {e}"
            logger.error(msg)
            _notify_text(f"❗ {msg}", key="rm_balance_load_err", cooldown=60)
            return

        if not balance_data:
            logger.info("보유 주식이 없습니다.")
            _notify_text("ℹ️ 보유 주식 없음", key="rm_empty_holdings", cooldown=300)
            return

        # 보조 정보(전체 스크리너 랭킹) 로드
        full_screener_file = find_latest_file("screener_full_*.json")
        all_stock_data: Dict[str, Dict[str, Any]] = {}
        if full_screener_file:
            try:
                with open(full_screener_file, "r", encoding="utf-8") as f:
                    all_stocks = json.load(f)
                for stock in all_stocks:
                    t = str(stock.get("Ticker", "")).zfill(6)
                    if t:
                        all_stock_data[t] = stock
                logger.info("보조 스톡 데이터 로드: %s (총 %d건)", full_screener_file.name, len(all_stock_data))
            except Exception as e:
                logger.warning("보조 스톡 데이터 로드 실패(%s): %s", full_screener_file, e)

        # 루프
        while True:
            if is_market_open_day() and is_market_open_time():
                ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
                logger.info("--- %s | 보유 종목 리스크 점검 ---", ts)
                _notify_text(f"🕒 리스크 점검 주기 실행 ({ts} KST)", key="rm_tick", cooldown=900)  # 15분마다 1회

                for holding in balance_data:
                    try:
                        qty = int(holding.get("hldg_qty", 0) or 0)
                    except Exception:
                        qty = 0
                    if qty <= 0:
                        continue

                    ticker = str(holding.get("pdno", "")).zfill(6)
                    prdt_name = holding.get("prdt_name", "N/A")

                    stock_info = dict(all_stock_data.get(ticker, {}))  # copy
                    decision, reason = self.check_sell_condition(holding, stock_info)

                    if decision == "SELL":
                        # 현재가/수익률 보강
                        current_price = float(stock_info.get("Price", 0) or 0)
                        avg_price_str = holding.get("pchs_avg_pric", "0")
                        try:
                            avg_price = float(avg_price_str)
                        except (ValueError, TypeError):
                            avg_price = 0.0
                        pl_pct = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0

                        logger.warning("!!! 매도 신호: %s(%s) - %s !!!", prdt_name, ticker, reason)

                        # 디스코드 임베드 알림 (쿨다운: 종목별 3분)
                        embed = create_trade_embed({
                            "side": "SELL",
                            "name": prdt_name,
                            "ticker": ticker,
                            "qty": qty,
                            "price": current_price,
                            "trade_status": "signal",  # 아직 '주문'은 아님
                            "strategy_details": {"error": None, "reason": reason, "pnl_pct": f"{pl_pct:.2f}%"},
                        })
                        _notify_embed(embed, key=f"sell_signal_{ticker}", cooldown=180)

                # 1분 주기
                time_mod.sleep(60)
            else:
                now_str = datetime.now(KST).strftime("%H:%M:%S")
                print(f"장이 열리지 않았습니다. 대기합니다. (현재 시간: {now_str} KST)", end="\r")
                _notify_text("⏸️ 장외 시간 대기 중", key="rm_waiting", cooldown=1800)
                time_mod.sleep(300)


# ─────────── 독립 실행 ───────────
if __name__ == "__main__":
    try:
        # settings 인스턴스를 직접 주입
        from settings import settings
        rm = RiskManager(settings)
        rm.monitor_holdings()  # 스크리너 전체 데이터가 없으면 제한적으로 동작
        print("RiskManager가 성공적으로 종료되었습니다.")
    except Exception as e:
        logger.critical("RiskManager 실행 중 오류: %s", e, exc_info=True)
        # 치명적 예외는 즉시 알림(쿨다운 짧게)
        _notify_text(f"🛑 RiskManager 치명적 오류: {str(e)[:1800]}", key="rm_fatal", cooldown=30)
