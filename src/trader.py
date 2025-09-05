# src/trader.py
import json
import logging
import os
import random
import time
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict
import uuid

# ── 공통 유틸리티 / 설정 ──────────────────────────────────────────────
from utils import (
    setup_logging,
    find_latest_file,
    OUTPUT_DIR,
    load_account_files_with_retry,
    extract_cash_from_summary,
    KST,
)
from api.kis_auth import KIS
from risk_manager import RiskManager
from settings import settings

# recorder: DB 초기화/기록 + 마지막 매수 조회(FIFO 연결)
from recorder import initialize_db, record_trade, fetch_trades_by_tickers

# ── 디스코드 노티파이어 ───────────────────────────────────────────────
from notifier import (
    DiscordLogHandler,
    WEBHOOK_URL,
    is_valid_webhook,
    send_discord_message,
    create_trade_embed,
)

# ── 로깅 초기화 ───────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger("trader")

# 루트 로거에 디스코드 에러 핸들러 장착(중복 방지)
_root = logging.getLogger()
if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

# ── 간단 레이트 리밋(스팸 방지) ───────────────────────────────────────
_last_sent_ts = defaultdict(float)
DEFAULT_COOLDOWN_SEC = 120  # 동일 키 알림 최소 간격(초)

def _can_send(key: str, cooldown: int = DEFAULT_COOLDOWN_SEC) -> bool:
    now = time.time()
    if now - _last_sent_ts[key] >= cooldown:
        _last_sent_ts[key] = now
        return True
    return False

def _scope_key_with_run_id(key: str) -> str:
    """key 가 'run:'으로 시작하지 않으면 RUN_ID 네임스페이스를 앞에 붙인다."""
    if key.startswith("run:"):
        return key
    return f"run:{os.getenv('RUN_ID', 'na')}:{key}"

def _notify_text(content: str, key: str = "trader_generic", cooldown: int = DEFAULT_COOLDOWN_SEC):
    key = _scope_key_with_run_id(key)
    if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL) and _can_send(key, cooldown):
        try:
            send_discord_message(content=content)
        except Exception:
            pass

def _notify_embed(embed: Dict, key: str, cooldown: int = DEFAULT_COOLDOWN_SEC):
    key = _scope_key_with_run_id(key)
    if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL) and _can_send(key, cooldown):
        try:
            send_discord_message(embeds=[embed])
        except Exception:
            pass

# ── 경로/상수 ─────────────────────────────────────────────────────────
COOLDOWN_FILE = OUTPUT_DIR / "cooldown.json"
ACCOUNT_SCRIPT_PATH = "/app/src/account.py"  # 계좌 스냅샷 생성 전용 스크립트

# ── 호가단위 ───────────────────────────────────────────────────────────
def get_tick_size(price: float) -> float:
    """호가 단위를 반환합니다."""
    if price < 2000: return 1
    elif price < 5000: return 5
    elif price < 20000: return 10
    elif price < 50000: return 50
    elif price < 200000: return 100
    elif price < 500000: return 500
    else: return 1000

# ── 보조 파서 ─────────────────────────────────────────────────────────
def _to_int(v) -> int:
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        s = v.replace(",", "").strip()
        try:
            return int(float(s))
        except Exception:
            return 0
    return 0

def _to_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace(",", "").strip()
        return float(s)
    except Exception:
        return default

# ── 스키마 관용성(디폴트) ────────────────────────────────────────────
# 과거/슬림 산출물과의 호환을 위해 빈 값 자동 보정
SCHEMA_DEFAULTS: Dict[str, Any] = {
    "Sector": "N/A",
    "SectorSource": "unknown",  # 누락 경고 다수 → 기본값 주입
    "ATR": 0.0,
    "RSI": 50.0,
    "MA50": None,
    "MA200": None,
    "손절가": None,
    "목표가": None,
    "source": "unknown",
    "daily_chart": [],
    "investor_flow": [],
    "Price": 0,
}

# ── Trader 본체 ───────────────────────────────────────────────────────
class Trader:
    def __init__(self, settings_obj):
        self.settings = settings_obj._config
        self.env = self.settings.get("trading_environment", "vps")
        self.is_real_trading = (self.env == "prod")
        self.risk_params = self.settings.get("risk_params", {}) or {}
        self.trading_params = self.settings.get("trading_params", {}) or {}   # [REBUY] 새 섹션

        # [REBUY] 파라미터 로드
        self.allow_rebuy = bool(self.trading_params.get("allow_rebuy", False))
        self.max_positions = int(self.trading_params.get("max_positions", self.risk_params.get("max_positions", 5)))
        self.max_legs_per_ticker = int(self.trading_params.get("max_legs_per_ticker", 1))
        self.per_ticker_max_weight = float(self.trading_params.get("per_ticker_max_weight", 1.0))
        self.min_order_cash = int(self.trading_params.get("min_order_cash", 0))
        self.rebuy_atr_k = float(self.trading_params.get("rebuy_atr_k", 0.0))
        self.rebuy_rsi_ceiling = float(self.trading_params.get("rebuy_rsi_ceiling", 100.0))

        self.cooldown_list = self._load_cooldown_list()
        self.cooldown_period_days = self.risk_params.get("cooldown_period_days", 10)

        # 1) run_id: env 우선, 없으면 로컬 생성
        self.run_id = os.getenv("RUN_ID") or (datetime.now(KST).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])

        # 2) 계좌 갱신 최소 간격: settings.notifications.discord_cooldown_sec 기반
        notif_cfg = self.settings.get("notifications", {}) or {}
        self._account_update_min_interval = float(notif_cfg.get("discord_cooldown_sec", 60))

        self._last_account_update_ts = 0.0

        initialize_db()
        logger.info("거래 기록용 데이터베이스가 초기화되었습니다.")

        # KIS 초기화(주문/가격조회만 사용; 잔고 조회는 account.py로 대체)
        try:
            self.kis = KIS(config={}, env=self.env)
            if not getattr(self, "kis", None) or not getattr(self.kis, "auth_token", None):
                raise ConnectionError("KIS API 인증에 실패했습니다 (토큰 없음).")
            logger.info(f"'{self.env}' 모드로 KIS API 인증 완료.")
        except Exception as e:
            logger.error(f"KIS API 초기화 중 오류 발생: {e}", exc_info=True)
            raise ConnectionError("KIS API 초기화에 실패했습니다.") from e

        self.risk_manager = RiskManager(settings_obj)

        self.all_stock_data = self._load_all_stock_data()

        _notify_text(
            f" Trader 초기화 완료 (env={self.env}, real_trading={self.is_real_trading}, run_id={self.run_id})",
            key="phase:init", cooldown=60
        )

    # ── 스크리너 전체 데이터 (파일 네이밍 변경 대응) ─────────────────────
    def _load_all_stock_data(self) -> Dict[str, Dict]:
        """
        screener.py 저장 파일 네이밍:
          - 전체 후보(풀): screener_candidates_full_{YYYYMMDD}_{MARKET}.json
          - 전체 랭킹(풀):   screener_rank_full_{YYYYMMDD}_{MARKET}.json
          - 후보(슬림):     screener_candidates_{YYYYMMDD}_{MARKET}.json
        위 순서로 탐색하여 최초 발견 파일을 로딩.
        """
        patterns = [
            "screener_candidates_full_*_*.json",
            "screener_rank_full_*_*.json",
            "screener_candidates_*_*.json",  # 슬림(무거운 컬럼 없음)
        ]
        picked = None
        for pat in patterns:
            f = find_latest_file(pat)
            if f:
                picked = f
                break

        if not picked:
            logger.info("스크리너 결과 파일을 찾지 못했습니다. (candidates_full/rank_full/candidates)")
            _notify_text("ℹ️ 스크리너 전체 데이터 없음 -> 실시간 조회로 진행",
                         key="phase:load_full_missing", cooldown=600)
            return {}

        try:
            with open(picked, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.info(f"{picked.name}의 형식이 리스트가 아닙니다. 빈 데이터로 진행합니다.")
                return {}

            mapping: Dict[str, Dict] = {}
            missing_total = 0
            missing_by_key: Dict[str, int] = defaultdict(int)

            for stock in data:
                if not isinstance(stock, dict):
                    continue
                t = str(stock.get('Ticker', '')).zfill(6)
                if not t or t == "000000":
                    continue

                # 스키마 디폴트 보정 (관용성)
                for k, dv in SCHEMA_DEFAULTS.items():
                    if k not in stock or stock.get(k) is None:
                        stock[k] = dv
                        missing_total += 1
                        missing_by_key[k] += 1

                mapping[t] = stock

            # 집계 로그(경고 대신 정보)
            if missing_total > 0:
                logger.info(
                    f"{picked.name} 스키마 결손 자동 보정: 총 {missing_total}건 | "
                    + ", ".join([f"{k}={v}" for k, v in sorted(missing_by_key.items())])
                )
            logger.info(f"스크리너 데이터 로드 완료: {picked.name} ({len(mapping)}종목)")
            return mapping

        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"{picked.name} 파일 로드 실패: {e}")
            _notify_text(f"⚠️ 스크리너 데이터 로드 실패: {e}",
                         key="phase:load_full_error", cooldown=600)
            return {}

    # ── account.py 트리거 (중복 최소화) ────────────────────────────────
    def _update_account_info(self, force: bool = False):
        now = time.time()
        # 최소 20초 또는 설정의 절반 간격 보다 짧으면 스킵
        if not force and (now - self._last_account_update_ts) < max(20, self._account_update_min_interval / 2):
            logger.info("account.py 호출 스킵(최근에 갱신됨).")
            return
        self._last_account_update_ts = now

        logger.info("[CALL] account.py 실행(계좌 스냅샷 갱신)...")
        try:
            for attempt in range(2):
                try:
                    subprocess.run(
                        ["python", str(ACCOUNT_SCRIPT_PATH)],
                        capture_output=True, text=True, check=True, encoding="utf-8",
                        timeout=60  # 핵심: 타임아웃
                    )
                    logger.info("[RET] account.py 실행 완료.")
                    _notify_text(" account.py 실행 완료(요약/잔고 갱신)",
                                 key="phase:account_update", cooldown=60)
                    break
                except subprocess.TimeoutExpired:
                    logger.error("account.py 타임아웃(60s). 재시도 중...")
                    if attempt == 1:
                        raise
                except subprocess.CalledProcessError as e:
                    head = (e.stderr or "")[:400]  # 너무 긴 stderr 방지
                    logger.error(f"account.py 실행 오류(Exit {e.returncode}). stderr:\n{head}")
                    if attempt == 1:
                        raise
        except FileNotFoundError:
            msg = f"스크립트를 찾을 수 없습니다: {ACCOUNT_SCRIPT_PATH}"
            logger.error(msg)
            _notify_text(f"❗ {msg}", key="phase:account_not_found", cooldown=300)
        except Exception as e:
            msg = f"계좌 정보 업데이트 중 예외: {e}"
            logger.error(msg, exc_info=True)
            _notify_text(f"❗ {msg}", key="phase:account_exc", cooldown=300)

    # ── 스냅샷 로더/헬퍼 ──────────────────────────────────────────────
    def _load_snapshot(self) -> Tuple[int, List[Dict], Dict[str, int]]:
        summary_dict, balance_list, *_ = load_account_files_with_retry(
            summary_pattern="summary_*.json",
            balance_pattern="balance_*.json",
            max_wait_sec=5,
        )
        cash_map = extract_cash_from_summary(summary_dict)
        available_cash = cash_map.get("available_cash", 0)
        holdings: List[Dict] = []
        if balance_list:
            holdings = [h for h in balance_list if _to_int(h.get("hldg_qty", 0)) > 0]
        return available_cash, holdings, cash_map

    @staticmethod
    def _get_qty(holdings: List[Dict], ticker: str) -> int:
        for h in holdings:
            if str(h.get("pdno", "")).zfill(6) == ticker:
                return _to_int(h.get("hldg_qty", 0))
        return 0

    # [REBUY] 포트폴리오 스냅샷(비중/평단)
    def _portfolio_snapshot(self, holdings: List[Dict]) -> Dict[str, Any]:
        """
        by_ticker_value: {ticker: 평가금액 추정}
        avg_price_by_ticker: {ticker: 평균단가}
        portfolio_value: 총 평가금액(대략)
        """
        by_val: Dict[str, float] = defaultdict(float)
        avg_price: Dict[str, float] = {}
        pv = 0.0
        for h in holdings:
            t = str(h.get("pdno", "")).zfill(6)
            qty = _to_int(h.get("hldg_qty", 0))
            prpr = _to_float(h.get("prpr"), 0.0)  # 현재가
            avgp = _to_float(h.get("pchs_avg_pric"), 0.0) or prpr
            val = prpr * qty if (prpr and qty) else 0.0
            by_val[t] += val
            pv += val
            if qty > 0:
                avg_price[t] = avgp
        return {
            "by_ticker_value": by_val,
            "avg_price_by_ticker": avg_price,
            "portfolio_value": pv,
        }

    # [REBUY] 티커별 레그 수(간단: 보유면 1레그, 추가매수시에도 1로 관리 가능)
    def _legs_count_for_ticker(self, holdings: List[Dict], ticker: str) -> int:
        return 1 if any(str(h.get("pdno", "")).zfill(6) == ticker and _to_int(h.get("hldg_qty", 0)) > 0 for h in holdings) else 0

    # [REBUY] 추가매수 가능성 판단
    def _can_rebuy(self, ticker: str, info: Dict[str, Any], holdings: List[Dict], available_cash: int) -> Tuple[bool, str]:
        # 레그 수 체크
        if self._legs_count_for_ticker(holdings, ticker) >= self.max_legs_per_ticker:
            return False, "레그 한도 초과"

        snap = self._portfolio_snapshot(holdings)
        pv = float(snap["portfolio_value"])
        tv = float(snap["by_ticker_value"].get(ticker, 0.0))
        avgp = float(snap["avg_price_by_ticker"].get(ticker, 0.0))

        # 비중 한도
        if pv > 0 and (tv / pv) >= self.per_ticker_max_weight:
            return False, "티커 비중 한도 초과"

        # 현금 한도
        if available_cash < max(self.min_order_cash, 0):
            return False, "현금 부족"

        # ATR/RSI 조건
        price = _to_float(info.get("Price"), 0.0)
        atr = _to_float(info.get("ATR"), 0.0)
        rsi = _to_float(info.get("RSI"), 50.0)

        if atr > 0 and price < (avgp + self.rebuy_atr_k * atr):
            return False, f"가격조건 미충족(px<{avgp}+{self.rebuy_atr_k}*ATR)"
        if rsi > self.rebuy_rsi_ceiling:
            return False, f"RSI 상한 초과({rsi:.1f}>{self.rebuy_rsi_ceiling})"

        return True, "OK"

    # ── 계좌 파일에서 가용 현금/보유 종목 로드 ─────────────────────────
    def get_account_info_from_files(self) -> Tuple[int, List[Dict], Dict[str, int]]:
        available_cash, holdings, cash_map = self._load_snapshot()

        d2 = cash_map.get("prvs_rcdl_excc_amt", 0)
        nx = cash_map.get("nxdy_excc_amt", 0)
        dn = cash_map.get("dnca_tot_amt", 0)
        tot = cash_map.get("tot_evlu_amt", 0)

        logger.info(
            f" 계좌 조회 완료\n"
            f"보유종목: {len(holdings)}개\n"
            f"D+2: {d2:,}원\n"
            f"익일: {nx:,}원\n"
            f"예수금: {dn:,}원\n"
            f"총평가: {tot:,}원\n"
            f"→ 사용 가용예산: {available_cash:,}원"
        )
        return available_cash, holdings, cash_map

    # ── 주문 안전 래퍼 ────────────────────────────────────────────────
    def _order_cash_safe(self, **kwargs) -> Dict[str, Any]:
        """
        KIS 주문 래퍼.
        - 성공: {'ok': True, 'rt_cd': '0', 'msg_cd': 'XXXX', 'msg1': '성공메시지', 'raw': dict, 'df': DataFrame}
        - 실패: {'ok': False, 'rt_cd': '8'(등), 'msg1': '에러메시지', 'raw': dict, ...}
        """
        try:
            df = self.kis.order_cash(**kwargs)
            if df is None or df.empty:
                return {'ok': False, 'msg1': 'API 응답 없음'}
            rec = df.to_dict('records')[0]
            rt_cd = rec.get('rt_cd', '')
            ok = (rt_cd == '0')
            return {
                'ok': ok,
                'rt_cd': rt_cd,
                'msg_cd': rec.get('msg_cd'),
                'msg1': rec.get('msg1', '메시지 없음'),
                'raw': rec,
                'df': df,
            }
        except Exception as e:
            logger.error(f"주문 API 호출 중 예외 발생: {e}", exc_info=True)
            return {'ok': False, 'msg1': str(e), 'error': str(e)}

    # ── [보강] 일시적 오류 감지 & 재시도 ───────────────────────────────
    @staticmethod
    def _is_transient_error(result: Dict[str, Any]) -> bool:
        msg = (result.get('msg1') or '').lower()
        # 네트워크/일시장애/레이트리밋/빈응답/타임아웃 계열 키워드
        hints = [
            'timeout', 'timed out', 'temporarily', '일시', 'too many requests',
            '429', 'service unavailable', '서버', '네트워크', 'api 응답 없음'
        ]
        return any(h in msg for h in hints)

    def _order_cash_retry(self, max_retries: int = 3, backoff_base: float = 0.5, **kwargs) -> Dict[str, Any]:
        """
        주문 재시도 래퍼: 일시적 오류로 판단되면 지수 백오프로 재시도.
        """
        backoff = backoff_base
        for attempt in range(1, max_retries + 1):
            res = self._order_cash_safe(**kwargs)
            if res.get('ok'):
                return res
            # 비일시적(명확 거절)로 보이면 즉시 반환
            if not self._is_transient_error(res):
                return res
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
        return res  # 마지막 시도 결과 반환

    # ── 매도 로직(주문 후 스냅샷 검증 + PnL/부모연결) ───────────────────
    def _parse_reason_code(self, reason: str) -> str:
        if "전략=StopLoss" in reason or "손절가 도달" in reason:
            return "STOP_LOSS_HIT"
        if "전략=TakeProfit" in reason or "목표가 도달" in reason:
            return "TAKE_PROFIT_HIT"
        if "전략=RSI_TP" in reason or "RSI 과열" in reason:
            return "RSI_OVERBOUGHT"
        if "전략=MaxHoldingDays" in reason or "보유일수 초과" in reason:
            return "MAX_HOLDING_DAYS"
        if reason.startswith("유지"):
            return "HOLD"
        return "UNKNOWN"

    def run_sell_logic(self, holdings: List[Dict]):
        logger.info(f"--------- 보유 종목 {len(holdings)}개 매도 로직 실행 ---------")

        executed_sell = False
        if not holdings:
            logger.info("매도할 보유 종목이 없습니다.")
            return

        holding_tickers = [str(h.get("pdno", "")).zfill(6) for h in holdings if _to_int(h.get("hldg_qty", 0)) > 0]
        last_buy_trades = fetch_trades_by_tickers(holding_tickers)

        for holding in holdings:
            ticker = str(holding.get("pdno", "")).zfill(6)
            name = holding.get("prdt_name", "N/A")
            quantity = _to_int(holding.get("hldg_qty", 0))
            if not ticker or quantity <= 0:
                continue

            stock_info = self.all_stock_data.get(ticker, {})
            for k, dv in SCHEMA_DEFAULTS.items():
                stock_info.setdefault(k, dv)

            decision, reason = self.risk_manager.check_sell_condition(holding, stock_info)
            if decision != "SELL":
                logger.info(f"유지 판단: {reason}")
                continue

            reason_code = self._parse_reason_code(reason)
            logger.info(f"매도 결정: {name}({ticker}) {quantity}주. 사유: {reason} | code={reason_code}")
            executed_sell = True

            if self.is_real_trading:
                pre_qty = self._get_qty(holdings, ticker)

                # 주문: 시장가(01) 매도 (재시도 포함)
                result = self._order_cash_retry(
                    ord_dv="01", pdno=ticker, ord_dvsn="01", ord_qty=str(quantity), ord_unpr="0"
                )

                # 주문 후 스냅샷으로 체결 반영 확인
                time.sleep(2)
                self._update_account_info()
                _, holdings_after, _ = self._load_snapshot()
                post_qty = self._get_qty(holdings_after, ticker)
                filled_qty = max(0, pre_qty - post_qty)

                # 체결가(보수적) → 현재가 조회 실패 시 0
                try:
                    price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                    current_price = _to_int(price_df['stck_prpr'].iloc[0]) if (price_df is not None and not price_df.empty) else 0
                except Exception:
                    current_price = 0

                parent_trade_id = None
                pnl_amount = None
                last_buy = last_buy_trades.get(ticker)
                if last_buy and filled_qty > 0:
                    parent_trade_id = last_buy.get('id')
                    buy_price = _to_int(last_buy.get('price', holding.get('pchs_avg_pric', 0)))
                    if buy_price and current_price:
                        pnl_amount = (current_price - buy_price) * filled_qty

                if result.get('ok'):
                    trade_status = "completed" if filled_qty > 0 else "submitted"
                    record_trade({
                        "side": "sell", "ticker": ticker, "name": name,
                        "qty": filled_qty if filled_qty > 0 else quantity,
                        "price": current_price,
                        "trade_status": trade_status,
                        "strategy_details": {
                            "reason": reason,
                            "reason_code": reason_code,
                            "broker_msg": result.get('msg1')
                        },
                        "parent_trade_id": parent_trade_id,
                        "pnl_amount": pnl_amount,
                        "sell_reason": reason
                    })
                    _notify_embed(create_trade_embed({
                        "side": "SELL", "name": name, "ticker": ticker,
                        "qty": filled_qty if filled_qty > 0 else quantity,
                        "price": current_price, "trade_status": trade_status,
                        "strategy_details": {
                            "reason": reason,
                            "reason_code": reason_code,
                            "broker_msg": result.get('msg1')
                        }
                    }), key=f"phase:sell:{ticker}", cooldown=30)

                    if filled_qty == 0:
                        logger.info("  -> 응답은 성공이나 즉시 체결 없음(미체결 가능). submitted로 기록.")
                else:
                    if filled_qty > 0:
                        record_trade({
                            "side": "sell", "ticker": ticker, "name": name,
                            "qty": filled_qty, "price": current_price,
                            "trade_status": "completed",
                            "strategy_details": {"broker_msg": "응답 실패→체결 확인", "reason": reason, "reason_code": reason_code},
                            "parent_trade_id": parent_trade_id,
                            "pnl_amount": pnl_amount,
                            "sell_reason": reason
                        })
                        _notify_embed(create_trade_embed({
                            "side": "SELL", "name": name, "ticker": ticker,
                            "qty": filled_qty, "price": current_price,
                            "trade_status": "completed",
                            "strategy_details": {"broker_msg": "응답 실패→체결 확인", "reason_code": reason_code}
                        }), key=f"phase:sell_fb:{ticker}", cooldown=30)
                    else:
                        err = result.get('msg1', 'Unknown error')
                        record_trade({
                            "side": "sell", "ticker": ticker, "name": name,
                            "qty": quantity, "price": current_price,
                            "trade_status": "failed",
                            "strategy_details": {
                                "error": err,
                                "rt_cd": result.get('rt_cd'),
                                "msg_cd": result.get('msg_cd'),
                                "reason": reason,
                                "reason_code": reason_code
                            },
                            "sell_reason": reason
                        })
                        _notify_embed(create_trade_embed({
                            "side": "SELL", "name": name, "ticker": ticker,
                            "qty": quantity, "price": current_price,
                            "trade_status": "failed",
                            "strategy_details": {"error": err, "reason_code": reason_code}
                        }), key=f"phase:sell_fail:{ticker}", cooldown=30)
                        self._add_to_cooldown(ticker, "매도 주문 실패")
            else:
                # 모의매매
                logger.info(f"[모의] {name}({ticker}) {quantity}주 시장가 매도 실행.")
                record_trade({
                    "side": "sell", "ticker": ticker, "name": name,
                    "qty": quantity, "price": 0, "trade_status": "completed",
                    "strategy_details": {"reason": reason, "reason_code": reason_code},
                    "sell_reason": reason
                })
                _notify_text(
                    f" [모의] SELL {name}({ticker}) x{quantity} | {reason_code}",
                    key=f"phase:paper_sell:{ticker}", cooldown=30
                )
                self._add_to_cooldown(ticker, "모의 매도 완료")

        if executed_sell:
            time.sleep(5)
            self._update_account_info()

    # ── 매수 로직(주문 후 스냅샷 검증/보정) ─────────────────────────────
    def run_buy_logic(self, available_cash: int, holdings: List[Dict]):
        logger.info(f"--------- 신규/추가 매수 로직 실행 (가용 예산: {available_cash:,} 원) ---------")

        if available_cash < 10000:
            logger.info("가용 예산이 부족하여 매수 로직을 실행할 수 없습니다.")
            _notify_text("⚠️ 가용 예산 부족 → 매수 스킵",
                         key="phase:no_cash", cooldown=300)
            return

        trade_plan_file = find_latest_file("gpt_trades_*.json")
        if not trade_plan_file:
            logger.info("매수 계획 파일(gpt_trades_*.json)이 없어 매수를 건너뜁니다.")
            _notify_text("ℹ️ gpt_trades 파일 없음 → 매수 스킵",
                         key="phase:no_trades", cooldown=600)
            return

        with open(trade_plan_file, 'r', encoding='utf-8') as f:
            trade_plans = json.load(f)

        buy_plans = [p for p in trade_plans if p.get("결정") == "매수"]
        if not buy_plans:
            logger.info("매수 결정이 내려진 종목이 없습니다.")
            _notify_text("ℹ️ 매수 결정된 종목 없음",
                         key="phase:no_buy", cooldown=300)
            return

        # 안전 디폴트 주입
        for p in buy_plans:
            p.setdefault("stock_info", {})
            for k, dv in SCHEMA_DEFAULTS.items():
                p["stock_info"].setdefault(k, dv)

        holding_tickers = {str(h.get("pdno", "")).zfill(6) for h in holdings if _to_int(h.get("hldg_qty", 0)) > 0}

        # [REBUY] 후보 분리: 신규 / 추가매수
        new_targets = []
        rebuy_candidates = []
        for plan in buy_plans:
            info = plan["stock_info"]
            ticker = str(info.get("Ticker", "")).zfill(6)
            name = info.get("Name", "N/A")

            if ticker in holding_tickers:
                if not self.allow_rebuy:
                    logger.info(f"[{name}({ticker})] 이미 보유 → 추가매수 비활성이라 제외")
                    continue
                if self._is_in_cooldown(ticker):
                    logger.info(f"[{name}({ticker})] 쿨다운 중 → 추가매수 제외")
                    continue
                ok, why = self._can_rebuy(ticker, info, holdings, available_cash)
                if not ok:
                    logger.info(f"[REBUY-블록] {name}({ticker}) 제외: {why}")
                    continue
                logger.info(f"[REBUY] {name}({ticker}) 추가매수 후보 등록 ({why})")
                rebuy_candidates.append(plan)
            else:
                if self._is_in_cooldown(ticker):
                    logger.info(f"[{name}({ticker})] 쿨다운 중 → 신규매수 제외")
                    continue
                new_targets.append(plan)

        # [REBUY] 추가매수는 슬롯과 무관하게 진행(현금/한도 조건만)
        remaining_cash = available_cash
        any_order_placed = False

        def _execute_buy_batch(plans: List[Dict], batch_name: str):
            nonlocal remaining_cash, any_order_placed
            if not plans:
                return
            _notify_text(f" {batch_name} 매수 시도 {len(plans)}종목 (예산 {remaining_cash:,}원)",
                         key=f"phase:{batch_name.lower()}_start", cooldown=120)
            logger.info(f"총 {len(plans)}개 종목 {batch_name} 매수 시도. 유동적 예산 배분 적용.")

            for i, plan in enumerate(plans):
                info = plan["stock_info"]
                ticker, name = str(info["Ticker"]).zfill(6), info.get("Name", "N/A")
                slots_left = len(plans) - i
                budget_for_this_stock = remaining_cash // max(slots_left, 1)
                logger.info(f"  -> [{i+1}/{len(plans)}] {name}({ticker}) 배분 예산: {budget_for_this_stock:,.0f}원")

                # 현재가 확보
                current_price = _to_int(info.get("Price", 0))
                if current_price == 0:
                    try:
                        price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                        if price_df is not None and not price_df.empty:
                            current_price = _to_int(price_df['stck_prpr'].iloc[0])
                    except Exception:
                        current_price = 0
                    if current_price == 0:
                        logger.info(f"  -> [{name}({ticker})] 현재가 조회 실패. 매수를 건너뜁니다.")
                        _notify_embed(create_trade_embed({
                            "side": "BUY", "name": name, "ticker": ticker,
                            "qty": 0, "price": 0, "trade_status": "skipped",
                            "strategy_details": {"reason_code": "PRICE_FETCH_FAILED", "batch": batch_name}
                        }), key=f"phase:buy_skip_price:{ticker}", cooldown=120)
                        continue

                # 수량/호가 계산
                tick_size = get_tick_size(current_price)
                order_price = current_price + (tick_size * random.randint(1, 3))
                quantity = int(budget_for_this_stock // order_price)
                if quantity == 0 or budget_for_this_stock < max(self.min_order_cash, 0):
                    logger.info(f"  -> [{name}({ticker})] 예산 부족으로 매수 불가.")
                    _notify_embed(create_trade_embed({
                        "side": "BUY", "name": name, "ticker": ticker,
                        "qty": 0, "price": order_price, "trade_status": "skipped",
                        "strategy_details": {
                            "reason_code": "INSUFFICIENT_CASH",
                            "required": max(order_price, self.min_order_cash),
                            "available": int(budget_for_this_stock),
                            "batch": batch_name
                        }
                    }), key=f"phase:buy_insufficient:{ticker}", cooldown=120)
                    continue

                pre_qty = self._get_qty(holdings, ticker)
                logger.info(f"  -> 매수 준비: {name}({ticker}), 수량: {quantity}주, 지정가: {order_price:,.0f}원 [{batch_name}]")

                if self.is_real_trading:
                    # 재시도 포함 주문
                    result = self._order_cash_retry(
                        ord_dv="02", pdno=ticker, ord_dvsn="00", ord_qty=str(quantity), ord_unpr=str(int(order_price))
                    )

                    # 주문 즉시 후행 스냅샷으로 실제 반영 확인
                    time.sleep(2)
                    self._update_account_info()
                    _, holdings_after, _ = self._load_snapshot()
                    post_qty = self._get_qty(holdings_after, ticker)
                    qty_delta = max(0, post_qty - pre_qty)

                    if result.get('ok'):
                        trade_status = "completed" if qty_delta > 0 else "submitted"
                        record_trade({
                            "side": "buy", "ticker": ticker, "name": name,
                            "qty": qty_delta if qty_delta > 0 else quantity,
                            "price": order_price,
                            "trade_status": trade_status,
                            "gpt_analysis": plan,
                            "strategy_details": {"broker_msg": result.get('msg1'), "batch": batch_name}
                        })
                        if qty_delta > 0:
                            remaining_cash -= (qty_delta * order_price)
                        any_order_placed = True
                        _notify_embed(create_trade_embed({
                            "side": "BUY", "name": name, "ticker": ticker,
                            "qty": qty_delta if qty_delta > 0 else quantity,
                            "price": order_price, "trade_status": "submitted" if qty_delta == 0 else "completed",
                            "strategy_details": {"broker_msg": result.get('msg1'), "batch": batch_name}
                        }), key=f"phase:buy:{ticker}", cooldown=30)
                    else:
                        if qty_delta > 0:
                            record_trade({
                                "side": "buy", "ticker": ticker, "name": name,
                                "qty": qty_delta, "price": order_price, "trade_status": "completed",
                                "gpt_analysis": plan,
                                "strategy_details": {"broker_msg": "응답 실패→체결 확인", "batch": batch_name},
                            })
                            remaining_cash -= (qty_delta * order_price)
                            any_order_placed = True
                            _notify_embed(create_trade_embed({
                                "side": "BUY", "name": name, "ticker": ticker,
                                "qty": qty_delta, "price": order_price, "trade_status": "completed",
                                "strategy_details": {"broker_msg": "응답 실패→체결 확인", "batch": batch_name}
                            }), key=f"phase:buy_fb:{ticker}", cooldown=30)
                        else:
                            err = result.get('msg1', 'Unknown error')
                            record_trade({
                                "side": "buy", "ticker": ticker, "name": name, "qty": quantity,
                                "price": order_price, "trade_status": "failed",
                                "strategy_details": {
                                    "error": err,
                                    "rt_cd": result.get('rt_cd'),
                                    "msg_cd": result.get('msg_cd'),
                                    "reason_code": "BROKER_REJECT",
                                    "batch": batch_name
                                },
                                "gpt_analysis": plan
                            })
                            self._add_to_cooldown(ticker, "매수 주문 실패")
                            _notify_embed(create_trade_embed({
                                "side": "BUY", "name": name, "ticker": ticker, "qty": quantity,
                                "price": order_price, "trade_status": "failed",
                                "strategy_details": {"error": err, "reason_code": "BROKER_REJECT", "batch": batch_name}
                            }), key=f"phase:buy_fail:{ticker}", cooldown=30)
                else:
                    # 모의매매
                    actual_spent = quantity * order_price
                    remaining_cash -= actual_spent
                    any_order_placed = True
                    record_trade({
                        "side": "buy", "ticker": ticker, "name": name,
                        "qty": quantity, "price": order_price, "trade_status": "completed",
                        "gpt_analysis": plan,
                        "strategy_details": {"batch": batch_name}
                    })
                    logger.info(f"  -> [모의] {name}({ticker}) {quantity}주 @{order_price:,.0f}원 지정가 매수 실행. [{batch_name}]")
                    _notify_text(
                        f" [모의] BUY {name}({ticker}) x{quantity} @ {order_price:,.0f} [{batch_name}]",
                        key=f"phase:paper_buy:{ticker}", cooldown=30
                    )

                logger.info(f"  -> 남은 예산: {remaining_cash:,.0f}원")
                time.sleep(0.3)

        # 1) [REBUY] 먼저 수행 (슬롯과 무관)
        _execute_buy_batch(rebuy_candidates, batch_name="REBUY")

        # 2) 신규 진입은 슬롯 제한 적용
        slots_to_fill = self.max_positions - len(holdings)
        if slots_to_fill <= 0:
            logger.info(f"신규 매수 슬롯이 없습니다 (최대: {self.max_positions}, 현재: {len(holdings)}).")
            _notify_text(f"ℹ️ 신규 매수 슬롯 없음 (max={self.max_positions}, curr={len(holdings)})",
                         key="phase:no_slots", cooldown=300)
        else:
            targets_to_buy = new_targets[:slots_to_fill]
            if not targets_to_buy:
                logger.info("신규로 매수할 최종 대상이 없습니다.")
                _notify_text("ℹ️ 신규 매수 대상 없음",
                             key="phase:no_targets", cooldown=300)
            else:
                _execute_buy_batch(targets_to_buy, batch_name="NEW")

        if any_order_placed:
            time.sleep(5)
            self._update_account_info()

    # ── 쿨다운 리스트 ─────────────────────────────────────────────────
    def _load_cooldown_list(self) -> dict:
        if not COOLDOWN_FILE.exists():
            return {}
        try:
            with open(COOLDOWN_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return {}

    def _save_cooldown_list(self):
        COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(COOLDOWN_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.cooldown_list, f, indent=2, ensure_ascii=False)

    def _add_to_cooldown(self, ticker: str, reason: str):
        end_date = (datetime.now() + timedelta(days=self.cooldown_period_days)).isoformat()
        self.cooldown_list[ticker] = end_date
        self._save_cooldown_list()
        logger.info(f"[{ticker}] {reason}으로 인해 쿨다운 목록에 추가. ({end_date}까지)")
        _notify_text(f"⛔ {ticker} 쿨다운 등록: {reason} (until {end_date[:19]})",
                     key=f"phase:cooldown:{ticker}", cooldown=60)

    def _is_in_cooldown(self, ticker: str) -> bool:
        if ticker not in self.cooldown_list:
            return False
        cooldown_end_date = datetime.fromisoformat(self.cooldown_list[ticker])
        if datetime.now() < cooldown_end_date:
            return True
        else:
            del self.cooldown_list[ticker]
            self._save_cooldown_list()
            return False

# ── 엔트리포인트 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        # 초기화
        trader = Trader(settings)

        # 최신 계좌 스냅샷 생성(파일 갱신) 후 파일에서 로드
        trader._update_account_info(force=True)
        cash0, holdings0, _ = trader.get_account_info_from_files()

        # 매도 로직
        if holdings0:
            trader.run_sell_logic(holdings0)
        else:
            logger.info("보유 종목이 없어 매도 로직을 건너뜁니다.")

        # 매도 이후 잔여 현금/보유 재조회
        cash1, holdings1, _ = trader.get_account_info_from_files()
        trader.run_buy_logic(cash1, holdings1)

        logger.info("모든 트레이딩 로직 실행 완료.")
        _notify_text("✅ 트레이딩 로직 실행 완료", key="phase:done", cooldown=60)

    except Exception as e:
        logger.critical(f"트레이더 실행 중 심각한 오류 발생: {e}", exc_info=True)
        _notify_text(f" 트레이더 치명적 오류: {str(e)[:1800]}",
                     key="phase:fatal", cooldown=30)
