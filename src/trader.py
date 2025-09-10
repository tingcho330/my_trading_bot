import json
import logging
import os
import random
import time
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict
import uuid

# ── 공통 유틸리티 / 설정 ──────────────────────────────────────────────
from utils import (
    setup_logging,
    find_latest_file,
    OUTPUT_DIR,
    extract_cash_from_summary,
    KST,
    in_time_windows,                 # 시간대 필터
    get_account_snapshot_cached,     # 계좌 스냅샷 캐시 리더
    get_tick_size,                   # 호가단위
    round_to_tick,                   # 호가 반올림
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
SCHEMA_DEFAULTS: Dict[str, Any] = {
    "Sector": "N/A",
    "SectorSource": "unknown",
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
    "Score": 0.0,
}

# ── 가성비 통계 로그 ───────────────────────────────────────────
def log_affordability_stats(usable_cash: int, buffer_ratio: float, candidates: List[Dict[str, Any]], min_order_cash: Optional[int] = None):
    cheapest = min((_to_int(c.get("Price", 0)) for c in candidates), default=0)
    buyable_cnt = sum(1 for c in candidates if _to_int(c.get("Price", 0)) <= int(usable_cash * (1 - buffer_ratio)))
    base = f"[Affordability] usable_cash={usable_cash:,}, buffer={buffer_ratio:.2%}, cheapest={cheapest:,}, buyable_count={buyable_cnt}"
    if isinstance(min_order_cash, (int, float)) and min_order_cash is not None:
        base += f", min_order_cash={int(min_order_cash):,}"
    logger.info(base)

# ── Trader 본체 ───────────────────────────────────────────────────────
class Trader:
    def __init__(self, settings_obj):
        self.settings = settings_obj._config
        self.env = self.settings.get("trading_environment", "vps")
        self.is_real_trading = (self.env == "prod")
        self.risk_params = self.settings.get("risk_params", {}) or {}
        self.trading_params = self.settings.get("trading_params", {}) or {}
        self.trading_guards = self.settings.get("trading_guards", {}) or {}
        self.screener_params = self.settings.get("screener_params", {}) or {}
        self.reporting = self.settings.get("reporting", {}) or {}

        # 시간대 로직
        self.buy_time_windows: List[str] = self.trading_params.get("buy_time_windows", ["09:05-14:50"])
        self.sell_time_windows: List[str] = self.trading_params.get("sell_time_windows", ["09:05-15:10"])

        # REBUY 파라미터
        self.allow_rebuy = bool(self.trading_params.get("allow_rebuy", False))
        self.max_positions = int(self.trading_params.get("max_positions", self.risk_params.get("max_positions", 5)))
        self.max_legs_per_ticker = int(self.trading_params.get("max_legs_per_ticker", 1))
        self.per_ticker_max_weight = float(self.trading_params.get("per_ticker_max_weight", 1.0))
        self.min_order_cash = int(self.trading_params.get("min_order_cash", 0))
        self.rebuy_atr_k = float(self.trading_params.get("rebuy_atr_k", 0.0))
        self.rebuy_rsi_ceiling = float(self.trading_params.get("rebuy_rsi_ceiling", 100.0))
        self.min_cash_reserve = int(self.trading_params.get("min_cash_reserve", 0))
        self.cash_buffer_ratio = float(self.trading_params.get("cash_buffer_ratio", 0.0))

        self.cooldown_list = self._load_cooldown_list()
        self.cooldown_period_days = self.risk_params.get("cooldown_period_days", 10)

        # 쿨다운: 연속 실패 카운트(메모리), 임계치
        self._fail_counts = defaultdict(int)
        self.cooldown_fail_threshold = int(self.risk_params.get("cooldown_fail_threshold", 2))

        # 회전 중복 방지 셋
        self._sold_once = set()         # RUN 내 동일 티커 중복 SELL 방지
        self._attempted_pairs = set()   # (sell_ticker, buy_ticker) 중복 회전 방지

        # 런타임 ID
        self.run_id = os.getenv("RUN_ID") or (datetime.now(KST).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])

        # 알림 레이트 리밋
        notif_cfg = self.settings.get("notifications", {}) or {}
        self._account_update_min_interval = float(notif_cfg.get("discord_cooldown_sec", 60))
        self._last_account_update_ts = 0.0

        # 요약/통계
        self.stats = {"buy": 0, "sell": 0, "hold": 0}
        self.summary_reason_code: Optional[str] = None
        self.summary_reason_detail: Optional[str] = None

        initialize_db()
        logger.info("거래 기록용 데이터베이스가 초기화되었습니다.")

        # KIS 초기화
        try:
            self.kis = KIS(config={}, env=self.env)
            if not getattr(self, "kis", None) or not getattr(self.kis, "auth_token", None):
                raise ConnectionError("KIS API 인증에 실패했습니다 (토큰 없음).")
            logger.info(f"'{self.env}' 모드로 KIS API 인증 완료.")
        except Exception as e:
            logger.error(f"KIS API 초기화 중 오류 발생: {e}", exc_info=True)
            raise ConnectionError("KIS API 초기화에 실패했습니다.") from e

        self.risk_manager = RiskManager(settings_obj)

        # 스크리너 전체 데이터(랭킹/후보)
        self.all_stock_data = self._load_all_stock_data()

        _notify_text(
            f" Trader 초기화 완료 (env={self.env}, real_trading={self.is_real_trading}, run_id={self.run_id})",
            key="phase:init", cooldown=60
        )

    # ── 스크리너 전체 데이터 로드 ──────────────────────────────────────
    def _load_all_stock_data(self) -> Dict[str, Dict]:
        patterns = [
            "screener_candidates_full_*_*.json",
            "screener_rank_full_*_*.json",
            "screener_candidates_*_*.json",
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
                for k, dv in SCHEMA_DEFAULTS.items():
                    if k not in stock or stock.get(k) is None:
                        stock[k] = dv
                        missing_total += 1
                        missing_by_key[k] += 1
                mapping[t] = stock

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

    # ── account.py 트리거 ────────────────────────────────────────────
    def _update_account_info(self, force: bool = False):
        now = time.time()
        # 회전 핵심 지점에서는 force=True로 강제 갱신 허용
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
                        timeout=60
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
                    head = (e.stderr or "")[:400]
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
        summary_dict, balance_list, *_ = get_account_snapshot_cached(
            summary_pattern="summary_*.json",
            balance_pattern="balance_*.json",
            ttl_sec=None,
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

    def _portfolio_snapshot(self, holdings: List[Dict]) -> Dict[str, Any]:
        by_val: Dict[str, float] = defaultdict(float)
        avg_price: Dict[str, float] = {}
        pv = 0.0
        for h in holdings:
            t = str(h.get("pdno", "")).zfill(6)
            qty = _to_int(h.get("hldg_qty", 0))
            prpr = _to_float(h.get("prpr"), 0.0)
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

    def _legs_count_for_ticker(self, holdings: List[Dict], ticker: str) -> int:
        return 1 if any(str(h.get("pdno", "")).zfill(6) == ticker and _to_int(h.get("hldg_qty", 0)) > 0 for h in holdings) else 0

    def _can_rebuy(self, ticker: str, info: Dict[str, Any], holdings: List[Dict], available_cash: int) -> Tuple[bool, str]:
        if self._legs_count_for_ticker(holdings, ticker) >= self.max_legs_per_ticker:
            return False, "레그 한도 초과"

        snap = self._portfolio_snapshot(holdings)
        pv = float(snap["portfolio_value"])
        tv = float(snap["by_ticker_value"].get(ticker, 0.0))
        avgp = float(snap["avg_price_by_ticker"].get(ticker, 0.0))

        if pv > 0 and (tv / pv) >= self.per_ticker_max_weight:
            return False, "티커 비중 한도 초과"

        if available_cash < max(self.min_order_cash, 0):
            return False, "현금 부족"

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
        try:
            df = self.kis.order_cash(**kwargs)
            if df is None or df.empty:
                return {'ok': False, 'rt_cd': None, 'msg_cd': None, 'msg1': 'API 응답 없음', 'http_status': None}
            rec = df.to_dict('records')[0]
            rt_cd = rec.get('rt_cd', '')
            ok = (rt_cd == '0')  # 표준 성공 판정
            return {
                'ok': ok,
                'rt_cd': rt_cd,
                'msg_cd': rec.get('msg_cd'),
                'msg1': rec.get('msg1', '메시지 없음'),
                'http_status': rec.get('status_code'),
                'raw': rec,
                'df': df,
            }
        except Exception as e:
            logger.error(f"주문 API 호출 중 예외 발생: {e}", exc_info=True)
            return {'ok': False, 'rt_cd': 'EXC', 'msg_cd': None, 'msg1': str(e), 'http_status': None, 'error': str(e)}

    @staticmethod
    def _is_transient_error(result: Dict[str, Any]) -> bool:
        msg = (result.get('msg1') or '').lower()
        status = result.get('http_status')
        # 5xx, 네트워크, 게이트웨이/서비스 불가, 일시 오류
        hints = [
            'timeout', 'timed out', 'temporarily', '일시', 'too many requests',
            'service unavailable', 'bad gateway', 'gateway', '네트워크', 'api 응답 없음'
        ]
        if isinstance(status, int) and 500 <= status < 600:
            return True
        return any(h in msg for h in hints)

    def _order_cash_retry(self, max_retries: int = 3, backoff_base: float = 0.5, **kwargs) -> Dict[str, Any]:
        backoff = backoff_base
        last_res = None
        for attempt in range(1, max_retries + 1):
            res = self._order_cash_safe(**kwargs)
            last_res = res
            if res.get('ok'):
                return res
            if not self._is_transient_error(res):
                return res
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
        return last_res if last_res is not None else {'ok': False, 'rt_cd': None, 'msg1': 'no result'}

    # ── 쿨다운 보조: 연속 실패 누적 후에만 등록 ───────────────────────
    def _maybe_add_cooldown(self, ticker: str, reason: str, increment_fail: bool = True):
        if increment_fail:
            self._fail_counts[ticker] += 1
        else:
            # 성공 시 리셋
            self._fail_counts[ticker] = 0

        cnt = self._fail_counts[ticker]
        if cnt >= self.cooldown_fail_threshold:
            self._add_to_cooldown(ticker, f"{reason} (연속실패 {cnt}회)")
            # 등록 후 카운트 리셋
            self._fail_counts[ticker] = 0

    # ── 공용 시장가 매도 (리밸런싱 등) ────────────────────────────────
    def _execute_market_sell(self, ticker: str, quantity: int, name: str, reason_text: str, reason_code: str = "REBALANCE_SWAP") -> Dict[str, Any]:
        if quantity <= 0:
            return {"status": "sell_fail", "filled_qty": 0, "rt_cd": None, "msg1": "qty<=0"}

        logger.info(f"[REBALANCE] 매도 실행: {name}({ticker}) {quantity}주 | 사유={reason_text}")

        if self.is_real_trading:
            pre_qty = quantity

            result = self._order_cash_retry(
                ord_dv="01", pdno=ticker, ord_dvsn="01", ord_qty=str(quantity), ord_unpr="0"
            )

            time.sleep(2)
            self._update_account_info(force=True)
            _, holdings_after, _ = self._load_snapshot()
            post_qty = self._get_qty(holdings_after, ticker)
            filled_qty = max(0, pre_qty - post_qty)

            # 체결가 근사(시세 조회 실패 시 0 허용)
            try:
                price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                current_price = _to_int(price_df['stck_prpr'].iloc[0]) if (price_df is not None and not price_df.empty) else 0
            except Exception:
                current_price = 0

            # 결과 기록 및 쿨다운 처리
            if result.get('ok') or filled_qty > 0:
                trade_status = "completed" if filled_qty > 0 else "submitted"
                record_trade({
                    "side": "sell", "ticker": ticker, "name": name,
                    "qty": filled_qty if filled_qty > 0 else quantity,
                    "price": current_price,
                    "trade_status": trade_status,
                    "strategy_details": {
                        "reason": reason_text,
                        "reason_code": reason_code,
                        "broker_msg": result.get('msg1')
                    },
                    "sell_reason": reason_text
                })
                _notify_embed(create_trade_embed({
                    "side": "SELL", "name": name, "ticker": ticker,
                    "qty": filled_qty if filled_qty > 0 else quantity,
                    "price": current_price, "trade_status": trade_status,
                    "strategy_details": {"reason": reason_text, "reason_code": reason_code, "broker_msg": result.get('msg1')}
                }), key=f"phase:rebalance_sell:{ticker}", cooldown=30)

                # 응답 실패지만 체결 확인 포함 → 쿨다운 금지 & 실패 카운터 리셋
                self._maybe_add_cooldown(ticker, "매도 주문 실패", increment_fail=False)
                return {"status": "executed", "filled_qty": filled_qty, "rt_cd": result.get('rt_cd'), "msg1": result.get('msg1')}
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
                        "reason": reason_text,
                        "reason_code": reason_code
                    },
                    "sell_reason": reason_text
                })
                _notify_embed(create_trade_embed({
                    "side": "SELL", "name": name, "ticker": ticker,
                    "qty": quantity, "price": current_price, "trade_status": "failed",
                    "strategy_details": {"error": err, "reason_code": reason_code}
                }), key=f"phase:rebalance_sell_fail:{ticker}", cooldown=30)

                # 연속 실패 누적 → 기준치 도달 시에만 쿨다운
                self._maybe_add_cooldown(ticker, "리밸런스 매도 주문 실패", increment_fail=True)
                return {"status": "sell_fail", "filled_qty": 0, "rt_cd": result.get('rt_cd'), "msg1": result.get('msg1')}
        else:
            record_trade({
                "side": "sell", "ticker": ticker, "name": name,
                "qty": quantity, "price": 0, "trade_status": "completed",
                "strategy_details": {"reason": reason_text, "reason_code": reason_code},
                "sell_reason": reason_text
            })
            logger.info(f"  -> [모의] REBALANCE SELL {name}({ticker}) x{quantity}")
            _notify_text(
                f" [모의] REBALANCE SELL {name}({ticker}) x{quantity}",
                key=f"phase:paper_rebalance_sell:{ticker}", cooldown=30
            )
            return {"status": "executed", "filled_qty": quantity, "rt_cd": "0", "msg1": "paper"}

    # ── 점수 캐시 로드 ────────────────────────────────────────────────
    def _load_latest_scores(self) -> Tuple[Dict[str, float], Optional[str]]:
        f = find_latest_file("screener_scores_*.json")
        if not f:
            logger.info("점수 캐시 파일(screener_scores_*.json)을 찾지 못했습니다.")
            return {}, None
        try:
            with open(f, "r", encoding="utf-8") as fh:
                arr = json.load(f)
            if not isinstance(arr, list):
                return {}, None
            m: Dict[str, float] = {}
            for row in arr:
                t = str(row.get("ticker", "")).zfill(6)
                sc = _to_float(row.get("score_total"), 0.0)
                if t and t != "000000":
                    m[t] = sc
            logger.info("점수 캐시 로드: %s (tickers=%d)", f.name, len(m))
            return m, f.name
        except Exception as e:
            logger.warning("점수 캐시 로드 실패(%s): %s", f.name, e)
            return {}, None

    # ── 회전(교체) 시도 ───────────────────────────────────────────────
    def try_rotation(self, candidates: List[Dict[str, Any]], holdings: List[Dict[str, Any]], usable_cash: int) -> bool:
        """
        보유 최약체 ↔ 신규 최고 점수 후보 간 회전 시도.
        - SELL 확정(체결 확인) 이후에만 BUY
        - 동일 티커 중복 SELL 방지, 동일 페어 중복 시도 방지
        """
        if not candidates or not holdings:
            return False

        scores_map, _ = self._load_latest_scores()
        if not scores_map:
            logger.info("ROTATE 스킵: 점수 캐시 없음.")
            return False

        # 보유: (ticker, name, qty, score, prpr, proceeds)
        held: List[Tuple[str, str, int, float, int, int]] = []
        for h in holdings:
            t = str(h.get("pdno", "")).zfill(6)
            nm = h.get("prdt_name", "N/A")
            qty = _to_int(h.get("hldg_qty", 0))
            sc = float(scores_map.get(t, 0.0))
            prpr = _to_int(h.get("prpr", 0))
            proceeds = prpr * qty
            if qty > 0:
                held.append((t, nm, qty, sc, prpr, proceeds))
        if not held:
            return False

        # 후보: (ticker, name, price, score)
        cand_list: List[Tuple[str, str, int, float, Dict[str, Any]]] = []
        for c in candidates:
            t = str(c.get("Ticker", "")).zfill(6)
            nm = c.get("Name", "N/A")
            px = _to_int(c.get("Price", 0))
            sc = _to_float(c.get("Score", scores_map.get(t, 0.0)), 0.0)
            if t and px > 0:
                cand_list.append((t, nm, px, sc, c))

        if not cand_list:
            return False

        # 정렬
        held_sorted = sorted(held, key=lambda x: x[3])               # 보유: 점수 낮은 것 우선
        cand_sorted = sorted(cand_list, key=lambda x: x[3], reverse=True)  # 후보: 점수 높은 것 우선

        buffer = float(self.trading_params.get("cash_buffer_ratio", 0.0))
        delta_thr = float(self.settings.get("rotation", {}).get("delta_score_min", 0.10))

        for wt, wname, wqty, wscore, wprpr, wproc in held_sorted:
            if wt in self._sold_once:
                continue
            if self._is_in_cooldown(wt):
                logger.info(f"ROTATE_TRY | sell={wname}({wt}) 쿨다운 중 → 스킵")
                continue

            for ct, cname, cprice, cscore, crow in cand_sorted:
                if ct == wt:
                    continue
                if (wt, ct) in self._attempted_pairs:
                    continue
                if self._is_in_cooldown(ct):
                    logger.info(f"ROTATE_TRY | buy={cname}({ct}) 쿨다운 중 → 스킵")
                    continue

                affordable_budget = int((usable_cash + wproc) * (1 - buffer))
                affordable = cprice <= affordable_budget
                delta = float(cscore) - float(wscore)

                logger.info(
                    f"ROTATE_TRY | sell={wname}({wt})[{wscore:.3f}] qty={wqty} prpr={wprpr:,} "
                    f"| buy={cname}({ct})[{cscore:.3f}] px={cprice:,} "
                    f"| delta=+{delta:.3f} thr={delta_thr:.3f} affordable={'YES' if affordable else 'NO'}"
                )

                self._attempted_pairs.add((wt, ct))

                if delta < delta_thr or not affordable:
                    continue

                # SELL 먼저, 확정 시에만 BUY
                sell_res = self._execute_market_sell(wt, wqty, wname, reason_text=f"REBALANCE_SWAP → {ct}", reason_code="REBALANCE_SWAP")
                if sell_res.get("status") != "executed":
                    logger.info(f"[ROTATION] SELL 확정 실패 → BUY 스킵 (pair {wt}->{ct})")
                    # 실패 누적은 _execute_market_sell 내부에서 처리
                    continue

                # 동일 티커 반복 SELL 방지
                self._sold_once.add(wt)

                # 매도 후 스냅샷 강제 갱신 → 최신 현금으로 BUY 판단
                time.sleep(3)
                self._update_account_info(force=True)
                new_cash, new_holdings, _ = self._load_snapshot()

                # 단일 매수 시도
                bought = self._execute_buy_single(crow, new_cash, batch_name="ROTATION")
                return bool(bought)

        return False

    # ── 단일 매수 실행(회전/특수 케이스용) ────────────────────────────
    def _execute_buy_single(self, stock_info: Dict[str, Any], available_cash: int, batch_name: str = "SINGLE") -> bool:
        name = stock_info.get("Name", "N/A")
        ticker = str(stock_info.get("Ticker", "")).zfill(6)
        if not ticker:
            return False

        # 예산 계산 (버퍼 적용)
        buffer = float(self.trading_params.get("cash_buffer_ratio", 0.0))
        budget = int(available_cash * (1 - buffer))
        if budget < 1:
            logger.info(f"[{batch_name}] 예산 부족으로 매수 불가: {name}({ticker}), budget={budget:,}")
            return False

        # 현재가 조회 (실패 시 stock_info 가격 사용)
        current_price = _to_int(stock_info.get("Price", 0))
        if current_price <= 0:
            try:
                price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                if price_df is not None and not price_df.empty and 'stck_prpr' in price_df.columns:
                    current_price = _to_int(price_df['stck_prpr'].iloc[0])
            except Exception:
                current_price = 0
        if current_price <= 0:
            logger.info(f"[{batch_name}] 현재가 조회 실패: {name}({ticker})")
            return False

        tick = get_tick_size(current_price)
        raw_price = current_price + (tick * random.randint(1, 3))
        order_price = round_to_tick(raw_price, mode="up")
        qty = int(budget // order_price)

        # 최소 주문 금액 판정은 **금액 기준**으로 (주가와 비교 ❌)
        if qty <= 0 or (self.min_order_cash > 0 and (qty * order_price) < self.min_order_cash):
            logger.info(f"[{batch_name}] 예산/최소주문 조건 불충족: {name}({ticker}) qty={qty}, "
                        f"price={order_price:,}, spent={qty*order_price:,}, min_order_cash={self.min_order_cash:,}")
            return False

        logger.info(f"[{batch_name}] BUY {name}({ticker}) x{qty} @ {order_price:,}")
        self.stats["buy"] += 1

        if self.is_real_trading:
            res = self._order_cash_retry(
                ord_dv="02", pdno=ticker, ord_dvsn="00", ord_qty=str(qty), ord_unpr=str(int(order_price))
            )
            time.sleep(2)
            self._update_account_info(force=True)
            if not res.get("ok"):
                err = res.get("msg1", "Unknown error")
                record_trade({
                    "side": "buy", "ticker": ticker, "name": name,
                    "qty": qty, "price": order_price, "trade_status": "failed",
                    "strategy_details": {"error": err, "batch": batch_name}
                })
                _notify_embed(create_trade_embed({
                    "side": "BUY", "name": name, "ticker": ticker,
                    "qty": qty, "price": order_price, "trade_status": "failed",
                    "strategy_details": {"error": err, "batch": batch_name}
                }), key=f"phase:buy_fail:{ticker}", cooldown=30)
                # 연속 실패 누적 → 기준 도달 시에만 쿨다운
                self._maybe_add_cooldown(ticker, "매수 주문 실패", increment_fail=True)
                return False

            record_trade({
                "side": "buy", "ticker": ticker, "name": name,
                "qty": qty, "price": order_price, "trade_status": "submitted",
                "strategy_details": {"broker_msg": res.get('msg1'), "batch": batch_name}
            })
            _notify_embed(create_trade_embed({
                "side": "BUY", "name": name, "ticker": ticker,
                "qty": qty, "price": order_price, "trade_status": "submitted",
                "strategy_details": {"broker_msg": res.get('msg1'), "batch": batch_name}
            }), key=f"phase:buy_single:{ticker}", cooldown=30)
            # 성공 시 실패카운트 리셋
            self._maybe_add_cooldown(ticker, "매수 주문 실패", increment_fail=False)
            return True
        else:
            record_trade({
                "side": "buy", "ticker": ticker, "name": name,
                "qty": qty, "price": order_price, "trade_status": "completed",
                "strategy_details": {"batch": batch_name}
            })
            _notify_text(
                f" [모의] BUY {name}({ticker}) x{qty} @ {order_price:,} [{batch_name}]",
                key=f"phase:paper_buy_single:{ticker}", cooldown=30
            )
            return True

    # ── 매도 로직 ────────────────────────────────────────────────────
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

        now_kst = datetime.now(KST)
        if not in_time_windows(now_kst, self.sell_time_windows):
            logger.info(f"현재 시간 {now_kst.strftime('%H:%M')}은 매도 시간대가 아닙니다. sell_time_windows={self.sell_time_windows}")
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

            # 실시간 레벨/RSI/Price 오버레이
            try:
                cur_price = 0
                try:
                    price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                    if price_df is not None and not price_df.empty and 'stck_prpr' in price_df.columns:
                        cur_price = _to_int(price_df['stck_prpr'].iloc[0])
                except Exception:
                    cur_price = 0
                if cur_price <= 0:
                    cur_price = _to_int(stock_info.get("Price", 0))
                if cur_price > 0:
                    stock_info["Price"] = cur_price

                rt_levels = self.risk_manager.compute_realtime_levels(ticker, cur_price) or {}
                if "RSI" in rt_levels and rt_levels["RSI"] is not None:
                    stock_info["RSI"] = float(rt_levels["RSI"])
                if "손절가" in rt_levels and rt_levels["손절가"] is not None:
                    stock_info["손절가"] = _to_int(rt_levels["손절가"])
                if "목표가" in rt_levels and rt_levels["목표가"] is not None:
                    stock_info["목표가"] = _to_int(rt_levels["목표가"])
                if "Price" in rt_levels and rt_levels["Price"] is not None:
                    stock_info["Price"] = _to_int(rt_levels["Price"])
                if "source" in rt_levels and rt_levels["source"]:
                    stock_info["source"] = rt_levels["source"]
            except Exception as e:
                logger.debug(f"[{ticker}] 실시간 레벨 오버레이 실패: {e}")

            decision, reason = self.risk_manager.check_sell_condition(holding, stock_info)
            if decision != "SELL":
                logger.info(f"유지 판단: {reason}")
                self.stats["hold"] += 1
                continue

            reason_code = self._parse_reason_code(reason)
            logger.info(f"매도 결정: {name}({ticker}) {quantity}주. 사유: {reason} | code={reason_code}")
            executed_sell = True

            if self.is_real_trading:
                pre_qty = self._get_qty(holdings, ticker)

                result = self._order_cash_retry(
                    ord_dv="01", pdno=ticker, ord_dvsn="01", ord_qty=str(quantity), ord_unpr="0"
                )

                time.sleep(2)
                self._update_account_info(force=True)
                _, holdings_after, _ = self._load_snapshot()
                post_qty = self._get_qty(holdings_after, ticker)
                filled_qty = max(0, pre_qty - post_qty)

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

                if result.get('ok') or filled_qty > 0:
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

                    # 성공/체결확인 → 실패카운트 리셋
                    self._maybe_add_cooldown(ticker, "매도 주문 실패", increment_fail=False)

                    if filled_qty == 0:
                        logger.info("  -> 응답은 성공이나 즉시 체결 없음(미체결 가능). submitted로 기록.")
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
                        "qty": quantity, "price": current_price, "trade_status": "failed",
                        "strategy_details": {"error": err, "reason_code": reason_code}
                    }), key=f"phase:sell_fail:{ticker}", cooldown=30)
                    # 연속 실패 누적 → 기준 도달 시에만 쿨다운
                    self._maybe_add_cooldown(ticker, "매도 주문 실패", increment_fail=True)

            else:
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
            self._update_account_info(force=True)

    # ── 매수 로직 ─────────────────────────────────────────────────────
    def run_buy_logic(self, available_cash: int, holdings: List[Dict]):
        logger.info(f"--------- 신규/추가 매수 로직 실행 (가용 예산: {available_cash:,} 원) ---------")

        now_kst = datetime.now(KST)
        if not in_time_windows(now_kst, self.buy_time_windows):
            logger.info(f"현재 시간 {now_kst.strftime('%H:%M')}은 매수 시간대가 아닙니다. buy_time_windows={self.buy_time_windows}")
            _notify_text("ℹ️ 매수 시간대 외 → 매수 스킵",
                         key="phase:buy_out_of_window", cooldown=300)
            return

        # 동적 슬롯 축소
        if self.trading_guards.get("auto_shrink_slots", False):
            eff_slots_cap = int(self.trading_params.get("max_positions", self.max_positions))
            eff_slots_by_cash = max(available_cash // max(1, self.min_order_cash), 0) if self.min_order_cash > 0 else eff_slots_cap
            effective_slots = min(eff_slots_cap, eff_slots_by_cash)
        else:
            effective_slots = self.max_positions

        # GPT 추천 계획 로드
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

        # 1차 후보(가성비) 판단: **주가 vs (현금×(1-버퍼))**만 적용
        buffer = float(self.trading_params.get("cash_buffer_ratio", 0.0))
        min_order = int(self.trading_params.get("min_order_cash", 0))

        candidates_all = list(self.all_stock_data.values())
        affordable = [c for c in candidates_all if _to_int(c.get("Price", 0)) <= int(available_cash * (1 - buffer))]

        # 통계 로그(참고용) — 필터에는 min_order_cash를 사용하지 않음
        log_affordability_stats(available_cash, buffer, candidates_all, min_order_cash=min_order)

        if self.screener_params.get("affordability_filter", False) and not affordable:
            # LOW_FUNDS 전에 회전 시도
            rotated = self.try_rotation(candidates_all, holdings, available_cash)
            if not rotated:
                cheapest = min((_to_int(c.get("Price", 0)) for c in candidates_all), default=0)
                self._set_summary_reason("SKIPPED_LOW_FUNDS_NO_ROTATION",
                                         f"cheapest={cheapest:,} cash={available_cash:,} buffer={buffer:.2%} min_order_cash={min_order:,}")
                logger.info("가용 예산 부족 & 회전 실패 → 매수 종료.")
                return
            else:
                # 회전 성공 시 현금/보유 최신화 후 계속 신규/추가 매수 진행
                time.sleep(3)
                self._update_account_info(force=True)
                available_cash, holdings, _ = self._load_snapshot()

        # 보유 집합
        holding_tickers = {str(h.get("pdno", "")).zfill(6) for h in holdings if _to_int(h.get("hldg_qty", 0)) > 0}

        # 후보 분리: 신규 / 추가매수
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

        remaining_cash = available_cash
        any_order_placed = False

        def _execute_buy_batch(plans: List[Dict], batch_name: str):
            nonlocal remaining_cash, any_order_placed
            if not plans:
                return
            if remaining_cash <= max(self.min_order_cash, self.min_cash_reserve):
                logger.info(f"잔여 현금이 최소치 이하({remaining_cash:,}원)로 {batch_name} 스킵.")
                return

            _notify_text(f" {batch_name} 매수 시도 {len(plans)}종목 (예산 {remaining_cash:,}원)",
                         key=f"phase:{batch_name.lower()}_start", cooldown=120)
            logger.info(f"총 {len(plans)}개 종목 {batch_name} 매수 시도. 유동적 예산 배분 + 버퍼 적용.")

            for i, plan in enumerate(plans):
                info = plan["stock_info"]
                ticker, name = str(info["Ticker"]).zfill(6), info.get("Name", "N/A")
                slots_left = len(plans) - i

                slot_cash = remaining_cash // max(1, slots_left if effective_slots <= 0 else min(slots_left, effective_slots))
                budget_for_this_stock = int(slot_cash * (1 - buffer))
                logger.info(f"  -> [{i+1}/{len(plans)}] {name}({ticker}) 배분 예산: {budget_for_this_stock:,.0f}원")

                current_price = _to_int(info.get("Price", 0))
                if current_price == 0:
                    try:
                        price_df = self.kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd=ticker)
                        if price_df is not None and not price_df.empty and 'stck_prpr' in price_df.columns:
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

                tick_size = get_tick_size(current_price)
                raw_price = current_price + (tick_size * random.randint(1, 3))
                order_price = round_to_tick(raw_price, mode="up")
                quantity = int(budget_for_this_stock // order_price)

                # 최소 주문 금액은 **수량×가격**으로 판정
                if quantity == 0 or (self.min_order_cash > 0 and (quantity * order_price) < self.min_order_cash):
                    logger.info(f"  -> [{name}({ticker})] 예산/최소주문 미충족. "
                                f"price={order_price:,}, budget={budget_for_this_stock:,}, "
                                f"qty={quantity}, spent={quantity*order_price:,}, min_order_cash={self.min_order_cash:,}")
                    _notify_embed(create_trade_embed({
                        "side": "BUY", "name": name, "ticker": ticker,
                        "qty": 0, "price": order_price, "trade_status": "skipped",
                        "strategy_details": {
                            "reason_code": "INSUFFICIENT_CASH",
                            "required": int(max(order_price, self.min_order_cash)),
                            "available": int(budget_for_this_stock),
                            "batch": batch_name
                        }
                    }), key=f"phase:buy_insufficient:{ticker}", cooldown=120)
                    continue

                pre_qty = self._get_qty(holdings, ticker)
                logger.info(f"  -> 매수 준비: {name}({ticker}), 수량: {quantity}주, 지정가: {order_price:,.0f}원 [{batch_name}]")

                if self.is_real_trading:
                    result = self._order_cash_retry(
                        ord_dv="02", pdno=ticker, ord_dvsn="00", ord_qty=str(quantity), ord_unpr=str(int(order_price))
                    )

                    time.sleep(2)
                    self._update_account_info(force=True)
                    new_cash, holdings_after, _ = self._load_snapshot()
                    post_qty = self._get_qty(holdings_after, ticker)
                    qty_delta = max(0, post_qty - pre_qty)

                    if result.get('ok') or qty_delta > 0:
                        self.stats["buy"] += 1
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
                            remaining_cash = new_cash
                        any_order_placed = True
                        _notify_embed(create_trade_embed({
                            "side": "BUY", "name": name, "ticker": ticker,
                            "qty": qty_delta if qty_delta > 0 else quantity,
                            "price": order_price, "trade_status": trade_status,
                            "strategy_details": {"broker_msg": result.get('msg1'), "batch": batch_name}
                        }), key=f"phase:buy:{ticker}", cooldown=30)
                        # 성공/체결확인 → 실패 카운트 리셋
                        self._maybe_add_cooldown(ticker, "매수 주문 실패", increment_fail=False)
                    else:
                        err = result.get('msg1', 'Unknown error')
                        record_trade({
                            "side": "buy", "ticker": ticker, "name": name,
                            "qty": quantity, "price": order_price, "trade_status": "failed",
                            "strategy_details": {
                                "error": err,
                                "rt_cd": result.get('rt_cd'),
                                "msg_cd": result.get('msg_cd'),
                                "reason_code": "BROKER_REJECT",
                                "batch": batch_name
                            },
                            "gpt_analysis": plan
                        })
                        logger.info(f"  -> [{name}({ticker})] 브로커 거절. 남은 종목에 예산 재배분.")
                        _notify_embed(create_trade_embed({
                            "side": "BUY", "name": name, "ticker": ticker, "qty": quantity,
                            "price": order_price, "trade_status": "failed",
                            "strategy_details": {"error": err, "reason_code": "BROKER_REJECT", "batch": batch_name}
                        }), key=f"phase:buy_fail:{ticker}", cooldown=30)
                        # 연속 실패 누적 → 기준 도달 시에만 쿨다운
                        self._maybe_add_cooldown(ticker, "매수 주문 실패", increment_fail=True)
                else:
                    actual_spent = quantity * order_price
                    remaining_cash -= actual_spent
                    any_order_placed = True
                    self.stats["buy"] += 1
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

        # 1) 추가매수 먼저
        _execute_buy_batch(rebuy_candidates, batch_name="REBUY")

        # 2) 신규 진입: 슬롯 확인 (동적 슬롯 반영)
        current_slots_used = len({str(h.get("pdno", "")).zfill(6) for h in holdings if _to_int(h.get("hldg_qty", 0)) > 0})
        slots_to_fill = max(0, effective_slots - current_slots_used)

        if slots_to_fill <= 0 and new_targets:
            # 보유 최저 점수 vs 신규 후보 비교 → 교체(리밸런싱)
            to_buy_plans, to_sell_list = self._determine_rebalance_swaps(new_targets, holdings)
            if to_sell_list:
                _notify_text(f" 리밸런싱 매도 {len(to_sell_list)}건 실행", key="phase:rebalance_sell_batch", cooldown=120)
                for s in to_sell_list:
                    # SELL 하드 게이트
                    sell_res = self._execute_market_sell(
                        ticker=s["ticker"],
                        quantity=s["qty"],
                        name=s["name"],
                        reason_text=f"REBALANCE_SWAP (old={s['old_score']:.3f} → new={s['new_score']:.3f} for {s['new_ticker']})",
                        reason_code="REBALANCE_SWAP"
                    )
                    time.sleep(0.5)

                time.sleep(3)
                self._update_account_info(force=True)
                new_cash, holdings_after, _ = self._load_snapshot()
                slots_now = max(0, effective_slots - len(holdings_after))
                if slots_now > 0:
                    buy_now = to_buy_plans[:slots_now]
                    _execute_buy_batch(buy_now, batch_name="REBALANCE_NEW")
            else:
                logger.info("리밸런싱 기준을 충족하는 교체 대상이 없어 신규 매수는 생략합니다.")

        else:
            if slots_to_fill <= 0:
                logger.info(f"신규 매수 슬롯이 없습니다 (effective_slots={effective_slots}, 현재보유={current_slots_used}).")
                _notify_text(f"ℹ️ 신규 매수 슬롯 없음 (effective_slots={effective_slots}, curr={current_slots_used})",
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
            self._update_account_info(force=True)

    # ── 리밸런싱 페어링 (점수 기반) ───────────────────────────────────
    def _determine_rebalance_swaps(self, new_targets: List[Dict], holdings: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        scores_map, score_file = self._load_latest_scores()
        if not scores_map:
            logger.info("리밸런싱 건너뜀: 점수 캐시 없음.")
            return [], []

        held: List[Tuple[str, str, int, float]] = []
        for h in holdings:
            t = str(h.get("pdno", "")).zfill(6)
            nm = h.get("prdt_name", "N/A")
            qty = _to_int(h.get("hldg_qty", 0))
            sc = float(scores_map.get(t, 0.0))
            if qty > 0:
                held.append((t, nm, qty, sc))
        if not held:
            return [], []

        new_list: List[Tuple[float, Dict]] = []
        for plan in new_targets:
            info = plan.get("stock_info", {})
            sc = _to_float(info.get("Score"), 0.0)
            new_list.append((float(sc), plan))

        if not new_list:
            return [], []

        held_sorted = sorted(held, key=lambda x: x[3])
        new_sorted = sorted(new_list, key=lambda x: x[0], reverse=True)

        to_buy: List[Dict] = []
        to_sell: List[Dict] = []

        hi = 0
        for new_score, plan in new_sorted:
            if hi >= len(held_sorted):
                break
            worst_t, worst_name, worst_qty, worst_score = held_sorted[hi]
            if new_score > worst_score:
                to_buy.append(plan)
                to_sell.append({
                    "ticker": worst_t,
                    "name": worst_name,
                    "qty": worst_qty,
                    "old_score": worst_score,
                    "new_score": new_score,
                    "new_ticker": str(plan.get("stock_info", {}).get("Ticker", "")).zfill(6),
                })
                hi += 1
        if to_buy and to_sell:
            pairs = len(to_buy)
            msg_lines = [f"점수 기반 리밸런싱 매칭 {pairs}건"]
            for i in range(pairs):
                s = to_sell[i]
                new_plan = to_buy[i]
                nt = str(new_plan.get('stock_info', {}).get('Ticker', '')).zfill(6)
                nn = new_plan.get('stock_info', {}).get('Name', 'N/A')
                msg_lines.append(
                    f"- SELL {s['name']}({s['ticker']}) [{s['old_score']:.3f}]  →  BUY {nn}({nt}) [{s['new_score']:.3f}]"
                )
            _notify_text(" " + "\n".join(msg_lines), key="phase:rebalance_pairs", cooldown=120)
        else:
            logger.info("리밸런싱 조건을 만족하는 신규 후보가 없습니다.")
        return to_buy, to_sell

    # ── 쿨다운 관리 ───────────────────────────────────────────────────
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

    # ── 코히어런트 요약 ──────────────────────────────────────────────
    def _set_summary_reason(self, code: str, detail: str = ""):
        self.summary_reason_code = code
        self.summary_reason_detail = detail

    def emit_final_summary(self, start_ts: float, status: str = "SUCCESS", warnings: int = 0):
        duration = int(time.time() - start_ts)
        reason = self.summary_reason_code or "N/A"
        detail = (f" | {self.summary_reason_detail}" if self.summary_reason_detail else "")
        line1 = f"RUN: {status} | WARNINGS: {warnings} | DURATION: {duration}s"
        line2 = f"TRADES: {self.stats['buy']} buy / {self.stats['sell']} sell / {self.stats['hold']} hold | REASON: {reason}{detail}"
        logger.info(line1)
        logger.info(line2)
        if self.reporting.get("coherent_summary", False):
            _notify_text(f"✅ 파이프라인 요약\n{line1}\n{line2}", key="phase:summary", cooldown=60)

# ── 엔트리포인트 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    start_ts = time.time()
    try:
        trader = Trader(settings)

        # 최신 계좌 스냅샷 생성(파일 갱신) 후 파일에서 로드
        trader._update_account_info(force=True)
        cash0, holdings0, _ = trader.get_account_info_from_files()

        # 세션 시작 가드
        usable_cash = cash0
        if trader.reporting.get("include_cash_breakdown", False):
            logger.info(f"usable_cash={usable_cash:,}")
        if trader.trading_guards.get("skip_when_low_funds", False) and \
           usable_cash < int(trader.trading_guards.get("min_total_cash_to_trade", 0)):
            trader._set_summary_reason(
                "SKIPPED_LOW_FUNDS_SESSION",
                f"cash {usable_cash:,} < min_total_cash_to_trade {int(trader.trading_guards.get('min_total_cash_to_trade', 0)):,}"
            )
            trader.emit_final_summary(start_ts, status="SUCCESS", warnings=0)
            sys.exit(0)

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

        # 최종 요약
        trader.emit_final_summary(start_ts, status="SUCCESS", warnings=0)

    except Exception as e:
        logger.critical(f"트레이더 실행 중 심각한 오류 발생: {e}", exc_info=True)
        _notify_text(f" 트레이더 치명적 오류: {str(e)[:1800]}",
                     key="phase:fatal", cooldown=30)
        try:
            trader._set_summary_reason("FATAL_EXCEPTION", str(e))
            trader.emit_final_summary(start_ts, status="FAILED", warnings=1)
        except Exception:
            pass
