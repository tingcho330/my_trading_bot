# src/notifier.py
import os
import time
import re
import logging
import threading
from typing import Optional, List, Dict, Any

import httpx

# ────────────────────────────────────────────────────────────────────
# 기본 설정
# ────────────────────────────────────────────────────────────────────
WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# httpx 타임아웃 (connect/read/write 개별 설정)
HTTPX_TIMEOUT = httpx.Timeout(10.0, connect=5.0, read=5.0, write=5.0)

DEFAULT_TIMEOUT = 7.0
MAX_RETRIES = 2              # 429/일시 오류 재시도 횟수
BASE_BACKOFF = 0.6           # 지수 백오프 시작(초)
MIN_INTERVAL = 0.75          # 메시지 최소 간격(초) - 러프한 토큰버킷

# noisy 로거 셋 (emit에서 필터링)
_NOISY_LOGGERS = {"httpx", "httpcore", "urllib3"}

# emit 재진입 가드용 thread-local
_emit_local = threading.local()

# 내부 전용 로거(핸들러 재귀 방지 위해 사용 최소화: 기본적으로 찍지 않음)
_logger = logging.getLogger("notifier")
_logger.propagate = False  # 상위(root)로 전파 금지

# 전역 HTTP 클라이언트
_client = httpx.Client(timeout=HTTPX_TIMEOUT, headers={"Content-Type": "application/json"})

# 간단 레이트리밋(최소 간격)
_last_sent_ts = 0.0

# ────────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────────
_WEBHOOK_RE = re.compile(r"^https://discord\.com/api/webhooks/\d+/[A-Za-z0-9_\-]+$")

def is_valid_webhook(url: Optional[str]) -> bool:
    # 간결 체크(사용자 제안) + 정규식 강화 체크 병행
    if not url:
        return False
    if not url.startswith("https://discord.com/api/webhooks/"):
        return False
    return bool(_WEBHOOK_RE.match(url))

# ────────────────────────────────────────────────────────────────────
# 디스코드 전송
#  - content 없이 embeds만으로도 전송 가능
#  - 429/일시 오류 백오프 재시도
#  - 내부에서 logging 호출하지 않아 핸들러 재귀 방지
# ────────────────────────────────────────────────────────────────────
def send_discord_message(
    content: Optional[str] = None,
    embeds: Optional[List[Dict[str, Any]]] = None,
    username: Optional[str] = None,
) -> None:
    """Discord Webhook 전송. 실패는 조용히 무시(로깅 재귀 방지)."""
    global _last_sent_ts
    if not is_valid_webhook(WEBHOOK_URL):
        return

    # 최소 간격 레이트리밋
    now = time.time()
    gap = now - _last_sent_ts
    if gap < MIN_INTERVAL:
        try:
            time.sleep(MIN_INTERVAL - gap)
        except Exception:
            # sleep 중 인터럽트 등은 무시
            pass

    payload: Dict[str, Any] = {}
    if content:
        # 디스코드 content 최대 2000자, 여유 있게 1900자로 제한
        payload["content"] = str(content)[:1900]
    if embeds:
        # embed 10개 제한 고려(보수적으로 5개)
        payload["embeds"] = embeds[:5]
    if username:
        payload["username"] = username

    if not payload.get("content") and not payload.get("embeds"):
        # 아무 것도 없으면 전송 안 함
        return

    backoff = BASE_BACKOFF
    for attempt in range(0, MAX_RETRIES + 1):
        try:
            resp = _client.post(WEBHOOK_URL, json=payload)
            if resp.status_code == 204:
                _last_sent_ts = time.time()
                return
            if resp.status_code == 429:
                # 디스코드 레이트리밋 헤더 기반 대기
                retry_after = 0.0
                try:
                    # 우선순위: Retry-After(초) → X-RateLimit-Reset-After(초)
                    retry_after = float(resp.headers.get("Retry-After", "0"))
                except Exception:
                    retry_after = 0.0
                if retry_after <= 0:
                    retry_after = backoff
                    backoff *= 2
                try:
                    time.sleep(min(10.0, max(0.5, retry_after)))
                except Exception:
                    pass
                continue  # 재시도
            # 5xx/일시적 장애: 지수 백오프 후 재시도
            if 500 <= resp.status_code < 600 and attempt < MAX_RETRIES:
                try:
                    time.sleep(backoff)
                except Exception:
                    pass
                backoff *= 2
                continue
            # 이외 오류는 조용히 중단
            return
        except Exception:
            # 네트워크 예외 등: 지수 백오프 재시도
            if attempt < MAX_RETRIES:
                try:
                    time.sleep(backoff)
                except Exception:
                    pass
                backoff *= 2
                continue
            return

# ────────────────────────────────────────────────────────────────────
# 로깅 핸들러
#  - noisy 로거 무시
#  - emit 내에서 어떤 로거도 호출하지 않음(재귀 방지)
#  - 재진입 가드(thread-local) 적용
#  - content-only 간결 메시지 전송
# ────────────────────────────────────────────────────────────────────
class DiscordLogHandler(logging.Handler):
    def __init__(self, webhook_url: Optional[str] = None, level=logging.ERROR):
        super().__init__(level=level)
        self.webhook_url = webhook_url or WEBHOOK_URL

    def emit(self, record: logging.LogRecord) -> None:
        # noisy 로거 무시
        if record.name in _NOISY_LOGGERS:
            return

        # 재진입 가드
        if getattr(_emit_local, "emitting", False):
            return

        if not is_valid_webhook(self.webhook_url):
            return

        # 너무 긴 메시지 방어 및 포맷
        try:
            msg = self.format(record)
        except Exception:
            try:
                msg = record.getMessage()
            except Exception:
                msg = ""

        # 스택트레이스 등은 길어질 수 있으므로 content에만 담고 잘라냄
        text = msg[:1900] if msg else ""

        # 절대 logging 호출 금지 (여기서 다시 로깅하면 재귀됨)
        try:
            _emit_local.emitting = True
            if text:
                send_discord_message(content=text)
        finally:
            _emit_local.emitting = False

# ────────────────────────────────────────────────────────────────────
# 주문/체결용 임베드 포맷터
# ────────────────────────────────────────────────────────────────────
def create_trade_embed(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload keys (예시):
      side: "BUY"|"SELL"
      name: "삼성전자"
      ticker: "005930"
      qty: 10
      price: 71200
      trade_status: "submitted"|"completed"|"failed"|"skipped"
      strategy_details: { ... 임의 ... }
    """
    side = str(payload.get("side", "?")).upper()
    name = payload.get("name", "N/A")
    ticker = str(payload.get("ticker", "N/A")).zfill(6)
    qty = payload.get("qty", 0)
    price = payload.get("price", 0)
    trade_status = payload.get("trade_status", "submitted")

    fields = [
        {"name": "티커", "value": f"`{ticker}`", "inline": True},
        {"name": "수량", "value": f"{qty}", "inline": True},
        {"name": "가격", "value": f"{price:,}", "inline": True},
        {"name": "상태", "value": f"{trade_status}", "inline": True},
    ]

    details = payload.get("strategy_details") or {}
    if details:
        # 긴 dict를 깔끔히 보여주기 위해 최대 900자
        try:
            import json as _json
            pretty = _json.dumps(details, ensure_ascii=False, indent=2)
            fields.append({"name": "Details", "value": f"```json\n{pretty[:900]}\n```", "inline": False})
        except Exception:
            fields.append({"name": "Details", "value": str(details)[:900], "inline": False})

    return {
        "type": "rich",
        "title": f" BUY {name}" if side == "BUY" else f" SELL {name}",
        "description": f"{name} ({ticker})",
        "fields": fields,
    }
