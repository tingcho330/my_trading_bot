# src/notifier.py
import os
import requests
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from urllib.parse import urlparse
import threading

# ───────────────── 로깅 설정 ─────────────────
# 루트 로거에 기본 핸들러 설정 (다른 모듈의 로그도 수집됨)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 이 모듈 전용 로거 (이 이름을 핸들러에서 루프 방지 키로 사용)
logger = logging.getLogger("notifier")

# ───────────────── .env 로딩 (고정 경로 + 폴백) ─────────────────
def load_env_with_fallback() -> str:
    """
    /app/config/.env 우선 → 파일 기준 후보 → CWD 후보 → find_dotenv 순으로 탐색.
    로드 성공 시 경로 문자열을 반환, 없으면 빈 문자열 반환.
    """
    candidates = [
        Path("/app/config/.env"),                                    # 절대 경로 우선
        Path(__file__).resolve().parents[1] / "config" / ".env",     # .../src → /config/.env
        Path(__file__).resolve().parent / "config" / ".env",         # 현재 폴더 하위 config/.env
        Path(__file__).resolve().parent / ".env",                    # 현재 폴더 .env
        Path.cwd() / "config" / ".env",                              # CWD/config/.env
        Path.cwd() / ".env",                                         # CWD/.env
    ]

    loaded = ""
    for p in candidates:
        try:
            if p.is_file():
                if load_dotenv(dotenv_path=p, override=False):
                    loaded = str(p)
                    break
        except Exception:
            continue

    if not loaded:
        try:
            found = find_dotenv(usecwd=True)
            if found:
                load_dotenv(found, override=False)
                loaded = found
        except Exception:
            pass

    logger.info(f".env loaded from: {loaded if loaded else 'None'}")
    return loaded

_ = load_env_with_fallback()
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# ───────────────── 유틸: 웹훅 URL 검증 ─────────────────
def is_valid_webhook(url: str) -> bool:
    """
    Discord 웹훅 URL 형식 검증:
    - 스킴: http/https
    - 도메인: discord.com 또는 discordapp.com
    - 경로: /api/webhooks/ 포함
    """
    try:
        if not url:
            return False
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = (parsed.netloc or "").lower()
        if not (host.endswith("discord.com") or host.endswith("discordapp.com")):
            return False
        if "/api/webhooks/" not in parsed.path:
            return False
        return True
    except Exception:
        return False

# ───────────────── 디스코드 전송 ─────────────────
def send_discord_message(content: Optional[str] = None, embeds: Optional[List[Dict]] = None, *, _silent: bool = False) -> None:
    """
    디스코드로 메시지 전송. notifier 로거를 사용해 루프를 회피하고,
    잘못된 URL은 경고만 남기고 리턴.
    _silent=True 이면 실패 로그를 남기지 않음(핸들러 내부용).
    """
    if not is_valid_webhook(WEBHOOK_URL):
        if not _silent:
            logger.warning("DISCORD_WEBHOOK_URL이 유효하지 않습니다. (예: https://discord.com/api/webhooks/...)")
        return

    payload: Dict[str, object] = {}
    if content:
        # 디스코드 메시지 길이 제한(2000자) 대응
        payload["content"] = content[:2000]
    if embeds:
        payload["embeds"] = embeds

    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        if not _silent:
            logger.info("디스코드 메시지 전송 성공!")
    except requests.RequestException as e:
        if not _silent:
            # 루트 로거가 아닌 notifier 로거를 사용 -> DiscordLogHandler가 자체적으로 무시
            logger.error(f"디스코드 메시지 전송 실패: {e}")

# ───────────────── 임베드 빌더 ─────────────────
def create_trade_embed(trade_info: Dict) -> Dict:
    """매매 내역 정보를 받아 디스코드 임베드 형식으로 만듭니다."""
    side = (trade_info.get('side') or 'N/A').upper()
    status = (trade_info.get('trade_status') or 'N/A').lower()

    # 주문 상태에 따른 색상 및 제목
    if status == 'failed':
        color = 16711680  # 빨강
        title = f"❌ 주문 실패: {side}"
    elif side == 'SELL':
        color = 15105570  # SELL 톤
        title = f"🔔 주문 실행 알림: {side}"
    else:
        color = 3066993   # BUY 톤
        title = f"🔔 주문 실행 알림: {side}"

    fields = [
        {"name": "종목명", "value": str(trade_info.get('name', 'N/A')), "inline": True},
        {"name": "티커", "value": str(trade_info.get('ticker', 'N/A')), "inline": True},
        {"name": "주문 수량", "value": str(trade_info.get('qty', 0)), "inline": False},
        {"name": "주문 가격", "value": f"{trade_info.get('price', 0):,.0f} 원", "inline": True},
        {"name": "주문 상태", "value": status.capitalize(), "inline": True},
    ]

    # 실패 사유 추가
    strategy_details = trade_info.get('strategy_details', {})
    if status == 'failed' and isinstance(strategy_details, dict) and strategy_details.get('error'):
        err_text = str(strategy_details['error'])
        # 코드블럭 길이 제한을 고려해 슬라이스
        fields.append({"name": "실패 사유", "value": f"```{err_text[:1800]}```", "inline": False})

    embed = {
        "title": title,
        "color": color,
        "fields": fields,
        "footer": {"text": "AI Trading Bot"}
    }
    return embed

# ───────────────── 로그 → 디스코드 핸들러 ─────────────────
class DiscordLogHandler(logging.Handler):
    """
    ERROR 레벨 이상의 로그를 디스코드 웹훅으로 전송하는 핸들러.
    - 이 핸들러는 내부에서 절대 logging.* 을 호출하지 않음 (무한 루프 방지)
    - 재진입 방지 플래그로 동일 스레드 중복 전송 차단
    """
    _tls = threading.local()

    def __init__(self, webhook_url: str):
        super().__init__(level=logging.ERROR)
        self.webhook_url = webhook_url

    def emit(self, record: logging.LogRecord):
        # 1) notifier 로거에서 발생한 로그는 무시 (자기 호출 차단)
        if record.name.startswith("notifier"):
            return

        # 2) 재진입(예: 전송 중 예외로 다시 emit 호출) 방지
        if getattr(self._tls, "busy", False):
            return

        # 3) 웹훅 URL 검증
        if not is_valid_webhook(self.webhook_url):
            # 여기서 print 사용: logging 호출 금지
            print("[DiscordLogHandler] Invalid webhook URL. Skip sending.")
            return

        try:
            self._tls.busy = True
            msg = self.format(record)
            formatted = f"**⚠️ ERROR LOG DETECTED ⚠️**\n```\n{msg[:1900]}\n```"

            payload = {"content": formatted}
            # logging 호출 없이 직접 POST
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            # 여기서도 logging 호출 금지
            print(f"[DiscordLogHandler] Failed to send log to Discord: {e}")
        finally:
            self._tls.busy = False

# ───────────────── 단독 실행 테스트 ─────────────────
if __name__ == '__main__':
    print("--- notifier.py 단독 테스트 시작 ---")

    # 루트 로거에 핸들러 추가 (중복 방지)
    if WEBHOOK_URL and is_valid_webhook(WEBHOOK_URL):
        root_logger = logging.getLogger()
        if not any(isinstance(h, DiscordLogHandler) for h in root_logger.handlers):
            root_logger.addHandler(DiscordLogHandler(WEBHOOK_URL))
    else:
        print("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그 전송 테스트를 건너뜁니다.")

    # 테스트 1: 텍스트 메시지
    print("\n1. 텍스트 메시지 전송 테스트...")
    send_discord_message(content="✅ 안녕하세요! notifier.py 단독 실행 테스트 메시지입니다.")

    # 테스트 2: 매수 성공 임베드
    print("\n2. 매수(BUY) 성공 임베드 전송 테스트...")
    sample_buy_trade = {"side": "buy", "name": "삼성전자", "ticker": "005930", "qty": 10, "price": 75000, "trade_status": "completed"}
    send_discord_message(embeds=[create_trade_embed(sample_buy_trade)])

    # 테스트 3: 매도 실패 임베드
    print("\n3. 매도(SELL) 실패 임베드 전송 테스트...")
    sample_sell_trade = {
        "side": "sell", "name": "카카오", "ticker": "035720", "qty": 20, "price": 55000,
        "trade_status": "failed", "strategy_details": {"error": "증거금 부족으로 주문이 거부되었습니다."}
    }
    send_discord_message(embeds=[create_trade_embed(sample_sell_trade)])

    # 테스트 4: 에러 로그 핸들러 동작 확인 (루트 로거에 ERROR 발행)
    print("\n4. 에러 로그 핸들러 테스트...")
    logging.error("이것은 notifier.py에서 보내는 테스트 에러 로그입니다. 디스코드에 전송되어야 합니다.")

    print("\n--- 테스트 종료 ---")
    print("디스코드 채널에서 메시지를 확인하세요.")
