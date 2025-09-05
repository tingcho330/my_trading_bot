# src/health_check.py
"""
KIS API 헬스체크
- 삼성전자(005930) 현재가 조회로 API 정상 여부 확인
- 성공 시 exit code 0, 실패 시 exit code 1
- notifier.py 연동: 시작/성공/실패시 디스코드 알림
"""

import sys
import logging
from api.kis_auth import KIS
from utils import setup_logging
from notifier import DiscordLogHandler, WEBHOOK_URL, send_discord_message

# ─────────── 로깅 설정 ───────────
setup_logging()
logger = logging.getLogger("HealthCheck")

_root = logging.getLogger()
if WEBHOOK_URL and WEBHOOK_URL.startswith(("http://", "https://")):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

def _notify(msg: str):
    """디스코드 간단 알림 (실패 시 무시)"""
    try:
        if WEBHOOK_URL and WEBHOOK_URL.startswith(("http://", "https://")):
            send_discord_message(content=msg)
    except Exception:
        pass

# ─────────── 메인 ───────────
def main():
    logger.info("API 헬스 체크를 시작합니다...")
    _notify("ߩꠋIS API 헬스체크 시작")

    try:
        kis = KIS(env="prod")
        if not getattr(kis, "auth_token", None):
            raise ConnectionError("KIS API 인증 실패 (토큰 없음)")

        # 삼성전자 현재가 조회
        price_df = kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd="005930")
        if price_df is None or price_df.empty:
            raise ValueError("API가 빈 데이터를 반환했습니다.")

        price = price_df["stck_prpr"].iloc[0]
        msg = f"✅ API 헬스체크 통과 (삼성전자 현재가: {price})"
        logger.info(msg)
        _notify(msg)
        sys.exit(0)

    except Exception as e:
        msg = f"❌ API 헬스체크 실패: {e}"
        logger.error(msg, exc_info=True)
        _notify(msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
