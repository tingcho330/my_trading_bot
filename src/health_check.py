# src/health_check.py

import sys
import logging
from api.kis_auth import KIS

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HealthCheck")

def main():
    """
    KIS API의 상태를 확인하기 위해 간단한 API(현재가 조회)를 호출합니다.
    성공 시 exit code 0, 실패 시 1을 반환합니다.
    """
    logger.info("API 헬스 체크를 시작합니다...")
    try:
        # KIS 인스턴스 생성 (설정 파일은 kis_auth.py 내부 로직에 따라 자동으로 로드됨)
        # trading_environment 설정에 따라 'prod' 또는 'vps' 모드로 자동 인증됩니다.
        kis = KIS(env='vps') 
        if not getattr(kis, "auth_token", None):
            raise ConnectionError("KIS API 인증에 실패했습니다 (토큰 없음).")

        # 가장 안정적인 종목인 삼성전자(005930) 현재가 조회를 시도
        price_df = kis.inquire_price(fid_cond_mrkt_div_code="J", fid_input_iscd="005930")

        if price_df is None or price_df.empty:
            raise ValueError("API가 비정상적인 응답(빈 데이터)을 반환했습니다.")
        
        price = price_df['stck_prpr'].iloc[0]
        logger.info(f"✅ API 헬스 체크 통과. 서버가 정상 응답합니다. (삼성전자 현재가: {price})")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ API 헬스 체크 실패. 서버가 불안정하거나 설정에 문제가 있습니다.", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()