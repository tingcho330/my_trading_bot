# src/main.py
import os
import sys
import pprint

# 현재 파일의 경로를 sys.path에 추가하여 모듈 임포트 경로를 설정
# 이렇게 해야 Docker 컨테이너에서도 kis_auth와 domestic_stock_functions를 찾을 수 있음
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import kis_auth as ka
from domestic_stock import domestic_stock_functions as ds

# 'prod' (실전투자) 또는 'vps' (모의투자) 중 선택
# kis_devlp.yaml 파일에 설정된 값에 따라 자동으로 올바른 서버에 접속합니다.
ka.auth(svr='prod')

def check_my_account():
    """ domestic_stock_functions.py에 정의된 주식 잔고 조회 함수를 호출합니다. """
    # inquire_balance 함수의 파라미터는 KIS API 명세를 참고하여 필요한 값으로 설정합니다.
    # 예시: 실전투자, 종합계좌, 시간외단일가 미포함, 종목별 조회 등
    trenv = ka.getTREnv()
    df1, df2 = ds.inquire_balance(
        env_dv="real", # 실전
        cano=trenv.my_acct,           # 계좌번호 앞 8자리
        acnt_prdt_cd=trenv.my_prod,   # 계좌번호 뒤 2자리
        afhr_flpr_yn="N",
        inqr_dvsn="02",
        unpr_dvsn="01",
        fund_sttl_icld_yn="N",
        fncg_amt_auto_rdpt_yn="N",
        prcs_dvsn="00"
    )

    if not df1.empty:
        print("--- 보유 주식 목록 ---")
        pprint.pprint(df1.to_dict('records'))
    else:
        print("보유 주식이 없습니다.")

    if not df2.empty:
        print("\n--- 계좌 종합 정보 ---")
        pprint.pprint(df2.to_dict('records')[0])
    else:
        print("계좌 종합 정보를 가져오지 못했습니다.")

if __name__ == "__main__":
    print("계좌 조회를 시작합니다.")
    check_my_account()