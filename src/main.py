# src/main.py
import pprint

# 이제 sys.path를 직접 건드릴 필요 없이 모듈을 바로 가져올 수 있습니다.
import kis_auth as ka
from domestic_stock import domestic_stock_functions as ds

# --- 실행 부분 ---
if __name__ == "__main__":
    # 'prod' (실전투자) 또는 'vps' (모의투자) 중 선택하여 인증
    # Docker 컨테이너 내부의 /app/config/kis_devlp.yaml 파일을 사용합니다.
    ka.auth(svr='prod') 
    
    # 인증 정보를 가져옵니다.
    trenv = ka.getTREnv()
    
    print("계좌 잔고 조회를 시작합니다...")
    
    # domestic_stock_functions의 잔고 조회 함수를 호출합니다.
    # 필요한 파라미터는 KIS API 문서를 참고하여 채워줍니다.
    # 계좌번호와 상품코드는 trenv 객체에서 안전하게 가져옵니다.
    df_balance, df_summary = ds.inquire_balance(
        env_dv="real",                  # 실전계좌
        cano=trenv.my_acct,             # 계좌번호 앞 8자리
        acnt_prdt_cd=trenv.my_prod,     # 계좌번호 뒤 2자리
        afhr_flpr_yn="N",               # 시간외단일가여부
        inqr_dvsn="02",                 # 조회구분 (01: 대출일별, 02: 종목별)
        unpr_dvsn="01",                 # 단가구분
        fund_sttl_icld_yn="N",          # 펀드결제분포함여부
        fncg_amt_auto_rdpt_yn="N",      # 융자금액자동상환여부
        prcs_dvsn="00"                  # 처리구분 (전일매매포함)
    )
    
    print("\n--- 보유 종목 현황 ---")
    if not df_balance.empty:
        pprint.pprint(df_balance.to_dict('records'))
    else:
        print("보유 종목이 없습니다.")
        
    print("\n--- 계좌 종합 평가 ---")
    if not df_summary.empty:
        pprint.pprint(df_summary.to_dict('records')[0])
    else:
        print("계좌 종합 정보를 가져오는 데 실패했습니다.")