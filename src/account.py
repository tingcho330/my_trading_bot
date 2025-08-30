# kis_api/account.py

import pandas as pd
from . import auth as ka # 같은 패키지 내의 auth.py를 ka라는 별칭으로 가져옵니다.

def inquire_total_balance(cano: str, acnt_prdt_cd: str):
    """
    [투자계좌자산현황조회] API
    계좌의 전체적인 자산 현황(총자산, 평가손익 등)을 조회합니다.
    """
    API_URL = "/uapi/domestic-stock/v1/trading/inquire-account-balance"
    TR_ID = "CTRP6548R"

    params = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt_prdt_cd,
        "INQR_DVSN_1": "",
        "BSPR_BF_DT_APLY_YN": ""
    }
    
    res = ka._url_fetch(API_URL, TR_ID, params)
    
    if res and res.isOK():
        df_assets = pd.DataFrame(res.getBody()['output1'])
        df_summary = pd.DataFrame([res.getBody()['output2']])
        return df_assets, df_summary
    else:
        if res: res.printError()
        return pd.DataFrame(), pd.DataFrame()

def inquire_holdings(cano: str, acnt_prdt_cd: str):
    """
    [주식잔고조회] API
    계좌에 보유 중인 종목의 상세 내역을 조회합니다.
    """
    API_URL = "/uapi/domestic-stock/v1/trading/inquire-balance"
    TR_ID = "TTTC8434R"
    
    params = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt_prdt_cd,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02", # 종목별 조회
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "00", # 전일매매포함
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": ""
    }
    
    res = ka._url_fetch(API_URL, TR_ID, params)
    
    if res and res.isOK():
        df_holdings = pd.DataFrame(res.getBody()['output1'])
        df_summary = pd.DataFrame([res.getBody()['output2']])
        return df_holdings, df_summary
    else:
        if res: res.printError()
        return pd.DataFrame(), pd.DataFrame()
