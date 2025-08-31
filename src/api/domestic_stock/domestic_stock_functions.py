import requests
import json
import pandas as pd
from datetime import datetime, timedelta

class DomesticStock:
    def __init__(self):
        # This will be initialized by the KIS class
        self.url_base = getattr(self, 'url_base', '')
        self.headers = getattr(self, 'headers', {})
        self.cano = getattr(self, 'cano', '')
        self.acnt_prdt_cd = getattr(self, 'acnt_prdt_cd', '')

    def inquire_price(self, fid_cond_mrkt_div_code: str, fid_input_iscd: str):
        """
        주식현재가 시세
        """
        url = f"{self.url_base}/uapi/domestic-stock/v1/quotations/inquire-price"
        tr_id = "FHKST01010100"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": fid_cond_mrkt_div_code,
            "FID_INPUT_ISCD": fid_input_iscd,
        }
        
        res = requests.get(url, headers={"tr_id": tr_id, **self.headers}, params=params)
        
        # Check if the request was successful
        if res.status_code == 200:
            return pd.DataFrame([res.json()['output']])
        else:
            print("Error:", res.status_code, res.text)
            return pd.DataFrame()

    def inquire_balance(self, inqr_dvsn: str, afhr_flpr_yn: str, ofl_yn: str, 
                        unpr_dvsn: str, fund_sttl_icld_yn: str, fncg_amt_auto_rdpt_yn: str, 
                        prcs_dvsn: str, ctx_area_fk100: str = "", ctx_area_nk100: str = ""):
        """
        주식 잔고 조회
        """
        url = f"{self.url_base}/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = "TTTC8434R"
        
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": afhr_flpr_yn,
            "OFL_YN": ofl_yn,
            "INQR_DVSN": inqr_dvsn,
            "UNPR_DVSN": unpr_dvsn,
            "FUND_STTL_ICLD_YN": fund_sttl_icld_yn,
            "FNCG_AMT_AUTO_RDPT_YN": fncg_amt_auto_rdpt_yn,
            "PRCS_DVSN": prcs_dvsn,
            "CTX_AREA_FK100": ctx_area_fk100,
            "CTX_AREA_NK100": ctx_area_nk100
        }
        
        res = requests.get(url, headers={"tr_id": tr_id, **self.headers}, params=params)
        
        if res.status_code == 200:
            df_balance = pd.DataFrame(res.json()['output1'])
            df_summary = pd.DataFrame([res.json()['output2']])
            return df_balance, df_summary
        else:
            print("Error:", res.status_code, res.text)
            return pd.DataFrame(), pd.DataFrame()

    def order_cash(self, ord_dv: str, pdno: str, ord_dvsn: str, ord_qty: int, ord_unpr: int):
        """
        현금 주문
        ord_dv: "01": 매도, "02": 매수
        """
        url = f"{self.url_base}/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = "TTTC0802U" if ord_dv == "02" else "TTTC0801U"
        
        data = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": pdno,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(ord_qty),
            "ORD_UNPR": str(ord_unpr),
        }
        
        res = requests.post(url, headers={"tr_id": tr_id, **self.headers}, data=json.dumps(data))
        
        if res.status_code == 200:
            return pd.DataFrame([res.json()['output']])
        else:
            print("Error:", res.status_code, res.text)
            return pd.DataFrame()

    # Add other domestic stock functions here as needed