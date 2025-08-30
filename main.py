# main.py

import kis_api
import yaml
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_account_info():
    """
    설정 파일에서 계좌 정보를 읽어옵니다.
    이 함수는 설정 파일이 있는 경우에만 동작합니다.
    """
    config_path = os.path.join(os.path.expanduser("~"), "KIS", "config", "kis_devlp.yaml")
    try:
        with open(config_path, 'r', encoding='UTF-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 'my_acct_stock'과 'my_prod' 키가 있는지 확인
        if 'my_acct_stock' in config and 'my_prod' in config:
            cano = config['my_acct_stock']
            acnt_prdt_cd = config['my_prod']
            return cano, acnt_prdt_cd
        else:
            logging.error("'my_acct_stock' 또는 'my_prod'를 설정 파일에서 찾을 수 없습니다.")
            return None, None
            
    except FileNotFoundError:
        logging.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return None, None
    except Exception as e:
        logging.error(f"설정 파일 로딩 중 오류 발생: {e}")
        return None, None


def main():
    """
    메인 실행 함수
    """
    # 1. API 인증
    kis_api.auth()
    
    # 2. 계좌 정보 가져오기
    # TODO: kis_devlp.yaml 파일에 본인의 계좌번호 8자리(my_acct_stock)와
    # 상품코드 2자리(my_prod)를 정확히 입력해주세요.
    cano, acnt_prdt_cd = get_account_info()
    
    if not cano or not acnt_prdt_cd:
        logging.error("계좌정보를 가져오지 못했습니다. 프로그램을 종료합니다.")
        return

    # 3. 보유 종목 조회
    logging.info("\n--- 보유 종목 조회 시작 ---")
    holdings_df, _ = kis_api.inquire_holdings(cano, acnt_prdt_cd)
    
    if not holdings_df.empty:
        # 원하는 컬럼만 선택하여 출력
        cols_to_show = ['prdt_name', 'hldg_qty', 'pchs_avg_pric', 'evlu_amt', 'evlu_pfls_rt']
        print(holdings_df[cols_to_show])
    else:
        print("보유 종목이 없습니다.")
    logging.info("--- 보유 종목 조회 완료 ---\n")

    # 4. 계좌 총자산 현황 조회
    logging.info("--- 계좌 총자산 현황 조회 시작 ---")
    _, summary_df = kis_api.inquire_total_balance(cano, acnt_prdt_cd)
    
    if not summary_df.empty:
        # 원하는 컬럼만 선택하여 출력
        cols_to_show = ['nass_tot_amt', 'evlu_pfls_amt_smtl', 'tot_evlu_amt', 'pchs_amt_smtl']
        # 컬럼 이름 변경
        summary_df.rename(columns={
            'nass_tot_amt': '순자산총액',
            'evlu_pfls_amt_smtl': '총평가손익',
            'tot_evlu_amt': '총평가금액',
            'pchs_amt_smtl': '총매입금액'
        }, inplace=True)
        print(summary_df[summary_df.columns.intersection(cols_to_show)]) # 변경된 이름으로 출력
    else:
        print("계좌 총자산 현황을 조회할 수 없습니다.")
    logging.info("--- 계좌 총자산 현황 조회 완료 ---")

if __name__ == "__main__":
    main()
