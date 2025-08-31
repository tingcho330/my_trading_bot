# src/account.py
import pprint
import json
from pathlib import Path
from datetime import datetime

# KIS API 모듈 임포트 경로 및 방식 (프로젝트 구조에 맞게 유지)
from api.kis_auth import KIS

# ───────────────── 경로 설정 ─────────────────
# 요청대로 /app/out 디렉터리에 저장
CONFIG_PATH = Path("/app/config/config.json")
OUTPUT_DIR = Path("/app/out")
COOLDOWN_FILE = OUTPUT_DIR / "cooldown.json"  # (미사용 시에도 경로 유지)


# ───────────────── 설정 및 유틸리티 함수 ─────────────────

def load_settings() -> dict:
    """설정 파일(config.json)을 로드합니다."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_today_tag() -> str:
    """파일명에 사용할 날짜 태그 (YYYYMMDD). 같은 날 실행 시 동일 파일명으로 덮어쓰기."""
    return datetime.now().strftime("%Y%m%d")


def build_balance_comments() -> dict:
    """df_balance 컬럼 설명(주석) 사전."""
    return {
        "pdno": "종목코드 (예: 005930 = 삼성전자)",
        "prdt_name": "종목명",
        "hldg_qty": "보유수량",
        "ord_psbl_qty": "매도 가능 수량",
        "pchs_avg_pric": "평균매입단가",
        "prpr": "현재가",
        "evlu_amt": "평가금액 (보유수량 × 현재가)",
        "evlu_pfls_amt": "평가손익 금액",
        "evlu_pfls_rt": "평가손익률 (%)",
        "bfdy_buy_amt": "전일 매수 금액",
        "thdt_buy_amt": "당일 매수 금액",
        "bfdy_sll_amt": "전일 매도 금액",
        "thdt_sll_amt": "당일 매도 금액"
        # 필요 시 실제 응답 컬럼을 보며 추가하세요.
    }


def build_summary_comments() -> dict:
    """df_summary 컬럼 설명(주석) 사전."""
    return {
        "dnca_tot_amt": "예수금 총액",
        "nxdy_excc_amt": "익일 출금 가능 금액",
        "prvs_rcdl_excc_amt": "이전 영업일 출금 가능 금액",
        "cma_evlu_amt": "CMA 평가 금액",
        "tot_evlu_amt": "총 평가 금액 (주식 평가금 + 예수금)",
        "pchs_amt_smtl_amt": "매입 금액 합계",
        "evlu_pfls_smtl_amt": "평가 손익 합계",
        "evlu_pfls_rt": "평가 손익률 (%)",
        "bfdy_buy_amt": "전일 매수 총액",
        "thdt_buy_amt": "당일 매수 총액",
        "bfdy_sll_amt": "전일 매도 총액",
        "thdt_sll_amt": "당일 매도 총액"
        # 필요 시 실제 응답 컬럼을 보며 추가하세요.
    }


def dump_with_comments(filepath: Path, comments: dict, df) -> None:
    """
    주석과 데이터(레코드 리스트)를 하나의 JSON에 저장.
    - 파일 구조: {"comments": {...}, "data": [...]}
    - 같은 경로에 동일 파일명 존재 시 덮어쓰기.
    - df가 비어도 파일을 생성(빈 리스트 저장)해 일관성 유지.
    """
    payload = {
        "comments": comments,
        "data": ([] if df is None or df.empty else df.to_dict(orient="records"))
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --- 실행 부분 ---
if __name__ == "__main__":
    try:
        # 설정 파일에서 환경 정보 로드
        settings = load_settings()
        # config.json의 'trading_environment' 키 사용: 'prod' 또는 'vps' 등
        trading_env = settings.get("trading_environment", "vps")

        # KIS 클래스 인스턴스화 및 인증 확인
        kis = KIS(config={}, env=trading_env)
        if not kis.auth_token:
            raise ConnectionError("KIS API 인증에 실패했습니다.")
        
        print(f"'{trading_env}' 모드로 인증 완료. 계좌 잔고 조회를 시작합니다...")

        # KIS 인스턴스를 통해 잔고 조회
        # 반환: df_balance (보유 종목별 내역), df_summary (계좌 요약 정보)
        df_balance, df_summary = kis.inquire_balance(
            inqr_dvsn="02",            # 조회구분: 02=잔고 조회
            afhr_flpr_yn="N",          # 시간외 단일가 반영 여부 (N: 미반영)
            ofl_yn="",                 # 오프라인 여부 (공란 유지)
            unpr_dvsn="01",            # 단가 구분 (01: 일반)
            fund_sttl_icld_yn="N",     # 펀드 결제 포함 여부
            fncg_amt_auto_rdpt_yn="N", # 금융금액 자동상환 여부
            prcs_dvsn="00"             # 처리 구분
        )

        # 콘솔 출력 (가독성 유지)
        print("\n--- 보유 종목 현황 ---")
        if df_balance is not None and not df_balance.empty:
            pprint.pprint(df_balance.to_dict('records'))
        else:
            print("보유 종목이 없습니다.")

        print("\n--- 계좌 종합 평가 ---")
        if df_summary is not None and not df_summary.empty:
            # 요약은 1행이 일반적이므로 첫 레코드만 출력
            recs = df_summary.to_dict('records')
            pprint.pprint(recs[0] if recs else {})
        else:
            print("계좌 종합 정보를 가져오는 데 실패했습니다.")

        # ───────────── JSON 저장 ─────────────
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        today = get_today_tag()

        balance_file = OUTPUT_DIR / f"balance_{today}.json"
        summary_file = OUTPUT_DIR / f"summary_{today}.json"

        # 주석(컬럼 설명) + 데이터 동시 저장
        dump_with_comments(balance_file, build_balance_comments(), df_balance)
        dump_with_comments(summary_file, build_summary_comments(), df_summary)

        print(f"\n파일 저장 완료:")
        print(f"- 보유 종목: {balance_file}")
        print(f"- 계좌 요약: {summary_file}")

    except Exception as e:
        print(f"오류 발생: {e}")
