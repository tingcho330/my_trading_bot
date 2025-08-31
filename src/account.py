# src/account.py
import pprint
import json
from pathlib import Path
from datetime import datetime

# KIS API 모듈 임포트 경로 및 방식 수정
from api.kis_auth import KIS

# ───────────────── 경로 설정 ─────────────────
CONFIG_PATH = Path("/app/config/config.json")
OUTPUT_DIR = Path("/app/output")
COOLDOWN_FILE = OUTPUT_DIR / "cooldown.json"


# ───────────────── 설정 및 유틸리티 함수 ─────────────────

def load_settings() -> dict:
    """설정 파일(config.json)을 로드합니다."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --- 실행 부분 ---
if __name__ == "__main__":
    try:
        # 설정 파일에서 환경 정보 로드
        settings = load_settings()
        trading_env = settings.get("trading_environment", "vps")

        # KIS 클래스 인스턴스화
        kis = KIS(config={}, env=trading_env)
        if not kis.auth_token:
            raise ConnectionError("KIS API 인증에 실패했습니다.")
        
        print(f"'{trading_env}' 모드로 인증 완료. 계좌 잔고 조회를 시작합니다...")

        # KIS 인스턴스를 통해 잔고 조회
        df_balance, df_summary = kis.inquire_balance(
            inqr_dvsn="02",
            afhr_flpr_yn="N",
            ofl_yn="",
            unpr_dvsn="01",
            fund_sttl_icld_yn="N",
            fncg_amt_auto_rdpt_yn="N",
            prcs_dvsn="00"
        )

        # ───────────── JSON 저장 ─────────────
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 오늘 날짜 (YYYYMMDD) 포맷
        today = datetime.now().strftime("%Y%m%d")
        balance_file = OUTPUT_DIR / f"balance_{today}.json"
        summary_file = OUTPUT_DIR / f"summary_{today}.json"

        if not df_balance.empty:
            df_balance.to_json(balance_file, orient="records", force_ascii=False, indent=2)
            print(f"보유 종목 현황 저장 완료: {balance_file}")
        else:
            print("보유 종목이 없습니다.")

        if not df_summary.empty:
            df_summary.to_json(summary_file, orient="records", force_ascii=False, indent=2)
            print(f"계좌 종합 평가 저장 완료: {summary_file}")
        else:
            print("계좌 종합 정보를 가져오는 데 실패했습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
