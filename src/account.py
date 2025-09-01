# src/account.py
"""
계좌 잔고/요약 조회 스크립트
- 로깅/경로/시간대/설정은 src/utils.py 사용
- KIS API로 잔고/요약 조회 후 JSON 저장
- 같은 날짜(KST)로 재실행 시 동일 파일명으로 '덮어쓰기' 동작
- 각 JSON에 컬럼 주석(comments) 포함
- notifier.py 연동: 시작/성공/에러 디스코드 알림 + ERROR 이상 로그 자동 전송
"""

import pprint
import json
import sys
import logging
from datetime import datetime

# 공통 유틸리티 모듈 (src/utils.py)
from utils import (
    setup_logging, load_config, OUTPUT_DIR, KST
)

# KIS API 모듈
from api.kis_auth import KIS

# 디스코드 알림
from notifier import DiscordLogHandler, WEBHOOK_URL, send_discord_message

# ───────────────── 로깅 설정 ─────────────────
setup_logging()
logger = logging.getLogger("Account")

# 루트 로거에 DiscordLogHandler 부착(중복 방지)
_root = logging.getLogger()
if WEBHOOK_URL and WEBHOOK_URL.startswith(("http://", "https://")):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL이 없어 에러 로그의 디스코드 전송을 비활성화합니다.")

def _notify(msg: str):
    """실패해도 파이프라인 막지 않도록 안전 전송"""
    try:
        if WEBHOOK_URL and WEBHOOK_URL.startswith(("http://", "https://")):
            send_discord_message(content=msg)
    except Exception:
        pass

# ───────────────── 유틸리티 함수 ─────────────────
def get_today_tag() -> str:
    """파일명에 사용할 날짜 태그 (YYYYMMDD) - KST 기준."""
    return datetime.now(KST).strftime("%Y%m%d")

def build_balance_comments() -> dict:
    """df_balance 컬럼 설명(주석) 사전."""
    return {
        "pdno": "종목코드 (예: 005930 = 삼성전자)",
        "prdt_name": "종목명",
        "hldg_qty": "보유수량",
        "pchs_avg_pric": "평균매입단가",
        "prpr": "현재가",
        "evlu_amt": "평가금액",
        "evlu_pfls_amt": "평가손익 금액",
        "evlu_pfls_rt": "평가손익률 (%)",
    }

def build_summary_comments() -> dict:
    """df_summary 컬럼 설명(주석) 사전."""
    return {
        "dnca_tot_amt": "예수금 총액",
        "prvs_rcdl_excc_amt": "D+2 출금 가능 금액 (가장 확실한 주문 가능금)",
        "tot_evlu_amt": "총 평가 금액 (주식 평가금 + 예수금)",
        "pchs_amt_smtl_amt": "매입 금액 합계",
        "evlu_pfls_smtl_amt": "평가 손익 합계",
    }

def dump_with_comments(filepath, comments: dict, df) -> None:
    """
    주석과 데이터를 하나의 JSON 파일에 저장합니다.
    같은 파일 경로에 저장하므로 동일 일자 재실행 시 '덮어쓰기' 됩니다.
    """
    payload = {
        "comments": comments,
        "data": ([] if df is None or df.empty else df.to_dict(orient="records"))
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _to_int(v) -> int:
    """문자/숫자 혼합 금액 안전 파싱"""
    try:
        if v is None:
            return 0
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).replace(",", "").strip()
        return int(float(s)) if s else 0
    except Exception:
        return 0

# ───────────────── 메인 ─────────────────
if __name__ == "__main__":
    try:
        today = get_today_tag()
        _notify(f"💼 계좌 조회 시작 (date={today})")

        # 설정 파일 로드 (src/utils.load_config)
        settings = load_config()
        trading_env = settings.get("trading_environment", "mock")
        kis_cfg = settings.get("kis_broker", {})  # 계정/키 등은 config.json의 kis_broker에 저장

        # KIS API 인증
        kis = KIS(config=kis_cfg, env=trading_env)
        if not getattr(kis, "auth_token", None):
            raise ConnectionError("KIS API 인증에 실패했습니다.")

        logger.info("'%s' 모드로 인증 완료. 계좌 잔고 조회를 시작합니다...", trading_env)

        # 잔고/요약 조회
        df_balance, df_summary = kis.inquire_balance(
            inqr_dvsn="02",           # 조회구분(보유/당일 등)
            afhr_flpr_yn="N",
            ofl_yn="",
            unpr_dvsn="01",
            fund_sttl_icld_yn="N",
            fncg_amt_auto_rdpt_yn="N",
            prcs_dvsn="00"
        )

        # 응답 검증
        if df_summary is None or df_summary.empty:
            raise RuntimeError("KIS API로부터 계좌 요약 정보(summary)를 가져오는 데 실패했습니다.")

        # 콘솔 출력
        logger.info("\n--- 보유 종목 현황 ---")
        if df_balance is not None and not df_balance.empty:
            pprint.pprint(df_balance.to_dict("records"))
        else:
            logger.info("보유 종목이 없습니다.")

        logger.info("\n--- 계좌 종합 평가 ---")
        recs = df_summary.to_dict("records")
        pprint.pprint(recs[0] if recs else {})

        # JSON 저장 — 같은날 재실행 시 덮어쓰기
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        balance_file = OUTPUT_DIR / f"balance_{today}.json"
        summary_file = OUTPUT_DIR / f"summary_{today}.json"

        dump_with_comments(balance_file, build_balance_comments(), df_balance)
        dump_with_comments(summary_file, build_summary_comments(), df_summary)

        logger.info("\n파일 저장 완료:")
        logger.info("- 보유 종목: %s", balance_file)
        logger.info("- 계좌 요약: %s", summary_file)

        # 성공 요약 디스코드 노티
        try:
            summ = recs[0] if recs else {}
            cash_d2 = _to_int(summ.get("prvs_rcdl_excc_amt"))
            cash_tot = _to_int(summ.get("dnca_tot_amt"))
            tot_eval = _to_int(summ.get("tot_evlu_amt"))
            hold_cnt = 0 if df_balance is None or df_balance.empty else len(df_balance)
            _notify(
                "✅ 계좌 조회 완료\n"
                f"- 보유종목: {hold_cnt}개\n"
                f"- D+2 출금가능: {cash_d2:,}원\n"
                f"- 예수금: {cash_tot:,}원\n"
                f"- 총평가: {tot_eval:,}원\n"
                f"- files: {balance_file.name}, {summary_file.name}"
            )
        except Exception:
            _notify("✅ 계좌 조회 완료 (요약 구성 실패)")

    except Exception as e:
        # 루트 로거에 붙은 DiscordLogHandler가 ERROR 이상 자동 전송함
        logger.critical("account.py 실행 중 심각한 오류 발생: %s", e, exc_info=True)
        # 추가로 간단 알림(중복 가능성 있으나 명시적 알림 선호 시 유지)
        try:
            _notify(f"❌ account.py 실패: {str(e)[:400]}")
        except Exception:
            pass
        sys.exit(1)
