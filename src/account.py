# src/account.py
"""
계좌 잔고/요약 조회 스크립트 (안전본)
- utils.py 로깅/경로/시간대 사용
- KIS API 조회 → JSON 저장(같은 날은 덮어쓰기)
- 디스코드/로그 모두 '저장된 JSON'을 기준으로 동일 로직 계산(불일치 제거)
- 토큰 만료/일시 오류 자동 복구(reauth + 재시도)
- ❗️degraded(실패) 발생 시: 본파일(bal/summary_YYYYMMDD.json)을 절대 덮지 않고 *_degraded.json에만 기록
"""

import pprint
import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

from utils import setup_logging, OUTPUT_DIR, KST
from settings import settings
from api.kis_auth import KIS
from notifier import DiscordLogHandler, WEBHOOK_URL, send_discord_message

# ───────── 로깅 ─────────
setup_logging()
logger = logging.getLogger("Account")
_root = logging.getLogger()
if WEBHOOK_URL and WEBHOOK_URL.startswith(("http://", "https://")):
    if not any(isinstance(h, DiscordLogHandler) for h in _root.handlers):
        _root.addHandler(DiscordLogHandler(WEBHOOK_URL))
        logger.info("DiscordLogHandler attached to root logger.")
else:
    logger.warning("유효한 DISCORD_WEBHOOK_URL 없음 → 에러 로그 전송 비활성화")

def _notify(msg: str):
    try:
        if WEBHOOK_URL and WEBHOOK_URL.startswith(("http://", "https://")):
            send_discord_message(content=msg)
    except Exception:
        pass

# ───────── 유틸 ─────────
def get_today_tag() -> str:
    return datetime.now(KST).strftime("%Y%m%d")

def build_balance_comments() -> dict:
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
    return {
        "dnca_tot_amt": "예수금 총액",
        "prvs_rcdl_excc_amt": "D+2 출금 가능 금액",
        "nxdy_excc_amt": "익일(D+1) 출금 가능 금액",
        "tot_evlu_amt": "총 평가 금액 (주식 평가금 + 예수금)",
        "pchs_amt_smtl_amt": "매입 금액 합계",
        "evlu_pfls_smtl_amt": "평가 손익 합계",
        "status": "데이터 상태 (ok/degraded)",
    }

def dump_with_comments(filepath: Path, comments: dict, df: pd.DataFrame | None, extra_fields: dict | None = None) -> None:
    payload = {
        "comments": comments,
        "data": ([] if df is None or df.empty else df.to_dict(orient="records"))
    }
    if extra_fields:
        payload.update(extra_fields)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _to_int(v) -> int:
    try:
        if v is None:
            return 0
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).replace(",", "").strip()
        return int(float(s)) if s else 0
    except Exception:
        return 0

def _pick_int(d: dict, candidates: list[str], default: int = 0) -> int:
    for k in candidates:
        if k in d:
            return _to_int(d.get(k))
    return default

def _denest_first_record(data_list) -> dict:
    """payload['data'][0]가 {"0": {...}} 또는 {0: {...}}처럼 중첩될 수 있어 전개"""
    if not data_list:
        return {}
    rec = data_list[0]
    if isinstance(rec, dict):
        if 0 in rec and isinstance(rec[0], dict):   # 숫자키
            return rec[0]
        if "0" in rec and isinstance(rec["0"], dict):  # 문자열 키
            return rec["0"]
    return rec

def _extract_summary_fields_from_saved_json(summary_path: Path) -> tuple[int, int, int, int, int]:
    """
    저장된 summary_JSON을 다시 읽어 안전하게 수치 추출
    반환: (hold_cnt_placeholder, cash_d2, cash_tot, tot_eval, nxdy_excc)
    hold_cnt는 balance에서 세므로 여기선 0 리턴
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("data", [])
    s = _denest_first_record(data)

    cash_d2 = _pick_int(s, ["prvs_rcdl_excc_amt", "d2_excc_amt", "rcdl_excc_amt_d2"], 0)
    cash_tot = _pick_int(s, ["dnca_tot_amt", "cash_amt", "dnca_avl_amt"], 0)
    tot_eval = _pick_int(s, ["tot_evlu_amt", "total_eval_amt", "tot_evlu", "tot_eval"], 0)
    nxdy_excc = _pick_int(s, ["nxdy_excc_amt", "d1_excc_amt", "rcdl_excc_amt_d1"], 0)
    return 0, cash_d2, cash_tot, tot_eval, nxdy_excc

def _load_status(filepath: Path) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return (json.load(f) or {}).get("status", "")
    except Exception:
        return ""

# ───────── KIS 래핑 ─────────
def inquire_balance_with_retry(kis: KIS, *, max_tries: int = 3) -> Tuple[pd.DataFrame | None, pd.DataFrame | None, bool, str]:
    """
    kis.inquire_balance 안전 호출:
    - kis.safe_call로 토큰 만료시 자동 reauth + 재시도
    - 네트워크/일시 오류에 대해 최대 max_tries 재시도
    반환: (df_balance, df_summary, degraded_flag, error_msg)
    """
    last_err = None
    for _ in range(max_tries):
        try:
            df_balance, df_summary = kis.safe_call(
                kis.inquire_balance,
                inqr_dvsn="02",
                afhr_flpr_yn="N",
                ofl_yn="",
                unpr_dvsn="01",
                fund_sttl_icld_yn="N",
                fncg_amt_auto_rdpt_yn="N",
                prcs_dvsn="00"
            )
            if df_summary is not None and not df_summary.empty:
                return df_balance, df_summary, False, ""
            last_err = RuntimeError("요약 데이터 비어있음")
        except Exception as e:
            last_err = e
    logger.warning("inquire_balance 재시도 종료: %s", last_err)
    return None, None, True, (str(last_err)[:400] if last_err else "unknown")

def _make_empty_balance_df() -> pd.DataFrame:
    cols = list(build_balance_comments().keys())
    return pd.DataFrame(columns=cols)

def _make_empty_summary_df() -> pd.DataFrame:
    cols = [c for c in build_summary_comments().keys() if c != "status"]
    return pd.DataFrame(columns=cols)

# ───────── 메인 ─────────
if __name__ == "__main__":
    try:
        today = get_today_tag()
        _notify(f"계좌 조회 시작 (date={today})")

        trading_env = settings._config.get("trading_environment", "prod")
        kis_cfg = settings._config.get("kis_broker", {})

        kis = KIS(config=kis_cfg, env=trading_env)
        if not getattr(kis, "auth_token", None):
            raise ConnectionError("KIS API 인증 실패")

        logger.info("'%s' 모드로 인증 완료. 계좌 잔고 조회를 시작합니다...", trading_env)

        # 조회 (안전 래퍼)
        df_balance, df_summary, degraded, err = inquire_balance_with_retry(kis)

        # 저장 경로
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        balance_file = OUTPUT_DIR / f"balance_{today}.json"
        summary_file = OUTPUT_DIR / f"summary_{today}.json"

        # ── degraded 보호 로직 ──
        existing_status = _load_status(summary_file)

        if degraded:
            logger.error("KIS 요약(summary) 수신 실패 → degraded 감지")

            # 본파일은 보존하고(이미 ok라면), 디버깅용 *_degraded.json만 생성
            degraded_sum = summary_file.with_name(summary_file.stem + "_degraded.json")
            degraded_bal = balance_file.with_name(balance_file.stem + "_degraded.json")

            dump_with_comments(
                degraded_bal, build_balance_comments(), _make_empty_balance_df(),
                extra_fields={"status": "degraded", "status_reason": err or "api_empty_or_error"}
            )
            dump_with_comments(
                degraded_sum, build_summary_comments(), _make_empty_summary_df(),
                extra_fields={"status": "degraded", "status_reason": err or "api_empty_or_error"}
            )

            if existing_status == "ok":
                logger.warning("기존 OK 스냅샷 보존: 본파일은 유지, *_degraded.json만 갱신.")
            else:
                logger.warning("오늘자 OK 스냅샷이 없거나 기존도 degraded: 본파일은 건드리지 않음.")

            _notify(
                "⚠️ 계좌 조회 실패(degraded) — 기존 OK 스냅샷 유지\n"
                f"files: {balance_file.name}, {summary_file.name} (참고: *_degraded.json 생성)"
            )
            sys.exit(0)

        # ── 정상 케이스: 로그(원본 확인용) ──
        logger.info("\n--- 보유 종목 현황 ---")
        if df_balance is not None and not df_balance.empty:
            pprint.pprint(df_balance.to_dict("records"))
        else:
            logger.info("보유 종목이 없습니다.")

        logger.info("\n--- 계좌 종합 평가(원본 DataFrame) ---")
        recs = df_summary.to_dict("records")
        pprint.pprint(recs[0] if recs else {})

        # 저장(덮어쓰기) - 정상
        dump_with_comments(balance_file, build_balance_comments(), df_balance, extra_fields={"status": "ok"})
        dump_with_comments(summary_file, build_summary_comments(), df_summary, extra_fields={"status": "ok"})

        # 저장된 JSON 재파싱
        _, cash_d2, cash_tot, tot_eval, nxdy_excc = _extract_summary_fields_from_saved_json(summary_file)

        # 보유종목 수: balance JSON 기준 (수량>0)
        hold_cnt = 0
        if df_balance is not None and not df_balance.empty:
            hold_cnt = sum(1 for row in df_balance.to_dict("records") if _to_int(row.get("hldg_qty", 0)) > 0)

        # 표준화된 요약 로그 (f-string으로 콤마표기)
        logger.info("\n--- 계좌 종합 평가(표준화) ---")
        logger.info("보유종목: %d개", hold_cnt)
        logger.info(f"D+2 출금가능: {cash_d2:,}원")
        if nxdy_excc:
            logger.info(f"익일 출금가능: {nxdy_excc:,}원")
        logger.info(f"예수금: {cash_tot:,}원")
        logger.info(f"총평가(요약): {tot_eval:,}원")
        logger.info("files: %s, %s", str(balance_file), str(summary_file))

        # 디스코드 전송
        try:
            _notify(
                "✅ 계좌 조회 완료\n"
                f"보유종목: {hold_cnt}개\n"
                f"D+2 출금가능: {cash_d2:,}원\n"
                f"예수금: {cash_tot:,}원\n"
                f"총평가: {tot_eval:,}원\n"
                f"files: {balance_file.name}, {summary_file.name}"
            )
        except Exception as e:
            logger.error("디스코드 요약 전송 실패: %s", e, exc_info=True)
            _notify("✅ 계좌 조회 완료 (요약 전송 실패)")

    except Exception as e:
        logger.critical("account.py 실행 중 심각한 오류: %s", e, exc_info=True)
        try:
            _notify(f"❌ account.py 실패: {str(e)[:400]}")
        except Exception:
            pass
        sys.exit(1)
