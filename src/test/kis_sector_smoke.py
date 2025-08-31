# kis_sector_smoke.py (inquire_price only)

import sys
import re
import argparse
import logging
import pandas as pd

# ────────── 로깅 + 민감정보 마스킹 ──────────
class RedactFilter(logging.Filter):
    PATTERNS = [
        (re.compile(r'(authorization\'?:\s*[\'"]?)Bearer\s+[A-Za-z0-9\-\._]+', re.IGNORECASE), r"\1Bearer ****"),
        (re.compile(r'(appkey\'?:\s*[\'"]?)[A-Za-z0-9\-\._]+', re.IGNORECASE), r"\1****"),
        (re.compile(r'(appsecret\'?:\s*[\'"]?)[^\'",]+', re.IGNORECASE), r"\1****"),
        (re.compile(r'("authorization":\s*)"Bearer\s+[^"]+"', re.IGNORECASE), r'\1"Bearer ****"'),
        (re.compile(r'("appkey":\s*)"[^"]+"', re.IGNORECASE), r'\1"****"'),
        (re.compile(r'("appsecret":\s*)"[^"]+"', re.IGNORECASE), r'\1"****"'),
    ]
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            m = record.msg
            for pat, repl in self.PATTERNS:
                m = pat.sub(repl, m)
            record.msg = m
        return True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
for h in logging.getLogger().handlers:
    h.addFilter(RedactFilter())
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ────────── KIS 모듈 ──────────
sys.path.append('./src')
import src.api.kis_auth as ka
from domestic_stock import domestic_stock_functions as ds

# ────────── 섹터 정규화 ──────────
def normalize_sector(x: str | None) -> str:
    if not x:
        return "N/A"
    s = str(x).strip()
    if s.upper() in {"", "NAN", "NA", "N/A", "NONE"}:
        return "N/A"
    mapping = {
        "보험":"금융","증권":"금융","은행":"금융",
        "IT 서비스":"IT서비스","정보기술":"IT서비스",
        "반도체":"전기전자","전자":"전기전자",
        "건설":"건설","조선":"제조","기계":"제조","화학":"화학",
        "유통":"유통","통신":"통신","의료정밀":"의료정밀","의약품":"의약품",
    }
    if s in mapping: return mapping[s]
    for k,v in mapping.items():
        if k in s: return v
    return s

def extract_sector(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    for col in ["bstp_kor_isnm", "sect_kr_nm", "induty_kor_isnm"]:
        if col in df.columns:
            val = str(df[col].iloc[0]).strip()
            if val and val.upper() not in {"N/A","NONE"}:
                return val
    return None

# ────────── inquire_price만 사용 ──────────
def call_inquire_price(ticker: str) -> pd.DataFrame | None:
    try:
        df = ds.inquire_price(
            env_dv="real",
            fid_cond_mrkt_div_code="J",
            fid_input_iscd=str(ticker).zfill(6),
        )
        return df if isinstance(df, pd.DataFrame) and not df.empty else None
    except Exception as e:
        logging.debug("inquire_price 예외(%s): %s", ticker, e)
        return None

# ────────── 스크리너용 공개 함수 ──────────
def get_kis_sector_map(tickers: list[str], debug: bool = False) -> dict[str,str]:
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logging.info("KIS 인증…")
    ka.auth(svr='prod')
    trenv = ka.getTREnv()
    if not (trenv and getattr(trenv, "my_token", None)):
        raise RuntimeError("KIS 인증 실패(토큰 없음)")

    result: dict[str,str] = {}
    for t in tickers:
        code = str(t).zfill(6)
        sec_raw = None

        df = call_inquire_price(code)
        if df is not None:
            sec_raw = extract_sector(df)

        norm = normalize_sector(sec_raw)
        result[code] = norm
        logging.debug("ticker=%s sector_raw=%s → sector_norm=%s", code, sec_raw, norm)

    return result

# ────────── CLI ──────────
def parse_args():
    ap = argparse.ArgumentParser(description="KIS 섹터맵(스크리너 최소 정보, inquire_price only)")
    ap.add_argument("--tickers", nargs="*", default=["005930","000660","035420"])
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sectors = get_kis_sector_map(args.tickers, debug=args.debug)
    print("\nTicker, Sector")
    for k,v in sectors.items():
        print(f"{k}, {v}")
