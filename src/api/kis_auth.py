# api/kis_auth.py
import yaml
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from .domestic_stock.domestic_stock_functions import DomesticStock

# 토큰을 저장할 파일 경로 설정
TOKEN_FILE = Path("/app/output/cache/kis_token.json")


class KIS(DomesticStock):
    """
    개선 사항
    1) 토큰 캐시: /app/output/cache/kis_token.json
    2) 토큰 만료 자동 감지(EGW00123/401/invalid token 등) → reauthenticate() → 1회 재시도
    3) 안전 요청 래퍼: request_get()/request_post()/safe_call()
       - 앞으로 KIS API 호출은 가능한 한 이 래퍼들로 감싸서 호출
    4) Python 3.9 호환: Optional[dict] 사용
    """

    def __init__(self, config: dict = {}, env: str = 'prod'):
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        self.is_korea_time = True

        # ---- 설정 로드 ----
        if not config:
            yaml_path = Path('/app/config/kis_devlp.yaml')
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"ERROR: '{yaml_path}' 파일을 찾을 수 없습니다.")
                raise

        if env == 'prod':
            self.app_key = config['my_app']
            self.app_secret = config['my_sec']
            self.cano = config['my_acct_stock']
            self.acnt_prdt_cd = config['my_prod']
            self.url_base = config['prod']
        elif env == 'vps':
            self.app_key = config['paper_app']
            self.app_secret = config['paper_sec']
            self.cano = config['my_paper_stock']
            self.acnt_prdt_cd = config['my_prod']
            self.url_base = config['vps']
        else:
            raise ValueError(f"알 수 없는 env: {env}")

        self.env = env
        self.auth_token = self.auth()  # 최초 인증/로딩

    # =========================
    # 토큰 저장/로드/발급/재인증
    # =========================
    def _save_token(self, token_data: dict):
        """발급받은 토큰과 만료 시간을 파일에 저장 (KIS는 통상 24h 유효)"""
        expires_at = datetime.utcnow() + timedelta(hours=24)
        token_data['expires_at'] = expires_at.isoformat() + "Z"

        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, 'w', encoding='utf-8') as f:
            json.dump(token_data, f)
        print("[KIS] 새로운 토큰을 파일에 저장했습니다.")

    def _load_token(self) -> Optional[dict]:
        """파일에서 유효한 토큰을 로드 (만료 5분 전이면 무효 처리)"""
        if not TOKEN_FILE.exists():
            return None
        try:
            with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            expiry = self._parse_iso_utc(data.get('expires_at'))
            if (expiry is None) or (expiry - timedelta(minutes=5) < datetime.utcnow()):
                print("[KIS] 기존 토큰 만료 또는 임박 → 새 발급 필요")
                return None
            print("[KIS] 기존 토큰을 재사용합니다.")
            return data
        except (json.JSONDecodeError, KeyError):
            print("[KIS] 토큰 파일 손상 → 새로 발급합니다.")
            return None

    def _create_new_token(self) -> Optional[str]:
        """KIS API 서버로부터 새 토큰 발급"""
        p = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        url = f"{self.url_base}/oauth2/tokenP"
        try:
            res = requests.post(url, data=json.dumps(p), headers={"Content-Type": "application/json"}, timeout=10)
        except Exception as e:
            print(f"[KIS] 토큰 발급 요청 실패: {e}")
            return None

        try:
            res_data = res.json()
        except Exception:
            print(f"[KIS] 토큰 발급 응답 파싱 실패: status={res.status_code}, text={res.text[:200]}")
            return None

        if 'access_token' in res_data:
            self._save_token(res_data)
            return res_data['access_token']
        else:
            print("[KIS] Authentication failed:", res_data)
            return None

    def auth(self) -> Optional[str]:
        """인증 토큰 관리 (로드 또는 신규 발급) + 헤더 장착"""
        token_data = self._load_token()
        if token_data and 'access_token' in token_data:
            access_token = token_data['access_token']
        else:
            access_token = self._create_new_token()

        if access_token:
            self._bind_headers(access_token)
        return access_token

    def reauthenticate(self):
        """만료/무효 토큰일 때 강제 재인증. 캐시 삭제 후 재발급."""
        try:
            if TOKEN_FILE.exists():
                TOKEN_FILE.unlink()
        except Exception:
            pass
        print("[KIS] 토큰 재인증(re-auth) 수행")
        new_token = self._create_new_token()
        if not new_token:
            raise RuntimeError("[KIS] 재인증 실패")
        self._bind_headers(new_token)

    def _bind_headers(self, access_token: str):
        self.headers["authorization"] = "Bearer " + access_token
        self.headers["appkey"] = self.app_key
        self.headers["appsecret"] = self.app_secret
        # 주의: tr_id 는 엔드포인트마다 다름. 각 호출부에서 세팅하거나 메서드 인자로 받도록.

    @staticmethod
    def _parse_iso_utc(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            # '...Z' 지원
            if s.endswith("Z"):
                s = s[:-1]
            return datetime.fromisoformat(s)
        except Exception:
            return None

    # =========================
    # 토큰 만료 감지 & 안전 호출
    # =========================
    @staticmethod
    def is_token_expired_error_from_resp(resp: requests.Response) -> bool:
        """HTTP 응답 기반 만료 감지"""
        try:
            if resp.status_code == 401:
                return True
            data = resp.json()
            msg = f"{data}".upper()
        except Exception:
            msg = (resp.text or "").upper()

        return ("EGW00123" in msg) or ("TOKEN" in msg and ("EXPIRE" in msg or "INVALID" in msg))

    @staticmethod
    def is_token_expired_error_from_exc(e: Exception) -> bool:
        m = str(e).upper()
        return ("EGW00123" in m) or ("TOKEN" in m and ("EXPIRE" in m or "INVALID" in m)) or (" 401" in m)

    def safe_call(self, fn, *args, **kwargs):
        """
        임의의 함수 호출을 감싸서 만료 감지 시 1회 reauthenticate 후 재시도.
        fn은 requests 호출 또는 KIS 메서드여도 됨.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if self.is_token_expired_error_from_exc(e):
                self.reauthenticate()
                return fn(*args, **kwargs)
            raise

    # -------------------------
    # 요청 래퍼: GET / POST
    # -------------------------
    def request_get(self, url: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> requests.Response:
        """
        KIS GET 호출용 안전 래퍼. 만료 감지 시 1회 재인증 후 재시도.
        """
        merged_headers = dict(self.headers)
        if headers:
            merged_headers.update(headers)

        def _do():
            return requests.get(url, headers=merged_headers, params=params, timeout=timeout)

        resp = _do()
        if self.is_token_expired_error_from_resp(resp):
            self.reauthenticate()
            resp = _do()
        return resp

    def request_post(self, url: str, *, headers: Optional[Dict[str, str]] = None, data: Any = None, json_body: Any = None, timeout: int = 10) -> requests.Response:
        """
        KIS POST 호출용 안전 래퍼. 만료 감지 시 1회 재인증 후 재시도.
        """
        merged_headers = dict(self.headers)
        if headers:
            merged_headers.update(headers)

        def _do():
            if json_body is not None:
                return requests.post(url, headers=merged_headers, json=json_body, timeout=timeout)
            else:
                # data가 dict면 JSON으로 직렬화해주는 게 안전
                payload = data if isinstance(data, (str, bytes)) else json.dumps(data) if data is not None else None
                return requests.post(url, headers=merged_headers, data=payload, timeout=timeout)

        resp = _do()
        if self.is_token_expired_error_from_resp(resp):
            self.reauthenticate()
            resp = _do()
        return resp

    # =========================
    # 기존(예시) 기능 메서드들
    # =========================
    def get_time_diff_ratio(self) -> float:
        """
        서버시간과 로컬시간의 차이를 초 단위로 계산하여 반환
        - 토큰 만료 시 자동 재인증 후 1회 재시도
        """
        url = f"{self.url_base}/uapi/domestic-stock/v1/quotations/chk-server"
        tr_id = 'PS-HP-01'
        resp = self.request_get(url, headers={"tr_id": tr_id})
        # 일부 환경에서 헤더가 없거나 포맷이 다른 경우가 있음
        svr_key = 'X-DT-BaseDateTime'
        if svr_key not in resp.headers:
            # 실패 시 0 반환(보수적 처리)
            return 0.0
        try:
            server_time = datetime.strptime(resp.headers[svr_key], '%Y-%m-%d %H:%M:%S:%f')
        except Exception:
            server_time = datetime.strptime(resp.headers[svr_key], '%Y-%m-%d %H:%M:%S')
        local_time = datetime.utcnow()
        time_diff = (server_time - local_time).total_seconds()
        return float(time_diff)

    def set_time_diff_ratio(self, is_korea_time: bool = True) -> bool:
        """한국시간 사용 여부 설정"""
        self.is_korea_time = is_korea_time
        return self.is_korea_time

    def get_current_time(self) -> datetime:
        """서버시간과 로컬시간의 차이를 반영한 현재시간을 반환"""
        return datetime.utcnow() + timedelta(seconds=self.get_time_diff_ratio())
