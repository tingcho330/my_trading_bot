# kis_api/auth.py

import json
import os
import requests
import yaml
from datetime import datetime
from collections import namedtuple

# --- 설정값 ---
# 설정파일은 홈디렉토리 아래 KIS/config 폴더에 위치해야 합니다.
# (예: /Users/사용자이름/KIS/config/kis_devlp.yaml)
CONFIG_ROOT = os.path.join(os.path.expanduser("~"), "KIS", "config")
TOKEN_FILE = os.path.join(CONFIG_ROOT, f"KIS_token_{datetime.today().strftime('%Y%m%d')}")
_TRENV = None
_BASE_HEADERS = {"Content-Type": "application/json", "Accept": "text/plain", "charset": "UTF-8"}

# --- 내부 함수 ---
def _save_token(token_data):
    """발급받은 토큰을 파일에 저장"""
    try:
        if not os.path.exists(CONFIG_ROOT):
            os.makedirs(CONFIG_ROOT)
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(token_data, f)
    except Exception as e:
        print(f"토큰 저장 실패: {e}")

def _read_token():
    """파일에서 토큰 읽기"""
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def _is_token_valid(token_data):
    """토큰 유효기간 체크"""
    if not token_data or 'access_token_token_expired' not in token_data:
        return False
    
    expire_dt_str = token_data['access_token_token_expired']
    expire_dt = datetime.strptime(expire_dt_str, "%Y-%m-%d %H:%M:%S")
    return expire_dt > datetime.now()

def _set_env(cfg, token):
    """환경변수 설정"""
    global _TRENV
    KISEnv = namedtuple("KISEnv", ["my_app", "my_sec", "my_acct", "my_prod", "my_url", "my_token"])
    _TRENV = KISEnv(
        my_app=cfg["my_app"],
        my_sec=cfg["my_sec"],
        my_acct=cfg["my_acct_stock"],
        my_prod=cfg["my_prod"],
        my_url=cfg["my_url"],
        my_token=token,
    )

def _get_base_header():
    """기본 헤더 생성"""
    if _TRENV is None or not _TRENV.my_token:
        raise Exception("인증이 필요합니다. auth() 함수를 먼저 호출해주세요.")
        
    headers = _BASE_HEADERS.copy()
    headers["authorization"] = f"Bearer {_TRENV.my_token}"
    headers["appkey"] = _TRENV.my_app
    headers["appsecret"] = _TRENV.my_sec
    headers["custtype"] = "P"
    return headers

class APIResp:
    """API 응답 처리를 위한 클래스"""
    def __init__(self, resp):
        self._rescode = resp.status_code
        self._resp = resp
        self._json = resp.json()

    def isOK(self):
        return self._json.get("rt_cd") == "0"

    def getBody(self):
        return self._json

    def getErrorCode(self):
        return self._json.get("msg_cd")

    def getErrorMessage(self):
        return self._json.get("msg1")

    def printError(self):
        print(f"API 오류: [{self.getErrorCode()}] {self.getErrorMessage()}")

# --- 공개 함수 ---
def auth(svr="prod"):
    """API 인증 토큰 발급"""
    config_file = os.path.join(CONFIG_ROOT, "kis_devlp.yaml")
    try:
        with open(config_file, encoding="UTF-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"설정 파일({config_file})을 찾을 수 없습니다. 파일을 생성하고 API 키를 입력해주세요.")
        return

    # 1. 기존 토큰 확인
    token_data = _read_token()
    if _is_token_valid(token_data):
        print("기존 토큰이 유효하여 재사용합니다.")
        token = token_data['access_token']
        url_key = 'prod_url' if svr == 'prod' else 'vps_url'
        cfg['my_url'] = cfg[url_key]
        _set_env(cfg, token)
        return

    # 2. 신규 토큰 발급
    print("신규 토큰을 발급합니다.")
    url_base = cfg['prod_url'] if svr == 'prod' else cfg['vps_url']
    url = f"{url_base}/oauth2/tokenP"
    
    p = {
        "grant_type": "client_credentials",
        "appkey": cfg["my_app"] if svr == "prod" else cfg["paper_app"],
        "appsecret": cfg["my_sec"] if svr == "prod" else cfg["paper_sec"]
    }

    res = requests.post(url, data=json.dumps(p), headers=_BASE_HEADERS)
    if res.status_code == 200:
        token_data = res.json()
        _save_token(token_data)
        token = token_data['access_token']
        cfg['my_url'] = url_base
        _set_env(cfg, token)
        print("토큰 발급 성공!")
    else:
        print(f"토큰 발급 실패: {res.text}")

def _url_fetch(api_url, tr_id, params):
    """API 호출 공통 함수"""
    if _TRENV is None:
        raise Exception("인증이 필요합니다. auth() 함수를 먼저 호출해주세요.")

    url = f"{_TRENV.my_url}{api_url}"
    headers = _get_base_header()
    headers["tr_id"] = tr_id
    
    res = requests.get(url, headers=headers, params=params)
    
    if res.status_code == 200:
        return APIResp(res)
    else:
        print(f"API 요청 실패: {res.status_code}, {res.text}")
        return None
