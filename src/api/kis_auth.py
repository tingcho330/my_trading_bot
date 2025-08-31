import yaml
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from .domestic_stock.domestic_stock_functions import DomesticStock

# 토큰을 저장할 파일 경로 설정
TOKEN_FILE = Path("/app/output/cache/kis_token.json")

class KIS(DomesticStock):
    def __init__(self, config: dict = {}, env: str = 'prod'):
        self.headers = {"Content-Type": "application/json"}
        self.is_korea_time = True
        
        if not config:
            try:
                yaml_path = Path('/app/config/kis_devlp.yaml')
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
        
        self.auth_token = self.auth()

    def _save_token(self, token_data: dict):
        """발급받은 토큰과 만료 시간을 파일에 저장"""
        # KIS 토큰은 발급 후 24시간 유효
        expires_at = datetime.now() + timedelta(hours=24)
        token_data['expires_at'] = expires_at.isoformat()
        
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, 'w', encoding='utf-8') as f:
            json.dump(token_data, f)
        print("새로운 토큰을 발급받아 파일에 저장했습니다.")

    def _load_token(self) -> dict | None:
        """파일에서 유효한 토큰을 로드"""
        if not TOKEN_FILE.exists():
            return None
        
        try:
            with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 만료 5분 전이면 만료된 것으로 간주
            expiry = datetime.fromisoformat(data['expires_at'])
            if expiry - timedelta(minutes=5) < datetime.now():
                print("기존 토큰이 만료되어 새로 발급받습니다.")
                return None
            
            print("기존 토큰을 재사용합니다.")
            return data
        except (json.JSONDecodeError, KeyError):
            print("토큰 파일이 손상되었습니다. 새로 발급받습니다.")
            return None

    def _create_new_token(self) -> str | None:
        """KIS API 서버로부터 새 토큰을 발급받음"""
        p = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        url = f"{self.url_base}/oauth2/tokenP"
        res = requests.post(url, data=json.dumps(p), headers=self.headers)
        res_data = res.json()
        
        if 'access_token' in res_data:
            self._save_token(res_data)
            return res_data['access_token']
        else:
            print("Authentication failed:", res_data)
            return None

    def auth(self):
        """인증 토큰을 관리 (로드 또는 신규 발급)"""
        token_data = self._load_token()
        
        if token_data and 'access_token' in token_data:
            access_token = token_data['access_token']
        else:
            access_token = self._create_new_token()

        if access_token:
            self.headers["authorization"] = "Bearer " + access_token
            self.headers["appkey"] = self.app_key
            self.headers["appsecret"] = self.app_secret
        
        return access_token

    # --- 이하 기존 함수들은 동일 ---

    def get_time_diff_ratio(self):
        """서버시간과 로컬시간의 차이를 초 단위로 계산하여 반환"""
        url = f"{self.url_base}/uapi/domestic-stock/v1/quotations/chk-server"
        tr_id = 'PS-HP-01'
        self.headers["tr_id"] = tr_id
        res = requests.get(url, headers=self.headers)
        try:
            server_time = datetime.strptime(res.headers['X-DT-BaseDateTime'], '%Y-%m-%d %H:%M:%S:%f')
        except:
            server_time = datetime.strptime(res.headers['X-DT-BaseDateTime'], '%Y-%m-%d %H:%M:%S')
        local_time = datetime.utcnow()
        time_diff = (server_time - local_time).total_seconds()
        return time_diff

    def set_time_diff_ratio(self, is_korea_time:bool=True):
        """한국시간 사용 여부 설정"""
        self.is_korea_time = is_korea_time
        return self.is_korea_time

    def get_current_time(self):
        """서버시간과 로컬시간의 차이를 반영한 현재시간을 반환"""
        return datetime.now() + timedelta(seconds=self.get_time_diff_ratio())