# kis_api/__init__.py

# auth.py 파일의 auth 함수를 kis_api.auth() 로 바로 호출할 수 있게 합니다.
from .kis_auth import auth

# account.py 파일의 함수들을 kis_api.inquire_balance() 등으로 바로 호출할 수 있게 합니다.
from .account import inquire_holdings, inquire_total_balance

print("KIS API 패키지를 성공적으로 불러왔습니다.")
