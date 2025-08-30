# 1. 베이스 이미지 선택 (Debian 12 기반)
FROM python:3.11-slim-bookworm

# 2. 시스템 패키지 설치
# build-essential: C/C++ 컴파일러 등 기본 빌드 도구
# libssl-dev: SSL/TLS 암호화 통신 관련 개발 라이브러리
# libffi-dev: 외부 C 함수 호출 관련 라이브러리
# 위 라이브러리들은 FinanceDataReader 같은 패키지가 소스에서 빌드될 때 필요합니다.
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 소스코드 복사
# (KIS API를 사용하지 않는 screener.py 전용 Dockerfile)
# COPY ./src .