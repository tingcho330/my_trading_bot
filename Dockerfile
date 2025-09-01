# 1. 베이스 이미지
FROM python:3.11-slim-bookworm

# 2. 시스템 패키지 + tzdata 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    tzdata \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 3. 타임존/KST 설정 (OS 레벨)
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 4. 런타임 환경
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 5. 작업 디렉토리
WORKDIR /app

# 6. 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. 앱 소스 복사
COPY ./src ./src
# (필요 시) 설정/환경파일 복사
# COPY ./config ./config

# 8. 비루트 유저 (권장)
RUN useradd -m bot && chown -R bot:bot /app
USER bot

# 9. 엔트리포인트: 스케줄러 실행
CMD ["python", "-u", "/app/src/scheduler.py"]
