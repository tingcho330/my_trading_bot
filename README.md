# 자동매매 트레이딩 봇 (Automated Trading Bot)

## 1\. 프로젝트 개요

본 프로젝트는 한국 주식 시장(KOSPI, KOSDAQ)을 대상으로 하는 완전 자동화된 퀀트 트레이딩 봇입니다. 정해진 스케줄에 따라 종목 선정(Screener), 뉴스 수집(News Collector), AI 기반 분석(GPT Analyzer) 및 자동 매매(Trader)까지의 전 과정을 수행합니다. 또한, 지속적인 성과 분석(Reviewer)을 통해 스스로 전략 파라미터를 조정하는 피드백 루프를 갖추고 있습니다.

모든 과정은 Docker 컨테이너 환경에서 실행되도록 설계되어 이식성과 확장성을 확보했으며, Discord 웹훅을 통해 실시간으로 주요 이벤트와 거래 내역을 사용자에게 알립니다.

-----

## 2\. 주요 기능

  * **📈 자동화된 데이터 파이프라인**: 스케줄러에 의해 종목 발굴부터 매매까지의 전 과정이 자동으로 실행됩니다.
  * **🧠 AI 기반 투자 결정**: 기술적/기본적 분석 점수와 최신 뉴스를 종합하여 OpenAI의 GPT 모델이 최종 매수/보류 결정을 내립니다. (API 미설정 시 휴리스틱 모드로 자동 전환)
  * **🛡️ 견고한 리스크 관리**:
      * `risk_manager.py`가 독립적으로 실행되어 보유 종목의 상태를 실시간으로 모니터링합니다.
      * ATR, 스윙 저점, RSI 등 다양한 지표를 조합하여 동적으로 손절/익절 라인을 설정합니다.
      * 포트폴리오가 비었을 경우, 자동으로 매매 파이프라인을 재가동하여 거래 연속성을 확보합니다.
  * **🔄 자동 파라미터 튜닝**: `reviewer.py`가 매매 성과(승률, 손익비)를 주기적으로 분석하고, 성과가 부진할 경우 손절/익절 등 핵심 전략 파라미터를 자동으로 미세 조정합니다.
  * **🐳 Docker 기반 배포**: `docker-compose.yml`을 통해 스케줄러와 리스크 매니저를 별도의 서비스로 실행하여 안정성을 높였습니다.
  * **📢 실시간 알림**: 거래 체결, 오류 발생, 파이프라인 시작/종료 등 주요 이벤트를 Discord 웹훅으로 실시간 전송합니다.
  * **🗃️ 상세한 거래 기록**: 모든 거래 내역과 AI의 분석 결과는 SQLite 데이터베이스(`trading_log.db`)에 영구적으로 기록되어 상세한 사후 분석을 지원합니다.

-----

## 3\. 시스템 아키텍처 및 데이터 흐름

본 시스템은 모듈화된 파이썬 스크립트들이 파일 기반 데이터 파이프라인을 통해 상호작용하는 구조로 설계되었습니다.

```
+----------------------+
|   Scheduler.py       |  <-- (Docker Entrypoint) Orchestrates the entire pipeline
+----------------------+
           |
           | (Schedules Jobs)
           ▼
+---------------------------------------------------------------------------------+
|                                 Trading Pipeline                                |
|                                                                                 |
|  [1] Screener.py  -->  [2] News_collector.py  -->  [3] Gpt_analyzer.py           |
|  (종목 후보 생성)      (뉴스 데이터 수집)        (AI/휴리스틱 분석)             |
|       |                     |                          |                        |
|       ▼                     ▼                          ▼                        |
|  screener_...json      collected_news...json      gpt_trades.json               |
|                                                                                 |
|                                      +--------------------+                     |
|                                      |    Trader.py       |  <-- Reads gpt_trades.json
|                                      | (매수/매도 실행)    |      Executes orders via KIS API
|                                      +--------------------+                     |
|                                                |                                |
|                                                ▼                                |
|  +------------------+             +----------------------+                      |
|  |  Risk_Manager.py | <---------- |     Recorder.py      |                      |
|  | (실시간 모니터링)  |             | (DB에 거래 기록)     |                      |
|  +------------------+             +----------------------+                      |
|          ^                                     |                                |
|          | (Triggers Trader if empty)          ▼                                |
|          |                             trading_log.db                           |
|          |                                     |                                |
|          |                                     ▼                                |
|          +--------------------------+-----------------------+                    |
|                                     |  Reviewer.py        |                    |
|                                     | (성과 분석 및 튜닝)  |                    |
|                                     +---------------------+                    |
+---------------------------------------------------------------------------------+
```

**데이터 흐름:**

1.  **`screener.py`**: 시장 데이터(가격, 재무)를 분석하여 조건에 맞는 \*\*종목 후보군 파일(`screener_candidates_*.json`)\*\*을 생성합니다.
2.  **`news_collector.py`**: 후보군 파일을 입력받아 종목별 최신 뉴스를 스크레이핑하여 \*\*뉴스 데이터 파일(`collected_news_*.json`)\*\*을 생성합니다.
3.  **`gpt_analyzer.py`**: 종목 정보와 뉴스 데이터를 종합하여 AI가 분석 후 최종 \*\*매매 계획 파일(`gpt_trades_*.json`)\*\*을 생성합니다.
4.  **`trader.py`**: 매매 계획 파일을 읽어 한국투자증권(KIS) API를 통해 실제 **매수/매도 주문을 실행**합니다.
5.  **`recorder.py`**: `trader.py`에서 발생한 모든 거래 내역을 \*\*SQLite DB(`trading_log.db`)\*\*에 기록합니다.
6.  **`reviewer.py`**: DB의 거래 기록을 분석하여 성과를 평가하고, 필요시 \*\*설정 파일(`config.json`)\*\*의 전략 파라미터를 자동 조정합니다.
7.  **`risk_manager.py`**: 주기적으로 계좌 잔고를 확인하고, 자체적인 리스크 로직에 따라 매도 조건을 판단합니다. 포트폴리오가 비어있으면 `trader.py`를 직접 실행하여 거래를 재개합니다.

-----

## 4\. 모듈 설명

### `src/` 디렉토리 주요 스크립트:

  * **`scheduler.py`**: 전체 파이프라인의 실행을 담당하는 메인 스케줄러. 정해진 시간에 각 모듈을 순서대로 실행합니다.
  * **`screener.py`**: 1차 종목 스크리닝을 수행합니다. 시가총액, 거래대금, 재무 지표(PER, PBR) 등을 기준으로 초기 후보군을 필터링하고, 기술적 지표, 섹터 트렌드 등을 종합하여 점수를 매깁니다.
  * **`news_collector.py`**: `screener.py`가 선정한 종목들의 최신 뉴스를 네이버 API 및 웹 스크레이핑을 통해 수집합니다.
  * **`gpt_analyzer.py`**: 스크리닝된 종목 정보와 수집된 뉴스를 바탕으로 GPT-4 모델을 활용하여 최종 투자 결정을 내립니다. API 키가 없는 경우, 점수 기반의 휴리스틱 분석으로 대체됩니다.
  * **`trader.py`**: `gpt_analyzer.py`의 결정을 바탕으로 실제 매수 주문을 실행하고, `risk_manager.py`의 매도 조건에 따라 보유 종목을 매도합니다.
  * **`risk_manager.py`**: 독립적인 서비스로 실행되며, 보유 종목의 가격 변동을 실시간으로 모니터링하여 ATR, 스윙 저점, RSI 과열 등의 조건에 따라 매도 시점을 판단합니다.
  * **`recorder.py`**: 모든 매매 기록을 SQLite 데이터베이스에 저장하고 조회하는 인터페이스를 제공합니다.
  * **`reviewer.py`**: 데이터베이스의 거래 기록을 바탕으로 FIFO 방식으로 손익을 계산하고, 승률, 손익비 등의 성과 지표를 분석합니다. 분석 결과에 따라 `config.json`의 전략 파라미터를 자동 튜닝합니다.
  * **`strategies.py`**: 다양한 매도 전략(RSI 역추세, 추세 추종, ATR 기반 등)의 로직을 정의합니다.
  * **`utils.py`**: 로깅 설정, 경로 관리, 시간대 설정, 파일 검색 등 프로젝트 전반에서 사용되는 공통 유틸리티 함수를 포함합니다.
  * **`notifier.py`**: Discord 웹훅을 통해 메시지와 임베드를 전송하는 기능을 담당합니다.
  * **`api/kis_auth.py`**: 한국투자증권 API 인증 및 토큰 관리를 담당합니다. 토큰 만료 시 자동 재인증 기능이 포함되어 있습니다.
  * **`health_check.py`**: KIS API의 정상 작동 여부를 간단히 확인하는 스크립트입니다.
  * **`cleanup_output.py`**: 오래된 로그 및 결과 파일을 주기적으로 삭제하여 디스크 공간을 관리합니다.

-----

## 5\. 기술 스택

  * **언어**: Python 3.11
  * **주요 라이브러리**:
      * `pandas`, `numpy`: 데이터 분석 및 처리
      * `requests`, `httpx`: API 요청 및 웹 통신
      * `schedule`: 작업 스케줄링
      * `pykrx`, `FinanceDataReader`: 국내 주식 데이터 수집
      * `openai`: GPT 모델 연동
      * `beautifulsoup4`: 뉴스 본문 스크레이핑
      * `python-dotenv`: 환경 변수 관리
  * **API**:
      * 한국투자증권(KIS) REST API: 실시간 시세 조회 및 주문 실행
      * Naver Search API: 뉴스 검색
      * OpenAI API: 투자 분석 및 의사결정
  * **데이터베이스**: SQLite
  * **배포**: Docker, Docker Compose

-----

## 6\. 설치 및 실행

### 사전 준비 사항

1.  **Docker 및 Docker Compose 설치**: 시스템에 Docker와 Docker Compose가 설치되어 있어야 합니다.
2.  **API 키 발급**:
      * **한국투자증권(KIS) API**: 실전/모의 투자 API Key를 발급받아야 합니다.
      * **Naver Search API**: 네이버 개발자 센터에서 검색 API의 Client ID와 Secret을 발급받아야 합니다.
      * **OpenAI API (선택 사항)**: OpenAI에서 API Key를 발급받습니다.
      * **Discord Webhook URL (선택 사항)**: 알림을 받을 디스코드 채널의 웹훅 URL을 생성합니다.

### 설치 과정

1.  **프로젝트 클론**:

    ```bash
    git clone <repository_url>
    cd my_trading_bot
    ```

2.  **설정 파일 생성**:

      * `config/` 디렉토리 안에 `.env` 파일을 생성하고 아래 내용을 채웁니다.

        ```env
        # .env
        DISCORD_WEBHOOK_URL="여기에_디스코드_웹훅_URL을_입력하세요"
        NAVER_CLIENT_ID="여기에_네이버_클라이언트_ID를_입력하세요"
        NAVER_CLIENT_SECRET="여기에_네이버_시크릿을_입력하세요"
        OPENAI_API_KEY="여기에_OpenAI_API_키를_입력하세요"

        # 실행 파라미터 (선택적 오버라이드)
        MARKET="KOSPI" # KOSPI, KOSDAQ, KONEX 중 선택
        SLOTS="3"      # GPT가 생성할 최대 매수 계획 수
        ```

      * `config/` 디렉토리 안에 `kis_devlp.yaml` 파일을 생성하고 KIS API 정보를 입력합니다. (실제 운영 시에는 파일 접근 권한에 유의하세요.)

        ```yaml
        # kis_devlp.yaml
        prod: https://openapi.koreainvestment.com:9443
        vps: https://openapivts.koreainvestment.com:29443
        my_app: "실전_APP_KEY"
        my_sec: "실전_APP_SECRET"
        my_acct_stock: "실전_계좌번호-01"
        paper_app: "모의_APP_KEY"
        paper_sec: "모의_APP_SECRET"
        my_paper_stock: "모의_계좌번호-01"
        my_prod: "01" # 상품 코드
        ```

      * `config/` 디렉토리의 `config.json` 파일을 필요에 맞게 수정합니다. (기본값으로도 동작 가능)

3.  **Docker 이미지 빌드 및 컨테이너 실행**:
    프로젝트 루트 디렉토리에서 아래 명령어를 실행합니다.

    ```bash
    docker-compose up --build -d
    ```

    이 명령어는 `Dockerfile`을 사용하여 이미지를 빌드하고, `docker-compose.yml`에 정의된 `scheduler`와 `risk_manager` 두 개의 서비스를 백그라운드에서 실행합니다.

### 사용법

  * **실행**: `docker-compose up` 명령어로 컨테이너가 실행되면 `scheduler.py`가 자동으로 시작되어 정해진 스케줄에 따라 파이프라인을 실행합니다.
  * **로그 확인**:
    ```bash
    # 스케줄러 로그 확인
    docker-compose logs -f scheduler

    # 리스크 매니저 로그 확인
    docker-compose logs -f risk_manager
    ```
  * **중지**:
    ```bash
    docker-compose down
    ```

-----

## 7\. 프로젝트 구조

```
my_trading_bot/
├── .devcontainer/         # VSCode 원격 개발 환경 설정
├── config/
│   ├── config.json        # 시스템 주요 파라미터 설정
│   ├── .env               # API 키 등 민감 정보 (Git 추적 안함)
│   └── kis_devlp.yaml     # KIS API 계정 정보 (Git 추적 안함)
├── output/
│   ├── cache/             # API 토큰, 섹터 정보 등 캐시 파일
│   ├── debug/             # 디버깅용 중간 산출물
│   ├── balance_...json    # 일별 계좌 잔고
│   ├── summary_...json    # 일별 계좌 요약
│   ├── screener_...json   # 스크리너 결과 파일
│   ├── collected_news...json # 수집된 뉴스 파일
│   ├── gpt_trades...json  # GPT 분석 및 매매 계획 파일
│   └── trading_log.db     # 모든 거래 기록 (SQLite)
├── src/
│   ├── api/               # 외부 API 연동 모듈
│   ├── __init__.py
│   ├── account.py
│   ├── cleanup_output.py
│   ├── gpt_analyzer.py
│   ├── health_check.py
│   ├── news_collector.py
│   ├── notifier.py
│   ├── recorder.py
│   ├── reviewer.py
│   ├── risk_manager.py
│   ├── scheduler.py
│   ├── screener.py
│   ├── screener_core.py
│   ├── settings.py
│   ├── strategies.py
│   ├── trader.py
│   └── utils.py
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

-----

## 8\. 면책 조항 (Disclaimer)

본 프로젝트는 알고리즘 트레이딩 학습 및 연구 목적으로 개발되었습니다. 제공되는 코드와 정보는 투자 조언이 아니며, 실제 투자에 따른 모든 책임은 사용자 본인에게 있습니다. 자동매매 시스템은 예기치 않은 버그, API 오류, 시장 상황의 급격한 변화 등으로 인해 금전적 손실을 유발할 수 있습니다. 실제 자금으로 운영하기 전에 반드시 충분한 모의 투자를 통해 시스템의 동작을 검증하시기 바랍니다.
