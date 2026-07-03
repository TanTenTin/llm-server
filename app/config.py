from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    google_ai_api_key: str = ""
    # Ollama가 뜨는 주소. 온프레미스 서버 주소로 변경 가능
    ollama_base_url: str = "http://localhost:11434"
    # 게이트웨이 공유 인증 토큰. 설정 시 /v1/* 요청에 'Authorization: Bearer <키>' 필요.
    # 비우면 인증 없이 개방 → 외부(llm.tan-kim.com) 노출 시 반드시 설정할 것.
    gateway_api_key: str = ""
    # 회로차단기 쿨다운(초). provider가 일시 장애(429/5xx/연결오류)를 내면 이 시간 동안
    # 해당 provider를 폴백 체인 뒤로 미뤄(=헛때리는 지연 제거), 만료되면 자동으로 다시 시도(half-open).
    # 0 이하로 두면 회로차단 비활성화(기존처럼 매 요청 primary부터 시도).
    # 업스트림 429에 Retry-After/RetryInfo가 있으면 그 값이 이 기본값 대신 쓰인다(상한 1시간).
    breaker_cooldown_seconds: float = 30.0
    # 응답 캐시 TTL(초). 비스트리밍 + temperature 미지정/0 인 동일 요청(/v1/chat/completions)을
    # 이 시간 동안 캐시해 무료 티어 쿼터 소모를 줄인다. 0 이하면 캐시 비활성화.
    cache_ttl_seconds: float = 300.0
    # 과금(paid) provider 일일 토큰 예산. Claude 등 is_free=False 모델의 하루(UTC) 사용
    # 토큰 합이 이 값을 넘으면 해당 후보를 건너뛴다(폴백 체인에 무료 후보가 있으면 그쪽으로,
    # 없으면 402). 에이전트 루프 폭주로 인한 과금 사고 방지. 0 이하면 무제한.
    paid_daily_token_budget: int = 0
    # 게이트웨이 분당 요청 상한(키/클라이언트 단위). 0 이하면 무제한.
    # GATEWAY_API_KEY 설정 시 토큰별, 미설정 시 클라이언트 IP별로 집계한다.
    rate_limit_rpm: int = 0
    # ── Realtime(음성) 브리지 설정 — /v1/realtime ───────────────────────────
    # 클라이언트가 model을 지정하지 않을 때 쓸 기본 Gemini Live 모델 id.
    # 계정/대시보드에서 사용 가능한 정확한 id로 교체할 것(예: native audio dialog 모델).
    realtime_default_model: str = "gemini-2.5-flash-native-audio-preview-09-2025"
    # 클라이언트가 보내는 입력 PCM16 샘플레이트(Hz). OpenAI Realtime 기본은 24000.
    # Gemini Live 입력은 16000을 요구하므로, 다르면 브리지가 16kHz로 리샘플한다.
    # 클라이언트가 이미 16000으로 보낸다면 16000으로 두면 리샘플을 건너뛴다.
    realtime_input_sample_rate: int = 24000


settings = Settings()
