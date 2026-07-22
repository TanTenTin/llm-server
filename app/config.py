from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    google_ai_api_key: str = ""
    # Ollama가 뜨는 주소. 온프레미스 서버 주소로 변경 가능
    ollama_base_url: str = "http://localhost:11434"
    # Ollama 요청 컨텍스트 창(num_ctx, 토큰). 0 이하면 미지정(=Ollama 서버 기본, 보통 2~4k).
    # OpenAI 호환 엔드포인트는 num_ctx를 못 받으므로 게이트웨이는 네이티브 /api/chat 경로로
    # 이 값을 주입한다 — 코딩 에이전트의 큰 시스템 프롬프트·도구 정의가 잘리는 것을 막는다.
    # 원격 Ollama 호스트에 OLLAMA_CONTEXT_LENGTH env를 함께 두면 서버 기본값도 올라간다.
    #
    # (E-02) 이 값이 라우터의 로컬 컨텍스트 판단의 '단일 소스'다 — registry가 ollama 모델의
    # context_window를 이 값에서 파생시킨다(예전엔 registry에 32_000이 하드코딩돼 있어 런타임
    # 16384와 어긋나 조용한 잘림이 났다). 기본값을 32768로 두어 로컬 우선 라우팅 범위를
    # 유지한다. 서버 RAM 한도에 맞춰 .env에서 조정 가능 — 낮추면 라우터가 자동으로 보수적으로
    # 판단하므로(=큰 입력은 클라우드로) 값을 바꿔도 잘림 위험이 생기지 않는다.
    ollama_num_ctx: int = 32768
    # qwen3 등 '사고(thinking)' 모델의 내부 reasoning을 기본 비활성화할지 여부.
    # True면 요청이 think를 명시하지 않았고 업스트림이 thinking 계열 모델일 때 think=False를
    # 보낸다 — 에이전트 요청에서 reasoning이 출력 예산을 다 써 content가 비는 문제를 막는다.
    # thinking 미지원 모델(gemma 등)엔 보내지 않으며, 그래도 400이 나면 think 없이 자동 재시도한다.
    ollama_disable_think: bool = True
    # (E-05) Ollama가 모델을 메모리에 유지하는 시간(keep_alive). "30m"·"1h"·초 정수·"-1"(상주).
    # 기본 5분 후 언로드되면 큰 num_ctx의 KV 캐시까지 재적재해 콜드스타트가 크다 → 길게 잡는다.
    # 빈 문자열이면 payload에 싣지 않아 Ollama 기본(5분)을 따른다.
    ollama_keep_alive: str = "30m"
    # (E-05) 기동 시 로컬 기본 모델을 예열(짧은 generate)해 첫 실요청 지연을 없앨지 여부.
    # True면 lifespan에서 DEFAULT_MODEL(ollama)이면 keep_alive로 1회 로드한다. 배포가 잦으면
    # 기동이 느려질 수 있어 기본 False(keep_alive만으로도 두 번째 요청부터는 빠르다).
    ollama_warmup: bool = False
    # (E-06) 로컬 Ollama 동시 요청 상한(게이트웨이 세마포어). 단일 GPU/CPU 서버에서 동시
    # 요청이 몰리면 모델 스왑 스래싱·큐 대기 타임아웃이 나므로 직렬화 정도를 제어한다.
    # 0 이하면 무제한(제한 없음). 스트리밍은 스트림이 끝날 때까지 슬롯을 점유한다.
    ollama_max_concurrency: int = 0
    # (E-07) 원격(http/https) 이미지 URL을 게이트웨이가 fetch해 base64로 변환할 때의 최대 크기(바이트).
    # Ollama images 필드는 base64만 받으므로 원격 URL은 게이트웨이가 받아 변환한다. 초과 시 거부.
    ollama_max_image_bytes: int = 20_000_000
    # 게이트웨이 공유 인증 토큰. 설정 시 /v1/* 요청에 'Authorization: Bearer <키>' 필요.
    # 비우면 인증 없이 개방 → 외부(llm.tan-kim.com) 노출 시 반드시 설정할 것.
    gateway_api_key: str = ""
    # 회로차단기 쿨다운(초). provider가 일시 장애(429/5xx/연결오류)를 내면 이 시간 동안
    # 해당 provider를 폴백 체인 뒤로 미뤄(=헛때리는 지연 제거), 만료되면 자동으로 다시 시도(half-open).
    # 0 이하로 두면 회로차단 비활성화(기존처럼 매 요청 primary부터 시도).
    # 업스트림 429에 Retry-After/RetryInfo가 있으면 그 값이 이 기본값 대신 쓰인다(상한 1시간).
    breaker_cooldown_seconds: float = 30.0
    # (E-13) 회로를 열기 전 요구하는 연속 실패 횟수. 단발 429/일시 오류 하나로 provider 전체를
    # 뒤로 미루는 과잉 개방을 막는다. 단, 업스트림이 Retry-After를 명시하면(명시적 백오프 신호)
    # 임계치와 무관하게 즉시 연다. 1이면 기존처럼 첫 실패에 개방.
    breaker_failure_threshold: int = 2
    # (E-14) 리버스 프록시(Caddy/nginx) 뒤에서 X-Forwarded-For의 첫 IP를 클라이언트로 신뢰할지.
    # 켜면 프록시 뒤에서도 실제 클라이언트 IP 단위로 레이트리밋이 걸린다. 프록시가 XFF를
    # 덮어써 주는 신뢰된 배치에서만 켤 것(직접 노출 시 스푸핑 가능). 기본 False(=프록시 IP 사용).
    trust_proxy_forwarded_for: bool = False
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

    # ── 로컬 Realtime 파이프라인 (provider=local, model=local-live) ────────────
    # STT/TTS/VAD를 게이트웨이 프로세스 안에서 CPU로 돌린다(로컬 전용 원칙 유지 —
    # 음성·프롬프트가 클라우드로 나가지 않는다). 무거운 모델 의존성
    # (faster-whisper·MeloTTS·silero-vad)은 requirements-realtime.txt로 분리하고,
    # 로컬 Live 세션이 '처음 열릴 때' 지연 로드한다 — 기본 게이트웨이 기동엔 영향이 없다.
    realtime_local_stt_model: str = "small"        # faster-whisper 크기(tiny/base/small/medium)
    realtime_local_stt_language: str = "ko"        # 전사 언어 고정(자동감지 오판·언어 튐 방지)
    # STT 실행 디바이스. "cpu"(기본) 또는 "cuda". cuda는 LLM과 VRAM을 나눠 쓰므로
    # 음성 LLM을 경량 모델로 내렸을 때만 켤 것. cuda 로드 실패 시 cpu로 자동 폴백한다.
    realtime_local_stt_device: str = "cpu"
    realtime_local_tts_language: str = "KR"        # MeloTTS 언어 코드
    realtime_local_tts_speed: float = 1.0          # 발화 속도(1.0=기본)
    realtime_local_llm_temperature: float = 0.7    # 대화용 샘플링(텍스트 봇과 동일 철학)
    # 음성 대화에 쓸 LLM 모델. 기본은 게이트웨이 DEFAULT_MODEL과 같은 qwen3:14b —
    # 텍스트 경로와 모델을 공유해 상주(keep_alive) 모델을 재사용한다(모델 스왑 지연 회피).
    # 더 빠른 첫 응답이 필요하면 ollama/qwen3.5:9b 등 경량 모델로 바꾼다
    # (선택지·실측치는 docs/ollama-bench-2026-07.md 참고).
    realtime_local_llm_model: str = "ollama/qwen3:14b"
    # 음성 턴 하나의 LLM 출력 상한(토큰). 음성은 2~3문장이면 충분하다 —
    # 폭주 응답이 TTS 큐를 수십 초씩 붙잡는 것을 막는다. 0 이하면 무제한.
    realtime_local_llm_max_tokens: int = 200
    # VAD가 '발화 종료'로 판정할 침묵 길이(ms). 짧으면 말 도중 끊기고, 길면 응답이 늦어진다.
    realtime_local_vad_silence_ms: int = 700
    # 클라이언트로 내보낼 출력 PCM16 레이트(Hz). OpenAI Realtime 관례는 24000.
    realtime_local_output_sample_rate: int = 24000


settings = Settings()
