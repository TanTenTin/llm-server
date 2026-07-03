# llm-server

여러 LLM provider(Gemini, Ollama, Anthropic)를 단일 OpenAI 호환 API로 묶는 게이트웨이 서버.

## 목적

- 에이전트/클라이언트가 provider를 몰라도 되도록 추상화
- 모델 이름만 바꿔서 Gemini ↔ Ollama ↔ Anthropic 전환
- 기본은 무료 티어 Gemini(`DEFAULT_MODEL=gemini-2.5-flash`), 장애·키 미설정 시 로컬 Ollama로 자동 폴백
- Oracle 온프레미스 서버에서 실행하는 것을 전제로 설계

## 기술 스택

- **Python** + **FastAPI** (ASGI 서버: uvicorn)
- **httpx** — Ollama·Gemini 비동기 HTTP 프록시 (Gemini는 OpenAI 호환 엔드포인트 사용)
- **anthropic SDK** — Anthropic API 호출 (`AsyncAnthropic`)
- **websockets** — Gemini Live(네이티브 WSS) 업스트림 클라이언트 (`/v1/realtime` 음성 브리지)
- **pydantic-settings** — 환경변수 관리

## 핵심 구조

```
app/
├── main.py        — 엔드포인트 + lifespan(풀·캐시 생성/정리): POST /v1/chat/completions(OpenAI)·/v1/embeddings, POST /v1/messages(Anthropic 네이티브), POST /v1beta/models/{model}:generateContent·:streamGenerateContent(Gemini 네이티브), GET /v1/models, GET /v1/usage(사용량 집계), GET /health(무인증)·/health/providers(심층: 등록/breaker/Ollama 프로브), WS /v1/realtime. 인증(_gateway_keys 쉼표 구분 복수 키)·레이트리밋(_enforce_rate_limit)도 여기
├── config.py      — Settings (GOOGLE_AI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_BASE_URL, GATEWAY_API_KEY(복수 키), BREAKER_COOLDOWN_SECONDS, CACHE_TTL_SECONDS, PAID_DAILY_TOKEN_BUDGET, RATE_LIMIT_RPM, REALTIME_*)
├── models.py      — ChatCompletionRequest·EmbeddingsRequest 등 OpenAI 호환 Pydantic 모델 (내부표준 canonical 포맷)
├── registry.py    — 라우팅 결정: ModelSpec(+cost_tier/is_free/supports_vision 메타) / RouteDecision / MODELS / ALIASES / DEFAULT_MODEL / resolve() / EMBEDDING_MODELS·EMBED_ALIASES·resolve_embedding() / LIVE_ALIASES·resolve_live_model()
├── service.py     — ProviderPool(인스턴스 재사용) + CircuitBreaker(동적 쿨다운) + RouteTrace + fallback 실행 + 에러 분류 + retry_after_seconds(Retry-After/RetryInfo 파싱) + 과금 예산 가드(_over_paid_budget→BudgetExceeded=402)
├── usage.py       — UsageTracker: provider:model × UTC 일 단위 요청/토큰/에러 인메모리 집계(7일 보존, 단일 프로세스 전제). OpenAI·Anthropic·Gemini 3종 usage 필드 정규화 + paid_tokens_today()(예산 가드 근거). /v1/usage로 노출
├── cache.py       — ResponseCache: exact-match(요청 전체 sha256) + TTL + LRU(256). 비스트리밍·temperature 미지정/0만 대상(cache_key_for). 히트는 usage에 "cache" 라벨 집계, x-llm-cache 헤더
├── ratelimit.py   — RateLimiter: 분 단위 고정 윈도우 RPM 제한(키 해시 또는 IP). RATE_LIMIT_RPM=0 비활성
├── realtime.py    — /v1/realtime 음성 브리지: OpenAI Realtime 이벤트 ↔ Gemini Live(네이티브 WSS) 양방향 중계 (RealtimeBridge)
├── audio.py       — PCM16 리샘플러(resample_pcm16): 클라 입력 24kHz → Gemini 16kHz (순수 파이썬, 의존성 0)
├── adapters/      — 네이티브 입력 포맷 어댑터 (edge에서 네이티브 ⇄ OpenAI 내부표준 변환만 담당, 라우팅은 기존 파이프라인 재사용)
│   ├── anthropic_io.py — Anthropic Messages ⇄ OpenAI: anthropic_to_chat_request / openai_to_anthropic_response / stream_openai_to_anthropic
│   ├── gemini_io.py    — Gemini generateContent ⇄ OpenAI: gemini_to_chat_request / openai_to_gemini_response / stream_openai_to_gemini
│   └── sse.py          — 어댑터 공용 SSE 헬퍼(sse_payloads / format_event / format_data)
└── providers/
    ├── base.py       — LLMProvider ABC: chat(request, spec) / stream(request, spec) / aclose()
    ├── openai_payload.py — OpenAI 패스스루 payload 공용 빌더(build_openai_payload). Gemini·Ollama 공용
    ├── ollama.py     — Ollama의 OpenAI 호환 API로 프록시(+think는 네이티브 /api/chat 경로)
    ├── gemini.py     — Gemini의 OpenAI 호환 엔드포인트로 프록시 (영속 httpx client, Bearer 인증) + 네이티브 패스스루(generate_native/stream_native: 별도 native_client, x-goog-api-key 인증, /v1beta/models/{m}:generateContent)
    └── anthropic.py  — OpenAI ↔ Anthropic 포맷 변환 후 SDK 호출 (client 재사용)
```

> 과거의 `router.py`는 제거됨. 라우팅 결정은 `registry.resolve()`, provider 선택/실행은 `service.py`가 담당한다.

> **네이티브 포맷 엔드포인트 = 순수 포맷 어댑터**: `/v1/messages`(Anthropic)·`:generateContent`(Gemini)는 입력을 `adapters/`에서 `ChatCompletionRequest`(내부표준)로 변환해 **기존 `route()`→`*_with_fallback()` 파이프라인을 그대로** 태운다. OpenAI 응답을 다시 네이티브 포맷으로 역변환할 뿐, `model` 필드는 보존되어 라우팅/폴백/회로차단기/`x-llm-route` 헤더가 OpenAI 엔드포인트와 동일하게 동작한다(예: `/v1/messages`로 `gemini-2.5-flash` 요청 → Gemini로 라우팅). `providers/anthropic.py`의 변환은 '내부표준→업스트림' 방향, `adapters/`는 '클라이언트 네이티브→내부표준'(반대 방향)이라 분리한다. Gemini 엔드포인트는 main.py의 `_run_native()`가 공통 실행 경로(비스트리밍 `to_response`/스트리밍 `to_stream` 콜백 주입), 인증은 `require_native_auth()`가 `Authorization: Bearer`/`x-api-key`(Anthropic)/`x-goog-api-key`·`?key=`(Gemini)를 모두 게이트웨이 토큰으로 검증.

> **네이티브 패스스루 fast-path (Anthropic·Gemini 공통)**: 네이티브 엔드포인트에서 라우팅된 후보가 '해당 네이티브 provider'면 OpenAI 이중 변환을 건너뛰고 **클라이언트 원본 body를 그대로 업스트림으로** 보낸다.
> - `/v1/messages` + `claude-*` → `AnthropicProvider.chat_native`/`stream_native`: `cache_control`·정확한 content 블록·멀티턴/스트리밍 `tool_use` 보존(기존 OpenAI→Anthropic 변환의 tool_use 누락 한계를 회피). 표준 외 top-level 필드는 `extra_body`로 전달(`_ANTHROPIC_PASSTHROUGH` 밖).
> - `:generateContent` + `gemini-*` → `GeminiProvider.generate_native`/`stream_native`: 별도 `native_client`(x-goog-api-key 인증)로 네이티브 `/v1beta/models/{m}:generateContent` 호출. `safetySettings`·`thinkingConfig`·`cachedContent` 등 Gemini 전용 필드와 네이티브 응답 구조 보존(OpenAI-compat이 못 싣는 필드).
>
> 구현은 fallback 루프를 **포맷 무관 제너릭**(`service.run_chat_fallback`/`run_stream_fallback`, 후보별 호출 방식을 `invoke`/`open_stream` 콜백으로 주입)으로 분리하고, main.py의 공통 헬퍼 `_run_native_passthrough(native_provider, native_chat, native_stream, to_response, to_stream)`가 '네이티브 후보→패스스루 / 그 외→어댑터 경로'를 콜백 안에서 분기한다(Anthropic·Gemini 핸들러가 이 헬퍼 공유). `chat_with_fallback`/`stream_with_fallback`(OpenAI 경로)도 같은 제너릭 루프의 얇은 래퍼. **단, OpenAI 엔드포인트(`/v1/chat/completions`)로 `claude-*`를 부르면 패스스루가 아니라 `AnthropicProvider.chat`(OpenAI 변환)을 타므로 기존 tool_use 한계가 그대로 남는다.**

## 요청 처리 흐름

```
POST /v1/chat/completions
   → resolve(request.model)                 # registry.py — RouteDecision(chain = primary + fallback들)
   → request.stream 이면:
        stream_with_fallback(...)            # service.py — 첫 청크 전까지만 fallback, SSE
     아니면:
        chat_with_fallback(...)              # service.py — 체인 순서대로 시도
   → 각 후보는 ProviderPool.get(provider)로 재사용 인스턴스를 받아 호출
   → 실패 시 분류:
        ProviderUnavailable(키 없음) / 재시도 가능(연결·타임아웃·404·408·409·429·5xx·529) → 다음 후보
        그 외(400·401·403 등) → 즉시 실패
   → 최종 예외는 http_status_for()로 원래 상태 코드를 살려 반환 (모두 500으로 뭉개지 않음)
```

## 라우팅 규칙 (registry.py)

진입점은 `route(request)`. `model="auto"`면 `_auto_route(request)`(요청 특성 기반), 그 외엔 `resolve(model)`이 모델명을 RouteDecision으로 변환한다. resolve는 위에서부터 먼저 매칭:

| 단계 | 처리 |
|------|------|
| 1. 별칭 | `ALIASES`에 있으면 실제 모델 키로 치환 (`fast`→gemini-flash-lite, `smart`→gemini-flash) |
| 2. 레지스트리 | `MODELS`에 등록돼 있으면 그 `ModelSpec`(+fallback 체인) 사용 |
| 3. 패스스루 | 미등록이라도 형태로 추론: `gemini-`/`gemini/` → Gemini, `ollama/` → Ollama, `anthropic/`·`claude-` → Anthropic, 그 외 `:` 포함 → Ollama |
| 4. 기본값 | 위 어디에도 안 걸리면 `DEFAULT_MODEL`(gemini-2.5-flash, 키 없으면 로컬 폴백) |

### auto 라우트 (Phase 2~4 — `_auto_route`)

`model="auto"`면 요청 특성으로 게이트웨이가 직접 모델 선택:
- **티어 분기(Phase 3~4)**: `_classify_tier(request, estimated_tokens)`가 long/complex/simple 판단 → `AUTO_CANDIDATES_BY_TIER[tier]` 선택. 토큰 추정은 `_auto_route`에서 1회만 수행해 전달.
  - **long이 난이도보다 우선(Phase 4)**: `_estimate_tokens` ≥ `_LONG_INPUT_THRESHOLD(25_000)` → long. 후보 [gemini-2.5-flash, gemini-2.5-flash-lite] (둘 다 1M 컨텍스트, 로컬 32k는 어차피 필터 탈락이라 미포함).
  - complex 조건(하나라도): `tools` 사용 / 추정 토큰 ≥ `_COMPLEX_TOKEN_THRESHOLD(1200)` / 메시지 수 ≥ `_COMPLEX_MESSAGE_COUNT(6)` / 사용자 메시지에 `_COMPLEX_KEYWORDS` 포함.
  - simple → [gemini-2.5-flash-lite, ollama/qwen3:14b] / complex → [gemini-2.5-flash, ollama/qwen3:14b]. 모든 티어 무료만, **과금(Claude) 미포함**(비용 0 보장).
- **capability 필터(Phase 2·4·5)**: `tools` 있는데 `supports_tools=False`면 제외. 이미지(`image_url` 파트, `_has_images`) 있는데 `supports_vision=False`면 제외(reason에 `,vision=1`) — 네이티브 어댑터가 Anthropic `image`/Gemini `inlineData`를 이미 `image_url`로 변환하므로 OpenAI 포맷만 보면 됨. 컨텍스트는 `추정 입력 + request.max_tokens(출력 예산)` > `_usable_context(spec)`(= `context_window` × `_CONTEXT_SAFETY_RATIO(0.8)`)이면 제외 — 추정 오차 + 로컬의 입력·출력 공유 창을 흡수하는 안전 마진.
- **overflow best-effort(Phase 4)**: 컨텍스트 초과로 전원 탈락 시 후보 중 `context_window` 최대 모델 1개를 시도(reason에 `,overflow=1`). DEFAULT_MODEL 무조건 폴백은 이미 탈락한 모델 재선택이라 폐기. 도구 필터로 전원 탈락하는 경우(현 레지스트리엔 없음)만 DEFAULT_MODEL 최후 폴백.
- **토큰 추정(`_estimate_tokens`)**: 메시지 content + 멀티턴 `tool_calls`(함수 인자 — 에이전트 대화에서 커짐) + 도구 정의를 문자수÷`_CHARS_PER_TOKEN=3`으로 근사.
- 선택 사유는 `RouteDecision.reason`("auto:tier=long,est=45000" 형태)으로 실려 `x-llm-route` 헤더에 `reason=`으로 노출. 살아남은 후보가 그대로 체인이 되어 Phase 1 회로차단기·폴백 동일 적용.
- capability 메타는 `ModelSpec.supports_tools`·`context_window`(MODELS에서 모델별 지정). 회귀 테스트: `tests/test_auto_route.py`.

> 패스스루는 **명시적 prefix 단서(`ollama/`/`anthropic/`/`claude-`)를 콜론보다 먼저** 평가한다. 따라서 미등록 `claude-x:snapshot` 도 Anthropic으로 가고, `qwen3:14b` 처럼 prefix 없이 콜론만 있는 건 Ollama로 간다.
> 모델명은 `resolve()`에서 `strip()` 으로 앞뒤 공백을 제거한다.

## Fallback / 풀링 (service.py)

- **ProviderPool**: lifespan 시작 시 provider 인스턴스를 1회 생성해 재사용(httpx keep-alive, `AsyncAnthropic` 재사용), 종료 시 `aclose()`. **Anthropic은 `ANTHROPIC_API_KEY`가 있을 때만 등록** — 빈 키로 SDK 초기화하다 기동이 깨지는 걸 막고, Ollama 전용 배포에서 claude 요청이 와도 `ProviderUnavailable → 로컬 fallback`으로 동작하게 함.
- **chat_with_fallback / stream_with_fallback**: `RouteDecision.chain`을 순서대로 시도. `ProviderUnavailable` 또는 재시도 가능 에러면 다음 후보로.
- **CircuitBreaker (Phase 1 + 동적 쿨다운)**: provider별 일시 장애(429/5xx/연결오류)를 기억해 쿨다운 동안 그 provider를 폴백 체인 **뒤로 미룬다**(`_order_by_breaker`). 무료 티어 Gemini가 429로 막히면 잠깐 Ollama를 우선시켜 헛때리는 지연을 제거. 쿨다운 만료 시 자동 복귀(half-open), 성공 시 즉시 닫힘. `ProviderUnavailable`(키 미설정)은 영구 설정 문제라 회로차단 대상이 아님. 단일 프로세스 코루틴 공유 상태(읽기-쓰기 사이 await 없어 락 불필요).
  - **동적 쿨다운**: `record_failure(provider, cooldown_hint)` — fallback 루프가 `retry_after_seconds(exc)`로 업스트림 힌트(표준 `Retry-After` 헤더 → Gemini 429 본문 `RetryInfo.retryDelay` 순)를 파싱해 전달. 힌트가 있으면 기본 `BREAKER_COOLDOWN_SECONDS` 대신 사용(`_MAX_DYNAMIC_COOLDOWN_SECONDS=3600` 클램프 — RPD 소진 같은 장기 대기도 1시간마다 half-open 탐침). 힌트가 기본값보다 짧으면 그대로 신뢰. 쿨다운 0(비활성) 설정은 힌트보다 우선. `status()`로 open 상태 인트로스펙션(/health/providers가 사용).
- **UsageTracker (usage.py)**: fallback 루프의 성공/실패 지점에서 집계 — 성공 시 `record_success(label, body, fell_back, is_free)`(3종 usage 필드 정규화, 스트리밍은 body=None으로 요청 수만, is_free=False면 paid 토큰 별도 적립), 실패 시 `record_error(label, kind)`(kind는 "429"/"connect"/"unavailable"/"budget" 등). `ProviderPool.usage`에 인스턴스가 살고 `/v1/usage`가 `snapshot()` 노출(+`paid_tokens_by_day`). 무료 티어 쿼터 관측 + 예산 가드의 데이터 기반.
- **과금 예산 가드 (P1)**: fallback 루프가 후보 시도 전 `_over_paid_budget(spec, pool)` 확인 — `PAID_DAILY_TOKEN_BUDGET` > 0이고 `spec.is_free=False`이며 `usage.paid_tokens_today()` ≥ 예산이면 그 후보를 건너뜀(trace `#budget`, usage kind "budget"). 무료 폴백이 없으면 `BudgetExceeded` → 402. 스트리밍은 토큰 미집계라 예산 소모로 안 잡히는 한계 있음.
- **응답 캐시 (cache.py, P1)**: `/v1/chat/completions` 핸들러에서 fallback 루프 진입 전 조회 — `cache_key_for()`가 비스트리밍+temperature 미지정/0만 키 발급(요청 전체 정렬 JSON sha256). 히트 시 업스트림 무호출, `x-llm-route: served=cache` + `x-llm-cache: hit`, usage에 "cache" 라벨 집계. miss면 성공 응답을 `put`. 네이티브 엔드포인트는 캐시 미적용(패스스루 보존 우선).
- **레이트리밋 (ratelimit.py, P1)**: `require_auth`/`require_native_auth`가 인증 통과 후 `_enforce_rate_limit` — 키 sha256(앞 16자) 또는 IP 단위 분당 고정 윈도우. 초과 시 429 + `Retry-After`(다음 분까지 남은 초). WS는 지속 연결이라 미적용.
- **RouteTrace / `x-llm-route` 헤더 (Phase 1)**: 실제로 어떤 후보가 응답했는지 추적해 응답 헤더로 노출(`requested=`/`served=`/`fallback=1`/`deferred=`). 응답 **본문은 OpenAI 형식 그대로** 두므로 호출 측 SDK 호환 유지. silent fallback 관측·로깅(`logger = logging.getLogger("llm_gateway")`)에 사용.
- **스트리밍 fallback은 첫 청크 전까지만** 가능(이미 바이트를 보낸 뒤엔 불가). 어떤 경로로 끝나든 `aclose_quietly()`로 업스트림 스트림을 정리해 커넥션 누수를 막는다.

## 환경변수 (config.py)

`.env`에서 로드 (pydantic-settings, 변수명 대소문자 무시).

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GOOGLE_AI_API_KEY` | `""` | Gemini API 키. `gemini-*`(기본 모델) 사용 시 필요. 비면 Gemini 미등록 → 로컬 fallback |
| `ANTHROPIC_API_KEY` | `""` | Anthropic API 키. `claude-*` 모델 사용 시 필요. 비면 Anthropic 미등록 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 서버 주소 |
| `GATEWAY_API_KEY` | `""` | 게이트웨이 인증 토큰(쉼표 구분 복수 키). 설정 시 `/v1/*`에 `Authorization: Bearer <키>` 필요 |
| `BREAKER_COOLDOWN_SECONDS` | `30.0` | 회로차단기 쿨다운(초). 일시 장애 provider를 이 시간 동안 폴백 체인 뒤로 미룸. `0` 이하면 비활성화. Retry-After/RetryInfo 힌트 우선(상한 1시간) |
| `CACHE_TTL_SECONDS` | `300.0` | 응답 캐시 TTL(초). 비스트리밍+temperature 0/미지정 동일 요청 캐시. `0` 이하면 비활성 |
| `PAID_DAILY_TOKEN_BUDGET` | `0` | 과금 provider 일일 토큰 예산. 초과 시 유료 후보 건너뜀(무료 폴백 or 402). `0` 무제한 |
| `RATE_LIMIT_RPM` | `0` | 분당 요청 상한(키별/IP별). 초과 시 429+Retry-After. `0` 무제한 |
| `REALTIME_DEFAULT_MODEL` | `gemini-2.5-flash-native-audio-preview-09-2025` | `/v1/realtime` 기본 Gemini Live 모델 id(클라가 `?model=` 미지정 시). **계정의 정확한 id로 교체 필요** |
| `REALTIME_INPUT_SAMPLE_RATE` | `24000` | 클라 입력 PCM16 레이트(Hz). Gemini Live는 16000 요구 → 다르면 16kHz로 리샘플(같으면 건너뜀) |

## Realtime 음성 브리지 (realtime.py)

`WS /v1/realtime` — **OpenAI Realtime API 호환** 엔드포인트를 **Gemini Live(네이티브 WSS)** 로 중계한다. 무료 티어 Gemini Live를 OpenAI Realtime 클라이언트로 그대로 쓰게 하는 것이 목적. 텍스트 경로(ProviderPool/fallback/회로차단기)와 **완전히 독립** — Live는 로컬 대체가 없어 폴백 개념이 없다.

```
Client(OpenAI Realtime JSON, pcm16 24kHz)  ⇄  RealtimeBridge  ⇄  Gemini Live(WSS, pcm16 16kHz in / 24kHz out)
   두 펌프 코루틴(_client_to_gemini / _gemini_to_client)을 asyncio.wait(FIRST_COMPLETED)로 동시 구동,
   한쪽 종료 시 나머지 취소 + 양쪽 연결 정리
```

- **연결/인증**: 업스트림 URL은 `wss://.../BidiGenerateContent?key=<GOOGLE_AI_API_KEY>`. 클라이언트 측 게이트웨이 인증은 `ws_authorized()`(main.py)가 `Authorization: Bearer` 헤더 **또는** `?api_key=` 쿼리로 검증(브라우저 WS는 헤더 불가 → 쿼리 허용). `GATEWAY_API_KEY` 미설정 시 개방.
- **setup은 lazy 1회**: 첫 입력 이벤트 시점에 `BidiGenerateContentSetup`을 보낸다(그 전에 온 `session.update`의 instructions/voice/model 반영). `generationConfig.responseModalities=["AUDIO"]` + `input/outputAudioTranscription` 활성.
- **모델 해석**: `registry.resolve_live_model(requested, default)` — `LIVE_ALIASES`(`gemini-live` 등) 치환 후 `models/` 접두사 보장. `?model=` 쿼리 또는 `REALTIME_DEFAULT_MODEL`.
- **오디오**: 입력 24kHz → `audio.resample_pcm16`으로 16kHz 변환 후 `realtimeInput.audio`. 출력 24kHz는 `response.audio.delta`로 패스스루. 리샘플은 순수 파이썬 선형보간(의존성 0).
- **이벤트 매핑**: README의 "Realtime (음성) API" 표 참고. 응답 턴 경계는 `_begin_response_if_needed`(modelTurn 첫 청크 → `response.created`)·`_finish_response`(`turnComplete`/`interrupted` → `response.done`)로 관리.
- **한계(v1)**: `toolCall`(함수 호출)·이미지/비디오 입력 미지원(오디오·텍스트만). `session.update`의 model 변경은 setup 전송 전에만 유효. `REALTIME_DEFAULT_MODEL` 기본값은 **추정 id라 실제 계정의 Live 모델 id로 교체** 필요.

## Anthropic 포맷 변환 주의사항 (providers/anthropic.py)

OpenAI와 Anthropic의 차이가 있어서 변환 로직이 들어있다. 수정 시 아래를 확인할 것.

- **system 메시지**: Anthropic은 `messages` 배열이 아닌 별도 `system` 파라미터로 받음 → `_extract_system()` (복수면 마지막만 사용, 비-문자열 content는 무시)
- **tool result**: OpenAI `role: "tool"` → Anthropic `role: "user"` + `type: "tool_result"` 블록
- **tool definition**: OpenAI `parameters` → Anthropic `input_schema`
- **tool 응답**: Anthropic `tool_use` 블록 → OpenAI `tool_calls` 배열
- **모델명/파라미터**: `spec.upstream`(레지스트리가 결정)을 사용. `max_tokens`는 요청 > `spec.max_tokens` > `DEFAULT_MAX_TOKENS`(8192).

## 알려진 동작 / 주의사항 (gotchas)

코드 수정 전 알아둘 현재 동작:

- **에러 상태 코드**: `http_status_for()`가 `httpx.HTTPStatusError`/`anthropic.APIStatusError`의 status를 그대로 노출, 연결 실패는 502, `ProviderUnavailable`은 503, 그 외 500. (예전처럼 전부 500이 아님)
- **`/v1/models`는 레지스트리(MODELS)에서 자동 생성**. 모델 추가는 `registry.py`만 손대면 목록에도 반영됨.
- **`tool_choice`는 OpenAI 패스스루(Gemini·Ollama)로 전달**: `AnthropicProvider`만 `tool_choice`를 변환/전달하지 않음.
- **패스스루 미지 필드 보존(extra="allow")**: `models.py`의 요청/메시지 모델이 `extra="allow"`라 모델이 모르는 메시지 구조 필드(예: `tool_calls`)도 버리지 않고 업스트림에 전달. 요청 레벨 파라미터는 `openai_payload.build_openai_payload`의 `_FORWARD_PARAMS` 화이트리스트로 전달(미지의 요청 레벨 필드는 무차별 전달하지 않음 — 메시지 보존/요청 파라미터 선별). passthrough 손실은 `tests/test_passthrough.py`가 회귀로 막음.
- **silent fallback 주의**: 등록된 모델이 404/연결오류거나 키 미설정이면 fallback 체인의 다음 후보(로컬 Ollama 등)로 조용히 떨어질 수 있다. 실제 사용된 모델은 응답 `model` 필드 또는 **`x-llm-route` 헤더**로 확인.
- **게이트웨이 자체 인증 없음**: 앞단에 인증/레이트리밋 없음. 외부 노출 시 별도 보호 필요.

### 미구현 / 기존 한계 (라우팅과 별개, 추후 과제)

> 아래 ①②는 **OpenAI→Anthropic 변환 경로(`AnthropicProvider.chat`/`stream`)에 한정**된다. 즉 `/v1/chat/completions`로 `claude-*`를 부르거나, `/v1/messages` 폴백이 Anthropic이 아닌 다른 후보에서 다시 Anthropic으로 갈 때만 해당. **`/v1/messages` + `claude-*`(주 경로)는 패스스루 fast-path(`chat_native`/`stream_native`)라 이 한계가 없다** — body를 그대로 SDK로 보내므로 멀티턴/스트리밍 `tool_use`가 보존됨.

- **① assistant `tool_calls` → Anthropic `tool_use` 역변환 없음**: `AnthropicProvider._convert_messages`가 assistant의 `tool_calls`(속성)를 `tool_use` 블록으로 되돌리지 않아(content가 list일 때만 처리), OpenAI 멀티턴 tool 대화를 Anthropic 변환 경로로 보내면 직전 `tool_use`가 누락 → 400 가능.
- **② Anthropic 변환 경로 스트리밍 시 `tool_use` 누락**: `AnthropicProvider.stream()`이 `text_stream`만 처리 → 스트리밍 모드에서 함수 호출이 빠짐(비스트리밍 `chat()`은 정상).
- **③ `finish_reason` 비변환**: Anthropic 원본 `stop_reason`(`end_turn`/`max_tokens` 등)을 그대로 노출 → OpenAI의 `stop`/`length`/`tool_calls`와 다름. (네이티브 `/v1/messages`는 애초에 Anthropic 포맷이라 무관.)

## 새 Provider 추가 절차

1. `app/providers/<name>.py` 생성, `LLMProvider` 상속
2. `chat(request, spec)` — 단일 응답 반환 (dict, OpenAI 형식). 모델명은 `spec.upstream` 사용
3. `stream(request, spec)` — `AsyncGenerator[str, None]`, SSE (`data: {...}\n\n`, 종료 시 `data: [DONE]\n\n`)
4. `aclose()` — 보유 client 정리 (풀 종료 시 호출됨)
5. `app/service.py` → `ProviderPool.__init__` 에 인스턴스 등록
6. `app/registry.py` → `MODELS`/`_passthrough_spec` 에 라우팅 추가

## 개발 / 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload          # 개발 (자동 리로드)
# 운영: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- 테스트: `tests/`(pytest — test_passthrough·test_auto_route·test_observability·test_p1). 실행: `pip install pytest && python -m pytest tests/ -q`. passthrough 보존, auto 라우팅(티어·안전 마진·overflow·vision), 동적 회로 쿨다운(Retry-After/RetryInfo 파싱·클램프), UsageTracker(3종 usage 정규화·에러 집계), embeddings 라우팅/payload, 응답 캐시(키 조건·LRU), 예산 가드(스킵→폴백/402), 레이트리밋·복수 키를 회귀로 검증. 배포 이미지에는 pytest 미포함(런타임 비대화 방지). **deploy.yml이 pytest 게이트를 통과해야만 배포**(test job → needs: test).

## 로컬 Ollama 모델

Oracle 서버 (4 OCPU, 24GB RAM) 기준 추천 모델:

```bash
ollama pull qwen3:14b           # 속도/품질 균형 (~9GB)
ollama pull qwen3.6:27b         # 품질 우선 (~17GB)
ollama pull nomic-embed-text    # /v1/embeddings 로컬 폴백용 (~275MB)
```
