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
- **pydantic-settings** — 환경변수 관리

## 핵심 구조

```
app/
├── main.py        — 엔드포인트 + lifespan(풀 생성/정리): POST /v1/chat/completions, GET /v1/models, GET /health
├── config.py      — Settings (GOOGLE_AI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_BASE_URL, GATEWAY_API_KEY, BREAKER_COOLDOWN_SECONDS)
├── models.py      — ChatCompletionRequest 등 OpenAI 호환 Pydantic 모델
├── registry.py    — 라우팅 결정: ModelSpec(+cost_tier/is_free 메타) / RouteDecision / MODELS / ALIASES / DEFAULT_MODEL / resolve()
├── service.py     — ProviderPool(인스턴스 재사용) + CircuitBreaker + RouteTrace + fallback 실행 + 에러 분류
└── providers/
    ├── base.py       — LLMProvider ABC: chat(request, spec) / stream(request, spec) / aclose()
    ├── ollama.py     — Ollama의 OpenAI 호환 API로 그대로 프록시 (영속 httpx client)
    ├── gemini.py     — Gemini의 OpenAI 호환 엔드포인트로 프록시 (영속 httpx client, Bearer 인증)
    └── anthropic.py  — OpenAI ↔ Anthropic 포맷 변환 후 SDK 호출 (client 재사용)
```

> 과거의 `router.py`는 제거됨. 라우팅 결정은 `registry.resolve()`, provider 선택/실행은 `service.py`가 담당한다.

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

### auto 라우트 (Phase 2~3 — `_auto_route`)

`model="auto"`면 요청 특성으로 게이트웨이가 직접 모델 선택:
- **난이도 분기(Phase 3)**: `_classify_complexity()`가 simple/complex 판단 → `AUTO_CANDIDATES_BY_TIER[tier]` 선택.
  - complex 조건(하나라도): `tools` 사용 / `_estimate_tokens` ≥ `_COMPLEX_TOKEN_THRESHOLD(1200)` / 메시지 수 ≥ `_COMPLEX_MESSAGE_COUNT(6)` / 사용자 메시지에 `_COMPLEX_KEYWORDS` 포함.
  - simple → [gemini-2.5-flash-lite, ollama/qwen3:14b] / complex → [gemini-2.5-flash, ollama/qwen3.6:27b]. 모두 무료, **과금(Claude) 미포함**(비용 0 보장).
- **capability 필터(Phase 2)**: `tools` 있는데 `supports_tools=False`면 제외, `_estimate_tokens()`(문자수÷`_CHARS_PER_TOKEN=3`) > `context_window`면 제외. 모두 탈락 시 DEFAULT_MODEL로 폴백.
- 선택 사유는 `RouteDecision.reason`("auto:tier=...")으로 실려 `x-llm-route` 헤더에 `reason=`으로 노출. 살아남은 후보가 그대로 체인이 되어 Phase 1 회로차단기·폴백 동일 적용.
- capability 메타는 `ModelSpec.supports_tools`·`context_window`(MODELS에서 모델별 지정).

> 패스스루는 **명시적 prefix 단서(`ollama/`/`anthropic/`/`claude-`)를 콜론보다 먼저** 평가한다. 따라서 미등록 `claude-x:snapshot` 도 Anthropic으로 가고, `qwen3:14b` 처럼 prefix 없이 콜론만 있는 건 Ollama로 간다.
> 모델명은 `resolve()`에서 `strip()` 으로 앞뒤 공백을 제거한다.

## Fallback / 풀링 (service.py)

- **ProviderPool**: lifespan 시작 시 provider 인스턴스를 1회 생성해 재사용(httpx keep-alive, `AsyncAnthropic` 재사용), 종료 시 `aclose()`. **Anthropic은 `ANTHROPIC_API_KEY`가 있을 때만 등록** — 빈 키로 SDK 초기화하다 기동이 깨지는 걸 막고, Ollama 전용 배포에서 claude 요청이 와도 `ProviderUnavailable → 로컬 fallback`으로 동작하게 함.
- **chat_with_fallback / stream_with_fallback**: `RouteDecision.chain`을 순서대로 시도. `ProviderUnavailable` 또는 재시도 가능 에러면 다음 후보로.
- **CircuitBreaker (Phase 1)**: provider별 일시 장애(429/5xx/연결오류)를 기억해 `BREAKER_COOLDOWN_SECONDS` 동안 그 provider를 폴백 체인 **뒤로 미룬다**(`_order_by_breaker`). 무료 티어 Gemini가 429로 막히면 잠깐 Ollama를 우선시켜 헛때리는 지연을 제거. 쿨다운 만료 시 자동 복귀(half-open), 성공 시 즉시 닫힘. `ProviderUnavailable`(키 미설정)은 영구 설정 문제라 회로차단 대상이 아님. 단일 프로세스 코루틴 공유 상태(읽기-쓰기 사이 await 없어 락 불필요).
- **RouteTrace / `x-llm-route` 헤더 (Phase 1)**: 실제로 어떤 후보가 응답했는지 추적해 응답 헤더로 노출(`requested=`/`served=`/`fallback=1`/`deferred=`). 응답 **본문은 OpenAI 형식 그대로** 두므로 호출 측 SDK 호환 유지. silent fallback 관측·로깅(`logger = logging.getLogger("llm_gateway")`)에 사용.
- **스트리밍 fallback은 첫 청크 전까지만** 가능(이미 바이트를 보낸 뒤엔 불가). 어떤 경로로 끝나든 `aclose_quietly()`로 업스트림 스트림을 정리해 커넥션 누수를 막는다.

## 환경변수 (config.py)

`.env`에서 로드 (pydantic-settings, 변수명 대소문자 무시).

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GOOGLE_AI_API_KEY` | `""` | Gemini API 키. `gemini-*`(기본 모델) 사용 시 필요. 비면 Gemini 미등록 → 로컬 fallback |
| `ANTHROPIC_API_KEY` | `""` | Anthropic API 키. `claude-*` 모델 사용 시 필요. 비면 Anthropic 미등록 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 서버 주소 |
| `GATEWAY_API_KEY` | `""` | 게이트웨이 공유 인증 토큰. 설정 시 `/v1/*`에 `Authorization: Bearer <키>` 필요 |
| `BREAKER_COOLDOWN_SECONDS` | `30.0` | 회로차단기 쿨다운(초). 일시 장애 provider를 이 시간 동안 폴백 체인 뒤로 미룸. `0` 이하면 비활성화 |

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
- **`tool_choice`는 Ollama로만 전달**: `AnthropicProvider`는 `tool_choice`를 변환/전달하지 않음.
- **silent fallback 주의**: 등록된 모델이 404/연결오류거나 키 미설정이면 fallback 체인의 다음 후보(로컬 Ollama 등)로 조용히 떨어질 수 있다. 실제 사용된 모델은 응답 `model` 필드 또는 **`x-llm-route` 헤더**로 확인.
- **게이트웨이 자체 인증 없음**: 앞단에 인증/레이트리밋 없음. 외부 노출 시 별도 보호 필요.

### 미구현 / 기존 한계 (라우팅과 별개, 추후 과제)

- **assistant `tool_calls` → Anthropic `tool_use` 역변환 없음**: `models.py`의 `Message`에 `tool_calls` 필드가 없어, OpenAI 멀티턴 tool 대화를 Anthropic으로 보내면 직전 assistant의 `tool_use`가 누락 → Anthropic이 400으로 거부될 수 있음.
- **Anthropic 스트리밍 시 `tool_use` 누락**: `stream()`이 `text_stream`만 처리 → 스트리밍 모드에서 함수 호출이 빠짐(비스트리밍 `chat()`은 정상).
- **`finish_reason` 비변환**: Anthropic 원본 `stop_reason`(`end_turn`/`max_tokens` 등)을 그대로 노출 → OpenAI의 `stop`/`length`/`tool_calls`와 다름.

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

- 테스트 코드 없음 (pytest 미구성). 추가 시 `tests/` 디렉터리 사용.

## 로컬 Ollama 모델

Oracle 서버 (4 OCPU, 24GB RAM) 기준 추천 모델:

```bash
ollama pull qwen3:14b       # 속도/품질 균형 (~9GB)
ollama pull qwen3.6:27b     # 품질 우선 (~17GB)
```
