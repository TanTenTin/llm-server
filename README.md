# llm-server

여러 LLM provider를 단일 **OpenAI 호환 API**로 묶어주는 경량 게이트웨이.

에이전트나 클라이언트는 이 서버 하나만 바라보면 되고, 뒤에서 어떤 모델을 쓰든 코드 변경 없이 **모델 이름만 바꿔** 전환할 수 있다.

OpenAI 포맷뿐 아니라 **Anthropic Messages**(`POST /v1/messages`)·**Gemini generateContent**(`POST /v1beta/models/{model}:generateContent`) **네이티브 포맷 엔드포인트**도 제공한다. 각 SDK(anthropic·google-genai)를 `base_url`만 바꿔 그대로 붙일 수 있다. 네이티브 엔드포인트는 **순수 포맷 어댑터**라서, 입출력 포맷만 해당 네이티브로 맞출 뿐 `model` 필드는 그대로 라우팅을 결정한다 — 즉 `/v1/messages`로 `gemini-2.5-flash`를 요청하면 Anthropic 포맷으로 받아 Gemini로 라우팅·폴백된다.

## 지원 Provider

| Provider | 모델 이름 형식 | 비고 |
|----------|-------------|------|
| **Gemini** (무료 티어) | `gemini-2.5-flash`, `gemini-2.5-flash-lite` | 기본 provider (`DEFAULT_MODEL`). `GOOGLE_AI_API_KEY` 필요 |
| **Ollama** (로컬) | `ollama/qwen3:14b` 또는 `qwen3:14b` | Oracle 온프레미스 서버에서 실행 |
| **Anthropic** | `claude-sonnet-4-6`, `claude-opus-4-7` | API 키 필요 |

## 아키텍처

모델 이름으로 provider를 결정(registry)하고, 재사용 인스턴스 풀로 호출하며, 실패 시 fallback 체인을 따른다.

```
클라이언트 / 에이전트
        │
        │  POST /v1/chat/completions   (OpenAI 호환 요청)
        ▼
┌────────────────────────────────────────────────────┐
│                     llm-server                      │
│                                                     │
│   registry.resolve(model) → 후보 체인               │
│        │   ProviderPool(인스턴스 재사용) + fallback │
│        ├─ ollama/* · "qwen3:14b"  ──────────────────┼──▶ Ollama  (OpenAI 호환 API 그대로 프록시)
│        │   (그 외 ":" 포함 / 기본값)                │     localhost:11434
│        │                                            │
│        └─ claude-* · anthropic/*  ──────────────────┼──▶ Anthropic API  (OpenAI ↔ Anthropic 변환)
│            (실패 시 체인의 다음 후보로 자동 전환)   │
└────────────────────────────────────────────────────┘
```

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

루트에 `.env` 파일을 만들고 값을 채운다. (Ollama만 쓸 경우 `ANTHROPIC_API_KEY`는 비워둬도 됨)

**PowerShell**

```powershell
@'
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
'@ | Set-Content -Encoding utf8 .env
```

**bash**

```bash
cat > .env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
EOF
```

설정 가능한 환경변수는 아래 [환경변수](#환경변수) 표 참고.

### 3. Ollama 설치 및 모델 다운로드 (로컬 사용 시)

```bash
# Ollama 설치 (Linux/ARM 포함)
curl -fsSL https://ollama.ai/install.sh | sh

# 모델 다운로드
ollama pull qwen3:14b       # 무난한 선택 (~9GB)
ollama pull qwen3.6:27b     # 품질 우선 (~17GB, Oracle 24GB RAM 서버 권장)
```

### 4. 서버 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

기동 후 `GET http://localhost:8000/health` 로 확인.

## 환경변수

`app/config.py`의 `Settings`가 `.env`에서 로드한다 (pydantic-settings, 변수명 대소문자 무시).

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GOOGLE_AI_API_KEY` | `""` (빈 값) | Gemini API 키. `gemini-*`(기본 모델) 사용 시 필요. 비면 Gemini 미등록 → 로컬 fallback |
| `ANTHROPIC_API_KEY` | `""` (빈 값) | Anthropic API 키. `claude-*` 모델 사용 시 필요 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 서버 주소. Docker 배포 시 compose가 `http://ollama:11434`로 주입 |
| `OLLAMA_NUM_CTX` | `16384` | Ollama 요청 컨텍스트 창(num_ctx, 토큰). 게이트웨이는 Ollama를 **네이티브 `/api/chat`** 경로로 호출해 이 값을 주입한다(OpenAI 호환 `/v1` 경로는 num_ctx를 못 받아 서버 기본 2~4k로 프롬프트가 잘림 — 코딩 에이전트의 큰 시스템 프롬프트·도구 정의 손실의 주원인). `0` 이하면 미지정(서버 기본). 원격 Ollama 호스트에 `OLLAMA_CONTEXT_LENGTH` env를 함께 두면 서버 기본값도 올라간다 |
| `OLLAMA_DISABLE_THINK` | `true` | qwen3 등 thinking 계열 모델의 내부 reasoning을 기본 비활성화. 요청이 `think`를 명시하지 않고 업스트림이 thinking 모델일 때 `think=false`를 보낸다 — 에이전트 요청에서 reasoning이 출력 예산을 다 써 `content`가 비는 문제를 막는다. thinking 미지원 모델(gemma 등)엔 보내지 않으며, 그래도 400이 나면 think 없이 자동 재시도한다 |
| `GATEWAY_API_KEY` | `""` (빈 값) | 게이트웨이 인증 토큰. 설정 시 `/v1/*` 요청에 `Authorization: Bearer <키>` 필요. **쉼표 구분 복수 키 지원**(`"key-a, key-b"` — 클라이언트별 발급/회수). **외부 노출 시 반드시 설정** |
| `BREAKER_COOLDOWN_SECONDS` | `30.0` | 회로차단기 쿨다운(초). provider가 일시 장애(429/5xx/연결오류)를 내면 이 시간 동안 폴백 체인 뒤로 미뤄 헛때리는 지연을 제거하고, 만료 시 자동 복귀. `0` 이하면 비활성화. 업스트림 429의 `Retry-After`/`RetryInfo` 힌트가 있으면 그 값을 우선(상한 1시간) |
| `CACHE_TTL_SECONDS` | `300.0` | 응답 캐시 TTL(초). 비스트리밍 + `temperature` 미지정/0 인 동일 요청(`/v1/chat/completions`)을 캐시해 무료 쿼터 소모를 줄인다. `0` 이하면 비활성화 |
| `PAID_DAILY_TOKEN_BUDGET` | `0` (무제한) | 과금(paid) provider 일일 토큰 예산. Claude 등 유료 모델의 하루(UTC) 토큰 합이 넘으면 해당 후보를 건너뜀(무료 폴백이 있으면 그쪽으로, 없으면 402). 에이전트 루프 폭주로 인한 과금 사고 방지 |
| `RATE_LIMIT_RPM` | `0` (무제한) | 분당 요청 상한. 키 설정 시 키별, 미설정 시 클라이언트 IP별로 집계. 초과 시 429 + `Retry-After` |
| `REALTIME_DEFAULT_MODEL` | `gemini-2.5-flash-native-audio-preview-09-2025` | `/v1/realtime`에서 클라이언트가 모델을 지정하지 않을 때 쓸 Gemini Live 모델 id. **계정에서 사용 가능한 정확한 id로 교체할 것** |
| `REALTIME_INPUT_SAMPLE_RATE` | `24000` | 클라이언트가 보내는 입력 PCM16 샘플레이트(Hz). OpenAI Realtime 기본 24000. Gemini Live 입력은 16000을 요구하므로 다르면 브리지가 16kHz로 리샘플(같으면 건너뜀) |

## API

OpenAI SDK / 호환 클라이언트에서 `base_url`만 바꾸면 바로 사용 가능하다.

> **인증**: `GATEWAY_API_KEY`가 설정돼 있으면 `/v1/*` 요청에 `Authorization: Bearer <키>` 헤더가 필요하다(미설정 시 개방, 쉼표 구분 복수 키 허용). `/health`는 인증 없이 접근 가능. `RATE_LIMIT_RPM` 설정 시 키별(미설정이면 IP별) 분당 요청 상한이 걸리고 초과 시 429 + `Retry-After`를 반환한다.
>
> 네이티브 엔드포인트(`/v1/messages`·`generateContent`)는 각 SDK가 키를 싣는 방식을 그대로 받는다 — `Authorization: Bearer <키>` 외에 Anthropic SDK의 `x-api-key: <키>`, Gemini SDK의 `x-goog-api-key: <키>`/`?key=<키>` 도 게이트웨이 토큰으로 검증한다.

### Chat Completions

```
POST /v1/chat/completions
```

**요청 예시 — Ollama**

```json
{
  "model": "ollama/qwen3:14b",
  "messages": [
    { "role": "user", "content": "안녕하세요" }
  ]
}
```

**요청 예시 — Anthropic (tool 사용)**

```json
{
  "model": "claude-sonnet-4-6",
  "messages": [
    { "role": "system", "content": "당신은 CS 상담사입니다." },
    { "role": "user", "content": "주문 취소하고 싶어요" }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "cancel_order",
        "description": "주문을 취소한다",
        "parameters": {
          "type": "object",
          "properties": {
            "order_id": { "type": "string" }
          },
          "required": ["order_id"]
        }
      }
    }
  ]
}
```

**스트리밍** — `stream: true` 시 `text/event-stream`(SSE)으로 응답한다.

```json
{
  "model": "ollama/qwen3:14b",
  "messages": [],
  "stream": true
}
```

> Ollama는 Ollama가 내려주는 SSE 라인을 그대로 전달하고, Anthropic은 토큰 스트림을 OpenAI `chat.completion.chunk` 형식으로 재구성한 뒤 마지막에 `data: [DONE]`을 보낸다.

**응답 형식 (비스트리밍)** — OpenAI `chat.completion` 형식.

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "claude-sonnet-4-6",
  "choices": [
    {
      "index": 0,
      "finish_reason": "end_turn",
      "message": { "role": "assistant", "content": "..." }
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

> Anthropic 응답의 `finish_reason`은 OpenAI의 `"stop"`이 아니라 Anthropic 원본 `stop_reason`(`end_turn` · `tool_use` · `max_tokens` 등)을 그대로 반영한다. `tool_use`인 경우 `message.tool_calls` 배열이 채워진다.

### 네이티브 포맷 엔드포인트 (Anthropic / Gemini)

OpenAI 포맷 외에, 각 provider의 **네이티브 요청/응답 포맷**을 그대로 받는 엔드포인트를 제공한다. 내부적으로는 네이티브 요청을 OpenAI 표준으로 들여와(inbound 어댑터) 동일한 라우팅·폴백·회로차단기 파이프라인을 태우고, 결과를 다시 네이티브 포맷으로 돌려준다(outbound 어댑터). **`model` 필드는 보존**되므로 라우팅은 OpenAI 엔드포인트와 똑같이 동작한다(폴백·`x-llm-route` 헤더 동일 적용).

```
Anthropic SDK ─→ POST /v1/messages ──────────────────┐
                                                      │  네이티브→OpenAI 변환
google-genai ─→ POST .../{model}:generateContent ─────┤        ↓
            └─→ POST .../{model}:streamGenerateContent ┘  route() → fallback (기존 파이프라인)
                                                               ↓  OpenAI 응답
                                              OpenAI→네이티브 역변환 후 응답
```

#### Anthropic Messages

```
POST /v1/messages
```

Anthropic Messages API 요청/응답 포맷. `anthropic` SDK / Claude Code 클라이언트를 `base_url`만 바꿔 붙일 수 있다.

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(
    base_url="http://localhost:8000",   # /v1/messages 는 SDK가 자동으로 붙임
    api_key="<GATEWAY_API_KEY>",         # x-api-key 헤더로 전송 → 게이트웨이가 검증
)
msg = await client.messages.create(
    model="gemini-2.5-flash",            # ← 포맷은 Anthropic, 라우팅은 Gemini로
    max_tokens=256,
    messages=[{"role": "user", "content": "안녕"}],
)
```

> **Anthropic 패스스루 fast-path**: 라우팅된 후보가 **Anthropic provider**(`claude-*`)면, 게이트웨이는 OpenAI 내부표준을 거치지 않고 **클라이언트의 Anthropic body를 그대로 SDK로 보낸다**(이중 변환 회피). 덕분에 `cache_control`(프롬프트 캐싱)·정확한 content 블록·**멀티턴 및 스트리밍 `tool_use`가 손실 없이 보존**된다 — Claude Code처럼 도구를 많이 쓰는 에이전트를 `claude-*`로 붙일 때 핵심. 표준 외 top-level 필드(예: `thinking`·beta 옵션)는 `extra_body`로 그대로 전달한다. 폴백 후보(예: 장애 시 Ollama)는 아래 어댑터 경로(OpenAI 변환 후 Anthropic 응답으로 역변환)를 타며, 회로차단기·재시도·`x-llm-route`는 동일하게 적용된다.

아래 변환은 **폴백 후보(비-Anthropic provider)** 가 응답할 때 적용된다(Anthropic 후보는 위 패스스루로 손실 없음):

- **입력 변환**: `system`(문자열/블록) → system 메시지, `tool_use`↔`tool_calls`, user의 `tool_result` 블록 → OpenAI `tool` 메시지, `image` 블록 → OpenAI `image_url`(data URL), `tool_choice`(`auto`/`any`/`tool`/`none`) 변환, `stop_sequences`→`stop`.
- **응답 변환**: OpenAI `finish_reason` → Anthropic `stop_reason`(`stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`), `tool_calls` → `tool_use` 블록, `usage` → `input_tokens`/`output_tokens`.
- **스트리밍**(`"stream": true`): Anthropic 네이티브 SSE 이벤트 시퀀스로 변환한다 — `message_start` → `content_block_start`/`content_block_delta`(`text_delta`·`tool_use`는 `input_json_delta`)/`content_block_stop` → `message_delta` → `message_stop`.

#### Gemini generateContent

```
POST /v1beta/models/{model}:generateContent          # 비스트리밍
POST /v1beta/models/{model}:streamGenerateContent     # 스트리밍(SSE)
```

Gemini `generateContent` API 요청/응답 포맷. `google-genai` SDK가 붙을 수 있다(`/v1/models/...` 경로도 동일하게 받는다). **모델은 URL 경로**에서 받는다.

```bash
curl "http://localhost:8000/v1beta/models/claude-sonnet-4-6:generateContent?key=$GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"안녕"}]}]}'
# ← 포맷은 Gemini, 라우팅은 Anthropic(claude)으로
```

> **Gemini 패스스루 fast-path**: 라우팅된 후보가 **Gemini provider**면, 게이트웨이는 OpenAI-compat 변환을 거치지 않고 **클라이언트의 Gemini body를 네이티브 `generateContent` 엔드포인트로 그대로 보낸다**. 덕분에 `safetySettings`·`thinkingConfig`·`cachedContent`(컨텍스트 캐싱) 등 **Gemini 전용 필드**와 응답의 네이티브 구조(`candidates`/`usageMetadata`/`safetyRatings`)가 손실 없이 보존된다. 폴백 후보(예: 장애 시 Ollama)는 아래 어댑터 경로(OpenAI 변환 후 Gemini 응답으로 역변환)를 탄다.

아래 변환은 **폴백 후보(비-Gemini provider)** 가 응답할 때 적용된다(Gemini 후보는 위 패스스루로 손실 없음):

- **입력 변환**: `systemInstruction` → system 메시지, `contents`의 role `model`→`assistant`, `functionCall`↔`tool_calls`, `functionResponse` → OpenAI `tool` 메시지, `inlineData` → `image_url`, `generationConfig`(`temperature`/`maxOutputTokens`/`topP`/`topK`/`stopSequences`) 매핑, `toolConfig.functionCallingConfig`(`AUTO`/`ANY`/`NONE`) → `tool_choice`.
- **응답 변환**: `candidates[].content.parts`(text/`functionCall`) + `finishReason`(`stop`→`STOP`, `length`→`MAX_TOKENS`) + `usageMetadata`.
- **스트리밍**(`:streamGenerateContent`): Gemini SSE(`data: <GenerateContentResponse>`)로 변환한다. text 델타는 즉시 흘려보내고, tool 호출은 부분 `arguments`를 누적했다가 마지막 청크에서 완성된 `functionCall`로 내보낸다.

> **네이티브 어댑터 한계(v1)**: ① Gemini `functionResponse`에는 호출 id가 없어 `functionCall`/`functionResponse`를 **이름 기반(`call_<name>`)으로 매칭**한다 — 같은 이름 도구를 한 턴에 여러 번 호출하면 충돌할 수 있다. ② `/v1/messages`의 Anthropic 후보는 패스스루 fast-path라 스트리밍 tool_use가 정상 보존되지만, **OpenAI 엔드포인트(`/v1/chat/completions`)로 `claude-*`를 호출할 때**는 여전히 provider의 스트리밍 tool_use 미지원 한계(아래 [제약사항](#제약사항--알려진-한계))를 따른다. ③ 에러 응답은 게이트웨이 공통 포맷(`{"detail": ...}`)으로, 각 SDK의 네이티브 에러 포맷과 다르다.

### Embeddings

```
POST /v1/embeddings
```

OpenAI 호환 embeddings 엔드포인트(RAG 에이전트용). **Gemini embedding(무료) 우선, 장애·키 미설정 시 로컬 Ollama로 폴백** — chat과 동일한 fallback 루프(회로차단기·`x-llm-route`·사용량 집계)를 재사용한다. Anthropic은 임베딩 API가 없어 라우팅 대상에서 제외된다.

```json
{ "model": "embed", "input": ["문장 1", "문장 2"] }
```

- **모델**: `gemini-embedding-001`(기본) / `ollama/nomic-embed-text` / `ollama/<모델>` 패스스루. OpenAI SDK 기본 모델명(`text-embedding-3-small`/`-large`)과 별칭 `embed`는 기본 모델로 매핑되므로 **클라이언트 코드 수정 없이 연동**된다.
- 로컬 폴백을 쓰려면 서버에 `ollama pull nomic-embed-text` 필요.

### 응답 캐시

`/v1/chat/completions`의 **비스트리밍 + `temperature` 미지정/0** 요청은 `CACHE_TTL_SECONDS`(기본 5분) 동안 exact-match 캐시된다 — 에이전트가 동일 요청을 재시도할 때 업스트림을 다시 때리지 않아 **무료 티어 쿼터를 절약**한다. 캐시 여부는 `x-llm-cache: hit|miss` 헤더로 노출되고, 히트는 `/v1/usage`에 `cache` 라벨로 집계된다. 요청이 1비트라도 다르면(모델·메시지·파라미터) 다른 키가 되어 오염이 없다.

### 모델 목록

```
GET /v1/models
```

**provider에 따라 정적/유동으로 나뉜다.**

- **SaaS(gemini·anthropic)** — `app/registry.py`의 `MODELS`·`EMBEDDING_MODELS` 레지스트리에서 **정적**으로 나열한다. 모델 추가는 레지스트리만 수정하면 이 목록에 반영된다.
- **Ollama(로컬)** — 서버의 `/api/tags`로 **실제 설치된 모델을 실시간 조회**한다(유동). `ollama pull`/`ollama rm`으로 로컬 모델이 바뀌면 **재기동 없이 즉시** 목록에 반영된다. id는 `ollama/<태그>` 형태라 그대로 `model` 파라미터로 호출할 수 있고, 임베딩 모델 여부는 Ollama가 주는 `capabilities`(`["embedding"]`)로 판별한다. Ollama 서버가 응답하지 않으면 레지스트리의 정적 ollama 항목으로 graceful degrade 한다.

각 항목의 `source` 필드(`"registry"` | `"ollama"`)로 정적/유동 출처를 구분할 수 있다(OpenAI 호환 클라이언트는 미지 필드를 무시).

### Realtime (음성) API

```
WS /v1/realtime
```

**OpenAI Realtime API 호환** WebSocket 엔드포인트. 클라이언트는 OpenAI Realtime 이벤트(JSON)를 그대로 주고받고, 게이트웨이가 내부에서 **Gemini Live API**로 양방향 중계한다(음성 입력 → 음성 출력). 무료 티어 Gemini Live를 OpenAI Realtime SDK/클라이언트로 그대로 쓸 수 있다.

- **모델 지정**: `?model=` 쿼리(미지정 시 `REALTIME_DEFAULT_MODEL`). 친화적 별칭 `gemini-live`도 사용 가능(`app/registry.py`의 `LIVE_ALIASES`).
- **인증**: `GATEWAY_API_KEY` 설정 시 `Authorization: Bearer <키>` 헤더 **또는** `?api_key=<키>` 쿼리(브라우저 WS는 헤더를 못 붙이므로 쿼리 허용).
- **오디오 포맷**: 입력 PCM16 24kHz(OpenAI 기본) → 내부에서 16kHz로 리샘플해 Gemini로 전달. 출력은 PCM16 24kHz 패스스루.

**이벤트 매핑** (OpenAI ↔ Gemini Live)

| 방향 | OpenAI 이벤트 | Gemini Live |
|------|--------------|-------------|
| → | `session.update` | setup 설정(instructions/voice/model) 반영 |
| → | `input_audio_buffer.append` | `realtimeInput.audio` (24k→16k 리샘플) |
| → | `conversation.item.create`(input_text) | `realtimeInput.text` |
| → | `input_audio_buffer.commit` / `response.create` | no-op (Gemini 자동 VAD가 턴 감지) |
| ← | `response.audio.delta` | `serverContent.modelTurn.inlineData` (24k 패스스루) |
| ← | `response.audio_transcript.delta` | `serverContent.outputTranscription` |
| ← | `conversation.item.input_audio_transcription.delta` | `serverContent.inputTranscription` |
| ← | `response.done` / `input_audio_buffer.speech_started` | `turnComplete` / `interrupted`(barge-in) |

> **한계(v1)**: 함수 호출(`toolCall`)·이미지/비디오 입력은 미지원(오디오·텍스트만). `session.update`의 model 변경은 첫 입력 전(=setup 전송 전)에만 반영된다(Gemini setup은 1회성). Live는 로컬 대체가 없어 **fallback 하지 않는다**(텍스트 경로의 ProviderPool/회로차단기와 독립).

### 헬스 체크 / 관측

```
GET /health             →   { "status": "ok" }                    # 무인증 생존 확인 (docker healthcheck용)
GET /health/providers   →   provider별 심층 상태 (인증 필요)
GET /v1/usage           →   모델별 사용량 집계 (인증 필요)
```

`/health/providers` — "왜 자꾸 로컬로 폴백되지?"를 로그 없이 진단:

```json
{
  "status": "ok",
  "providers": {
    "gemini":    { "registered": true, "breaker_open": true, "breaker_remaining_seconds": 512.3 },
    "anthropic": { "registered": false },
    "ollama":    { "registered": true, "breaker_open": false, "reachable": true }
  }
}
```

- `registered: false` = API 키 미설정 → 해당 provider 요청은 폴백으로 빠진다.
- `breaker_open: true` = 일시 장애(429 등)로 회로차단기가 열려 폴백 체인 뒤로 밀린 상태.
- `reachable`(Ollama만) = `/api/version` 실측 프로브 — 로컬 폴백의 최후 보루 생존 확인.

`/v1/usage` — 무료 티어 쿼터를 얼마나 썼는지 게이트웨이 스스로 집계(provider:model × UTC 일 단위, 최근 7일, 인메모리라 재기동 시 리셋). 요청 수·프롬프트/응답 토큰·에러 종류별 횟수(`429` 등)·폴백 응답 횟수를 노출한다. 스트리밍 응답은 요청 수만 집계된다.

### curl로 빠른 테스트

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -d '{"model":"ollama/qwen3:14b","messages":[{"role":"user","content":"안녕"}]}'
```

### Python SDK 연동 예시

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="<GATEWAY_API_KEY>",  # GATEWAY_API_KEY 설정 시 필수, 미설정이면 아무 값
)

response = await client.chat.completions.create(
    model="ollama/qwen3:14b",
    messages=[{"role": "user", "content": "안녕"}],
)
```

## 라우팅 · 별칭 · Fallback

`app/registry.py`의 진입점은 `route(request)`다. `model="auto"`면 요청 특성 기반 선택,
그 외엔 이름 기반 `resolve()`(**별칭 → 레지스트리 → 패스스루 → 기본값** 순)로 후보 체인을 만든다.

| 단계 | 처리 |
|------|------|
| 별칭(`ALIASES`) | `fast` → `gemini-2.5-flash-lite`, `smart` → `gemini-2.5-flash` 등으로 치환 |
| 레지스트리(`MODELS`) | 등록된 모델이면 그 spec(+fallback) 사용 |
| 패스스루 | 미등록이라도 `gemini-`·`ollama/`·`anthropic/`·`claude-` prefix, 그 외 `:` 포함으로 provider 추론 |
| 기본값(`DEFAULT_MODEL`) | 위에 안 걸리는/모르는 모델은 `gemini-2.5-flash`(키 없으면 로컬 Ollama로 폴백) |

### auto 라우트 (요청 특성 기반 자동 선택)

`model="auto"`로 보내면 게이트웨이가 **요청 특성을 보고 직접 모델을 고른다**. 클라이언트는 어떤 모델이 있는지 몰라도 된다.

- **티어 분기(`tier`)**: 먼저 입력 크기를 추정하고 요청을 `simple`/`complex`/`long` 셋 중 하나로 분류해 티어별 후보 셋을 고른다.
  - `simple` → `gemini-2.5-flash-lite` → `ollama/qwen3:14b` (가볍고 빠름): 인사·단순 단발 질의
  - `complex` → `gemini-2.5-flash` → `ollama/qwen3:14b` (강한 모델): 도구 사용·긴 입력(≥1200토큰)·긴 멀티턴(≥6메시지)·추론성 키워드(분석/설계/디버그/구현 등)
  - `long` → `gemini-2.5-flash` → `gemini-2.5-flash-lite` (1M 컨텍스트): 추정 입력 ≥25k 토큰. **난이도보다 우선** — 아무리 단순한 요청도 로컬 창(32k)을 넘으면 큰 컨텍스트 모델로 직행한다.
- **후보**: 무료만, 비용·품질 우선순위 순. **과금(Claude)은 auto가 자동 선택하지 않는다**(비용 0 보장 — Claude가 필요하면 모델명을 명시).
- **필터**: 도구(`tools`)를 쓰는 요청인데 도구 미지원 모델은 제외. **이미지(`image_url`)가 포함된 요청인데 vision 미지원 모델도 제외**(`supports_vision` — 이미지가 텍스트 전용 로컬로 폴백돼 조용히 무시되는 것을 방지, reason에 `vision=1` 표시). 컨텍스트는 `추정 입력 + max_tokens(출력 예산)` 이 **usable 창(`context_window` × 0.8 안전 마진)** 을 초과하는 모델을 제외한다 — 추정이 근사치인 데다 로컬은 입력·출력이 한 창을 나눠 쓰기 때문. (예: 입력 20k 토큰 + `max_tokens=8k` 면 32k 로컬은 빠진다.)
- **overflow best-effort**: 모든 후보의 usable 창(1M×0.8=800k)마저 넘는 초대형 입력이면, 후보 중 창이 가장 큰 모델 1개를 best-effort로 시도하고 `reason`에 `overflow=1`을 표시한다(무의미한 DEFAULT_MODEL 폴백 대신 — 진짜 한계 초과면 업스트림 4xx로 드러난다).
- **토큰 추정**: 메시지 + 멀티턴 `tool_calls`(함수 인자) + 도구 정의의 문자 수 ÷ 3 (정확한 토큰화가 아니라 "로컬에 들어가나, 큰 컨텍스트가 필요한가" 판단용 근사치).
- 선택 사유는 `x-llm-route` 헤더에 `reason=auto:tier=complex,est=1500` 형태(티어 + 추정 토큰)로 노출된다. 살아남은 후보가 그대로 폴백 체인이 되므로 **회로차단기·`x-llm-route` 헤더가 동일하게 적용**된다.

- **명시적 prefix가 콜론보다 우선**한다 → 미등록 `claude-x:snapshot`도 Anthropic으로, prefix 없는 `qwen3:14b`는 Ollama로 라우팅된다.
- **Fallback**: 레지스트리 모델에 `fallback`을 지정하면(예: `claude-sonnet-4-6` → `ollama/qwen3:14b`) provider 장애·과부하(연결/타임아웃/5xx/529/모델없음)나 키 미설정 시 다음 후보로 자동 전환된다. 스트리밍은 첫 토큰 전까지만 fallback 가능.
- **회로차단기(circuit breaker)**: 어떤 provider가 일시 장애(429/5xx/연결오류)를 내면 그 provider를 폴백 체인 **뒤로 미룬다**. 무료 티어(Gemini)가 429로 막히면 잠깐 로컬(Ollama)을 우선시켜 **매 요청 primary부터 헛때리는 지연을 제거**한다. 쿨다운이 지나면 자동으로 다시 우선순위에 올린다(half-open). 미뤄진 후보도 다른 후보가 모두 실패하면 결국 시도하므로 누락은 없다.
- **동적 쿨다운(Retry-After 반영)**: 쿨다운은 기본 `BREAKER_COOLDOWN_SECONDS`(30초)지만, 업스트림 429 응답에 `Retry-After` 헤더나 Gemini `RetryInfo`(`retryDelay`)가 실려 있으면 **그 값을 그대로 쿨다운으로 쓴다**(상한 1시간 클램프). RPM 초과(수십 초)와 RPD 소진(수 시간)을 같은 30초로 취급해 헛때리던 문제를 해소 — 힌트가 5초면 5초만 기다리고, 하루 쿼터 소진이면 1시간 단위 half-open 탐침으로 전환된다.
- **과금 예산 가드**: `PAID_DAILY_TOKEN_BUDGET` 설정 시 과금(`is_free=False`) 모델의 하루(UTC) 토큰 합이 예산을 넘으면 **그 후보를 건너뛴다** — 폴백 체인에 무료 후보가 있으면 그쪽으로(trace에 `#budget`), 없으면 402를 반환한다. 에이전트 루프 폭주로 인한 Claude 과금 사고를 게이트웨이 수준에서 차단. (한계: 스트리밍 응답은 토큰이 집계되지 않아 예산 소모로 잡히지 않는다.)
- **관측성(`x-llm-route` 헤더)**: 응답 본문은 OpenAI 형식 그대로 두고, 실제 라우팅 결과는 `x-llm-route` 응답 헤더로 노출한다 — 예: `requested=gemini-2.5-flash; served=ollama:qwen3:14b; fallback=1`. silent fallback이 일어나도 *무엇이 실제로 응답했는지* 헤더/로그로 바로 확인할 수 있다(호출 측 코드 변경 불필요).
- **모델 추가/별칭/fallback 변경은 `app/registry.py`의 `MODELS`·`ALIASES`만** 수정하면 된다(SaaS 모델은 `/v1/models` 목록에도 자동 반영). **로컬 Ollama 모델은 레지스트리와 무관하게** `/api/tags` 실시간 조회로 `/v1/models`에 나타나므로, `ollama pull`/`rm`만으로 목록이 갱신된다(패스스루라 호출도 등록 없이 가능).

### 로컬 전용 추론 (`x-llm-local-only`)

기본 정책은 **모든 로컬 체인 끝에 SaaS 폴백(`gemini-2.5-flash`)을 붙이는 것**이다(`_ensure_saas_fallback`).
로컬이 죽어도 응답을 받게 해주지만, *프롬프트·코드가 외부 provider로 나가면 안 되는 호출자*에게는
`ollama/qwen3:14b`를 콕 집어도 조용히 클라우드로 넘어가는 정책 위반이 된다.

요청에 `x-llm-local-only: 1` 헤더를 실으면 그 폴백을 끈다. **로컬이 죽으면 그냥 실패한다.**

```bash
curl https://llm.tan-kim.com/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "x-llm-local-only: 1" \
  -H "Content-Type: application/json" \
  -d '{"model":"ollama/qwen3:14b","messages":[{"role":"user","content":"안녕"}]}'
```

| 요청 | 기본 체인 | `x-llm-local-only: 1` |
|------|-----------|------------------------|
| `ollama/qwen3:14b` | `[qwen3:14b, gemini-2.5-flash]` | `[qwen3:14b]` |
| `auto` (tools 포함 → complex) | `[gemini-2.5-flash, qwen3:14b]` | `[qwen3:14b]` |
| `auto` (입력 ≥25k → long) | `[gemini-2.5-flash, gemini-2.5-flash-lite]` | `[qwen3:14b]` |
| `gemini-2.5-flash` (SaaS 명시) | `[gemini-2.5-flash]` | `[qwen3:14b]` (로컬로 강등) |

- 참으로 인정하는 값: `1`·`true`·`yes`·`on` (대소문자·앞뒤 공백 무시). 그 외/미지정은 거짓.
- SaaS 모델을 명시해도 **로컬로 강등**된다. 조용한 강등이 아니라 `x-llm-route` 헤더의
  `reason=local_only=1`로 드러나므로 호출자가 확인할 수 있다.
- long 티어는 후보가 전부 SaaS라 전멸하는데, 이때 `DEFAULT_MODEL`(로컬)로 되돌아온다.
  입력이 로컬 창을 정말 넘으면 폴백으로 뭉개지 않고 `413 ContextTooLarge`로 정직하게 거절한다.
- **응답 캐시 공간이 분리된다.** 같은 body라도 일반 요청은 Gemini가 만든 응답을 캐시에 남길 수 있는데,
  그걸 로컬 전용 호출자에게 돌려주면 보장이 깨지기 때문이다(캐시 키에 `:local-only` 접미사).
- 적용 범위는 `/v1/chat/completions`다. 네이티브 포맷 엔드포인트(`/v1/messages` 등)는 아직 지원하지 않는다.

## 자동 모델 감시 (model-watch)

`.github/workflows/model-watch.yml`가 **매일 09:00 KST**에 provider의 모델 목록 API를
조회해 `app/registry.py`의 `MODELS`와 비교한다(릴리스 노트 스크래핑이 아니라 models API가 신뢰원).

- **감지**: `new`(라이브엔 있는데 레지스트리에 없는 새 무료 모델 후보) / `removed`(레지스트리엔 있는데 라이브에서 사라진 deprecated 후보). preview·스냅샷 변형은 노이즈로 제외.
- **알림**: 드리프트가 있으면 단일 GitHub 이슈(`model-drift` 라벨)를 생성/갱신하고, 해소되면 자동 종료한다(매일 중복 이슈 없음). **코드는 자동 수정하지 않는다** — 이슈의 제안 diff를 보고 사람이 `registry.py`를 반영한다.
- **provider 추가**: `scripts/check_models.py`의 `ModelSource`를 상속한 클래스 1개 작성 + `build_sources()`에 등록하면 끝(현재 Gemini만 활성).
- **필요 설정**: GitHub Secret `GOOGLE_AI_API_KEY`(models 목록 조회용, 읽기 전용). 이슈 생성은 기본 `GITHUB_TOKEN`으로 동작.

## 제약사항 / 알려진 한계

- **게이트웨이 자체 인증 없음** — API 키 검증·레이트리밋이 없다. 외부에 노출한다면 reverse proxy 등에서 별도 보호 필요.
- **silent fallback** — 등록된 모델이 장애·키 미설정이면 fallback 체인의 다음 후보(로컬 Ollama 등)로 조용히 떨어질 수 있다. 실제 사용된 모델은 응답 `model` 필드 또는 **`x-llm-route` 헤더**로 확인(헤더에 `served=`·`fallback=1`·`deferred=` 표기).
- **스트리밍 fallback은 첫 토큰 전까지만** — 토큰이 나가기 시작한 뒤 업스트림이 끊기면 스트림이 중단된다(복구 불가).
- **`tool_choice`는 OpenAI 패스스루(Gemini·Ollama)로 전달** — Anthropic provider는 여전히 `tool_choice`를 변환/전달하지 않는다.
- **OpenAI 패스스루는 미지 필드를 보존한다** — 요청/메시지 모델이 `extra="allow"`라, 게이트웨이가 모르는 메시지 구조 필드(예: `tool_calls`)도 버리지 않고 그대로 업스트림에 전달한다. 요청 레벨 파라미터는 `app/providers/openai_payload.py`의 화이트리스트(`temperature`·`top_p`·`stop`·`response_format`·`tool_choice` 등)로 전달하며, 미지의 요청 레벨 필드는 엄격한 업스트림의 400을 피하려 전달하지 않는다(메시지는 보존, 요청 파라미터는 선별).
- **Anthropic 멀티턴 tool / 스트리밍 tool_use 미지원 (OpenAI 엔드포인트 한정)** — `/v1/chat/completions`로 `claude-*`를 호출하면 `AnthropicProvider`가 OpenAI→Anthropic 변환을 하는데, assistant `tool_calls`를 Anthropic `tool_use`로 역변환하지 않아 tool 왕복 대화가 깨질 수 있고 스트리밍 시 함수 호출이 누락된다. **`/v1/messages`(네이티브 엔드포인트)로 `claude-*`를 호출하면 패스스루 fast-path가 이 변환 자체를 건너뛰어 정상 동작한다** — Claude Code 등 Anthropic SDK 클라이언트는 이 경로를 쓰면 된다.

## 프로젝트 구조

```
llm-server/
├── app/
│   ├── main.py           # FastAPI 앱, 엔드포인트 정의
│   ├── config.py         # 환경변수 설정 (Settings)
│   ├── models.py         # Pydantic 모델 (OpenAI 호환 요청)
│   ├── registry.py       # 라우팅 결정 (ModelSpec/MODELS/ALIASES/resolve, LIVE_ALIASES)
│   ├── service.py        # ProviderPool(재사용) + fallback 실행
│   ├── realtime.py       # /v1/realtime — OpenAI Realtime ↔ Gemini Live 음성 브리지
│   ├── audio.py          # PCM16 리샘플러 (24k→16k, 순수 파이썬)
│   ├── adapters/         # 네이티브 포맷 어댑터 (네이티브 요청 ⇄ OpenAI 내부표준)
│   │   ├── anthropic_io.py   # Anthropic Messages ⇄ OpenAI (/v1/messages)
│   │   ├── gemini_io.py      # Gemini generateContent ⇄ OpenAI (.../{model}:generateContent)
│   │   └── sse.py            # 어댑터 공용 SSE 헬퍼
│   └── providers/
│       ├── base.py            # Provider 추상 클래스 (LLMProvider)
│       ├── openai_payload.py  # OpenAI 패스스루 payload 공용 빌더
│       ├── ollama.py          # Ollama provider (OpenAI API 프록시)
│       ├── gemini.py          # Gemini provider (OpenAI 호환 엔드포인트 프록시)
│       └── anthropic.py       # Anthropic provider (포맷 변환 포함)
├── Dockerfile            # llm-server 컨테이너 이미지 (python:3.12-slim)
├── docker-compose.yml    # llm-server + ollama (Oracle 단일 호스트)
├── .dockerignore
├── requirements.txt
├── .gitignore
├── .github/workflows/
│   └── deploy.yml        # CI/CD: push main → Oracle SSH 배포
└── .claude/              # Claude Code 프로젝트 설정 + 가이드
```

## 새 Provider 추가하기

1. `app/providers/` 에 새 파일 생성 (`LLMProvider` 상속), `chat(request, spec)`·`stream(request, spec)`·`aclose()` 구현
2. `app/service.py` 의 `ProviderPool` 에 인스턴스 등록
3. `app/registry.py` 의 `MODELS`/`_passthrough_spec` 에 라우팅 추가 (SaaS provider면 `/v1/models` 목록은 `MODELS`에서 자동 반영. Ollama처럼 설치 모델을 실시간 조회하려면 provider에 목록 조회 메서드를 만들고 `/v1/models` 핸들러에서 병합)

## 배포 (Oracle + Docker Compose + CI/CD)

Oracle Always Free ARM (4 OCPU, 24GB RAM)에서 **Docker Compose**로 `llm-server`를 띄운다
(Ollama는 이 호스트가 아닌 **원격 로컬-LLM 서버** — `OLLAMA_BASE_URL`로 접속, `62a97d4`).
`main` 브랜치 push 시 **GitHub Actions**가 SSH로 자동 배포한다.
외부 도메인(`llm.tan-kim.com`) 노출은 **같은 호스트의 nginx(+Cloudflare)**가 담당한다 —
컨테이너는 `127.0.0.1:8000`만 퍼블리시한다(`32e6420`).

```
   Cloudflare ──▶ Oracle 호스트 nginx (llm.tan-kim.com)
                     ├─ location /       →  127.0.0.1:8000  (llm-server)
                     └─ location /aiobs/ →  127.0.0.1:8081  (ai-observability collector)
                     ▼
   Oracle ─ docker compose
        llm-server 127.0.0.1:8000 ──(OLLAMA_BASE_URL)──▶ 원격 로컬-LLM :11434
```

### 1. 최초 1회 — 서버 준비

```bash
# Docker / compose 플러그인 설치는 사전 완료 가정
git clone https://github.com/<사용자명>/llm-server.git ~/llm-server
cd ~/llm-server

# .env 작성 (git 제외 대상)
cat > .env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
GATEWAY_API_KEY=<충분히-긴-랜덤-토큰>
EOF

docker compose up -d --build

# Ollama 모델은 원격 로컬-LLM 서버에서 관리한다 (이 compose에 ollama 서비스 없음).
# 레지스트리가 참조하는 모델은 그 서버에서 pull 해둔다: ollama pull qwen3:14b 등
```

### 2. CI/CD (GitHub Actions)

`main`에 `app/**` · `tests/**` · `Dockerfile` · `docker-compose.yml` · `requirements.txt` 변경을 push하면
`.github/workflows/deploy.yml`이 **pytest를 먼저 실행하고(테스트 게이트), 통과 시에만**
서버에서 `git pull`(`.env`는 gitignore라 보존) → `docker compose up -d --build` 한다.
테스트가 깨지면 운영에 나가지 않는다.

필요한 GitHub Secrets:

| Secret | 값 |
|--------|----|
| `ORACLE_HOST` | Oracle 인스턴스 공인 IP |
| `ORACLE_USER` | SSH 유저 (Ubuntu 이미지 `ubuntu`, Oracle Linux `opc`) |
| `ORACLE_SSH_KEY` | SSH 개인키 전체 내용 |

### 3. 네트워크 / 방화벽

- **컨테이너는 `127.0.0.1:8000`만 퍼블리시** — 외부에서 직접 접근할 수 없고,
  같은 호스트의 nginx가 `llm.tan-kim.com`으로 프록시한다. 포트 개방(VCN/iptables)이 필요 없다.
- **DNS**: `llm.tan-kim.com` → Oracle (Cloudflare 경유). TLS는 Cloudflare/nginx 계층에서 처리한다.
- **인증**: `GATEWAY_API_KEY`를 설정하면 `/v1/*` 호출에 `Authorization: Bearer <키>`가 필요하다. 공인 노출 시 필수.
- **같은 호스트의 다른 경로**: `llm.tan-kim.com/aiobs/*`는 ai-observability collector(127.0.0.1:8081)로
  프록시된다 — 이 게이트웨이와 무관한 별도 서비스다 (ai-observability 레포 `docs/deploy.md` 참조).
