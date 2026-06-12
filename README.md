# llm-server

여러 LLM provider를 단일 **OpenAI 호환 API**로 묶어주는 경량 게이트웨이.

에이전트나 클라이언트는 이 서버 하나만 바라보면 되고, 뒤에서 어떤 모델을 쓰든 코드 변경 없이 **모델 이름만 바꿔** 전환할 수 있다.

## 지원 Provider

| Provider | 모델 이름 형식 | 비고 |
|----------|-------------|------|
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
| `ANTHROPIC_API_KEY` | `""` (빈 값) | Anthropic API 키. `claude-*` 모델 사용 시 필요 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 서버 주소. 온프레미스 호스트로 변경 가능 |

## API

OpenAI SDK / 호환 클라이언트에서 `base_url`만 바꾸면 바로 사용 가능하다.

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

### 모델 목록

```
GET /v1/models
```

> `app/registry.py`의 `MODELS` 레지스트리에서 자동 생성된다(실제 Ollama 설치 모델을 조회하지는 않음). 모델 추가는 `MODELS`만 수정하면 이 목록에도 반영된다.

### 헬스 체크

```
GET /health   →   { "status": "ok" }
```

### curl로 빠른 테스트

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ollama/qwen3:14b","messages":[{"role":"user","content":"안녕"}]}'
```

### Python SDK 연동 예시

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # 게이트웨이 자체는 키 검증 안 함
)

response = await client.chat.completions.create(
    model="ollama/qwen3:14b",
    messages=[{"role": "user", "content": "안녕"}],
)
```

## 라우팅 · 별칭 · Fallback

`app/registry.py`의 `resolve()`가 모델 이름을 후보 체인으로 변환한다. **별칭 → 레지스트리 → 패스스루 → 기본값** 순.

| 단계 | 처리 |
|------|------|
| 별칭(`ALIASES`) | `fast` → `ollama/qwen3:14b`, `smart` → `claude-sonnet-4-6` 등으로 치환 |
| 레지스트리(`MODELS`) | 등록된 모델이면 그 spec(+fallback) 사용 |
| 패스스루 | 미등록이라도 `ollama/`·`anthropic/`·`claude-` prefix, 그 외 `:` 포함으로 provider 추론 |
| 기본값(`DEFAULT_MODEL`) | 위에 안 걸리는/모르는 모델은 로컬 Ollama로 |

- **명시적 prefix가 콜론보다 우선**한다 → 미등록 `claude-x:snapshot`도 Anthropic으로, prefix 없는 `qwen3:14b`는 Ollama로 라우팅된다.
- **Fallback**: 레지스트리 모델에 `fallback`을 지정하면(예: `claude-sonnet-4-6` → `ollama/qwen3.6:27b`) provider 장애·과부하(연결/타임아웃/5xx/529/모델없음)나 키 미설정 시 다음 후보로 자동 전환된다. 스트리밍은 첫 토큰 전까지만 fallback 가능.
- **모델 추가/별칭/fallback 변경은 `app/registry.py`의 `MODELS`·`ALIASES`만** 수정하면 된다(`/v1/models` 목록도 자동 반영).

## 제약사항 / 알려진 한계

- **게이트웨이 자체 인증 없음** — API 키 검증·레이트리밋이 없다. 외부에 노출한다면 reverse proxy 등에서 별도 보호 필요.
- **silent fallback** — 등록된 claude 모델이 장애·키 미설정이면 fallback 체인의 로컬 Ollama로 조용히 떨어질 수 있다. 실제 사용된 모델은 응답 `model` 필드로 확인.
- **스트리밍 fallback은 첫 토큰 전까지만** — 토큰이 나가기 시작한 뒤 업스트림이 끊기면 스트림이 중단된다(복구 불가).
- **`tool_choice`는 Ollama로만 전달** — Anthropic provider는 `tool_choice`를 변환/전달하지 않는다.
- **Anthropic 멀티턴 tool / 스트리밍 tool_use 미지원** — assistant `tool_calls`를 Anthropic `tool_use`로 역변환하지 않아 tool 왕복 대화가 깨질 수 있고, 스트리밍 시 함수 호출이 누락된다. `finish_reason`도 Anthropic 원본 값(`end_turn` 등)을 그대로 노출한다.

## 프로젝트 구조

```
llm-server/
├── app/
│   ├── main.py           # FastAPI 앱, 엔드포인트 정의
│   ├── config.py         # 환경변수 설정 (Settings)
│   ├── models.py         # Pydantic 모델 (OpenAI 호환 요청)
│   ├── registry.py       # 라우팅 결정 (ModelSpec/MODELS/ALIASES/resolve)
│   ├── service.py        # ProviderPool(재사용) + fallback 실행
│   └── providers/
│       ├── base.py       # Provider 추상 클래스 (LLMProvider)
│       ├── ollama.py     # Ollama provider (OpenAI API 프록시)
│       └── anthropic.py  # Anthropic provider (포맷 변환 포함)
├── requirements.txt
├── .gitignore
└── .claude/              # Claude Code 프로젝트 설정 + 가이드
```

## 새 Provider 추가하기

1. `app/providers/` 에 새 파일 생성 (`LLMProvider` 상속), `chat(request, spec)`·`stream(request, spec)`·`aclose()` 구현
2. `app/service.py` 의 `ProviderPool` 에 인스턴스 등록
3. `app/registry.py` 의 `MODELS`/`_passthrough_spec` 에 라우팅 추가 (`/v1/models` 목록은 `MODELS`에서 자동 반영)

## Oracle 서버 배포

Oracle Always Free ARM (4 OCPU, 24GB RAM) 기준.

```bash
# 서비스로 등록 (systemd)
sudo tee /etc/systemd/system/llm-server.service > /dev/null <<EOF
[Unit]
Description=LLM Gateway Server
After=network.target

[Service]
WorkingDirectory=/home/ubuntu/llm-server
ExecStart=/home/ubuntu/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable llm-server
sudo systemctl start llm-server
```
