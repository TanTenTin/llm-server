import json
import time
import uuid
import httpx
from typing import AsyncGenerator

from app.models import ChatCompletionRequest, EmbeddingsRequest
from app.providers.base import LLMProvider
from app.providers.openai_payload import build_embeddings_payload, build_openai_payload
from app.registry import ModelSpec

TIMEOUT = 120.0


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str):
        # Ollama는 /v1/chat/completions 엔드포인트로 OpenAI 호환 API를 제공함.
        # client는 앱 생애주기 동안 재사용 → 커넥션 keep-alive.
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=TIMEOUT)

    def _build_payload(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        # 공용 OpenAI 패스스루 빌더 사용 (think는 네이티브 /api/chat 경로에서 별도 처리)
        return build_openai_payload(request, spec)

    def _build_native_payload(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        # /api/chat 네이티브 엔드포인트용 payload.
        # think 파라미터가 /v1/chat/completions 에서 무시되므로 네이티브 API를 사용.
        payload: dict = {
            "model": spec.upstream,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "stream": request.stream or False,
            "think": request.think,
        }
        options: dict = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        max_tokens = request.max_tokens or spec.max_tokens
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options
        return payload

    def _native_to_openai(self, native: dict, model: str) -> dict:
        """네이티브 /api/chat 단일 응답 → OpenAI chat.completion 형식 변환"""
        msg = native.get("message", {})
        prompt_tokens = native.get("prompt_eval_count", 0)
        completion_tokens = native.get("eval_count", 0)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                },
                "finish_reason": native.get("done_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def chat(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        if request.think is not None:
            # think 파라미터는 /v1에서 무시됨 → 네이티브 /api/chat 사용
            payload = self._build_native_payload(request, spec)
            payload["stream"] = False
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            return self._native_to_openai(response.json(), spec.upstream)

        payload = self._build_payload(request, spec)
        payload["stream"] = False
        response = await self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def list_models(self) -> list[dict]:
        """
        Ollama 서버에 실제 설치된 모델 목록을 조회한다(/api/tags).
        로컬 모델은 운영 중 pull/rm으로 자주 바뀌므로 레지스트리에 고정하지 않고
        실시간 조회한다(SaaS provider는 정적 레지스트리로 관리). 서버 미가용 시
        예외를 그대로 올려 호출 측(/v1/models)이 graceful degrade 하도록 둔다.

        각 항목은 최소 "name"(태그)과, Ollama가 제공하면 "capabilities"
        (["embedding"] · ["completion","tools",...] 등)를 담는다. 호출 측이
        capabilities로 chat/embedding 여부를 정확히 구분한다(구버전엔 없을 수 있음).
        """
        # /v1/models 응답이 로컬 서버 다운으로 오래 매달리지 않도록 짧은 타임아웃 사용.
        response = await self.client.get("/api/tags", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return [m for m in data.get("models", []) if m.get("name")]

    async def embed(self, request: EmbeddingsRequest, spec: ModelSpec) -> dict:
        """Ollama OpenAI 호환 /v1/embeddings 프록시. 모델은 사전 pull 필요(미설치면 404 → 폴백)."""
        response = await self.client.post(
            "/v1/embeddings", json=build_embeddings_payload(request, spec)
        )
        response.raise_for_status()
        return response.json()

    async def stream(
        self, request: ChatCompletionRequest, spec: ModelSpec
    ) -> AsyncGenerator[str, None]:
        if request.think is not None:
            # 네이티브 NDJSON 스트림 → OpenAI SSE 변환
            payload = self._build_native_payload(request, spec)
            payload["stream"] = True
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())
            first = True

            async with self.client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    done = chunk.get("done", False)

                    delta: dict = {"content": content}
                    if first:
                        delta["role"] = "assistant"
                        first = False

                    sse = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": spec.upstream,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": chunk.get("done_reason") if done else None,
                        }],
                    }
                    yield f"data: {json.dumps(sse)}\n\n"

            yield "data: [DONE]\n\n"
            return

        payload = self._build_payload(request, spec)
        payload["stream"] = True

        async with self.client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as response:
            # 상태 코드 오류(예: 모델 미로드 404)는 첫 청크 전에 여기서 raise됨
            response.raise_for_status()
            # Ollama가 내려주는 SSE 중 'data:' 라인만 골라 그대로 전달
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n\n"

    async def aclose(self) -> None:
        await self.client.aclose()
