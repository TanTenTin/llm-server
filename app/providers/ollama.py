import json
import time
import uuid
import httpx
from typing import AsyncGenerator

from app.models import ChatCompletionRequest
from app.providers.base import LLMProvider
from app.registry import ModelSpec

TIMEOUT = 120.0


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str):
        # Ollama는 /v1/chat/completions 엔드포인트로 OpenAI 호환 API를 제공함.
        # client는 앱 생애주기 동안 재사용 → 커넥션 keep-alive.
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=TIMEOUT)

    def _build_payload(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        payload: dict = {
            "model": spec.upstream,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "stream": request.stream or False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        # 요청값 우선, 없으면 레지스트리 기본값
        max_tokens = request.max_tokens or spec.max_tokens
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if request.tools:
            payload["tools"] = [tool.model_dump() for tool in request.tools]
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        return payload

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
