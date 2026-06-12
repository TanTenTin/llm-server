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
        if request.think is not None:
            payload["think"] = request.think
        return payload

    async def chat(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        payload = self._build_payload(request, spec)
        payload["stream"] = False
        response = await self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def stream(
        self, request: ChatCompletionRequest, spec: ModelSpec
    ) -> AsyncGenerator[str, None]:
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
