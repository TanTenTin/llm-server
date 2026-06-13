import httpx
from typing import AsyncGenerator

from app.models import ChatCompletionRequest
from app.providers.base import LLMProvider
from app.registry import ModelSpec

# Gemini는 OpenAI 호환 엔드포인트를 제공함.
# 인증은 Bearer 토큰(GOOGLE_AI_API_KEY)으로 처리.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
TIMEOUT = 120.0


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = httpx.AsyncClient(
            base_url=GEMINI_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )

    def _build_payload(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        payload: dict = {
            "model": spec.upstream,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "stream": request.stream or False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        max_tokens = request.max_tokens or spec.max_tokens
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if request.tools:
            payload["tools"] = [tool.model_dump() for tool in request.tools]
        return payload

    async def chat(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        payload = self._build_payload(request, spec)
        payload["stream"] = False
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def stream(
        self, request: ChatCompletionRequest, spec: ModelSpec
    ) -> AsyncGenerator[str, None]:
        payload = self._build_payload(request, spec)
        payload["stream"] = True

        async with self.client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n\n"

    async def aclose(self) -> None:
        await self.client.aclose()
