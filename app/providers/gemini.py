import httpx
from typing import AsyncGenerator

from app.models import ChatCompletionRequest
from app.providers.base import LLMProvider
from app.providers.openai_payload import build_openai_payload
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
        # 공용 OpenAI 패스스루 빌더 사용 — 메시지 구조(tool_calls 등) 보존 + 표준 파라미터 전달
        return build_openai_payload(request, spec)

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
            # 스트리밍 응답은 본문을 자동으로 읽지 않으므로, 에러 시 사유가 비어버린다.
            # 4xx/5xx면 본문을 먼저 당겨와 raise_for_status가 .text를 담은 채 던지게 한다.
            if response.status_code >= 400:
                await response.aread()
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n\n"

    async def aclose(self) -> None:
        await self.client.aclose()
