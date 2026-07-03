import httpx
from typing import AsyncGenerator

from app.models import ChatCompletionRequest, EmbeddingsRequest
from app.providers.base import LLMProvider
from app.providers.openai_payload import build_embeddings_payload, build_openai_payload
from app.registry import ModelSpec

# Gemini는 OpenAI 호환 엔드포인트를 제공함.
# 인증은 Bearer 토큰(GOOGLE_AI_API_KEY)으로 처리.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
# 네이티브 generateContent 엔드포인트(패스스루용). OpenAI-compat와 달리 '/openai' 가 없고
# 인증은 x-goog-api-key 헤더(또는 ?key=)로 한다(Bearer는 OAuth 토큰용이라 API 키엔 안 맞음).
GEMINI_NATIVE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = 120.0


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = httpx.AsyncClient(
            base_url=GEMINI_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
        # 네이티브 패스스루 전용 client (별도 base_url·인증 방식). 영속 재사용.
        self.native_client = httpx.AsyncClient(
            base_url=GEMINI_NATIVE_BASE_URL,
            headers={"x-goog-api-key": api_key},
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

    async def embed(self, request: EmbeddingsRequest, spec: ModelSpec) -> dict:
        """Gemini OpenAI 호환 /embeddings 프록시 (chat과 같은 client·인증 재사용)."""
        response = await self.client.post(
            "/embeddings", json=build_embeddings_payload(request, spec)
        )
        response.raise_for_status()
        return response.json()

    # ── 네이티브 패스스루 (/v1beta/models/{model}:generateContent → Gemini) ──────
    # OpenAI-compat 이중 변환을 건너뛰고 클라이언트의 Gemini 요청 body를 그대로 네이티브
    # 엔드포인트로 보낸다. safetySettings·thinkingConfig·cachedContent 등 Gemini 전용 필드와
    # 응답의 네이티브 구조(candidates/usageMetadata/safetyRatings 등)가 손실 없이 보존된다.
    def _native_payload(self, body: dict) -> dict:
        # model은 URL 경로(spec.upstream)에서 지정하므로 body에서 제외. 그 외는 그대로 통과.
        return {k: v for k, v in body.items() if k != "model"}

    async def generate_native(self, body: dict, spec: ModelSpec) -> dict:
        """Gemini 네이티브 패스스루(비스트리밍). 응답도 GenerateContentResponse 그대로 반환."""
        payload = self._native_payload(body)
        response = await self.native_client.post(
            f"/models/{spec.upstream}:generateContent", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def stream_native(
        self, body: dict, spec: ModelSpec
    ) -> AsyncGenerator[str, None]:
        """
        Gemini 네이티브 패스스루(스트리밍). `:streamGenerateContent?alt=sse` 의 네이티브 SSE
        라인을 그대로 중계한다(변환 없음 → 네이티브 응답 구조 보존).
        """
        payload = self._native_payload(body)
        async with self.native_client.stream(
            "POST",
            f"/models/{spec.upstream}:streamGenerateContent",
            params={"alt": "sse"},
            json=payload,
        ) as response:
            # 스트리밍은 본문을 자동으로 읽지 않으므로, 에러 시 사유가 비지 않도록 먼저 당겨온다.
            if response.status_code >= 400:
                await response.aread()
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n\n"

    async def aclose(self) -> None:
        await self.client.aclose()
        await self.native_client.aclose()
