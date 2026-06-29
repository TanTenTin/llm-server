"""
Gemini 네이티브 패스스루(/v1beta/models/{model}:generateContent → Gemini) 회귀 테스트.

핵심 보장:
  - OpenAI-compat 변환 없이 클라이언트 Gemini body가 네이티브 엔드포인트로 거의 그대로 전달된다
    (safetySettings·thinkingConfig·cachedContent 등 Gemini 전용 필드 보존).
  - model은 body가 아니라 URL 경로(spec.upstream)로 지정되므로 payload에서 제외된다.
  - 네이티브 client가 별도 base_url(/v1beta)·x-goog-api-key 인증으로 구성된다.

> GeminiProvider는 anthropic SDK에 의존하지 않아 최소 환경에서도 그대로 실행된다.
"""

from app.providers.gemini import (
    GEMINI_NATIVE_BASE_URL,
    GeminiProvider,
)
from app.registry import MODELS


def _provider() -> GeminiProvider:
    # httpx client 2개를 만들지만 호출 전까지 네트워크 접근은 없다.
    return GeminiProvider("g-test-key")


def test_native_payload_strips_model_and_preserves_gemini_fields():
    body = {
        "model": "gemini-2.5-flash",  # URL 경로로 지정 → payload에서 빠져야
        "contents": [{"role": "user", "parts": [{"text": "안녕"}]}],
        "systemInstruction": {"parts": [{"text": "sys"}]},
        "generationConfig": {"temperature": 0.3, "thinkingConfig": {"thinkingBudget": 1024}},
        "safetySettings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
        "cachedContent": "cachedContents/abc123",
    }
    payload = _provider()._native_payload(body)

    # model은 URL 경로(spec.upstream)에서 지정 → payload에는 없어야
    assert "model" not in payload
    # Gemini 전용 필드가 손실 없이 그대로 보존(OpenAI-compat 이중 변환이 못 싣는 것들)
    assert payload["safetySettings"] == body["safetySettings"]
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}
    assert payload["cachedContent"] == "cachedContents/abc123"
    assert payload["contents"] == body["contents"]
    assert payload["systemInstruction"] == body["systemInstruction"]


def test_native_client_uses_goog_api_key_and_native_base():
    provider = _provider()
    # 네이티브 client는 별도 base_url(/v1beta, '/openai' 없음)과 x-goog-api-key 인증을 쓴다
    assert str(provider.native_client.base_url).rstrip("/") == GEMINI_NATIVE_BASE_URL
    assert provider.native_client.headers.get("x-goog-api-key") == "g-test-key"
    # OpenAI-compat client는 여전히 Bearer 인증(별도 경로)
    assert provider.client.headers.get("authorization") == "Bearer g-test-key"
