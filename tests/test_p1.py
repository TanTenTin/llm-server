"""
P1 기능 회귀 테스트 — vision 라우팅 · embeddings 라우팅/payload · 응답 캐시 ·
과금 예산 가드 · 레이트리밋 · 복수 키 인증.

핵심 보장:
  - 이미지 포함 auto 요청이 vision 미지원 로컬(qwen3)로 가지 않는다.
  - 임베딩 별칭(text-embedding-3-*)이 기본 모델로 매핑되고, Anthropic으로는
    절대 라우팅되지 않는다(임베딩 API 없음).
  - 캐시 키는 스트리밍/temperature>0 요청을 거부하고, 요청이 1비트라도 다르면
    다른 키가 된다(오염 방지).
  - 과금 예산 소진 시 유료 후보를 건너뛰어 무료 폴백으로 가고, 무료 후보가
    없으면 402(BudgetExceeded)로 실패한다.
"""

import asyncio

from app.cache import ResponseCache, cache_key_for
from app.config import settings
from app.models import ChatCompletionRequest, EmbeddingsRequest
from app.ratelimit import RateLimiter
from app.registry import (
    EMBEDDING_MODELS,
    MODELS,
    RouteDecision,
    resolve_embedding,
    route,
)
from app.providers.openai_payload import build_embeddings_payload
from app.service import BudgetExceeded, ProviderPool, RouteTrace, http_status_for, run_chat_fallback


# ── vision capability 라우팅 ─────────────────────────────────
def _image_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="auto", messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "what is in this picture"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ])


def test_auto_with_image_excludes_non_vision_models():
    decision = route(_image_request())
    assert ",vision=1" in decision.reason
    assert all(spec.supports_vision for spec in decision.chain)
    assert all(spec.provider != "ollama" for spec in decision.chain)


def test_auto_without_image_keeps_local_candidate():
    decision = route(ChatCompletionRequest(model="auto", messages=[
        {"role": "user", "content": "hi"},
    ]))
    assert ",vision=1" not in decision.reason
    assert any(spec.provider == "ollama" for spec in decision.chain)


# ── embeddings 라우팅 / payload ──────────────────────────────
def test_embedding_openai_alias_maps_to_default():
    decision = resolve_embedding("text-embedding-3-small")
    assert decision.chain[0].upstream == "gemini-embedding-001"
    # 로컬 폴백이 체인에 있어야 함
    assert any(spec.provider == "ollama" for spec in decision.chain)


def test_embedding_never_routes_to_anthropic():
    # Anthropic은 임베딩 API가 없다 — claude 이름이 와도 기본 모델로 대체
    decision = resolve_embedding("claude-sonnet-4-6")
    assert all(spec.provider != "anthropic" for spec in decision.chain)
    assert decision.chain[0].upstream == "gemini-embedding-001"


def test_embedding_ollama_passthrough():
    decision = resolve_embedding("ollama/mxbai-embed-large")
    assert decision.chain[0].provider == "ollama"
    assert decision.chain[0].upstream == "mxbai-embed-large"


def test_embeddings_payload_forwards_optional_params():
    request = EmbeddingsRequest(
        model="embed", input=["a", "b"], dimensions=256, encoding_format="float",
    )
    payload = build_embeddings_payload(request, EMBEDDING_MODELS["gemini-embedding-001"])
    assert payload["model"] == "gemini-embedding-001"
    assert payload["input"] == ["a", "b"]
    assert payload["dimensions"] == 256
    assert payload["encoding_format"] == "float"
    assert "user" not in payload  # None 파라미터는 전달하지 않음


# ── 응답 캐시 ─────────────────────────────────────────────────
def _chat(content: str, **extra) -> ChatCompletionRequest:
    body = {"model": "auto", "messages": [{"role": "user", "content": content}]}
    body.update(extra)
    return ChatCompletionRequest(**body)


def test_cache_key_rejects_stream_and_sampling():
    assert cache_key_for(_chat("hi", stream=True)) is None
    assert cache_key_for(_chat("hi", temperature=0.7)) is None
    assert cache_key_for(_chat("hi")) is not None
    assert cache_key_for(_chat("hi", temperature=0)) is not None


def test_cache_key_differs_when_request_differs():
    assert cache_key_for(_chat("hi")) == cache_key_for(_chat("hi"))
    assert cache_key_for(_chat("hi")) != cache_key_for(_chat("hello"))
    assert cache_key_for(_chat("hi")) != cache_key_for(_chat("hi", max_tokens=100))


def test_cache_put_get_and_lru_eviction():
    cache = ResponseCache(ttl_seconds=60.0, max_entries=2)
    cache.put("k1", {"id": 1})
    cache.put("k2", {"id": 2})
    assert cache.get("k1") == {"id": 1}
    cache.put("k3", {"id": 3})  # 상한 초과 → 가장 오래 안 쓰인 k2 제거(k1은 방금 조회됨)
    assert cache.get("k2") is None
    assert cache.get("k1") == {"id": 1}
    assert cache.get("k3") == {"id": 3}


def test_cache_disabled_stores_nothing():
    cache = ResponseCache(ttl_seconds=0)
    assert not cache.enabled
    cache.put("k", {"id": 1})
    assert cache.get("k") is None


# ── 과금 예산 가드 ────────────────────────────────────────────
def _spend_paid_tokens(pool: ProviderPool, tokens: int) -> None:
    """과금 사용량을 미리 적립해 예산 소진 상태를 만든다."""
    pool.usage.record_success(
        "anthropic:claude-sonnet-4-6",
        {"usage": {"input_tokens": tokens, "output_tokens": 0}},
        fell_back=False, is_free=False,
    )


def test_budget_exhausted_skips_paid_and_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "paid_daily_token_budget", 100)
    pool = ProviderPool()
    _spend_paid_tokens(pool, 200)  # 예산(100) 초과 상태

    decision = RouteDecision(chain=[MODELS["claude-sonnet-4-6"], MODELS["ollama/qwen3:14b"]])
    trace = RouteTrace(requested="claude-sonnet-4-6")

    async def invoke(spec):
        assert spec.provider != "anthropic", "예산 소진 상태에서 유료 후보가 호출되면 안 된다"
        return {"id": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    result = asyncio.run(run_chat_fallback(decision, pool, trace, invoke))
    assert result["id"] == "ok"
    assert trace.served == "ollama:qwen3:14b"
    assert any(a.endswith("#budget") for a in trace.attempts)


def test_budget_exhausted_without_free_fallback_returns_402(monkeypatch):
    monkeypatch.setattr(settings, "paid_daily_token_budget", 100)
    pool = ProviderPool()
    _spend_paid_tokens(pool, 200)

    decision = RouteDecision(chain=[MODELS["claude-sonnet-4-6"]])  # 무료 폴백 없음

    async def invoke(spec):
        raise AssertionError("호출되면 안 됨")

    try:
        asyncio.run(run_chat_fallback(decision, pool, None, invoke))
        raise AssertionError("BudgetExceeded가 나야 함")
    except BudgetExceeded as e:
        assert http_status_for(e) == 402


def test_budget_disabled_by_default(monkeypatch):
    monkeypatch.setattr(settings, "paid_daily_token_budget", 0)
    pool = ProviderPool()
    _spend_paid_tokens(pool, 10_000)

    decision = RouteDecision(chain=[MODELS["claude-sonnet-4-6"], MODELS["ollama/qwen3:14b"]])

    async def invoke(spec):
        return {"id": "paid-ok"}

    result = asyncio.run(run_chat_fallback(decision, pool, None, invoke))
    assert result["id"] == "paid-ok"  # 예산 0 = 무제한 → 유료 후보 그대로 호출


# ── 레이트리밋 / 복수 키 ──────────────────────────────────────
def test_rate_limiter_blocks_over_rpm():
    limiter = RateLimiter(rpm=2)
    assert limiter.allow("client-a")
    assert limiter.allow("client-a")
    assert not limiter.allow("client-a")   # 3번째부터 차단
    assert limiter.allow("client-b")       # 다른 identity는 독립 윈도우


def test_rate_limiter_disabled():
    limiter = RateLimiter(rpm=0)
    assert not limiter.enabled
    assert all(limiter.allow("x") for _ in range(100))


def test_multiple_gateway_keys(monkeypatch):
    from app.main import _gateway_keys, _token_matches

    monkeypatch.setattr(settings, "gateway_api_key", "key-one, key-two")
    keys = _gateway_keys()
    assert keys == ["key-one", "key-two"]
    assert _token_matches("key-one", keys)
    assert _token_matches("key-two", keys)
    assert not _token_matches("key-three", keys)

    monkeypatch.setattr(settings, "gateway_api_key", "")
    assert _gateway_keys() == []  # 빈 값 = 인증 개방
