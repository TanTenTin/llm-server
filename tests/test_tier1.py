"""
Tier 1 고도화(E-01·E-02·E-03) 회귀 테스트.

- E-01: 스트리밍 SSE에서 usage를 sniff해 집계 + 과금 예산 근거 반영
- E-02: registry의 ollama context_window가 런타임 num_ctx와 한 소스로 묶임
- E-03: Ollama /api/tags capability가 패스스루 라우팅의 tools/vision에 환류
"""

from app.config import settings
from app.registry import (
    MODELS,
    _OLLAMA_CONTEXT_WINDOW,
    resolve,
    update_ollama_capabilities,
)
from app.usage import UsageTracker, sniff_stream_usage


# ─────────────────────────────────────────────────────────────
# E-01 — 스트리밍 토큰 집계
# ─────────────────────────────────────────────────────────────
def test_sniff_openai_stream_usage() -> None:
    """OpenAI-compat 스트림의 마지막 usage 청크를 누적한다(Gemini·Ollama 공통 포맷)."""
    acc: dict = {}
    sniff_stream_usage('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', acc)
    sniff_stream_usage(
        'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":8}}\n\n', acc
    )
    assert acc == {"prompt": 12, "completion": 8}


def test_sniff_gemini_native_stream_usage() -> None:
    """Gemini 네이티브 스트림은 usageMetadata(누적)에서 마지막 값이 최종."""
    acc: dict = {}
    sniff_stream_usage('data: {"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3}}\n\n', acc)
    sniff_stream_usage('data: {"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":20}}\n\n', acc)
    assert acc == {"prompt": 5, "completion": 20}


def test_sniff_anthropic_native_stream_usage() -> None:
    """Anthropic 네이티브는 message_start(input) + message_delta(output)에 나눠 실린다."""
    acc: dict = {}
    sniff_stream_usage('data: {"type":"message_start","message":{"usage":{"input_tokens":40,"output_tokens":1}}}\n\n', acc)
    sniff_stream_usage('data: {"type":"message_delta","usage":{"output_tokens":15}}\n\n', acc)
    assert acc == {"prompt": 40, "completion": 15}


def test_sniff_skips_plain_delta() -> None:
    """usage 마커가 없는 순수 델타 청크는 파싱 없이 건너뛴다(누적 변화 없음)."""
    acc: dict = {}
    sniff_stream_usage('data: {"choices":[{"delta":{"content":"hello"}}]}\n\n', acc)
    sniff_stream_usage("data: [DONE]\n\n", acc)
    assert acc == {}


def test_record_stream_tokens_paid_feeds_budget() -> None:
    """과금 스트림 토큰이 paid_tokens_today에 적립돼 예산 가드 근거가 된다(E-01 핵심)."""
    tracker = UsageTracker()
    tracker.record_success("anthropic:claude-sonnet-4-6", None, False, is_free=False)
    assert tracker.paid_tokens_today() == 0  # body=None이라 아직 토큰 없음
    tracker.record_stream_tokens("anthropic:claude-sonnet-4-6", 100, 50, is_free=False)
    assert tracker.paid_tokens_today() == 150


def test_record_stream_tokens_free_not_counted_as_paid() -> None:
    """무료 스트림 토큰은 집계되되 과금 예산에는 잡히지 않는다."""
    tracker = UsageTracker()
    tracker.record_stream_tokens("ollama:qwen3:14b", 100, 50, is_free=True)
    assert tracker.paid_tokens_today() == 0
    snap = tracker.snapshot()
    assert snap["totals"]["ollama:qwen3:14b"]["total_tokens"] == 150


# ─────────────────────────────────────────────────────────────
# E-02 — context_window ↔ num_ctx 단일 소스
# ─────────────────────────────────────────────────────────────
def test_registered_ollama_context_matches_num_ctx() -> None:
    """등록 ollama 모델의 context_window가 런타임 num_ctx에서 파생된다."""
    expected = settings.ollama_num_ctx if settings.ollama_num_ctx > 0 else 8192
    assert _OLLAMA_CONTEXT_WINDOW == expected
    assert MODELS["ollama/qwen3:14b"].context_window == _OLLAMA_CONTEXT_WINDOW


def test_passthrough_ollama_context_matches_num_ctx() -> None:
    """패스스루 ollama 모델도 같은 소스의 context_window를 받는다(라우터-런타임 정합)."""
    chain = resolve("ollama/some-unregistered:latest").chain
    assert chain[0].context_window == _OLLAMA_CONTEXT_WINDOW


# ─────────────────────────────────────────────────────────────
# E-03 — capability 환류
# ─────────────────────────────────────────────────────────────
def test_capability_reflects_vision_and_tools() -> None:
    """/api/tags capability가 패스스루 spec의 supports_vision/supports_tools에 반영된다."""
    update_ollama_capabilities([
        {"name": "llava:13b", "capabilities": ["completion", "vision"]},
        {"name": "mistral:7b", "capabilities": ["completion", "tools"]},
    ])
    try:
        llava = resolve("ollama/llava:13b").chain[0]
        assert llava.supports_vision is True
        assert llava.supports_tools is False   # vision만, tools 없음

        mistral = resolve("ollama/mistral:7b").chain[0]
        assert mistral.supports_tools is True
        assert mistral.supports_vision is False
    finally:
        update_ollama_capabilities([])  # 캐시 정리(다른 테스트 오염 방지)


def test_capability_empty_cache_falls_back_to_defaults() -> None:
    """캐시가 비면(미조회/구버전 Ollama) 보수적 기본값으로 폴백해 회귀가 없다."""
    update_ollama_capabilities([])  # 빈 캐시
    spec = resolve("ollama/unknown-model:latest").chain[0]
    assert spec.supports_tools is True
    assert spec.supports_vision is False
