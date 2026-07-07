"""
auto 라우팅(Phase 2~4) 회귀 테스트.

핵심 보장:
  - 티어 분류: 짧은 단발 질의=simple / 도구·긴 입력=complex / 대용량 입력=long.
  - long 티어는 난이도와 무관하게 대용량 컨텍스트 모델(1M Gemini)로 직행하고
    로컬(32k)은 체인에서 빠진다.
  - 컨텍스트 적합 판정에 안전 마진(_CONTEXT_SAFETY_RATIO)과 출력 예산(max_tokens)이
    반영된다 — 창 크기 1:1 비교로 회귀하면 로컬이 못 담는 요청이 로컬로 간다.
  - 모든 후보가 컨텍스트를 감당 못 하면 창이 가장 큰 후보로 best-effort 폴백하고
    reason에 overflow=1 이 표시된다.
  - 토큰 추정에 멀티턴 tool_calls(함수 인자)가 포함된다.
"""

from app.models import ChatCompletionRequest, Message
from app.registry import (
    _CHARS_PER_TOKEN,
    _CONTEXT_SAFETY_RATIO,
    _LONG_INPUT_THRESHOLD,
    _estimate_tokens,
    route,
)


def _req(content: str, **extra) -> ChatCompletionRequest:
    """model=auto 단일 user 메시지 요청 헬퍼."""
    body = {"model": "auto", "messages": [{"role": "user", "content": content}]}
    body.update(extra)
    return ChatCompletionRequest(**body)


def _text_of_tokens(tokens: int) -> str:
    """추정 토큰이 정확히 `tokens`가 되는 더미 텍스트."""
    return "a" * (tokens * _CHARS_PER_TOKEN)


def test_simple_short_query_prefers_local():
    decision = route(_req("hi"))
    assert decision.reason.startswith("auto:tier=simple")
    # 로컬 우선 정책: 로컬(qwen3)이 primary, Gemini(flash-lite)가 폴백으로 남는다
    assert decision.chain[0].provider == "ollama"
    assert any(spec.provider == "gemini" for spec in decision.chain)


def test_tools_request_classified_complex():
    decision = route(_req("read x", tools=[{"type": "function", "function": {"name": "read"}}]))
    assert decision.reason.startswith("auto:tier=complex")
    # complex 도 로컬 우선 — qwen3는 tools 지원이라 primary로 남고 Gemini가 폴백
    assert decision.chain[0].provider == "ollama"
    assert any(spec.provider == "gemini" for spec in decision.chain)


def test_long_input_routes_to_big_context_only():
    # 로컬 usable 창(32k×0.8)을 넘는 입력 → long 티어, 로컬은 체인에서 제외
    decision = route(_req(_text_of_tokens(_LONG_INPUT_THRESHOLD + 1000)))
    assert decision.reason.startswith("auto:tier=long")
    assert decision.chain[0].upstream == "gemini-2.5-flash"
    assert all(spec.provider != "ollama" for spec in decision.chain)
    # long 티어 안에서도 폴백(flash-lite)이 남아 단일 실패점이 아니어야 함
    assert len(decision.chain) >= 2


def test_output_budget_excludes_local():
    # 입력 20k는 로컬 usable(25.6k)에 들어가지만, max_tokens=8k를 더하면 초과 → 로컬 제외
    decision = route(_req(_text_of_tokens(20_000), max_tokens=8_000))
    assert all(spec.provider != "ollama" for spec in decision.chain)
    # max_tokens 없이는 로컬이 남는다 (안전 마진만 적용)
    decision_no_budget = route(_req(_text_of_tokens(20_000)))
    assert any(spec.provider == "ollama" for spec in decision_no_budget.chain)


def test_safety_margin_applied_to_context_fit():
    # 창 크기(32k)보다 작지만 usable(25.6k)을 넘는 입력 — 1:1 비교로 회귀하면 로컬이 남는다
    tokens = int(32_000 * _CONTEXT_SAFETY_RATIO) + 500
    decision = route(_req(_text_of_tokens(tokens)))
    assert all(spec.provider != "ollama" for spec in decision.chain)


def test_overflow_falls_back_to_largest_window():
    # 1M usable(800k)마저 넘는 초대형 입력 → 창이 가장 큰 후보로 best-effort
    decision = route(_req(_text_of_tokens(900_000)))
    assert "overflow=1" in decision.reason
    assert len(decision.chain) == 1
    assert decision.chain[0].upstream == "gemini-2.5-flash"


def test_estimate_includes_tool_calls():
    big_args = '{"content": "' + "x" * 30_000 + '"}'
    with_calls = ChatCompletionRequest(model="auto", messages=[
        {"role": "user", "content": "write file"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "write", "arguments": big_args}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ])
    without_calls = ChatCompletionRequest(model="auto", messages=[
        {"role": "user", "content": "write file"},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ])
    assert _estimate_tokens(with_calls) - _estimate_tokens(without_calls) >= 10_000


def test_reason_exposes_estimated_tokens():
    decision = route(_req("hi"))
    assert ",est=" in decision.reason
