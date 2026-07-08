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


# ── P1: 명시 라우팅 컨텍스트 가드 (_guard_context) ──────────────
def test_explicit_local_overflow_reorders_to_large_context():
    """
    ollama/qwen3:14b(32k) 명시 + 대용량 입력이면, fallback의 Gemini(1M)를
    앞으로 재정렬하고 reason에 truncate_risk를 실어야 한다(조용한 잘림 방지).
    """
    big = "가" * (40_000 * _CHARS_PER_TOKEN)  # 추정 ~40k 토큰 (로컬 usable 25.6k 초과)
    decision = route(ChatCompletionRequest(
        model="ollama/qwen3:14b",
        messages=[{"role": "user", "content": big}],
    ))
    assert decision.reason is not None and "truncate_risk=1" in decision.reason
    assert decision.chain[0].provider == "gemini"          # 1M 창 후보가 앞으로
    assert decision.chain[0].context_window == 1_000_000


def test_explicit_local_small_input_untouched():
    """작은 입력이면 결정을 손대지 않는다(primary 유지, reason=None)."""
    decision = route(ChatCompletionRequest(
        model="ollama/qwen3:14b",
        messages=[{"role": "user", "content": "안녕"}],
    ))
    assert decision.reason is None
    assert decision.chain[0].upstream == "qwen3:14b"


# ── 모든 로컬 모델에 SaaS 폴백 보장 (_ensure_saas_fallback) ──────
def test_passthrough_local_gets_saas_fallback():
    """미등록 패스스루 로컬 모델(ollama/gemma4:12b)도 SaaS(Gemini) 폴백을 받아야 한다."""
    decision = route(ChatCompletionRequest(
        model="ollama/gemma4:12b",
        messages=[{"role": "user", "content": "안녕"}],
    ))
    providers = [spec.provider for spec in decision.chain]
    assert decision.chain[0].provider == "ollama"      # 로컬 primary 유지
    assert "gemini" in providers                        # SaaS 폴백 자동 부착


def test_registry_local_no_duplicate_saas():
    """이미 gemini 폴백이 있는 로컬 모델엔 중복으로 붙이지 않는다."""
    decision = route(ChatCompletionRequest(
        model="ollama/qwen3:14b",
        messages=[{"role": "user", "content": "안녕"}],
    ))
    gemini_count = sum(1 for spec in decision.chain if spec.provider == "gemini")
    assert gemini_count == 1


def test_saas_primary_chain_untouched():
    """SaaS가 primary면 로컬-only가 아니므로 폴백을 덧붙이지 않는다(기존 체인 유지)."""
    decision = route(ChatCompletionRequest(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "안녕"}],
    ))
    assert decision.chain[0].provider == "gemini"
    # 기존 fallback(ollama/qwen3:14b)만 유지
    assert [s.provider for s in decision.chain] == ["gemini", "ollama"]
