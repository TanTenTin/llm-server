"""
P0 관측 계층 회귀 테스트 — 동적 회로 쿨다운 · 사용량 집계.

핵심 보장:
  - retry_after_seconds가 표준 Retry-After 헤더(초/HTTP-date)와 Gemini RetryInfo
    본문(retryDelay)을 파싱하고, 힌트가 없으면 None을 돌려준다.
  - CircuitBreaker가 힌트를 쿨다운으로 반영하되 상한(1시간)으로 클램프하고,
    쿨다운 0(비활성화) 설정을 존중한다.
  - UsageTracker가 OpenAI/Anthropic/Gemini 3종 usage 필드를 정규화해 집계하고,
    에러 종류·폴백 응답 횟수를 스냅샷(totals 포함)으로 노출한다.
"""

import httpx

from app.service import CircuitBreaker, retry_after_seconds
from app.usage import UsageTracker


def _http_error(status: int = 429, headers: dict | None = None, body: dict | None = None) -> httpx.HTTPStatusError:
    """지정한 헤더/본문을 가진 httpx.HTTPStatusError 생성 헬퍼."""
    request = httpx.Request("POST", "http://upstream/v1/chat")
    response = httpx.Response(status, headers=headers or {}, json=body or {}, request=request)
    return httpx.HTTPStatusError("upstream error", request=request, response=response)


# ── retry_after_seconds ──────────────────────────────────────
def test_retry_after_header_seconds():
    assert retry_after_seconds(_http_error(headers={"retry-after": "42"})) == 42.0


def test_retry_after_gemini_retry_info_body():
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "58s"},
    ]}}
    assert retry_after_seconds(_http_error(body=body)) == 58.0


def test_retry_after_no_hint_returns_none():
    assert retry_after_seconds(_http_error()) is None
    assert retry_after_seconds(RuntimeError("no response attr")) is None


def test_retry_after_header_takes_precedence_over_body():
    body = {"error": {"details": [
        {"@type": "google.rpc.RetryInfo", "retryDelay": "99s"},
    ]}}
    assert retry_after_seconds(_http_error(headers={"retry-after": "10"}, body=body)) == 10.0


# ── CircuitBreaker 동적 쿨다운 ────────────────────────────────
def test_breaker_uses_hint_cooldown():
    breaker = CircuitBreaker(cooldown_seconds=30.0)
    applied = breaker.record_failure("gemini", cooldown_hint=300.0)
    assert applied == 300.0
    # 기본 쿨다운(30s)보다 길게 열려 있어야 함
    assert breaker.status()["gemini"] > 30.0


def test_breaker_clamps_hint_to_max():
    breaker = CircuitBreaker(cooldown_seconds=30.0)
    applied = breaker.record_failure("gemini", cooldown_hint=86_400.0)  # RPD 소진: 24시간
    assert applied == 3600.0  # 상한 1시간 — half-open 탐침이 실제 회복을 잡게 둔다


def test_breaker_hint_shorter_than_default_is_trusted():
    breaker = CircuitBreaker(cooldown_seconds=30.0)
    applied = breaker.record_failure("gemini", cooldown_hint=5.0)
    assert applied == 5.0  # 업스트림이 5초 뒤 재시도 가능하다면 30초를 기다릴 이유가 없다


def test_breaker_disabled_ignores_hint():
    breaker = CircuitBreaker(cooldown_seconds=0)
    assert breaker.record_failure("gemini", cooldown_hint=300.0) == 0.0
    assert not breaker.is_open("gemini")
    assert breaker.status() == {}


def test_breaker_success_closes_and_status_clears():
    breaker = CircuitBreaker(cooldown_seconds=30.0)
    breaker.record_failure("gemini")
    assert breaker.is_open("gemini")
    breaker.record_success("gemini")
    assert not breaker.is_open("gemini")
    assert "gemini" not in breaker.status()


# ── UsageTracker ─────────────────────────────────────────────
def test_usage_normalizes_three_provider_formats():
    tracker = UsageTracker()
    tracker.record_success("gemini:gemini-2.5-flash", {
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }, fell_back=False)
    tracker.record_success("anthropic:claude-sonnet-4-6", {
        "usage": {"input_tokens": 20, "output_tokens": 7},
    }, fell_back=False)
    tracker.record_success("gemini:gemini-2.5-flash", {
        "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 9, "totalTokenCount": 39},
    }, fell_back=False)

    totals = tracker.snapshot()["totals"]
    assert totals["gemini:gemini-2.5-flash"]["requests"] == 2
    assert totals["gemini:gemini-2.5-flash"]["prompt_tokens"] == 40
    assert totals["gemini:gemini-2.5-flash"]["completion_tokens"] == 14
    assert totals["anthropic:claude-sonnet-4-6"]["total_tokens"] == 27


def test_usage_streaming_counts_requests_only():
    tracker = UsageTracker()
    tracker.record_success("ollama:qwen3:14b", None, fell_back=True)
    totals = tracker.snapshot()["totals"]["ollama:qwen3:14b"]
    assert totals["requests"] == 1
    assert totals["total_tokens"] == 0
    assert totals["served_as_fallback"] == 1


def test_usage_records_error_kinds():
    tracker = UsageTracker()
    tracker.record_error("gemini:gemini-2.5-flash", "429")
    tracker.record_error("gemini:gemini-2.5-flash", "429")
    tracker.record_error("anthropic:claude-sonnet-4-6", "unavailable")
    totals = tracker.snapshot()["totals"]
    assert totals["gemini:gemini-2.5-flash"]["errors"] == {"429": 2}
    assert totals["anthropic:claude-sonnet-4-6"]["errors"] == {"unavailable": 1}


def test_usage_snapshot_has_day_buckets():
    tracker = UsageTracker()
    tracker.record_success("gemini:gemini-2.5-flash", {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}, fell_back=False)
    snap = tracker.snapshot()
    assert snap["retention_days"] == 7
    assert len(snap["days"]) == 1  # 오늘 하나
    (day_usage,) = snap["days"].values()
    assert day_usage["gemini:gemini-2.5-flash"]["requests"] == 1
