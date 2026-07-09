"""
Tier 3 저리스크 고도화(E-13·E-14·E-15·E-16) 회귀 테스트.

- E-13: 회로차단기 연속 실패 임계치(단발 오류 과잉 개방 방지, 명시적 힌트는 즉시 개방)
- E-14: X-Forwarded-For 신뢰 시 원 클라이언트 IP로 레이트리밋 식별
- E-15: 캐시 single-flight(동일 요청 동시 다발 시 결과 공유)
- E-16: connect/read 타임아웃 분리
"""

import asyncio

import httpx
import pytest

from app.cache import ResponseCache
from app.config import settings
from app.main import _client_ip
from app.providers.gemini import TIMEOUT as GEMINI_TIMEOUT
from app.providers.ollama import TIMEOUT as OLLAMA_TIMEOUT
from app.service import CircuitBreaker


# ── E-13 회로차단기 임계치 ────────────────────────────────────
def test_threshold_requires_consecutive_failures() -> None:
    b = CircuitBreaker(cooldown_seconds=30.0, failure_threshold=2)
    assert b.record_failure("gemini") == 0.0   # 1회: 아직 미개방
    assert not b.is_open("gemini")
    assert b.record_failure("gemini") == 30.0  # 2회: 임계치 도달 → 개방
    assert b.is_open("gemini")


def test_explicit_hint_opens_immediately_despite_threshold() -> None:
    """업스트림 Retry-After(명시적 백오프)는 임계치와 무관하게 즉시 연다."""
    b = CircuitBreaker(cooldown_seconds=30.0, failure_threshold=3)
    assert b.record_failure("gemini", cooldown_hint=120.0) == 120.0
    assert b.is_open("gemini")


def test_success_resets_failure_count() -> None:
    b = CircuitBreaker(cooldown_seconds=30.0, failure_threshold=2)
    b.record_failure("gemini")            # 1회
    b.record_success("gemini")            # 성공 → 카운터 리셋
    assert b.record_failure("gemini") == 0.0   # 다시 1회일 뿐 → 미개방
    assert not b.is_open("gemini")


# ── E-14 X-Forwarded-For ──────────────────────────────────────
def test_client_ip_uses_socket_by_default() -> None:
    headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
    assert _client_ip(headers, "10.0.0.1") == "10.0.0.1"  # 신뢰 꺼짐 → 소켓 IP


def test_client_ip_uses_xff_first_when_trusted(monkeypatch) -> None:
    monkeypatch.setattr(settings, "trust_proxy_forwarded_for", True)
    headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
    assert _client_ip(headers, "10.0.0.1") == "203.0.113.9"  # 원 클라이언트 IP


# ── E-15 캐시 single-flight ───────────────────────────────────
def test_single_flight_shares_result() -> None:
    async def scenario() -> None:
        cache = ResponseCache(300.0)
        is_leader, fut = cache.begin("k")
        assert is_leader
        is_leader2, fut2 = cache.begin("k")
        assert not is_leader2 and fut2 is fut   # follower가 leader의 future 공유
        cache.settle("k", result={"ok": 1})
        assert await fut2 == {"ok": 1}
        # settle 후 키가 비워져 다음 요청은 다시 leader
        assert cache.begin("k")[0] is True

    asyncio.run(scenario())


def test_single_flight_propagates_exception() -> None:
    async def scenario() -> None:
        cache = ResponseCache(300.0)
        _, _ = cache.begin("k")
        _, follower = cache.begin("k")
        cache.settle("k", exc=ValueError("boom"))
        with pytest.raises(ValueError):
            await follower

    asyncio.run(scenario())


# ── E-16 connect/read 타임아웃 분리 ───────────────────────────
def test_timeouts_separate_connect_and_read() -> None:
    for timeout in (GEMINI_TIMEOUT, OLLAMA_TIMEOUT):
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect < timeout.read   # 죽은 호스트는 빨리 포기, 생성은 길게
