"""
로컬 전용 라우팅(x-llm-local-only) 회귀 테스트.

배경: auto 라우팅(_ensure_saas_fallback)은 '로컬 후보만 남은 체인 끝에 Gemini를 붙인다'이다.
로컬 모델만 쓰기로 한 호출자(예: 코드가 외부로 나가면 안 되는 에이전트)에게는 이게
조용한 정책 위반이 된다 — auto에 맡겼는데 로컬이 죽으면 Gemini가 답한다.

핵심 보장:
  - local_only=True면 라우팅 체인에 SaaS(비-ollama) provider가 단 하나도 없다.
  - 명시 라우팅·auto 라우팅·long 티어(후보가 전부 SaaS인 경우) 모두에서 성립한다.
  - SaaS 모델을 콕 집어 요청해도 로컬(DEFAULT_MODEL)로 강등되며, 그 사실이
    reason(local_only=1)에 드러나 조용히 넘어가지 않는다.
  - 헤더 파서 is_local_only는 "1"/"true"/"yes"/"on"(대소문자·공백 무시)만 참으로 본다.

주의: 명시 라우팅은 애초에 폴백이 없으므로(resolve — 조용한 provider 바꿔치기 금지)
local_only의 실질 효과는 'auto 체인에서 SaaS 제거'와 'SaaS 지정 시 로컬 강등'이다.
"""

import pytest

from app.main import is_local_only
from app.models import ChatCompletionRequest
from app.registry import (
    DEFAULT_MODEL,
    _LONG_INPUT_THRESHOLD,
    _ASCII_CHARS_PER_TOKEN,
    route,
)


def _req(model: str, content: str = "안녕", **extra) -> ChatCompletionRequest:
    """단일 user 메시지 요청 헬퍼."""
    body = {"model": model, "messages": [{"role": "user", "content": content}]}
    body.update(extra)
    return ChatCompletionRequest(**body)


def _providers(decision) -> list[str]:
    return [spec.provider for spec in decision.chain]


# ── 명시 라우팅 ────────────────────────────────────────────────

def test_명시_로컬모델은_기본도_단일후보다() -> None:
    """
    명시 라우팅은 local_only 여부와 무관하게 폴백하지 않는다 — 이름을 콕 집었으면 그 모델뿐.
    (SaaS 폴백은 model="auto"에서만 붙는다. 아래 test_auto_* 참고.)
    """
    decision = route(_req("ollama/qwen3:14b"))
    assert _providers(decision) == ["ollama"]


def test_명시_로컬모델_local_only면_saas가_전부_빠진다() -> None:
    decision = route(_req("ollama/qwen3:14b"), local_only=True)
    assert set(_providers(decision)) == {"ollama"}


def test_saas모델을_콕_집어도_local_only면_로컬로_강등된다() -> None:
    """조용한 강등이 아니라 reason에 드러나야 한다."""
    decision = route(_req("gemini-2.5-flash"), local_only=True)
    assert set(_providers(decision)) == {"ollama"}
    assert decision.chain[0].upstream == "qwen3:14b"
    assert decision.reason is not None and "local_only=1" in decision.reason


def test_패스스루_미등록_ollama모델도_local_only가_적용된다() -> None:
    """레지스트리에 없는 ollama/* 는 패스스루 spec으로 만들어지며 여기도 폴백이 안 붙는다."""
    decision = route(_req("ollama/gemma4:12b"), local_only=True)
    assert set(_providers(decision)) == {"ollama"}


# ── auto 라우팅 ────────────────────────────────────────────────

def test_auto_simple티어_local_only는_로컬만_남긴다() -> None:
    decision = route(_req("auto", "안녕"), local_only=True)
    assert set(_providers(decision)) == {"ollama"}
    assert decision.reason is not None and "local_only=1" in decision.reason


def test_auto_tools요청_local_only는_로컬만_남긴다() -> None:
    """tools가 붙으면 complex 티어 — 원래는 Gemini가 선호되지만 로컬로 고정된다."""
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    decision = route(_req("auto", "코드 고쳐줘", tools=tools), local_only=True)
    assert set(_providers(decision)) == {"ollama"}


def test_auto_long티어_후보가_전부_saas여도_로컬로_되돌린다() -> None:
    """
    long 티어 후보는 전부 대용량 SaaS다. local_only면 후보가 전멸하는데, 이때
    조용히 Gemini로 새지 않고 DEFAULT_MODEL(로컬)로 되돌아와야 한다.
    (로컬 창을 넘는 입력이면 이후 fallback 루프가 413으로 정직하게 거절한다.)
    """
    huge = "a" * int(_LONG_INPUT_THRESHOLD * _ASCII_CHARS_PER_TOKEN * 1.2)
    decision = route(_req("auto", huge), local_only=True)
    assert set(_providers(decision)) == {"ollama"}
    assert decision.chain[0].upstream == "qwen3:14b"


def test_auto_long티어_기본은_여전히_saas로_간다() -> None:
    """회귀 방지 — local_only가 아니면 long 티어는 대용량 컨텍스트 모델을 쓴다."""
    huge = "a" * int(_LONG_INPUT_THRESHOLD * _ASCII_CHARS_PER_TOKEN * 1.2)
    decision = route(_req("auto", huge))
    assert decision.chain[0].provider != "ollama"


# ── 헤더 파서 ──────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", " on ", "True"])
def test_is_local_only_참으로_보는_값(value: str) -> None:
    assert is_local_only(value) is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "maybe"])
def test_is_local_only_거짓으로_보는_값(value: str | None) -> None:
    assert is_local_only(value) is False
