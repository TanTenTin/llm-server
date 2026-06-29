"""
Anthropic 네이티브 패스스루(/v1/messages → Anthropic) 회귀 테스트.

핵심 보장:
  - OpenAI 이중 변환 없이 클라이언트 Anthropic body가 SDK 호출 인자로 거의 그대로 전달된다
    (cache_control·정확한 content 블록·tool_use 보존).
  - model은 레지스트리 spec.upstream으로 보정되고, max_tokens는 요청값 우선.
  - 표준 외 top-level 필드는 extra_body로 라우팅(미지 kwargs로 인한 SDK TypeError 회피).

> anthropic SDK가 설치된 환경(CI/배포 이미지)에서만 실행되며, 미설치 환경에선 skip된다.
"""

import pytest

# anthropic 미설치 환경(개발용 최소 환경 등)에서는 이 파일 전체를 skip
pytest.importorskip("anthropic")

from app.providers.anthropic import DEFAULT_MAX_TOKENS, AnthropicProvider
from app.registry import MODELS


def _provider() -> AnthropicProvider:
    # 빈/더미 키로 생성해도 호출 전까지는 예외가 나지 않는다(_native_params는 순수 변환).
    return AnthropicProvider("sk-test")


def test_native_params_preserve_body_and_override_model():
    spec = MODELS["claude-sonnet-4-6"]  # upstream=claude-sonnet-4-6
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 100,
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "f", "input": {"x": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}]},
        ],
        "stream": True,  # 호출 측에서 별도 지정 → params에서 제외돼야
    }
    params = _provider()._native_params(body, spec)

    # model은 레지스트리가 결정한 upstream으로
    assert params["model"] == spec.upstream
    # stream은 _native_params가 손대지 않음(호출 측에서 지정)
    assert "stream" not in params
    # cache_control 등 content 블록이 손실 없이 그대로 보존(이중 변환 없음)
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # 멀티턴 tool_use / tool_result 가 평문으로 뭉개지지 않고 블록 그대로
    assert params["messages"][1]["content"][0]["type"] == "tool_use"
    assert params["messages"][2]["content"][0]["type"] == "tool_result"
    # max_tokens는 요청값 우선
    assert params["max_tokens"] == 100


def test_native_params_unknown_field_to_extra_body():
    spec = MODELS["claude-sonnet-4-6"]
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},  # SDK 버전 의존 → extra_body
        "some_beta_field": 123,
    }
    params = _provider()._native_params(body, spec)
    # 표준 외 top-level 필드는 extra_body로 모인다(known kwargs 오염 방지)
    assert params["extra_body"] == {"thinking": {"type": "enabled", "budget_tokens": 1024}, "some_beta_field": 123}
    assert "thinking" not in params
    assert "some_beta_field" not in params


def test_native_params_max_tokens_default():
    spec = MODELS["claude-sonnet-4-6"]  # spec.max_tokens=8192
    body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]}
    params = _provider()._native_params(body, spec)
    # 요청에 max_tokens가 없으면 spec.max_tokens, 그것도 없으면 DEFAULT_MAX_TOKENS
    assert params["max_tokens"] == spec.max_tokens or params["max_tokens"] == DEFAULT_MAX_TOKENS
