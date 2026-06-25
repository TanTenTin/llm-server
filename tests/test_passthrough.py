"""
OpenAI 패스스루(Gemini·Ollama) payload 구성 회귀 테스트.

핵심 보장:
  - 멀티턴 도구 왕복의 tool_calls 등 '메시지 구조 필드'가 업스트림 payload까지 보존된다.
  - 모델에 선언되지 않은 미지 필드도 extra="allow" 로 보존된다(조용한 누락 방지).
  - 표준 OpenAI 요청 파라미터(top_p·stop·response_format 등)가 전달된다.
  - 게이트웨이 내부/Ollama 전용 필드(think)는 OpenAI 패스스루 payload에 새지 않는다.
"""

from app.models import ChatCompletionRequest
from app.providers.openai_payload import build_openai_payload
from app.registry import MODELS

SPEC = MODELS["gemini-2.5-flash"]


def _roundtrip_request(**extra) -> ChatCompletionRequest:
    """assistant.tool_calls → tool 결과로 이어지는 멀티턴 요청."""
    body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "read x"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read", "arguments": "{\"path\":\"x\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "hi"},
            {"role": "user", "content": "summarize"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "read", "description": "read",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    }
    body.update(extra)
    return ChatCompletionRequest(**body)


def test_tool_calls_preserved_in_payload():
    payload = build_openai_payload(_roundtrip_request(), SPEC)
    assistant = payload["messages"][1]
    assert "tool_calls" in assistant, "assistant.tool_calls 가 누락되면 업스트림이 400을 낸다"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    # content=None 은 exclude_none 으로 빠져야 함
    assert assistant.get("content") is None
    # tool 결과 메시지의 tool_call_id 보존
    assert payload["messages"][2]["tool_call_id"] == "call_1"


def test_unknown_message_field_preserved():
    # 모델에 없는 미지 메시지 필드도 보존되어야(extra="allow")
    req = ChatCompletionRequest(model="auto", messages=[
        {"role": "user", "content": "hi", "reasoning_content": "secret", "some_future_field": 123},
    ])
    payload = build_openai_payload(req, SPEC)
    msg = payload["messages"][0]
    assert msg.get("reasoning_content") == "secret"
    assert msg.get("some_future_field") == 123


def test_standard_params_forwarded():
    req = _roundtrip_request(top_p=0.9, stop=["END"], seed=7,
                             response_format={"type": "json_object"},
                             presence_penalty=0.2, tool_choice="auto")
    payload = build_openai_payload(req, SPEC)
    assert payload["top_p"] == 0.9
    assert payload["stop"] == ["END"]
    assert payload["seed"] == 7
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["presence_penalty"] == 0.2
    assert payload["tool_choice"] == "auto"


def test_think_not_leaked_to_openai_payload():
    # think 는 Ollama 네이티브 전용 → OpenAI 패스스루 payload 에 들어가면 안 됨
    req = ChatCompletionRequest(model="auto", messages=[{"role": "user", "content": "hi"}], think=False)
    payload = build_openai_payload(req, SPEC)
    assert "think" not in payload


def test_model_overridden_to_upstream():
    # 요청 model 이 'auto' 여도 payload 에는 spec.upstream 이 들어가야
    payload = build_openai_payload(_roundtrip_request(), SPEC)
    assert payload["model"] == SPEC.upstream  # "gemini-2.5-flash"


def test_max_tokens_default_from_spec():
    spec = MODELS["claude-sonnet-4-6"]  # spec.max_tokens=8192
    req = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "hi"}])
    payload = build_openai_payload(req, spec)
    assert payload["max_tokens"] == 8192
    # 요청이 명시하면 요청값 우선
    req2 = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "hi"}], max_tokens=100)
    assert build_openai_payload(req2, spec)["max_tokens"] == 100
