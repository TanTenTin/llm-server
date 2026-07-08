"""
Ollama 네이티브 /api/chat 경로 회귀 테스트 (P0/P2).

핵심 보장:
  - OpenAI 포맷 메시지 → Ollama 네이티브 변환(tool_calls args 문자열→객체, image_url→images).
  - 네이티브 payload에 num_ctx(컨텍스트 잘림 방지)·num_predict·tools가 실린다.
  - thinking 계열 모델(qwen3)은 think=False로 억제, 비-thinking 모델(gemma)엔 think 미포함.
  - 네이티브 응답의 tool_calls(객체 arguments) → OpenAI(문자열 arguments) 역변환.
"""

from app.models import ChatCompletionRequest
from app.providers.ollama import (
    OllamaProvider,
    _messages_to_native,
    _native_tool_calls_to_openai,
    _is_thinking_model,
)
from app.registry import MODELS, _passthrough_spec

PROVIDER = OllamaProvider("http://localhost:11434")
QWEN = MODELS["ollama/qwen3:14b"]              # thinking 계열
GEMMA = _passthrough_spec("ollama/gemma4:12b")  # 비-thinking(패스스루)


def _agent_request(**extra) -> ChatCompletionRequest:
    """멀티턴 도구 왕복 + 이미지 파트를 포함한 에이전트형 요청."""
    body = {
        "model": "ollama/qwen3:14b",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "이 이미지 봐"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
            ]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read", "arguments": "{\"path\":\"x\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "name": "read", "content": "hi"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "read", "description": "read",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}],
        "max_tokens": 256,
    }
    body.update(extra)
    return ChatCompletionRequest(**body)


def test_messages_tool_calls_string_to_object():
    native = _messages_to_native(_agent_request().messages)
    assistant = native[1]
    # arguments는 JSON 문자열이 아니라 객체여야(Ollama 네이티브 규격)
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"path": "x"}


def test_messages_image_url_to_images():
    native = _messages_to_native(_agent_request().messages)
    user = native[0]
    assert user["content"] == "이 이미지 봐"
    assert user["images"] == ["QUJD"]  # data URL 접두사 제거된 base64


def test_messages_tool_result_name():
    native = _messages_to_native(_agent_request().messages)
    tool_msg = native[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_name"] == "read"


def test_native_payload_injects_num_ctx_and_tools():
    payload = PROVIDER._build_native_payload(_agent_request(), QWEN, think=False)
    assert payload["options"]["num_ctx"] > 0          # 컨텍스트 잘림 방지(P0 핵심)
    assert payload["options"]["num_predict"] == 256    # max_tokens 매핑
    assert payload["tools"][0]["function"]["name"] == "read"
    assert payload["think"] is False


def test_resolve_think_disables_qwen_by_default():
    # 요청이 think 미지정이고 qwen3(thinking 계열)면 기본 억제(False)
    assert PROVIDER._resolve_think(_agent_request(), QWEN) is False


def test_resolve_think_skips_non_thinking_model():
    # gemma 등 비-thinking 모델엔 think를 보내지 않는다(None) → 400 회피
    assert PROVIDER._resolve_think(_agent_request(), GEMMA) is None


def test_resolve_think_explicit_wins():
    req = _agent_request(think=True)
    assert PROVIDER._resolve_think(req, QWEN) is True


def test_thinking_model_prefixes():
    assert _is_thinking_model("qwen3:14b")
    assert _is_thinking_model("deepseek-r1:7b")
    assert not _is_thinking_model("gemma4:12b")
    assert not _is_thinking_model("llama3:8b")


def test_native_tool_calls_object_to_string():
    native = [{"function": {"name": "read", "arguments": {"path": "x"}}}]
    converted = _native_tool_calls_to_openai(native)
    # OpenAI 규격: arguments는 JSON 문자열
    assert converted[0]["function"]["arguments"] == '{"path": "x"}'
    assert converted[0]["type"] == "function"
    assert converted[0]["id"].startswith("call_")


def test_native_to_openai_sets_tool_calls_finish_reason():
    native_resp = {
        "message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read", "arguments": {"path": "x"}}}
        ]},
        "prompt_eval_count": 10, "eval_count": 5, "done_reason": "stop",
    }
    openai = PROVIDER._native_to_openai(native_resp, "qwen3:14b")
    choice = openai["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read"
    assert openai["usage"]["total_tokens"] == 15
