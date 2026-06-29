"""
네이티브 포맷 어댑터(Anthropic Messages / Gemini generateContent) 회귀 테스트.

핵심 보장:
  - 네이티브 요청이 내부표준(ChatCompletionRequest)으로 손실 없이 들어온다(system·tools·tool 왕복).
  - 내부표준 응답이 네이티브 응답 포맷으로 정확히 되돌아간다(stop_reason/finishReason, tool 블록).
  - 스트리밍 SSE가 각 네이티브 포맷의 이벤트 시퀀스로 변환된다.
  - model 필드는 그대로 보존된다(순수 포맷 어댑터 — 라우팅은 별도 결정).
"""

import asyncio
import json

from app.adapters import (
    anthropic_to_chat_request,
    gemini_to_chat_request,
    openai_to_anthropic_response,
    openai_to_gemini_response,
    stream_openai_to_anthropic,
    stream_openai_to_gemini,
)

# tool_calls 가 포함된 OpenAI 응답(비스트리밍) — 출력 어댑터 공통 입력
_OAI_TOOL_RESP = {
    "id": "chatcmpl-x",
    "choices": [{
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"seoul"}'},
            }],
        },
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


async def _collect(agen) -> list[str]:
    return [chunk async for chunk in agen]


async def _fake_openai_stream():
    """text 델타 2개 → tool_call 부분 arguments 2개 → finish 의 OpenAI 스트림."""
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":'}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"seoul"}'}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}},
    ]
    for c in chunks:
        yield f"data: {json.dumps(c)}\n\n"
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────
# Anthropic
# ─────────────────────────────────────────────────────────────
def test_anthropic_inbound_system_and_tool_roundtrip():
    body = {
        "model": "gemini-2.5-flash",  # /v1/messages 로 와도 model 이 라우팅을 결정 → 보존 검증
        "max_tokens": 256,
        "system": "you are helpful",
        "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"city": "seoul"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "sunny"}]},
        ],
    }
    req = anthropic_to_chat_request(body)
    assert req.model == "gemini-2.5-flash"  # 순수 포맷 어댑터 — model 보존
    assert req.max_tokens == 256
    d = req.model_dump(exclude_none=True)
    assert [m["role"] for m in d["messages"]] == ["system", "user", "assistant", "tool"]
    # tool_use → tool_calls
    assert d["messages"][2]["tool_calls"][0]["function"]["name"] == "get_weather"
    # tool_result → tool 메시지(tool_call_id 매칭)
    assert d["messages"][3]["tool_call_id"] == "tu1"
    # any → required
    assert d["tool_choice"] == "required"


def test_anthropic_outbound_tool_use():
    out = openai_to_anthropic_response(_OAI_TOOL_RESP, "gemini-2.5-flash")
    assert out["type"] == "message"
    assert out["stop_reason"] == "tool_use"
    assert [b["type"] for b in out["content"]] == ["text", "tool_use"]
    assert out["content"][1]["input"] == {"city": "seoul"}
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_anthropic_stream_event_sequence():
    events = asyncio.run(_collect(stream_openai_to_anthropic(_fake_openai_stream(), "gemini-2.5-flash")))
    types = [json.loads(e.split("data: ", 1)[1])["type"] for e in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    assert types[-2] == "message_delta"
    # text 블록(0)을 닫은 뒤 tool_use 블록(1)을 연다
    assert "content_block_stop" in types
    assert types.count("content_block_start") == 2  # text + tool_use


# ─────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────
def test_gemini_inbound_system_and_tool_roundtrip():
    body = {
        "systemInstruction": {"parts": [{"text": "sys"}]},
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 128, "stopSequences": ["END"]},
        "tools": [{"functionDeclarations": [{"name": "get_weather", "description": "w", "parameters": {"type": "object"}}]}],
        "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}},
        "contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "seoul"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"t": 20}}}]},
        ],
    }
    req = gemini_to_chat_request(body, "claude-sonnet-4-6", stream=False)
    assert req.model == "claude-sonnet-4-6"  # model 은 URL 경로에서, 보존
    assert req.max_tokens == 128
    d = req.model_dump(exclude_none=True)
    assert [m["role"] for m in d["messages"]] == ["system", "user", "assistant", "tool"]
    # functionCall → tool_calls, functionResponse → tool 메시지(이름 기반 id 매칭)
    assert d["messages"][2]["tool_calls"][0]["id"] == d["messages"][3]["tool_call_id"]
    assert d["stop"] == ["END"]
    # ANY + 단일 허용 함수 → 그 함수 강제
    assert d["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_gemini_outbound_function_call():
    out = openai_to_gemini_response(_OAI_TOOL_RESP, "claude-sonnet-4-6")
    cand = out["candidates"][0]
    assert cand["finishReason"] == "STOP"
    assert cand["content"]["role"] == "model"
    parts = cand["content"]["parts"]
    assert parts[0] == {"text": "ok"}
    assert parts[1]["functionCall"] == {"name": "get_weather", "args": {"city": "seoul"}}
    assert out["usageMetadata"]["totalTokenCount"] == 15


def test_gemini_stream_emits_text_then_functioncall():
    events = asyncio.run(_collect(stream_openai_to_gemini(_fake_openai_stream(), "claude-sonnet-4-6")))
    payloads = [json.loads(e.split("data: ", 1)[1]) for e in events]
    # 앞쪽은 text 청크
    assert payloads[0]["candidates"][0]["content"]["parts"][0]["text"] == "Hel"
    # 마지막 청크에 완성된 functionCall(누적된 부분 arguments 결합)과 finishReason
    final = payloads[-1]["candidates"][0]
    assert final["finishReason"] == "STOP"
    fc = final["content"]["parts"][0]["functionCall"]
    assert fc == {"name": "get_weather", "args": {"city": "seoul"}}
