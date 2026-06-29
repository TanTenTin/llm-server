"""
Anthropic Messages API ⇄ OpenAI chat.completions 변환.

inbound:  POST /v1/messages 의 Anthropic 요청 dict → ChatCompletionRequest(내부표준)
outbound: 내부 파이프라인의 OpenAI 응답 → Anthropic Messages 응답 / SSE 이벤트 스트림

순수 포맷 어댑터: model 필드는 그대로 두므로 라우팅(provider 선택/폴백)은 기존 로직이 결정한다.
즉 /v1/messages 로 'gemini-2.5-flash' 를 요청하면 Anthropic 포맷으로 받아 Gemini로 라우팅된다.
"""

import json
import uuid
from typing import Any, AsyncGenerator

from app.adapters.sse import format_event, sse_payloads
from app.models import ChatCompletionRequest

# ── finish_reason 매핑 ────────────────────────────────────────
# OpenAI finish_reason → Anthropic stop_reason
_OPENAI_TO_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────
def _stringify(content: Any) -> str:
    """tool_result content(문자열 또는 블록 배열)를 평문 문자열로 정규화."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                else:
                    texts.append(json.dumps(block, ensure_ascii=False))
            else:
                texts.append(str(block))
        return "\n".join(texts)
    return json.dumps(content, ensure_ascii=False)


def _flatten_system(system: Any) -> str | None:
    """Anthropic system(문자열 또는 [{"type":"text","text":...}] 블록) → 평문."""
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts = [
            b.get("text", "")
            for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        joined = "\n".join(p for p in parts if p)
        return joined or None
    return None


def _convert_user_message(content: Any) -> list[dict]:
    """
    Anthropic user 메시지 → OpenAI 메시지들. 한 user 메시지가 여러 OpenAI 메시지로 갈릴 수 있다:
      - tool_result 블록 → 별도 {"role":"tool", ...} 메시지 (선행 assistant tool_use 응답)
      - text/image 블록   → 하나의 {"role":"user", ...} 메시지 (이미지 있으면 멀티모달 배열)
    tool 메시지를 먼저, 그 다음 user 메시지 순으로 둔다(OpenAI는 tool 결과가 assistant 뒤에 와야 함).
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    tool_msgs: list[dict] = []
    parts: list[dict] = []  # 멀티모달 OpenAI content parts
    plain_texts: list[str] = []
    has_image = False

    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            plain_texts.append(text)
            parts.append({"type": "text", "text": text})
        elif btype == "image":
            has_image = True
            source = block.get("source", {})
            if source.get("type") == "base64":
                media = source.get("media_type", "image/png")
                url = f"data:{media};base64,{source.get('data', '')}"
            else:
                url = source.get("url", "")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif btype == "tool_result":
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id"),
                "content": _stringify(block.get("content", "")),
            })

    messages = list(tool_msgs)
    if has_image:
        messages.append({"role": "user", "content": parts})
    elif plain_texts:
        messages.append({"role": "user", "content": "\n".join(plain_texts)})
    return messages


def _convert_assistant_message(content: Any) -> dict:
    """
    Anthropic assistant 메시지 → OpenAI assistant 메시지.
    text 블록 → content, tool_use 블록 → tool_calls(arguments는 JSON 문자열).
    """
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                },
            })

    msg: dict = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tools(name/description/input_schema) → OpenAI tools(function/parameters)."""
    converted: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict) or "name" not in tool:
            continue  # 내장 도구(예: web_search 등 input_schema 없는 형태)는 건너뜀
        converted.append({
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return converted


def _convert_tool_choice(tool_choice: Any) -> Any:
    """Anthropic tool_choice → OpenAI tool_choice."""
    if not isinstance(tool_choice, dict):
        return None
    ctype = tool_choice.get("type")
    if ctype == "auto":
        return "auto"
    if ctype == "any":
        return "required"
    if ctype == "none":
        return "none"
    if ctype == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return None


# ─────────────────────────────────────────────────────────────
# inbound: Anthropic 요청 → ChatCompletionRequest
# ─────────────────────────────────────────────────────────────
def anthropic_to_chat_request(body: dict) -> ChatCompletionRequest:
    """Anthropic Messages 요청 dict → 내부표준 ChatCompletionRequest."""
    if "model" not in body:
        raise ValueError("Anthropic 요청에 'model' 이 없습니다")

    messages: list[dict] = []
    system = _flatten_system(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})

    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            messages.append(_convert_assistant_message(content))
        else:  # "user" (그 외 역할도 user로 취급)
            messages.extend(_convert_user_message(content))

    payload: dict = {
        "model": body["model"],
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }
    # Anthropic은 max_tokens 필수 — 그대로 전달
    if body.get("max_tokens") is not None:
        payload["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    # 아래는 extra="allow" 로 보존되고 openai_payload._FORWARD_PARAMS 화이트리스트로 업스트림 전달됨
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]
    if body.get("tools"):
        payload["tools"] = _convert_tools(body["tools"])
    tool_choice = _convert_tool_choice(body.get("tool_choice"))
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    return ChatCompletionRequest.model_validate(payload)


# ─────────────────────────────────────────────────────────────
# outbound: OpenAI 응답 → Anthropic 응답 (비스트리밍)
# ─────────────────────────────────────────────────────────────
def openai_to_anthropic_response(resp: dict, model: str) -> dict:
    """OpenAI chat.completion → Anthropic Messages 응답."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message", {})

    content_blocks: list[dict] = []
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name"),
            "input": args,
        })
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = resp.get("usage", {})
    return {
        "id": resp.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": _OPENAI_TO_STOP_REASON.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ─────────────────────────────────────────────────────────────
# outbound: OpenAI SSE → Anthropic SSE 이벤트 스트림
# ─────────────────────────────────────────────────────────────
async def stream_openai_to_anthropic(
    source: AsyncGenerator[str, None], model: str
) -> AsyncGenerator[str, None]:
    """
    OpenAI 스트림 청크 → Anthropic Messages 스트리밍 이벤트 시퀀스로 변환.
      message_start → (content_block_start → content_block_delta* → content_block_stop)+ →
      message_delta → message_stop
    text delta 와 tool_calls(input_json_delta) 모두 처리한다. text 와 tool_use 가 섞이면
    실제 Anthropic 스트림처럼 text 블록을 먼저 닫고 tool_use 블록을 연다.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield format_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    text_index: int | None = None       # 열린 text 블록의 인덱스 (없으면 None)
    tool_blocks: dict[int, int] = {}     # OpenAI tool_call index → Anthropic 블록 인덱스
    next_index = 0
    finish_reason: str | None = None
    output_tokens = 0

    try:
        async for raw in source:
            for data in sse_payloads(raw):
                if data == "[DONE]" or not data:
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta", {})
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                if chunk.get("usage"):
                    output_tokens = chunk["usage"].get("completion_tokens", output_tokens)

                # 1) 텍스트 델타
                content = delta.get("content")
                if content:
                    if text_index is None:
                        text_index = next_index
                        next_index += 1
                        yield format_event("content_block_start", {
                            "type": "content_block_start",
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    yield format_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": content},
                    })

                # 2) tool_call 델타
                for tc in delta.get("tool_calls") or []:
                    oi = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    if oi not in tool_blocks:
                        # text 블록이 열려 있으면 먼저 닫는다(블록은 순차적으로 시작/종료)
                        if text_index is not None:
                            yield format_event("content_block_stop", {
                                "type": "content_block_stop", "index": text_index,
                            })
                            text_index = None
                        ai = next_index
                        next_index += 1
                        tool_blocks[oi] = ai
                        yield format_event("content_block_start", {
                            "type": "content_block_start",
                            "index": ai,
                            "content_block": {
                                "type": "tool_use",
                                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": fn.get("name") or "",
                                "input": {},
                            },
                        })
                    args = fn.get("arguments")
                    if args:
                        yield format_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": tool_blocks[oi],
                            "delta": {"type": "input_json_delta", "partial_json": args},
                        })

        # 열린 블록 닫기
        if text_index is not None:
            yield format_event("content_block_stop", {
                "type": "content_block_stop", "index": text_index,
            })
        for ai in tool_blocks.values():
            yield format_event("content_block_stop", {
                "type": "content_block_stop", "index": ai,
            })

        yield format_event("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": _OPENAI_TO_STOP_REASON.get(finish_reason, "end_turn"),
                "stop_sequence": None,
            },
            "usage": {"output_tokens": output_tokens},
        })
        yield format_event("message_stop", {"type": "message_stop"})
    finally:
        await source.aclose()
