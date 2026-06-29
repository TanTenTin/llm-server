"""
Gemini generateContent API ⇄ OpenAI chat.completions 변환.

inbound:  POST /v1beta/models/{model}:generateContent 의 Gemini 요청 dict → ChatCompletionRequest
outbound: 내부 파이프라인의 OpenAI 응답 → Gemini GenerateContentResponse / SSE 스트림

순수 포맷 어댑터: model 은 URL 경로에서 받아 그대로 라우팅에 넘긴다(provider는 기존 로직이 결정).

한계(v1): Gemini functionResponse 에는 호출 id가 없어, functionCall/functionResponse 를
이름 기반(call_<name>)으로 매칭한다. 같은 이름의 도구를 한 턴에 여러 번 호출하면 충돌할 수 있다.
이미지(inlineData) 입력은 OpenAI image_url(data URL)로 변환한다.
"""

import json
from typing import Any, AsyncGenerator

from app.adapters.sse import format_data, sse_payloads
from app.models import ChatCompletionRequest

# ── finish_reason 매핑 ────────────────────────────────────────
# OpenAI finish_reason → Gemini finishReason
_OPENAI_TO_GEMINI_FINISH = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "tool_calls": "STOP",       # Gemini는 함수호출도 STOP으로 종료
    "function_call": "STOP",
    "content_filter": "SAFETY",
}


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────
def _fc_id(name: str | None) -> str:
    """functionCall/functionResponse 매칭용 결정적 tool_call_id (이름 기반)."""
    return f"call_{name or 'unknown'}"


def _parts_text(parts: Any) -> str | None:
    """parts 배열에서 text 조각만 모아 평문으로."""
    if not isinstance(parts, list):
        return None
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
    joined = "\n".join(t for t in texts if t)
    return joined or None


def _convert_content(content: dict) -> list[dict]:
    """
    Gemini content(role + parts) → OpenAI 메시지들.
    role "model" → assistant, 그 외 → user. functionResponse 는 별도 tool 메시지로 분리.
    """
    role = content.get("role", "user")
    oai_role = "assistant" if role == "model" else "user"
    parts = content.get("parts", [])

    text_segments: list[str] = []
    image_parts: list[dict] = []
    tool_calls: list[dict] = []
    tool_msgs: list[dict] = []

    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            continue
        if "text" in part:
            text_segments.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append({
                "id": _fc_id(fc.get("name")),
                "type": "function",
                "function": {
                    "name": fc.get("name"),
                    "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                },
            })
        elif "functionResponse" in part:
            fr = part["functionResponse"]
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": _fc_id(fr.get("name")),
                "content": json.dumps(fr.get("response", {}), ensure_ascii=False),
            })
        elif "inlineData" in part:
            data = part["inlineData"]
            mime = data.get("mimeType", "image/png")
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data.get('data', '')}"},
            })

    messages: list[dict] = list(tool_msgs)
    text = "\n".join(text_segments) if text_segments else None

    if oai_role == "assistant":
        if text or tool_calls:
            msg: dict = {"role": "assistant", "content": text}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
    else:
        if image_parts:
            content_parts: list[dict] = list(image_parts)
            if text:
                content_parts.insert(0, {"type": "text", "text": text})
            messages.append({"role": "user", "content": content_parts})
        elif text is not None:
            messages.append({"role": "user", "content": text})

    return messages


def _convert_tools(tools: Any) -> list[dict]:
    """Gemini tools([{functionDeclarations:[...]}]) → OpenAI tools."""
    converted: list[dict] = []
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        for decl in tool.get("functionDeclarations", []) or []:
            converted.append({
                "type": "function",
                "function": {
                    "name": decl.get("name"),
                    "description": decl.get("description", ""),
                    "parameters": decl.get("parameters", {"type": "object", "properties": {}}),
                },
            })
    return converted


def _convert_tool_config(tool_config: Any) -> Any:
    """Gemini toolConfig.functionCallingConfig → OpenAI tool_choice."""
    if not isinstance(tool_config, dict):
        return None
    cfg = tool_config.get("functionCallingConfig", {})
    mode = (cfg.get("mode") or "").upper()
    allowed = cfg.get("allowedFunctionNames") or []
    if mode == "NONE":
        return "none"
    if mode == "AUTO":
        return "auto"
    if mode == "ANY":
        # 특정 함수 1개만 허용하면 그 함수 강제, 아니면 'required'
        if len(allowed) == 1:
            return {"type": "function", "function": {"name": allowed[0]}}
        return "required"
    return None


# ─────────────────────────────────────────────────────────────
# inbound: Gemini 요청 → ChatCompletionRequest
# ─────────────────────────────────────────────────────────────
def gemini_to_chat_request(body: dict, model: str, stream: bool) -> ChatCompletionRequest:
    """Gemini generateContent 요청 dict → 내부표준 ChatCompletionRequest (model은 URL 경로에서)."""
    messages: list[dict] = []

    # systemInstruction (camelCase / snake_case 둘 다 허용)
    system = body.get("systemInstruction") or body.get("system_instruction")
    if isinstance(system, dict):
        sys_text = _parts_text(system.get("parts"))
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

    for content in body.get("contents", []):
        if isinstance(content, dict):
            messages.extend(_convert_content(content))

    payload: dict = {"model": model, "messages": messages, "stream": stream}

    gen_config = body.get("generationConfig") or body.get("generation_config") or {}
    if gen_config.get("temperature") is not None:
        payload["temperature"] = gen_config["temperature"]
    if gen_config.get("maxOutputTokens") is not None:
        payload["max_tokens"] = gen_config["maxOutputTokens"]
    # 아래는 extra="allow" 로 보존되어 openai_payload 화이트리스트로 업스트림에 전달됨
    if gen_config.get("topP") is not None:
        payload["top_p"] = gen_config["topP"]
    if gen_config.get("topK") is not None:
        payload["top_k"] = gen_config["topK"]
    if gen_config.get("stopSequences"):
        payload["stop"] = gen_config["stopSequences"]

    if body.get("tools"):
        payload["tools"] = _convert_tools(body["tools"])
    tool_choice = _convert_tool_config(body.get("toolConfig") or body.get("tool_config"))
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    return ChatCompletionRequest.model_validate(payload)


# ─────────────────────────────────────────────────────────────
# outbound: OpenAI 응답 → Gemini 응답 (비스트리밍)
# ─────────────────────────────────────────────────────────────
def _message_to_parts(message: dict) -> list[dict]:
    """OpenAI assistant 메시지 → Gemini parts(text / functionCall)."""
    parts: list[dict] = []
    text = message.get("content")
    if text:
        parts.append({"text": text})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        parts.append({"functionCall": {"name": fn.get("name"), "args": args}})
    if not parts:
        parts.append({"text": ""})
    return parts


def openai_to_gemini_response(resp: dict, model: str) -> dict:
    """OpenAI chat.completion → Gemini GenerateContentResponse."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message", {})
    usage = resp.get("usage", {})
    return {
        "candidates": [{
            "content": {"role": "model", "parts": _message_to_parts(message)},
            "finishReason": _OPENAI_TO_GEMINI_FINISH.get(choice.get("finish_reason"), "STOP"),
            "index": 0,
            "safetyRatings": [],
        }],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
        "modelVersion": model,
    }


# ─────────────────────────────────────────────────────────────
# outbound: OpenAI SSE → Gemini SSE 스트림
# ─────────────────────────────────────────────────────────────
async def stream_openai_to_gemini(
    source: AsyncGenerator[str, None], model: str
) -> AsyncGenerator[str, None]:
    """
    OpenAI 스트림 청크 → Gemini streamGenerateContent(alt=sse) 청크로 변환.
    text 델타는 즉시 청크로 흘려보낸다. tool_calls 는 부분 arguments를 누적했다가
    마지막에 완성된 functionCall part로 한 번에 내보낸다(Gemini는 부분 args 스트림이 없음).
    """
    tool_acc: dict[int, dict] = {}   # OpenAI tool index → {"name","args"}
    finish_reason: str | None = None
    usage: dict = {}

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
                    usage = chunk["usage"]

                content = delta.get("content")
                if content:
                    yield format_data({
                        "candidates": [{
                            "content": {"role": "model", "parts": [{"text": content}]},
                            "index": 0,
                        }],
                    })

                for tc in delta.get("tool_calls") or []:
                    oi = tc.get("index", 0)
                    acc = tool_acc.setdefault(oi, {"name": "", "args": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        acc["name"] = fn["name"]
                    if fn.get("arguments"):
                        acc["args"] += fn["arguments"]

        # 마지막 청크: 완성된 functionCall + finishReason + usage
        final_parts: list[dict] = []
        for acc in tool_acc.values():
            try:
                args = json.loads(acc["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            final_parts.append({"functionCall": {"name": acc["name"], "args": args}})

        final_candidate: dict = {
            "finishReason": _OPENAI_TO_GEMINI_FINISH.get(finish_reason, "STOP"),
            "index": 0,
        }
        if final_parts:
            final_candidate["content"] = {"role": "model", "parts": final_parts}
        payload: dict = {"candidates": [final_candidate]}
        if usage:
            payload["usageMetadata"] = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            }
        yield format_data(payload)
    finally:
        await source.aclose()
