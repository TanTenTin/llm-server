import json
import uuid
from typing import AsyncGenerator

import anthropic

from app.models import ChatCompletionRequest, Message, Tool
from app.providers.base import LLMProvider
from app.registry import ModelSpec

# 요청·레지스트리 모두 max_tokens가 없을 때 쓰는 최후 기본값
DEFAULT_MAX_TOKENS = 8192

# 네이티브 패스스루(/v1/messages → Anthropic)에서 SDK 호출 인자로 '그대로' 넘길 표준 파라미터.
# SDK 버전마다 지원 kwargs가 달라질 수 있어, 이 목록 밖의 top-level 필드(예: thinking, beta 옵션)는
# extra_body 로 보내 JSON 본문에 그대로 병합한다(미지 kwargs로 인한 TypeError 회피).
_ANTHROPIC_PASSTHROUGH = {
    "messages", "system", "max_tokens", "metadata", "stop_sequences",
    "temperature", "top_p", "top_k", "tools", "tool_choice",
}


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        # client는 앱 생애주기 동안 재사용. api_key가 빈 값이어도 생성 시점에는
        # 예외가 나지 않으며, 실제 호출 시 인증 오류로 드러난다(Ollama 전용 사용 가능).
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def _extract_system(self, messages: list[Message]) -> tuple[str, list[Message]]:
        """
        Anthropic은 system 프롬프트를 messages 배열이 아닌 별도 파라미터로 받음.
        system 메시지를 분리해서 반환한다.
        주의: system 메시지가 여러 개면 마지막 것만 사용하고,
        content가 문자열이 아니면(list 등) 무시된다.
        """
        system = ""
        others = []
        for msg in messages:
            if msg.role == "system":
                system = msg.content if isinstance(msg.content, str) else ""
            else:
                others.append(msg)
        return system, others

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """OpenAI 메시지 형식 → Anthropic 메시지 형식 변환"""
        result = []
        for msg in messages:
            if msg.role == "tool":
                # OpenAI tool result → Anthropic tool_result 블록
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": (
                                msg.content
                                if isinstance(msg.content, str)
                                else json.dumps(msg.content)
                            ),
                        }
                    ],
                })
            elif msg.role == "assistant" and isinstance(msg.content, list):
                # tool_calls 포함된 assistant 메시지 변환
                content = []
                for item in msg.content:
                    if isinstance(item, dict):
                        content.append(item)
                result.append({"role": "assistant", "content": content})
            else:
                result.append({
                    "role": msg.role,
                    "content": (
                        msg.content
                        if isinstance(msg.content, str)
                        else json.dumps(msg.content)
                    ),
                })
        return result

    def _convert_tools(self, tools: list[Tool]) -> list[dict]:
        """OpenAI tools 형식 → Anthropic tools 형식 변환"""
        return [
            {
                "name": t.function.name,
                "description": t.function.description or "",
                # Anthropic은 JSON Schema를 input_schema 키로 받음
                "input_schema": t.function.parameters or {
                    "type": "object",
                    "properties": {},
                },
            }
            for t in tools
        ]

    def _to_openai_response(self, response, model: str) -> dict:
        """Anthropic 응답 → OpenAI chat.completion 형식 변환"""
        choice: dict = {"index": 0, "finish_reason": response.stop_reason}

        if response.stop_reason == "tool_use":
            # tool_use 블록을 OpenAI tool_calls 형식으로
            tool_calls = [
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
                for block in response.content
                if block.type == "tool_use"
            ]
            text_content = " ".join(
                block.text for block in response.content if block.type == "text"
            )
            choice["message"] = {
                "role": "assistant",
                "content": text_content or None,
                "tool_calls": tool_calls,
            }
        else:
            text = " ".join(
                block.text for block in response.content if block.type == "text"
            )
            choice["message"] = {"role": "assistant", "content": text}

        return {
            "id": f"chatcmpl-{response.id}",
            "object": "chat.completion",
            "model": model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }

    def _build_params(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        system, messages = self._extract_system(request.messages)
        params: dict = {
            "model": spec.upstream,
            "messages": self._convert_messages(messages),
            # 요청값 > 레지스트리 기본값 > 최후 기본값
            "max_tokens": request.max_tokens or spec.max_tokens or DEFAULT_MAX_TOKENS,
        }
        if system:
            params["system"] = system
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.tools:
            params["tools"] = self._convert_tools(request.tools)
        return params

    # ── 네이티브 패스스루 (/v1/messages → Anthropic) ──────────────
    # OpenAI 내부표준을 거치지 않고 클라이언트의 Anthropic 요청을 그대로 SDK로 보낸다.
    # 이중 변환(Anthropic→OpenAI→Anthropic)을 없애 cache_control·정확한 content 블록·
    # 멀티턴 tool_use·스트리밍 tool_use(input_json_delta)가 손실 없이 보존된다.
    def _native_params(self, body: dict, spec: ModelSpec) -> dict:
        """
        클라이언트 Anthropic Messages body → SDK 호출 인자. model은 레지스트리 spec.upstream으로
        보정하고, max_tokens가 없으면 spec/기본값으로 채운다. 표준 외 top-level 필드는 extra_body로.
        """
        known: dict = {}
        extra: dict = {}
        for key, value in body.items():
            if key in ("model", "stream"):
                continue  # model은 레지스트리가 결정, stream은 호출 측에서 별도 지정
            (known if key in _ANTHROPIC_PASSTHROUGH else extra)[key] = value
        known["model"] = spec.upstream
        known.setdefault("max_tokens", spec.max_tokens or DEFAULT_MAX_TOKENS)
        if extra:
            known["extra_body"] = extra
        return known

    async def chat_native(self, body: dict, spec: ModelSpec) -> dict:
        """Anthropic 네이티브 패스스루(비스트리밍). 응답도 Anthropic Messages 포맷 dict 그대로 반환."""
        params = self._native_params(body, spec)
        response = await self.client.messages.create(**params)
        return response.model_dump()

    async def stream_native(
        self, body: dict, spec: ModelSpec
    ) -> AsyncGenerator[str, None]:
        """
        Anthropic 네이티브 패스스루(스트리밍). SDK 원본 raw 이벤트를 Anthropic 와이어 SSE로
        그대로 중계한다(`event: <type>\\ndata: <json>\\n\\n`). 변환을 거치지 않아 tool_use
        스트리밍이 누락되지 않는다(기존 OpenAI 변환 경로의 한계 회피).
        """
        params = self._native_params(body, spec)
        params["stream"] = True
        stream = await self.client.messages.create(**params)
        async for event in stream:
            yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"

    async def chat(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        params = self._build_params(request, spec)
        response = await self.client.messages.create(**params)
        return self._to_openai_response(response, request.model)

    async def stream(
        self, request: ChatCompletionRequest, spec: ModelSpec
    ) -> AsyncGenerator[str, None]:
        params = self._build_params(request, spec)

        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # 스트림 종료 후 최종 메시지의 usage를 청크로 노출한다(E-01 — 게이트웨이가
            # 스트리밍 토큰을 집계해 과금 예산 가드가 스트리밍으로 우회되지 않게 한다).
            final = await stream.get_final_message()
            prompt_tokens = final.usage.input_tokens
            completion_tokens = final.usage.output_tokens
            usage_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    async def aclose(self) -> None:
        # SDK 버전에 따라 close 메서드명이 다를 수 있어 방어적으로 처리
        closer = getattr(self.client, "aclose", None) or getattr(self.client, "close", None)
        if closer is not None:
            await closer()
