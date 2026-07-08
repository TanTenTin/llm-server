import json
import time
import uuid
import httpx
from typing import AsyncGenerator

from app.config import settings
from app.models import ChatCompletionRequest, EmbeddingsRequest, Message
from app.providers.base import LLMProvider
from app.providers.openai_payload import build_embeddings_payload
from app.registry import ModelSpec

TIMEOUT = 120.0

# thinking(사고) 계열로 알려진 Ollama 모델 접두사.
# 이 목록에 걸리는 모델에만 think=False를 보낸다 — thinking 미지원 모델(gemma 등)에
# think 필드를 보내면 Ollama가 400("does not support thinking")을 내기 때문이다.
# (heuristic이 놓쳐도 chat()/stream()이 400을 잡아 think 없이 자동 재시도한다.)
_THINKING_MODEL_PREFIXES = ("qwen3", "deepseek-r1", "qwq", "magistral")


def _is_thinking_model(upstream: str) -> bool:
    """업스트림 모델명이 thinking 계열인지 접두사로 추정한다."""
    name = upstream.lower()
    return any(name.startswith(prefix) for prefix in _THINKING_MODEL_PREFIXES)


def _content_to_native(content: object) -> tuple[str, list[str]]:
    """
    OpenAI content(문자열 또는 멀티모달 파트 배열) → (평문 text, images[base64]).
    Ollama 네이티브는 content가 문자열이고 이미지는 별도 images 배열(순수 base64)이다.
    image_url의 data URL(`data:image/png;base64,....`)은 접두사를 떼고 base64만 싣는다.
    """
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "" if content is None else str(content), []

    texts: list[str] = []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and "," in url:
                images.append(url.split(",", 1)[1])  # base64 payload만
            elif url:
                images.append(url)
    return "\n".join(texts), images


def _messages_to_native(messages: list[Message]) -> list[dict]:
    """
    내부표준(OpenAI 포맷) 메시지 → Ollama 네이티브 /api/chat 메시지로 변환한다.
    OpenAI-compat 경로와 달리 네이티브는 몇 가지가 다르다:
      - content 멀티모달 배열 → content(text) + images(base64 배열)
      - assistant tool_calls의 arguments: OpenAI는 JSON '문자열', 네이티브는 '객체'
      - tool 결과 메시지: name → tool_name (있으면)
    """
    out: list[dict] = []
    for message in messages:
        dumped = message.model_dump(exclude_none=True)
        role = dumped.get("role", "user")
        native: dict = {"role": role}

        text, images = _content_to_native(dumped.get("content"))
        native["content"] = text
        if images:
            native["images"] = images

        tool_calls = dumped.get("tool_calls")
        if tool_calls:
            converted: list[dict] = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        args = {}
                converted.append({"function": {"name": fn.get("name"), "arguments": args or {}}})
            native["tool_calls"] = converted

        if role == "tool" and dumped.get("name"):
            native["tool_name"] = dumped["name"]

        out.append(native)
    return out


def _native_tool_calls_to_openai(tool_calls: list) -> list[dict]:
    """Ollama 네이티브 tool_calls(arguments=객체) → OpenAI tool_calls(arguments=JSON 문자열)."""
    result: list[dict] = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments", {})
        result.append({
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": fn.get("name"),
                "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
            },
        })
    return result


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str):
        # Ollama 네이티브 /api/chat 엔드포인트를 사용한다(OpenAI-compat /v1 은 num_ctx·think를
        # 받지 못해 컨텍스트가 서버 기본값으로 잘리고 thinking을 못 끄기 때문).
        # client는 앱 생애주기 동안 재사용 → 커넥션 keep-alive.
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=TIMEOUT)

    def _resolve_think(self, request: ChatCompletionRequest, spec: ModelSpec) -> bool | None:
        """
        이 요청에 보낼 think 값을 결정한다.
          - 요청이 명시하면 그 값 우선
          - 미지정이고 설정(ollama_disable_think)이 켜져 있으며 thinking 계열 모델이면 False
          - 그 외 None(=필드 미포함)
        """
        if request.think is not None:
            return request.think
        if settings.ollama_disable_think and _is_thinking_model(spec.upstream):
            return False
        return None

    def _build_native_payload(
        self, request: ChatCompletionRequest, spec: ModelSpec, think: bool | None
    ) -> dict:
        """네이티브 /api/chat payload. num_ctx·think·tools·num_predict를 실어 보낸다."""
        payload: dict = {
            "model": spec.upstream,
            "messages": _messages_to_native(request.messages),
            "stream": request.stream or False,
        }
        if think is not None:
            payload["think"] = think
        if request.tools:
            # Ollama 네이티브 tools 스키마는 OpenAI와 동일({type, function{name, description, parameters}})
            payload["tools"] = [tool.model_dump(exclude_none=True) for tool in request.tools]

        options: dict = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        max_tokens = request.max_tokens or spec.max_tokens
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if settings.ollama_num_ctx > 0:
            options["num_ctx"] = settings.ollama_num_ctx  # ← 컨텍스트 잘림 방지(핵심)
        if options:
            payload["options"] = options
        return payload

    def _native_to_openai(self, native: dict, model: str) -> dict:
        """네이티브 /api/chat 단일 응답 → OpenAI chat.completion 형식 변환(tool_calls 포함)."""
        msg = native.get("message", {})
        prompt_tokens = native.get("prompt_eval_count", 0)
        completion_tokens = native.get("eval_count", 0)

        openai_msg: dict = {"role": msg.get("role", "assistant"), "content": msg.get("content", "")}
        if msg.get("tool_calls"):
            openai_msg["tool_calls"] = _native_tool_calls_to_openai(msg["tool_calls"])

        # tool_calls가 있으면 finish_reason은 OpenAI 관례상 tool_calls
        finish_reason = native.get("done_reason", "stop")
        if openai_msg.get("tool_calls"):
            finish_reason = "tool_calls"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": openai_msg, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _is_think_unsupported_error(self, exc: httpx.HTTPStatusError) -> bool:
        """400 응답이 'thinking 미지원' 때문인지 판별(heuristic이 놓친 모델 대비 graceful 재시도용)."""
        if exc.response.status_code != 400:
            return False
        try:
            body = exc.response.text.lower()
        except Exception:
            return False
        return "think" in body

    async def chat(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        think = self._resolve_think(request, spec)
        payload = self._build_native_payload(request, spec, think)
        payload["stream"] = False
        try:
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # thinking 미지원 모델에 think를 보낸 경우: think 빼고 1회 재시도
            if think is not None and self._is_think_unsupported_error(exc):
                payload.pop("think", None)
                response = await self.client.post("/api/chat", json=payload)
                response.raise_for_status()
            else:
                raise
        return self._native_to_openai(response.json(), spec.upstream)

    async def list_models(self) -> list[dict]:
        """
        Ollama 서버에 실제 설치된 모델 목록을 조회한다(/api/tags).
        로컬 모델은 운영 중 pull/rm으로 자주 바뀌므로 레지스트리에 고정하지 않고
        실시간 조회한다(SaaS provider는 정적 레지스트리로 관리). 서버 미가용 시
        예외를 그대로 올려 호출 측(/v1/models)이 graceful degrade 하도록 둔다.

        각 항목은 최소 "name"(태그)과, Ollama가 제공하면 "capabilities"
        (["embedding"] · ["completion","tools",...] 등)를 담는다. 호출 측이
        capabilities로 chat/embedding 여부를 정확히 구분한다(구버전엔 없을 수 있음).
        """
        # /v1/models 응답이 로컬 서버 다운으로 오래 매달리지 않도록 짧은 타임아웃 사용.
        response = await self.client.get("/api/tags", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return [m for m in data.get("models", []) if m.get("name")]

    async def embed(self, request: EmbeddingsRequest, spec: ModelSpec) -> dict:
        """Ollama OpenAI 호환 /v1/embeddings 프록시. 모델은 사전 pull 필요(미설치면 404 → 폴백)."""
        response = await self.client.post(
            "/v1/embeddings", json=build_embeddings_payload(request, spec)
        )
        response.raise_for_status()
        return response.json()

    async def _send_stream(self, payload: dict) -> httpx.Response:
        """스트림 응답 1회 전송. 에러면 본문을 읽어(raise 전) 상태 판별이 가능하게 한다."""
        response = await self.client.send(
            self.client.build_request("POST", "/api/chat", json=payload), stream=True
        )
        if response.is_error:
            # stream=True 응답은 에러 시 본문이 자동으로 안 읽힘 → .text 접근 전에 읽어둔다.
            await response.aread()
            response.raise_for_status()
        return response

    async def _open_native_stream(self, payload: dict) -> httpx.Response:
        """네이티브 /api/chat 스트림을 연다. thinking 미지원 400이면 think 빼고 재시도."""
        try:
            return await self._send_stream(payload)
        except httpx.HTTPStatusError as exc:
            if payload.get("think") is not None and self._is_think_unsupported_error(exc):
                await exc.response.aclose()
                payload.pop("think", None)
                return await self._send_stream(payload)
            raise

    async def stream(
        self, request: ChatCompletionRequest, spec: ModelSpec
    ) -> AsyncGenerator[str, None]:
        think = self._resolve_think(request, spec)
        payload = self._build_native_payload(request, spec, think)
        payload["stream"] = True

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        first = True

        response = await self._open_native_stream(payload)
        try:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message", {})
                done = chunk.get("done", False)

                delta: dict = {}
                if first:
                    delta["role"] = "assistant"
                    first = False
                content = msg.get("content", "")
                if content:
                    delta["content"] = content
                # 네이티브 스트림은 보통 마지막 청크에 tool_calls를 싣는다 → OpenAI 델타로 변환
                if msg.get("tool_calls"):
                    converted = _native_tool_calls_to_openai(msg["tool_calls"])
                    delta["tool_calls"] = [
                        {"index": i, **tc} for i, tc in enumerate(converted)
                    ]

                finish_reason = None
                if done:
                    finish_reason = "tool_calls" if msg.get("tool_calls") else chunk.get("done_reason", "stop")

                sse = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": spec.upstream,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
                # 마지막(done) 청크에 usage를 실어 게이트웨이가 스트리밍 토큰을 집계하게 한다
                # (E-01 — OpenAI 스트리밍 usage 청크 관례. 로컬은 무료라 관측 목적).
                if done:
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)
                    sse["usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                yield f"data: {json.dumps(sse)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        await self.client.aclose()
