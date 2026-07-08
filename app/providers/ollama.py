import asyncio
import base64
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx

from app.config import settings
from app.models import ChatCompletionRequest, EmbeddingsRequest, Message
from app.providers.base import LLMProvider
from app.providers.openai_payload import build_embeddings_payload
from app.registry import ModelSpec, estimate_tokens

TIMEOUT = 120.0

# (E-08) 동적 num_ctx 산정 파라미터. 요청 크기에 맞춰 창을 필요한 만큼만 잡아 작은 요청의
# KV 캐시 적재 비용을 줄인다(큰 요청은 설정 상한까지 확장). 추정이 근사치라 여유를 둔다.
_INPUT_HEADROOM = 1.25          # 입력 토큰 추정에 곱하는 안전 계수(과소추정 흡수)
_DEFAULT_OUTPUT_BUDGET = 2048   # max_tokens 미지정 시 출력용으로 예약할 토큰
_CTX_MARGIN = 1024              # 템플릿·특수토큰 등 추가 여유
_MIN_NUM_CTX = 4096             # 너무 작게 잡아 잘리는 것을 막는 하한
_CTX_ROUND = 2048               # num_ctx를 이 배수로 올림(할당 정렬)

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


# (E-09) OpenAI 요청 파라미터 → Ollama options 로 넘길 샘플링 키.
# extra="allow" 덕에 model_dump에는 모델에 선언되지 않은 top_p/seed 등도 담겨 있다.
_SAMPLING_PARAMS = (
    "temperature", "top_p", "top_k", "seed",
    "presence_penalty", "frequency_penalty", "repeat_penalty",
)


def _build_options(request: ChatCompletionRequest) -> dict:
    """
    (E-09) 요청의 샘플링 파라미터를 Ollama options 로 매핑한다. 예전엔 temperature만 전달돼
    top_p/top_k/stop/seed/penalty 계열이 무시됐다. stop은 문자열/배열 모두 배열로 정규화한다.
    """
    dumped = request.model_dump(exclude_none=True)
    options: dict = {}
    for key in _SAMPLING_PARAMS:
        if dumped.get(key) is not None:
            options[key] = dumped[key]
    stop = dumped.get("stop")
    if stop is not None:
        options["stop"] = stop if isinstance(stop, list) else [stop]
    return options


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
        # (E-07) 원격 이미지 URL을 base64로 받아오는 별도 client(base_url 없이 임의 호스트 호출).
        self._img_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        # (E-06) 로컬 동시 요청 상한. 단일 GPU/CPU에서 동시 호출이 몰릴 때 직렬화 정도를 제어.
        n = settings.ollama_max_concurrency
        self._sem: asyncio.Semaphore | None = asyncio.Semaphore(n) if n > 0 else None

    @asynccontextmanager
    async def _guard(self) -> AsyncGenerator[None, None]:
        """(E-06) 동시성 세마포어 가드. 설정이 0 이하면 아무 제한 없이 통과한다."""
        if self._sem is None:
            yield
            return
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()

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

    async def _fetch_as_data_url(self, url: str) -> str:
        """(E-07) 원격 이미지 URL을 받아 base64로 인코딩한다. 크기 상한 초과/실패 시 ValueError."""
        response = await self._img_client.get(url)
        response.raise_for_status()
        data = response.content
        if len(data) > settings.ollama_max_image_bytes:
            raise ValueError(
                f"원격 이미지가 너무 큼({len(data)} bytes > {settings.ollama_max_image_bytes})"
            )
        return base64.b64encode(data).decode("ascii")

    async def _resolve_remote_images(self, request: ChatCompletionRequest) -> None:
        """
        (E-07) 메시지의 원격(http/https) 이미지 URL을 게이트웨이가 fetch해 base64 data URL로
        바꿔 넣는다(Ollama images 필드는 base64만 받으므로). 요청 객체의 content를 제자리에서
        수정하며, data URL/파일 경로 등 원격이 아닌 것은 건드리지 않는다.
        """
        for message in request.messages:
            content = message.content
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url") or {}
                url = image_url.get("url", "")
                if url.startswith(("http://", "https://")):
                    b64 = await self._fetch_as_data_url(url)
                    image_url["url"] = f"data:image/*;base64,{b64}"
                    part["image_url"] = image_url

    def _resolve_num_ctx(self, request: ChatCompletionRequest) -> int | None:
        """
        (E-08) 요청 크기에 맞춰 num_ctx를 산정한다. 설정값(ollama_num_ctx)을 상한으로,
        '추정 입력×여유 + 출력 예산 + 마진'을 하한(_MIN_NUM_CTX)과 함께 클램프해 배수로 올린다.
        작은 요청은 창을 작게 잡아 KV 캐시 적재를 아끼고, 큰 요청은 설정 상한까지 확장한다.
        설정이 0 이하면 None(=서버 기본 사용, 라우터의 context_window도 보수적으로 잡힘).
        """
        configured = settings.ollama_num_ctx
        if configured <= 0:
            return None
        needed = int(estimate_tokens(request) * _INPUT_HEADROOM)
        needed += (request.max_tokens or _DEFAULT_OUTPUT_BUDGET) + _CTX_MARGIN
        needed = min(max(needed, _MIN_NUM_CTX), configured)
        rounded = ((needed + _CTX_ROUND - 1) // _CTX_ROUND) * _CTX_ROUND
        return min(rounded, configured)

    def _build_native_payload(
        self, request: ChatCompletionRequest, spec: ModelSpec, think: bool | None
    ) -> dict:
        """네이티브 /api/chat payload. num_ctx·think·tools·num_predict·샘플링·keep_alive를 싣는다."""
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
        # (E-05) 모델 상주 시간 — 콜드스타트(모델+KV 캐시 재적재) 지연을 줄인다.
        if settings.ollama_keep_alive:
            payload["keep_alive"] = settings.ollama_keep_alive

        options = _build_options(request)
        num_ctx = self._resolve_num_ctx(request)          # (E-08) 동적 컨텍스트 창
        if num_ctx is not None:
            options["num_ctx"] = num_ctx                  # ← 컨텍스트 잘림 방지(핵심)
        max_tokens = request.max_tokens or spec.max_tokens
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options
        return payload

    def _native_to_openai(self, native: dict, model: str) -> dict:
        """네이티브 /api/chat 단일 응답 → OpenAI chat.completion 형식 변환(tool_calls 포함)."""
        msg = native.get("message", {})
        prompt_tokens = native.get("prompt_eval_count", 0)
        completion_tokens = native.get("eval_count", 0)

        openai_msg: dict = {"role": msg.get("role", "assistant"), "content": msg.get("content", "")}
        # (E-04) Ollama는 reasoning을 별도 message.thinking으로 준다 — 예전엔 content만 읽어
        # think=True로 켜도 조용히 유실됐다. reasoning_content로 노출한다.
        if msg.get("thinking"):
            openai_msg["reasoning_content"] = msg["thinking"]
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
        await self._resolve_remote_images(request)   # (E-07) 원격 이미지 → base64
        think = self._resolve_think(request, spec)
        payload = self._build_native_payload(request, spec, think)
        payload["stream"] = False
        async with self._guard():                     # (E-06) 로컬 동시성 제어
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
        async with self._guard():                     # (E-06) 로컬 동시성 제어
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
        await self._resolve_remote_images(request)   # (E-07) 원격 이미지 → base64
        think = self._resolve_think(request, spec)
        payload = self._build_native_payload(request, spec, think)
        payload["stream"] = True

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        first = True
        # (E-10) 한 스트림 내 tool_call id를 인덱스별로 고정한다 — 예전엔 청크마다 새 uuid를
        # 생성해 tool_call이 여러 청크에 걸치면 id가 어긋났다. saw_tool_calls로 done 청크에
        # tool_calls가 없어도 finish_reason을 tool_calls로 정확히 잡는다.
        tool_call_ids: dict[int, str] = {}
        saw_tool_calls = False

        async with self._guard():                     # (E-06) 스트림 전체 동안 슬롯 점유
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
                    # (E-04) reasoning(thinking) 델타도 흘려보낸다(유실 방지)
                    if msg.get("thinking"):
                        delta["reasoning_content"] = msg["thinking"]
                    # 네이티브 스트림은 보통 마지막 청크에 tool_calls를 싣는다 → OpenAI 델타로 변환
                    if msg.get("tool_calls"):
                        saw_tool_calls = True
                        converted = _native_tool_calls_to_openai(msg["tool_calls"])
                        delta["tool_calls"] = [
                            {"index": i, **tc, "id": tool_call_ids.setdefault(i, tc["id"])}
                            for i, tc in enumerate(converted)
                        ]

                    finish_reason = None
                    if done:
                        finish_reason = "tool_calls" if saw_tool_calls else chunk.get("done_reason", "stop")

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

    async def warmup(self, spec: ModelSpec) -> None:
        """
        (E-05) 모델을 미리 로드해 첫 실요청 콜드스타트를 없앤다. 빈 프롬프트로 1토큰만
        생성하고 keep_alive로 상주시킨다. 실패는 조용히 무시(기동을 막지 않음).
        """
        payload: dict = {
            "model": spec.upstream,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_predict": 1},
        }
        if settings.ollama_keep_alive:
            payload["keep_alive"] = settings.ollama_keep_alive
        response = await self.client.post("/api/chat", json=payload)
        response.raise_for_status()

    async def aclose(self) -> None:
        await self.client.aclose()
        await self._img_client.aclose()
