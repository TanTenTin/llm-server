import secrets
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Awaitable, Callable, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

from app.adapters import (
    anthropic_to_chat_request,
    gemini_to_chat_request,
    openai_to_anthropic_response,
    openai_to_gemini_response,
    stream_openai_to_anthropic,
    stream_openai_to_gemini,
)
from app.config import settings
from app.models import ChatCompletionRequest
from app.realtime import RealtimeBridge
from app.registry import AUTO_ROUTE, MODELS, ModelSpec, RouteDecision, route
from app.service import (
    ProviderPool,
    RouteTrace,
    aclose_quietly,
    chat_with_fallback,
    error_detail,
    http_status_for,
    run_chat_fallback,
    run_stream_fallback,
    stream_with_fallback,
)


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    """
    게이트웨이 공유 토큰 검증. `GATEWAY_API_KEY`가 설정된 경우에만 강제한다.
    미설정(빈 값)이면 통과 — 내부/로컬 사용을 막지 않기 위함이나, 외부 노출 시엔
    반드시 키를 설정해야 한다(인증 없는 LLM 프록시 = Anthropic 과금/오남용 위험).
    타이밍 공격 방지를 위해 상수 시간 비교(compare_digest)를 사용한다.
    """
    if not settings.gateway_api_key:
        return
    expected = f"Bearer {settings.gateway_api_key}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def ws_authorized(websocket: WebSocket) -> bool:
    """
    WebSocket(/v1/realtime) 공유 토큰 검증. `GATEWAY_API_KEY` 미설정이면 통과.
    브라우저 WS는 커스텀 헤더를 못 붙이므로, `Authorization: Bearer` 헤더 또는
    `?api_key=` 쿼리 파라미터 둘 다 허용한다. 상수 시간 비교 사용.
    """
    if not settings.gateway_api_key:
        return True
    authorization = websocket.headers.get("authorization")
    if authorization and secrets.compare_digest(
        authorization, f"Bearer {settings.gateway_api_key}"
    ):
        return True
    token = websocket.query_params.get("api_key")
    if token and secrets.compare_digest(token, settings.gateway_api_key):
        return True
    return False


def _extract_native_token(http_request: Request) -> Optional[str]:
    """
    네이티브 SDK가 키를 싣는 여러 위치에서 게이트웨이 토큰 후보를 추출한다.
    - OpenAI/공통: `Authorization: Bearer <키>`
    - Anthropic SDK: `x-api-key: <키>`
    - Gemini(google-genai): `x-goog-api-key: <키>` 또는 `?key=<키>`
    """
    authorization = http_request.headers.get("authorization")
    if authorization and authorization.startswith("Bearer "):
        return authorization[len("Bearer ") :]
    x_api_key = http_request.headers.get("x-api-key")
    if x_api_key:
        return x_api_key
    x_goog = http_request.headers.get("x-goog-api-key")
    if x_goog:
        return x_goog
    return http_request.query_params.get("key")


def require_native_auth(http_request: Request) -> None:
    """
    네이티브 엔드포인트(/v1/messages, generateContent) 공유 토큰 검증.
    `GATEWAY_API_KEY` 미설정이면 통과. 설정 시 위 여러 헤더/쿼리 중 하나로 일치해야 한다.
    상수 시간 비교 사용.
    """
    if not settings.gateway_api_key:
        return
    token = _extract_native_token(http_request)
    if token is None or not secrets.compare_digest(token, settings.gateway_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 시작: provider 풀 생성 (client를 1회 만들어 재사용)
    app.state.pool = ProviderPool()
    yield
    # 종료: 보유한 client 정리
    await app.state.pool.aclose()


app = FastAPI(title="LLM Gateway", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(_auth: None = Depends(require_auth)) -> dict:
    """지원 모델 목록 반환 (레지스트리에서 자동 생성, OpenAI 호환)"""
    # auto는 실제 모델이 아니라 '요청 특성으로 게이트웨이가 고르는' 논리 라우트.
    data: list[dict] = [
        {"id": AUTO_ROUTE, "object": "model", "provider": "(auto)"}
    ]
    data += [
        {"id": name, "object": "model", "provider": spec.provider}
        for name, spec in MODELS.items()
    ]
    return {"object": "list", "data": data}


async def _streaming_response(
    gen: AsyncGenerator[str, None],
    trace: RouteTrace,
) -> StreamingResponse:
    """
    스트리밍 응답 구성. `gen`은 이미 출력 포맷(OpenAI/Anthropic/Gemini SSE)으로 완성된
    최종 제너레이터다. 첫 청크를 엔드포인트의 try/except 안에서 미리 당겨, '시작도 못 한'
    실패는 일반 HTTP 오류로 변환한다(네이티브 어댑터 제너레이터도 내부 stream_with_fallback의
    첫 청크를 당기면서 시작 에러가 그대로 전파된다). 첫 청크 이후의 오류는 이미 응답이
    시작됐으므로 스트림 중단으로 나타난다(스트리밍 fallback 한계).

    첫 청크를 당긴 시점에 trace가 채워지므로(실제 응답 provider 확정), 그 직후
    `x-llm-route` 헤더에 라우팅 결과를 실어 응답 시작 전에 함께 내려보낸다.
    """
    try:
        first = await gen.__anext__()
    except StopAsyncIteration:
        first = None

    async def body() -> AsyncGenerator[str, None]:
        try:
            if first is not None:
                yield first
            async for chunk in gen:
                yield chunk
        finally:
            # 클라이언트 조기 종료 등 어떤 경로로 끝나도 업스트림 스트림을 정리(누수 방지).
            # 네이티브 어댑터 제너레이터는 finally에서 내부 stream_with_fallback도 함께 닫는다.
            await aclose_quietly(gen)

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"x-llm-route": trace.header()},
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    _auth: None = Depends(require_auth),
) -> dict | StreamingResponse:
    """OpenAI 호환 chat completions 엔드포인트"""
    pool: ProviderPool = app.state.pool
    decision = route(request)  # auto면 요청 특성 기반 선택, 그 외엔 이름 기반
    # 실제로 어떤 모델이 응답했는지 관측용 트레이스. 응답 본문은 OpenAI 형식 그대로 두고
    # (호출 측 SDK 호환 유지), 라우팅 결과는 x-llm-route 헤더로만 노출한다.
    # decision.reason(auto:tier=... 등 선택 사유)을 헤더에 함께 싣는다.
    trace = RouteTrace(requested=request.model, reason=decision.reason)
    try:
        if request.stream:
            gen = stream_with_fallback(request, decision, pool, trace)
            return await _streaming_response(gen, trace)
        body = await chat_with_fallback(request, decision, pool, trace)
        return JSONResponse(content=body, headers={"x-llm-route": trace.header()})
    except Exception as e:
        # 업스트림/입력 오류의 원래 상태 + 본문(실제 사유)을 그대로 노출 (모두 500으로 뭉개지 않음)
        raise HTTPException(status_code=http_status_for(e), detail=error_detail(e))


# ─────────────────────────────────────────────────────────────
# 네이티브 포맷 엔드포인트 — 네이티브 패스스루(fast-path) + 폴백 어댑터 경로
#
# 후보가 '해당 네이티브 provider'면 클라이언트 원본 body를 그대로 업스트림에 보내고(이중 변환
# 없음 → provider 전용 필드·정확한 구조 보존), 그 외 후보(폴백 등)는 어댑터 경로(OpenAI 변환 후
# 네이티브 포맷으로 역변환)를 탄다. 회로차단기·재시도·트레이스는 공통 fallback 루프가 동일 적용.
# Anthropic·Gemini가 이 구조를 공유하므로, provider별 차이(네이티브 호출 메서드/출력 어댑터)만
# 콜백으로 주입받는 제너릭 헬퍼로 묶는다.
# ─────────────────────────────────────────────────────────────
async def _run_native_passthrough(
    request: ChatCompletionRequest,
    body: dict,
    native_provider: str,
    native_chat: Callable[[object, dict, ModelSpec], Awaitable[dict]],
    native_stream: Callable[[object, dict, ModelSpec], AsyncGenerator[str, None]],
    to_response: Callable[[dict], dict],
    to_stream: Callable[[AsyncGenerator[str, None]], AsyncGenerator[str, None]],
) -> JSONResponse | StreamingResponse:
    """
    네이티브 엔드포인트 공통 실행 경로.
    `native_provider`: 패스스루를 적용할 provider 이름("anthropic" | "gemini").
    `native_chat`/`native_stream`: 그 provider의 네이티브 호출(원본 body 그대로 전달).
    `to_response`/`to_stream`: 폴백(비-native provider) 결과(OpenAI)를 네이티브로 역변환.
    """
    pool: ProviderPool = app.state.pool
    decision = route(request)
    trace = RouteTrace(requested=request.model, reason=decision.reason)

    async def invoke(spec: ModelSpec) -> dict:
        provider = pool.get(spec.provider)  # ProviderUnavailable 가능 → 다음 후보로
        if spec.provider == native_provider:
            return await native_chat(provider, body, spec)   # 네이티브 패스스루
        return to_response(await provider.chat(request, spec))  # 폴백: 어댑터 경로

    def open_stream(spec: ModelSpec) -> AsyncGenerator[str, None]:
        provider = pool.get(spec.provider)
        if spec.provider == native_provider:
            return native_stream(provider, body, spec)       # 네이티브 SSE 그대로
        return to_stream(provider.stream(request, spec))     # 폴백: 어댑터 변환

    try:
        if request.stream:
            gen = run_stream_fallback(decision, pool, trace, open_stream)
            return await _streaming_response(gen, trace)
        body_out = await run_chat_fallback(decision, pool, trace, invoke)
        return JSONResponse(content=body_out, headers={"x-llm-route": trace.header()})
    except Exception as e:
        raise HTTPException(status_code=http_status_for(e), detail=error_detail(e))


@app.post("/v1/messages", response_model=None)
async def anthropic_messages(http_request: Request) -> JSONResponse | StreamingResponse:
    """
    Anthropic Messages API 호환 엔드포인트. 요청/응답은 Anthropic 포맷이지만 model 필드가
    라우팅을 결정하므로 Gemini·Ollama로도 라우팅/폴백된다(순수 포맷 어댑터).
    인증: GATEWAY_API_KEY 설정 시 `x-api-key` 또는 `Authorization: Bearer` 필요.

    **Anthropic 패스스루 fast-path**: 후보가 Anthropic provider면 원본 Anthropic body를
    그대로 SDK로 보낸다(OpenAI 이중 변환 없음 → cache_control·멀티턴/스트리밍 tool_use 보존).
    그 외 provider(폴백 등)는 어댑터 경로를 탄다.
    """
    require_native_auth(http_request)
    body = await http_request.json()
    try:
        request = anthropic_to_chat_request(body)  # 라우팅 판단용 내부표준(폴백 후보가 사용)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = request.model
    return await _run_native_passthrough(
        request,
        body,
        native_provider="anthropic",
        native_chat=lambda provider, b, spec: provider.chat_native(b, spec),
        native_stream=lambda provider, b, spec: provider.stream_native(b, spec),
        to_response=lambda openai_body: openai_to_anthropic_response(openai_body, model),
        to_stream=lambda inner: stream_openai_to_anthropic(inner, model),
    )


async def _gemini_generate(
    model_action: str, http_request: Request
) -> JSONResponse | StreamingResponse:
    """
    Gemini generateContent / streamGenerateContent 공통 핸들러.
    경로 `{model}:{action}` 에서 모델과 액션을 분리한다(예: 'gemini-2.5-flash:generateContent').
    인증: GATEWAY_API_KEY 설정 시 `x-goog-api-key`/`?key=`/`Authorization: Bearer` 중 하나.

    **Gemini 패스스루 fast-path**: 후보가 Gemini provider면 원본 Gemini body를 네이티브
    generateContent 엔드포인트로 그대로 보낸다(OpenAI-compat 이중 변환 없음 → safetySettings·
    thinkingConfig·cachedContent 등 Gemini 전용 필드와 네이티브 응답 구조 보존). 폴백 후보(예:
    Ollama)는 어댑터 경로(OpenAI 변환 후 Gemini 응답으로 역변환)를 탄다.
    """
    require_native_auth(http_request)
    model, _, action = model_action.rpartition(":")
    if not model or action not in ("generateContent", "streamGenerateContent"):
        raise HTTPException(status_code=404, detail="Unknown Gemini endpoint")

    body = await http_request.json()
    stream = action == "streamGenerateContent"
    try:
        request = gemini_to_chat_request(body, model, stream)  # 라우팅 판단용 내부표준
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return await _run_native_passthrough(
        request,
        body,
        native_provider="gemini",
        native_chat=lambda provider, b, spec: provider.generate_native(b, spec),
        native_stream=lambda provider, b, spec: provider.stream_native(b, spec),
        to_response=lambda openai_body: openai_to_gemini_response(openai_body, model),
        to_stream=lambda inner: stream_openai_to_gemini(inner, model),
    )


# google-genai SDK는 기본 v1beta, 일부 클라이언트는 v1 경로를 쓴다 → 둘 다 받는다.
# 경로 끝이 `{model}:{action}` 형태라 `:path` 컨버터로 콜론 포함 전체를 캡처한다.
@app.post("/v1beta/models/{model_action:path}", response_model=None)
async def gemini_generate_v1beta(
    model_action: str, http_request: Request
) -> JSONResponse | StreamingResponse:
    return await _gemini_generate(model_action, http_request)


@app.post("/v1/models/{model_action:path}", response_model=None)
async def gemini_generate_v1(
    model_action: str, http_request: Request
) -> JSONResponse | StreamingResponse:
    return await _gemini_generate(model_action, http_request)


@app.websocket("/v1/realtime")
async def realtime(websocket: WebSocket) -> None:
    """
    OpenAI Realtime API 호환 음성 엔드포인트. 내부에서 Gemini Live로 양방향 중계한다.
    - 모델: `?model=` 쿼리로 지정(미지정 시 settings.realtime_default_model).
    - 인증: GATEWAY_API_KEY 설정 시 Authorization 헤더 또는 `?api_key=` 필요.
    텍스트 경로(ProviderPool/fallback)와 독립 — Live는 로컬 대체가 없어 폴백하지 않는다.
    """
    if not ws_authorized(websocket):
        await websocket.close(code=4401)  # 4401: 애플리케이션 정의 Unauthorized
        return

    bridge = RealtimeBridge(
        api_key=settings.google_ai_api_key,
        default_model=settings.realtime_default_model,
        requested_model=websocket.query_params.get("model"),
        client_input_rate=settings.realtime_input_sample_rate,
    )
    await bridge.run(websocket)
