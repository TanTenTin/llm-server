from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.models import ChatCompletionRequest
from app.registry import MODELS, RouteDecision, resolve
from app.service import (
    ProviderPool,
    aclose_quietly,
    chat_with_fallback,
    http_status_for,
    stream_with_fallback,
)


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
async def list_models() -> dict:
    """지원 모델 목록 반환 (레지스트리에서 자동 생성, OpenAI 호환)"""
    data = [
        {"id": name, "object": "model", "provider": spec.provider}
        for name, spec in MODELS.items()
    ]
    return {"object": "list", "data": data}


async def _streaming_response(
    request: ChatCompletionRequest, decision: RouteDecision, pool: ProviderPool
) -> StreamingResponse:
    """
    스트리밍 응답 구성. 첫 청크를 엔드포인트의 try/except 안에서 미리 당겨,
    '시작도 못 한' 실패는 일반 HTTP 500으로 변환한다. 첫 청크 이후의 오류는
    이미 응답이 시작됐으므로 스트림 중단으로 나타난다(스트리밍 fallback 한계).
    """
    gen = stream_with_fallback(request, decision, pool)
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
            # 클라이언트 조기 종료 등 어떤 경로로 끝나도 업스트림 스트림을 정리(누수 방지)
            await aclose_quietly(gen)

    return StreamingResponse(body(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
) -> dict | StreamingResponse:
    """OpenAI 호환 chat completions 엔드포인트"""
    pool: ProviderPool = app.state.pool
    decision = resolve(request.model)
    try:
        if request.stream:
            return await _streaming_response(request, decision, pool)
        return await chat_with_fallback(request, decision, pool)
    except Exception as e:
        # 업스트림/입력 오류의 원래 상태를 그대로 노출 (모두 500으로 뭉개지 않음)
        raise HTTPException(status_code=http_status_for(e), detail=str(e))
