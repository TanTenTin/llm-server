import hashlib
import secrets
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Awaitable, Callable, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app.dashboard import DASHBOARD_HTML
from app.ratelimit import RateLimiter

from app.cache import ResponseCache, cache_key_for
from app.adapters import (
    anthropic_to_chat_request,
    gemini_to_chat_request,
    openai_to_anthropic_response,
    openai_to_gemini_response,
    stream_openai_to_anthropic,
    stream_openai_to_gemini,
)
from app.config import settings
from app.models import ChatCompletionRequest, EmbeddingsRequest
from app.providers.ollama import OllamaProvider
from app.realtime import RealtimeBackend, RealtimeBridge
from app.realtime_local import LocalRealtimeBridge
from app.registry import (
    AUTO_CANDIDATES_BY_TIER,
    AUTO_ROUTE,
    DEFAULT_MODEL,
    EMBEDDING_MODELS,
    MODELS,
    ModelSpec,
    RouteDecision,
    ollama_context_window,
    resolve_embedding,
    resolve_live,
    route,
    update_ollama_capabilities,
)
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


# 분당 요청 제한기 (RATE_LIMIT_RPM=0 이면 비활성). 단일 프로세스 인메모리.
_rate_limiter = RateLimiter(settings.rate_limit_rpm)


def _gateway_keys() -> list[str]:
    """
    허용 게이트웨이 키 목록. `GATEWAY_API_KEY`는 쉼표 구분 복수 키를 지원한다
    (클라이언트별 키 발급/회수 용도 — 예: "key-agent1, key-agent2").
    빈 값이면 인증 개방(내부/로컬 사용).
    """
    return [k.strip() for k in settings.gateway_api_key.split(",") if k.strip()]


def _token_matches(token: str, keys: list[str]) -> bool:
    """상수 시간 비교(compare_digest)로 키 목록과 대조. 매칭돼도 전체를 순회한다."""
    matched = False
    for key in keys:
        if secrets.compare_digest(token, key):
            matched = True
    return matched


def _client_ip(headers, client_host: Optional[str]) -> str:
    """
    (E-14) 레이트리밋 식별용 클라이언트 IP를 고른다. TRUST_PROXY_FORWARDED_FOR가 켜져 있으면
    X-Forwarded-For의 첫(=원 클라이언트) IP를 쓴다 — 리버스 프록시(Caddy/nginx) 뒤에서 모든
    클라이언트가 프록시 IP 하나로 묶여 전원 스로틀되는 문제를 막는다. 꺼져 있으면 소켓 IP.
    (XFF는 신뢰된 프록시 배치에서만 켤 것 — 직접 노출 시 헤더 스푸핑 가능.)
    """
    if settings.trust_proxy_forwarded_for and headers is not None:
        xff = headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return client_host or "unknown"


def _rate_limit_identity(token: Optional[str], headers, client_host: Optional[str]) -> str:
    """토큰이 있으면 그 해시, 없으면 클라이언트 IP를 레이트리밋 키로 쓴다."""
    if token:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    return _client_ip(headers, client_host)


def _enforce_rate_limit(http_request: Request, token: Optional[str]) -> None:
    """
    키(해시) 또는 클라이언트 IP 단위 분당 요청 제한. 초과 시 429 + Retry-After.
    인증을 통과한 뒤에 호출된다(무인증 401 요청으로 남의 윈도우를 소모하지 못하게).
    """
    if not _rate_limiter.enabled:
        return
    client_host = http_request.client.host if http_request.client else None
    identity = _rate_limit_identity(token, http_request.headers, client_host)
    if not _rate_limiter.allow(identity):
        raise HTTPException(
            status_code=429,
            detail="게이트웨이 레이트리밋 초과 (RATE_LIMIT_RPM)",
            headers={"Retry-After": str(_rate_limiter.seconds_until_reset())},
        )


def require_auth(
    http_request: Request, authorization: Optional[str] = Header(default=None)
) -> None:
    """
    게이트웨이 공유 토큰 검증 + 레이트리밋. `GATEWAY_API_KEY`가 설정된 경우에만
    인증을 강제한다(쉼표 구분 복수 키 허용). 미설정(빈 값)이면 인증은 통과하되
    레이트리밋은 IP 단위로 여전히 적용된다. 외부 노출 시엔 반드시 키를 설정할 것.
    """
    keys = _gateway_keys()
    token: Optional[str] = None
    if keys:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")
        token = authorization[len("Bearer ") :]
        if not _token_matches(token, keys):
            raise HTTPException(status_code=401, detail="Unauthorized")
    _enforce_rate_limit(http_request, token)


def ws_authorized(websocket: WebSocket) -> bool:
    """
    WebSocket(/v1/realtime) 공유 토큰 검증. 키 미설정이면 통과(복수 키 허용).
    브라우저 WS는 커스텀 헤더를 못 붙이므로, `Authorization: Bearer` 헤더 또는
    `?api_key=` 쿼리 파라미터 둘 다 허용한다. 상수 시간 비교 사용.
    (레이트리밋은 요청 단위 개념이라 지속 연결인 WS에는 적용하지 않는다.)
    """
    keys = _gateway_keys()
    if not keys:
        return True
    authorization = websocket.headers.get("authorization")
    if authorization and authorization.startswith("Bearer ") and _token_matches(
        authorization[len("Bearer ") :], keys
    ):
        return True
    token = websocket.query_params.get("api_key")
    if token and _token_matches(token, keys):
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
    네이티브 엔드포인트(/v1/messages, generateContent) 공유 토큰 검증 + 레이트리밋.
    `GATEWAY_API_KEY` 미설정이면 인증 통과(레이트리밋은 IP 단위 적용). 설정 시
    위 여러 헤더/쿼리 중 하나로 일치해야 한다(쉼표 구분 복수 키 허용, 상수 시간 비교).
    """
    keys = _gateway_keys()
    token = _extract_native_token(http_request)
    if keys and (token is None or not _token_matches(token, keys)):
        raise HTTPException(status_code=401, detail="Unauthorized")
    _enforce_rate_limit(http_request, token if keys else None)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 시작: provider 풀 생성 (client를 1회 만들어 재사용)
    app.state.pool = ProviderPool()
    # 응답 캐시 (exact-match TTL — 무료 티어 쿼터 절약. TTL 0이면 비활성)
    app.state.cache = ResponseCache(settings.cache_ttl_seconds)
    # (E-03) Ollama capability 캐시 워밍업 — 패스스루 라우팅이 실제 모델 특성(tools/vision)을
    # 반영하도록 시작 시 1회 조회한다. 실패해도 기동엔 지장 없고(빈 캐시=보수적 기본),
    # 이후 /v1/models 조회 때마다 최신으로 갱신된다.
    try:
        provider = app.state.pool.get("ollama")
        if isinstance(provider, OllamaProvider):
            update_ollama_capabilities(await provider.list_models())
            # (E-05) 기본 모델이 로컬이면 예열 — 첫 실요청 콜드스타트 제거(설정 켜졌을 때만).
            if settings.ollama_warmup:
                default_spec = MODELS.get(DEFAULT_MODEL)
                if default_spec is not None and default_spec.provider == "ollama":
                    await provider.warmup(default_spec)
    except Exception:
        pass
    yield
    # 종료: 보유한 client 정리
    await app.state.pool.aclose()


app = FastAPI(title="LLM Gateway", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """가벼운 생존 확인 (docker healthcheck 등 무인증 프로브용) — 항상 ok."""
    return {"status": "ok"}


# 게이트웨이가 알 수 있는 provider 전체 집합. 등록 여부와 무관하게 상태를 보여줘
# "키 미설정으로 미등록"을 침묵 아닌 명시적 false로 드러낸다.
_KNOWN_PROVIDERS = ("gemini", "anthropic", "ollama")


@app.get("/health/providers")
async def health_providers(_auth: None = Depends(require_auth)) -> dict:
    """
    심층 헬스체크 — "왜 자꾸 로컬로 폴백되지?"를 로그 없이 진단하기 위한 상태 노출.
      - registered: provider 등록 여부 (false면 API 키 미설정 → ProviderUnavailable 폴백)
      - breaker_open / breaker_remaining_seconds: 회로차단기 상태 (일시 장애로 뒤로 밀림)
      - reachable(Ollama만): /api/version 실측 프로브 — 로컬 폴백의 최후 보루가
        실제로 살아있는지. 클라우드 provider는 매 요청이 곧 프로브라 별도 확인 안 함.
    키 설정 여부가 드러나므로 게이트웨이 인증을 요구한다.
    """
    pool: ProviderPool = app.state.pool
    registered = set(pool.registered())
    breaker_status = pool.breaker.status()

    providers: dict[str, dict] = {}
    for name in _KNOWN_PROVIDERS:
        info: dict = {"registered": name in registered}
        if name in registered:
            info["breaker_open"] = name in breaker_status
            if name in breaker_status:
                info["breaker_remaining_seconds"] = breaker_status[name]
        providers[name] = info

    if "ollama" in registered:
        try:
            resp = await pool.get("ollama").client.get("/api/version", timeout=2.0)
            providers["ollama"]["reachable"] = resp.status_code == 200
        except Exception:
            providers["ollama"]["reachable"] = False

    return {"status": "ok", "providers": providers}


@app.get("/health/ollama")
async def health_ollama(_auth: None = Depends(require_auth)) -> dict:
    """
    LLM 백엔드 서버(Ollama) 자체의 상세 상태 — 대시보드용. 게이트웨이가 프록시하는 그
    별도 LLM 서버(OLLAMA_BASE_URL, 이 PC가 아니라 운영에선 Oracle)의 버전·설치된 모델
    (크기/파라미터/양자화)·현재 메모리에 로드된 모델(size_vram/만료)을 실측한다.
    Ollama 미등록(항상 등록되긴 하나 방어적)·미가용 시 reachable=False로 degrade.
    """
    pool: ProviderPool = app.state.pool
    provider = pool.get("ollama") if "ollama" in pool.registered() else None
    if not isinstance(provider, OllamaProvider):
        return {"reachable": False, "base_url": None, "version": None,
                "installed": [], "loaded": []}
    return await provider.server_status()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    """
    관측용 임시 대시보드(self-contained HTML). 셸 자체엔 비밀이 없어 무인증으로 서빙하고,
    페이지가 브라우저에서 동일 출처로 /health/providers·/v1/usage·/health/ollama를 폴링한다
    (동일 출처라 CORS·CSP 무관). 게이트웨이 키가 설정돼 있으면 페이지에서 입력한 키를
    Authorization 헤더로 붙인다. 인메모리 집계라 서버 재기동 시 통계는 리셋된다.
    """
    return DASHBOARD_HTML


@app.get("/v1/usage")
async def usage(_auth: None = Depends(require_auth)) -> dict:
    """
    모델별(provider:model × UTC 일 단위) 사용량 스냅샷 — 요청 수·토큰·에러(429 등)·
    폴백 응답 횟수. 무료 티어 쿼터를 얼마나 썼는지 게이트웨이 스스로 관측하기 위한
    인메모리 집계(재기동 시 리셋, 최근 7일 보존). 스트리밍은 요청 수만 집계된다.
    """
    pool: ProviderPool = app.state.pool
    return pool.usage.snapshot()


async def _ollama_models(pool: ProviderPool) -> list[dict]:
    """
    Ollama 서버에 실제 설치된 모델을 /api/tags로 실시간 조회해 model 항목으로 변환한다.
    로컬 모델은 운영 중 pull/rm으로 자주 바뀌므로 정적 레지스트리 대신 실측한다.
    id는 'ollama/<태그>' 형태라 그대로 model 파라미터로 호출 가능(패스스루 라우팅).
    임베딩 여부는 Ollama가 주는 capabilities(["embedding"])로 판별하고, 구버전이라
    capabilities가 없으면 이름 휴리스틱("embed" 포함)으로 보완한다.
    조회 실패(서버 다운 등) 시 레지스트리의 정적 ollama 항목으로 graceful degrade 한다.
    """
    provider = pool.get("ollama")   # ollama는 항상 등록되어 있음(로컬 기본 provider)
    try:
        models = await provider.list_models() if isinstance(provider, OllamaProvider) else []
    except Exception:
        # 서버 미가용 등 — 목록이 비지 않도록 레지스트리의 정적 ollama 항목으로 대체.
        # chat 모델만 컨텍스트 메타를 싣는다(EMBEDDING_MODELS엔 의미 없는 값).
        return [
            {"id": name, "object": "model", "provider": "ollama", "source": "registry",
             **({} if name in EMBEDDING_MODELS else {
                 "context_length": spec.context_window,
                 "max_output_tokens": spec.max_output_tokens,
             })}
            for name, spec in {**MODELS, **EMBEDDING_MODELS}.items()
            if spec.provider == "ollama"
        ]

    # (E-03) 실시간 조회 결과로 라우팅용 capability 캐시를 갱신 — pull/rm으로 로컬 모델이
    # 바뀌면 패스스루 라우팅의 tools/vision 판단도 재기동 없이 따라간다.
    update_ollama_capabilities(models)

    entries: list[dict] = []
    for model in models:
        tag = model["name"]
        capabilities = model.get("capabilities") or []
        entry: dict = {
            "id": f"ollama/{tag}", "object": "model",
            "provider": "ollama", "source": "ollama",
        }
        # capabilities 우선, 없으면 이름 기반 추정으로 embedding 모델 표시.
        if "embedding" in capabilities or ("embed" in tag.lower() and not capabilities):
            entry["type"] = "embedding"
        else:
            # 로컬 chat 모델의 창은 런타임 num_ctx와 한 소스(registry._OLLAMA_CONTEXT_WINDOW).
            # 입력·출력이 한 창을 나눠 쓰므로 출력 상한은 보수적으로 잡는다.
            entry["context_length"] = ollama_context_window()
            entry["max_output_tokens"] = 4096
        # /api/tags가 준 capabilities(["tools","vision","thinking",...])를 그대로 노출한다.
        # 커스텀 provider 모델은 models.dev 같은 공개 레지스트리에 없어, 클라이언트
        # (opencode 등)가 도구 지원 여부를 알 길이 없다 — 설정에 tool_call을 손으로
        # 적는 대신 이 필드로 판단할 수 있게 한다. 구버전 Ollama면 필드 자체를 생략.
        if capabilities:
            entry["capabilities"] = capabilities
        entries.append(entry)
    return entries


def _auto_context_length(local_only: bool) -> tuple[int, int]:
    """
    'auto' 논리 라우트가 실제로 보장하는 (컨텍스트 창, 출력 상한).

    auto의 후보는 티어별로 다르므로 '가장 큰 창'을 보고한다 — 클라이언트가 대화를
    얼마나 키워도 되는지 판단하는 상한이기 때문이다. 단 local_only 요청은 SaaS 후보가
    체인에서 걷혀 로컬 창(32k)이 실제 천장이므로, 1M을 보고하면 클라이언트가 압축을
    미루다 413을 맞는다. 헤더를 보고 정직한 값을 돌려준다.
    """
    names = {n for tier in AUTO_CANDIDATES_BY_TIER.values() for n in tier}
    specs = [MODELS[n] for n in names]
    if local_only:
        specs = [s for s in specs if s.provider == "ollama"] or [MODELS[DEFAULT_MODEL]]
    best = max(specs, key=lambda s: s.context_window)
    return best.context_window, best.max_output_tokens


@app.get("/v1/models")
async def list_models(
    _auth: None = Depends(require_auth),
    x_llm_local_only: Optional[str] = Header(default=None),
) -> dict:
    """
    지원 모델 목록 반환 (OpenAI 호환).

    SaaS provider(gemini·anthropic)는 레지스트리(MODELS/EMBEDDING_MODELS)에서 정적으로
    나열하고, Ollama는 서버에 실제 설치된 모델을 /api/tags로 실시간 조회해 유동적으로
    나열한다. 항목의 `source`("registry" | "ollama")로 출처를 구분할 수 있다.

    각 chat 항목은 `context_length`·`max_output_tokens`를 함께 싣는다. 클라이언트가 대화
    압축 시점을 정하는 근거이며, 로컬 모델은 OLLAMA_NUM_CTX를 그대로 반영하므로 설정을
    바꾸면 클라이언트 쪽 한계도 자동으로 따라간다(양쪽 하드코딩으로 인한 어긋남 제거).
    """
    pool: ProviderPool = app.state.pool
    local_only = is_local_only(x_llm_local_only)

    # auto는 실제 모델이 아니라 '요청 특성으로 게이트웨이가 고르는' 논리 라우트.
    auto_context, auto_output = _auto_context_length(local_only)
    data: list[dict] = [
        {"id": AUTO_ROUTE, "object": "model", "provider": "(auto)",
         "context_length": auto_context, "max_output_tokens": auto_output}
    ]
    # ── SaaS(정적): ollama 이외 provider만 레지스트리에서 나열 ──
    data += [
        {"id": name, "object": "model", "provider": spec.provider, "source": "registry",
         "context_length": spec.context_window, "max_output_tokens": spec.max_output_tokens}
        for name, spec in MODELS.items()
        if spec.provider != "ollama"
    ]
    data += [
        {"id": name, "object": "model", "provider": spec.provider,
         "type": "embedding", "source": "registry"}
        for name, spec in EMBEDDING_MODELS.items()
        if spec.provider != "ollama"
    ]
    # ── Ollama(유동): 서버에 실제 설치된 모델을 실시간 조회 ──
    data += await _ollama_models(pool)

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


def is_local_only(header_value: Optional[str]) -> bool:
    """
    x-llm-local-only 헤더 해석. 참이면 SaaS provider(Gemini·Anthropic)를 라우팅 체인에서
    배제하고 로컬 Ollama로만 추론한다 — 로컬이 죽어도 클라우드로 넘기지 않고 실패시킨다.
    프롬프트/코드를 외부로 내보내면 안 되는 호출자(로컬 모델 전용 에이전트 등)를 위한 것.
    """
    if header_value is None:
        return False
    return header_value.strip().lower() in ("1", "true", "yes", "on")


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    _auth: None = Depends(require_auth),
    x_llm_local_only: Optional[str] = Header(default=None),
) -> dict | StreamingResponse:
    """OpenAI 호환 chat completions 엔드포인트"""
    pool: ProviderPool = app.state.pool
    cache: ResponseCache = app.state.cache
    local_only = is_local_only(x_llm_local_only)

    # 응답 캐시 — 비스트리밍 + temperature 미지정/0 인 동일 요청만 대상(cache_key_for).
    # 히트 시 업스트림 무호출 → 무료 티어 쿼터 절약. 절약분은 'cache' 라벨로 /v1/usage에 집계.
    cache_key = cache_key_for(request) if cache.enabled else None
    # 로컬 전용 요청은 캐시 공간을 분리한다. 같은 body라도 일반 요청은 Gemini가 만든 응답을
    # 캐시에 남길 수 있는데, 그걸 로컬 전용 호출자에게 돌려주면 "로컬만 쓴다"는 보장이 깨진다.
    if cache_key and local_only:
        cache_key = f"{cache_key}:local-only"
    if cache_key:
        cached = cache.get(cache_key)
        if cached is not None:
            pool.usage.record_success("cache", cached, False)
            return JSONResponse(content=cached, headers={
                "x-llm-route": f"requested={request.model}; served=cache",
                "x-llm-cache": "hit",
            })

    decision = route(request, local_only=local_only)  # auto면 요청 특성 기반 선택, 그 외엔 이름 기반
    # 실제로 어떤 모델이 응답했는지 관측용 트레이스. 응답 본문은 OpenAI 형식 그대로 두고
    # (호출 측 SDK 호환 유지), 라우팅 결과는 x-llm-route 헤더로만 노출한다.
    # decision.reason(auto:tier=... 등 선택 사유)을 헤더에 함께 싣는다.
    trace = RouteTrace(requested=request.model, reason=decision.reason)
    leader = True
    try:
        if request.stream:
            gen = stream_with_fallback(request, decision, pool, trace)
            return await _streaming_response(gen, trace)
        # (E-15) single-flight — 동일 요청이 이미 계산 중이면 그 결과를 공유해 업스트림
        # 중복 호출(스탬피드)을 막는다. 캐시 미스 상태에서만(cache_key 존재) 동작한다.
        if cache_key:
            leader, inflight = cache.begin(cache_key)
            if not leader:
                shared = await inflight   # 진행 중인 leader의 결과를 기다린다
                return JSONResponse(content=shared, headers={
                    "x-llm-route": f"requested={request.model}; served=cache",
                    "x-llm-cache": "hit-inflight",
                })
        body = await chat_with_fallback(request, decision, pool, trace)
        headers = {"x-llm-route": trace.header()}
        if cache_key:
            cache.put(cache_key, body)
            cache.settle(cache_key, result=body)  # follower들을 결과로 깨운다
            headers["x-llm-cache"] = "miss"
        return JSONResponse(content=body, headers=headers)
    except Exception as e:
        # leader가 실패하면 대기 중인 follower들도 같은 예외로 깨운다(무한 대기 방지).
        if cache_key and leader:
            cache.settle(cache_key, exc=e)
        # 업스트림/입력 오류의 원래 상태 + 본문(실제 사유)을 그대로 노출 (모두 500으로 뭉개지 않음)
        raise HTTPException(status_code=http_status_for(e), detail=error_detail(e))


@app.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingsRequest,
    _auth: None = Depends(require_auth),
) -> JSONResponse:
    """
    OpenAI 호환 embeddings 엔드포인트 (RAG 에이전트용). Gemini embedding(무료) 우선,
    장애·키 미설정 시 로컬 Ollama로 폴백 — chat과 동일한 fallback 루프(회로차단기·
    x-llm-route·사용량 집계 포함)를 재사용한다. Anthropic은 임베딩 API가 없어 제외.
    OpenAI SDK 기본 모델명(text-embedding-3-*)은 별칭으로 기본 모델에 매핑된다.
    """
    pool: ProviderPool = app.state.pool
    decision = resolve_embedding(request.model)
    trace = RouteTrace(requested=request.model, reason=decision.reason)

    async def invoke(spec: ModelSpec) -> dict:
        return await pool.get(spec.provider).embed(request, spec)

    try:
        body = await run_chat_fallback(decision, pool, trace, invoke)
        return JSONResponse(content=body, headers={"x-llm-route": trace.header()})
    except Exception as e:
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
    OpenAI Realtime API 호환 음성 엔드포인트. `?model=`로 백엔드가 갈린다(텍스트 경로가
    model 필드로 로컬/클라우드를 가르는 것과 동일). 클라이언트는 프로토콜만 보므로 내부가
    Gemini인지 로컬인지 몰라도 된다.
      - model=gemini-live(기본) → Gemini Live 중계(클라우드).
      - model=local-live       → 완전 로컬(VAD→STT→Ollama→TTS).
    인증: GATEWAY_API_KEY 설정 시 Authorization 헤더 또는 `?api_key=` 필요.
    `x-llm-local-only: 1` 헤더를 실으면 클라우드 Live 지정을 거부한다(텍스트와 대칭).
    """
    if not ws_authorized(websocket):
        await websocket.close(code=4401)  # 4401: 애플리케이션 정의 Unauthorized
        return

    # (E-14) WS도 연결 단위로 레이트리밋 — 다수 동시 WS로 Gemini Live를 무제한 중계하며
    # usage/budget/ratelimit을 우회하던 구멍을 막는다(연결 1건 = 요청 1건으로 집계).
    if _rate_limiter.enabled:
        keys = _gateway_keys()
        ws_token: Optional[str] = None
        if keys:
            authorization = websocket.headers.get("authorization")
            if authorization and authorization.startswith("Bearer "):
                ws_token = authorization[len("Bearer ") :]
            else:
                ws_token = websocket.query_params.get("api_key")
        client_host = websocket.client.host if websocket.client else None
        identity = _rate_limit_identity(ws_token, websocket.headers, client_host)
        if not _rate_limiter.allow(identity):
            await websocket.close(code=4429)  # 4429: 애플리케이션 정의 Too Many Requests
            return

    # model → 백엔드 결정. local_only인데 클라우드 Live를 콕 집으면 거부(조용한 바꿔치기 금지).
    local_only = is_local_only(websocket.headers.get("x-llm-local-only"))
    try:
        spec = resolve_live(
            websocket.query_params.get("model"),
            settings.realtime_default_model,
            local_only,
        )
    except ValueError as e:
        await websocket.accept()   # send 전에 핸드셰이크를 완료해야 에러 이벤트를 보낼 수 있다
        await websocket.send_json({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": str(e)},
        })
        await websocket.close()
        return

    bridge: RealtimeBackend
    if spec.provider == "local":
        bridge = LocalRealtimeBridge(
            spec,
            pool=websocket.app.state.pool,   # Ollama 스트림 재사용
            client_input_rate=settings.realtime_input_sample_rate,
        )
    else:
        bridge = RealtimeBridge(
            api_key=settings.google_ai_api_key,
            default_model=spec.upstream or settings.realtime_default_model,
            requested_model=None,            # 이미 spec으로 해석 완료
            client_input_rate=settings.realtime_input_sample_rate,
        )
    await bridge.run(websocket)
