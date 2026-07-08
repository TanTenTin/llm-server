import json
from dataclasses import dataclass, field

from app.config import settings
from app.models import ChatCompletionRequest


# ─────────────────────────────────────────────────────────────
# 데이터 모델
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    """
    하나의 모델을 어떤 provider로, 어떤 upstream 이름으로 호출할지에 대한 정의.
    라우팅·모델명 변환의 단일 진실 공급원(SSOT).
    """
    provider: str                              # "ollama" | "anthropic" | "gemini"
    upstream: str                              # provider에 실제로 보낼 모델명
    max_tokens: int | None = None              # 기본 max_tokens (요청에 없을 때 사용)
    fallback: list[str] = field(default_factory=list)  # 실패 시 시도할 다른 모델 키
    # ── 비용/특성 메타 (Phase 1: 로그·헤더 표시용 / Phase 2: 자동선택 판단 근거) ──
    cost_tier: str = "local"                   # "local"(자가호스팅) | "free-cloud"(무료 티어) | "paid"(과금)
    is_free: bool = True                       # 한계비용 0 여부 (무료 티어·로컬=True, 과금 API=False)
    # ── capability 메타 (Phase 2: auto 라우트의 후보 필터 근거) ──
    supports_tools: bool = True                # function calling(도구) 지원 여부
    context_window: int = 32_000               # 최대 입력 컨텍스트(토큰). 보수적 기본값
    # 이미지 입력(vision) 지원 여부. 보수적 기본 False — 미지원 모델에 이미지가 가면
    # 조용히 무시되거나 400이 나므로, 확실한 모델만 True로 지정한다(Phase 5).
    supports_vision: bool = False


@dataclass(frozen=True)
class RouteDecision:
    """resolve()/route() 결과. 시도 순서대로 정렬된 spec 체인 (primary + fallback들)."""
    chain: list[ModelSpec]
    # 선택 사유(관측용). auto 라우트면 "auto:tier=complex" 등, 그 외엔 None.
    reason: str | None = None


# ─────────────────────────────────────────────────────────────
# Ollama 런타임 메타 (컨텍스트·capability) — 라우터가 실제 로컬 동작에 맞추기 위한 단일 소스
# ─────────────────────────────────────────────────────────────
# (E-02) 로컬 컨텍스트 창은 런타임에 주입되는 num_ctx(settings)와 '한 소스'로 묶는다.
# 예전엔 registry에 32_000을 하드코딩해 런타임 num_ctx=16384와 어긋나(라우터는 25.6k까지
# 로컬 OK로 판단, 실제 창은 16384) 그 사이 입력이 조용히 잘렸다. 이제 registry의 ollama
# context_window가 이 값을 따르므로, num_ctx를 어떤 값으로 두든 라우터 판단이 실제와 일치한다.
# num_ctx 미지정(≤0=서버 기본 ~2~4k)이면 보수적으로 8192로 잡는다.
_OLLAMA_CONTEXT_WINDOW = settings.ollama_num_ctx if settings.ollama_num_ctx > 0 else 8192

# (E-03) Ollama /api/tags capabilities 캐시(태그 → ["tools","vision","embedding",...]).
# 패스스루 ollama 모델의 supports_tools/supports_vision를 실제 모델에 맞추기 위한 근거.
# lifespan 시작 시 1회 워밍업하고 /v1/models 조회 때마다 갱신한다(update_ollama_capabilities).
# 비어 있으면(미조회/구버전 Ollama) 기존 보수적 기본값으로 폴백해 회귀가 없다.
_OLLAMA_CAPABILITIES: dict[str, list[str]] = {}


# ─────────────────────────────────────────────────────────────
# 레지스트리 (코드 내 관리 — 모델 추가/수정 시 여기만 손대면 됨)
# ─────────────────────────────────────────────────────────────
MODELS: dict[str, ModelSpec] = {
    # ── Gemini (기본 provider) ───────────────────────────────────
    "gemini-2.5-flash": ModelSpec(
        provider="gemini",
        upstream="gemini-2.5-flash",
        fallback=["ollama/qwen3:14b"],      # 키 미설정 또는 장애 시 로컬로 폴백
        cost_tier="free-cloud",
        is_free=True,
        supports_tools=True,
        context_window=1_000_000,
        supports_vision=True,
    ),
    "gemini-2.5-flash-lite": ModelSpec(
        provider="gemini",
        upstream="gemini-2.5-flash-lite",
        fallback=["ollama/qwen3:14b"],
        cost_tier="free-cloud",
        is_free=True,
        supports_tools=True,
        context_window=1_000_000,
        supports_vision=True,
    ),
    # ── Anthropic ───────────────────────────────────────────────
    "claude-sonnet-4-6": ModelSpec(
        provider="anthropic",
        upstream="claude-sonnet-4-6",
        max_tokens=8192,
        fallback=["ollama/qwen3:14b"],
        cost_tier="paid",
        is_free=False,
        supports_tools=True,
        context_window=200_000,
        supports_vision=True,
    ),
    "claude-opus-4-7": ModelSpec(
        provider="anthropic",
        upstream="claude-opus-4-7",
        max_tokens=8192,
        fallback=["ollama/qwen3:14b"],
        cost_tier="paid",
        is_free=False,
        supports_tools=True,
        context_window=200_000,
        supports_vision=True,
    ),
    # ── Ollama (로컬) ────────────────────────────────────────────
    # 서버에 실제 pull된 모델만 등록한다. qwen3.6:27b는 미설치(404)라 제거함.
    # 추후 'ollama pull qwen3.6:27b' 후 다시 등록하면 fallback/auto 후보로 쓸 수 있다.
    "ollama/qwen3:14b": ModelSpec(
        provider="ollama", upstream="qwen3:14b",
        fallback=["gemini-2.5-flash"],      # 로컬 장애/과부하 시 Gemini(무료 클라우드)로 폴백
        # (E-02) context_window는 런타임 num_ctx와 한 소스(_OLLAMA_CONTEXT_WINDOW)로 묶는다.
        supports_tools=True, context_window=_OLLAMA_CONTEXT_WINDOW,
    ),
}

# 논리적 별칭 → 실제 모델 키. 필요 없으면 비워둬도 됨.
ALIASES: dict[str, str] = {
    "fast": "gemini-2.5-flash-lite",
    "smart": "gemini-2.5-flash",
}

# 기본 모델: 로컬 Ollama(qwen3:14b) 우선. 로컬 장애/과부하 시 fallback으로 Gemini(무료 클라우드).
DEFAULT_MODEL = "ollama/qwen3:14b"

# 로컬(자가호스팅) 후보만 있는 체인에 자동으로 이어붙일 SaaS 폴백(무료 클라우드 우선).
# 정책: '어떤 로컬 모델이든' 로컬 장애/과부하 시 클라우드로 넘어갈 곳을 보장한다.
# 레지스트리에 fallback을 안 적은 로컬 모델이나, 패스스루로 들어온 미등록 ollama/* 모델
# (예: ollama/gemma4:12b)도 이 보장을 받는다. 과금(Claude)은 넣지 않는다 — 비용 0 유지.
LOCAL_SAAS_FALLBACK: list[str] = ["gemini-2.5-flash"]

# ── Realtime(음성) 모델 별칭 ────────────────────────────────
# /v1/realtime 의 친화적 별칭 → 실제 Gemini Live 모델 id.
# 텍스트 모델(MODELS)과 별개의 Live 전용 모델군이라 분리한다.
# 계정/대시보드에서 사용 가능한 정확한 id로 교체할 것.
LIVE_ALIASES: dict[str, str] = {
    "gemini-live": "gemini-2.5-flash-native-audio-preview-09-2025",
}

# ── Embeddings 모델 (Phase 5 — /v1/embeddings) ──────────────
# chat 모델(MODELS)과 별개의 임베딩 전용 모델군. Anthropic은 임베딩 API가 없어 제외.
# ollama/nomic-embed-text 는 서버에 `ollama pull nomic-embed-text` 필요(README 배포 절 참고).
EMBEDDING_MODELS: dict[str, ModelSpec] = {
    "gemini-embedding-001": ModelSpec(
        provider="gemini",
        upstream="gemini-embedding-001",
        fallback=["ollama/nomic-embed-text"],   # 키 미설정/장애 시 로컬로 폴백
        cost_tier="free-cloud",
        is_free=True,
    ),
    "ollama/nomic-embed-text": ModelSpec(
        provider="ollama", upstream="nomic-embed-text",
    ),
}

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

# OpenAI SDK 기본 모델명을 그대로 받아주기 위한 별칭 — 클라이언트 코드 수정 없이 연동.
EMBED_ALIASES: dict[str, str] = {
    "embed": DEFAULT_EMBEDDING_MODEL,
    "text-embedding-3-small": DEFAULT_EMBEDDING_MODEL,
    "text-embedding-3-large": DEFAULT_EMBEDDING_MODEL,
}

# ── auto 라우트 (Phase 2~4) ─────────────────────────────────
# 클라이언트가 model="auto"로 보내면, 게이트웨이가 직접 모델을 고른다.
# 후보는 '무료'만. 로컬 우선 정책: 로컬 Ollama → free-cloud Gemini 순으로 둔다
# (로컬 장애·과부하·컨텍스트 초과 시 클라우드로 폴백).
# 과금(Claude)은 auto에서 자동 선택하지 않는다 — 비용 0 보장. Claude가 필요하면 명시 지정.
#
# Phase 3: 요청 난이도(simple/complex)를 먼저 판단해 티어별로 후보 셋을 다르게 쓴다.
#   simple  → 로컬 qwen3:14b 우선, 폴백 flash-lite   : 인사·단순 질의·짧은 대화
#   complex → 로컬 qwen3:14b 우선, 폴백 flash        : 도구 사용·긴 입력·다중턴·추론성 키워드
# Phase 4: 대용량 컨텍스트 전용 'long' 티어. 추정 입력이 로컬 usable 창을 넘으면
#   난이도와 무관하게 1M 컨텍스트 Gemini로 직행한다(로컬 32k는 어차피 필터에서 탈락).
AUTO_ROUTE = "auto"
AUTO_CANDIDATES_BY_TIER: dict[str, list[str]] = {
    "simple": ["ollama/qwen3:14b", "gemini-2.5-flash-lite"],
    "complex": ["ollama/qwen3:14b", "gemini-2.5-flash"],
    "long": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],   # 대용량 입력은 로컬 32k 불가 → 클라우드
}

# 토큰 추정 계수: char 수 → 대략의 토큰 수(한글/혼합 보수적으로 3 chars/token 가정).
# 정확한 토큰화가 아니라 "32k 로컬에 들어가나, 1M 클라우드가 필요한가" 판단용 근사치.
_CHARS_PER_TOKEN = 3

# 컨텍스트 안전 마진 (Phase 4): context_window의 이 비율까지만 '들어간다'고 본다.
# _estimate_tokens 가 근사치(과소추정 가능)인 데다, 로컬 모델은 입력·출력이
# 한 창(num_ctx)을 나눠 쓰므로 창을 꽉 채워 보내면 실제로는 잘리거나 실패한다.
_CONTEXT_SAFETY_RATIO = 0.8

# long 티어 임계값 (Phase 4): 추정 입력 토큰이 이 값 이상이면 난이도와 무관하게 'long'.
# 로컬(32k)의 usable 창(32,000 × 0.8 = 25,600)을 넘보는 크기 = 대용량 컨텍스트 전용 라우팅.
_LONG_INPUT_THRESHOLD = 25_000

# 난이도 분류 임계값 — 아래 중 하나라도 걸리면 complex로 본다.
_COMPLEX_TOKEN_THRESHOLD = 1200        # 추정 입력 토큰 (긴 입력 = 복잡)
_COMPLEX_MESSAGE_COUNT = 6             # 메시지 수 (긴 대화 = 복잡)
# 추론·생성 부담이 큰 작업을 시사하는 키워드(한/영, 소문자 비교). 단순 조회와 구분용.
_COMPLEX_KEYWORDS = (
    "분석", "설계", "구현", "디버그", "리팩터", "리팩토링", "최적화", "비교", "요약",
    "단계", "이유", "왜 ", "원인", "증명", "알고리즘", "전략", "계획", "검토", "리뷰",
    "오류", "버그", "코드", "작성해", "만들어",
    "analyze", "design", "implement", "debug", "refactor", "optimize", "compare",
    "summarize", "explain", "why", "algorithm", "review", "strategy", "plan",
)

# DEFAULT_MODEL·auto 후보는 반드시 MODELS에 존재해야 함 (기동 시점에 즉시 검증)
if DEFAULT_MODEL not in MODELS:
    raise ValueError(f"DEFAULT_MODEL '{DEFAULT_MODEL}' 가 MODELS에 없습니다")
for _name in {n for names in AUTO_CANDIDATES_BY_TIER.values() for n in names}:
    if _name not in MODELS:
        raise ValueError(f"AUTO 후보 '{_name}' 가 MODELS에 없습니다")
if DEFAULT_EMBEDDING_MODEL not in EMBEDDING_MODELS:
    raise ValueError(f"DEFAULT_EMBEDDING_MODEL '{DEFAULT_EMBEDDING_MODEL}' 가 EMBEDDING_MODELS에 없습니다")


# ─────────────────────────────────────────────────────────────
# 라우팅 결정
# ─────────────────────────────────────────────────────────────
def update_ollama_capabilities(models: list[dict]) -> None:
    """
    (E-03) list_models(/api/tags) 결과로 capability 캐시를 교체한다(태그 → capabilities).
    lifespan 시작 워밍업과 /v1/models 조회 핸들러가 호출한다. 조회 실패 시엔 호출하지 않아
    직전 캐시가 유지된다(빈 캐시면 _ollama_spec이 보수적 기본으로 폴백).
    """
    _OLLAMA_CAPABILITIES.clear()
    for model in models:
        name = model.get("name")
        if name:
            _OLLAMA_CAPABILITIES[name] = model.get("capabilities") or []


def _ollama_spec(tag: str) -> ModelSpec:
    """
    패스스루 ollama 모델의 spec을 만든다. capability 캐시(/api/tags)가 있으면 tools/vision
    지원 여부를 실제 모델에 맞추고(예: llava→vision, 텍스트 전용→tools만), context_window는
    런타임 num_ctx와 한 소스로 묶는다(E-02). 캐시가 없으면(미조회/구버전) 기존 보수적
    기본값(tools=True, vision=False)으로 폴백해 회귀를 피한다.
    """
    caps = _OLLAMA_CAPABILITIES.get(tag)
    if caps:
        supports_tools = "tools" in caps
        supports_vision = "vision" in caps
    else:
        supports_tools = True
        supports_vision = False
    return ModelSpec(
        provider="ollama", upstream=tag,
        supports_tools=supports_tools, supports_vision=supports_vision,
        context_window=_OLLAMA_CONTEXT_WINDOW,
    )


def _passthrough_spec(model: str) -> ModelSpec | None:
    """
    레지스트리에 없지만 모델명 형태로 provider를 추론할 수 있는 경우 spec 생성.
    Ollama는 받은 이름을 그대로 실행하므로 미등록 모델도 패스스루로 동작한다.
    명시적 provider 단서(prefix)를 먼저 보고, 그 외 콜론 포함만 Ollama 태그로 본다.
    (예: 미등록 'claude-x:snapshot' 도 콜론보다 claude- 를 우선해 Anthropic으로)
    """
    if model.startswith("ollama/"):
        return _ollama_spec(model.removeprefix("ollama/"))
    if model.startswith("anthropic/"):
        return ModelSpec(
            provider="anthropic", upstream=model.removeprefix("anthropic/"),
            cost_tier="paid", is_free=False, supports_vision=True,
        )
    if model.startswith("gemini/"):
        return ModelSpec(
            provider="gemini", upstream=model.removeprefix("gemini/"),
            cost_tier="free-cloud", is_free=True, context_window=1_000_000,
            supports_vision=True,
        )
    if model.startswith("claude-"):
        return ModelSpec(
            provider="anthropic", upstream=model,
            cost_tier="paid", is_free=False, context_window=200_000,
            supports_vision=True,
        )
    if model.startswith("gemini-"):
        return ModelSpec(
            provider="gemini", upstream=model,
            cost_tier="free-cloud", is_free=True, context_window=1_000_000,
            supports_vision=True,
        )
    if ":" in model:
        return _ollama_spec(model)
    return None


def _spec_for(model: str) -> ModelSpec | None:
    """모델 키 하나에 대한 spec 조회 (별칭 치환 → 레지스트리 → 패스스루)."""
    model = ALIASES.get(model, model)
    return MODELS.get(model) or _passthrough_spec(model)


def _ensure_saas_fallback(chain: list[ModelSpec]) -> list[ModelSpec]:
    """
    체인이 전부 로컬(provider="ollama")이면 끝에 SaaS 폴백을 이어붙인다.
    '모든 로컬 모델은 로컬 장애 시 클라우드로 넘어갈 곳이 있어야 한다'는 보장 —
    레지스트리에 fallback을 안 적은 로컬 모델과 패스스루 ollama/* 모두 커버한다.
    이미 SaaS(비-ollama) 후보가 있으면 손대지 않는다(중복/불필요 방지).
    """
    if any(spec.provider != "ollama" for spec in chain):
        return chain
    extended = list(chain)
    for name in LOCAL_SAAS_FALLBACK:
        spec = _spec_for(name)
        if spec is not None and spec not in extended:
            extended.append(spec)
    return extended


def resolve(model: str) -> RouteDecision:
    """
    모델 이름 → RouteDecision(primary + fallback 체인).
      1. 별칭 치환
      2. 레지스트리 조회, 없으면 형태로 추론(패스스루)
      3. 그래도 모르면 DEFAULT_MODEL(로컬 Ollama)
      4. primary.fallback을 이어붙여 체인 구성
    """
    model = model.strip()                       # 앞뒤 공백으로 인한 오라우팅 방지
    spec = _spec_for(model) or MODELS[DEFAULT_MODEL]

    chain = [spec]
    for fb in spec.fallback:
        fb_spec = _spec_for(fb)
        if fb_spec is not None:
            chain.append(fb_spec)

    return RouteDecision(chain=_ensure_saas_fallback(chain))


# ─────────────────────────────────────────────────────────────
# auto 라우팅 (Phase 2~4) — 요청 특성으로 모델을 직접 선택
# ─────────────────────────────────────────────────────────────
def _estimate_tokens(request: ChatCompletionRequest) -> int:
    """
    요청 입력 크기를 토큰 단위로 근사한다(메시지 + 도구 호출 이력 + 도구 정의).
    문자열 content는 길이, 비-문자열(멀티모달 등)은 JSON 직렬화 길이로 센 뒤
    _CHARS_PER_TOKEN으로 나눈다. context_window 적합성 판단용 근사치일 뿐
    정확한 토큰화가 아니다.
    """
    chars = 0
    for message in request.messages:
        content = message.content
        if isinstance(content, str):
            chars += len(content)
        elif content is not None:
            chars += len(json.dumps(content, ensure_ascii=False))
        # 멀티턴 도구 왕복의 tool_calls(함수 인자)도 입력 컨텍스트를 차지한다 —
        # 에이전트 대화에선 인자가 파일 내용 등으로 커질 수 있어 빼면 과소추정된다.
        if message.tool_calls:
            chars += len(json.dumps(message.tool_calls, ensure_ascii=False, default=str))
    if request.tools:
        chars += len(json.dumps([t.model_dump() for t in request.tools], ensure_ascii=False))
    return chars // _CHARS_PER_TOKEN


# 공개 별칭 — 다른 모듈(ollama의 동적 num_ctx 산정, E-08)이 같은 추정치를 재사용한다.
estimate_tokens = _estimate_tokens


def _usable_context(spec: ModelSpec) -> int:
    """안전 마진(_CONTEXT_SAFETY_RATIO)을 반영한 실효 컨텍스트 크기(토큰)."""
    return int(spec.context_window * _CONTEXT_SAFETY_RATIO)


def _has_images(request: ChatCompletionRequest) -> bool:
    """
    요청에 이미지 입력이 있는지 감지한다(Phase 5: vision capability 필터 근거).
    내부표준은 OpenAI 포맷이므로 `image_url` 파트만 보면 된다 — 네이티브 어댑터가
    Anthropic `image` 블록·Gemini `inlineData`를 이미 `image_url`로 변환해 들어온다.
    """
    for message in request.messages:
        content = message.content
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
    return False


def _classify_tier(request: ChatCompletionRequest, estimated_tokens: int) -> str:
    """
    요청 티어를 'long' | 'complex' | 'simple'로 분류한다(추가 LLM 호출 없는 휴리스틱).

    long이 난이도보다 우선한다 — 아무리 단순한 요청이라도 입력이 로컬 창을 넘으면
    큰 컨텍스트 모델로 보내는 것 외에 선택지가 없다.
      - long: 추정 입력 토큰 ≥ _LONG_INPUT_THRESHOLD (로컬 usable 창 초과 크기)
      - complex: 아래 중 하나라도 해당
          · 도구(tools) 사용 (function calling 오케스트레이션은 강한 모델이 유리)
          · 추정 입력 토큰 ≥ _COMPLEX_TOKEN_THRESHOLD (긴 입력)
          · 메시지 수 ≥ _COMPLEX_MESSAGE_COUNT (긴 멀티턴)
          · 사용자 메시지에 추론/생성 부담 키워드 포함
      - simple: 그 외 (짧은 단발 질의·인사 등)
    """
    if estimated_tokens >= _LONG_INPUT_THRESHOLD:
        return "long"
    if request.tools:
        return "complex"
    if estimated_tokens >= _COMPLEX_TOKEN_THRESHOLD:
        return "complex"
    if len(request.messages) >= _COMPLEX_MESSAGE_COUNT:
        return "complex"
    text = " ".join(
        message.content for message in request.messages
        if isinstance(message.content, str)
    ).lower()
    if any(keyword in text for keyword in _COMPLEX_KEYWORDS):
        return "complex"
    return "simple"


def _auto_route(request: ChatCompletionRequest) -> RouteDecision:
    """
    model="auto" 처리 (Phase 2~4).
      1. 입력 크기를 1회 추정하고 티어(simple/complex/long)에 맞는 후보 셋을 고른다
      2. 후보를 요청 특성으로 필터링한다
         - 도구를 쓰는 요청인데 도구 미지원 모델 → 제외
         - 이미지가 있는 요청인데 vision 미지원 모델 → 제외 (Phase 5 — 이미지가
           qwen3 같은 텍스트 전용 로컬로 폴백돼 조용히 무시되는 것을 방지)
         - '추정 입력 + 요청 출력(max_tokens)'이 usable 컨텍스트(안전 마진 반영)를
           초과하는 모델 → 제외 (로컬은 입력·출력이 한 창을 나눠 쓰므로 출력분 포함)
      3. 살아남은 후보를 비용·품질 우선순위(정의된 순서) 그대로 체인으로 만든다
      4. 컨텍스트 초과로 전원 탈락하면 후보 중 창이 가장 큰 모델을 best-effort로
         1개 시도한다 — DEFAULT_MODEL 무조건 폴백은 이미 탈락한 모델을 다시 고르는
         셈이라 의미가 없고, 진짜 한계 초과라면 업스트림 4xx로 명확히 드러난다.
    선택 근거는 reason("auto:tier=...,est=...")으로 x-llm-route 헤더에 노출된다.
    """
    estimated_tokens = _estimate_tokens(request)
    tier = _classify_tier(request, estimated_tokens)
    needs_tools = bool(request.tools)
    needs_vision = _has_images(request)
    # 출력 예산까지 포함한 필요 토큰. max_tokens 미지정이면 입력만으로 판단
    # (출력 여유분은 _CONTEXT_SAFETY_RATIO 마진이 흡수한다).
    required_tokens = estimated_tokens + (request.max_tokens or 0)

    candidates = [MODELS[name] for name in AUTO_CANDIDATES_BY_TIER[tier]]
    if needs_tools:
        candidates = [spec for spec in candidates if spec.supports_tools]
    if needs_vision:
        candidates = [spec for spec in candidates if spec.supports_vision]

    chain = [spec for spec in candidates if required_tokens <= _usable_context(spec)]
    reason = f"auto:tier={tier},est={estimated_tokens}"
    if needs_vision:
        reason += ",vision=1"

    if not chain and candidates:
        # 모든 후보의 창을 넘는 초대형 입력 — 그나마 가장 큰 창으로 best-effort
        chain = [max(candidates, key=lambda spec: spec.context_window)]
        reason += ",overflow=1"
    if not chain:
        # 도구/vision 필터로 전원 탈락(예: 이미지 요청인데 Gemini 후보가 없는 티어) — 최후 폴백
        chain = [MODELS[DEFAULT_MODEL]]
    # auto 후보가 로컬뿐인 경우에도 SaaS 폴백 보장(명시 라우팅과 동일 정책)
    return RouteDecision(chain=_ensure_saas_fallback(chain), reason=reason)


def _embedding_spec_for(model: str) -> ModelSpec | None:
    """임베딩 모델 키 하나에 대한 spec 조회 (별칭 → 임베딩 레지스트리 → 패스스루)."""
    model = EMBED_ALIASES.get(model, model)
    spec = EMBEDDING_MODELS.get(model)
    if spec is not None:
        return spec
    guessed = _passthrough_spec(model)
    # 임베딩은 OpenAI 호환 /embeddings 를 제공하는 Gemini·Ollama만 지원
    # (Anthropic은 임베딩 API 자체가 없음 → 잘못 유추된 후보는 버린다)
    if guessed is not None and guessed.provider in ("gemini", "ollama"):
        return guessed
    return None


def _ensure_saas_embedding_fallback(chain: list[ModelSpec]) -> list[ModelSpec]:
    """
    (E-11) 임베딩 체인이 전부 로컬(ollama)이면 끝에 SaaS 임베딩 폴백을 이어붙인다.
    chat의 _ensure_saas_fallback와 같은 '로컬 장애 시 클라우드로 넘어갈 곳 보장' 정책을
    임베딩으로 확장한 것 — 사용자가 ollama/nomic-embed-text 를 직접 지정했는데 미설치(404)여도
    Gemini 임베딩으로 폴백된다(chat 폴백은 chat 모델이라 임베딩엔 못 쓰므로 전용 함수).
    이미 SaaS 후보가 있으면 손대지 않는다.
    """
    if any(spec.provider != "ollama" for spec in chain):
        return chain
    saas = EMBEDDING_MODELS.get(DEFAULT_EMBEDDING_MODEL)
    if saas is not None and saas.provider != "ollama" and saas not in chain:
        return [*chain, saas]
    return chain


def resolve_embedding(model: str) -> RouteDecision:
    """
    임베딩 모델 이름 → RouteDecision. chat의 resolve()와 같은 구조지만
    EMBEDDING_MODELS/EMBED_ALIASES 를 본다. 미지 모델은 DEFAULT_EMBEDDING_MODEL로.
    로컬 전용 체인엔 SaaS 임베딩 폴백을 보장한다(E-11).
    """
    spec = _embedding_spec_for(model.strip()) or EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]
    chain = [spec]
    for fb in spec.fallback:
        fb_spec = _embedding_spec_for(fb)
        if fb_spec is not None:
            chain.append(fb_spec)
    return RouteDecision(chain=_ensure_saas_embedding_fallback(chain))


def resolve_live_model(requested: str | None, default: str) -> str:
    """
    Realtime 요청 모델명 → Gemini Live 모델 id로 변환한다.
      1. 요청이 없으면 default 사용
      2. LIVE_ALIASES 별칭 치환
      3. Gemini setup이 요구하는 'models/' 접두사를 보장
    """
    name = (requested or default or "").strip()
    name = LIVE_ALIASES.get(name, name)
    if not name:
        name = default
    return name if name.startswith("models/") else f"models/{name}"


def _guard_context(request: ChatCompletionRequest, decision: RouteDecision) -> RouteDecision:
    """
    명시 라우팅(비-auto)의 컨텍스트 초과 방어(P1).

    auto는 _auto_route가 이미 usable 창으로 후보를 거른다. 하지만 사용자가 모델을 직접
    지정(예: ollama/qwen3:14b)하면 그 필터가 없어, 큰 입력이 로컬의 작은 창을 넘겨도
    그대로 보내져 업스트림이 프롬프트를 조용히 잘라버린다(에이전트가 지시·도구를 잃음).

    primary가 '추정 입력 + 출력 예산'을 못 담으면:
      - 체인 안에 담을 수 있는 후보(예: 1M 창의 Gemini 폴백)가 있으면 앞으로 재정렬
      - 없으면 순서는 두되 reason에 truncate_risk=1을 실어 x-llm-route로 관측되게 한다
    여유가 있으면 결정을 그대로 반환한다(reason=None 보존).
    """
    if not decision.chain:
        return decision
    required = _estimate_tokens(request) + (request.max_tokens or 0)
    if required <= _usable_context(decision.chain[0]):
        return decision  # primary가 담을 수 있음 — 손대지 않음

    fit = [spec for spec in decision.chain if required <= _usable_context(spec)]
    unfit = [spec for spec in decision.chain if required > _usable_context(spec)]
    reason = f"truncate_risk=1,est={required}"
    return RouteDecision(chain=(fit + unfit) if fit else decision.chain, reason=reason)


def route(request: ChatCompletionRequest) -> RouteDecision:
    """
    요청 → RouteDecision 진입점. model="auto"(또는 auto로 향하는 별칭)이면
    요청 특성 기반 선택(_auto_route), 그 외엔 기존 이름 기반 resolve()(+컨텍스트 가드).
    """
    model = ALIASES.get(request.model.strip(), request.model.strip())
    if model == AUTO_ROUTE:
        return _auto_route(request)
    return _guard_context(request, resolve(request.model))
