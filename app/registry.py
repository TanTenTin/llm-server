import json
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class RouteDecision:
    """resolve()/route() 결과. 시도 순서대로 정렬된 spec 체인 (primary + fallback들)."""
    chain: list[ModelSpec]
    # 선택 사유(관측용). auto 라우트면 "auto:tier=complex" 등, 그 외엔 None.
    reason: str | None = None


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
    ),
    "gemini-2.5-flash-lite": ModelSpec(
        provider="gemini",
        upstream="gemini-2.5-flash-lite",
        fallback=["ollama/qwen3:14b"],
        cost_tier="free-cloud",
        is_free=True,
        supports_tools=True,
        context_window=1_000_000,
    ),
    # ── Anthropic ───────────────────────────────────────────────
    "claude-sonnet-4-6": ModelSpec(
        provider="anthropic",
        upstream="claude-sonnet-4-6",
        max_tokens=8192,
        fallback=["ollama/qwen3.6:27b"],
        cost_tier="paid",
        is_free=False,
        supports_tools=True,
        context_window=200_000,
    ),
    "claude-opus-4-7": ModelSpec(
        provider="anthropic",
        upstream="claude-opus-4-7",
        max_tokens=8192,
        fallback=["ollama/qwen3.6:27b"],
        cost_tier="paid",
        is_free=False,
        supports_tools=True,
        context_window=200_000,
    ),
    # ── Ollama (로컬) ────────────────────────────────────────────
    "ollama/qwen3:14b": ModelSpec(
        provider="ollama", upstream="qwen3:14b",
        supports_tools=True, context_window=32_000,
    ),
    "ollama/qwen3.6:27b": ModelSpec(
        provider="ollama", upstream="qwen3.6:27b",
        supports_tools=True, context_window=32_000,
    ),
}

# 논리적 별칭 → 실제 모델 키. 필요 없으면 비워둬도 됨.
ALIASES: dict[str, str] = {
    "fast": "gemini-2.5-flash-lite",
    "smart": "gemini-2.5-flash",
}

# 기본 모델: GOOGLE_AI_API_KEY 있으면 Gemini, 키 미설정이면 fallback으로 로컬 Ollama
DEFAULT_MODEL = "gemini-2.5-flash"

# ── auto 라우트 (Phase 2~3) ─────────────────────────────────
# 클라이언트가 model="auto"로 보내면, 게이트웨이가 직접 모델을 고른다.
# 후보는 '무료'만, 비용·품질 우선순위 순으로 둔다(free-cloud Gemini → 로컬 Ollama).
# 과금(Claude)은 auto에서 자동 선택하지 않는다 — 비용 0 보장. Claude가 필요하면 명시 지정.
#
# Phase 3: 요청 난이도(simple/complex)를 먼저 판단해 티어별로 후보 셋을 다르게 쓴다.
#   simple  → 가볍고 빠른 모델(flash-lite, 14b)   : 인사·단순 질의·짧은 대화
#   complex → 강한 모델(flash, 27b)               : 도구 사용·긴 입력·다중턴·추론성 키워드
AUTO_ROUTE = "auto"
AUTO_CANDIDATES_BY_TIER: dict[str, list[str]] = {
    "simple": ["gemini-2.5-flash-lite", "ollama/qwen3:14b"],
    "complex": ["gemini-2.5-flash", "ollama/qwen3.6:27b"],
}

# 토큰 추정 계수: char 수 → 대략의 토큰 수(한글/혼합 보수적으로 3 chars/token 가정).
# 정확한 토큰화가 아니라 "32k 로컬에 들어가나, 1M 클라우드가 필요한가" 판단용 근사치.
_CHARS_PER_TOKEN = 3

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


# ─────────────────────────────────────────────────────────────
# 라우팅 결정
# ─────────────────────────────────────────────────────────────
def _passthrough_spec(model: str) -> ModelSpec | None:
    """
    레지스트리에 없지만 모델명 형태로 provider를 추론할 수 있는 경우 spec 생성.
    Ollama는 받은 이름을 그대로 실행하므로 미등록 모델도 패스스루로 동작한다.
    명시적 provider 단서(prefix)를 먼저 보고, 그 외 콜론 포함만 Ollama 태그로 본다.
    (예: 미등록 'claude-x:snapshot' 도 콜론보다 claude- 를 우선해 Anthropic으로)
    """
    if model.startswith("ollama/"):
        return ModelSpec(provider="ollama", upstream=model.removeprefix("ollama/"))
    if model.startswith("anthropic/"):
        return ModelSpec(
            provider="anthropic", upstream=model.removeprefix("anthropic/"),
            cost_tier="paid", is_free=False,
        )
    if model.startswith("gemini/"):
        return ModelSpec(
            provider="gemini", upstream=model.removeprefix("gemini/"),
            cost_tier="free-cloud", is_free=True, context_window=1_000_000,
        )
    if model.startswith("claude-"):
        return ModelSpec(
            provider="anthropic", upstream=model,
            cost_tier="paid", is_free=False, context_window=200_000,
        )
    if model.startswith("gemini-"):
        return ModelSpec(
            provider="gemini", upstream=model,
            cost_tier="free-cloud", is_free=True, context_window=1_000_000,
        )
    if ":" in model:
        return ModelSpec(provider="ollama", upstream=model)
    return None


def _spec_for(model: str) -> ModelSpec | None:
    """모델 키 하나에 대한 spec 조회 (별칭 치환 → 레지스트리 → 패스스루)."""
    model = ALIASES.get(model, model)
    return MODELS.get(model) or _passthrough_spec(model)


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

    return RouteDecision(chain=chain)


# ─────────────────────────────────────────────────────────────
# auto 라우팅 (Phase 2) — 요청 특성으로 모델을 직접 선택
# ─────────────────────────────────────────────────────────────
def _estimate_tokens(request: ChatCompletionRequest) -> int:
    """
    요청 입력 크기를 토큰 단위로 근사한다(메시지 + 도구 정의).
    문자열 content는 길이, 비-문자열(멀티모달 등)은 JSON 직렬화 길이로 센 뒤
    _CHARS_PER_TOKEN으로 나눈다. context_window 적합성 판단용 근사치일 뿐
    정확한 토큰화가 아니다.
    """
    chars = 0
    for message in request.messages:
        content = message.content
        if isinstance(content, str):
            chars += len(content)
        else:
            chars += len(json.dumps(content, ensure_ascii=False))
    if request.tools:
        chars += len(json.dumps([t.model_dump() for t in request.tools], ensure_ascii=False))
    return chars // _CHARS_PER_TOKEN


def _classify_complexity(request: ChatCompletionRequest) -> str:
    """
    요청 난이도를 'simple' | 'complex'로 분류한다(추가 LLM 호출 없는 휴리스틱).
    아래 중 하나라도 걸리면 complex:
      - 도구(tools) 사용 (function calling 오케스트레이션은 강한 모델이 유리)
      - 추정 입력 토큰 ≥ _COMPLEX_TOKEN_THRESHOLD (긴 입력)
      - 메시지 수 ≥ _COMPLEX_MESSAGE_COUNT (긴 멀티턴)
      - 사용자 메시지에 추론/생성 부담 키워드 포함
    그 외(짧은 단발 질의·인사 등)는 simple.
    """
    if request.tools:
        return "complex"
    if _estimate_tokens(request) >= _COMPLEX_TOKEN_THRESHOLD:
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
    model="auto" 처리. 먼저 난이도(simple/complex)를 판단해 티어별 후보 셋을 고르고,
    그 후보를 요청 특성으로 필터링한다.
      - 도구를 쓰는 요청인데 도구 미지원 모델 → 제외
      - 추정 입력 토큰이 context_window를 초과하는 모델 → 제외
    살아남은 후보를 비용·품질 우선순위(정의된 순서) 그대로 체인으로 만든다.
    모두 탈락하면(예: 입력이 모든 무료 후보의 컨텍스트를 초과) DEFAULT_MODEL로 폴백한다.
    """
    tier = _classify_complexity(request)
    needs_tools = bool(request.tools)
    estimated_tokens = _estimate_tokens(request)

    chain: list[ModelSpec] = []
    for name in AUTO_CANDIDATES_BY_TIER[tier]:
        spec = MODELS[name]
        if needs_tools and not spec.supports_tools:
            continue
        if spec.context_window < estimated_tokens:
            continue
        chain.append(spec)

    if not chain:
        chain = [MODELS[DEFAULT_MODEL]]
    return RouteDecision(chain=chain, reason=f"auto:tier={tier}")


def route(request: ChatCompletionRequest) -> RouteDecision:
    """
    요청 → RouteDecision 진입점. model="auto"(또는 auto로 향하는 별칭)이면
    요청 특성 기반 선택(_auto_route), 그 외엔 기존 이름 기반 resolve().
    """
    model = ALIASES.get(request.model.strip(), request.model.strip())
    if model == AUTO_ROUTE:
        return _auto_route(request)
    return resolve(request.model)
