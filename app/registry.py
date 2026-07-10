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
    # 실패 시 시도할 다른 모델 키. **임베딩(EMBEDDING_MODELS) 전용** — chat 모델의 명시
    # 라우팅은 폴백하지 않으므로(resolve 참고) chat spec에는 비워 둔다.
    fallback: list[str] = field(default_factory=list)
    # ── 비용/특성 메타 (Phase 1: 로그·헤더 표시용 / Phase 2: 자동선택 판단 근거) ──
    cost_tier: str = "local"                   # "local"(자가호스팅) | "free-cloud"(무료 티어) | "paid"(과금)
    is_free: bool = True                       # 한계비용 0 여부 (무료 티어·로컬=True, 과금 API=False)
    # ── capability 메타 (Phase 2: auto 라우트의 후보 필터 근거) ──
    supports_tools: bool = True                # function calling(도구) 지원 여부
    context_window: int = 32_000               # 최대 입력 컨텍스트(토큰). 보수적 기본값
    # 모델이 한 번에 낼 수 있는 출력 토큰 상한. /v1/models가 max_output_tokens로 노출해
    # 클라이언트(opencode 등)가 출력 예산을 잡는 근거로 쓴다. max_tokens(요청 기본값)와는
    # 다른 개념 — 이쪽은 '모델의 물리 상한', 저쪽은 '요청에 없을 때 쓸 기본값'이다.
    max_output_tokens: int = 4096
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
# 주의: chat 모델의 `fallback`은 비워 둔다. 모델을 이름으로 콕 집은 요청은 그 모델로만
# 시도하고, 실패하면 실패시킨다(조용한 provider 바꿔치기 금지 — 아래 resolve() 참고).
# provider를 넘나드는 폴백은 model="auto"에서만 일어난다(AUTO_CANDIDATES_BY_TIER).
MODELS: dict[str, ModelSpec] = {
    # ── Gemini (기본 provider) ───────────────────────────────────
    "gemini-2.5-flash": ModelSpec(
        provider="gemini",
        upstream="gemini-2.5-flash",
        cost_tier="free-cloud",
        is_free=True,
        supports_tools=True,
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
    ),
    "gemini-2.5-flash-lite": ModelSpec(
        provider="gemini",
        upstream="gemini-2.5-flash-lite",
        cost_tier="free-cloud",
        is_free=True,
        supports_tools=True,
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
    ),
    # ── Anthropic ───────────────────────────────────────────────
    # max_output_tokens는 모델의 물리 상한이 아니라 '게이트웨이가 실제로 허용하는 출력'
    # (= max_tokens 기본값)에 맞춘다. 실제 Claude는 더 큰 출력을 내지만, 여기서 크게
    # 불러 놓으면 클라이언트가 그만큼 요청했다가 업스트림 400을 맞는다 — 낮춰 잡는 쪽이 안전.
    "claude-sonnet-4-6": ModelSpec(
        provider="anthropic",
        upstream="claude-sonnet-4-6",
        max_tokens=8192,
        cost_tier="paid",
        is_free=False,
        supports_tools=True,
        context_window=200_000,
        max_output_tokens=8192,
        supports_vision=True,
    ),
    "claude-opus-4-7": ModelSpec(
        provider="anthropic",
        upstream="claude-opus-4-7",
        max_tokens=8192,
        cost_tier="paid",
        is_free=False,
        supports_tools=True,
        context_window=200_000,
        max_output_tokens=8192,
        supports_vision=True,
    ),
    # ── Ollama (로컬) ────────────────────────────────────────────
    # 서버에 실제 pull된 모델만 등록한다. qwen3.6:27b는 미설치(404)라 제거함.
    # 추후 'ollama pull qwen3.6:27b' 후 다시 등록하면 fallback/auto 후보로 쓸 수 있다.
    "ollama/qwen3:14b": ModelSpec(
        provider="ollama", upstream="qwen3:14b",
        # (E-02) context_window는 런타임 num_ctx와 한 소스(_OLLAMA_CONTEXT_WINDOW)로 묶는다.
        supports_tools=True, context_window=_OLLAMA_CONTEXT_WINDOW,
    ),
}

# 논리적 별칭 → 실제 모델 키. 필요 없으면 비워둬도 됨.
ALIASES: dict[str, str] = {
    "fast": "gemini-2.5-flash-lite",
    "smart": "gemini-2.5-flash",
}

# 기본 모델: 모델명을 못 알아본 요청이 떨어질 곳. 로컬 Ollama(qwen3:14b).
DEFAULT_MODEL = "ollama/qwen3:14b"

# auto 라우트의 후보가 전부 로컬일 때 체인 끝에 이어붙일 SaaS 폴백(무료 클라우드).
# 정책: 'auto로 맡긴 요청'은 로컬 장애/과부하 시 클라우드로 넘어갈 곳을 보장한다.
# 과금(Claude)은 넣지 않는다 — 비용 0 유지.
#
# 명시 라우팅(resolve)에는 붙지 않는다. 사용자가 ollama/qwen3:14b 를 콕 집었으면
# 그게 죽어도 Gemini로 코드를 내보내지 않는다 — 조용한 provider 바꿔치기 금지.
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
#
# Phase 6: 'agentic' 티어. 코딩 에이전트(opencode·Claude Code 등)의 하네스는 도구 정의만으로
#   수천 토큰을 쓰고, 턴이 쌓이며 파일 내용·도구 결과가 계속 누적된다. 후보 순서는 다른
#   티어와 같은 '로컬 우선'이다 — auto의 기본은 비용 0인 로컬이어야 한다.
#   컨텍스트가 로컬 창을 넘어서는 순간 _auto_route의 _usable_context 필터가 로컬을 떨어뜨려
#   1M Gemini가 자동으로 잡히므로, 로컬 우선이어도 대화가 커지면 막히지 않는다.
#   티어를 따로 두는 이유는 후보 순서가 아니라 관측이다 — x-llm-route의 reason=tier=agentic
#   으로 '이 세션은 하네스라 계속 커질 것'임을 로그에서 구분할 수 있다.
AUTO_ROUTE = "auto"
AUTO_CANDIDATES_BY_TIER: dict[str, list[str]] = {
    "simple": ["ollama/qwen3:14b", "gemini-2.5-flash-lite"],
    "complex": ["ollama/qwen3:14b", "gemini-2.5-flash"],
    "agentic": ["ollama/qwen3:14b", "gemini-2.5-flash"],     # 로컬 우선, 창 넘으면 1M으로 승격
    "long": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],   # 대용량 입력은 로컬 32k 불가 → 클라우드
}

# 토큰 추정 계수: char 수 → 대략의 토큰 수. 정확한 토큰화가 아니라
# "32k 로컬에 들어가나, 1M 클라우드가 필요한가" 판단용 근사치.
#
# ASCII와 비-ASCII를 나눠 센다. 예전엔 전부 3 chars/token으로 뭉뚱그렸는데, 그러면
# 한글 입력을 약 1.85배 과소추정한다. 그 결과 ollama의 동적 num_ctx(_resolve_num_ctx)가
# 실제 프롬프트보다 작게 잡히고, Ollama는 초과분을 에러 없이 '창의 절반만 남기고
# 앞부분을 버리는' 식으로 잘라낸다(에이전트가 시스템 프롬프트·도구 정의·초반 대화를 통째로 잃음).
#
# 실측(ollama/ornith:9b, 한글 산문 6,000자 = 3,704토큰): 문장 전체로는 1.62 chars/token이지만
# 그 문장은 30%가 ASCII(띄어쓰기·마침표)였다. 띄어쓰기는 뒤따르는 한글 토큰에 흡수돼 거의
# 공짜이므로, 한글 글자 자체의 밀도는 ~1.2 chars/token이다. 1.62를 그대로 쓰면 한글 비중이
# 높은 입력에서 다시 과소추정된다.
#
# 과소추정만이 잘림을 만들고(과다추정은 창을 넉넉히 잡아 KV 캐시만 더 쓴다) 낭비는
# OLLAMA_NUM_CTX 상한이 막아주므로, 양쪽 다 실측보다 낮은(=보수적) 값을 쓴다.
_ASCII_CHARS_PER_TOKEN = 3      # 영문 산문·JSON 도구 스키마 (실측 ~3.8)
_WIDE_CHARS_PER_TOKEN = 1.2     # 한글·CJK 등 비-ASCII (실측 ~1.2~1.27)

# 요청이 max_tokens를 안 줄 때 출력용으로 예약할 토큰 수.
# 컨텍스트 초과 판정(context_overflow)과 ollama의 num_ctx 산정(_resolve_num_ctx)이
# 같은 값을 써야 "보낼 땐 된다고 했는데 실제로 잘리는" 어긋남이 생기지 않는다.
DEFAULT_OUTPUT_BUDGET = 2048

# 컨텍스트 안전 마진 (Phase 4): context_window의 이 비율까지만 '들어간다'고 본다.
# _estimate_tokens 가 근사치(과소추정 가능)인 데다, 로컬 모델은 입력·출력이
# 한 창(num_ctx)을 나눠 쓰므로 창을 꽉 채워 보내면 실제로는 잘리거나 실패한다.
_CONTEXT_SAFETY_RATIO = 0.8

# long 티어 임계값 (Phase 4): 추정 입력 토큰이 이 값 이상이면 난이도와 무관하게 'long'.
# 로컬(32k)의 usable 창(32,000 × 0.8 = 25,600)을 넘보는 크기 = 대용량 컨텍스트 전용 라우팅.
_LONG_INPUT_THRESHOLD = 25_000

# agentic 티어 임계값 (Phase 6): 도구 '정의'만으로 이 토큰을 넘으면 코딩 에이전트 하네스로 본다.
# opencode·Claude Code류는 read/edit/bash/grep… 십수 개의 JSON 스키마를 매 턴 재전송해
# 도구 정의만 수천 토큰이다. 반면 일반 앱의 function calling은 도구 1~2개(수백 토큰)에 그친다.
# 이 경계가 '한 번 부르고 끝나는 요청'과 '턴이 쌓이며 컨텍스트가 커지는 요청'을 가른다.
_AGENTIC_TOOL_TOKENS = 1500

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


def ollama_supports_thinking(tag: str) -> bool | None:
    """
    태그가 thinking(사고)을 지원하는지 capability 캐시로 판정한다.
    캐시에 근거가 없으면(미조회·구버전 Ollama) None — 호출 측이 접두사 휴리스틱으로 폴백한다.
    """
    caps = _OLLAMA_CAPABILITIES.get(tag)
    if not caps:
        return None
    return "thinking" in caps


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
    **auto 라우트 전용** — 모델 선택을 게이트웨이에 맡긴 요청만 이 보장을 받는다.
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


def _local_only_chain(chain: list[ModelSpec]) -> list[ModelSpec]:
    """
    체인에서 SaaS(비-ollama) 후보를 모두 제거해 '로컬에서만 추론' 보장을 만든다.

    호출자가 로컬 전용을 요구한 경우(x-llm-local-only)에 쓴다. 기본 정책인
    _ensure_saas_fallback의 정반대 — 로컬이 죽으면 클라우드로 넘어가는 대신
    그냥 실패시킨다. 프롬프트/코드가 외부 provider로 나가면 안 되는 호출자
    (예: 로컬 모델만 쓰기로 한 에이전트)를 위한 것이다.

    필터 결과가 비면(사용자가 gemini-2.5-flash처럼 SaaS 모델을 콕 집은 경우)
    DEFAULT_MODEL(로컬)로 강등한다. 조용한 강등이 아니라 RouteDecision.reason과
    x-llm-route 헤더에 local_only=1로 드러나므로 호출자가 확인할 수 있다.
    """
    local = [spec for spec in chain if spec.provider == "ollama"]
    return local or [MODELS[DEFAULT_MODEL]]


def resolve(model: str, *, local_only: bool = False) -> RouteDecision:
    """
    모델 이름 → RouteDecision. **명시 라우팅은 폴백하지 않는다** — 체인은 후보 하나뿐이다.
      1. 별칭 치환
      2. 레지스트리 조회, 없으면 형태로 추론(패스스루)
      3. 그래도 모르면 DEFAULT_MODEL(로컬 Ollama)

    이름을 콕 집은 요청을 다른 provider로 넘기지 않는 이유: 사용자가 ollama/qwen3:14b를
    지정했다면 그건 '이 모델로 돌려라'가 아니라 대개 '이 모델로만 돌려라'라는 뜻이다.
    로컬이 죽었다고 코드·프롬프트를 Gemini로 내보내거나, Gemini가 429라고 품질이 다른
    로컬 14B가 조용히 답하면 호출자는 무엇이 답했는지 모른 채 결과만 받는다.
    provider를 넘나드는 폴백이 필요하면 model="auto"로 게이트웨이에 선택을 맡길 것.

    local_only=True인데 SaaS 모델을 지정했다면 DEFAULT_MODEL(로컬)로 강등한다
    (_local_only_chain — reason/x-llm-route 헤더에 local_only=1로 드러남).
    """
    model = model.strip()                       # 앞뒤 공백으로 인한 오라우팅 방지
    spec = _spec_for(model) or MODELS[DEFAULT_MODEL]
    chain = [spec]

    if local_only:
        return RouteDecision(chain=_local_only_chain(chain), reason="local_only=1")
    return RouteDecision(chain=chain)


# ─────────────────────────────────────────────────────────────
# auto 라우팅 (Phase 2~4) — 요청 특성으로 모델을 직접 선택
# ─────────────────────────────────────────────────────────────
def _text_tokens(text: str) -> float:
    """
    문자열 하나의 추정 토큰 수. ASCII/비-ASCII를 나눠 각자의 계수로 환산한다.

    글자를 순회하지 않고 UTF-8 바이트 길이로 비-ASCII 글자 수를 역산한다
    (ASCII=1바이트, 한글·CJK=3바이트 → 비-ASCII 글자수 ≈ (bytes - len) / 2).
    4바이트인 이모지는 약간 과다 계상되는데, 과다추정은 창을 넉넉히 잡을 뿐
    잘림을 만들지 않으므로 안전한 방향의 오차다.
    """
    total = len(text)
    wide = (len(text.encode("utf-8")) - total) // 2
    return (total - wide) / _ASCII_CHARS_PER_TOKEN + wide / _WIDE_CHARS_PER_TOKEN


def _estimate_tokens(request: ChatCompletionRequest) -> int:
    """
    요청 입력 크기를 토큰 단위로 근사한다(메시지 + 도구 호출 이력 + 도구 정의).
    문자열 content는 그대로, 비-문자열(멀티모달 등)은 JSON 직렬화 결과를
    _text_tokens로 환산한다. context_window 적합성 판단용 근사치일 뿐
    정확한 토큰화가 아니다.
    """
    tokens = 0.0
    for message in request.messages:
        content = message.content
        if isinstance(content, str):
            tokens += _text_tokens(content)
        elif content is not None:
            tokens += _text_tokens(json.dumps(content, ensure_ascii=False))
        # 멀티턴 도구 왕복의 tool_calls(함수 인자)도 입력 컨텍스트를 차지한다 —
        # 에이전트 대화에선 인자가 파일 내용 등으로 커질 수 있어 빼면 과소추정된다.
        if message.tool_calls:
            tokens += _text_tokens(json.dumps(message.tool_calls, ensure_ascii=False, default=str))
    if request.tools:
        tokens += _text_tokens(json.dumps([t.model_dump() for t in request.tools], ensure_ascii=False))
    return int(tokens)


# 공개 별칭 — 다른 모듈(ollama의 동적 num_ctx 산정, E-08)이 같은 추정치를 재사용한다.
estimate_tokens = _estimate_tokens


def _tool_tokens(request: ChatCompletionRequest) -> int:
    """도구 '정의'(스키마)만의 추정 토큰 수. agentic 티어 판정 근거."""
    if not request.tools:
        return 0
    return int(_text_tokens(json.dumps(
        [tool.model_dump() for tool in request.tools], ensure_ascii=False
    )))


def ollama_context_window() -> int:
    """로컬 Ollama 모델의 컨텍스트 창(런타임 num_ctx와 한 소스). /v1/models 메타 노출용."""
    return _OLLAMA_CONTEXT_WINDOW


def _usable_context(spec: ModelSpec) -> int:
    """안전 마진(_CONTEXT_SAFETY_RATIO)을 반영한 실효 컨텍스트 크기(토큰)."""
    return int(spec.context_window * _CONTEXT_SAFETY_RATIO)


def context_overflow(request: ChatCompletionRequest, spec: ModelSpec) -> tuple[int, int] | None:
    """
    요청이 spec의 컨텍스트 창에 담기지 않는지 판정한다. 담기면 None, 넘치면 (필요 토큰, 창 크기).

    _usable_context(0.8 마진)가 아니라 창 '전체'와 비교한다 — _estimate_tokens가 실측보다
    크게 잡는(보수적) 쪽으로 고쳐졌으므로 마진을 이중으로 걸면 실제로는 들어가는 요청까지
    거부하게 된다. 출력 몫을 함께 세는 건 로컬 모델이 입력·출력으로 한 창(num_ctx)을
    나눠 쓰기 때문이다 — 입력이 창을 꽉 채우면 답할 자리가 남지 않는다.
    """
    required = _estimate_tokens(request) + (request.max_tokens or DEFAULT_OUTPUT_BUDGET)
    return (required, spec.context_window) if required > spec.context_window else None


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
    요청 티어를 'long' | 'agentic' | 'complex' | 'simple'로 분류한다(추가 LLM 호출 없는 휴리스틱).

    long이 난이도보다 우선한다 — 아무리 단순한 요청이라도 입력이 로컬 창을 넘으면
    큰 컨텍스트 모델로 보내는 것 외에 선택지가 없다.
      - long: 추정 입력 토큰 ≥ _LONG_INPUT_THRESHOLD (로컬 usable 창 초과 크기)
      - agentic: 도구 정의만 ≥ _AGENTIC_TOOL_TOKENS (코딩 에이전트 하네스)
          후보 순서는 complex와 같은 로컬 우선. 입력이 로컬 창을 넘으면 컨텍스트 필터가
          로컬을 떨어뜨려 1M Gemini로 승격된다. 티어 구분은 관측(reason=tier=agentic)용.
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
        if _tool_tokens(request) >= _AGENTIC_TOOL_TOKENS:
            return "agentic"
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


def _auto_route(request: ChatCompletionRequest, *, local_only: bool = False) -> RouteDecision:
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
    # 로컬 전용 요청은 후보 단계에서 SaaS를 걷어낸다. long/agentic 티어처럼 후보가 전부
    # SaaS인 경우 여기서 전멸하고, 아래 _local_only_chain이 DEFAULT_MODEL로 되돌린다.
    # 그 강등은 '요청한 티어를 못 지켰다'는 뜻이므로 reason에 명시한다 — 클라이언트가
    # 1M 창을 기대하고 대화를 안 압축하다 413을 맞는 것을 x-llm-route로 진단할 수 있어야 한다.
    local_downgrade = False
    if local_only:
        local_candidates = [spec for spec in candidates if spec.provider == "ollama"]
        local_downgrade = bool(candidates) and not local_candidates
        candidates = local_candidates

    chain = [spec for spec in candidates if required_tokens <= _usable_context(spec)]
    reason = f"auto:tier={tier},est={estimated_tokens}"
    if needs_vision:
        reason += ",vision=1"
    if local_only:
        reason += ",local_only=1"
    if local_downgrade:
        reason += ",local_downgrade=1"

    if not chain and candidates:
        # 모든 후보의 창을 넘는 초대형 입력 — 그나마 가장 큰 창으로 best-effort
        chain = [max(candidates, key=lambda spec: spec.context_window)]
        reason += ",overflow=1"
    if not chain:
        # 도구/vision 필터로 전원 탈락(예: 이미지 요청인데 Gemini 후보가 없는 티어) — 최후 폴백
        chain = [MODELS[DEFAULT_MODEL]]
    if local_only:
        # SaaS 폴백을 붙이지 않는다 — 로컬이 죽으면 그대로 실패시킨다.
        return RouteDecision(chain=_local_only_chain(chain), reason=reason)
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
    명시 라우팅(비-auto)의 컨텍스트 초과 '표시'(P1).

    auto는 _auto_route가 이미 창 크기로 후보를 거른다. 하지만 사용자가 모델을 직접
    지정(예: ollama/qwen3:14b)하면 그 필터가 없어, 큰 입력이 로컬의 작은 창을 넘겨도
    그대로 보내져 업스트림이 프롬프트를 조용히 잘라버린다(에이전트가 지시·도구를 잃음).

    예전엔 담을 수 있는 후보(1M 창의 Gemini 폴백)를 앞으로 재정렬했다. 그런데 그건
    사용자가 콕 집어 지정한 모델을 말없이 다른 provider로 바꿔치기하는 셈이고, 그 Gemini가
    쿼터(429)로 죽으면 결국 로컬로 되돌아와 조용히 잘렸다 — 안전망이 아니라 지연된 실패였다.

    이제는 reason에 표시만 하고, 실제 거부는 fallback 루프가 ContextTooLarge(413)로 즉시
    처리한다. 컨텍스트 초과는 일시 장애가 아니라 입력 오류이므로 폴백으로 뭉개지 않는다.
    """
    if not decision.chain:
        return decision
    overflow = context_overflow(request, decision.chain[0])
    if overflow is None:
        return decision  # primary가 담을 수 있음 — 손대지 않음(reason=None 보존)

    required, window = overflow
    return RouteDecision(
        chain=decision.chain,
        reason=f"context_overflow=1,est={required},window={window}",
    )


def route(request: ChatCompletionRequest, *, local_only: bool = False) -> RouteDecision:
    """
    요청 → RouteDecision 진입점. model="auto"(또는 auto로 향하는 별칭)이면
    요청 특성 기반 선택(_auto_route), 그 외엔 기존 이름 기반 resolve()(+컨텍스트 가드).

    local_only=True(요청 헤더 x-llm-local-only)면 체인을 로컬(ollama) 후보로만 구성한다.
    """
    model = ALIASES.get(request.model.strip(), request.model.strip())
    if model == AUTO_ROUTE:
        return _auto_route(request, local_only=local_only)
    return _guard_context(request, resolve(request.model, local_only=local_only))
