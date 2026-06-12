from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# 데이터 모델
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    """
    하나의 모델을 어떤 provider로, 어떤 upstream 이름으로 호출할지에 대한 정의.
    라우팅·모델명 변환의 단일 진실 공급원(SSOT).
    """
    provider: str                              # "ollama" | "anthropic"
    upstream: str                              # provider에 실제로 보낼 모델명
    max_tokens: int | None = None              # 기본 max_tokens (요청에 없을 때 사용)
    fallback: list[str] = field(default_factory=list)  # 실패 시 시도할 다른 모델 키


@dataclass(frozen=True)
class RouteDecision:
    """resolve() 결과. 시도 순서대로 정렬된 spec 체인 (primary + fallback들)."""
    chain: list[ModelSpec]


# ─────────────────────────────────────────────────────────────
# 레지스트리 (코드 내 관리 — 모델 추가/수정 시 여기만 손대면 됨)
# ─────────────────────────────────────────────────────────────
MODELS: dict[str, ModelSpec] = {
    "claude-sonnet-4-6": ModelSpec(
        provider="anthropic",
        upstream="claude-sonnet-4-6",
        max_tokens=8192,
        fallback=["ollama/qwen3.6:27b"],   # Anthropic 장애 시 로컬로 폴백
    ),
    "claude-opus-4-7": ModelSpec(
        provider="anthropic",
        upstream="claude-opus-4-7",
        max_tokens=8192,
        fallback=["ollama/qwen3.6:27b"],
    ),
    "ollama/qwen3:14b": ModelSpec(provider="ollama", upstream="qwen3:14b"),
    "ollama/qwen3.6:27b": ModelSpec(provider="ollama", upstream="qwen3.6:27b"),
}

# 논리적 별칭 → 실제 모델 키. 필요 없으면 비워둬도 됨.
ALIASES: dict[str, str] = {
    "fast": "ollama/qwen3:14b",
    "smart": "claude-sonnet-4-6",
}

# 경로 없는/모르는 요청이 떨어지는 기본 모델 (로컬 Ollama)
DEFAULT_MODEL = "ollama/qwen3:14b"

# DEFAULT_MODEL은 반드시 MODELS에 존재해야 함 (기동 시점에 즉시 검증)
if DEFAULT_MODEL not in MODELS:
    raise ValueError(f"DEFAULT_MODEL '{DEFAULT_MODEL}' 가 MODELS에 없습니다")


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
        return ModelSpec(provider="anthropic", upstream=model.removeprefix("anthropic/"))
    if model.startswith("claude-"):
        return ModelSpec(provider="anthropic", upstream=model)
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
