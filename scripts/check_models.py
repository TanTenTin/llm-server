"""
모델 드리프트 감지 스크립트 (GitHub Actions에서 매일 실행).

provider(예: Gemini)의 '라이브 모델 목록 API'를 조회해, 게이트웨이 레지스트리
(app.registry.MODELS)가 알고 있는 모델과 비교한다. 다음 두 가지를 감지한다.

  - new   : 라이브 목록엔 있는데 레지스트리엔 없는 채팅 모델 (새 무료 모델 후보)
  - removed: 레지스트리엔 있는데 라이브 목록에서 사라진 모델 (deprecated/미지원 후보)

릴리스 노트 프로즈를 스크래핑하지 않고 models API를 신뢰원으로 쓴다(정확·안정).
결과를 마크다운 리포트로 쓰고, GITHUB_OUTPUT에 has_drift 플래그를 내보낸다.
실제 registry.py 수정은 하지 않는다 — 사람이 이슈를 보고 판단/반영한다.

provider 추가 절차: ModelSource를 상속한 클래스 1개 작성 + build_sources()에 등록.
"""

import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

# app.registry는 app.models(pydantic)만 의존 → CI에서 httpx+pydantic만 설치하면 임포트 가능
from app.registry import MODELS

# ─────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────
REPORT_PATH = "model-drift-report.md"

# 새 모델 후보에서 제외할 비-채팅/비-안정 변형(부분 문자열). 노이즈를 줄이기 위함.
_SKIP_SUBSTRINGS = (
    "preview", "exp", "tuning", "embedding", "aqa", "vision", "imagen",
    "learnlm", "thinking", "gemma", "-it", "tts", "audio", "image-generation",
)
# 날짜/숫자 스냅샷 변형 접미사 (예: -001, -0827, -09-2025) → 안정 모델만 보려고 제외
_SNAPSHOT_SUFFIX = re.compile(r"-\d{3,}$|-\d{2}-\d{4}$")

HTTP_TIMEOUT = 30.0


# ─────────────────────────────────────────────────────────────
# 데이터 모델
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LiveModel:
    """provider가 보고한 라이브 모델 한 건."""
    model_id: str                 # provider 접두사 제거한 모델명 (레지스트리 upstream과 비교)
    input_token_limit: int | None  # 컨텍스트 윈도우 추정에 활용 (없으면 None)


@dataclass(frozen=True)
class ProviderDiff:
    """한 provider에 대한 드리프트 결과."""
    provider: str
    new_models: list[LiveModel]    # 레지스트리에 없는 새 채팅 모델 후보
    removed_models: list[str]      # 레지스트리엔 있으나 라이브에서 사라진 upstream 이름

    @property
    def has_drift(self) -> bool:
        return bool(self.new_models or self.removed_models)


# ─────────────────────────────────────────────────────────────
# Provider별 모델 소스 (pluggable)
# ─────────────────────────────────────────────────────────────
class ModelSource(ABC):
    """provider의 라이브 채팅 모델 목록을 가져오는 추상화. provider 추가 시 이걸 상속."""

    provider: str = ""

    @abstractmethod
    def fetch_chat_models(self) -> dict[str, LiveModel]:
        """채팅(generateContent 등) 가능한 모델만 {model_id: LiveModel}로 반환."""
        raise NotImplementedError


class GeminiModelSource(ModelSource):
    """Gemini ListModels API (OpenAI 호환 아님, Google 네이티브 v1beta)."""

    provider = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch_chat_models(self) -> dict[str, LiveModel]:
        models: dict[str, LiveModel] = {}
        page_token: str | None = None
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            while True:
                params: dict[str, str | int] = {"key": self.api_key, "pageSize": 200}
                if page_token:
                    params["pageToken"] = page_token
                response = client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                for entry in data.get("models", []):
                    methods = entry.get("supportedGenerationMethods", [])
                    if "generateContent" not in methods:
                        continue  # 임베딩/토큰카운트 전용 등은 제외
                    model_id = entry.get("name", "").removeprefix("models/")
                    if not model_id:
                        continue
                    models[model_id] = LiveModel(
                        model_id=model_id,
                        input_token_limit=entry.get("inputTokenLimit"),
                    )
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
        return models


# ─────────────────────────────────────────────────────────────
# 진단 / 리포트
# ─────────────────────────────────────────────────────────────
def build_sources() -> list[ModelSource]:
    """API 키가 설정된 provider만 소스로 등록. (provider 추가 시 여기에 한 줄)"""
    sources: list[ModelSource] = []
    gemini_key = os.environ.get("GOOGLE_AI_API_KEY", "")
    if gemini_key:
        sources.append(GeminiModelSource(gemini_key))
    return sources


def _registry_upstreams(provider: str) -> set[str]:
    """레지스트리가 추적 중인 해당 provider의 upstream 모델명 집합."""
    return {spec.upstream for spec in MODELS.values() if spec.provider == provider}


def _is_noise(model_id: str) -> bool:
    """새 모델 후보에서 걸러낼 비-채팅/스냅샷 변형인지."""
    lowered = model_id.lower()
    if any(token in lowered for token in _SKIP_SUBSTRINGS):
        return True
    if _SNAPSHOT_SUFFIX.search(lowered):
        return True
    return False


def diff_source(source: ModelSource) -> ProviderDiff:
    """한 provider의 라이브 목록과 레지스트리를 비교해 new/removed 산출."""
    live = source.fetch_chat_models()
    tracked = _registry_upstreams(source.provider)

    removed = sorted(name for name in tracked if name not in live)
    new = sorted(
        (model for model_id, model in live.items()
         if model_id not in tracked and not _is_noise(model_id)),
        key=lambda m: m.model_id,
    )
    return ProviderDiff(provider=source.provider, new_models=new, removed_models=removed)


def render_report(diffs: list[ProviderDiff], drift: bool) -> str:
    """이슈 본문으로 쓸 마크다운 리포트 생성."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# 🤖 모델 드리프트 리포트")
    lines.append("")
    lines.append(f"- 생성: {now} (GitHub Actions `model-watch`)")
    lines.append(f"- 신뢰원: 각 provider의 models 목록 API ↔ `app/registry.py`의 `MODELS`")
    lines.append("")

    if not drift:
        lines.append("✅ **현재 드리프트 없음.** 레지스트리가 라이브 모델 목록과 일치합니다.")
        return "\n".join(lines) + "\n"

    lines.append("⚠️ **레지스트리 갱신이 필요한 변동이 감지되었습니다.** 아래를 검토 후 `app/registry.py`를 수정하세요.")
    lines.append("")

    for diff in diffs:
        if not diff.has_drift:
            continue
        lines.append(f"## provider: `{diff.provider}`")
        lines.append("")

        if diff.new_models:
            lines.append("### 🆕 새 모델 후보 (레지스트리에 없음)")
            lines.append("")
            lines.append("| 모델명 | inputTokenLimit |")
            lines.append("|--------|-----------------|")
            for model in diff.new_models:
                limit = f"{model.input_token_limit:,}" if model.input_token_limit else "—"
                lines.append(f"| `{model.model_id}` | {limit} |")
            lines.append("")
            lines.append("> 반영 시 `MODELS`에 추가 예시:")
            lines.append("> ```python")
            for model in diff.new_models:
                ctx = model.input_token_limit or 32_000
                lines.append(
                    f'> "{model.model_id}": ModelSpec(provider="{diff.provider}", '
                    f'upstream="{model.model_id}", cost_tier="free-cloud", is_free=True, '
                    f"supports_tools=True, context_window={ctx}, fallback=[...]),"
                )
            lines.append("> ```")
            lines.append("> 무료/도구지원 여부와 `auto` 후보(`AUTO_CANDIDATES_BY_TIER`) 편입은 사람이 판단하세요.")
            lines.append("")

        if diff.removed_models:
            lines.append("### ❌ 사라진 모델 (deprecated/미지원 후보)")
            lines.append("")
            for name in diff.removed_models:
                lines.append(f"- `{name}` — 라이브 목록에 없음. `MODELS`에서 제거하거나 후속 모델로 교체하고, "
                             f"이 모델을 `fallback`/`AUTO_CANDIDATES_BY_TIER`에서 참조하는 곳도 함께 수정하세요.")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    sources = build_sources()
    if not sources:
        print("ERROR: 감시할 provider가 없습니다. GOOGLE_AI_API_KEY를 설정하세요.", file=sys.stderr)
        return 2

    diffs = [diff_source(source) for source in sources]
    drift = any(diff.has_drift for diff in diffs)

    report = render_report(diffs, drift)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)
    print(report)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as file:
            file.write(f"has_drift={'true' if drift else 'false'}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
