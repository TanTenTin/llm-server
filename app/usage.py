"""
사용량 집계 (P0 관측 계층).

provider:model 라벨 × UTC 일 단위로 요청 수·토큰·에러를 인메모리 집계한다.
목적은 과금 리포트가 아니라 **무료 티어 쿼터 관측** — "오늘 Gemini를 몇 번 때렸고
429가 몇 번 났는지"를 게이트웨이 자신이 알게 하는 것. 이후 예산 가드(P1)·선제
라우팅의 데이터 기반이 된다.

설계 전제:
  - 단일 프로세스(uvicorn 워커 1) — 코루틴 간 공유 dict로 충분, 락 불필요
    (읽기-쓰기 사이 await 없음. CircuitBreaker와 동일한 전제).
  - 재기동 시 리셋 — 배포가 컨테이너 재빌드라 영속화는 하지 않는다.
    장기 보존이 필요해지면 주기적 JSONL 덤프를 추가한다.
  - 스트리밍 응답은 요청 수만 센다(토큰은 마지막 SSE 청크에만 실려 파싱 비용 대비
    이득이 작음). usage 필드가 없는 응답도 요청 수는 집계된다.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

# 보존 일수. 오래된 날짜 버킷은 기록 시점에 잘라 메모리를 상수로 유지한다.
RETENTION_DAYS = 7


def _extract_usage(body: dict) -> tuple[int, int]:
    """
    응답 본문에서 (입력 토큰, 출력 토큰)을 추출한다. 게이트웨이가 다루는 3종 포맷 지원:
      - OpenAI:    usage.prompt_tokens / completion_tokens
      - Anthropic: usage.input_tokens / output_tokens        (네이티브 패스스루 응답)
      - Gemini:    usageMetadata.promptTokenCount / candidatesTokenCount (네이티브 패스스루 응답)
    없으면 (0, 0) — 요청 수 집계는 그대로 유효하다.
    """
    usage = body.get("usage")
    if isinstance(usage, dict):
        if "prompt_tokens" in usage:  # OpenAI
            return usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
        if "input_tokens" in usage:  # Anthropic
            return usage.get("input_tokens") or 0, usage.get("output_tokens") or 0
    meta = body.get("usageMetadata")
    if isinstance(meta, dict):  # Gemini
        return meta.get("promptTokenCount") or 0, meta.get("candidatesTokenCount") or 0
    return 0, 0


@dataclass
class ModelDayUsage:
    """한 모델 라벨의 하루치 카운터."""
    requests: int = 0                                   # 성공 응답 수 (스트리밍 포함)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    served_as_fallback: int = 0                          # primary가 아닌 후보로 응답한 횟수
    errors: dict[str, int] = field(default_factory=dict)  # 에러 종류(429/503/connect 등) → 횟수

    def as_dict(self) -> dict:
        total = self.prompt_tokens + self.completion_tokens
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": total,
            "served_as_fallback": self.served_as_fallback,
            "errors": dict(self.errors),
        }


class UsageTracker:
    """provider:model 라벨 × UTC 일 단위 사용량 집계기 (인메모리)."""

    def __init__(self) -> None:
        # 날짜("2026-07-02") → 라벨("gemini:gemini-2.5-flash") → 카운터
        self._days: dict[str, dict[str, ModelDayUsage]] = {}
        # 날짜 → 과금(is_free=False) provider가 소비한 토큰 합. 예산 가드(P1)의 판단 근거
        self._paid_tokens: dict[str, int] = {}
        self._started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _bucket(self, label: str) -> ModelDayUsage:
        """오늘 날짜·라벨의 카운터를 확보하고, 보존 기간을 넘긴 날짜는 정리한다."""
        today = self._today()
        if today not in self._days and len(self._days) >= RETENTION_DAYS:
            for stale in sorted(self._days)[: len(self._days) - RETENTION_DAYS + 1]:
                del self._days[stale]
                self._paid_tokens.pop(stale, None)
        return self._days.setdefault(today, {}).setdefault(label, ModelDayUsage())

    def record_success(
        self, label: str, body: dict | None, fell_back: bool, is_free: bool = True
    ) -> None:
        """성공 응답 집계. body가 없으면(스트리밍) 요청 수만 센다."""
        bucket = self._bucket(label)
        bucket.requests += 1
        if fell_back:
            bucket.served_as_fallback += 1
        if body is not None:
            prompt, completion = _extract_usage(body)
            bucket.prompt_tokens += prompt
            bucket.completion_tokens += completion
            if not is_free:
                today = self._today()
                self._paid_tokens[today] = self._paid_tokens.get(today, 0) + prompt + completion

    def paid_tokens_today(self) -> int:
        """오늘(UTC) 과금 provider가 소비한 토큰 합. 예산 가드가 매 후보 시도 전에 조회."""
        return self._paid_tokens.get(self._today(), 0)

    def record_error(self, label: str, kind: str) -> None:
        """실패 시도 집계. kind는 상태 코드("429")나 분류("connect"/"unavailable")."""
        bucket = self._bucket(label)
        bucket.errors[kind] = bucket.errors.get(kind, 0) + 1

    def snapshot(self) -> dict:
        """/v1/usage 응답 본문. 날짜 → 라벨 → 카운터 + 전 기간 합계."""
        days = {
            day: {label: usage.as_dict() for label, usage in labels.items()}
            for day, labels in sorted(self._days.items())
        }
        totals: dict[str, dict] = {}
        for labels in self._days.values():
            for label, usage in labels.items():
                agg = totals.setdefault(label, {
                    "requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "served_as_fallback": 0, "errors": {},
                })
                snap = usage.as_dict()
                for key in ("requests", "prompt_tokens", "completion_tokens",
                            "total_tokens", "served_as_fallback"):
                    agg[key] += snap[key]
                for kind, count in snap["errors"].items():
                    agg["errors"][kind] = agg["errors"].get(kind, 0) + count
        return {
            "tracker_started_at": self._started_at,
            "retention_days": RETENTION_DAYS,
            "days": days,
            "totals": totals,
            "paid_tokens_by_day": dict(sorted(self._paid_tokens.items())),
        }
