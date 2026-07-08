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
  - 스트리밍 응답도 토큰을 집계한다(E-01) — SSE 스트림을 흘려보내며 usage를 실은 청크를
    sniff_stream_usage로 훑어 누적하고, 스트림 종료 시 record_stream_tokens로 반영한다.
    usage가 실리지 않는 스트림(구형 업스트림 등)은 요청 수만 집계된다.
"""

import json
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


def sniff_stream_usage(chunk: str, acc: dict) -> None:
    """
    스트리밍 SSE 청크 문자열에서 usage를 추출해 누적기(acc)에 last-wins로 반영한다(E-01).
    게이트웨이가 중계하는 3종 스트림 포맷을 모두 훑는다:
      - OpenAI-compat: 마지막 청크의 top-level usage.prompt_tokens/completion_tokens
        (Gemini는 stream_options.include_usage, Ollama 네이티브는 게이트웨이가 붙여 보낸다)
      - Gemini 네이티브: 각 청크의 usageMetadata(누적값이라 마지막이 최종)
      - Anthropic 네이티브: message_start의 message.usage.input_tokens +
        message_delta의 usage.output_tokens (두 이벤트에 나눠 실림 → 필드별 last-wins)
    각 usage는 최종 청크에 완성값(누적)으로 실리므로 last-wins가 곧 최종값이다.
    usage 마커가 없는 순수 델타 청크는 문자열 검사로 즉시 건너뛴다(파싱 비용 회피).
    """
    if "usage" not in chunk and "usageMetadata" not in chunk:
        return
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        # OpenAI(top-level usage) / Anthropic message_delta(usage.output_tokens)
        usage = obj.get("usage")
        if isinstance(usage, dict):
            if "prompt_tokens" in usage:  # OpenAI
                if usage.get("prompt_tokens"):
                    acc["prompt"] = usage["prompt_tokens"]
                if usage.get("completion_tokens"):
                    acc["completion"] = usage["completion_tokens"]
            else:  # Anthropic message_delta (input은 message_start에서, output이 여기서 갱신)
                if usage.get("input_tokens"):
                    acc["prompt"] = usage["input_tokens"]
                if usage.get("output_tokens"):
                    acc["completion"] = usage["output_tokens"]
        # Anthropic message_start: message.usage.input_tokens
        message = obj.get("message")
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            mu = message["usage"]
            if mu.get("input_tokens"):
                acc["prompt"] = mu["input_tokens"]
            if mu.get("output_tokens"):
                acc["completion"] = mu["output_tokens"]
        # Gemini 네이티브: usageMetadata
        meta = obj.get("usageMetadata")
        if isinstance(meta, dict):
            if meta.get("promptTokenCount"):
                acc["prompt"] = meta["promptTokenCount"]
            if meta.get("candidatesTokenCount"):
                acc["completion"] = meta["candidatesTokenCount"]


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

    def record_stream_tokens(
        self, label: str, prompt: int, completion: int, is_free: bool = True
    ) -> None:
        """
        스트리밍 응답의 토큰을 사후 반영한다(E-01). 요청 수는 이미 record_success(body=None)로
        집계됐으므로 여기선 토큰만 더한다. 과금(is_free=False) 스트림도 paid 토큰에 적립돼
        예산 가드가 다음 요청부터 이를 근거로 삼는다(진행 중 스트림은 소급 차단 불가).
        """
        if prompt <= 0 and completion <= 0:
            return
        bucket = self._bucket(label)
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
