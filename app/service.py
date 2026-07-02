import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import AsyncGenerator, Awaitable, Callable

import anthropic
import httpx

from app.config import settings
from app.models import ChatCompletionRequest
from app.providers.anthropic import AnthropicProvider
from app.providers.base import LLMProvider
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.registry import ModelSpec, RouteDecision
from app.usage import UsageTracker

# fallback을 유발하는 HTTP 상태 (provider가 일시적으로/구조적으로 못 받는 상황)
#   404: 모델 미로드 · 408/409/429: 일시 과부하 · 5xx: provider 내부 오류
#   529: Anthropic OverloadedError(가장 흔한 일시 장애)
# 400/401/403(입력·인증 오류)는 재시도해도 동일 실패 → 포함하지 않음(즉시 실패)
_RETRYABLE_STATUS = {404, 408, 409, 429, 500, 502, 503, 504, 529}

# 동적 회로 쿨다운 상한(초). 업스트림이 알려준 Retry-After가 이보다 커도 1시간까지만
# 미룬다 — 파싱 오류/비정상 값으로 provider가 사실상 영구 제외되는 것을 막는 안전판.
# (RPD 소진처럼 진짜 긴 대기도 1시간마다 half-open 탐침이 실제 회복 시점을 잡아낸다)
_MAX_DYNAMIC_COOLDOWN_SECONDS = 3600.0

logger = logging.getLogger("llm_gateway")
logger.setLevel(logging.INFO)  # uvicorn 루트 핸들러로 전파되어 출력됨


class ProviderUnavailable(Exception):
    """provider가 설정되지 않아(예: API 키 미설정) 사용 불가. fallback 대상으로 취급."""


# ─────────────────────────────────────────────────────────────
# 회로차단기 — provider별 일시 장애를 기억해 쿨다운 동안 뒤로 미룬다
# ─────────────────────────────────────────────────────────────
class CircuitBreaker:
    """
    provider 단위로 최근 일시 장애를 기록한다. '열림(open)' 상태인 provider는
    폴백 체인에서 뒤로 미뤄(=primary부터 헛때리는 지연을 제거) 다른 후보를 먼저 시도한다.
    쿨다운이 지나면 자동으로 닫혀(half-open) 다시 정상 우선순위로 시도된다.

    무료 티어(Gemini)처럼 429가 빈번한 환경에서, 한 번 막히면 잠깐 로컬(Ollama)을
    우선시켜 응답 지연을 줄이는 것이 목적이다. 단일 프로세스의 코루틴 간 공유 상태로,
    읽기-쓰기 사이에 await가 없어 별도 락 없이 안전하다.
    """

    def __init__(self, cooldown_seconds: float) -> None:
        self._cooldown = cooldown_seconds
        self._open_until: dict[str, float] = {}  # provider → 이 시각(monotonic)까지 열림

    def is_open(self, provider: str) -> bool:
        if self._cooldown <= 0:
            return False  # 쿨다운 0 이하 → 회로차단 비활성화
        until = self._open_until.get(provider)
        if until is None:
            return False
        if time.monotonic() >= until:
            del self._open_until[provider]  # 쿨다운 만료 → 자동 닫힘(half-open)
            return False
        return True

    def record_failure(self, provider: str, cooldown_hint: float | None = None) -> float:
        """
        장애를 기록하고 실제 적용된 쿨다운(초)을 반환한다(로그 표기용).
        `cooldown_hint`: 업스트림이 알려준 재시도 대기 시간(Retry-After/RetryInfo).
        있으면 기본 쿨다운 대신 사용한다 — RPM 초과(수십 초)와 RPD 소진(수 시간)을
        같은 30초로 취급해 헛때리던 문제를 해소. 상한으로 클램프해 안전을 보장한다.
        """
        if self._cooldown <= 0:
            return 0.0  # 회로차단 비활성화 설정 존중 (힌트가 있어도 열지 않음)
        cooldown = self._cooldown
        if cooldown_hint is not None and cooldown_hint > 0:
            cooldown = min(cooldown_hint, _MAX_DYNAMIC_COOLDOWN_SECONDS)
        self._open_until[provider] = time.monotonic() + cooldown
        return cooldown

    def record_success(self, provider: str) -> None:
        self._open_until.pop(provider, None)  # 성공하면 즉시 닫음

    def status(self) -> dict[str, float]:
        """현재 open 상태인 provider와 남은 쿨다운(초). 헬스체크/관측용."""
        now = time.monotonic()
        return {
            provider: round(until - now, 1)
            for provider, until in self._open_until.items()
            if until > now
        }


# ─────────────────────────────────────────────────────────────
# 라우팅 트레이스 — 실제로 어떤 모델이 응답했는지 관측(헤더/로그)
# ─────────────────────────────────────────────────────────────
@dataclass
class RouteTrace:
    """한 요청에서 어떤 후보를 시도/스킵하고 무엇이 최종 응답했는지 기록(silent fallback 관측용)."""
    requested: str                                  # 클라이언트가 요청한 모델명
    served: str | None = None                       # 실제 응답한 spec 라벨 ("provider:upstream")
    fell_back: bool = False                          # primary가 아닌 후보가 응답했는지
    reason: str | None = None                        # 선택 사유 (auto:tier=complex 등). RouteDecision에서 주입
    attempts: list[str] = field(default_factory=list)   # 시도 이력 (label#ok/#retryable/#unavailable)
    deferred: list[str] = field(default_factory=list)   # 회로 open으로 뒤로 미뤄진 provider 라벨

    def header(self) -> str:
        """`x-llm-route` 응답 헤더 값으로 직렬화 (ASCII 안전)."""
        parts = [f"requested={self.requested}"]
        if self.served:
            parts.append(f"served={self.served}")
        if self.fell_back:
            parts.append("fallback=1")
        if self.reason:
            parts.append(f"reason={self.reason}")
        if self.deferred:
            parts.append("deferred=" + ",".join(self.deferred))
        return "; ".join(parts)


# ─────────────────────────────────────────────────────────────
# Provider 풀 — lifespan 동안 인스턴스 재사용
# ─────────────────────────────────────────────────────────────
class ProviderPool:
    def __init__(self) -> None:
        # Ollama는 항상 등록 (로컬 기본 provider)
        self._providers: dict[str, LLMProvider] = {
            "ollama": OllamaProvider(settings.ollama_base_url),
        }
        # API 키가 있을 때만 Anthropic 등록.
        # 빈 키로 SDK를 초기화하다 기동이 깨지는 것을 막고, Ollama 전용 배포에서
        # claude 요청이 와도 '사용 불가 → fallback' 으로 로컬에 떨어지게 한다.
        if settings.anthropic_api_key:
            self._providers["anthropic"] = AnthropicProvider(settings.anthropic_api_key)
        if settings.google_ai_api_key:
            self._providers["gemini"] = GeminiProvider(settings.google_ai_api_key)
        # provider별 일시 장애를 기억하는 회로차단기 (폴백 경로에서 참조)
        self.breaker = CircuitBreaker(settings.breaker_cooldown_seconds)
        # 모델별 요청/토큰/에러 집계 (무료 티어 쿼터 관측 — /v1/usage로 노출)
        self.usage = UsageTracker()

    def registered(self) -> list[str]:
        """등록된 provider 이름 목록 (헬스체크용)."""
        return list(self._providers)

    def get(self, provider_type: str) -> LLMProvider:
        provider = self._providers.get(provider_type)
        if provider is None:
            raise ProviderUnavailable(provider_type)
        return provider

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()


# ─────────────────────────────────────────────────────────────
# 에러 분류 / 정리 헬퍼
# ─────────────────────────────────────────────────────────────
def _spec_label(spec: ModelSpec) -> str:
    """로그·헤더에 쓸 사람이 읽기 좋은 식별자."""
    return f"{spec.provider}:{spec.upstream}"


def _order_by_breaker(
    chain: list[ModelSpec], breaker: CircuitBreaker
) -> tuple[list[ModelSpec], list[str]]:
    """
    회로 open 상태인 provider의 후보를 체인 '뒤로' 미룬다(순서만 바꿈, 누락 없음).
    정상 후보 → (뒤로 미룬) open 후보 순. 모두 open이어도 결국 시도하므로
    영구 실패로 빠지지 않는다(미뤄진 후보는 회복 탐침 역할도 겸함).
    반환: (재정렬된 체인, 미뤄진 provider 라벨 목록)
    """
    fresh: list[ModelSpec] = []
    deferred: list[ModelSpec] = []
    for spec in chain:
        (deferred if breaker.is_open(spec.provider) else fresh).append(spec)
    deferred_labels = [_spec_label(s) for s in deferred]
    return fresh + deferred, deferred_labels


def _error_kind(exc: Exception) -> str:
    """사용량 집계용 에러 분류 라벨 — 상태 코드 우선("429" 등), 없으면 연결 계열."""
    if isinstance(exc, httpx.HTTPStatusError):
        return str(exc.response.status_code)
    if isinstance(exc, anthropic.APIStatusError):
        return str(exc.status_code)
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, (httpx.ConnectError, anthropic.APIConnectionError)):
        return "connect"
    return type(exc).__name__


def _is_retryable(exc: Exception) -> bool:
    """다음 fallback 후보로 넘어갈 만한 에러인지 판단."""
    # 연결 실패 / 타임아웃 계열
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exc, anthropic.APIConnectionError):  # APITimeoutError 포함
        return True
    # HTTP 상태 코드 계열
    status: int | None = None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    elif isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code
    return status in _RETRYABLE_STATUS if status is not None else False


def _parse_retry_after_header(value: str) -> float | None:
    """Retry-After 헤더 값 파싱: 초 단위 숫자 또는 HTTP-date 두 형식 모두 지원."""
    try:
        return float(value)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    delta = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return delta if delta > 0 else None


def _parse_gemini_retry_info(body: dict) -> float | None:
    """
    Gemini 429 본문의 google.rpc.RetryInfo에서 재시도 대기 시간을 추출한다.
    형식: {"error": {"details": [{"@type": ".../google.rpc.RetryInfo", "retryDelay": "58s"}]}}
    """
    details = body.get("error", {}).get("details", [])
    if not isinstance(details, list):
        return None
    for detail in details:
        if not isinstance(detail, dict):
            continue
        if "RetryInfo" not in str(detail.get("@type", "")):
            continue
        delay = str(detail.get("retryDelay", "")).rstrip("s")
        try:
            return float(delay)
        except ValueError:
            return None
    return None


def retry_after_seconds(exc: Exception) -> float | None:
    """
    업스트림 에러(주로 429)에서 '언제 다시 시도 가능한지' 힌트를 초 단위로 추출한다.
    회로차단기의 동적 쿨다운에 사용 — RPM 초과(수십 초)와 RPD 소진(수 시간)을
    구분해, 고정 쿨다운으로 헛때리거나 너무 일찍 재시도하는 것을 막는다.
      1. 표준 Retry-After 헤더 (httpx/anthropic 예외 모두 .response가 httpx.Response)
      2. Gemini 429 본문의 RetryInfo.retryDelay
    힌트가 없거나 파싱 불가면 None → 기본 쿨다운 사용.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    header = response.headers.get("retry-after")
    if header:
        parsed = _parse_retry_after_header(header)
        if parsed is not None:
            return parsed
    # 본문은 provider가 미리 읽어둔 경우에만 접근 가능(스트리밍 경로 포함) — 실패는 조용히 무시
    try:
        body = response.json()
    except Exception:
        return None
    return _parse_gemini_retry_info(body) if isinstance(body, dict) else None


def error_detail(exc: Exception) -> str:
    """
    클라이언트(및 로그)에 돌려줄 상세 메시지. 업스트림 HTTP 에러면 **응답 본문(실제 사유)**을
    함께 노출한다 — 예전엔 httpx의 일반 메시지만 나와 Gemini의 400 사유가 가려졌다.
    스트리밍 경로는 provider에서 body를 미리 읽어둬야 여기서 .text가 채워진다.
    """
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        try:
            body = exc.response.text
        except Exception:
            body = ""
        return f"{exc}: {body[:800]}" if body else str(exc)
    if isinstance(exc, anthropic.APIStatusError):
        body = getattr(exc, "body", "") or ""
        return f"{exc}: {body}"[:800]
    return str(exc)


def http_status_for(exc: Exception) -> int:
    """예외를 클라이언트에 돌려줄 HTTP 상태로 매핑 (기본 500). 원인을 그대로 노출."""
    if isinstance(exc, ProviderUnavailable):
        return 503
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, anthropic.APIConnectionError)):
        return 502  # 업스트림 연결 실패
    return 500


async def aclose_quietly(gen: AsyncGenerator[str, None]) -> None:
    """제너레이터 정리 중 발생하는 2차 예외가 fallback/응답을 막지 않도록 삼킨다."""
    try:
        await gen.aclose()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Fallback 실행
#
# 후보 체인 순회·회로차단기·재시도·트레이스 같은 '라우팅 메커니즘'은 포맷과 무관하다.
# 그래서 제너릭 루프(run_chat_fallback / run_stream_fallback)로 분리하고, "각 후보를 실제로
# 어떻게 호출해 결과를 만들지"만 콜백(invoke / open_stream)으로 주입한다.
#   - OpenAI 엔드포인트: provider.chat/stream 을 그대로 호출(결과는 OpenAI 포맷)
#   - 네이티브 엔드포인트(/v1/messages 등): Anthropic 후보는 네이티브 패스스루, 그 외는
#     OpenAI 호출 후 네이티브 포맷으로 역변환 — 둘 다 '이미 응답 포맷으로 완성된' 결과를 돌려준다.
# ─────────────────────────────────────────────────────────────
async def run_chat_fallback(
    decision: RouteDecision,
    pool: ProviderPool,
    trace: RouteTrace | None,
    invoke: Callable[[ModelSpec], Awaitable[dict]],
) -> dict:
    """
    비스트리밍 fallback 공통 루프. 각 후보 spec마다 `invoke(spec)`을 호출해 결과(이미 응답
    포맷으로 완성된 dict)를 받는다. provider 선택·결과 변환의 구체 방식은 invoke에 위임한다.
    미설정 provider(`ProviderUnavailable`)/재시도 가능 에러면 다음 후보로 넘어간다.
    """
    breaker = pool.breaker
    ordered, deferred = _order_by_breaker(decision.chain, breaker)
    if trace is not None:
        trace.deferred = deferred
    if deferred:
        logger.info("[route] 회로 open → 뒤로 미룸: %s", ", ".join(deferred))

    last_exc: Exception | None = None
    for idx, spec in enumerate(ordered):
        label = _spec_label(spec)
        try:
            result = await invoke(spec)
        except ProviderUnavailable as e:
            last_exc = e  # 미설정 provider → 다음 후보로 (회로차단 대상 아님: 영구 설정 문제)
            pool.usage.record_error(label, "unavailable")
            if trace is not None:
                trace.attempts.append(f"{label}#unavailable")
            continue
        except Exception as e:
            if not _is_retryable(e):
                raise  # 입력/인증 오류 등은 즉시 실패
            cooldown = breaker.record_failure(spec.provider, retry_after_seconds(e))
            pool.usage.record_error(label, _error_kind(e))
            last_exc = e
            if trace is not None:
                trace.attempts.append(f"{label}#retryable")
            logger.warning(
                "[route] %s 일시 장애(%s) → 회로 open %.0fs, 다음 후보로",
                label, type(e).__name__, cooldown,
            )
            continue
        # 성공
        breaker.record_success(spec.provider)
        fell_back = idx > 0 or bool(deferred)
        pool.usage.record_success(label, result, fell_back)
        if trace is not None:
            trace.served = label
            trace.fell_back = fell_back
            trace.attempts.append(f"{label}#ok")
        logger.info("[route] 응답=%s (fallback=%s)", label, fell_back)
        return result
    raise last_exc if last_exc is not None else RuntimeError("라우팅 후보가 없습니다")


async def run_stream_fallback(
    decision: RouteDecision,
    pool: ProviderPool,
    trace: RouteTrace | None,
    open_stream: Callable[[ModelSpec], AsyncGenerator[str, None]],
) -> AsyncGenerator[str, None]:
    """
    스트리밍 fallback 공통 루프 — 첫 청크를 받기 전에 실패한 후보만 건너뛴다.
    `open_stream(spec)`은 해당 후보의 출력 스트림(이미 응답 포맷으로 완성된 SSE 제너레이터)을
    돌려준다(미설정 provider면 호출 시점에 `ProviderUnavailable` 발생). 한 번 토큰을 내보낸
    뒤에는 fallback 불가하며, 어떤 경로로 끝나든 업스트림 제너레이터를 반드시 정리한다.
    """
    breaker = pool.breaker
    ordered, deferred = _order_by_breaker(decision.chain, breaker)
    if trace is not None:
        trace.deferred = deferred
    if deferred:
        logger.info("[route] 회로 open → 뒤로 미룸: %s", ", ".join(deferred))

    last_exc: Exception | None = None
    for idx, spec in enumerate(ordered):
        label = _spec_label(spec)
        try:
            gen = open_stream(spec)  # pool.get 등에서 ProviderUnavailable 가능
        except ProviderUnavailable as e:
            last_exc = e
            pool.usage.record_error(label, "unavailable")
            if trace is not None:
                trace.attempts.append(f"{label}#unavailable")
            continue

        try:
            # 첫 청크를 당겨본다. 여기서 나는 에러는 아직 아무것도 보내기 전이라
            # fallback(또는 상위에서 HTTP 상태 변환)이 가능하다.
            first = await gen.__anext__()
        except StopAsyncIteration:
            await aclose_quietly(gen)
            breaker.record_success(spec.provider)  # 빈 스트림이라도 연결은 정상
            pool.usage.record_success(label, None, idx > 0 or bool(deferred))
            if trace is not None:
                trace.served = label
                trace.fell_back = idx > 0 or bool(deferred)
                trace.attempts.append(f"{label}#ok-empty")
            return  # 빈 스트림
        except Exception as e:
            await aclose_quietly(gen)
            if not _is_retryable(e):
                raise
            cooldown = breaker.record_failure(spec.provider, retry_after_seconds(e))
            pool.usage.record_error(label, _error_kind(e))
            last_exc = e
            if trace is not None:
                trace.attempts.append(f"{label}#retryable")
            logger.warning(
                "[route] %s 스트림 시작 실패(%s) → 회로 open %.0fs, 다음 후보로",
                label, type(e).__name__, cooldown,
            )
            continue

        # 첫 청크 확보 — 이후엔 fallback 없이 끝까지 흘려보내되 항상 정리
        # (스트리밍은 토큰 집계 없이 요청 수만 센다 — usage.py 모듈 설명 참고)
        breaker.record_success(spec.provider)
        fell_back = idx > 0 or bool(deferred)
        pool.usage.record_success(label, None, fell_back)
        if trace is not None:
            trace.served = label
            trace.fell_back = fell_back
            trace.attempts.append(f"{label}#ok")
        logger.info("[route] 스트림=%s (fallback=%s)", label, fell_back)
        try:
            yield first
            async for chunk in gen:
                yield chunk
        finally:
            await aclose_quietly(gen)
        return

    raise last_exc if last_exc is not None else RuntimeError("라우팅 후보가 없습니다")


async def chat_with_fallback(
    request: ChatCompletionRequest,
    decision: RouteDecision,
    pool: ProviderPool,
    trace: RouteTrace | None = None,
) -> dict:
    """OpenAI 경로: 각 후보를 provider.chat(request, spec)로 호출(결과는 OpenAI 포맷)."""
    async def invoke(spec: ModelSpec) -> dict:
        return await pool.get(spec.provider).chat(request, spec)
    return await run_chat_fallback(decision, pool, trace, invoke)


async def stream_with_fallback(
    request: ChatCompletionRequest,
    decision: RouteDecision,
    pool: ProviderPool,
    trace: RouteTrace | None = None,
) -> AsyncGenerator[str, None]:
    """OpenAI 경로 스트리밍: 각 후보를 provider.stream(request, spec)로 호출."""
    def open_stream(spec: ModelSpec) -> AsyncGenerator[str, None]:
        return pool.get(spec.provider).stream(request, spec)
    async for chunk in run_stream_fallback(decision, pool, trace, open_stream):
        yield chunk
