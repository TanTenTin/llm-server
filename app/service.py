from typing import AsyncGenerator

import anthropic
import httpx

from app.config import settings
from app.models import ChatCompletionRequest
from app.providers.anthropic import AnthropicProvider
from app.providers.base import LLMProvider
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.registry import RouteDecision

# fallback을 유발하는 HTTP 상태 (provider가 일시적으로/구조적으로 못 받는 상황)
#   404: 모델 미로드 · 408/409/429: 일시 과부하 · 5xx: provider 내부 오류
#   529: Anthropic OverloadedError(가장 흔한 일시 장애)
# 400/401/403(입력·인증 오류)는 재시도해도 동일 실패 → 포함하지 않음(즉시 실패)
_RETRYABLE_STATUS = {404, 408, 409, 429, 500, 502, 503, 504, 529}


class ProviderUnavailable(Exception):
    """provider가 설정되지 않아(예: API 키 미설정) 사용 불가. fallback 대상으로 취급."""


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
# ─────────────────────────────────────────────────────────────
async def chat_with_fallback(
    request: ChatCompletionRequest, decision: RouteDecision, pool: ProviderPool
) -> dict:
    """체인을 순서대로 시도. 미설정 provider/재시도 가능 에러면 다음 후보로."""
    last_exc: Exception | None = None
    for spec in decision.chain:
        try:
            provider = pool.get(spec.provider)
            return await provider.chat(request, spec)
        except ProviderUnavailable as e:
            last_exc = e            # 미설정 provider → 다음 후보로
        except Exception as e:
            if not _is_retryable(e):
                raise               # 입력/인증 오류 등은 즉시 실패
            last_exc = e            # 재시도 가능 → 다음 후보로
    raise last_exc if last_exc is not None else RuntimeError("라우팅 후보가 없습니다")


async def stream_with_fallback(
    request: ChatCompletionRequest, decision: RouteDecision, pool: ProviderPool
) -> AsyncGenerator[str, None]:
    """
    스트리밍 fallback — 첫 청크를 받기 전에 실패한 후보만 건너뛴다.
    한 번 토큰을 내보낸 뒤에는(이미 바이트를 전송했으므로) fallback 불가하며,
    어떤 경로로 끝나든 업스트림 제너레이터를 반드시 정리한다(커넥션 누수 방지).
    """
    last_exc: Exception | None = None
    for spec in decision.chain:
        try:
            provider = pool.get(spec.provider)
        except ProviderUnavailable as e:
            last_exc = e
            continue

        gen = provider.stream(request, spec)
        try:
            # 첫 청크를 당겨본다. 여기서 나는 에러는 아직 아무것도 보내기 전이라
            # fallback(또는 상위에서 HTTP 상태 변환)이 가능하다.
            first = await gen.__anext__()
        except StopAsyncIteration:
            await aclose_quietly(gen)
            return  # 빈 스트림
        except Exception as e:
            await aclose_quietly(gen)
            if not _is_retryable(e):
                raise
            last_exc = e
            continue

        # 첫 청크 확보 — 이후엔 fallback 없이 끝까지 흘려보내되 항상 정리
        try:
            yield first
            async for chunk in gen:
                yield chunk
        finally:
            await aclose_quietly(gen)
        return

    raise last_exc if last_exc is not None else RuntimeError("라우팅 후보가 없습니다")
