from abc import ABC, abstractmethod
from typing import AsyncGenerator

from app.models import ChatCompletionRequest
from app.registry import ModelSpec


class LLMProvider(ABC):
    """
    provider 공통 인터페이스.
    인스턴스는 앱 생애주기 동안 1회 생성되어 재사용되며(ProviderPool),
    실제 호출할 모델 정보는 요청마다 ModelSpec으로 주입받는다.
    """

    @abstractmethod
    async def chat(self, request: ChatCompletionRequest, spec: ModelSpec) -> dict:
        """단일 응답 반환 (OpenAI chat.completion 형식)"""
        pass

    @abstractmethod
    async def stream(
        self, request: ChatCompletionRequest, spec: ModelSpec
    ) -> AsyncGenerator[str, None]:
        """SSE 형식 스트리밍 응답 (data: {...}\n\n)"""
        pass

    async def aclose(self) -> None:
        """풀 종료 시 보유한 client 자원 정리. 기본은 no-op."""
        return None
