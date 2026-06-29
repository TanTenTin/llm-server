"""
네이티브 포맷 어댑터 패키지.

게이트웨이 내부 표준(canonical)은 OpenAI chat.completions 포맷이다. 이 패키지는
edge에서 네이티브 요청(Anthropic Messages / Gemini generateContent)을 내부 표준으로
들여오고(inbound), 내부 표준 응답을 다시 네이티브 포맷으로 내보내는(outbound) 변환만
담당한다. 라우팅/폴백/회로차단기 등 실행 파이프라인은 그대로 재사용한다.

> providers/anthropic.py 의 변환은 '내부표준 → 업스트림 호출' 방향(outbound to provider)이고,
> 이 패키지의 변환은 '클라이언트 네이티브 요청 → 내부표준'(inbound from client) 방향이다.
> 방향이 반대라 별도 모듈로 둔다.
"""

from app.adapters.anthropic_io import (
    anthropic_to_chat_request,
    openai_to_anthropic_response,
    stream_openai_to_anthropic,
)
from app.adapters.gemini_io import (
    gemini_to_chat_request,
    openai_to_gemini_response,
    stream_openai_to_gemini,
)
from app.adapters.sse import sse_payloads

__all__ = [
    "anthropic_to_chat_request",
    "openai_to_anthropic_response",
    "stream_openai_to_anthropic",
    "gemini_to_chat_request",
    "openai_to_gemini_response",
    "stream_openai_to_gemini",
    "sse_payloads",
]
