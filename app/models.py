from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any, Literal, Union

# 모든 요청 모델 공통 설정: 모델에 선언하지 않은 필드도 '버리지 않고 보존'한다(extra="allow").
# 게이트웨이는 OpenAI 호환 요청을 그대로 받아 passthrough(Gemini·Ollama)로 흘려보내는데,
# Pydantic 기본값(extra="ignore")이면 모델이 모르는 필드를 조용히 버려 업스트림에 손실이 생긴다.
# (멀티턴 도구 왕복의 tool_calls 누락 → 400 이 정확히 이 문제였다.)
_ALLOW_EXTRA = ConfigDict(extra="allow")


class ToolFunction(BaseModel):
    model_config = _ALLOW_EXTRA

    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class Tool(BaseModel):
    model_config = _ALLOW_EXTRA

    type: Literal["function"] = "function"
    function: ToolFunction


class Message(BaseModel):
    model_config = _ALLOW_EXTRA

    role: str
    # OpenAI 스펙상 assistant가 tool_calls만 내는 메시지의 content는 null일 수 있다 → Optional.
    # (null 불가로 두면 opencode 등 멀티턴 도구 왕복에서 422가 난다)
    content: Optional[Union[str, List[Any]]] = None
    # assistant가 호출한 도구 목록. 멀티턴 도구 왕복에서 필수 — 빠지면 뒤따르는
    # tool 결과(tool_call_id)가 짝을 잃어 업스트림이 400으로 거부한다. 필드가 없으면
    # Pydantic이 조용히 버리므로 반드시 모델에 둬야 그대로 업스트림에 전달된다.
    tool_calls: Optional[List[Any]] = None
    # tool 응답 메시지에서 어떤 tool_call에 대한 결과인지 식별
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class EmbeddingsRequest(BaseModel):
    """OpenAI 호환 embeddings 요청 (/v1/embeddings). Gemini·Ollama 업스트림으로 패스스루."""
    model_config = _ALLOW_EXTRA

    model: str
    # 단일 문자열 또는 배치(문자열 배열). 토큰 id 배열 등도 그대로 업스트림에 전달
    input: Union[str, List[Any]]
    dimensions: Optional[int] = None
    encoding_format: Optional[str] = None
    user: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = _ALLOW_EXTRA

    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    # "auto" | "none" | {"type": "function", "function": {"name": "..."}}
    tool_choice: Optional[Union[str, dict]] = None
    # Ollama qwen3 전용: False로 설정 시 내부 reasoning(thinking) 비활성화
    think: Optional[bool] = None
