from pydantic import BaseModel
from typing import Optional, List, Any, Literal, Union


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class Message(BaseModel):
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


class ChatCompletionRequest(BaseModel):
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
