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
    content: Union[str, List[Any]]
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
