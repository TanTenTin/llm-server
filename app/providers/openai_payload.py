from app.models import ChatCompletionRequest, EmbeddingsRequest
from app.registry import ModelSpec

# OpenAI 호환 업스트림(Gemini·Ollama)으로 '그대로 전달'하는 표준 파라미터 화이트리스트.
# 메시지 구조(tool_calls 등)는 extra="allow" 덕분에 messages dump에 보존되어 자동 전달되므로
# 여기엔 요청 레벨 샘플링/포맷 파라미터만 둔다. 게이트웨이 내부/특정 provider 전용 필드
# (예: Ollama의 think)는 의도적으로 제외 — 다른 업스트림에 보내면 거부될 수 있다.
_FORWARD_PARAMS = (
    "temperature", "top_p", "top_k", "max_tokens", "stop", "seed",
    "presence_penalty", "frequency_penalty", "n", "logprobs", "top_logprobs",
    "response_format", "tool_choice", "parallel_tool_calls",
)


def build_openai_payload(request: ChatCompletionRequest, spec: ModelSpec) -> dict:
    """
    OpenAI 호환 업스트림용 payload를 구성한다(Gemini·Ollama 공용).

    핵심: extra="allow"로 모델이 미지 필드까지 보존하므로, `model_dump`된 messages에는
    tool_calls 같은 구조 필드가 그대로 들어 있다 → 업스트림에 손실 없이 전달된다.
    요청 레벨 파라미터는 _FORWARD_PARAMS 화이트리스트로 전달한다(미지의 요청 레벨 필드를
    무차별 전달하면 엄격한 업스트림이 400을 낼 수 있어, 메시지는 보존·요청 파라미터는 선별).
    """
    dumped = request.model_dump(exclude_none=True)  # 미지 필드 포함(extra="allow")

    payload: dict = {
        "model": spec.upstream,                      # 레지스트리가 정한 실제 업스트림 모델명
        "messages": dumped.get("messages", []),      # 메시지 구조/미지 필드 보존
        "stream": request.stream or False,
    }
    # 스트리밍이면 마지막에 usage 청크를 받도록 요청한다(E-01 — 게이트웨이가 스트리밍
    # 토큰을 집계할 수 있게. Gemini OpenAI-compat이 stream_options.include_usage를 지원).
    if request.stream:
        payload["stream_options"] = {"include_usage": True}
    for key in _FORWARD_PARAMS:
        if key in dumped:
            payload[key] = dumped[key]
    if "tools" in dumped:
        payload["tools"] = dumped["tools"]
    # max_tokens 기본값: 요청에 없으면 레지스트리 spec 값
    if "max_tokens" not in payload and spec.max_tokens is not None:
        payload["max_tokens"] = spec.max_tokens

    return payload


# embeddings 요청에서 업스트림으로 전달할 선택 파라미터 (chat과 동일한 화이트리스트 원칙)
_EMBED_FORWARD_PARAMS = ("dimensions", "encoding_format", "user")


def build_embeddings_payload(request: EmbeddingsRequest, spec: ModelSpec) -> dict:
    """OpenAI 호환 업스트림용 embeddings payload (Gemini·Ollama 공용)."""
    payload: dict = {"model": spec.upstream, "input": request.input}
    for key in _EMBED_FORWARD_PARAMS:
        value = getattr(request, key)
        if value is not None:
            payload[key] = value
    return payload
