"""어댑터 공용 SSE 헬퍼 (순환 import 방지를 위해 별도 모듈로 분리)."""

import json


def sse_payloads(raw: str) -> list[str]:
    """
    내부 파이프라인이 내보내는 OpenAI SSE 문자열("data: {...}\\n\\n")에서 data 페이로드만 추출.
    한 청크에 여러 data 라인이 들어올 수 있어 라인 단위로 모은다. "[DONE]" 도 그대로 반환한다.
    """
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            out.append(line[len("data:") :].strip())
    return out


def format_event(event: str, data: dict) -> str:
    """이름 있는 SSE 이벤트(Anthropic 스타일): `event: <name>\\ndata: <json>\\n\\n`."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_data(data: dict) -> str:
    """이름 없는 SSE 데이터(Gemini/OpenAI 스타일): `data: <json>\\n\\n`."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
