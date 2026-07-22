"""
로컬 Realtime 브리지 — OpenAI Realtime API(WebSocket)를 완전 로컬 파이프라인
(VAD → STT → Ollama → TTS)으로 처리한다. 프롬프트·음성이 클라우드로 나가지 않는다.

밖으로는 Gemini 브리지(realtime.py)와 '똑같은' OpenAI Realtime 이벤트를 주고받으므로,
클라이언트(봇)는 model=local-live 로 바꾸는 것 말고는 코드를 바꿀 필요가 없다.
안에서는 Gemini Live의 네이티브 speech-to-speech 대신 세 단계를 직접 잇는다:

    사용자 음성 ─→ VAD(발화경계) ─→ [버퍼] ─→ Whisper(STT) ─→ 텍스트
                                                                  │
    사용자 ←─ PCM16 ←─ MeloTTS(TTS) ←─ 문장분할 ←─ Ollama.stream(LLM)

핵심 설계:
- turn-taking: Gemini는 서버 VAD가 턴을 잡지만, 로컬은 Silero VAD를 직접 돌려
  '발화 종료'를 감지하고 그 시점에 STT→LLM→TTS 턴을 트리거한다. 클라이언트가 보내는
  input_audio_buffer.commit 도 강제 트리거로 받는다(서버 VAD 대안).
  VAD가 발화를 못 잡았어도(조용한 화자 등) commit이 오면 최근 수신 오디오 버퍼로
  턴을 강행한다 — '봇이 아예 대답하지 않는' 실패 모드를 없앤다.
- 지연 완화 1(병렬 파이프라인): LLM 스트리밍 소비와 TTS 합성을 분리한다. LLM 델타는
  문장 큐에 쌓이고 별도 태스크가 순서대로 합성·전송한다 — 문장 N을 합성하는 동안에도
  LLM은 문장 N+1을 계속 생성한다(예전엔 문장마다 LLM 소비가 멈췄다).
- 지연 완화 2(조기 분할): 첫 세그먼트는 문장 종결부를 기다리지 않고 쉼표에서도 잘라
  첫 오디오가 나오는 시점을 앞당긴다(이후 세그먼트는 문장 단위 유지).
- barge-in: 응답 재생 중 새 발화가 감지되면 진행 중 응답 태스크를 취소하고
  response.done(cancelled)로 통지한다(Gemini 브리지의 interrupted 처리와 동일 매핑).
- TTS 전처리: LLM이 지시를 어기고 마크다운·이모지를 내도 합성 전에 걷어낸다
  (기호를 소리내어 읽는 사고 방지).
- CPU 블로킹(STT/TTS)은 asyncio.to_thread로 감싸 수신 루프를 막지 않는다.

한계(v1):
- 함수 호출(tool)·이미지/비디오 입력은 다루지 않는다.
- 에코 제거는 클라이언트(Discord)의 몫으로 둔다(봇은 사용자 마이크만 수신).
"""
import asyncio
import base64
import json
import logging
import re
from itertools import count
from typing import Any, AsyncGenerator, Optional

from starlette.websockets import WebSocket, WebSocketDisconnect

from app.audio import resample_pcm16
from app.config import settings
from app.models import ChatCompletionRequest, Message
from app.realtime_media import (
    RealtimeMediaUnavailable,
    VadGate,
    synthesize,
    transcribe,
)
from app.registry import LiveModelSpec, ModelSpec

logger = logging.getLogger("llm_gateway")

_VAD_RATE = 16000                     # STT/VAD가 다루는 내부 레이트
_HISTORY_TURNS = 10                   # LLM에 넣을 최근 대화 턴 수(과거 맥락)
_MIN_UTTERANCE_BYTES = 6400           # ~200ms(16k·PCM16) 미만 발화는 잡음으로 보고 버림
_AUDIO_FRAME_BYTES = 4096 * 2         # response.audio.delta 한 프레임(4096샘플·PCM16)

# 문장 종결부 — 여기서 끊어 TTS로 조기 전송한다(한국어/영문 공통).
_SENTENCE_ENDINGS = set(".!?。！？\n")

# 첫 세그먼트 조기 분할용 절 구분자와 최소 길이. 첫 오디오까지의 시간을 줄이기 위해
# 응답의 첫 조각만 쉼표에서도 자른다(너무 짧은 조각은 TTS 품질이 떨어져 하한을 둔다).
_CLAUSE_BREAKS = set(",，、")
_FIRST_CLAUSE_MIN_CHARS = 12

# commit 폴백용 최근 수신 오디오 상한(16k PCM16 30초). VAD가 발화를 놓친 경우의 안전망.
_RECENT_MAX_BYTES = _VAD_RATE * 2 * 30

# TTS 전처리 — 마크다운 구조 기호와 이모지·특수기호를 걷어낸다.
# LLM이 프롬프트 지시를 어겨도 "별표", "우물 정" 같은 소리가 나가지 않게 한다.
_TTS_STRIP_PATTERN = re.compile(
    r"[*_`#>|~\[\]{}<>=+/\\^]"           # 마크다운·구조 기호
    r"|[\U0001F000-\U0001FAFF]"          # 이모지 주요 블록
    r"|[☀-➿]"                  # 기타 기호·딩뱃(Misc Symbols·Dingbats)
    r"|[️‍]"                   # 이모지 변형 셀렉터·ZWJ
)


def _clean_for_tts(text: str) -> str:
    """TTS에 넣기 전 기호를 제거하고 공백을 정리한다. 내용이 없으면 빈 문자열."""
    cleaned = _TTS_STRIP_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


# 음성 대화용 기본 시스템 프롬프트. 음성은 텍스트처럼 길게 답하면 안 되므로 간결함을 강제한다.
_DEFAULT_SYSTEM = (
    "당신은 음성으로 대화하는 한국어 비서입니다. "
    "말하듯 자연스럽고 간결하게, 2~3문장 이내로 답하세요. "
    "목록·코드·마크다운 기호는 쓰지 마세요(소리내어 읽기 어렵습니다)."
)


def _pop_segments(buf: str, allow_clause: bool) -> tuple[list[str], str]:
    """
    누적 텍스트에서 완성된 문장들을 떼어내고 미완성 꼬리를 남긴다.
    종결부(., !, ?, 。, 줄바꿈)를 만날 때마다 한 문장으로 자른다.

    allow_clause=True(응답의 첫 조각이 아직 안 나간 상태)면 완성 문장이 없어도
    쉼표 절에서 한 번 조기 분할한다 — 첫 오디오가 나오는 시점을 문장 완성보다
    앞당긴다. 너무 짧은 절은 TTS 운율이 어색해 최소 길이를 넘는 절만 자른다.
    """
    segments: list[str] = []
    start = 0
    for i, ch in enumerate(buf):
        if ch in _SENTENCE_ENDINGS:
            segment = buf[start:i + 1].strip()
            if segment:
                segments.append(segment)
            start = i + 1
    rest = buf[start:]
    if allow_clause and not segments:
        for i, ch in enumerate(rest):
            if ch in _CLAUSE_BREAKS and i + 1 >= _FIRST_CLAUSE_MIN_CHARS:
                head = rest[:i + 1].strip()
                if head:
                    segments.append(head)
                rest = rest[i + 1:]
                break
    return segments, rest


def _sse_content(chunk: str) -> Optional[str]:
    """OllamaProvider.stream이 내는 SSE 한 줄에서 delta.content(평문)를 뽑는다."""
    line = chunk.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    choices = data.get("choices") or [{}]
    return choices[0].get("delta", {}).get("content")


class LocalRealtimeBridge:
    """하나의 클라이언트 WebSocket 세션을 로컬 음성 파이프라인으로 처리하는 브리지."""

    def __init__(
        self, spec: LiveModelSpec, pool: Any, client_input_rate: int
    ) -> None:
        self.spec = spec
        self.pool = pool                 # ProviderPool — Ollama 스트림 재사용에 필요
        self.client_input_rate = client_input_rate
        self.out_rate = settings.realtime_local_output_sample_rate

        self._session_config: dict[str, Any] = {}
        self._history: list[tuple[str, str]] = []  # (role, text) 대화 기록

        self._vad: Optional[VadGate] = None
        self._listening = False                    # 발화 수집 중인지(VAD start~end)
        self._utterance = bytearray()              # 현재 발화의 16k PCM16 누적
        # 마지막 턴 이후 수신한 모든 오디오(상한 30초). VAD가 발화를 놓쳐도
        # 클라이언트 commit이 오면 이 버퍼로 턴을 강행한다(무응답 방지 안전망).
        self._recent = bytearray()

        self._response_task: Optional[asyncio.Task] = None
        self._response_active = False
        self._response_id: Optional[str] = None

        self._event_counter = count(1)
        self._response_counter = count(1)
        self._item_counter = count(1)

    # ── 진입점 ───────────────────────────────────────────────
    async def run(self, client_ws: WebSocket) -> None:
        await client_ws.accept()

        # VAD 모델 로드(블로킹). 로컬 음성 의존성이 없으면 정직하게 알리고 종료.
        try:
            self._vad = await asyncio.to_thread(VadGate)
        except RealtimeMediaUnavailable as e:
            await self._emit(client_ws, "error", error={
                "type": "server_error", "message": str(e),
            })
            await self._close_quietly(client_ws)
            return

        await self._emit(client_ws, "session.created", session=self._session_object())

        try:
            while True:
                try:
                    raw = await client_ws.receive_text()
                except WebSocketDisconnect:
                    return
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    await self._emit(client_ws, "error", error={
                        "type": "invalid_request_error", "message": "잘못된 JSON 프레임",
                    })
                    continue
                await self._handle_client_event(event, client_ws)
        finally:
            await self._cancel_response()
            await self._close_quietly(client_ws)

    # ── 클라이언트 이벤트 처리 ───────────────────────────────
    async def _handle_client_event(
        self, event: dict[str, Any], client_ws: WebSocket
    ) -> None:
        etype = event.get("type")

        if etype == "session.update":
            session = event.get("session") or {}
            self._session_config.update(session)
            await self._emit(client_ws, "session.updated", session=self._session_object())

        elif etype == "input_audio_buffer.append":
            audio_b64 = event.get("audio")
            if audio_b64:
                await self._on_audio(audio_b64, client_ws)

        elif etype == "input_audio_buffer.commit":
            # 클라이언트가 턴 종료를 명시 → 서버 VAD 대신 강제 트리거.
            await self._commit_turn(client_ws)

        elif etype == "input_audio_buffer.clear":
            self._listening = False
            self._utterance = bytearray()
            self._recent = bytearray()
            if self._vad:
                self._vad.reset()

        elif etype == "conversation.item.create":
            # 텍스트 입력 경로(STT 건너뜀) — 채널에 글로 말을 걸 때.
            text = self._extract_input_text(event.get("item") or {})
            if text:
                self._history.append(("user", text))
                self._start_turn(client_ws, utterance=None, text=text)

        elif etype == "response.create":
            # 로컬은 VAD/commit로 턴을 구동하므로 별도 트리거 불필요(no-op).
            pass

        else:
            logger.info("[local-live] 미처리 클라이언트 이벤트: %s", etype)

    async def _on_audio(self, audio_b64: str, client_ws: WebSocket) -> None:
        """입력 오디오 청크를 16kHz로 맞춰 VAD에 먹이고 발화 경계 이벤트를 처리한다."""
        pcm = base64.b64decode(audio_b64)
        pcm16 = (
            resample_pcm16(pcm, self.client_input_rate, _VAD_RATE)
            if self.client_input_rate != _VAD_RATE else pcm
        )

        # commit 폴백용 누적(항상). 상한을 넘으면 앞부분부터 버린다.
        self._recent.extend(pcm16)
        if len(self._recent) > _RECENT_MAX_BYTES:
            del self._recent[:len(self._recent) - _RECENT_MAX_BYTES]

        assert self._vad is not None
        events = self._vad.feed(pcm16)

        for ev in events:
            if ev == "start":
                # barge-in: 재생 중이면 진행 응답을 취소하고 새 발화를 받는다.
                if self._response_active:
                    await self._cancel_response()
                await self._emit(client_ws, "input_audio_buffer.speech_started")
                self._listening = True
                self._utterance = bytearray()
            elif ev == "end":
                if self._listening:
                    await self._emit(client_ws, "input_audio_buffer.speech_stopped")
                    utter = bytes(self._utterance)
                    self._listening = False
                    self._utterance = bytearray()
                    self._recent = bytearray()
                    self._start_turn(client_ws, utterance=utter, text=None)

        # 발화 수집 중이면 이번 청크를 누적(start가 이번에 켜졌으면 이 청크부터 포함).
        if self._listening:
            self._utterance.extend(pcm16)

    async def _commit_turn(self, client_ws: WebSocket) -> None:
        """
        input_audio_buffer.commit — 수집 중인 발화를 즉시 턴으로 확정한다.
        VAD가 발화를 감지하지 못했으면(조용한 화자·짧은 발화) 마지막 턴 이후 수신한
        오디오 전체(_recent)로 턴을 강행한다 — 클라이언트가 턴 종료를 명시했는데
        아무 반응이 없는 실패 모드를 없앤다.
        """
        if self._listening:
            utter = bytes(self._utterance)
        else:
            utter = bytes(self._recent)
            if len(utter) < _MIN_UTTERANCE_BYTES:
                return  # 폴백으로도 쓸 만한 오디오가 없다(잡음 수준)
        await self._emit(client_ws, "input_audio_buffer.speech_stopped")
        self._listening = False
        self._utterance = bytearray()
        self._recent = bytearray()
        if self._vad:
            self._vad.reset()
        self._start_turn(client_ws, utterance=utter, text=None)

    # ── 턴(응답) 실행 ────────────────────────────────────────
    def _start_turn(
        self, client_ws: WebSocket, utterance: Optional[bytes], text: Optional[str]
    ) -> None:
        """응답 태스크를 스폰한다(취소 가능 — barge-in 대응). 이미 처리 중이면 무시."""
        if self._response_task and not self._response_task.done():
            return
        self._response_task = asyncio.create_task(
            self._run_turn(client_ws, utterance, text)
        )

    async def _run_turn(
        self, client_ws: WebSocket, utterance: Optional[bytes], text: Optional[str]
    ) -> None:
        """STT(필요 시) → LLM → 문장분할 → TTS 한 턴을 수행한다."""
        try:
            # 텍스트 입력이 아니면 발화를 전사한다. 너무 짧은 발화(잡음)는 버린다.
            if text is None:
                if utterance is None or len(utterance) < _MIN_UTTERANCE_BYTES:
                    return
                text = await asyncio.to_thread(transcribe, utterance)
                if not text:
                    return
                await self._emit(
                    client_ws, "conversation.item.input_audio_transcription.completed",
                    transcript=text,
                )
                self._history.append(("user", text))

            await self._generate(client_ws)
        except asyncio.CancelledError:
            raise  # barge-in — _generate가 cancelled 통지를 이미 냈다
        except RealtimeMediaUnavailable as e:
            await self._emit_quietly(client_ws, "error", error={
                "type": "server_error", "message": str(e),
            })
        except Exception as e:
            logger.warning("[local-live] 턴 처리 오류: %r", e)
            await self._emit_quietly(client_ws, "error", error={
                "type": "server_error", "message": str(e),
            })

    async def _generate(self, client_ws: WebSocket) -> None:
        """
        LLM 텍스트를 스트리밍받아 세그먼트(문장/첫 절) 단위로 TTS해 오디오 델타로 내보낸다.

        LLM 소비(producer)와 TTS 합성·전송(consumer)을 태스크로 분리한다 —
        문장 N을 합성하는 동안에도 LLM은 다음 문장을 계속 생성하므로, 체감 지연은
        '첫 세그먼트 TTS 시간'만 남고 이후 문장 간 갭은 LLM 속도와 무관해진다.
        """
        await self._begin_response(client_ws)
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        speaker = asyncio.create_task(self._speaker_loop(client_ws, queue))
        collected: list[str] = []
        pending = ""
        first_sent = False
        try:
            async for delta in self._llm_text():
                await self._emit(client_ws, "response.audio_transcript.delta", delta=delta)
                collected.append(delta)
                pending += delta
                segments, pending = _pop_segments(pending, allow_clause=not first_sent)
                for segment in segments:
                    first_sent = True
                    queue.put_nowait(segment)
            # 마지막 미완성 꼬리도 발화한다.
            tail = pending.strip()
            if tail:
                queue.put_nowait(tail)
            queue.put_nowait(None)  # 종료 신호 — 큐에 쌓인 세그먼트를 모두 말한 뒤 끝난다
            await speaker

            self._history.append(("assistant", "".join(collected)))
            await self._finish_response(client_ws, status="completed")
        except asyncio.CancelledError:
            # barge-in — 진행 중인 합성·전송도 즉시 멈춘다.
            speaker.cancel()
            try:
                await speaker
            except (asyncio.CancelledError, Exception):
                pass
            await self._finish_response(client_ws, status="cancelled")
            raise

    async def _speaker_loop(
        self, client_ws: WebSocket, queue: "asyncio.Queue[Optional[str]]"
    ) -> None:
        """세그먼트 큐를 순서대로 비우며 TTS·전송한다(None = 종료 신호)."""
        while True:
            segment = await queue.get()
            if segment is None:
                return
            await self._speak(client_ws, segment)

    async def _llm_text(self) -> AsyncGenerator[str, None]:
        """기존 OllamaProvider.stream을 재사용해 LLM 텍스트 델타(평문)를 흘린다."""
        model_id = self.spec.llm_model or "ollama/qwen3:14b"
        upstream = model_id.split("/", 1)[1] if model_id.startswith("ollama/") else model_id
        model_spec = ModelSpec(provider="ollama", upstream=upstream)

        # 음성 턴은 짧아야 한다 — 출력 상한으로 폭주 응답(수십 초 TTS 큐 점유)을 막는다.
        max_tokens = settings.realtime_local_llm_max_tokens
        messages = [Message(**m) for m in self._build_messages()]
        request = ChatCompletionRequest(
            model=model_id,
            messages=messages,
            stream=True,
            temperature=settings.realtime_local_llm_temperature,
            max_tokens=max_tokens if max_tokens > 0 else None,
        )
        provider = self.pool.get("ollama")
        async for chunk in provider.stream(request, model_spec):
            content = _sse_content(chunk)
            if content:
                yield content

    def _build_messages(self) -> list[dict[str, str]]:
        """시스템 프롬프트 + 최근 대화 턴으로 LLM 메시지를 구성한다."""
        system = self._session_config.get("instructions") or _DEFAULT_SYSTEM
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for role, text in self._history[-_HISTORY_TURNS:]:
            messages.append({"role": role, "content": text})
        return messages

    async def _speak(self, client_ws: WebSocket, sentence: str) -> None:
        """문장 하나를 TTS해 PCM16 프레임 단위로 response.audio.delta 전송."""
        # 마크다운·이모지를 걷어낸다 — 기호만 남은 세그먼트는 발화하지 않는다.
        sentence = _clean_for_tts(sentence)
        if not sentence:
            return
        pcm = await asyncio.to_thread(synthesize, sentence, self.out_rate)
        for i in range(0, len(pcm), _AUDIO_FRAME_BYTES):
            frame = pcm[i:i + _AUDIO_FRAME_BYTES]
            await self._emit(
                client_ws, "response.audio.delta",
                delta=base64.b64encode(frame).decode("ascii"),
            )

    # ── 응답(턴) 경계 관리 ───────────────────────────────────
    async def _begin_response(self, client_ws: WebSocket) -> None:
        self._response_active = True
        self._response_id = f"resp_{next(self._response_counter)}"
        await self._emit(client_ws, "response.created", response={
            "id": self._response_id, "status": "in_progress",
        })

    async def _finish_response(self, client_ws: WebSocket, status: str) -> None:
        if not self._response_active:
            return
        await self._emit_quietly(client_ws, "response.audio.done")
        await self._emit_quietly(client_ws, "response.audio_transcript.done")
        await self._emit_quietly(client_ws, "response.done", response={
            "id": self._response_id, "status": status,
        })
        self._response_active = False
        self._response_id = None

    async def _cancel_response(self) -> None:
        """진행 중 응답 태스크를 취소하고 정리(barge-in·세션 종료 공통)."""
        task = self._response_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._response_task = None

    # ── 헬퍼 ─────────────────────────────────────────────────
    def _session_object(self) -> dict[str, Any]:
        return {
            "id": f"sess_{next(self._item_counter)}",
            "object": "realtime.session",
            "model": self.spec.name,
            "modalities": ["audio", "text"],
            "instructions": self._session_config.get("instructions", ""),
            "voice": self._session_config.get("voice"),
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
        }

    @staticmethod
    def _extract_input_text(item: dict[str, Any]) -> Optional[str]:
        content = item.get("content") or []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                return part.get("text")
        return None

    async def _emit(self, client_ws: WebSocket, type_: str, **fields: Any) -> None:
        payload = {"type": type_, "event_id": f"evt_{next(self._event_counter)}", **fields}
        await client_ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def _emit_quietly(self, client_ws: WebSocket, type_: str, **fields: Any) -> None:
        try:
            await self._emit(client_ws, type_, **fields)
        except Exception:
            pass

    @staticmethod
    async def _close_quietly(client_ws: WebSocket) -> None:
        try:
            await client_ws.close()
        except Exception:
            pass
