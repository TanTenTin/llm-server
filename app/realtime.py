"""
Realtime 음성 브리지 — OpenAI Realtime API(WebSocket) 호환 엔드포인트를
Gemini Live API(네이티브 WSS)로 양방향 중계한다.

설계 요지:
- 클라이언트는 OpenAI Realtime 이벤트(JSON 텍스트 프레임)를 주고받는다.
  내부에서 Gemini Live 프로토콜로 변환해 중계하므로, 호출 측은 provider를 몰라도 된다.
- 오디오: 클라이언트 입력(PCM16 24kHz) → Gemini 입력(16kHz)로 리샘플,
  Gemini 출력(24kHz) → 클라이언트 출력(24kHz)는 그대로 패스스루.
- 두 펌프 코루틴(client→gemini, gemini→client)을 동시에 돌리고,
  한쪽이 끝나면 다른 쪽을 취소한 뒤 양쪽 연결을 정리한다.
- 텍스트 추론 경로(/v1/chat/completions)의 ProviderPool/회로차단기/fallback과는
  독립적이다. Live는 로컬 대체가 없어 폴백 개념이 성립하지 않는다.

미구현/한계(v1):
- toolCall(함수 호출)은 변환하지 않고 로그만 남긴다.
- session.update의 model 변경은 setup 전송 전에만 반영된다(Gemini setup은 1회성).
- 이미지/비디오 입력은 다루지 않는다(오디오·텍스트만).
"""
import asyncio
import base64
import json
import logging
from itertools import count
from typing import Any, Optional, Protocol

import websockets
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.audio import resample_pcm16
from app.registry import resolve_live_model

logger = logging.getLogger("llm_gateway")


class RealtimeBackend(Protocol):
    """
    Realtime 백엔드 공통 계약. 어떤 provider(gemini/local)든 이 하나만 만족하면
    /v1/realtime 디스패처가 동일하게 구동한다. 클라이언트(봇)는 OpenAI Realtime
    프로토콜만 보므로 내부가 클라우드인지 로컬인지 몰라도 된다.
    """
    async def run(self, client_ws: WebSocket) -> None: ...

# Gemini Live 네이티브 WebSocket 엔드포인트 (인증은 ?key= 쿼리 파라미터)
GEMINI_LIVE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
GEMINI_INPUT_RATE = 16000   # Gemini Live가 요구하는 입력 PCM16 레이트(mono, LE)
GEMINI_OUTPUT_RATE = 24000  # Gemini Live 출력 레이트(OpenAI 호환 24kHz와 동일 → 패스스루)
WS_MAX_SIZE = 2 ** 24       # 업스트림 프레임 최대 크기(오디오 base64가 커질 수 있어 넉넉히)


class RealtimeBridge:
    """
    하나의 클라이언트 WebSocket 세션을 Gemini Live 세션으로 중계하는 브리지.
    인스턴스는 세션(연결) 단위로 생성된다 — 세션 상태(설정/응답 진행)를 담기 때문.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str,
        requested_model: Optional[str],
        client_input_rate: int,
    ) -> None:
        self.api_key = api_key
        self.default_model = default_model
        self.client_input_rate = client_input_rate
        # 세션 설정(OpenAI session 객체에 대응). 클라이언트 ?model= 을 초기값으로 둔다.
        self._session_config: dict[str, Any] = {}
        if requested_model:
            self._session_config["model"] = requested_model
        # Gemini setup은 첫 입력 시점에 1회만 보낸다(그 전 session.update 반영 위해 lazy).
        self._setup_sent = False
        # 현재 응답(턴)이 진행 중인지 — response.created/response.done 짝을 맞추기 위함
        self._response_active = False
        self._response_id: Optional[str] = None
        # id 발급 카운터들(단일 스레드 이벤트루프라 락 불필요)
        self._event_counter = count(1)
        self._response_counter = count(1)
        self._item_counter = count(1)

    # ── 진입점 ───────────────────────────────────────────────
    async def run(self, client_ws: WebSocket) -> None:
        await client_ws.accept()

        # Gemini 키가 없으면 Live 사용 불가 — 에러 이벤트 후 종료
        if not self.api_key:
            await self._emit(client_ws, "error", error={
                "type": "invalid_request_error",
                "message": "Realtime은 GOOGLE_AI_API_KEY 설정이 필요합니다.",
            })
            await self._close_quietly(client_ws)
            return

        # 세션 생성 통지(OpenAI 클라이언트는 보통 이 직후 session.update를 보낸다)
        await self._emit(client_ws, "session.created", session=self._session_object())

        uri = f"{GEMINI_LIVE_URL}?key={self.api_key}"
        try:
            async with websockets.connect(uri, max_size=WS_MAX_SIZE) as gemini_ws:
                c2g = asyncio.create_task(self._client_to_gemini(client_ws, gemini_ws))
                g2c = asyncio.create_task(self._gemini_to_client(client_ws, gemini_ws))
                # 한쪽이 끝나면(연결 종료/에러) 나머지를 취소
                done, pending = await asyncio.wait(
                    {c2g, g2c}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                # 끝난 펌프의 예외를 표면화(로그)
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        logger.warning("[realtime] 펌프 종료 예외: %r", exc)
        except Exception as e:
            logger.warning("[realtime] 브리지 오류: %r", e)
            await self._emit_quietly(client_ws, "error", error={
                "type": "server_error", "message": str(e),
            })
        finally:
            await self._close_quietly(client_ws)

    # ── Client → Gemini 펌프 ─────────────────────────────────
    async def _client_to_gemini(
        self, client_ws: WebSocket, gemini_ws: Any
    ) -> None:
        while True:
            try:
                raw = await client_ws.receive_text()
            except WebSocketDisconnect:
                return  # 클라이언트가 끊음 → 펌프 정상 종료
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await self._emit(client_ws, "error", error={
                    "type": "invalid_request_error", "message": "잘못된 JSON 프레임",
                })
                continue
            await self._handle_client_event(event, client_ws, gemini_ws)

    async def _handle_client_event(
        self, event: dict[str, Any], client_ws: WebSocket, gemini_ws: Any
    ) -> None:
        etype = event.get("type")

        if etype == "session.update":
            # 세션 설정 병합(instructions/voice/model 등). setup 전이면 다음 setup에 반영된다.
            session = event.get("session") or {}
            self._session_config.update(session)
            await self._emit(client_ws, "session.updated", session=self._session_object())

        elif etype == "input_audio_buffer.append":
            await self._ensure_setup(gemini_ws)
            audio_b64 = event.get("audio")
            if audio_b64:
                await self._forward_audio(audio_b64, gemini_ws)

        elif etype == "conversation.item.create":
            await self._ensure_setup(gemini_ws)
            text = self._extract_input_text(event.get("item") or {})
            if text:
                await gemini_ws.send(json.dumps({"realtimeInput": {"text": text}}))

        elif etype in ("input_audio_buffer.commit", "response.create",
                       "input_audio_buffer.clear"):
            # Gemini 자동 VAD가 턴 시작/종료를 감지하므로 별도 트리거 불필요(no-op).
            # setup만 보장해 둔다(아직 입력이 없었던 경우 대비).
            await self._ensure_setup(gemini_ws)

        else:
            logger.info("[realtime] 미처리 클라이언트 이벤트: %s", etype)

    async def _forward_audio(self, audio_b64: str, gemini_ws: Any) -> None:
        """
        클라이언트 입력 오디오(base64 PCM16)를 Gemini 입력 레이트로 리샘플해 전달.
        base64 디코드 → 리샘플 → base64 인코드 후 realtimeInput.audio 로 보낸다.
        """
        pcm = base64.b64decode(audio_b64)
        if self.client_input_rate != GEMINI_INPUT_RATE:
            pcm = resample_pcm16(pcm, self.client_input_rate, GEMINI_INPUT_RATE)
        payload = {
            "realtimeInput": {
                "audio": {
                    "data": base64.b64encode(pcm).decode("ascii"),
                    "mimeType": f"audio/pcm;rate={GEMINI_INPUT_RATE}",
                }
            }
        }
        await gemini_ws.send(json.dumps(payload))

    # ── Gemini → Client 펌프 ─────────────────────────────────
    async def _gemini_to_client(
        self, client_ws: WebSocket, gemini_ws: Any
    ) -> None:
        async for message in gemini_ws:
            # websockets는 텍스트(str)/바이너리(bytes)를 모두 줄 수 있다 → 통일해서 파싱
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("[realtime] Gemini 비-JSON 프레임 무시")
                continue
            await self._handle_gemini_message(data, client_ws)

    async def _handle_gemini_message(
        self, data: dict[str, Any], client_ws: WebSocket
    ) -> None:
        if "setupComplete" in data:
            logger.info("[realtime] Gemini setup 완료")
            return

        server_content = data.get("serverContent")
        if server_content:
            await self._handle_server_content(server_content, client_ws)

        if "toolCall" in data:
            # v1 미구현 — 함수 호출 변환은 추후 과제
            logger.info("[realtime] toolCall 수신(미처리): %s", data["toolCall"])

        if "error" in data or "goAway" in data:
            detail = data.get("error") or data.get("goAway")
            await self._emit(client_ws, "error", error={
                "type": "server_error", "message": json.dumps(detail, ensure_ascii=False),
            })

    async def _handle_server_content(
        self, sc: dict[str, Any], client_ws: WebSocket
    ) -> None:
        # 사용자 발화 전사 → OpenAI 입력 전사 델타
        input_tx = sc.get("inputTranscription")
        if input_tx and input_tx.get("text"):
            await self._emit(
                client_ws, "conversation.item.input_audio_transcription.delta",
                delta=input_tx["text"],
            )

        # 모델 발화 전사 → OpenAI 오디오 전사 델타
        output_tx = sc.get("outputTranscription")
        if output_tx and output_tx.get("text"):
            await self._emit(
                client_ws, "response.audio_transcript.delta", delta=output_tx["text"],
            )

        # 모델 응답 본문(오디오/텍스트 파트)
        model_turn = sc.get("modelTurn")
        if model_turn:
            await self._begin_response_if_needed(client_ws)
            for part in model_turn.get("parts", []):
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    # Gemini 출력 24kHz == OpenAI 출력 24kHz → 패스스루
                    await self._emit(
                        client_ws, "response.audio.delta", delta=inline["data"],
                    )
                text = part.get("text")
                if text:
                    await self._emit(
                        client_ws, "response.audio_transcript.delta", delta=text,
                    )

        # 사용자 끼어들기(barge-in) → 진행 중 응답 취소 통지
        if sc.get("interrupted"):
            await self._emit(client_ws, "input_audio_buffer.speech_started")
            await self._finish_response(client_ws, status="cancelled")

        # 턴 완료 → 응답 종료 통지
        if sc.get("turnComplete"):
            await self._finish_response(client_ws, status="completed")

    # ── 응답(턴) 경계 관리 ───────────────────────────────────
    async def _begin_response_if_needed(self, client_ws: WebSocket) -> None:
        if self._response_active:
            return
        self._response_active = True
        self._response_id = f"resp_{next(self._response_counter)}"
        await self._emit(client_ws, "response.created", response={
            "id": self._response_id, "status": "in_progress",
        })

    async def _finish_response(self, client_ws: WebSocket, status: str) -> None:
        if not self._response_active:
            return
        await self._emit(client_ws, "response.audio.done")
        await self._emit(client_ws, "response.audio_transcript.done")
        await self._emit(client_ws, "response.done", response={
            "id": self._response_id, "status": status,
        })
        self._response_active = False
        self._response_id = None

    # ── Gemini setup(1회성) ─────────────────────────────────
    async def _ensure_setup(self, gemini_ws: Any) -> None:
        if self._setup_sent:
            return
        self._setup_sent = True
        await gemini_ws.send(json.dumps(self._build_setup()))

    def _build_setup(self) -> dict[str, Any]:
        """현재 세션 설정으로 Gemini BidiGenerateContentSetup 메시지를 구성한다."""
        model = resolve_live_model(self._session_config.get("model"), self.default_model)
        generation_config: dict[str, Any] = {"responseModalities": ["AUDIO"]}

        voice = self._session_config.get("voice")
        if voice:
            generation_config["speechConfig"] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            }

        setup: dict[str, Any] = {
            "model": model,
            "generationConfig": generation_config,
            # 양방향 전사 활성화 — OpenAI의 입력/출력 transcript 이벤트로 매핑하기 위함
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }

        instructions = self._session_config.get("instructions")
        if instructions:
            setup["systemInstruction"] = {"parts": [{"text": instructions}]}

        return {"setup": setup}

    # ── 헬퍼 ─────────────────────────────────────────────────
    def _session_object(self) -> dict[str, Any]:
        """클라이언트에 돌려줄 OpenAI 호환 session 객체(최소 필드)."""
        return {
            "id": f"sess_{next(self._item_counter)}",
            "object": "realtime.session",
            "model": self._session_config.get("model") or self.default_model,
            "modalities": ["audio", "text"],
            "instructions": self._session_config.get("instructions", ""),
            "voice": self._session_config.get("voice"),
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
        }

    @staticmethod
    def _extract_input_text(item: dict[str, Any]) -> Optional[str]:
        """conversation.item.create 의 item에서 input_text를 추출한다."""
        content = item.get("content") or []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                return part.get("text")
        return None

    async def _emit(self, client_ws: WebSocket, type_: str, **fields: Any) -> None:
        """OpenAI Realtime 서버 이벤트 한 건을 클라이언트로 전송."""
        payload = {"type": type_, "event_id": f"evt_{next(self._event_counter)}", **fields}
        await client_ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def _emit_quietly(self, client_ws: WebSocket, type_: str, **fields: Any) -> None:
        """종료 경로에서 이벤트 전송 중 2차 예외가 정리를 막지 않도록 삼킨다."""
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
