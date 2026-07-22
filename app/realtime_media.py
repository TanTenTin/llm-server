"""
로컬 Realtime 파이프라인의 미디어 어댑터 — VAD·STT·TTS를 CPU에서 돌린다.

이 모듈은 무거운 ML 의존성(numpy·torch·faster-whisper·MeloTTS·silero-vad)을
**한곳에 격리**한다. LocalRealtimeBridge가 로컬 세션이 처음 열릴 때만 지연 import
하므로, 기본 게이트웨이(텍스트/Gemini 경로) 기동에는 이 파일이 로드되지 않는다.

세 어댑터는 앱 생애주기 동안 1회 로드되는 싱글턴이다(모델 적재 비용이 크므로).
모델은 CPU에 상주하고, GPU(RTX 3060 12GB)는 qwen3:14b가 독점하도록 둔다 —
STT/TTS를 같은 GPU에 올리면 VRAM 경합으로 둘 다 느려진다.

의존성이 설치돼 있지 않으면 RealtimeMediaUnavailable을 던져, 브리지가 클라이언트에
'로컬 음성 미설치' 에러 이벤트로 정직하게 알리고 세션을 닫게 한다(조용한 실패 금지).
"""
import logging
from typing import Optional

from app.audio import resample_pcm16
from app.config import settings

logger = logging.getLogger("llm_gateway")

# Silero VAD가 16kHz에서 요구하는 창 크기(샘플). v5 기준 고정값.
_VAD_WINDOW = 512
_VAD_RATE = 16000


class RealtimeMediaUnavailable(RuntimeError):
    """로컬 음성 의존성(faster-whisper·MeloTTS·silero-vad) 미설치/로드 실패."""


# ── 지연 로드 싱글턴 ────────────────────────────────────────────
_stt_model = None          # faster_whisper.WhisperModel
_tts_model = None          # melo.api.TTS
_tts_speaker_id = None     # MeloTTS 화자 id
_tts_rate: int = 44100     # MeloTTS 출력 샘플레이트(로드 시 실제 값으로 갱신)
_vad_model = None          # silero_vad 모델(핸들)


def _require(module: str):
    """무거운 의존성을 import하되, 없으면 설치 안내를 담아 RealtimeMediaUnavailable로 바꾼다."""
    try:
        return __import__(module)
    except Exception as e:  # ImportError 외에 네이티브 로드 실패도 포함
        raise RealtimeMediaUnavailable(
            f"로컬 음성 의존성 '{module}' 로드 실패: {e!r} "
            f"(requirements-realtime.txt 설치 필요)"
        ) from e


def _get_stt():
    """
    faster-whisper STT 모델(싱글턴).
    기본은 CPU int8(VRAM 절약 — GPU는 LLM이 독점). 설정으로 cuda를 지정하면
    float16으로 GPU에 올리고, 로드 실패(드라이버·VRAM 부족) 시 CPU로 폴백한다.
    """
    global _stt_model
    if _stt_model is None:
        _require("faster_whisper")
        from faster_whisper import WhisperModel
        size = settings.realtime_local_stt_model
        device = settings.realtime_local_stt_device
        if device == "cuda":
            try:
                logger.info("[local-live] STT 로드: faster-whisper %s (cuda/float16)", size)
                _stt_model = WhisperModel(size, device="cuda", compute_type="float16")
                return _stt_model
            except Exception as e:
                logger.warning("[local-live] STT cuda 로드 실패 — cpu로 폴백: %r", e)
        logger.info("[local-live] STT 로드: faster-whisper %s (cpu/int8)", size)
        _stt_model = WhisperModel(size, device="cpu", compute_type="int8")
    return _stt_model


def _get_tts():
    """MeloTTS 모델(싱글턴). 화자 id와 출력 레이트를 함께 확정한다."""
    global _tts_model, _tts_speaker_id, _tts_rate
    if _tts_model is None:
        _require("melo")
        from melo.api import TTS
        lang = settings.realtime_local_tts_language
        logger.info("[local-live] TTS 로드: MeloTTS %s (cpu)", lang)
        model = TTS(language=lang, device="cpu")
        speakers = model.hps.data.spk2id
        # 언어 코드와 같은 이름의 화자를 우선, 없으면 첫 화자를 쓴다.
        _tts_speaker_id = speakers.get(lang, next(iter(speakers.values())))
        _tts_rate = int(model.hps.data.sampling_rate)
        _tts_model = model
    return _tts_model


def _load_vad_model():
    """Silero VAD 모델(싱글턴)."""
    global _vad_model
    if _vad_model is None:
        _require("silero_vad")
        from silero_vad import load_silero_vad
        logger.info("[local-live] VAD 로드: silero-vad")
        _vad_model = load_silero_vad()
    return _vad_model


# ── VAD 게이트(세션 단위 상태) ──────────────────────────────────
class VadGate:
    """
    16kHz PCM16 스트림을 받아 발화 시작/종료 이벤트를 뽑는 세션 단위 상태기.

    Silero VADIterator에 침묵 임계(min_silence_duration_ms)를 위임하므로, 브리지는
    'start'/'end' 이벤트만 소비하면 된다. VADIterator는 정확히 512샘플 창을 요구해
    입력 청크를 창 단위로 잘라 먹인다(남는 꼬리는 다음 feed로 이월).
    """

    def __init__(self) -> None:
        model = _load_vad_model()
        from silero_vad import VADIterator  # _load_vad_model이 import 성공을 보장
        import numpy as np
        self._np = np
        self._iter = VADIterator(
            model,
            threshold=0.5,
            sampling_rate=_VAD_RATE,
            min_silence_duration_ms=settings.realtime_local_vad_silence_ms,
        )
        self._buf = np.zeros(0, dtype=np.float32)  # 창 정렬용 잔여 버퍼

    def feed(self, pcm16_16k: bytes) -> list[str]:
        """16kHz PCM16 바이트를 먹이고 발생한 이벤트 목록(["start"|"end", ...])을 돌려준다."""
        import torch
        np = self._np
        samples = np.frombuffer(pcm16_16k, dtype="<i2").astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, samples])

        events: list[str] = []
        while len(self._buf) >= _VAD_WINDOW:
            window = self._buf[:_VAD_WINDOW]
            self._buf = self._buf[_VAD_WINDOW:]
            out = self._iter(torch.from_numpy(window), return_seconds=False)
            if out:
                if "start" in out:
                    events.append("start")
                if "end" in out:
                    events.append("end")
        return events

    def reset(self) -> None:
        """턴 사이에 VADIterator 내부 상태를 초기화한다(다음 발화를 깨끗이 감지)."""
        self._iter.reset_states()
        self._buf = self._np.zeros(0, dtype=self._np.float32)


# ── STT / TTS ───────────────────────────────────────────────────
def transcribe(pcm16_16k: bytes) -> str:
    """
    16kHz PCM16 발화 전체를 텍스트로 전사한다(블로킹 — 호출부에서 to_thread로 감쌀 것).
    언어를 고정해 자동감지 오판·언어 튐을 막는다. 짧은 발화면 beam_size=1로 지연을 줄인다.
    """
    import numpy as np
    model = _get_stt()
    audio = np.frombuffer(pcm16_16k, dtype="<i2").astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio, language=settings.realtime_local_stt_language, beam_size=1
    )
    return "".join(seg.text for seg in segments).strip()


def synthesize(text: str, out_rate: int) -> bytes:
    """
    텍스트 한 문장을 PCM16(mono, out_rate)로 합성한다(블로킹 — to_thread로 감쌀 것).
    MeloTTS는 float32 파형을 모델 레이트(보통 44.1kHz)로 내므로, PCM16 변환 후
    out_rate(클라이언트 기대 레이트, 보통 24kHz)로 리샘플한다.
    """
    import numpy as np
    model = _get_tts()
    # tts_to_file(output_path=None)은 float32 파형 ndarray를 반환한다(파일 미기록).
    wav = model.tts_to_file(
        text, _tts_speaker_id, output_path=None,
        speed=settings.realtime_local_tts_speed, quiet=True,
    )
    wav = np.asarray(wav, dtype=np.float32)
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    if _tts_rate != out_rate:
        pcm16 = resample_pcm16(pcm16, _tts_rate, out_rate)
    return pcm16
