"""
PCM16 오디오 유틸 — Realtime 브리지에서 클라이언트(OpenAI 24kHz)와
Gemini Live(입력 16kHz) 간 샘플레이트를 맞추기 위한 리샘플러.

추가 의존성(numpy/scipy/audioop) 없이 동작하도록 순수 파이썬 선형보간으로 구현한다.
음성 대역에는 충분한 품질이며, 실시간 청크(수십 ms) 단위 비용도 무시할 만하다.

주의:
- PCM16 little-endian, mono 가정. (배포 대상 x86/ARM 모두 LE라 array 'h' 네이티브 순서와 일치)
- 다운샘플 시 안티앨리어싱 필터를 적용하지 않는 단순 선형보간이라 고역에 약간의 앨리어싱이
  생길 수 있으나, 음성 명료도에는 영향이 미미하다.
- 청크 단위로 독립 리샘플하므로 청크 경계의 보간 연속성은 보장하지 않는다(실용상 무시 가능).
"""
import array


def resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """
    PCM16 mono 바이트열을 src_rate → dst_rate로 선형보간 리샘플한다.
    레이트가 같으면 입력을 그대로 반환한다(복사 없음).
    """
    if src_rate == dst_rate or not data:
        return data

    # 바이트열 → 16비트 정수 샘플 배열
    samples = array.array("h")
    samples.frombytes(data)
    n_in = len(samples)
    if n_in == 0:
        return data

    n_out = max(1, int(n_in * dst_rate / src_rate))
    out = array.array("h", bytes(2 * n_out))
    ratio = src_rate / dst_rate

    for i in range(n_out):
        pos = i * ratio
        idx = int(pos)
        frac = pos - idx
        s0 = samples[idx] if idx < n_in else samples[n_in - 1]
        s1 = samples[idx + 1] if idx + 1 < n_in else s0
        # 선형보간 후 int16 범위로 클램프
        value = int(s0 + (s1 - s0) * frac)
        if value > 32767:
            value = 32767
        elif value < -32768:
            value = -32768
        out[i] = value

    return out.tobytes()
