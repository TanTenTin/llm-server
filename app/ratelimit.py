"""
게이트웨이 레이트리밋 (P1) — 키/클라이언트 단위 분당 요청 상한.

외부 노출 시 폭주(에이전트 루프·오남용)로부터 업스트림 쿼터와 과금을 보호한다.
고정 윈도우(분 단위) 카운터 — 슬라이딩 윈도우 대비 경계에서 최대 2배 순간 허용이
있지만, 단일 사용자 규모의 보호 목적엔 충분하고 메모리·연산이 상수다.
단일 프로세스 전제(코루틴 공유 dict, 읽기-쓰기 사이 await 없음 → 락 불필요).
RATE_LIMIT_RPM=0 이면 비활성화.
"""

import time


class RateLimiter:
    """분 단위 고정 윈도우 요청 카운터. identity는 키 해시 또는 클라이언트 IP."""

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._window_minute: int = -1          # 현재 윈도우의 분 epoch
        self._counts: dict[str, int] = {}      # identity → 이번 분의 요청 수

    @property
    def enabled(self) -> bool:
        return self._rpm > 0

    def allow(self, identity: str) -> bool:
        """요청 1건을 집계하고 상한 이내인지 반환한다."""
        if not self.enabled:
            return True
        minute = int(time.time() // 60)
        if minute != self._window_minute:
            self._window_minute = minute
            self._counts = {}                  # 새 분 → 윈도우 리셋
        self._counts[identity] = self._counts.get(identity, 0) + 1
        return self._counts[identity] <= self._rpm

    @staticmethod
    def seconds_until_reset() -> int:
        """다음 윈도우(다음 분)까지 남은 초 — 429 응답의 Retry-After 값."""
        return 60 - int(time.time()) % 60
