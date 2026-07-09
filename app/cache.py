"""
응답 캐시 (P1) — exact-match + TTL.

목적은 응답 속도가 아니라 **무료 티어 쿼터 절약** — 에이전트가 동일 요청을
재시도/반복할 때 업스트림을 다시 때리지 않는다. 시맨틱 캐시는 이 규모에선 과함.

적용 조건(모두 만족해야 캐시 대상):
  - 비스트리밍 (/v1/chat/completions 만 — 네이티브 엔드포인트는 패스스루 보존 우선)
  - temperature 미지정 또는 0 — 그 외 값은 매 호출 다른 응답이 기대되므로 부적합.
    미지정도 캐시하는 이유: 에이전트 재시도는 '같은 답이면 충분'한 경우가 대부분이고,
    '재생성' 의도의 동일 요청이 낡은 답을 받는 창은 짧은 TTL로 제한한다.

단일 프로세스 인메모리(OrderedDict LRU). CACHE_TTL_SECONDS=0 으로 비활성화.
"""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict

from app.models import ChatCompletionRequest

# 캐시 엔트리 수 상한. 초과 시 가장 오래 안 쓰인 것부터 제거(LRU).
MAX_ENTRIES = 256


def cache_key_for(request: ChatCompletionRequest) -> str | None:
    """
    캐시 가능한 요청이면 결정적 키(sha256)를, 아니면 None을 반환한다.
    키는 요청 전체(model·messages·tools·파라미터·미지 필드 포함)의 정렬 JSON 해시 —
    어떤 필드든 다르면 다른 키가 되어 오염(잘못된 히트)이 없다.
    """
    if request.stream:
        return None
    if request.temperature not in (None, 0, 0.0):
        return None
    dumped = request.model_dump(exclude_none=True)
    raw = json.dumps(dumped, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    """TTL + LRU 응답 캐시. 값은 업스트림 응답 body(dict) 그대로 저장한다."""

    def __init__(self, ttl_seconds: float, max_entries: int = MAX_ENTRIES) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        # key → (만료 시각 monotonic, 응답 body)
        self._store: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        # (E-15) 진행 중인 동일 요청의 계산을 공유하기 위한 single-flight future 맵.
        self._inflight: dict[str, asyncio.Future] = {}

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def get(self, key: str) -> dict | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, body = entry
        if time.monotonic() >= expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)  # LRU 갱신
        return body

    def put(self, key: str, body: dict) -> None:
        if not self.enabled:
            return
        self._store[key] = (time.monotonic() + self._ttl, body)
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)  # 가장 오래 안 쓰인 엔트리 제거

    # ── single-flight (E-15) — 동일 요청 동시 다발 시 업스트림 중복 호출(스탬피드) 방지 ──
    def begin(self, key: str) -> tuple[bool, asyncio.Future]:
        """
        동일 키가 이미 계산 중이면 (False, 그 future)를 반환해 follower가 결과를 기다리게 하고,
        없으면 (True, 새 future)를 반환해 leader가 계산하도록 한다. leader는 계산 후 settle 필수.
        단일 프로세스 코루틴 전제(읽기-쓰기 사이 await 없음 → 락 불필요).
        """
        fut = self._inflight.get(key)
        if fut is not None:
            return False, fut
        fut = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        return True, fut

    def settle(
        self, key: str, result: dict | None = None, exc: BaseException | None = None
    ) -> None:
        """leader가 계산을 마치면 대기 중인 follower들을 결과(또는 예외)로 깨운다."""
        fut = self._inflight.pop(key, None)
        if fut is None or fut.done():
            return
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(result)
