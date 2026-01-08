# app/services/stt_store.py
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class _CallState:
    texts: List[Tuple[float, str]]  # (ts, text)
    updated_at: float


class STTBufferStore:
    """
    call_id 별로 최근 N개의 STT chunk를 누적하는 in-memory store.
    TTL 지나면 자동 제거(접근 시 정리).
    """
    def __init__(self, ttl_sec: int = 60 * 60, max_keep: int = 20):
        self.ttl_sec = ttl_sec
        self.max_keep = max_keep
        self._lock = asyncio.Lock()
        self._data: Dict[str, _CallState] = {}

    async def add_text(self, call_id: str, text: str) -> None:
        now = time.time()
        async with self._lock:
            self._gc_locked(now)

            st = self._data.get(call_id)
            if st is None:
                st = _CallState(texts=[], updated_at=now)
                self._data[call_id] = st

            st.texts.append((now, text))
            st.updated_at = now

            # 너무 많이 쌓이면 앞쪽 제거
            if len(st.texts) > self.max_keep:
                st.texts = st.texts[-self.max_keep :]

    async def get_last_texts(self, call_id: str, n: int) -> List[str]:
        now = time.time()
        async with self._lock:
            self._gc_locked(now)
            st = self._data.get(call_id)
            if not st:
                return []
            return [t for _, t in st.texts[-n:]]

    def _gc_locked(self, now: float) -> None:
        # TTL 지난 call 제거
        dead = [cid for cid, st in self._data.items() if (now - st.updated_at) > self.ttl_sec]
        for cid in dead:
            self._data.pop(cid, None)
