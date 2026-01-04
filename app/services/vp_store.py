# app/services/vp_store.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import asyncio

@dataclass
class CallScores:
    scores: List[float] = field(default_factory=list)
    last_updated: float = field(default_factory=lambda: time.time())

class VoicePhishingStore:
    """
    5초마다 들어오는 mfcc phishing_score를 call_id별로 저장해두고,
    통화 종료 시(stt 호출 시점) 최종 점수/플래그를 계산해주는 간단 저장소.
    """
    def __init__(self, ttl_sec: int = 60 * 60):
        self.ttl_sec = ttl_sec
        self._data: Dict[str, CallScores] = {}
        self._lock = asyncio.Lock()
        self._last_call_id: str | None = None
        
    async def add_score(self, call_id: str, score: float) -> None:
        async with self._lock:
            self._last_call_id = call_id
            item = self._data.get(call_id)
            if item is None:
                item = CallScores()
                self._data[call_id] = item
            item.scores.append(float(score))
            item.last_updated = time.time()

    async def finalize(self, call_id: str) -> Tuple[bool, Optional[float], Dict]:
        """
        call_id에 쌓인 점수들을 집계해서 최종 (flag, score, debug_info) 반환 후 삭제.
        """
        async with self._lock:
            item = self._data.pop(call_id, None)

        if item is None or not item.scores:
            # mfcc 점수가 하나도 없으면 안전하게 False 처리(원하면 None 처리도 가능)
            return False, None, {"count": 0}

        scores = item.scores
        mean_score = sum(scores) / len(scores)
        max_score = max(scores)

        # 추천 집계: mean + max 혼합
        final_score = 0.7 * mean_score + 0.3 * max_score

        # 최종 플래그 룰
        flag = final_score >= 0.5

        debug = {
            "count": len(scores),
            "mean": mean_score,
            "max": max_score,
            "final": final_score,
        }
        return flag, float(final_score), debug

    async def cleanup(self) -> None:
        """
        혹시 통화 종료(stt)가 안 오거나 앱이 죽은 경우를 대비한 TTL 정리.
        """
        now = time.time()
        async with self._lock:
            stale = [cid for cid, v in self._data.items() if now - v.last_updated > self.ttl_sec]
            for cid in stale:
                self._data.pop(cid, None)
