from __future__ import annotations

from dataclasses import dataclass
import threading
from time import monotonic


@dataclass(slots=True)
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    def __init__(self, requests_per_minute: int, burst: int):
        self._capacity = float(max(1, burst))
        self._refill_rate = float(requests_per_minute) / 60.0
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        now = monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self._capacity, last_refill=now)
                self._buckets[key] = bucket

            elapsed = max(0.0, now - bucket.last_refill)
            bucket.last_refill = now
            bucket.tokens = min(self._capacity, bucket.tokens + (elapsed * self._refill_rate))

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0

            if self._refill_rate <= 0.0:
                return False, 60.0
            retry_after = (1.0 - bucket.tokens) / self._refill_rate
            return False, max(0.0, retry_after)

