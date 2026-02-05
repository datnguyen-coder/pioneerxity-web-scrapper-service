"""Async rate limiting utilities (per-domain politeness).

This module enforces a minimum interval between requests to the same domain.
It is intentionally simple and designed for infrastructure safety, not maximum throughput.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict
from urllib.parse import urlparse


def _now() -> float:
    return time.monotonic()


@dataclass
class _DomainState:
    lock: asyncio.Lock
    next_allowed_at: float


class DomainRateLimiter:
    """Per-domain rate limiter (async).

    Args:
        requests_per_second: Allowed RPS per domain. If <= 0, no limiting is applied.
    """

    def __init__(self, requests_per_second: float):
        self._rps = float(requests_per_second)
        self._states: Dict[str, _DomainState] = {}
        self._global_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._rps > 0

    def _min_interval_s(self) -> float:
        return 1.0 / self._rps if self._rps > 0 else 0.0

    async def _get_state(self, domain: str) -> _DomainState:
        async with self._global_lock:
            state = self._states.get(domain)
            if state is None:
                state = _DomainState(lock=asyncio.Lock(), next_allowed_at=0.0)
                self._states[domain] = state
            return state

    async def wait_for_slot(self, url: str) -> None:
        """Wait until it is allowed to perform a request to the given URL's domain."""
        if not self.enabled:
            return

        parsed = urlparse(url)
        domain = (parsed.netloc or "").lower()
        if not domain:
            return

        state = await self._get_state(domain)
        async with state.lock:
            now = _now()
            if state.next_allowed_at > now:
                await asyncio.sleep(state.next_allowed_at - now)

            state.next_allowed_at = _now() + self._min_interval_s()


