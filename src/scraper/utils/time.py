"""Time utilities."""

from __future__ import annotations

import time


def current_time_ms() -> int:
    return int(time.time() * 1000)


def elapsed_ms(start_time_ms: int) -> int:
    return max(0, current_time_ms() - start_time_ms)


