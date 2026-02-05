"""LLM runtime interface (Phase 2).

The service must talk to LLMs through this interface to allow future adapters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int


class LLMRuntime:
    async def complete(self, req: LLMRequest) -> str:  # pragma: no cover - interface
        raise NotImplementedError


