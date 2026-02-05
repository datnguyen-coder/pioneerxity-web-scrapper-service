"""Ollama adapter (Phase 2).

Uses Ollama HTTP API: POST /api/generate
This must remain internal-only (same host/network).
"""

from __future__ import annotations

import asyncio

import aiohttp

from ..domain.errors import NetworkTimeoutError, ContentProcessingError
from ..observability.logger import get_logger
from .runtime import LLMRequest, LLMRuntime

logger = get_logger(__name__)


class OllamaAdapter(LLMRuntime):
    def __init__(self, *, host: str, port: int):
        self._base_url = f"http://{host}:{port}"

    async def complete(self, req: LLMRequest) -> str:
        payload = {
            "model": req.model,
            "prompt": req.prompt,
            "stream": False,
            # Force JSON output for deterministic parsing in Phase 2.
            # Ollama supports `format: "json"` for /api/generate.
            "format": "json",
            "options": {
                "temperature": float(req.temperature),
                # Ollama uses `num_predict` for max tokens.
                "num_predict": int(req.max_tokens),
            },
        }

        timeout = aiohttp.ClientTimeout(total=max(1, int(req.timeout_seconds)))
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self._base_url}/api/generate", json=payload) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        raise ContentProcessingError(
                            "ollama_request_failed",
                            detail=f"status={resp.status} body={body[:500]}",
                        )
                    data = await resp.json()
                    if isinstance(data, dict) and data.get("error"):
                        raise ContentProcessingError("ollama_error", detail=str(data.get("error")))
                    text = data.get("response", "")
                    if not isinstance(text, str):
                        raise ContentProcessingError("ollama_response_invalid")
                    return text
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            raise NetworkTimeoutError("ollama_timeout_or_network_error", detail=str(e)) from e


