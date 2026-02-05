"""OpenAI adapter (Phase 3).

Uses OpenAI API for LLM completions.
Supports GPT-4, GPT-4o, GPT-3.5-turbo, etc.
"""

from __future__ import annotations

import asyncio
import os

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore

import aiohttp

from ..domain.errors import NetworkTimeoutError, ContentProcessingError
from ..observability.logger import get_logger
from .runtime import LLMRequest, LLMRuntime

logger = get_logger(__name__)


class OpenAIAdapter(LLMRuntime):
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            base_url: Custom base URL (for OpenAI-compatible APIs). If None, uses OpenAI default.
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        self._base_url = base_url or "https://api.openai.com/v1"
        
        # Use official OpenAI client if available, otherwise fallback to HTTP
        self._use_official_client = AsyncOpenAI is not None
        if self._use_official_client:
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        else:
            logger.warning("openai package not installed. Using HTTP fallback. Install with: pip install openai")

    async def complete(self, req: LLMRequest) -> str:
        """Complete LLM request using OpenAI API."""
        if self._use_official_client:
            return await self._complete_with_client(req)
        else:
            return await self._complete_with_http(req)

    async def _complete_with_client(self, req: LLMRequest) -> str:
        """Use official OpenAI client."""
        try:
            response = await self._client.chat.completions.create(
                model=req.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds in JSON format when requested."},
                    {"role": "user", "content": req.prompt}
                ],
                temperature=float(req.temperature),
                max_tokens=int(req.max_tokens),
                timeout=max(1, int(req.timeout_seconds)),
            )
            text = response.choices[0].message.content or ""
            if not isinstance(text, str):
                raise ContentProcessingError("openai_response_invalid")
            return text
        except asyncio.TimeoutError as e:
            raise NetworkTimeoutError("openai_timeout", detail=str(e)) from e
        except Exception as e:
            raise ContentProcessingError("openai_request_failed", detail=str(e)) from e

    async def _complete_with_http(self, req: LLMRequest) -> str:
        """Fallback HTTP implementation for OpenAI API."""
        payload = {
            "model": req.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that responds in JSON format when requested."},
                {"role": "user", "content": req.prompt}
            ],
            "temperature": float(req.temperature),
            "max_tokens": int(req.max_tokens),
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=max(1, int(req.timeout_seconds)))
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        raise ContentProcessingError(
                            "openai_request_failed",
                            detail=f"status={resp.status} body={body[:500]}",
                        )
                    data = await resp.json()
                    if isinstance(data, dict) and data.get("error"):
                        raise ContentProcessingError("openai_error", detail=str(data.get("error")))
                    
                    # Extract text from OpenAI response format
                    choices = data.get("choices", [])
                    if not choices:
                        raise ContentProcessingError("openai_response_invalid", detail="no choices in response")
                    
                    message = choices[0].get("message", {})
                    text = message.get("content", "")
                    if not isinstance(text, str):
                        raise ContentProcessingError("openai_response_invalid")
                    return text
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            raise NetworkTimeoutError("openai_timeout_or_network_error", detail=str(e)) from e

