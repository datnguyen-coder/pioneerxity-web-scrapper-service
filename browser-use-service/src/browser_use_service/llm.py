"""LLM provider factory (minimal)."""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use.llm.openai.chat import ChatOpenAI as BrowserUseOpenAI
from langchain_ollama import ChatOllama


class BrowserUseChatOllama(ChatOllama):
    model_config = ConfigDict(extra="allow")
    provider: str = "ollama"

from .config import Settings


def build_llm(
    *,
    provider: str,
    model: Optional[str],
    temperature: float,
    settings: Settings,
) -> BaseChatModel:
    provider = (provider or "openai").lower()
    if provider == "ollama":
        return BrowserUseChatOllama(
            model=model or "qwen2.5:7b",
            temperature=temperature,
            base_url=settings.ollama_endpoint,
            num_ctx=16000,
        )
    return BrowserUseOpenAI(
        model=model or "gpt-4o",
        temperature=temperature,
        base_url=settings.openai_base_url or "https://api.openai.com/v1",
        api_key=settings.openai_api_key,
    )


