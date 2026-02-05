"""Config and request models for BrowserUse service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1)
    maxSteps: int = Field(default=20, ge=1, le=200)
    model: Optional[str] = None
    provider: str = "openai"  # openai | ollama
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    headless: bool = True
    baseUrl: Optional[str] = None


@dataclass(frozen=True)
class Settings:
    internal_api_key: Optional[str]
    openai_api_key: Optional[str]
    openai_base_url: Optional[str]
    ollama_endpoint: str


def get_settings() -> Settings:
    return Settings(
        internal_api_key=os.getenv("BROWSER_USE_INTERNAL_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        ollama_endpoint=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
    )


