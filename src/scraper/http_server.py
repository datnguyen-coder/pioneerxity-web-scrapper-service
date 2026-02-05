"""FastAPI server runner."""

from __future__ import annotations

import asyncio

import uvicorn

from .config.settings import get_settings
from .http_app import app
from .observability.logger import get_logger

logger = get_logger(__name__)


async def run_http_server() -> None:
    settings = get_settings()
    if not settings.http_enable:
        return

    config = uvicorn.Config(
        app=app,
        host=settings.http_host,
        port=settings.http_port,
        log_level="warning",  # structlog is the primary logger
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    logger.info("http_server_started", address=f"http://{settings.http_host}:{settings.http_port}")
    await server.serve()


