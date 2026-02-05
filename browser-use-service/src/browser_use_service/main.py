"""FastAPI entrypoint for BrowserUse service."""

from __future__ import annotations

import json
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .agent_runner import run_agent_stream
from .config import RunRequest, get_settings

load_dotenv()

app = FastAPI(title="BrowserUse Service", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/browser-use/run/stream")
async def run_stream(
    request: Request,
    payload: RunRequest,
    x_gateway_key: str | None = Header(default=None, alias="X-Gateway-Key"),
) -> StreamingResponse:
    settings = get_settings()
    if settings.internal_api_key and x_gateway_key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="invalid_gateway_key")

    async def event_stream() -> AsyncIterator[bytes]:
        async for chunk in run_agent_stream(req=payload, settings=settings):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9010)


