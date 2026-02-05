"""FastAPI app (internal-only).

This service is gRPC-first; FastAPI is used for lightweight health checks and
internal operational endpoints (no public UI).
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import aiohttp
import uuid
from typing import Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config.settings import get_settings
from .domain.errors import InvalidInputError
from .domain.models import ErrorCode, ProgressEvent
from .grpc_server import app_state
from .services.agent_service import AgentConstraints
from .utils.validators import is_valid_http_url

app = FastAPI(title="Web Scraper Service", version="0.1.0")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/api/v1/browser-use/run/stream")
async def browser_use_stream(payload: dict) -> StreamingResponse:
    settings = get_settings()
    service_url = settings.browser_use_service_base_url.rstrip("/")
    if not service_url:
        raise HTTPException(status_code=400, detail="browser_use_service_base_url_missing")

    headers = {"Content-Type": "application/json"}
    if settings.browser_use_internal_api_key:
        headers["X-Gateway-Key"] = settings.browser_use_internal_api_key

    timeout = aiohttp.ClientTimeout(total=float(settings.browser_use_timeout_seconds))
    target_url = f"{service_url}/api/v1/browser-use/run/stream"

    async def event_stream() -> AsyncIterator[bytes]:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(target_url, json=payload, headers=headers) as response:
                if response.status >= 400:
                    detail = await response.text()
                    error_payload = {
                        "type": "error",
                        "error": f"{response.status}: {detail}",
                        "isComplete": True,
                    }
                    yield f"data: {json.dumps(error_payload)}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"
                    return
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _serialize_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


class Phase3ScrapeRequest(BaseModel):
    baseUrl: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    maxDepth: int = Field(default=2, ge=1, le=10)
    maxPages: int = Field(default=50, ge=1, le=5000)
    allowedDomains: list[str] = []
    excludePatterns: list[str] = []
    llmProvider: str = "openai"
    llmModel: Optional[str] = None
    outputFormat: str = "pdf"
    includeImages: bool = False
    imageHandling: str = "embed_or_link"
    includeMetadata: bool = True
    noiseSelectors: list[str] = []


def _serialize_progress(ev: ProgressEvent) -> dict:
    return {
        "type": "progress",
        "jobId": ev.job_id,
        "stage": ev.stage.value,
        "message": ev.message,
        "currentUrl": ev.current_url,
        "currentIndex": ev.current_index,
        "totalDocuments": ev.total_documents,
        "isComplete": ev.is_complete,
        "success": ev.success,
        "errorCode": ev.error_code.value if ev.error_code else "",
        "errorMessage": ev.error_message,
        "lastDocument": ev.last_document.__dict__ if ev.last_document else None,
        "stats": ev.stats.__dict__ if ev.stats else None,
    }


@app.post("/api/v1/phase3/scrape/stream")
async def phase3_scrape_stream(payload: Phase3ScrapeRequest) -> StreamingResponse:
    if not is_valid_http_url(payload.baseUrl):
        raise HTTPException(status_code=400, detail="invalid_base_url")
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt_required")
    agent_service = app_state.get("agent_service")
    if agent_service is None:
        raise HTTPException(status_code=503, detail="agent_service_unavailable")

    settings = get_settings()
    constraints = AgentConstraints(
        max_depth=payload.maxDepth,
        max_pages=payload.maxPages,
        allowed_domains=tuple(payload.allowedDomains or ()),
        exclude_patterns=tuple(payload.excludePatterns or ()),
        llm_provider=(payload.llmProvider or settings.llm_provider),
        llm_model=payload.llmModel or None,
    )
    scraping_options = {
        "output_format": payload.outputFormat,
        "include_images": payload.includeImages,
        "image_handling": payload.imageHandling,
        "noise_selectors": payload.noiseSelectors or settings.default_noise_selectors,
        "include_metadata": payload.includeMetadata,
    }
    job_id = str(uuid.uuid4())
    client_id = "platform-browser-agent"

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            async for ev in agent_service.scrape_phase3(
                job_id=job_id,
                client_id=client_id,
                base_url=payload.baseUrl,
                prompt=payload.prompt,
                constraints=constraints,
                scraping_options=scraping_options,
            ):
                yield _serialize_event(_serialize_progress(ev))
                if ev.is_complete:
                    break
        except InvalidInputError as exc:
            payload_err = {
                "type": "progress",
                "jobId": job_id,
                "stage": "FAILED",
                "message": "invalid_input",
                "isComplete": True,
                "success": False,
                "errorCode": ErrorCode.INVALID_INPUT.value,
                "errorMessage": str(exc),
            }
            yield _serialize_event(payload_err)
        except Exception as exc:
            payload_err = {
                "type": "progress",
                "jobId": job_id,
                "stage": "FAILED",
                "message": "unexpected_error",
                "isComplete": True,
                "success": False,
                "errorCode": ErrorCode.INTERNAL_ERROR.value,
                "errorMessage": str(exc),
            }
            yield _serialize_event(payload_err)
        finally:
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


