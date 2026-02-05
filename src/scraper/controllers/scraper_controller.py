"""gRPC Controller - Transport layer only.

Responsibilities:
- Convert gRPC requests to domain inputs
- Call service layer
- Convert domain progress events to gRPC messages
- Map domain errors to gRPC status codes deterministically

NO business logic should be here.
"""

from __future__ import annotations

import contextlib
import json
import uuid
import grpc
import structlog

from ..domain.errors import (
    ContentProcessingError,
    DatabaseError,
    InvalidInputError,
    InvalidURLError,
    NetworkTimeoutError,
    RateLimitExceededError,
    StorageError,
)
from ..domain.models import ErrorCode, ProgressEvent, ProgressStage
from ..domain.models import DiscoveryProgressEvent, DiscoveryStage
from ..observability.logger import get_logger
from ..services.discovery_service import DiscoveryService
from ..services.agent_service import AgentScraperService, AgentConstraints
from ..services.scraper_service import WebScraperService

logger = get_logger(__name__)

try:
    from proto.generated import scraper_pb2, scraper_pb2_grpc
except ImportError as e:
    raise ImportError("gRPC code not generated. Run: python scripts/generate_proto.py") from e


_STAGE_MAP = {
    ProgressStage.VALIDATING: scraper_pb2.PROGRESS_STAGE_VALIDATING,
    ProgressStage.INITIALIZING: scraper_pb2.PROGRESS_STAGE_INITIALIZING,
    ProgressStage.SCRAPING: scraper_pb2.PROGRESS_STAGE_SCRAPING,
    ProgressStage.FILTERING: scraper_pb2.PROGRESS_STAGE_FILTERING,
    ProgressStage.GENERATING: scraper_pb2.PROGRESS_STAGE_GENERATING,
    ProgressStage.UPLOADING: scraper_pb2.PROGRESS_STAGE_UPLOADING,
    ProgressStage.PERSISTING: scraper_pb2.PROGRESS_STAGE_PERSISTING,
    ProgressStage.COMPLETED: scraper_pb2.PROGRESS_STAGE_COMPLETED,
    ProgressStage.FAILED: scraper_pb2.PROGRESS_STAGE_FAILED,
}

_ERROR_MAP = {
    ErrorCode.INVALID_INPUT: scraper_pb2.ERROR_CODE_INVALID_INPUT,
    ErrorCode.INVALID_URL: scraper_pb2.ERROR_CODE_INVALID_URL,
    ErrorCode.NETWORK_TIMEOUT: scraper_pb2.ERROR_CODE_NETWORK_TIMEOUT,
    ErrorCode.RATE_LIMIT_EXCEEDED: scraper_pb2.ERROR_CODE_RATE_LIMIT_EXCEEDED,
    ErrorCode.CONTENT_PROCESSING_ERROR: scraper_pb2.ERROR_CODE_CONTENT_PROCESSING_ERROR,
    ErrorCode.STORAGE_ERROR: scraper_pb2.ERROR_CODE_STORAGE_ERROR,
    ErrorCode.DATABASE_ERROR: scraper_pb2.ERROR_CODE_DATABASE_ERROR,
    ErrorCode.INTERNAL_ERROR: scraper_pb2.ERROR_CODE_INTERNAL_ERROR,
}

_DISCOVERY_STAGE_MAP = {
    DiscoveryStage.VALIDATING: scraper_pb2.DISCOVERY_STAGE_VALIDATING,
    DiscoveryStage.CRAWLING: scraper_pb2.DISCOVERY_STAGE_CRAWLING,
    DiscoveryStage.SUMMARIZING: scraper_pb2.DISCOVERY_STAGE_SUMMARIZING,
    DiscoveryStage.LLM_ANALYZING: scraper_pb2.DISCOVERY_STAGE_LLM_ANALYZING,
    DiscoveryStage.READY_FOR_APPROVAL: scraper_pb2.DISCOVERY_STAGE_READY_FOR_APPROVAL,
    DiscoveryStage.FAILED: scraper_pb2.DISCOVERY_STAGE_FAILED,
}


class WebScraperController(scraper_pb2_grpc.WebScraperServiceServicer):
    def __init__(
        self,
        scraper_service: WebScraperService,
        discovery_service: DiscoveryService,
        agent_service: AgentScraperService,
    ):
        self._service = scraper_service
        self._discovery = discovery_service
        self._agent = agent_service

    async def ScrapeConfigured(self, request, context: grpc.aio.ServicerContext):
        resolved_job_id = request.job_id or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(
            correlation_id=resolved_job_id,
            job_id=resolved_job_id,
            client_id=request.client_id or "",
        )
        try:
            async for ev in self._service.scrape_configured(
                job_id=resolved_job_id,
                client_id=request.client_id,
                config_json=request.config_json,
            ):
                yield self._progress_to_pb(ev)
                if ev.is_complete:
                    break
        except InvalidInputError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            yield self._error_progress(
                job_id=resolved_job_id,
                error_code=ErrorCode.INVALID_INPUT,
                message=str(e),
            )
        except InvalidURLError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            yield self._error_progress(
                job_id=resolved_job_id,
                error_code=ErrorCode.INVALID_URL,
                message=str(e),
            )
        except NetworkTimeoutError as e:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(str(e))
            yield self._error_progress(
                job_id=resolved_job_id,
                error_code=ErrorCode.NETWORK_TIMEOUT,
                message=str(e),
            )
        except RateLimitExceededError as e:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(str(e))
            yield self._error_progress(
                job_id=resolved_job_id,
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                message=str(e),
            )
        except ContentProcessingError as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            yield self._error_progress(
                job_id=resolved_job_id,
                error_code=ErrorCode.CONTENT_PROCESSING_ERROR,
                message=str(e),
            )
        except StorageError as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            yield self._error_progress(
                job_id=resolved_job_id,
                error_code=ErrorCode.STORAGE_ERROR,
                message=str(e),
            )
        except DatabaseError as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            yield self._error_progress(
                job_id=resolved_job_id,
                error_code=ErrorCode.DATABASE_ERROR,
                message=str(e),
            )
        except grpc.aio.AioRpcError:
            raise
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Unexpected error: {str(e)}")
            yield self._error_progress(
                job_id=resolved_job_id,
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Unexpected error: {str(e)}",
            )
        finally:
            with contextlib.suppress(Exception):
                structlog.contextvars.clear_contextvars()

    async def GetJobStatus(self, request, context: grpc.aio.ServicerContext):
        try:
            job, docs = await self._service.get_job_status(request.job_id)
            if job is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("job not found")
                return scraper_pb2.JobStatusResponse(
                    job_id=request.job_id,
                    status=scraper_pb2.JOB_STATUS_UNSPECIFIED,
                    success=False,
                    error_message="job not found",
                )

            status_map = {
                "PENDING": scraper_pb2.JOB_STATUS_PENDING,
                "RUNNING": scraper_pb2.JOB_STATUS_RUNNING,
                "COMPLETED": scraper_pb2.JOB_STATUS_COMPLETED,
                "FAILED": scraper_pb2.JOB_STATUS_FAILED,
                "CANCELLED": scraper_pb2.JOB_STATUS_CANCELLED,
            }
            status_pb = status_map.get(job.status, scraper_pb2.JOB_STATUS_UNSPECIFIED)

            stats = scraper_pb2.ProcessingStats(
                total_pages_processed=job.total_pages,
                successful_pages=job.successful_pages,
                failed_pages=job.failed_pages,
                total_processing_time_ms=job.total_processing_time_ms,
            )

            pb_docs = [
                scraper_pb2.ProcessedDocument(
                    minio_path=d.minio_path,
                    source_url=d.source_url,
                    file_size_bytes=d.file_size_bytes,
                    content_type=d.content_type,
                    processing_time_ms=d.processing_time_ms,
                    output_format=d.output_format,
                )
                for d in docs
                if d.success
            ]

            return scraper_pb2.JobStatusResponse(
                job_id=job.job_id,
                status=status_pb,
                success=bool(job.success),
                error_message=job.error_message or "",
                stats=stats,
                documents=pb_docs,
            )
        except InvalidInputError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return scraper_pb2.JobStatusResponse(
                job_id=request.job_id,
                status=scraper_pb2.JOB_STATUS_UNSPECIFIED,
                success=False,
                error_message=str(e),
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Unexpected error: {str(e)}")
            return scraper_pb2.JobStatusResponse(
                job_id=request.job_id,
                status=scraper_pb2.JOB_STATUS_UNSPECIFIED,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    async def DiscoverStructure(self, request, context: grpc.aio.ServicerContext):
        resolved_job_id = request.job_id or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(
            correlation_id=resolved_job_id,
            job_id=resolved_job_id,
            client_id=request.client_id or "",
        )
        try:
            discovery_options = {}
            if request.HasField("discovery_options"):
                discovery_options = {
                    "max_depth": request.discovery_options.max_depth,
                    "max_pages": request.discovery_options.max_pages,
                    "allowed_domains": list(request.discovery_options.allowed_domains),
                    "exclude_patterns": list(request.discovery_options.exclude_patterns),
                }
                if request.discovery_options.HasField("quality_thresholds"):
                    discovery_options["quality_thresholds"] = {
                        "min_word_count": request.discovery_options.quality_thresholds.min_word_count,
                        "min_text_to_html_ratio": request.discovery_options.quality_thresholds.min_text_to_html_ratio,
                    }

            llm_options = {}
            if request.HasField("llm_options"):
                llm_options = {
                    "provider": request.llm_options.provider,
                    "model": request.llm_options.model,
                    "temperature": request.llm_options.temperature,
                    "max_tokens": request.llm_options.max_tokens,
                    "timeout_seconds": request.llm_options.timeout_seconds,
                }

            scraping_options = {}
            if request.HasField("scraping_options"):
                scraping_options = {
                    "output_format": request.scraping_options.output_format,
                    "include_images": request.scraping_options.include_images,
                    "image_handling": request.scraping_options.image_handling,
                    "noise_selectors": list(request.scraping_options.noise_selectors),
                    "include_metadata": request.scraping_options.include_metadata,
                }

            async for ev in self._discovery.discover_structure(
                job_id=resolved_job_id,
                client_id=request.client_id,
                base_url=request.base_url,
                discovery_options=discovery_options,
                llm_options=llm_options,
                scraping_options=scraping_options,
            ):
                yield self._discovery_progress_to_pb(ev)
                if ev.is_complete:
                    break
        except InvalidInputError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            yield self._discovery_error_progress(resolved_job_id, ErrorCode.INVALID_INPUT, str(e))
        except InvalidURLError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            yield self._discovery_error_progress(resolved_job_id, ErrorCode.INVALID_URL, str(e))
        except NetworkTimeoutError as e:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(str(e))
            yield self._discovery_error_progress(resolved_job_id, ErrorCode.NETWORK_TIMEOUT, str(e))
        except ContentProcessingError as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            yield self._discovery_error_progress(resolved_job_id, ErrorCode.CONTENT_PROCESSING_ERROR, str(e))
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Unexpected error: {str(e)}")
            yield self._discovery_error_progress(resolved_job_id, ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")
        finally:
            with contextlib.suppress(Exception):
                structlog.contextvars.clear_contextvars()

    async def GetDiscoveredStructure(self, request, context: grpc.aio.ServicerContext):
        try:
            rec = await self._discovery.get_discovered(request.job_id)
            if rec is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("discovery job not found")
                return scraper_pb2.DiscoveredStructureResponse(
                    job_id=request.job_id,
                    success=False,
                    error_message="discovery job not found",
                )
            return scraper_pb2.DiscoveredStructureResponse(
                job_id=rec.job_id,
                success=bool(rec.success),
                error_message=rec.error_message or "",
                base_url=rec.base_url,
                discovered_structure_json=rec.discovered_structure_json or "",
                approved=bool(rec.approved),
                approved_structure_json=rec.approved_structure_json or "",
                llm_provider=rec.llm_provider,
                llm_model=rec.llm_model,
                scraping_options_json=rec.scraping_options_json or "",
            )
        except InvalidInputError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return scraper_pb2.DiscoveredStructureResponse(
                job_id=request.job_id,
                success=False,
                error_message=str(e),
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Unexpected error: {str(e)}")
            return scraper_pb2.DiscoveredStructureResponse(
                job_id=request.job_id,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    async def ApproveAndScrape(self, request, context: grpc.aio.ServicerContext):
        # Approval step builds a Phase 1 compatible config_json and then reuses the existing pipeline.
        resolved_job_id = request.job_id.strip() if request.job_id else str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(
            correlation_id=resolved_job_id,
            job_id=resolved_job_id,
            client_id="",
        )
        try:
            result = await self._discovery.approve_and_build_config(
                job_id=resolved_job_id,
                approved_structure_json=request.approved_structure_json or None,
            )
            config_json = json.dumps(
                {
                    "base_url": result.base_url,
                    "structure": result.structure,
                    "options": result.scraping_options,
                },
                ensure_ascii=False,
            )
            async for ev in self._service.scrape_configured(
                job_id=resolved_job_id,
                client_id="",
                config_json=config_json,
            ):
                yield self._progress_to_pb(ev)
                if ev.is_complete:
                    break
        except InvalidInputError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            yield self._error_progress(resolved_job_id, ErrorCode.INVALID_INPUT, str(e))
        except InvalidURLError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            yield self._error_progress(resolved_job_id, ErrorCode.INVALID_URL, str(e))
        except NetworkTimeoutError as e:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(str(e))
            yield self._error_progress(resolved_job_id, ErrorCode.NETWORK_TIMEOUT, str(e))
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Unexpected error: {str(e)}")
            yield self._error_progress(resolved_job_id, ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")
        finally:
            with contextlib.suppress(Exception):
                structlog.contextvars.clear_contextvars()

    async def ScrapePhase3(self, request, context: grpc.aio.ServicerContext):
        resolved_job_id = request.job_id.strip() if request.job_id else str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(
            correlation_id=resolved_job_id,
            job_id=resolved_job_id,
            client_id=request.client_id or "",
        )
        try:
            agent_options = {}
            if request.HasField("agent_options"):
                agent_options = {
                    "max_depth": request.agent_options.max_depth,
                    "max_pages": request.agent_options.max_pages,
                    "allowed_domains": list(request.agent_options.allowed_domains),
                    "exclude_patterns": list(request.agent_options.exclude_patterns),
                }

            llm_provider = ""
            llm_model = ""
            if request.HasField("llm_options"):
                llm_provider = request.llm_options.provider
                llm_model = request.llm_options.model

            scraping_options = {}
            if request.HasField("scraping_options"):
                scraping_options = {
                    "output_format": request.scraping_options.output_format,
                    "include_images": request.scraping_options.include_images,
                    "image_handling": request.scraping_options.image_handling,
                    "noise_selectors": list(request.scraping_options.noise_selectors),
                    "include_metadata": request.scraping_options.include_metadata,
                }
            if not scraping_options:
                scraping_options = {
                    "output_format": "pdf",
                    "include_images": False,
                    "image_handling": "embed_or_link",
                    "noise_selectors": ["header", "footer", "nav", "aside"],
                    "include_metadata": True,
                }

            constraints = AgentConstraints(
                max_depth=int(agent_options.get("max_depth") or 2),
                max_pages=int(agent_options.get("max_pages") or 50),
                allowed_domains=tuple(agent_options.get("allowed_domains") or ()),
                exclude_patterns=tuple(agent_options.get("exclude_patterns") or ()),
                llm_provider=(llm_provider or "ollama"),
                llm_model=llm_model or None,
            )

            async for ev in self._agent.scrape_phase3(
                job_id=resolved_job_id,
                client_id=request.client_id or "",
                base_url=request.base_url,
                prompt=request.prompt,
                constraints=constraints,
                scraping_options=scraping_options,
            ):
                yield self._progress_to_pb(ev)
                if ev.is_complete:
                    break
        except InvalidInputError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            yield self._error_progress(resolved_job_id, ErrorCode.INVALID_INPUT, str(e))
        except InvalidURLError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            yield self._error_progress(resolved_job_id, ErrorCode.INVALID_URL, str(e))
        except NetworkTimeoutError as e:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(str(e))
            yield self._error_progress(resolved_job_id, ErrorCode.NETWORK_TIMEOUT, str(e))
        except ContentProcessingError as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            yield self._error_progress(resolved_job_id, ErrorCode.CONTENT_PROCESSING_ERROR, str(e))
        except StorageError as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            yield self._error_progress(resolved_job_id, ErrorCode.STORAGE_ERROR, str(e))
        except DatabaseError as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            yield self._error_progress(resolved_job_id, ErrorCode.DATABASE_ERROR, str(e))
        except grpc.aio.AioRpcError:
            raise
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Unexpected error: {str(e)}")
            yield self._error_progress(
                resolved_job_id,
                ErrorCode.INTERNAL_ERROR,
                f"Unexpected error: {str(e)}",
            )
        finally:
            with contextlib.suppress(Exception):
                structlog.contextvars.clear_contextvars()

    def _progress_to_pb(self, ev: ProgressEvent) -> scraper_pb2.ScrapeProgress:
        pb = scraper_pb2.ScrapeProgress(
            job_id=ev.job_id,
            stage=_STAGE_MAP.get(ev.stage, scraper_pb2.PROGRESS_STAGE_UNSPECIFIED),
            message=ev.message,
            current_url=ev.current_url or "",
            current_index=ev.current_index,
            total_documents=ev.total_documents,
            is_complete=ev.is_complete,
            success=ev.success,
        )

        if ev.error_code:
            pb.error_code = _ERROR_MAP.get(ev.error_code, scraper_pb2.ERROR_CODE_UNSPECIFIED)
            pb.error_message = ev.error_message or ""

        if ev.last_document:
            pb.last_document.CopyFrom(
                scraper_pb2.ProcessedDocument(
                    minio_path=ev.last_document.minio_path,
                    source_url=ev.last_document.source_url,
                    file_size_bytes=ev.last_document.file_size_bytes,
                    content_type=ev.last_document.content_type,
                    processing_time_ms=ev.last_document.processing_time_ms,
                    output_format=ev.last_document.output_format,
                )
            )

        if ev.stats:
            pb.stats.CopyFrom(
                scraper_pb2.ProcessingStats(
                    total_pages_processed=ev.stats.total_pages_processed,
                    successful_pages=ev.stats.successful_pages,
                    failed_pages=ev.stats.failed_pages,
                    total_processing_time_ms=ev.stats.total_processing_time_ms,
                )
            )

        return pb

    def _error_progress(self, job_id: str, error_code: ErrorCode, message: str) -> scraper_pb2.ScrapeProgress:
        return scraper_pb2.ScrapeProgress(
            job_id=job_id,
            stage=scraper_pb2.PROGRESS_STAGE_FAILED,
            message="job_failed",
            error_code=_ERROR_MAP.get(error_code, scraper_pb2.ERROR_CODE_UNSPECIFIED),
            error_message=message,
            is_complete=True,
            success=False,
        )

    def _discovery_progress_to_pb(self, ev: DiscoveryProgressEvent) -> scraper_pb2.DiscoveryProgress:
        pb = scraper_pb2.DiscoveryProgress(
            job_id=ev.job_id,
            stage=_DISCOVERY_STAGE_MAP.get(ev.stage, scraper_pb2.DISCOVERY_STAGE_UNSPECIFIED),
            message=ev.message,
            current_url=ev.current_url or "",
            pages_crawled=ev.pages_crawled,
            max_pages=ev.max_pages,
            discovered_structure_json=ev.discovered_structure_json or "",
            is_complete=ev.is_complete,
            success=ev.success,
        )
        if ev.error_code:
            pb.error_code = _ERROR_MAP.get(ev.error_code, scraper_pb2.ERROR_CODE_UNSPECIFIED)
            pb.error_message = ev.error_message or ""
        return pb

    def _discovery_error_progress(
        self, job_id: str, error_code: ErrorCode, message: str
    ) -> scraper_pb2.DiscoveryProgress:
        return scraper_pb2.DiscoveryProgress(
            job_id=job_id,
            stage=scraper_pb2.DISCOVERY_STAGE_FAILED,
            message="discovery_failed",
            error_code=_ERROR_MAP.get(error_code, scraper_pb2.ERROR_CODE_UNSPECIFIED),
            error_message=message,
            is_complete=True,
            success=False,
        )

