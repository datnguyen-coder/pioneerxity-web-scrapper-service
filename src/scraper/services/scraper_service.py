"""Scraper orchestration service (business logic)."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict
from typing import AsyncIterator
from urllib.parse import urljoin, urlparse

from ..config.settings import get_settings
from ..domain.errors import (
    ContentProcessingError,
    DatabaseError,
    InvalidInputError,
    InvalidURLError,
    NetworkTimeoutError,
    StorageError,
)
from ..domain.models import ErrorCode, ProcessedDocument, ProcessingStats, ProgressEvent, ProgressStage
from ..models.requests import ScrapeConfig
from ..observability.logger import get_logger
from ..storage.minio_client import MinIOClient
from ..storage.repositories import DocumentRepository, JobRepository
from ..utils.paths import (
    build_document_object_name,
    build_image_object_name,
    build_metadata_object_name,
    build_minio_base_prefix,
)
from ..utils.time import current_time_ms, elapsed_ms
from ..utils.validators import is_valid_http_url, sanitize_path_segment
from ..utils.quality import assess_quality
from .content_filter import ContentFilter
from .document_processor import DocumentProcessor
from ..scraping.playwright_scraper import PlaywrightScraper
from ..processing.image_handler import ImageHandler

logger = get_logger(__name__)


class WebScraperService:
    """Service layer for web scraping orchestration.

    Responsibilities:
    - Validate input configuration
    - Apply system constraints (limits)
    - Orchestrate scraping -> filtering -> generation -> storage -> metadata persistence
    - Stream progress events
    """

    def __init__(
        self,
        scraper: PlaywrightScraper,
        content_filter: ContentFilter,
        document_processor: DocumentProcessor,
        image_handler: ImageHandler,
        minio: MinIOClient,
        jobs: JobRepository,
        documents: DocumentRepository,
    ):
        self._scraper = scraper
        self._filter = content_filter
        self._processor = document_processor
        self._images = image_handler
        self._minio = minio
        self._jobs = jobs
        self._documents = documents
        self._settings = get_settings()

    async def scrape_configured(self, job_id: str, client_id: str, config_json: str) -> AsyncIterator[ProgressEvent]:
        start_ms = current_time_ms()

        resolved_job_id = job_id.strip() if job_id else str(uuid.uuid4())

        # Validate request shape early
        if not config_json or not config_json.strip():
            raise InvalidInputError("config_json is required")

        yield ProgressEvent(
            job_id=resolved_job_id,
            stage=ProgressStage.VALIDATING,
            message="validating_config",
            is_complete=False,
        )

        # Parse JSON config
        try:
            raw = json.loads(config_json)
        except json.JSONDecodeError as e:
            raise InvalidInputError("config_json must be valid JSON", detail=str(e)) from e

        try:
            cfg = ScrapeConfig.model_validate(raw)
        except Exception as e:
            raise InvalidInputError("invalid configuration schema", detail=str(e)) from e

        # Flatten document URLs
        documents = self._flatten_structure(cfg.base_url, cfg.structure)
        if len(documents) == 0:
            raise InvalidInputError("structure produced 0 documents")

        if len(documents) > self._settings.max_total_documents_per_job:
            raise InvalidInputError(
                f"too many documents: {len(documents)} exceeds max_total_documents_per_job "
                f"({self._settings.max_total_documents_per_job})"
            )

        base_prefix = build_minio_base_prefix(self._settings.service_name, cfg.base_url)

        # Persist job record
        try:
            await self._jobs.create_job(
                job_id=resolved_job_id,
                client_id=client_id,
                base_url=cfg.base_url,
                config_json=config_json,
            )
            await self._jobs.mark_running(resolved_job_id, total_pages=len(documents))
        except Exception as e:
            raise DatabaseError("failed to create job record", detail=str(e)) from e

        yield ProgressEvent(
            job_id=resolved_job_id,
            stage=ProgressStage.INITIALIZING,
            message="job_initialized",
            total_documents=len(documents),
        )

        try:
            total = len(documents)
            successful = 0
            failed = 0

            for idx, (category_path, url, doc_name) in enumerate(documents, start=1):
                doc_start = current_time_ms()
                try:
                    yield ProgressEvent(
                        job_id=resolved_job_id,
                        stage=ProgressStage.SCRAPING,
                        message="scraping_page",
                        current_url=url,
                        current_index=idx,
                        total_documents=total,
                    )

                    page = await self._fetch_with_retries(url)

                    yield ProgressEvent(
                        job_id=resolved_job_id,
                        stage=ProgressStage.FILTERING,
                        message="filtering_content",
                        current_url=url,
                        current_index=idx,
                        total_documents=total,
                    )

                    filtered_html = self._filter.filter_html(
                        page.html,
                        noise_selectors=cfg.options.noise_selectors or None,
                    )

                    # Quality validation before processing (Phase 1 mandatory)
                    q = assess_quality(filtered_html)
                    if (
                        q.word_count < self._settings.min_word_count
                        or q.text_to_html_ratio < self._settings.min_text_to_html_ratio
                    ):
                        failed += 1
                        await self._documents.save_document(
                            job_id=resolved_job_id,
                            doc=ProcessedDocument(
                                minio_path="",
                                source_url=url,
                                file_size_bytes=0,
                                content_type="",
                                processing_time_ms=elapsed_ms(doc_start),
                                output_format="",
                            ),
                            category_path={"path": category_path},
                            success=False,
                            error_code=ErrorCode.CONTENT_PROCESSING_ERROR,
                            error_message="content_quality_too_low",
                        )
                        yield ProgressEvent(
                            job_id=resolved_job_id,
                            stage=ProgressStage.FAILED,
                            message="document_failed",
                            current_url=url,
                            current_index=idx,
                            total_documents=total,
                            error_code=ErrorCode.CONTENT_PROCESSING_ERROR,
                            error_message="content_quality_too_low",
                            is_complete=False,
                            success=False,
                        )
                        continue

                    embedded_images = 0
                    linked_images = 0
                    image_urls: list[str] = []
                    image_infos: list[dict[str, str]] = []
                    reportlab_images: list[tuple[str, bytes, str, str, str]] = []
                    uploaded_images: list[str] = []

                    # Image handling (Phase 1 mandatory)
                    if cfg.options.include_images:
                        refs = self._images.extract_image_refs(filtered_html, base_url=cfg.base_url)
                        image_urls = [r.url for r in refs]
                        image_infos = [{"url": r.url, "alt": r.alt} for r in refs if r.url]
                        alt_by_url = {r.url: r.alt for r in refs if r.url and r.alt}

                        downloaded = []
                        if cfg.options.image_handling in ("embed", "embed_or_link", "both"):
                            downloaded = await self._images.download_images(image_urls)
                            reportlab_images = [
                                (
                                    img.filename,
                                    img.bytes_data,
                                    img.content_type,
                                    alt_by_url.get(img.source_url, ""),
                                    img.source_url,
                                )
                                for img in downloaded
                            ]

                        # Upload images to MinIO when linking is requested
                        uploaded_map: dict[str, str] = {}
                        if cfg.options.image_handling in ("both",):
                            for img in downloaded:
                                img_obj = build_image_object_name(
                                    base_prefix=base_prefix,
                                    category_path=category_path,
                                    doc_name=doc_name,
                                    filename=img.filename,
                                )
                                up = await self._minio.upload_bytes(img_obj, img.bytes_data, img.content_type)
                                ref = f"{self._minio.bucket}/{up.object_name}"
                                uploaded_map[img.source_url] = ref
                                uploaded_images.append(ref)

                        filtered_html, embedded_images, linked_images, _ = self._images.rewrite_html(
                            filtered_html,
                            base_url=cfg.base_url,
                            image_handling=cfg.options.image_handling,
                            downloaded=downloaded,
                            uploaded_image_urls=uploaded_map or None,
                        )

                    yield ProgressEvent(
                        job_id=resolved_job_id,
                        stage=ProgressStage.GENERATING,
                        message="generating_document",
                        current_url=url,
                        current_index=idx,
                        total_documents=total,
                    )

                    generated_docs = []
                    fmt = cfg.options.output_format
                    if fmt in ("pdf", "both"):
                        generated_docs.append(
                            await asyncio.to_thread(
                                self._processor.generate_pdf,
                                filtered_html,
                                cfg.base_url,
                                url,
                                embedded_images=embedded_images,
                                linked_images=linked_images,
                                reportlab_images=reportlab_images,
                            )
                        )
                    if fmt in ("docx", "both"):
                        generated_docs.append(
                            await asyncio.to_thread(
                                self._processor.generate_docx,
                                filtered_html,
                                url,
                                embedded_images=embedded_images,
                                linked_images=linked_images,
                            )
                        )

                    # Upload docs and metadata
                    for gen in generated_docs:
                        if len(gen.bytes_data) > self._settings.max_document_bytes:
                            raise ContentProcessingError(
                                "generated document exceeds max_document_bytes",
                                detail=f"size={len(gen.bytes_data)} max={self._settings.max_document_bytes}",
                            )

                        yield ProgressEvent(
                            job_id=resolved_job_id,
                            stage=ProgressStage.UPLOADING,
                            message="uploading_to_minio",
                            current_url=url,
                            current_index=idx,
                            total_documents=total,
                        )

                        object_name = build_document_object_name(
                            base_prefix=base_prefix,
                            category_path=category_path,
                            doc_name=doc_name,
                            extension=gen.output_format,
                        )
                        try:
                            upload = await self._minio.upload_bytes(object_name, gen.bytes_data, gen.content_type)
                        except Exception as e:
                            raise StorageError("failed to upload document to MinIO", detail=str(e)) from e

                        proc_ms = elapsed_ms(doc_start)
                        processed = ProcessedDocument(
                            minio_path=f"{self._minio.bucket}/{upload.object_name}",
                            source_url=url,
                            file_size_bytes=upload.size_bytes,
                            content_type=upload.content_type,
                            processing_time_ms=proc_ms,
                            output_format=gen.output_format,
                        )

                        # Upload metadata JSON (always available)
                        metadata_obj = build_metadata_object_name(upload.object_name)
                        metadata = {
                            "source_url": url,
                            "extraction_timestamp_ms": current_time_ms(),
                            "content_type": upload.content_type,
                            "file_size_bytes": upload.size_bytes,
                            "processing_duration_ms": proc_ms,
                            "output_format": gen.output_format,
                            "minio_object": upload.object_name,
                            "include_images": bool(cfg.options.include_images),
                            "image_handling": cfg.options.image_handling,
                            "embedded_images": int(getattr(gen, "embedded_images", 0)),
                            "linked_images": int(getattr(gen, "linked_images", 0)),
                            "image_urls": image_urls,
                            "images": image_infos,
                            "uploaded_images": uploaded_images if cfg.options.image_handling == "both" else [],
                            "quality": {
                                "word_count": q.word_count,
                                "text_length": q.text_length,
                                "text_to_html_ratio": q.text_to_html_ratio,
                                "min_word_count": self._settings.min_word_count,
                                "min_text_to_html_ratio": self._settings.min_text_to_html_ratio,
                            },
                        }
                        await self._minio.upload_bytes(
                            metadata_obj,
                            json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
                            "application/json",
                        )

                        yield ProgressEvent(
                            job_id=resolved_job_id,
                            stage=ProgressStage.PERSISTING,
                            message="persisting_metadata",
                            current_url=url,
                            current_index=idx,
                            total_documents=total,
                            last_document=processed,
                        )

                        await self._documents.save_document(
                            job_id=resolved_job_id,
                            doc=processed,
                            category_path={"path": category_path},
                            success=True,
                        )

                    successful += 1

                except InvalidInputError:
                    raise
                except InvalidURLError:
                    failed += 1
                    await self._documents.save_document(
                        job_id=resolved_job_id,
                        doc=ProcessedDocument(
                            minio_path="",
                            source_url=url,
                            file_size_bytes=0,
                            content_type="",
                            processing_time_ms=elapsed_ms(doc_start),
                            output_format="",
                        ),
                        category_path={"path": category_path},
                        success=False,
                        error_code=ErrorCode.INVALID_URL,
                        error_message="invalid_url",
                    )
                    yield ProgressEvent(
                        job_id=resolved_job_id,
                        stage=ProgressStage.FAILED,
                        message="document_failed",
                        current_url=url,
                        current_index=idx,
                        total_documents=total,
                        error_code=ErrorCode.INVALID_URL,
                        error_message="invalid_url",
                        is_complete=False,
                        success=False,
                    )
                except ContentProcessingError as e:
                    failed += 1
                    await self._documents.save_document(
                        job_id=resolved_job_id,
                        doc=ProcessedDocument(
                            minio_path="",
                            source_url=url,
                            file_size_bytes=0,
                            content_type="",
                            processing_time_ms=elapsed_ms(doc_start),
                            output_format="",
                        ),
                        category_path={"path": category_path},
                        success=False,
                        error_code=ErrorCode.CONTENT_PROCESSING_ERROR,
                        error_message=str(e),
                    )
                    yield ProgressEvent(
                        job_id=resolved_job_id,
                        stage=ProgressStage.FAILED,
                        message="document_failed",
                        current_url=url,
                        current_index=idx,
                        total_documents=total,
                        error_code=ErrorCode.CONTENT_PROCESSING_ERROR,
                        error_message=str(e),
                        is_complete=False,
                        success=False,
                    )
                except StorageError as e:
                    failed += 1
                    await self._documents.save_document(
                        job_id=resolved_job_id,
                        doc=ProcessedDocument(
                            minio_path="",
                            source_url=url,
                            file_size_bytes=0,
                            content_type="",
                            processing_time_ms=elapsed_ms(doc_start),
                            output_format="",
                        ),
                        category_path={"path": category_path},
                        success=False,
                        error_code=ErrorCode.STORAGE_ERROR,
                        error_message=str(e),
                    )
                    yield ProgressEvent(
                        job_id=resolved_job_id,
                        stage=ProgressStage.FAILED,
                        message="document_failed",
                        current_url=url,
                        current_index=idx,
                        total_documents=total,
                        error_code=ErrorCode.STORAGE_ERROR,
                        error_message=str(e),
                        is_complete=False,
                        success=False,
                    )
                except Exception as e:
                    failed += 1
                    yield ProgressEvent(
                        job_id=resolved_job_id,
                        stage=ProgressStage.FAILED,
                        message="document_failed",
                        current_url=url,
                        current_index=idx,
                        total_documents=total,
                        error_code=ErrorCode.INTERNAL_ERROR,
                        error_message=str(e),
                        is_complete=False,
                        success=False,
                    )

            stats = ProcessingStats(
                total_pages_processed=total,
                successful_pages=successful,
                failed_pages=failed,
                total_processing_time_ms=elapsed_ms(start_ms),
            )

            # Persist final job status
            try:
                await self._jobs.mark_completed(resolved_job_id, stats)
            except Exception as e:
                logger.error("job_status_persist_failed", job_id=resolved_job_id, error=str(e))

            yield ProgressEvent(
                job_id=resolved_job_id,
                stage=ProgressStage.COMPLETED,
                message="job_completed",
                stats=stats,
                is_complete=True,
                success=(failed == 0),
            )
        except asyncio.CancelledError:
            # Must stop instantly on cancellation and persist cancellation state.
            try:
                await self._jobs.mark_cancelled(resolved_job_id, error_message="client_cancelled")
            except Exception as e:
                logger.error("job_cancel_persist_failed", job_id=resolved_job_id, error=str(e))
            raise

    async def get_job_status(self, job_id: str):
        if not job_id:
            raise InvalidInputError("job_id is required")
        job = await self._jobs.get_job(job_id)
        docs = await self._documents.list_documents_for_job(job_id)
        return job, docs

    def _flatten_structure(self, base_url: str, structure: dict) -> list[tuple[list[str], str, str]]:
        """Flatten nested dict/list structure into a list of (category_path, url, doc_name)."""
        out: list[tuple[list[str], str, str]] = []

        def walk(node: object, path: list[str]) -> None:
            if isinstance(node, list):
                for item in node:
                    if not isinstance(item, str):
                        continue
                    url = urljoin(base_url + "/", item.lstrip("/"))
                    if not is_valid_http_url(url):
                        raise InvalidURLError(f"Invalid URL: {url}")
                    name = self._doc_name_from_path(item)
                    out.append((path.copy(), url, name))
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, path + [str(k)])
                return

        walk(structure, [])
        return out

    def _doc_name_from_path(self, path: str) -> str:
        # Use the last segment without extension as doc name
        s = path.strip().replace("\\", "/").rstrip("/")
        last = s.split("/")[-1] if s else "document"
        if "." in last:
            last = last.rsplit(".", 1)[0]
        return sanitize_path_segment(last)

    async def _fetch_with_retries(self, url: str):
        """Fetch HTML with bounded retries for transient timeouts."""
        attempts = max(0, int(self._settings.scrape_max_retries)) + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._scraper.fetch_html(url, timeout_ms=self._settings.scrape_timeout_ms)
            except NetworkTimeoutError as e:
                last_exc = e
                if attempt >= attempts:
                    raise
                await asyncio.sleep(max(0.0, float(self._settings.scrape_retry_backoff_ms) / 1000.0))
        if last_exc:
            raise last_exc
        raise NetworkTimeoutError("failed to fetch url")  # fallback


