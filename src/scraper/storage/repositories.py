"""Repository pattern for database access (jobs + documents)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.models import ErrorCode, JobStatus, ProcessedDocument, ProcessingStats
from ..models.database import DiscoveryJobRecord, DocumentRecord, ScrapingJob
from ..observability.logger import get_logger

logger = get_logger(__name__)


class JobRepository:
    """Repository for scraping job operations."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    async def create_job(self, job_id: str, client_id: str, base_url: str, config_json: str) -> None:
        async with self._session_factory() as session:
            job = ScrapingJob(
                job_id=job_id,
                client_id=client_id or "",
                base_url=base_url,
                job_type="configured",
                status=JobStatus.PENDING.value,
                configuration_json=config_json,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(job)
            await session.commit()

    async def mark_running(self, job_id: str, total_pages: int) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(ScrapingJob)
                .where(ScrapingJob.job_id == job_id)
                .values(
                    status=JobStatus.RUNNING.value,
                    started_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    total_pages=total_pages,
                )
            )
            await session.commit()

    async def mark_completed(self, job_id: str, stats: ProcessingStats) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(ScrapingJob)
                .where(ScrapingJob.job_id == job_id)
                .values(
                    status=JobStatus.COMPLETED.value,
                    completed_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    successful_pages=stats.successful_pages,
                    failed_pages=stats.failed_pages,
                    total_processing_time_ms=stats.total_processing_time_ms,
                    success=(stats.failed_pages == 0),
                )
            )
            await session.commit()

    async def mark_failed(self, job_id: str, error_code: ErrorCode, error_message: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(ScrapingJob)
                .where(ScrapingJob.job_id == job_id)
                .values(
                    status=JobStatus.FAILED.value,
                    completed_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    success=False,
                    error_code=error_code.value,
                    error_message=error_message,
                )
            )
            await session.commit()

    async def mark_cancelled(self, job_id: str, error_message: str = "cancelled") -> None:
        """Mark a job as cancelled (e.g., client disconnected / cancellation requested)."""
        async with self._session_factory() as session:
            await session.execute(
                update(ScrapingJob)
                .where(ScrapingJob.job_id == job_id)
                .values(
                    status=JobStatus.CANCELLED.value,
                    completed_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    success=False,
                    error_code=ErrorCode.INTERNAL_ERROR.value,  # keep within existing enum space
                    error_message=error_message,
                )
            )
            await session.commit()

    async def get_job(self, job_id: str) -> Optional[ScrapingJob]:
        async with self._session_factory() as session:
            result = await session.execute(select(ScrapingJob).where(ScrapingJob.job_id == job_id))
            return result.scalar_one_or_none()


class DocumentRepository:
    """Repository for document metadata operations."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    async def save_document(
        self,
        job_id: str,
        doc: ProcessedDocument,
        category_path: dict | None,
        success: bool = True,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
    ) -> None:
        """Save a document metadata record.

        This method should not raise exceptions that break the request flow.
        """
        try:
            async with self._session_factory() as session:
                record = DocumentRecord(
                    job_id=job_id,
                    source_url=doc.source_url,
                    minio_path=doc.minio_path,
                    content_type=doc.content_type,
                    output_format=doc.output_format,
                    file_size_bytes=doc.file_size_bytes,
                    processing_time_ms=doc.processing_time_ms,
                    category_path=category_path,
                    extraction_timestamp=datetime.utcnow(),
                    created_at=datetime.utcnow(),
                    success=success,
                    error_code=error_code.value if error_code else None,
                    error_message=error_message,
                )
                session.add(record)
                await session.commit()
        except Exception as e:
            logger.error("failed_to_save_document_metadata", job_id=job_id, error=str(e))

    async def list_documents_for_job(self, job_id: str) -> list[DocumentRecord]:
        async with self._session_factory() as session:
            result = await session.execute(select(DocumentRecord).where(DocumentRecord.job_id == job_id))
            return list(result.scalars().all())


class DiscoveryRepository:
    """Repository for Phase 2 discovery job operations."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    async def create_job(
        self,
        *,
        job_id: str,
        client_id: str,
        base_url: str,
        discovery_options_json: str,
        scraping_options_json: str,
        llm_provider: str,
        llm_model: str,
        max_pages: int,
    ) -> None:
        async with self._session_factory() as session:
            rec = DiscoveryJobRecord(
                job_id=job_id,
                client_id=client_id or "",
                base_url=base_url,
                status=JobStatus.PENDING.value,
                discovery_options_json=discovery_options_json,
                scraping_options_json=scraping_options_json,
                llm_provider=llm_provider,
                llm_model=llm_model,
                max_pages=max_pages,
                pages_crawled=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(rec)
            await session.commit()

    async def mark_running(self, job_id: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(DiscoveryJobRecord)
                .where(DiscoveryJobRecord.job_id == job_id)
                .values(status=JobStatus.RUNNING.value, updated_at=datetime.utcnow())
            )
            await session.commit()

    async def update_progress(self, job_id: str, *, pages_crawled: int) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(DiscoveryJobRecord)
                .where(DiscoveryJobRecord.job_id == job_id)
                .values(pages_crawled=pages_crawled, updated_at=datetime.utcnow())
            )
            await session.commit()

    async def mark_ready_for_approval(self, job_id: str, *, discovered_structure_json: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(DiscoveryJobRecord)
                .where(DiscoveryJobRecord.job_id == job_id)
                .values(
                    status=JobStatus.COMPLETED.value,
                    discovered_structure_json=discovered_structure_json,
                    success=True,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()

    async def update_scraping_options(self, job_id: str, *, scraping_options_json: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(DiscoveryJobRecord)
                .where(DiscoveryJobRecord.job_id == job_id)
                .values(scraping_options_json=scraping_options_json, updated_at=datetime.utcnow())
            )
            await session.commit()

    async def mark_failed(self, job_id: str, *, error_code: ErrorCode, error_message: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(DiscoveryJobRecord)
                .where(DiscoveryJobRecord.job_id == job_id)
                .values(
                    status=JobStatus.FAILED.value,
                    success=False,
                    error_code=error_code.value,
                    error_message=error_message,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()

    async def mark_cancelled(self, job_id: str, *, error_message: str = "client_cancelled") -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(DiscoveryJobRecord)
                .where(DiscoveryJobRecord.job_id == job_id)
                .values(
                    status=JobStatus.CANCELLED.value,
                    success=False,
                    error_code=ErrorCode.INTERNAL_ERROR.value,
                    error_message=error_message,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()

    async def mark_approved(self, job_id: str, *, approved_structure_json: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(DiscoveryJobRecord)
                .where(DiscoveryJobRecord.job_id == job_id)
                .values(
                    approved=True,
                    approved_structure_json=approved_structure_json,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()

    async def get_job(self, job_id: str) -> Optional[DiscoveryJobRecord]:
        async with self._session_factory() as session:
            result = await session.execute(select(DiscoveryJobRecord).where(DiscoveryJobRecord.job_id == job_id))
            return result.scalar_one_or_none()


