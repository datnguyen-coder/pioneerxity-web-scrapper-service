"""SQLAlchemy models for job tracking and document metadata."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ScrapingJob(Base):
    __tablename__ = "scraping_jobs"

    job_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    base_url: Mapped[str] = mapped_column(Text, nullable=False)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False, default="configured")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    configuration_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Statistics
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successful_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Failure
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    minio_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    output_format: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    category_path: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    extraction_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    # Optional per-document error (if a document failed we can still keep a record)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class DiscoveryJobRecord(Base):
    """Phase 2 discovery job tracking + discovered structure storage."""

    __tablename__ = "discovery_jobs"

    job_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    base_url: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    discovery_options_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    scraping_options_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    llm_provider: Mapped[str] = mapped_column(String(50), nullable=False, default="ollama")
    llm_model: Mapped[str] = mapped_column(String(255), nullable=False, default="qwen2.5:3b")

    pages_crawled: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    discovered_structure_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    approved_structure_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


