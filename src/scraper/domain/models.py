"""Framework-agnostic domain models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProgressStage(str, Enum):
    VALIDATING = "VALIDATING"
    INITIALIZING = "INITIALIZING"
    SCRAPING = "SCRAPING"
    FILTERING = "FILTERING"
    GENERATING = "GENERATING"
    UPLOADING = "UPLOADING"
    PERSISTING = "PERSISTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_URL = "INVALID_URL"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    CONTENT_PROCESSING_ERROR = "CONTENT_PROCESSING_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class ProcessingStats:
    total_pages_processed: int
    successful_pages: int
    failed_pages: int
    total_processing_time_ms: int


@dataclass(frozen=True)
class ProcessedDocument:
    minio_path: str
    source_url: str
    file_size_bytes: int
    content_type: str
    processing_time_ms: int
    output_format: str


@dataclass(frozen=True)
class ProgressEvent:
    job_id: str
    stage: ProgressStage
    message: str
    current_url: str = ""
    current_index: int = 0
    total_documents: int = 0
    last_document: Optional[ProcessedDocument] = None
    error_code: Optional[ErrorCode] = None
    error_message: str = ""
    stats: Optional[ProcessingStats] = None
    is_complete: bool = False
    success: bool = True


class DiscoveryStage(str, Enum):
    VALIDATING = "VALIDATING"
    CRAWLING = "CRAWLING"
    SUMMARIZING = "SUMMARIZING"
    LLM_ANALYZING = "LLM_ANALYZING"
    READY_FOR_APPROVAL = "READY_FOR_APPROVAL"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DiscoveryProgressEvent:
    job_id: str
    stage: DiscoveryStage
    message: str
    current_url: str = ""
    pages_crawled: int = 0
    max_pages: int = 0
    discovered_structure_json: str = ""
    error_code: Optional[ErrorCode] = None
    error_message: str = ""
    is_complete: bool = False
    success: bool = True


