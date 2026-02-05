"""Domain-specific errors.

These errors are mapped to gRPC status codes in the controller layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class ScraperDomainError(Exception):
    """Base class for all domain errors."""


@dataclass(frozen=True)
class DomainErrorInfo:
    code: str
    message: str
    detail: Optional[str] = None


class InvalidInputError(ScraperDomainError):
    """Raised when request/config validation fails."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.info = DomainErrorInfo(code="INVALID_INPUT", message=message, detail=detail)


class InvalidURLError(ScraperDomainError):
    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.info = DomainErrorInfo(code="INVALID_URL", message=message, detail=detail)


class NetworkTimeoutError(ScraperDomainError):
    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.info = DomainErrorInfo(code="NETWORK_TIMEOUT", message=message, detail=detail)


class RateLimitExceededError(ScraperDomainError):
    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.info = DomainErrorInfo(code="RATE_LIMIT_EXCEEDED", message=message, detail=detail)


class ContentProcessingError(ScraperDomainError):
    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.info = DomainErrorInfo(code="CONTENT_PROCESSING_ERROR", message=message, detail=detail)


class StorageError(ScraperDomainError):
    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.info = DomainErrorInfo(code="STORAGE_ERROR", message=message, detail=detail)


class DatabaseError(ScraperDomainError):
    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.info = DomainErrorInfo(code="DATABASE_ERROR", message=message, detail=detail)


