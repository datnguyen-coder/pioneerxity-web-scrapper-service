"""MinIO client adapter (object storage)."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass

from minio import Minio
from minio.error import S3Error

from ..observability.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class UploadResult:
    object_name: str
    size_bytes: int
    content_type: str


class MinIOClient:
    """Storage adapter for MinIO.

    Rules:
    - Documents and metadata files are stored in MinIO (never in PostgreSQL).
    - This adapter must not contain business logic.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool,
        bucket: str,
    ):
        self._bucket = bucket
        self._client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    @property
    def bucket(self) -> str:
        return self._bucket

    async def health_check(self) -> bool:
        try:
            # MinIO SDK is sync; run in thread to avoid blocking request/event loop.
            def _ensure_bucket() -> None:
                if not self._client.bucket_exists(self._bucket):
                    self._client.make_bucket(self._bucket)

            await asyncio.to_thread(_ensure_bucket)
            return True
        except Exception:
            return False

    async def upload_bytes(self, object_name: str, data: bytes, content_type: str) -> UploadResult:
        """Upload raw bytes to MinIO."""
        try:
            def _put() -> None:
                stream = io.BytesIO(data)
                self._client.put_object(
                    bucket_name=self._bucket,
                    object_name=object_name,
                    data=stream,
                    length=len(data),
                    content_type=content_type,
                )

            await asyncio.to_thread(_put)
            return UploadResult(object_name=object_name, size_bytes=len(data), content_type=content_type)
        except S3Error as e:
            logger.error("minio_upload_failed", object_name=object_name, error=str(e))
            raise


