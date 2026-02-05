"""Path helpers (MinIO object naming)."""

from __future__ import annotations

from urllib.parse import urlparse

from .validators import sanitize_path_segment


def build_minio_base_prefix(service_name: str, base_url: str) -> str:
    parsed = urlparse(base_url)
    host = sanitize_path_segment(parsed.netloc or "unknown-host")
    return f"{sanitize_path_segment(service_name)}/{host}"


def build_document_object_name(
    base_prefix: str,
    category_path: list[str],
    doc_name: str,
    extension: str,
) -> str:
    parts = [base_prefix] + [sanitize_path_segment(p) for p in category_path if p] + [
        f"{sanitize_path_segment(doc_name)}.{extension}"
    ]
    return "/".join([p for p in parts if p])


def build_metadata_object_name(document_object_name: str) -> str:
    return f"{document_object_name}.metadata.json"


def build_image_object_name(
    base_prefix: str,
    category_path: list[str],
    doc_name: str,
    filename: str,
) -> str:
    """Build MinIO object name for an image stored alongside a document.

    Folder convention (per design docs):
      .../<category>/<subcategory>/images/<doc_name>--<filename>
    """
    safe_doc = sanitize_path_segment(doc_name)
    safe_filename = sanitize_path_segment(filename)
    combined = f"{safe_doc}--{safe_filename}"
    parts = [base_prefix] + [sanitize_path_segment(p) for p in category_path if p] + ["images", combined]
    return "/".join([p for p in parts if p])


