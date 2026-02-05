"""Validation helpers."""

from __future__ import annotations

from urllib.parse import urlparse


def is_valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def sanitize_path_segment(segment: str) -> str:
    """Sanitize a path segment for MinIO object names."""
    s = segment.strip().replace("\\", "/")
    s = s.replace("..", ".")
    s = s.replace(":", "-")
    s = s.replace("|", "-")
    s = s.replace("?", "")
    s = s.replace("#", "")
    s = s.replace('"', "")
    s = s.replace("<", "")
    s = s.replace(">", "")
    return s.strip("/").strip() or "untitled"


