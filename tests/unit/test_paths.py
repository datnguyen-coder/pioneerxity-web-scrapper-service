from __future__ import annotations

from scraper.utils.paths import build_document_object_name, build_metadata_object_name, build_minio_base_prefix


def test_build_minio_base_prefix() -> None:
    p = build_minio_base_prefix("web-scraper", "https://example.com/docs")
    assert p.startswith("web-scraper/")
    assert "example.com" in p


def test_build_document_object_name_and_metadata() -> None:
    obj = build_document_object_name(
        base_prefix="svc/example.com",
        category_path=["cat", "sub"],
        doc_name="intro",
        extension="pdf",
    )
    assert obj == "svc/example.com/cat/sub/intro.pdf"
    meta = build_metadata_object_name(obj)
    assert meta == "svc/example.com/cat/sub/intro.pdf.metadata.json"


