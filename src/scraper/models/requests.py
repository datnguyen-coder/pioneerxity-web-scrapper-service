"""Request/config models (Phase 1 configured scraping)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


ImageHandling = Literal["embed", "link_only", "both", "embed_or_link"]
OutputFormat = Literal["pdf", "docx", "both"]


class ScrapingOptions(BaseModel):
    include_images: bool = True
    image_handling: ImageHandling = "embed_or_link"
    output_format: OutputFormat = "pdf"
    noise_selectors: List[str] = Field(default_factory=list)
    include_metadata: bool = True


StructureNode = Union[Dict[str, Any], List[str]]


class ScrapeConfig(BaseModel):
    base_url: str
    structure: Dict[str, Any]  # category/subcategory nesting
    options: ScrapingOptions = Field(default_factory=ScrapingOptions)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        v = v.strip()
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip("/")


