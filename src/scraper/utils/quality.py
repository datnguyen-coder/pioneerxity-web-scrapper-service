"""Content quality validation utilities.

Phase 1 requires validating extracted content quality before document processing.
"""

from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class QualityReport:
    word_count: int
    text_length: int
    text_to_html_ratio: float


def assess_quality(filtered_html: str) -> QualityReport:
    soup = BeautifulSoup(filtered_html, "lxml")
    # Remove common non-content elements that can heavily skew HTML size.
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text_len = len(text)
    cleaned_html = str(soup)
    html_len = max(1, len(cleaned_html))
    words = [w for w in text.split() if w]
    return QualityReport(
        word_count=len(words),
        text_length=text_len,
        text_to_html_ratio=float(text_len) / float(html_len),
    )


