"""Content filtering (noise removal + main-content extraction)."""

from __future__ import annotations

from bs4 import BeautifulSoup


class ContentFilter:
    """Processing layer component: noise filtering.

    Rules:
    - Apply noise filters before content extraction
    - Never return unfiltered content if noise selectors are provided
    """

    def __init__(self, default_noise_selectors: list[str] | None = None):
        self._default_noise_selectors = default_noise_selectors or []
        self._content_selectors = [
            "main",
            "article",
            ".content",
            ".documentation",
            ".docs",
            ".main-content",
            "#content",
            ".markdown-body",
        ]

    def filter_html(self, html: str, noise_selectors: list[str] | None = None) -> str:
        soup = BeautifulSoup(html, "lxml")

        selectors = noise_selectors if noise_selectors is not None and len(noise_selectors) > 0 else self._default_noise_selectors
        for selector in selectors:
            for el in soup.select(selector):
                el.decompose()

        main = self._find_main_content(soup)
        if main is not None:
            return str(main)
        return str(soup)

    def _find_main_content(self, soup: BeautifulSoup):
        for selector in self._content_selectors:
            candidate = soup.select_one(selector)
            if candidate is None:
                continue
            text = candidate.get_text(strip=True)
            if len(text) >= 200:
                return candidate
        return None


