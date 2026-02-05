#!/usr/bin/env python3
"""Probe a few URLs to see if they pass current filtering + quality thresholds.

This is a dev helper to pick stable E2E test pages.
"""

from __future__ import annotations

import asyncio

from scraper.scraping.playwright_scraper import PlaywrightScraper
from scraper.services.content_filter import ContentFilter
from scraper.utils.quality import assess_quality


URLS = [
    "https://www.cyon.ch/support",
    "https://www.cyon.ch/support/e-mail",
    "https://www.cyon.ch/support/website",
    "https://www.cyon.ch/support/domains",
]


async def main() -> None:
    scraper = PlaywrightScraper(
        default_timeout_ms=30000,
        user_agent="WebScraperService/0.1.0",
        respect_robots_txt=False,
    )
    f = ContentFilter(default_noise_selectors=["header", "footer", "nav"])

    for u in URLS:
        try:
            page = await scraper.fetch_html(u, timeout_ms=30000)
            filtered = f.filter_html(page.html, noise_selectors=["header", "footer", "nav"])
            q = assess_quality(filtered)
            print(f"{u} words={q.word_count} ratio={q.text_to_html_ratio:.4f} textlen={q.text_length}")
        except Exception as e:
            print(f"{u} ERROR {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    asyncio.run(main())


