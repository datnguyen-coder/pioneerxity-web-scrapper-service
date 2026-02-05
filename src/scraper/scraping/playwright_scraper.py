"""Playwright-based scraper for dynamic content."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright

from ..domain.errors import InvalidURLError, NetworkTimeoutError
from ..utils.rate_limiter import DomainRateLimiter
from ..utils.robots import RobotsChecker


@dataclass(frozen=True)
class ScrapedPage:
    url: str
    html: str


class PlaywrightScraper:
    """Scraping layer.

    Responsibilities:
    - Fetch web content (dynamic pages)
    - Respect timeouts and basic politeness (hook points)
    - Return raw HTML (no filtering, no storage)
    """

    def __init__(
        self,
        default_timeout_ms: int,
        user_agent: str,
        *,
        rate_limiter: DomainRateLimiter | None = None,
        robots_checker: RobotsChecker | None = None,
        respect_robots_txt: bool = True,
    ):
        self._timeout_ms = default_timeout_ms
        self._user_agent = user_agent
        self._rate_limiter = rate_limiter
        self._robots = robots_checker
        self._respect_robots = bool(respect_robots_txt)

    async def fetch_html(self, url: str, timeout_ms: int | None = None) -> ScrapedPage:
        timeout = timeout_ms or self._timeout_ms
        try:
            # Politeness: robots.txt + per-domain rate limiting
            if self._respect_robots and self._robots is not None:
                allowed = await self._robots.is_allowed(url, user_agent=self._user_agent)
                if not allowed:
                    raise InvalidURLError("robots.txt disallows scraping this URL", detail=url)

            if self._rate_limiter is not None:
                await self._rate_limiter.wait_for_slot(url)

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(user_agent=self._user_agent)
                    page = await context.new_page()
                    # NOTE: "networkidle" is fragile on modern sites due to background requests.
                    # For Phase 1 reliability, prefer "domcontentloaded".
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                    # Give client-side rendered pages a brief moment to populate meaningful DOM.
                    # This helps avoid false "content_quality_too_low" on JS-heavy pages.
                    try:
                        await page.wait_for_load_state("networkidle", timeout=min(5000, timeout))
                    except Exception:
                        # ignore - networkidle can be noisy; fallback to a short render wait
                        pass
                    await page.wait_for_timeout(500)
                    html = await page.content()
                    return ScrapedPage(url=url, html=html)
                finally:
                    await browser.close()
        except (asyncio.TimeoutError, PlaywrightTimeoutError) as e:
            raise NetworkTimeoutError(f"Timeout while fetching {url}", detail=str(e)) from e


