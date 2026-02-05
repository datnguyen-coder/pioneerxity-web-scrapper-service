"""robots.txt politeness support.

The service must respect robots.txt per requirements. This module provides a small
async helper with caching.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import aiohttp


def _now() -> float:
    return time.monotonic()


@dataclass
class _RobotsEntry:
    parser: RobotFileParser
    expires_at: float


class RobotsChecker:
    """robots.txt checker with in-memory cache.

    Policy:
    - If robots.txt cannot be fetched or parsed, allow by default (fail-open),
      but keep this behavior configurable later if needed.
    """

    def __init__(self, *, cache_ttl_seconds: int = 3600, timeout_seconds: int = 10):
        self._ttl = int(cache_ttl_seconds)
        self._timeout = int(timeout_seconds)
        self._cache: Dict[str, _RobotsEntry] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._global_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    def _cache_key_for(self, url: str) -> Tuple[str, str]:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "http").lower()
        netloc = (parsed.netloc or "").lower()
        return scheme, netloc

    def _robots_url(self, scheme: str, netloc: str) -> str:
        return f"{scheme}://{netloc}/robots.txt"

    async def _fetch_and_parse(self, scheme: str, netloc: str) -> RobotFileParser:
        robots_url = self._robots_url(scheme, netloc)
        parser = RobotFileParser()
        parser.set_url(robots_url)

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(robots_url) as resp:
                if resp.status >= 400:
                    # Treat as missing robots -> allow all.
                    parser.parse([])
                    return parser
                text = await resp.text(errors="ignore")
                lines = text.splitlines()
                parser.parse(lines)
                return parser

    async def _get_parser(self, scheme: str, netloc: str) -> RobotFileParser:
        if not netloc:
            # No domain => nothing to check
            parser = RobotFileParser()
            parser.parse([])
            return parser

        key = f"{scheme}://{netloc}"
        entry = self._cache.get(key)
        if entry and entry.expires_at > _now():
            return entry.parser

        lock = await self._get_lock(key)
        async with lock:
            # Double-check under lock
            entry = self._cache.get(key)
            if entry and entry.expires_at > _now():
                return entry.parser

            try:
                parser = await self._fetch_and_parse(scheme, netloc)
            except Exception:
                # Fail-open per policy
                parser = RobotFileParser()
                parser.parse([])

            self._cache[key] = _RobotsEntry(parser=parser, expires_at=_now() + self._ttl)
            return parser

    async def is_allowed(self, url: str, *, user_agent: str) -> bool:
        """Return True if robots.txt allows fetching the given URL."""
        parsed = urlparse(url)
        scheme = (parsed.scheme or "http").lower()
        netloc = (parsed.netloc or "").lower()
        parser = await self._get_parser(scheme, netloc)
        try:
            return bool(parser.can_fetch(user_agent, url))
        except Exception:
            # Conservative fallback: allow (fail-open)
            return True


