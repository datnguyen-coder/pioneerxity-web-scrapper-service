"""Phase 3 agent service (LLM-guided browsing).

Responsibilities:
- Use LLM to navigate and collect relevant document URLs for a user prompt
- Transform collected URLs into a Phase 1 compatible config
- Delegate scraping to Phase 1 pipeline
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, AsyncIterator, Any
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp

from ..config.settings import get_settings
from ..domain.errors import InvalidInputError
from ..domain.models import ProgressEvent, ProgressStage
from ..llm.ollama_adapter import OllamaAdapter
from ..llm.openai_adapter import OpenAIAdapter
from ..llm.runtime import LLMRequest, LLMRuntime
from ..observability.logger import get_logger
from ..scraping.playwright_scraper import PlaywrightScraper
from ..services.scraper_service import WebScraperService
from ..utils.validators import is_valid_http_url

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

logger = get_logger(__name__)


@dataclass(frozen=True)
class AgentConstraints:
    max_depth: int = 2
    max_pages: int = 50
    allowed_domains: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    llm_provider: str = "ollama"  # "ollama" | "openai"
    llm_model: str | None = None


@dataclass(frozen=True)
class CollectedURL:
    url: str
    path: str
    title: str | None = None


def _canonicalize(u: str) -> str:
    p = urlparse(u)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, p.query, ""))


def _same_or_allowed_domain(u: str, base: str, allowed_domains: Iterable[str]) -> bool:
    host = urlparse(u).netloc.lower()
    base_host = urlparse(base).netloc.lower()
    if host == base_host:
        return True
    allowed = [d.strip().lower() for d in allowed_domains if d and d.strip()]
    return any(host == d or host.endswith("." + d) for d in allowed)


def _excluded(u: str, exclude_patterns: Iterable[str]) -> bool:
    for pat in exclude_patterns:
        if not pat:
            continue
        try:
            if re.search(pat, u):
                return True
        except re.error:
            if pat in u:
                return True
    return False


def _extract_title(html: str) -> str | None:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        t = soup.find("title")
        if t and t.get_text(strip=True):
            return t.get_text(strip=True)
        return None
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"<[^>]+>", " ", m.group(1)).strip() or None
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"<[^>]+>", " ", m.group(1)).strip() or None
    return None


def _extract_links(html: str, *, current_url: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href or href.startswith("#"):
                continue
            if href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            anchor_text = (a.get_text(strip=True) or "").strip()[:200]
            full_url = urljoin(current_url, href)
            out.append((full_url, anchor_text))
        return out
    for m in re.finditer(r"""<a\b[^>]*\bhref\s*=\s*["']([^"']+)["']""", html, flags=re.IGNORECASE):
        href = (m.group(1) or "").strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        full_url = urljoin(current_url, href)
        out.append((full_url, ""))
    return out


def _extract_text_snippet(html: str, max_chars: int = 1000) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


_URL_RE = re.compile(r"https?://[^\s)\]\"']+")


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*$", "", cleaned)
    return cleaned.strip()


def _maybe_load_json(text: str) -> Any | None:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None
    if cleaned.startswith("{") or cleaned.startswith("["):
        try:
            return json.loads(cleaned)
        except ValueError:
            return None
    match = re.search(r"(\{.*?\}|\[.*?\])", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except ValueError:
            return None
    return None


def _extract_urls_from_json(data: Any) -> list[str]:
    urls: list[str] = []
    if data is None:
        return urls
    if isinstance(data, str):
        return [data]
    if isinstance(data, list):
        for item in data:
            urls.extend(_extract_urls_from_json(item))
        return urls
    if isinstance(data, dict):
        for key in ("urls", "relevant_urls", "links", "documents"):
            if key in data:
                urls.extend(_extract_urls_from_json(data.get(key)))
        for key in ("url", "link"):
            if key in data:
                urls.extend(_extract_urls_from_json(data.get(key)))
        return urls
    return urls


def _extract_urls_from_text(text: str) -> list[str]:
    data = _maybe_load_json(text)
    urls = _extract_urls_from_json(data)
    if urls:
        return urls
    return _URL_RE.findall(text or "")


def _filter_browser_use_urls(
    *,
    base_url: str,
    urls: list[str],
    constraints: AgentConstraints,
) -> list[str]:
    base_path = urlparse(base_url).path.rstrip("/") or ""
    out: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        if not isinstance(raw, str):
            continue
        cleaned = raw.strip().rstrip(",")
        if not cleaned or not is_valid_http_url(cleaned):
            continue
        cleaned = _canonicalize(cleaned)
        if cleaned in seen:
            continue
        if not _same_or_allowed_domain(cleaned, base_url, constraints.allowed_domains):
            continue
        if _excluded(cleaned, constraints.exclude_patterns):
            continue
        if base_path and not urlparse(cleaned).path.startswith(base_path):
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _build_browser_use_task(
    *,
    base_url: str,
    prompt: str,
    constraints: AgentConstraints,
) -> str:
    allowed_domains = [d for d in constraints.allowed_domains if d]
    if not allowed_domains:
        allowed_domains = [urlparse(base_url).netloc]
    exclude_patterns = [p for p in constraints.exclude_patterns if p]
    domain_text = ", ".join(allowed_domains)
    exclude_text = ", ".join(exclude_patterns) if exclude_patterns else "none"
    return (
        "You are a web navigation agent.\n"
        f"Start from: {base_url}\n"
        f"User request: \"{prompt}\"\n\n"
        "Goal: collect URLs to actual document pages that match the user request.\n"
        f"- Max depth: {constraints.max_depth}\n"
        f"- Max URLs: {constraints.max_pages}\n"
        f"- Allowed domains: {domain_text}\n"
        f"- Exclude patterns (regex or substring): {exclude_text}\n"
        "- Avoid category pages that only list links.\n"
        "- Prefer direct document/article pages with substantial content.\n"
        "- Do not use Google if the site has native navigation.\n\n"
        "Return JSON only (no markdown, no extra text) in this format:\n"
        "{\"urls\": [\"https://example.com/doc1\", \"https://example.com/doc2\"]}\n"
        "If none found, return: {\"urls\": []}\n"
    )


def _extract_keywords_from_prompt(prompt: str) -> list[str]:
    keywords: list[str] = []
    prompt_lower = prompt.lower()
    matches = re.findall(
        r"(?:topic|about|document|documents?|related to|with topic)\s+([a-z0-9\-\s]+)",
        prompt_lower,
    )
    for match in matches:
        keywords.extend(match.strip().split())
    capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", prompt)
    for cap in capitalized:
        keywords.append(cap.lower())
        keywords.extend(re.split(r"[\s\-]+", cap.lower()))
    stop_words = {
        "the",
        "all",
        "find",
        "get",
        "collect",
        "scrape",
        "document",
        "documents",
        "topic",
        "about",
        "with",
        "related",
        "to",
    }
    keywords = [k for k in keywords if k and k not in stop_words and len(k) > 2]
    return list(set(keywords))


def _is_likely_document_page(html: str) -> tuple[bool, dict[str, int | bool]]:
    metadata: dict[str, int | bool] = {
        "has_main_content": False,
        "text_length": 0,
        "link_count": 0,
        "has_article_tag": False,
        "has_substantial_paragraphs": False,
    }
    if BeautifulSoup is None:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        metadata["text_length"] = len(text)
        return len(text) > 500, metadata
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()
    article = soup.find("article") or soup.find("main")
    if article:
        metadata["has_main_content"] = True
        metadata["has_article_tag"] = True
        content_area = article
    else:
        main = soup.find("main")
        if main:
            content_area = main
            metadata["has_main_content"] = True
        else:
            content_area = soup
    links = content_area.find_all("a", href=True)
    metadata["link_count"] = len(links)
    text = content_area.get_text(separator=" ", strip=True)
    metadata["text_length"] = len(text)
    paragraphs = content_area.find_all(["p", "div"])
    substantial_paras = sum(1 for p in paragraphs if len(p.get_text(strip=True)) > 100)
    metadata["has_substantial_paragraphs"] = substantial_paras >= 2
    is_doc = (
        int(metadata["text_length"]) > 800
        and (metadata["has_main_content"] or metadata["has_substantial_paragraphs"])
        and (int(metadata["link_count"]) == 0 or int(metadata["text_length"]) / max(int(metadata["link_count"]), 1) > 50)
    )
    return bool(is_doc), metadata


async def _ask_llm_is_url_relevant(
    *,
    llm: LLMRuntime,
    prompt: str,
    url: str,
    title: str | None,
    snippet: str,
    is_document_page: bool,
    page_metadata: dict[str, int | bool],
    model: str,
) -> bool:
    keywords = _extract_keywords_from_prompt(prompt)
    keywords_hint = f"\nKeywords to match: {', '.join(keywords)}" if keywords else ""
    page_type = "document page (has substantial content)" if is_document_page else "category/topic page (mostly links)"
    llm_prompt = f"""You are a web navigation agent. The user wants to: "{prompt}"
{keywords_hint}

CRITICAL: The user is looking for documents about a SPECIFIC TOPIC. You must be very strict about relevance.
- ONLY pages about the topic mentioned in the user's request are relevant
- Pages about other topics are NOT relevant, even if they are on the same website
- The page must be an ACTUAL DOCUMENT PAGE with real content, NOT a category/topic page that only lists links

URL: {url}
Title: {title or 'N/A'}
Page type: {page_type}
Page metadata: text_length={page_metadata.get('text_length', 0)}, link_count={page_metadata.get('link_count', 0)}, has_main_content={page_metadata.get('has_main_content', False)}
Content snippet: {snippet[:500]}

Task: Is this URL a RELEVANT DOCUMENT PAGE that matches the user's specific topic request?
- Return true ONLY if:
  (1) The page content is DIRECTLY about the topic mentioned in the user's request
  (2) It's an actual document page with substantial content (not just a category page listing links)
- Return false if:
  - The page is about a different topic (even if on the same website)
  - It's just a category/topic page listing links
  - The content doesn't match the user's specific topic request

Response format (JSON only):
{{
  "relevant": true or false,
  "reason": "brief explanation"
}}
"""
    req = LLMRequest(
        prompt=llm_prompt,
        provider="",
        model=model,
        temperature=0.1,
        max_tokens=256,
        timeout_seconds=30,
    )
    try:
        response = await llm.complete(req)
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?\s*", "", response, flags=re.MULTILINE)
            response = re.sub(r"```\s*$", "", response, flags=re.MULTILINE)
        response = response.strip()
        data = json.loads(response)
        return bool(data.get("relevant", False))
    except Exception as e:
        logger.warning("llm_relevance_check_failed", url=url, error=str(e))
        return False


async def _ask_llm_which_links_relevant(
    *,
    llm: LLMRuntime,
    prompt: str,
    current_url: str,
    page_title: str | None,
    page_snippet: str,
    links: list[tuple[str, str]],
    model: str,
) -> list[str]:
    if not links:
        return []
    keywords = _extract_keywords_from_prompt(prompt)
    keywords_hint = f"Keywords to look for: {', '.join(keywords)}" if keywords else ""
    links_text_parts = []
    for url, anchor_text in links[:50]:
        url_lower = url.lower()
        anchor_lower = (anchor_text or "").lower()
        contains_keyword = any(kw in url_lower or kw in anchor_lower for kw in keywords) if keywords else False
        marker = "⭐ " if contains_keyword else "  "
        links_text_parts.append(
            f"{marker}{url} (text: {anchor_text[:100] if anchor_text else 'no text'})"
        )
    links_text = "\n".join(links_text_parts)
    llm_prompt = f"""You are a web navigation agent. The user wants to: "{prompt}"
{keywords_hint}

CRITICAL: The user is looking for documents about a SPECIFIC TOPIC. You must be very selective.
- ONLY follow links that are clearly about the topic mentioned in the user's request
- Look for links with anchor text or URLs that contain keywords related to the user's topic
- Avoid links to other topics, even if they are on the same website
- Prefer links that lead to actual document pages (articles, guides, tutorials) over category pages
- If you see a link to a topic/category page that matches the user's topic, you should follow it to find documents within that topic

Current page: {current_url}
Page title: {page_title or 'N/A'}
Page content snippet: {page_snippet[:500]}

Available links on this page:
{links_text}

Task: Return a JSON array of URLs that are RELEVANT to the user's SPECIFIC TOPIC request.
- Only include links that are clearly related to the topic mentioned in the user's prompt
- Match links based on keywords in the user's request (check URL path and anchor text)
- Include category/topic pages if they lead to the right topic
- Exclude links to unrelated topics

Response format (JSON only, no other text):
{{
  "relevant_urls": ["url1", "url2", ...]
}}

If no links are relevant to the specific topic, return: {{"relevant_urls": []}}
"""
    req = LLMRequest(
        prompt=llm_prompt,
        provider="",
        model=model,
        temperature=0.1,
        max_tokens=2048,
        timeout_seconds=30,
    )
    try:
        response = await llm.complete(req)
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?\s*", "", response, flags=re.MULTILINE)
            response = re.sub(r"```\s*$", "", response, flags=re.MULTILINE)
        response = response.strip()
        data = json.loads(response)
        relevant = data.get("relevant_urls", [])
        if isinstance(relevant, list):
            return [str(url) for url in relevant if url]
    except Exception as e:
        logger.warning("llm_link_filter_failed", error=str(e))
    return []


def _to_relative_paths(base_url: str, urls: list[str]) -> list[str]:
    base = urlparse(base_url)
    base_path = (base.path or "").rstrip("/")
    out: list[str] = []
    for u in urls:
        p = urlparse(u)
        if p.netloc and p.netloc != base.netloc:
            continue
        path = p.path or "/"
        if base_path and path.startswith(base_path + base_path + "/"):
            path = base_path + path[len(base_path + base_path) :]
        if base_path and path.startswith(base_path + "/"):
            path = path[len(base_path) :]
        out.append(path)
    seen = set()
    uniq = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


class AgentScraperService:
    def __init__(self, *, scraper: PlaywrightScraper, web_scraper: WebScraperService):
        self._scraper = scraper
        self._web_scraper = web_scraper
        self._settings = get_settings()

    def _build_llm(self, provider: str) -> tuple[LLMRuntime, str]:
        provider = (provider or "ollama").strip().lower()
        if provider == "openai":
            llm = OpenAIAdapter(
                api_key=self._settings.openai_api_key,
                base_url=self._settings.openai_base_url,
            )
            model = self._settings.openai_default_model
            return llm, model
        llm = OllamaAdapter(host=self._settings.ollama_host, port=self._settings.ollama_port)
        model = self._settings.ollama_default_model
        return llm, model

    async def _collect_document_urls_browser_use(
        self,
        *,
        base_url: str,
        prompt: str,
        constraints: AgentConstraints,
    ) -> list[CollectedURL]:
        service_url = (self._settings.browser_use_service_base_url or "").strip().rstrip("/")
        if not service_url:
            raise InvalidInputError("browser_use_service_base_url_missing")

        provider = (constraints.llm_provider or self._settings.llm_provider).strip().lower()
        if provider == "openai":
            model = constraints.llm_model or self._settings.openai_default_model
        else:
            provider = "ollama"
            model = constraints.llm_model or self._settings.ollama_default_model

        payload = {
            "task": _build_browser_use_task(
                base_url=base_url,
                prompt=prompt,
                constraints=constraints,
            ),
            "provider": provider,
            "model": model,
            "temperature": float(self._settings.browser_use_temperature),
            "maxSteps": int(self._settings.browser_use_max_steps),
            "headless": bool(self._settings.browser_use_headless),
        }
        headers = {"Content-Type": "application/json"}
        if self._settings.browser_use_internal_api_key:
            headers["X-Gateway-Key"] = self._settings.browser_use_internal_api_key

        final_text = ""
        success = False
        timeout = aiohttp.ClientTimeout(total=float(self._settings.browser_use_timeout_seconds))
        url = f"{service_url}/api/v1/browser-use/run/stream"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status >= 400:
                    detail = await response.text()
                    raise InvalidInputError(
                        "browser_use_http_error",
                        detail=f"{response.status}: {detail}",
                    )
                buffer = ""
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    buffer += chunk.decode("utf-8", errors="ignore")
                    parts = buffer.split("\n")
                    buffer = parts.pop() if parts else ""
                    for raw_line in parts:
                        line = raw_line.strip("\r")
                        if not line or not line.startswith("data:"):
                            continue
                        payload_text = line[5:].strip()
                        if not payload_text or payload_text == "[DONE]":
                            continue
                        try:
                            event = json.loads(payload_text)
                        except ValueError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        if event.get("type") == "error":
                            raise InvalidInputError(
                                "browser_use_error",
                                detail=str(event.get("error") or "unknown_error"),
                            )
                        if event.get("type") == "result":
                            final_text = str(event.get("finalText") or "")
                            success = bool(event.get("success", False))

        if not final_text:
            raise InvalidInputError("browser_use_no_result")
        if not success:
            logger.warning("browser_use_result_not_success", final_text=final_text[:200])

        urls = _extract_urls_from_text(final_text)
        urls = _filter_browser_use_urls(
            base_url=base_url,
            urls=urls,
            constraints=constraints,
        )
        urls = urls[: int(constraints.max_pages)]
        if not urls:
            raise InvalidInputError("browser_use_collected_0_urls")
        return [
            CollectedURL(
                url=u,
                path=urlparse(u).path or "/",
                title=None,
            )
            for u in urls
        ]

    async def collect_document_urls(
        self,
        *,
        base_url: str,
        prompt: str,
        constraints: AgentConstraints,
        timeout_ms: int,
    ) -> list[CollectedURL]:
        base_url = base_url.strip().rstrip("/")
        seed = base_url
        prompt = (prompt or "").strip()

        if self._settings.phase3_agent_backend.strip().lower() == "browser_use":
            logger.info("phase3_backend_browser_use_enabled", base_url=base_url)
            return await self._collect_document_urls_browser_use(
                base_url=base_url,
                prompt=prompt,
                constraints=constraints,
            )

        llm, default_model = self._build_llm(constraints.llm_provider)
        model = constraints.llm_model or default_model

        seen: set[str] = set()
        queue: list[tuple[str, int]] = [(seed, 0)]
        collected: dict[str, CollectedURL] = {}

        while queue and len(seen) < int(constraints.max_pages):
            url, depth = queue.pop(0)
            url = _canonicalize(url)
            if url in seen:
                continue
            if depth > int(constraints.max_depth):
                continue
            if not _same_or_allowed_domain(url, base_url, constraints.allowed_domains):
                continue
            if _excluded(url, constraints.exclude_patterns):
                continue
            seen.add(url)
            page = await self._scraper.fetch_html(url, timeout_ms=timeout_ms)
            title = _extract_title(page.html)
            snippet = _extract_text_snippet(page.html)
            is_doc_page, page_metadata = _is_likely_document_page(page.html)

            is_relevant = await _ask_llm_is_url_relevant(
                llm=llm,
                prompt=prompt,
                url=url,
                title=title,
                snippet=snippet,
                is_document_page=is_doc_page,
                page_metadata=page_metadata,
                model=model,
            )
            if is_relevant and is_doc_page:
                if url not in collected:
                    collected[url] = CollectedURL(
                        url=url,
                        path=urlparse(url).path or "/",
                        title=title,
                    )

            if depth >= int(constraints.max_depth):
                continue

            links_with_text = _extract_links(page.html, current_url=url)
            if links_with_text:
                relevant_links = await _ask_llm_which_links_relevant(
                    llm=llm,
                    prompt=prompt,
                    current_url=url,
                    page_title=title,
                    page_snippet=snippet,
                    links=links_with_text,
                    model=model,
                )
                base_path = urlparse(base_url).path.rstrip("/") or ""
                for link in relevant_links:
                    link = _canonicalize(link)
                    if link in seen:
                        continue
                    if base_path and not urlparse(link).path.startswith(base_path):
                        continue
                    queue.append((link, depth + 1))

        return [collected[k] for k in sorted(collected.keys())]

    async def scrape_phase3(
        self,
        *,
        job_id: str,
        client_id: str,
        base_url: str,
        prompt: str,
        constraints: AgentConstraints,
        scraping_options: dict[str, object],
    ) -> AsyncIterator[ProgressEvent]:
        yield ProgressEvent(
            job_id=job_id,
            stage=ProgressStage.VALIDATING,
            message="validating_phase3_request",
            is_complete=False,
        )

        urls = await self.collect_document_urls(
            base_url=base_url,
            prompt=prompt,
            constraints=constraints,
            timeout_ms=self._settings.scrape_timeout_ms,
        )
        rel_paths = _to_relative_paths(base_url, [c.url for c in urls])
        if not rel_paths:
            raise InvalidInputError("agent_collected_0_urls")

        config = {
            "base_url": base_url.rstrip("/"),
            "structure": {"phase3": rel_paths},
            "options": scraping_options,
        }
        config_json = json.dumps(config, ensure_ascii=False)

        async for ev in self._web_scraper.scrape_configured(
            job_id=job_id,
            client_id=client_id,
            config_json=config_json,
        ):
            yield ev

