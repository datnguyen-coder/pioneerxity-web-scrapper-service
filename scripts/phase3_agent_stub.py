#!/usr/bin/env python3
"""
Phase 3 demo - "BrowserUse Agent Service" stub.

Purpose:
- Keep the Gateway thin for the demo.
- Given (base_url, prompt, constraints), collect a small/medium list of relevant document URLs.

This uses Ollama LLM to understand the user prompt and intelligently navigate/filter URLs.
"""

from __future__ import annotations

import json
import re
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse

# Ensure repo root is on sys.path so `import scraper.*` resolves when running `python scripts/...`.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
for p in (str(_SRC_DIR), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

from scraper.scraping.playwright_scraper import PlaywrightScraper
from scraper.utils.rate_limiter import DomainRateLimiter
from scraper.utils.robots import RobotsChecker
from scraper.llm.ollama_adapter import OllamaAdapter
from scraper.llm.openai_adapter import OpenAIAdapter
from scraper.llm.runtime import LLMRequest, LLMRuntime


@dataclass(frozen=True)
class AgentConstraints:
    max_depth: int = 2
    max_pages: int = 50
    allowed_domains: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    # LLM provider selection
    llm_provider: str = "ollama"  # "ollama" or "openai"
    # Ollama config
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_model: str = "qwen2.5:3b"
    # OpenAI config
    openai_api_key: str | None = None
    openai_base_url: str | None = None  # For OpenAI-compatible APIs
    openai_model: str = "gpt-4o"  # or "gpt-4", "gpt-3.5-turbo", etc.


@dataclass(frozen=True)
class CollectedURL:
    url: str
    path: str
    title: str | None = None


def _canonicalize(u: str) -> str:
    p = urlparse(u)
    # drop fragment; keep query (some docs are query-based) but normalize common tracking by leaving as-is for demo
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
            # treat invalid regex as substring
            if pat in u:
                return True
    return False


def _is_probable_doc_url(u: str) -> bool:
    # Minimal heuristic for the demo: prefer canonical "article" URLs and common doc patterns.
    p = urlparse(u).path.lower()
    if "/a/" in p:
        return True
    if any(p.endswith(ext) for ext in (".pdf", ".html", ".htm")):
        return True
    # common doc slugs: /docs/, /documentation/, /guide/, /kb/
    if any(seg in p for seg in ("/docs/", "/documentation/", "/guide/", "/kb/")):
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
    # Fallback: super-light extraction
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"<[^>]+>", " ", m.group(1)).strip() or None
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"<[^>]+>", " ", m.group(1)).strip() or None
    return None


def _extract_links(html: str, *, current_url: str) -> list[tuple[str, str]]:
    """Extract links with their anchor text."""
    out: list[tuple[str, str]] = []
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href or href.startswith("#"):
                continue
            if href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
                continue
            anchor_text = (a.get_text(strip=True) or "").strip()[:200]  # Limit length
            full_url = urljoin(current_url, href)
            out.append((full_url, anchor_text))
        return out

    # Fallback: regex-based href extraction (good enough for demo)
    for m in re.finditer(r"""<a\b[^>]*\bhref\s*=\s*["']([^"']+)["']""", html, flags=re.IGNORECASE):
        href = (m.group(1) or "").strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
            continue
        full_url = urljoin(current_url, href)
        out.append((full_url, ""))
    return out


def _extract_keywords_from_prompt(prompt: str) -> list[str]:
    """Extract potential keywords from user prompt for better matching."""
    # Simple keyword extraction: look for quoted phrases or common patterns
    keywords = []
    prompt_lower = prompt.lower()
    
    # Look for "topic X" or "about X" patterns
    import re
    # Match "topic E-mail", "about E-mail", "E-mail documents", etc.
    matches = re.findall(r'(?:topic|about|document|documents?|related to|with topic)\s+([a-z0-9\-\s]+)', prompt_lower)
    for match in matches:
        keywords.extend(match.strip().split())
    
    # Also extract any capitalized words (likely topic names)
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', prompt)
    for cap in capitalized:
        keywords.append(cap.lower())
        # Split compound words like "E-mail" -> ["e", "mail"]
        keywords.extend(re.split(r'[\s\-]+', cap.lower()))
    
    # Remove common stop words
    stop_words = {'the', 'all', 'find', 'get', 'collect', 'scrape', 'document', 'documents', 'topic', 'about', 'with', 'related', 'to'}
    keywords = [k for k in keywords if k and k not in stop_words and len(k) > 2]
    
    # Unique and return
    return list(set(keywords))


def _extract_text_snippet(html: str, max_chars: int = 1000) -> str:
    """Extract a text snippet from HTML for LLM context."""
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]
    # Fallback: remove tags
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _is_likely_document_page(html: str, url: str) -> tuple[bool, dict[str, any]]:
    """
    Heuristic to detect if a page is a document page (with actual content) 
    vs a category/topic page (mostly links).
    
    Returns: (is_document, metadata_dict)
    """
    metadata = {
        "has_main_content": False,
        "text_length": 0,
        "link_count": 0,
        "has_article_tag": False,
        "has_substantial_paragraphs": False,
    }
    
    if BeautifulSoup is None:
        # Fallback: simple text-based check
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        metadata["text_length"] = len(text)
        # If text is substantial (>500 chars) and not mostly links, likely a document
        return len(text) > 500, metadata
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()
    
    # Check for article tag
    article = soup.find("article") or soup.find("main")
    if article:
        metadata["has_main_content"] = True
        metadata["has_article_tag"] = True
        content_area = article
    else:
        # Try to find main content area
        main = soup.find("main")
        if main:
            content_area = main
            metadata["has_main_content"] = True
        else:
            content_area = soup
    
    # Count links in content area
    links = content_area.find_all("a", href=True)
    metadata["link_count"] = len(links)
    
    # Extract text
    text = content_area.get_text(separator=" ", strip=True)
    metadata["text_length"] = len(text)
    
    # Count substantial paragraphs (more than 100 chars)
    paragraphs = content_area.find_all(["p", "div"])
    substantial_paras = sum(1 for p in paragraphs if len(p.get_text(strip=True)) > 100)
    metadata["has_substantial_paragraphs"] = substantial_paras >= 2
    
    # Heuristic: Document page if:
    # 1. Has substantial text (>800 chars) AND
    # 2. Either has article/main tag OR has multiple substantial paragraphs AND
    # 3. Link-to-text ratio is reasonable (not mostly links)
    is_doc = (
        metadata["text_length"] > 800
        and (metadata["has_main_content"] or metadata["has_substantial_paragraphs"])
        and (metadata["link_count"] == 0 or metadata["text_length"] / max(metadata["link_count"], 1) > 50)
    )
    
    return is_doc, metadata


async def _ask_llm_which_links_relevant(
    *,
    llm: OllamaAdapter,
    prompt: str,
    current_url: str,
    page_title: str | None,
    page_snippet: str,
    links: list[tuple[str, str]],
    model: str,
) -> list[str]:
    """Ask LLM which links are relevant to the user prompt."""
    if not links:
        return []

    # Extract keywords from prompt to help LLM
    keywords = _extract_keywords_from_prompt(prompt)
    keywords_hint = f"Keywords to look for: {', '.join(keywords)}" if keywords else ""

    # Build context for LLM, highlighting links that contain keywords
    links_text_parts = []
    for url, anchor_text in links[:50]:  # Limit to 50 links per page
        url_lower = url.lower()
        anchor_lower = (anchor_text or "").lower()
        # Check if URL or anchor contains any keywords
        contains_keyword = any(kw in url_lower or kw in anchor_lower for kw in keywords) if keywords else False
        marker = "⭐ " if contains_keyword else "  "
        links_text_parts.append(f"{marker}{url} (text: {anchor_text[:100] if anchor_text else 'no text'})")
    
    links_text = "\n".join(links_text_parts)

    # Build dynamic examples based on extracted keywords
    keyword_examples = ""
    if keywords:
        keyword_list = ", ".join(f'"{kw}"' for kw in keywords[:5])  # Limit to 5 keywords
        keyword_examples = f"\n- Look for links containing these keywords in URL or anchor text: {keyword_list}"
    
    llm_prompt = f"""You are a web navigation agent. The user wants to: "{prompt}"
{keywords_hint}

CRITICAL: The user is looking for documents about a SPECIFIC TOPIC. You must be very selective.
- ONLY follow links that are clearly about the topic mentioned in the user's request
- Look for links with anchor text or URLs that contain keywords related to the user's topic{keyword_examples}
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

    try:
        req = LLMRequest(
            prompt=llm_prompt,
            provider="ollama",
            model=model,
            temperature=0.1,  # Low temperature for deterministic filtering
            max_tokens=2048,
            timeout_seconds=30,
        )
        response = await llm.complete(req)
        
        # Parse JSON response (Ollama with format: "json" should return valid JSON)
        response = response.strip()
        # Remove markdown code blocks if present
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?\s*", "", response, flags=re.MULTILINE)
            response = re.sub(r"```\s*$", "", response, flags=re.MULTILINE)
        response = response.strip()
        
        data = json.loads(response)
        relevant = data.get("relevant_urls", [])
        if isinstance(relevant, list):
            return [str(url) for url in relevant if url]
        return []
    except Exception as e:
        # Fallback: if LLM fails, return all links (graceful degradation)
        print(f"Warning: LLM filtering failed: {e}. Falling back to all links.", file=sys.stderr)
        return [url for url, _ in links]


async def _ask_llm_is_url_relevant(
    *,
    llm: OllamaAdapter,
    prompt: str,
    url: str,
    title: str | None,
    snippet: str,
    is_document_page: bool,
    page_metadata: dict[str, any],
    model: str,
) -> bool:
    """
    Ask LLM if a URL/content is relevant to the user prompt.
    
    IMPORTANT: The user wants ACTUAL DOCUMENTS, not category/topic pages that just list links.
    """
    page_type = "document page (has substantial content)" if is_document_page else "category/topic page (mostly links)"
    
    # Extract keywords to help LLM understand what to look for
    keywords = _extract_keywords_from_prompt(prompt)
    keywords_hint = f"\nKeywords to match: {', '.join(keywords)}" if keywords else ""
    
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
  "reason": "brief explanation of why it is/isn't relevant to the specific topic"
}}
"""

    try:
        req = LLMRequest(
            prompt=llm_prompt,
            provider="ollama",
            model=model,
            temperature=0.1,
            max_tokens=256,
            timeout_seconds=15,
        )
        response = await llm.complete(req)
        
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?\s*", "", response, flags=re.MULTILINE)
            response = re.sub(r"```\s*$", "", response, flags=re.MULTILINE)
        response = response.strip()
        
        data = json.loads(response)
        return bool(data.get("relevant", False))
    except Exception as e:
        # Fallback: if LLM fails, use heuristic (only collect if it's a document page)
        print(f"Warning: LLM relevance check failed for {url}: {e}. Using heuristic.", file=sys.stderr)
        return is_document_page


async def collect_document_urls(
    *,
    base_url: str,
    prompt: str,
    constraints: AgentConstraints,
    user_agent: str = "pioneerxity-demo-agent/0.1",
    timeout_ms: int = 30_000,
) -> list[CollectedURL]:
    """Collect document URLs using LLM-guided navigation based on user prompt."""
    base_url = base_url.strip().rstrip("/")
    seed = base_url
    prompt = (prompt or "").strip()
    
    # Extract keywords for debugging
    keywords = _extract_keywords_from_prompt(prompt)
    if keywords:
        print(f"[Agent] Extracted keywords from prompt: {', '.join(keywords)}", file=sys.stderr)

    # Initialize LLM adapter based on provider
    provider = constraints.llm_provider.lower()
    if provider == "openai":
        llm = OpenAIAdapter(
            api_key=constraints.openai_api_key,
            base_url=constraints.openai_base_url,
        )
        model = constraints.openai_model
        print(f"[Agent] Using OpenAI provider with model: {model}", file=sys.stderr)
    elif provider == "ollama":
        llm = OllamaAdapter(host=constraints.ollama_host, port=constraints.ollama_port)
        model = constraints.ollama_model
        print(f"[Agent] Using Ollama provider with model: {model}", file=sys.stderr)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'ollama' or 'openai'")

    # DomainRateLimiter takes requests_per_second (RPS). 0.2s delay => ~5 RPS.
    rate_limiter = DomainRateLimiter(requests_per_second=5.0)
    robots = RobotsChecker()
    scraper = PlaywrightScraper(
        default_timeout_ms=timeout_ms,
        user_agent=user_agent,
        rate_limiter=rate_limiter,
        robots_checker=robots,
        respect_robots_txt=True,
    )

    seen: set[str] = set()
    queue: list[tuple[str, int]] = [(seed, 0)]
    collected: dict[str, CollectedURL] = {}
    
    print(f"[Agent] Starting collection with prompt: '{prompt}'", file=sys.stderr)
    print(f"[Agent] Base URL: {base_url}, max_depth={constraints.max_depth}, max_pages={constraints.max_pages}", file=sys.stderr)

    while queue and len(seen) < int(constraints.max_pages):
        url, depth = queue.pop(0)
        url = _canonicalize(url)
        if url in seen:
            continue
        if depth > int(constraints.max_depth):
            print(f"[Agent] Skipping {url} (depth {depth} > max {constraints.max_depth})", file=sys.stderr)
            continue
        if not _same_or_allowed_domain(url, base_url, constraints.allowed_domains):
            print(f"[Agent] Skipping {url} (domain not allowed)", file=sys.stderr)
            continue
        if _excluded(url, constraints.exclude_patterns):
            print(f"[Agent] Skipping {url} (excluded by pattern)", file=sys.stderr)
            continue

        seen.add(url)
        print(f"[Agent] Visiting [{depth}] {url}", file=sys.stderr)
        
        try:
            page = await scraper.fetch_html(url)
            title = _extract_title(page.html)
            snippet = _extract_text_snippet(page.html)
            
            # Check if this is a document page (vs category page)
            is_doc_page, page_metadata = _is_likely_document_page(page.html, url)
            
            print(f"[Agent]   Title: {title or 'N/A'}", file=sys.stderr)
            print(f"[Agent]   Is document page: {is_doc_page} (text={page_metadata.get('text_length', 0)}, links={page_metadata.get('link_count', 0)})", file=sys.stderr)

            # Ask LLM if this page is relevant to the prompt
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
            
            print(f"[Agent]   LLM says relevant: {is_relevant}", file=sys.stderr)

            # IMPORTANT: Only collect if BOTH conditions are met:
            # 1. LLM says it's relevant to the user's prompt (e.g., "E-mail" topic)
            # 2. It's a document page (not just a category page)
            # This ensures we only collect actual documents that match the user's request
            should_collect = is_relevant and is_doc_page
            
            if should_collect:
                if url not in collected:
                    collected[url] = CollectedURL(
                        url=url,
                        path=urlparse(url).path or "/",
                        title=title,
                    )
                    print(f"[Agent]   ✓ Collected: {url} (relevant + document page)", file=sys.stderr)
            else:
                reason = []
                if not is_relevant:
                    reason.append("not relevant to prompt")
                if not is_doc_page:
                    reason.append("not a document page")
                print(f"[Agent]   ✗ Not collected ({', '.join(reason)})", file=sys.stderr)

            # If we've hit max depth, don't explore further
            if depth >= int(constraints.max_depth):
                print(f"[Agent]   Max depth reached, not exploring further", file=sys.stderr)
                continue

            # Extract links and ask LLM which ones to follow
            links_with_text = _extract_links(page.html, current_url=url)
            print(f"[Agent]   Found {len(links_with_text)} links on page", file=sys.stderr)
            
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
                
                print(f"[Agent]   LLM selected {len(relevant_links)} relevant links to follow", file=sys.stderr)
                if relevant_links:
                    print(f"[Agent]   Selected links: {relevant_links[:5]}", file=sys.stderr)  # Show first 5

                # Only queue links that LLM deemed relevant
                base_path = urlparse(base_url).path.rstrip("/") or ""
                queued_count = 0
                skipped_seen = 0
                skipped_path = 0
                for link in relevant_links:
                    original_link = link
                    link = _canonicalize(link)
                    if link in seen:
                        print(f"[Agent]     Skipped {link} (already seen)", file=sys.stderr)
                        skipped_seen += 1
                        continue
                    # Avoid runaway: only traverse within base path prefix when possible
                    link_path = urlparse(link).path
                    if base_path and not link_path.startswith(base_path):
                        print(f"[Agent]     Skipped {link} (path {link_path} doesn't start with {base_path})", file=sys.stderr)
                        skipped_path += 1
                        continue
                    queue.append((link, depth + 1))
                    queued_count += 1
                    print(f"[Agent]     ✓ Queued: {link}", file=sys.stderr)
                
                if skipped_seen > 0 or skipped_path > 0:
                    print(f"[Agent]   Skipped: {skipped_seen} already seen, {skipped_path} path mismatch", file=sys.stderr)
                print(f"[Agent]   Queued {queued_count} new links", file=sys.stderr)
        except Exception as e:
            print(f"[Agent]   ERROR processing {url}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue

    print(f"[Agent] Collection complete. Collected {len(collected)} URLs", file=sys.stderr)
    # Return stable order (by URL)
    return [collected[k] for k in sorted(collected.keys())]


