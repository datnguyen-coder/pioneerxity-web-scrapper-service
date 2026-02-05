"""Phase 2 discovery service (business logic).

Responsibilities:
- Validate discovery request
- Crawl a bounded set of pages starting from base_url (deterministic; LLM does NOT decide topics)
- Use LLM (Ollama) to suggest noise filtering rules (selectors) to reduce boilerplate (header/footer/nav/search/ads)
- Persist discovered pages (Phase-1-compatible `structure`) for approval
- On approval, run Phase 1 pipeline over ALL discovered pages using suggested filtering rules
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..config.settings import get_settings
from ..domain.errors import ContentProcessingError, InvalidInputError, InvalidURLError, NetworkTimeoutError
from ..domain.models import DiscoveryProgressEvent, DiscoveryStage, ErrorCode
from ..llm.runtime import LLMRequest, LLMRuntime
from ..observability.logger import get_logger
from ..scraping.playwright_scraper import PlaywrightScraper
from ..storage.repositories import DiscoveryRepository
from ..utils.validators import sanitize_path_segment
from ..utils.validators import is_valid_http_url

logger = get_logger(__name__)


@dataclass(frozen=True)
class DiscoveryResult:
    job_id: str
    base_url: str
    structure: dict[str, Any]
    scraping_options: dict[str, Any]


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _normalize_scope_and_start(url: str) -> tuple[str, str]:
    """Return (scope_base_url, start_url).

    Phase 2 accepts `base_url` as either:
    - a section/root URL (scope == start), e.g. https://x.com/support
    - a specific document URL, e.g. https://x.com/support/a/some-article

    For document URLs, we treat the *parent directory* as scope so that relative
    paths don't double-join (e.g. /support/support/...) and the crawl can pick up siblings.
    """
    u = url.strip().rstrip("/")
    p = urlparse(u)
    if not p.scheme or not p.netloc:
        return u, u
    path = (p.path or "/").rstrip("/")
    if path in ("", "/"):
        return u, u
    # Heuristic: treat deep paths (>= 3 segments) as document-like URLs.
    # Example: /support/a/<slug> or /support/favoriten/<slug>
    segs = [s for s in path.split("/") if s]
    if len(segs) >= 3:
        parent = "/".join(segs[:-1])
        scope = f"{p.scheme}://{p.netloc}/{parent}"
        return scope, u
    return u, u

def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from a potentially chatty LLM response."""
    if not text or "{" not in text:
        snippet = (text or "").strip().replace("\n", " ")
        raise ContentProcessingError("llm_response_missing_json", detail=f"first={snippet[:200]}")
    start = text.find("{")
    end = text.rfind("}")
    if end <= start:
        snippet = (text or "").strip().replace("\n", " ")
        raise ContentProcessingError("llm_response_missing_json", detail=f"first={snippet[:200]}")
    blob = text[start : end + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ContentProcessingError("llm_response_invalid_json", detail=str(e)) from e
    if not isinstance(obj, dict):
        raise ContentProcessingError("llm_response_json_not_object")
    return obj


def _validate_structure(node: Any) -> None:
    def walk(n: Any, path: list[str]) -> None:
        if isinstance(n, list):
            for i, x in enumerate(n):
                if not isinstance(x, str):
                    raise InvalidInputError(
                        "structure leaf must be a list of strings",
                        detail=f"path={'/'.join(path)} idx={i} type={type(x).__name__}",
                    )
            return
        if isinstance(n, dict):
            for k, v in n.items():
                if not isinstance(k, str):
                    raise InvalidInputError(
                        "structure keys must be strings",
                        detail=f"path={'/'.join(path)} key_type={type(k).__name__}",
                    )
                walk(v, path + [k])
            return
        raise InvalidInputError(
            "structure must be a nested object of dicts and lists",
            detail=f"path={'/'.join(path)} type={type(n).__name__}",
        )

    walk(node, [])


def _coerce_to_structure(node: Any) -> Any:
    """Coerce common LLM output shapes into Phase-1-compatible `structure`.

    We accept (and normalize):
    - leaf string -> [string]
    - {"documents": [...]} -> [...]
    - {"urls": [...]} -> [...]
    - {"subcategories": {...}} -> {...}
    - {"categories": {...}} -> {...}
    """
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        # Keep only string leaves; coerce nested leaf strings if any.
        out: list[Any] = []
        for x in node:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict) or isinstance(x, list):
                # Avoid weird nesting; try to coerce and flatten if it becomes list of strings.
                cx = _coerce_to_structure(x)
                if isinstance(cx, list):
                    out.extend([y for y in cx if isinstance(y, str)])
        return out
    if isinstance(node, dict):
        # Common wrapper keys
        for key in ("structure", "categories", "subcategories"):
            if key in node and isinstance(node.get(key), dict):
                return _coerce_to_structure(node[key])
        # Common leaf keys
        for key in ("documents", "urls", "links"):
            if key in node and isinstance(node.get(key), list):
                return _coerce_to_structure(node[key])
        # Otherwise treat as nested categories
        return {str(k): _coerce_to_structure(v) for k, v in node.items()}
    return node


def _normalize_structure_paths(node: Any, *, base_url: str) -> Any:
    """Normalize structure leaves to relative paths under base_url."""
    parsed_base = urlparse(base_url)
    base_host = parsed_base.netloc.lower()

    if isinstance(node, list):
        out: list[str] = []
        for x in node:
            if not isinstance(x, str):
                continue
            s = x.strip()
            if not s:
                continue
            if s.startswith("http://") or s.startswith("https://"):
                p = urlparse(s)
                if p.netloc.lower() != base_host:
                    # Drop cross-domain URLs.
                    continue
                s = p.path or "/"
                if p.query:
                    s = f"{s}?{p.query}"
            if not s.startswith("/"):
                s = "/" + s.lstrip("/")
            out.append(s)
        return out
    if isinstance(node, dict):
        return {str(k): _normalize_structure_paths(v, base_url=base_url) for k, v in node.items()}
    return node


def _collect_relative_candidates(pages: list[dict[str, Any]], *, base_url: str) -> list[str]:
    """Collect relative path candidates under base_url from crawled pages."""
    parsed_base = urlparse(base_url)
    base_host = parsed_base.netloc.lower()
    base_path_prefix = (parsed_base.path or "/").rstrip("/") or "/"

    rels: list[str] = []
    for p in pages:
        u = str(p.get("url") or "").strip()
        if not u:
            continue
        pu = urlparse(u)
        if pu.netloc.lower() != base_host:
            continue
        path = (pu.path or "/").rstrip("/") or "/"
        if base_path_prefix != "/" and not path.startswith(base_path_prefix):
            continue
        # Drop query variants of the scope root (e.g. /support?c=email) since they produce
        # broken Phase-1 joins (base_url + "support?c=email" -> /support/support?...).
        if path.rstrip("/") == base_path_prefix.rstrip("/") and pu.query:
            continue
        rel = path
        if pu.query:
            rel = f"{rel}?{pu.query}"
        if not rel.startswith("/"):
            rel = "/" + rel.lstrip("/")

        # IMPORTANT: leaf paths must be relative *under base_url*.
        # Example: base_url=https://x.com/support and a page path=/support/a/123
        # -> leaf should be /a/123 (NOT /support/a/123), otherwise Phase 1 urljoin
        # produces /support/support/a/123.
        if base_path_prefix != "/" and rel.startswith(base_path_prefix.rstrip("/") + "/"):
            rel = rel[len(base_path_prefix.rstrip("/")) :]
            if not rel.startswith("/"):
                rel = "/" + rel.lstrip("/")
        rels.append(rel)

    # Keep all candidates in stable order.
    out = rels
    # Remove base page itself if present
    base_rel = base_path_prefix if base_path_prefix.startswith("/") else f"/{base_path_prefix}"
    out = [r for r in out if r.rstrip("/") != base_rel.rstrip("/")]
    # Deduplicate while preserving order
    seen: set[str] = set()
    dedup: list[str] = []
    for r in out:
        if r in seen:
            continue
        seen.add(r)
        dedup.append(r)
    return dedup


def _count_document_paths(structure: Any) -> int:
    if isinstance(structure, list):
        return sum(1 for x in structure if isinstance(x, str) and x.strip())
    if isinstance(structure, dict):
        return sum(_count_document_paths(v) for v in structure.values())
    return 0


def _fallback_structure_from_candidates(candidates: list[str]) -> dict[str, Any]:
    """Build a minimal Phase-1-compatible structure from candidate relative paths."""
    if not candidates:
        return {}
    # Keep the structure simple and stable: one bucket containing all relative paths.
    return {"discovered": {"paths": candidates}}


def _canonical_url(u: str) -> str:
    s = (u or "").strip()
    if not s:
        return ""
    s = s.split("#", 1)[0]
    # Normalize trailing slash (except for scheme://host/)
    try:
        p = urlparse(s)
        if p.scheme and p.netloc:
            path = p.path or "/"
            if path != "/" and path.endswith("/"):
                s = s.rstrip("/")
    except Exception:
        pass
    return s


def _rel_path_under_base(abs_url: str, *, base_url: str) -> str:
    """Return a Phase-1-compatible leaf path for urljoin(base_url + '/', leaf.lstrip('/'))."""
    abs_url = _canonical_url(abs_url)
    parsed_base = urlparse(base_url)
    base_host = parsed_base.netloc.lower()
    base_path_prefix = (parsed_base.path or "/").rstrip("/") or "/"

    pu = urlparse(abs_url)
    if pu.netloc.lower() != base_host:
        return ""
    path = (pu.path or "/").rstrip("/") or "/"
    if base_path_prefix != "/" and not path.startswith(base_path_prefix):
        return ""
    # Drop query variants of the scope root (e.g. /support?c=email)
    if path.rstrip("/") == base_path_prefix.rstrip("/") and pu.query:
        return ""
    rel = path
    if pu.query:
        rel = f"{rel}?{pu.query}"
    if base_path_prefix != "/" and rel.startswith(base_path_prefix.rstrip("/") + "/"):
        rel = rel[len(base_path_prefix.rstrip("/")) :]
        if not rel.startswith("/"):
            rel = "/" + rel.lstrip("/")
    if not rel.startswith("/"):
        rel = "/" + rel.lstrip("/")
    return rel


def _is_section_page(p: dict[str, Any]) -> bool:
    """Heuristic: section/index pages tend to link to many internal pages and have low text content.

    IMPORTANT: doc pages can also contain many links (related articles, ToC, etc). We bias towards
    classifying high-word-count pages as docs to avoid collapsing everything into "sections".
    """
    # Prefer content-only link count (excludes header/nav/footer/aside), fall back to raw link count.
    try:
        links = int(p.get("content_internal_links_count") or p.get("internal_links_count") or 0)
    except Exception:
        links = 0
    try:
        wc = int(p.get("word_count") or 0)
    except Exception:
        wc = 0
    rel = str(p.get("rel_path") or "").strip()
    segs = [s for s in rel.split("?", 1)[0].split("/") if s] if rel else []

    # Heuristic: article-like pages are almost never "sections".
    # This is intentionally generic (short marker segment like /a/<slug>), but it matches CYON well.
    if len(segs) >= 2 and len(segs[0]) == 1:
        return False

    # Heuristic: shallow paths tend to be section/index pages when they link out heavily.
    # Example: /support/<topic> or /support/<topic>/<subtopic>
    if len(segs) <= 2 and links >= 8 and wc <= 2000:
        return True

    # Default: section pages have many links + relatively low content.
    # (Docs can also have many links, so we require low-ish word count.)
    return (links >= 20 and wc <= 700) or (links >= 12 and wc <= 400) or (links >= 8 and wc <= 250)


def _unique_key(base: str, used: set[str]) -> str:
    k = sanitize_path_segment(base)
    if k not in used:
        used.add(k)
        return k
    i = 2
    while True:
        kk = f"{k}-{i}"
        if kk not in used:
            used.add(kk)
            return kk
        i += 1


def _key_from_page(p: dict[str, Any], *, fallback: str) -> str:
    h1 = str(p.get("h1") or "").strip()
    if h1:
        return h1
    t = str(p.get("title") or "").strip()
    if t:
        return t
    return fallback


def _build_hierarchical_structure(
    pages: list[dict[str, Any]],
    *,
    base_url: str,
    start_url: str,
) -> dict[str, Any]:
    """Build a Phase-1-compatible 2-level structure: {topic: {subtopic: [leaf_paths]}}.

    Strategy:
    - Use link-graph parenting from deterministic crawl (parent_url).
    - Detect "section" pages (topic/subtopic) via internal link density.
    - Assign non-section pages (docs) to nearest subtopic/topic ancestor.
    - Fallback: URL path clustering when section pages cannot be inferred.
    """
    if not pages:
        return {}

    # Use the scope base_url as the logical root. This keeps discovery stable even when the client
    # passes an article URL as base_url (we still want to infer topics from the section root).
    root = _canonical_url(base_url)
    by_url: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for p in pages:
        u = _canonical_url(str(p.get("url") or ""))
        if not u:
            continue
        if u in by_url:
            continue
        by_url[u] = p
        order.append(u)

    # Build parent chain map.
    parent_of: dict[str, str] = {}
    for u, p in by_url.items():
        parent = _canonical_url(str(p.get("parent_url") or ""))
        if parent:
            parent_of[u] = parent

    # Determine section pages.
    is_section: dict[str, bool] = {u: _is_section_page(p) for u, p in by_url.items()}
    # Never classify the root as a section leaf container.
    if root in is_section:
        is_section[root] = True

    # Topic pages are section pages directly under root.
    topic_urls = {u for u, par in parent_of.items() if par == root and is_section.get(u, False)}
    # If the root page provided explicit topic hints, prefer those as stable topic roots.
    topic_name_override: dict[str, str] = {}
    root_page = by_url.get(root) or {}
    root_topics = root_page.get("root_topics") if isinstance(root_page, dict) else None
    if isinstance(root_topics, list):
        for t in root_topics:
            try:
                rel = str(t.get("rel_path") or "").strip()
                abs_u = str(t.get("url") or "").strip()
                title = str(t.get("title") or "").strip()
                if not abs_u and rel:
                    abs_u = urljoin(root, rel.lstrip("/"))
                abs_u = _canonical_url(abs_u)
                if not abs_u:
                    continue
                topic_urls.add(abs_u)
                if title:
                    topic_name_override[abs_u] = title
            except Exception:
                continue
    # If we couldn't find any topics, fall back to URL path clustering.
    if not topic_urls:
        used_topics: set[str] = set()
        used_subs: dict[str, set[str]] = {}
        topic_key_by_raw: dict[str, str] = {}
        sub_key_by_raw: dict[tuple[str, str], str] = {}
        out: dict[str, Any] = {}
        for u in order:
            p = by_url[u]
            if is_section.get(u, False) and u != root:
                continue
            rel = _rel_path_under_base(u, base_url=base_url)
            if not rel:
                continue
            segs = [s for s in rel.split("?")[0].split("/") if s]
            topic_raw = segs[0] if len(segs) >= 1 else "root"
            # Heuristic: treat 2-segment paths as "topic/doc" and group them under a stable bucket.
            # Use a second segment as subtopic only when we have at least 3 segments: "topic/subtopic/doc".
            sub_raw = segs[1] if len(segs) >= 3 else "pages"
            topic = topic_key_by_raw.get(topic_raw)
            if not topic:
                topic = _unique_key(topic_raw, used_topics)
                topic_key_by_raw[topic_raw] = topic
            used_subs.setdefault(topic, set())
            sub = sub_key_by_raw.get((topic, sub_raw))
            if not sub:
                sub = _unique_key(sub_raw, used_subs[topic])
                sub_key_by_raw[(topic, sub_raw)] = sub
            out.setdefault(topic, {}).setdefault(sub, [])
            if rel not in out[topic][sub]:
                out[topic][sub].append(rel)
        return out

    # Subtopic pages are section pages directly under a topic page.
    subtopic_urls = {u for u, par in parent_of.items() if par in topic_urls and is_section.get(u, False)}

    def nearest_ancestor_in(u: str, candidates: set[str]) -> str:
        cur = u
        seen: set[str] = set()
        while True:
            par = parent_of.get(cur)
            if not par or par in seen:
                return ""
            if par in candidates:
                return par
            seen.add(par)
            cur = par

    # Map URLs to (topic_url, subtopic_url)
    assigned: dict[str, tuple[str, str]] = {}
    for u in order:
        if u == root:
            continue
        # Skip section/index pages as leaf docs.
        if is_section.get(u, False):
            continue
        sub = nearest_ancestor_in(u, subtopic_urls) if subtopic_urls else ""
        if sub:
            topic = nearest_ancestor_in(u, topic_urls) or parent_of.get(sub, "")
            if topic in topic_urls:
                assigned[u] = (topic, sub)
                continue
        topic = nearest_ancestor_in(u, topic_urls)
        if topic:
            assigned[u] = (topic, "")

    used_topics: set[str] = set()
    used_subs: dict[str, set[str]] = {}
    topic_key_by_url: dict[str, str] = {}
    sub_key_by_url: dict[str, str] = {}
    default_sub_key_by_topic: dict[str, str] = {}

    # Pre-assign stable topic/subtopic keys
    for tu in sorted(topic_urls):
        p = by_url.get(tu) or {}
        fallback = urlparse(tu).path.split("/")[-1] or "topic"
        preferred = topic_name_override.get(tu)
        topic_key_by_url[tu] = _unique_key(preferred or _key_from_page(p, fallback=fallback), used_topics)
        used_subs[topic_key_by_url[tu]] = set()
    for su in sorted(subtopic_urls):
        p = by_url.get(su) or {}
        fallback = urlparse(su).path.split("/")[-1] or "subtopic"
        topic_u = parent_of.get(su, "")
        topic_k = topic_key_by_url.get(topic_u) or "topic"
        sub_key_by_url[su] = _unique_key(_key_from_page(p, fallback=fallback), used_subs.setdefault(topic_k, set()))

    # Build output in crawl order for deterministic lists.
    out: dict[str, Any] = {}
    for u in order:
        if u not in assigned:
            continue
        topic_u, sub_u = assigned[u]
        rel = _rel_path_under_base(u, base_url=base_url)
        if not rel:
            continue
        topic_k = topic_key_by_url.get(topic_u)
        if not topic_k:
            continue
        if sub_u:
            sub_k = sub_key_by_url.get(sub_u)
        else:
            sub_k = None
        if not sub_k:
            # Put under a stable bucket when no subtopic can be inferred.
            if topic_k not in default_sub_key_by_topic:
                used_subs.setdefault(topic_k, set())
                default_sub_key_by_topic[topic_k] = _unique_key("pages", used_subs[topic_k])
            sub_k = default_sub_key_by_topic[topic_k]
        out.setdefault(topic_k, {}).setdefault(sub_k, [])
        if rel not in out[topic_k][sub_k]:
            out[topic_k][sub_k].append(rel)

    return out


class DiscoveryService:
    def __init__(self, *, scraper: PlaywrightScraper, llm: LLMRuntime, discovery_repo: DiscoveryRepository):
        self._scraper = scraper
        self._llm = llm
        self._repo = discovery_repo
        self._settings = get_settings()

    async def discover_structure(
        self,
        *,
        job_id: str,
        client_id: str,
        base_url: str,
        discovery_options: dict[str, Any],
        llm_options: dict[str, Any],
        scraping_options: dict[str, Any],
    ) -> AsyncIterator[DiscoveryProgressEvent]:
        if not base_url or not base_url.strip():
            raise InvalidInputError("base_url is required")
        raw_base_url = base_url.strip().rstrip("/")
        if not is_valid_http_url(raw_base_url):
            raise InvalidURLError(f"Invalid URL: {raw_base_url}")

        scope_base_url, start_url = _normalize_scope_and_start(raw_base_url)

        max_depth = _clamp_int(
            discovery_options.get("max_depth", 2),
            0,
            self._settings.discovery_max_depth_cap,
        )
        max_pages = _clamp_int(
            discovery_options.get("max_pages", 50),
            1,
            self._settings.discovery_max_pages_cap,
        )

        allowed_domains = discovery_options.get("allowed_domains") or []
        exclude_patterns = discovery_options.get("exclude_patterns") or []
        quality_thresholds = discovery_options.get("quality_thresholds") or {}
        # Discovery-time quality thresholds are intentionally more permissive than Phase 1.
        # We only allow tightening (higher min_word_count / higher min_text_to_html_ratio), never loosening.
        min_word_count = int(getattr(self._settings, "discovery_min_word_count", 0))
        min_text_to_html_ratio = float(getattr(self._settings, "discovery_min_text_to_html_ratio", 0.0))
        try:
            if "min_word_count" in quality_thresholds and quality_thresholds["min_word_count"] is not None:
                min_word_count = max(min_word_count, int(quality_thresholds["min_word_count"]))
            if "min_text_to_html_ratio" in quality_thresholds and quality_thresholds["min_text_to_html_ratio"] is not None:
                min_text_to_html_ratio = max(min_text_to_html_ratio, float(quality_thresholds["min_text_to_html_ratio"]))
        except Exception:
            # Ignore malformed quality thresholds rather than crashing.
            min_word_count = int(getattr(self._settings, "discovery_min_word_count", 0))
            min_text_to_html_ratio = float(getattr(self._settings, "discovery_min_text_to_html_ratio", 0.0))
        compiled_excludes = []
        for p in exclude_patterns:
            try:
                compiled_excludes.append(re.compile(str(p)))
            except re.error:
                # Ignore invalid regex patterns rather than crashing.
                continue

        llm_provider = (llm_options.get("provider") or "ollama").strip().lower()
        if llm_provider != "ollama":
            raise InvalidInputError("only ollama provider is supported in Phase 2")

        llm_model = (llm_options.get("model") or self._settings.ollama_default_model).strip()
        temperature = float(llm_options.get("temperature") or self._settings.llm_temperature_default)
        max_tokens = _clamp_int(llm_options.get("max_tokens") or self._settings.llm_max_tokens_cap, 1, self._settings.llm_max_tokens_cap)
        timeout_seconds = _clamp_int(
            llm_options.get("timeout_seconds") or self._settings.llm_timeout_seconds_cap,
            1,
            self._settings.llm_timeout_seconds_cap,
        )

        yield DiscoveryProgressEvent(
            job_id=job_id,
            stage=DiscoveryStage.VALIDATING,
            message="validating_discovery_request",
            max_pages=max_pages,
        )
        logger.info(
            "discovery_started",
            job_id=job_id,
            base_url=scope_base_url,
            start_url=start_url,
            max_depth=max_depth,
            max_pages=max_pages,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

        # Persist initial discovery job record (structure filled later)
        await self._repo.create_job(
            job_id=job_id,
            client_id=client_id,
            base_url=scope_base_url,
            discovery_options_json=json.dumps(
                {
                    "max_depth": max_depth,
                    "max_pages": max_pages,
                    "allowed_domains": allowed_domains,
                    "exclude_patterns": exclude_patterns,
                    "start_url": start_url,
                    "quality_thresholds": {
                        "min_word_count": min_word_count,
                        "min_text_to_html_ratio": min_text_to_html_ratio,
                    },
                },
                ensure_ascii=False,
            ),
            scraping_options_json=json.dumps(scraping_options, ensure_ascii=False),
            llm_provider=llm_provider,
            llm_model=llm_model,
            max_pages=max_pages,
        )
        await self._repo.mark_running(job_id)

        try:
            pages: list[dict[str, Any]] = []
            async for ev, page in self._crawl_pages_streaming(
                job_id=job_id,
                scope_base_url=scope_base_url,
                start_url=start_url,
                max_depth=max_depth,
                max_pages=max_pages,
                allowed_domains=allowed_domains,
                excludes=compiled_excludes,
            ):
                yield ev
                if page is not None:
                    pages.append(page)

            yield DiscoveryProgressEvent(
                job_id=job_id,
                stage=DiscoveryStage.SUMMARIZING,
                message="summarizing_pages",
                pages_crawled=len(pages),
                max_pages=max_pages,
            )

            # Phase 2 quality filtering: filter obviously low-quality pages before building structure.
            def _passes_quality(p: dict[str, Any]) -> bool:
                try:
                    wc = int(p.get("word_count") or 0)
                    ratio = float(p.get("text_to_html_ratio") or 0.0)
                    return wc >= min_word_count and ratio >= min_text_to_html_ratio
                except Exception:
                    return False

            good_pages = [p for p in pages if _passes_quality(p)]
            dropped = len(pages) - len(good_pages)
            logger.info(
                "discovery_quality_filter_applied",
                job_id=job_id,
                pages_total=len(pages),
                pages_kept=len(good_pages),
                pages_dropped=dropped,
                min_word_count=min_word_count,
                min_text_to_html_ratio=min_text_to_html_ratio,
            )
            # Discovery should not hard-fail if quality gates are too strict (or the site is JS-heavy).
            # If too many pages were dropped, fall back to using raw crawled pages for structure generation.
            if len(pages) > 0 and (len(good_pages) == 0 or len(good_pages) < max(5, int(0.1 * len(pages)))):
                logger.warning(
                    "discovery_quality_filter_dropped_too_many_pages_falling_back_to_raw_pages",
                    job_id=job_id,
                    pages_total=len(pages),
                    pages_kept=len(good_pages),
                    min_word_count=min_word_count,
                    min_text_to_html_ratio=min_text_to_html_ratio,
                )
                good_pages = pages

            # Deterministic discovery: build a Phase-1-compatible hierarchy from the link graph.
            structure = _build_hierarchical_structure(good_pages, base_url=scope_base_url, start_url=start_url)
            # Safety net: if hierarchy inference fails, fall back to a single bucket of paths.
            if not structure or _count_document_paths(structure) == 0:
                candidates = _collect_relative_candidates(good_pages, base_url=scope_base_url)
                # Ensure the user-provided start_url is always included as a document candidate.
                candidates_start = _collect_relative_candidates([{"url": start_url}], base_url=scope_base_url)
                for c in candidates_start:
                    if c not in candidates:
                        candidates.insert(0, c)
                structure = _fallback_structure_from_candidates(candidates)
            if not structure or _count_document_paths(structure) == 0:
                raise InvalidInputError("discovered structure contains 0 documents")
            _validate_structure(structure)
            structure_json = json.dumps(structure, ensure_ascii=False)

            # LLM is used ONLY for noise filtering suggestions (not for deciding topics).
            yield DiscoveryProgressEvent(
                job_id=job_id,
                stage=DiscoveryStage.LLM_ANALYZING,
                message="llm_suggesting_noise_filters",
                pages_crawled=len(pages),
                max_pages=max_pages,
            )
            suggested_noise = await self._suggest_noise_selectors(
                base_url=scope_base_url,
                pages=pages,
                llm_provider=llm_provider,
                llm_model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )

            # IMPORTANT: LLM is suggestion-only. Persist suggestions separately so the gateway can decide
            # whether to apply them. We keep `noise_selectors` as provided by the client.
            merged_options = dict(scraping_options or {})
            existing_noise = merged_options.get("noise_selectors") or []
            if not isinstance(existing_noise, list):
                existing_noise = []

            safe_suggested: list[str] = []
            seen: set[str] = set()
            for s in list(suggested_noise):
                if not isinstance(s, str):
                    continue
                ss = s.strip()
                if not ss or ss in seen:
                    continue
                # Never allow selectors that could remove main content containers.
                if ss in (
                    "main",
                    "article",
                    "body",
                    "html",
                    "#content",
                    ".content",
                    ".documentation",
                    ".docs",
                    ".main-content",
                    ".markdown-body",
                ):
                    continue
                seen.add(ss)
                safe_suggested.append(ss)

            merged_options["noise_selectors"] = [
                str(x).strip() for x in existing_noise if isinstance(x, str) and str(x).strip()
            ]
            merged_options["suggested_noise_selectors"] = safe_suggested
            await self._repo.update_scraping_options(
                job_id, scraping_options_json=json.dumps(merged_options, ensure_ascii=False)
            )

            await self._repo.mark_ready_for_approval(job_id, discovered_structure_json=structure_json)
            logger.info(
                "discovery_ready_for_approval",
                job_id=job_id,
                pages_crawled=len(pages),
            )

            yield DiscoveryProgressEvent(
                job_id=job_id,
                stage=DiscoveryStage.READY_FOR_APPROVAL,
                message="structure_ready_for_approval",
                pages_crawled=len(pages),
                max_pages=max_pages,
                discovered_structure_json=structure_json,
                is_complete=True,
                success=True,
            )
        except asyncio.CancelledError:
            await self._repo.mark_cancelled(job_id, error_message="client_cancelled")
            logger.info("discovery_cancelled", job_id=job_id)
            raise
        except NetworkTimeoutError as e:
            # Preserve deterministic error mapping for LLM/runtime/network timeouts.
            await self._repo.mark_failed(job_id, error_code=ErrorCode.NETWORK_TIMEOUT, error_message=str(e))
            logger.warning(
                "discovery_failed",
                job_id=job_id,
                error_code=ErrorCode.NETWORK_TIMEOUT.value,
                error=str(e),
            )
            raise
        except (InvalidInputError, InvalidURLError) as e:
            code = ErrorCode.INVALID_INPUT if isinstance(e, InvalidInputError) else ErrorCode.INVALID_URL
            await self._repo.mark_failed(job_id, error_code=code, error_message=str(e))
            logger.warning("discovery_failed", job_id=job_id, error_code=code.value, error=str(e))
            raise
        except ContentProcessingError as e:
            await self._repo.mark_failed(job_id, error_code=ErrorCode.CONTENT_PROCESSING_ERROR, error_message=str(e))
            logger.warning(
                "discovery_failed",
                job_id=job_id,
                error_code=ErrorCode.CONTENT_PROCESSING_ERROR.value,
                error=str(e),
            )
            raise
        except Exception as e:
            await self._repo.mark_failed(job_id, error_code=ErrorCode.INTERNAL_ERROR, error_message=str(e))
            logger.error("discovery_failed", job_id=job_id, error_code=ErrorCode.INTERNAL_ERROR.value, error=str(e))
            raise ContentProcessingError("discovery_internal_error", detail=str(e)) from e

    async def get_discovered(self, job_id: str):
        if not job_id:
            raise InvalidInputError("job_id is required")
        return await self._repo.get_job(job_id)

    async def approve_and_build_config(self, *, job_id: str, approved_structure_json: str | None) -> DiscoveryResult:
        rec = await self.get_discovered(job_id)
        if rec is None:
            raise InvalidInputError("discovery job not found")
        if not rec.discovered_structure_json:
            raise InvalidInputError("discovered structure is not ready")

        structure_json = (approved_structure_json or "").strip() or rec.discovered_structure_json
        try:
            structure = json.loads(structure_json)
        except json.JSONDecodeError as e:
            raise InvalidInputError("approved_structure_json must be valid JSON", detail=str(e)) from e
        if not isinstance(structure, dict):
            raise InvalidInputError("approved_structure_json must be a JSON object")
        _validate_structure(structure)

        await self._repo.mark_approved(job_id, approved_structure_json=json.dumps(structure, ensure_ascii=False))

        scraping_opts = {}
        try:
            scraping_opts = json.loads(rec.scraping_options_json or "{}")
        except Exception:
            scraping_opts = {}
        if not isinstance(scraping_opts, dict):
            scraping_opts = {}

        return DiscoveryResult(job_id=job_id, base_url=rec.base_url, structure=structure, scraping_options=scraping_opts)

    async def _crawl_pages_streaming(
        self,
        *,
        job_id: str,
        scope_base_url: str,
        start_url: str,
        max_depth: int,
        max_pages: int,
        allowed_domains: list[str],
        excludes: list[re.Pattern],
    ) -> AsyncIterator[Tuple[DiscoveryProgressEvent, Optional[dict[str, Any]]]]:
        parsed_base = urlparse(scope_base_url)
        base_domain = parsed_base.netloc.lower()
        base_path_prefix = (parsed_base.path or "/").rstrip("/") or "/"
        allow_set = {d.lower().strip() for d in allowed_domains if str(d).strip()}

        def allowed(u: str) -> bool:
            try:
                p = urlparse(u)
            except Exception:
                return False
            if p.scheme not in ("http", "https"):
                return False
            host = (p.netloc or "").lower()
            if allow_set:
                if host not in allow_set:
                    return False
            else:
                if host != base_domain:
                    return False
            # Keep discovery scoped to the base path (prevents wandering into marketing pages).
            path = (p.path or "/").rstrip("/") or "/"
            if base_path_prefix != "/" and not path.startswith(base_path_prefix):
                return False
            for rx in excludes:
                if rx.search(u):
                    return False
            return True

        visited: set[str] = set()
        q: deque[tuple[str, int, str]] = deque()
        # Seed both the start_url and scope root to ensure broad discovery even when the
        # client passes a deep article URL as "base_url".
        start_u = _canonical_url(start_url)
        root_u = _canonical_url(scope_base_url)
        if root_u:
            q.append((root_u, 0, ""))
            visited.add(root_u)
        if start_u and start_u not in visited:
            q.append((start_u, 0, ""))
            visited.add(start_u)

        out: list[dict[str, Any]] = []

        while q and len(out) < max_pages:
            url, depth, parent_url = q.popleft()
            if depth > max_depth:
                continue
            yield (
                DiscoveryProgressEvent(
                job_id=job_id,
                stage=DiscoveryStage.CRAWLING,
                message="crawling_page",
                current_url=url,
                pages_crawled=len(out),
                max_pages=max_pages,
                ),
                None,
            )
            try:
                page = await self._scraper.fetch_html(url, timeout_ms=self._settings.scrape_timeout_ms)
            except Exception as e:
                logger.warning("discovery_fetch_failed", url=url, error=str(e))
                continue

            html = page.html or ""
            soup = BeautifulSoup(html, "html.parser")
            title = (soup.title.get_text(strip=True) if soup.title else "").strip()
            h1 = ""
            try:
                h1_el = soup.select_one("h1")
                if h1_el:
                    h1 = h1_el.get_text(" ", strip=True).strip()
            except Exception:
                h1 = ""
            text = soup.get_text(" ", strip=True)
            if len(text) > 4000:
                text = text[:4000]

            # Lightweight quality metrics for discovery-time filtering.
            try:
                word_count = len([w for w in text.split() if w])
            except Exception:
                word_count = 0
            try:
                text_to_html_ratio = (len(text) / max(1, len(html))) if html else 0.0
            except Exception:
                text_to_html_ratio = 0.0

            def _select_primary_sections(doc: BeautifulSoup) -> set:
                """Pick primary content sections for seed links on the root page.

                This is a generic heuristic: prefer early sections with many meaningful links,
                while de-emphasizing forms, contact/utility blocks, and status-like lists.
                """
                try:
                    main = doc.find("main") or doc.find(attrs={"role": "main"}) or doc.body or doc
                    sections = list(main.find_all("section", recursive=True))
                except Exception:
                    return set()
                if not sections:
                    return set()

                total = len(sections)
                cutoff = max(1, int(total * 0.6))
                candidates = sections[:cutoff] if sections[:cutoff] else sections
                date_re = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")

                scored: list[tuple[float, int]] = []
                for idx, sec in enumerate(candidates):
                    try:
                        links = sec.find_all("a", href=True)
                        if not links:
                            continue
                        link_count = len(links)
                        link_text_len = sum(len(a.get_text(" ", strip=True)) for a in links)
                        text = sec.get_text(" ", strip=True)
                        headings = len(sec.find_all(["h1", "h2", "h3"]))
                        forms = len(sec.find_all("form"))
                        tel_mailto = sum(
                            1
                            for a in links
                            if str(a.get("href") or "").startswith(("tel:", "mailto:"))
                        )
                        date_like = len(date_re.findall(text))
                        score = (
                            link_count * 2
                            + headings * 3
                            + min(link_text_len / 40.0, 10.0)
                            - forms * 5
                            - tel_mailto * 5
                            - date_like * 2
                        )
                        scored.append((score, idx))
                    except Exception:
                        continue

                if not scored:
                    return set()
                scored.sort(reverse=True)
                top_n = min(3, max(1, len(scored) // 3))
                keep = {idx for _, idx in scored[:top_n]}
                return {candidates[i] for i in keep}

            # Determine primary sections early so root-topic hints can be captured.
            primary_sections: set = set()
            if depth == 0:
                primary_sections = _select_primary_sections(soup)

            page_summary = {
                "url": url,
                "parent_url": parent_url,
                "depth": depth,
                "title": title,
                "h1": h1,
                "text": text,
                "word_count": word_count,
                "text_to_html_ratio": float(text_to_html_ratio),
            }
            # Precompute relative path under the discovery scope for better heuristics in hierarchy inference.
            # Example: base_url=https://x.com/support and url=https://x.com/support/a/doc -> /a/doc
            page_summary["rel_path"] = _rel_path_under_base(url, base_url=scope_base_url)

            def _is_primary_anchor(tag) -> bool:
                if depth != 0 or not primary_sections:
                    return True
                try:
                    for parent in tag.parents:
                        if parent in primary_sections:
                            return True
                except Exception:
                    return False
                return False

            def _is_noise_anchor(tag) -> bool:
                """True if anchor is inside boilerplate containers (header/nav/footer/aside/sidebar)."""
                try:
                    for parent in tag.parents:
                        name = (getattr(parent, "name", "") or "").lower()
                        if name in ("header", "nav", "footer", "aside"):
                            return True
                        classes = parent.get("class") if hasattr(parent, "get") else None
                        if classes:
                            for c in classes:
                                cc = str(c).lower()
                                if any(
                                    k in cc
                                    for k in (
                                        "nav",
                                        "breadcrumb",
                                        "sidebar",
                                        "menu",
                                        "footer",
                                        "header",
                                        "search",
                                    )
                                ):
                                    return True
                except Exception:
                    return False
                return False

            # Root-page topic hints for hierarchy (generic: shallow links in primary sections).
            if depth == 0 and primary_sections:
                root_topics: list[dict[str, str]] = []
                for a in soup.select("a[href]"):
                    if _is_noise_anchor(a):
                        continue
                    if not _is_primary_anchor(a):
                        continue
                    href = str(a.get("href") or "").strip()
                    if not href or href.startswith("#") or href.startswith(("mailto:", "javascript:", "tel:")):
                        continue
                    abs_url = _canonical_url(urljoin(url, href))
                    if not abs_url or not allowed(abs_url):
                        continue
                    rel = _rel_path_under_base(abs_url, base_url=scope_base_url)
                    if not rel or "?" in rel:
                        continue
                    segs = [s for s in rel.split("/") if s]
                    if len(segs) != 1:
                        continue
                    title_text = a.get_text(" ", strip=True)
                    if not title_text:
                        continue
                    root_topics.append({"rel_path": rel, "title": title_text, "url": abs_url})
                if root_topics:
                    page_summary["root_topics"] = root_topics

            out.append(page_summary)
            await self._repo.update_progress(job_id, pages_crawled=len(out))
            yield (
                DiscoveryProgressEvent(
                    job_id=job_id,
                    stage=DiscoveryStage.CRAWLING,
                    message="crawled_page",
                    current_url=url,
                    pages_crawled=len(out),
                    max_pages=max_pages,
                ),
                page_summary,
            )

            # Extract links for next layer (+ collect internal link count for hierarchy inference)
            internal_links: set[str] = set()
            content_links: set[str] = set()

            for a in soup.select("a[href]"):
                href = a.get("href")
                if not href:
                    continue
                href = str(href).strip()
                if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                    continue
                abs_url = _canonical_url(urljoin(url, href))
                if not abs_url:
                    continue
                if not allowed(abs_url):
                    continue
                internal_links.add(abs_url)
                if not _is_noise_anchor(a):
                    content_links.add(abs_url)
                # On the root page, only enqueue links from primary content sections.
                if not _is_primary_anchor(a):
                    continue
                if abs_url in visited:
                    continue
                visited.add(abs_url)
                q.append((abs_url, depth + 1, url))

            page_summary["internal_links_count"] = len(internal_links)
            page_summary["content_internal_links_count"] = len(content_links)

        return

    async def _suggest_noise_selectors(
        self,
        *,
        base_url: str,
        pages: list[dict[str, Any]],
        llm_provider: str,
        llm_model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
    ) -> list[str]:
        """Ask LLM for noise selectors. Best-effort; failure falls back to defaults only."""
        # Keep prompt bounded: only include up to N pages.
        sample = pages[: min(5, len(pages))]
        lines: list[str] = []
        for i, p in enumerate(sample, start=1):
            lines.append(f"PAGE {i}")
            lines.append(f"URL: {p.get('url','')}")
            if p.get("title"):
                lines.append(f"TITLE: {p.get('title')}")
            # Use text summary only; selector suggestions should be generic.
            lines.append(f"TEXT_SNIPPET: {p.get('text','')[:1000]}")
            lines.append("")

        prompt = (
            "You are helping configure a web scraper to remove boilerplate elements.\n"
            f"Base URL: {base_url}\n\n"
            "Return ONLY valid JSON with this shape:\n"
            "{\n"
            '  "noise_selectors": ["header", "footer", "nav", "..."]\n'
            "}\n\n"
            "Rules:\n"
            "- noise_selectors must be an array of CSS selectors.\n"
            "- Focus on removing: header, footer, nav, sidebar, breadcrumbs, ads, cookie banners, search bars.\n"
            "- Prefer generic selectors (tag names and common class/id patterns).\n"
            "- Do NOT include selectors that would remove the main article content.\n\n"
            "Page samples:\n"
            + "\n".join(lines)
        )

        try:
            llm_text = await self._llm.complete(
                LLMRequest(
                    prompt=prompt,
                    provider=llm_provider,
                    model=llm_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
            )
            obj = _extract_json_object(llm_text)
            noise = obj.get("noise_selectors", [])
            if isinstance(noise, list):
                out: list[str] = []
                seen = set()
                for s in noise:
                    if not isinstance(s, str):
                        continue
                    ss = s.strip()
                    if not ss or ss in seen:
                        continue
                    seen.add(ss)
                    out.append(ss)
                return out
        except Exception as e:
            logger.warning("llm_noise_selector_suggestion_failed", error=str(e))
        return []


