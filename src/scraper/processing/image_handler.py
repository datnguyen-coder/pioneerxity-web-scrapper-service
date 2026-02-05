"""Image handling for Phase 1.

Responsibilities:
- Extract image URLs from (filtered) HTML
- Optionally download images
- Optionally rewrite HTML to embed images (data: URIs) or keep links

Rules:
- No storage operations here (upload is handled by service + MinIO client)
- Keep logic deterministic and configurable per job
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class ExtractedImage:
    source_url: str
    resolved_url: str
    filename: str
    content_type: str
    bytes_data: bytes


@dataclass(frozen=True)
class ImageRef:
    """Reference to an image found in HTML."""

    url: str
    alt: str = ""


def _safe_filename(url: str) -> str:
    parsed = urlparse(url)
    name = (parsed.path or "").split("/")[-1] or "image"
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "image"
    if "." not in name:
        name = f"{name}.bin"
    return name


def _guess_content_type(filename: str, default: str = "application/octet-stream") -> str:
    lower = filename.lower()
    if lower.endswith(".avif"):
        return "image/avif"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".svg"):
        return "image/svg+xml"
    return default


def _parse_srcset(srcset: str) -> list[str]:
    """Parse a srcset attribute into a list of candidate URLs (in-order)."""
    out: list[str] = []
    for part in (srcset or "").split(","):
        p = part.strip()
        if not p:
            continue
        # Each entry is like: "url 1x" or "url 640w"
        url = p.split()[0].strip()
        if url:
            out.append(url)
    return out


def _first_image_candidate_url(img_tag, *, base_url: str) -> str:
    """Pick the best-guess source URL for an <img>, preferring <picture><source srcset>."""
    # Prefer <picture><source srcset> (often contains avif/webp on modern sites)
    pic = img_tag.find_parent("picture")
    if pic is not None:
        for source in pic.find_all("source"):
            candidates = _parse_srcset(str(source.get("srcset") or ""))
            if candidates:
                return urljoin(base_url.rstrip("/") + "/", candidates[0].strip())
    # Next: img srcset
    candidates = _parse_srcset(str(img_tag.get("srcset") or ""))
    if candidates:
        return urljoin(base_url.rstrip("/") + "/", candidates[0].strip())
    # Fallback: img src
    src = (img_tag.get("src") or "").strip()
    if src:
        return urljoin(base_url.rstrip("/") + "/", src)
    return ""


class ImageHandler:
    def __init__(self, *, timeout_seconds: int = 15, max_images: int = 50, max_bytes: int = 5 * 1024 * 1024):
        self._timeout_seconds = int(timeout_seconds)
        self._max_images = int(max_images)
        self._max_bytes = int(max_bytes)

    def extract_image_refs(self, html: str, *, base_url: str) -> list[ImageRef]:
        """Extract image URLs (including <source srcset>) + alt text."""
        soup = BeautifulSoup(html, "lxml")
        refs: list[ImageRef] = []

        def add(u: str, alt: str) -> None:
            if not u:
                return
            resolved = urljoin(base_url.rstrip("/") + "/", u.strip())
            refs.append(ImageRef(url=resolved, alt=(alt or "").strip()))

        for img in soup.find_all("img"):
            alt = (img.get("alt") or "").strip()
            src = (img.get("src") or "").strip()
            if src:
                add(src, alt)

            # Also consider img srcset
            for u in _parse_srcset(str(img.get("srcset") or "")):
                add(u, alt)

            # If inside <picture>, also consider <source srcset>
            pic = img.find_parent("picture")
            if pic is not None:
                for source in pic.find_all("source"):
                    for u in _parse_srcset(str(source.get("srcset") or "")):
                        add(u, alt)

        # Dedupe by URL in-order, and cap
        seen: set[str] = set()
        out: list[ImageRef] = []
        for r in refs:
            if r.url in seen:
                continue
            seen.add(r.url)
            out.append(r)
            if len(out) >= self._max_images:
                break
        return out

    def extract_image_urls(self, html: str, *, base_url: str) -> List[str]:
        return [r.url for r in self.extract_image_refs(html, base_url=base_url)]

    async def download_images(self, urls: Iterable[str]) -> List[ExtractedImage]:
        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        out: list[ExtractedImage] = []
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for url in list(urls)[: self._max_images]:
                try:
                    async with session.get(url) as resp:
                        if resp.status >= 400:
                            continue
                        data = await resp.read()
                        if len(data) > self._max_bytes:
                            continue
                        filename = _safe_filename(url)
                        ctype = resp.headers.get("content-type") or _guess_content_type(filename)
                        out.append(
                            ExtractedImage(
                                source_url=url,
                                resolved_url=url,
                                filename=filename,
                                content_type=ctype.split(";")[0].strip(),
                                bytes_data=data,
                            )
                        )
                except Exception:
                    continue
        return out

    def rewrite_html(
        self,
        html: str,
        *,
        base_url: str,
        image_handling: str,
        downloaded: Optional[List[ExtractedImage]] = None,
        uploaded_image_urls: Optional[dict[str, str]] = None,
    ) -> tuple[str, int, int, list[str]]:
        """Rewrite HTML according to image_handling.

        Args:
            image_handling: "embed" | "link_only" | "both" | "embed_or_link"
            downloaded: downloaded images for embedding
            uploaded_image_urls: mapping from original absolute image url -> public/reference url (e.g. MinIO path)

        Returns:
            (rewritten_html, embedded_count, linked_count, image_urls)
        """
        image_urls = self.extract_image_urls(html, base_url=base_url)
        soup = BeautifulSoup(html, "lxml")

        downloaded_by_url = {img.source_url: img for img in (downloaded or [])}

        embedded = 0
        linked = 0

        for img in soup.find_all("img"):
            src = (img.get("src") or "").strip()
            if not src:
                continue
            abs_url = urljoin(base_url.rstrip("/") + "/", src)
            # Preserve both the original <img src> URL and the best candidate from srcset/picture sources.
            # This prevents losing the original URL when we embed to data: URIs.
            candidate_url = _first_image_candidate_url(img, base_url=base_url) or abs_url
            img["data-ws-img-src"] = abs_url
            img["data-ws-img-candidate"] = candidate_url
            alt_text = (img.get("alt") or "").strip()
            should_caption = False

            if image_handling in ("link_only",):
                # Always resolve relative src to absolute so the document remains usable.
                img["src"] = uploaded_image_urls.get(abs_url, abs_url) if uploaded_image_urls else abs_url
                linked += 1
                should_caption = True

            elif image_handling in ("embed", "embed_or_link", "both"):
                found = downloaded_by_url.get(abs_url)
                if found:
                    b64 = base64.b64encode(found.bytes_data).decode("ascii")
                    img["src"] = f"data:{found.content_type};base64,{b64}"
                    embedded += 1
                    should_caption = True
                elif image_handling in ("embed_or_link", "both"):
                    # Fall back to a link when embedding isn't available.
                    img["src"] = uploaded_image_urls.get(abs_url, abs_url) if uploaded_image_urls else abs_url
                    linked += 1
                    should_caption = True
                else:
                    # embed-only and not found -> drop image
                    img.decompose()
                    continue

            # Add inline caption with image URL + alt text (best-effort).
            # Skip if the image is already inside a figure with a figcaption.
            if should_caption:
                fig = img.find_parent("figure")
                if fig is None or fig.find("figcaption") is None:
                    # Avoid duplicating captions
                    nxt = img.find_next_sibling()
                    if not (getattr(nxt, "get", None) and "ws-img-meta" in (nxt.get("class") or [])):
                        meta = soup.new_tag("div")
                        meta["class"] = ["ws-img-meta"]
                        meta["style"] = "font-size: 12px; color: #666; margin-top: 2px; margin-bottom: 10px;"
                        # Image URL
                        url_div = soup.new_tag("div")
                        a = soup.new_tag("a", href=abs_url)
                        a.string = abs_url
                        url_div.append("Image: ")
                        url_div.append(a)
                        meta.append(url_div)
                        if candidate_url and candidate_url != abs_url:
                            srcset_div = soup.new_tag("div")
                            a2 = soup.new_tag("a", href=candidate_url)
                            a2.string = candidate_url
                            srcset_div.append("SourceSet: ")
                            srcset_div.append(a2)
                            meta.append(srcset_div)
                        # Alt
                        if alt_text:
                            alt_div = soup.new_tag("div")
                            alt_div.string = f"Alt: {alt_text}"
                            meta.append(alt_div)
                        img.insert_after(meta)

        return str(soup), embedded, linked, image_urls


