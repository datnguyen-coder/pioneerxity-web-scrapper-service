from __future__ import annotations

from scraper.processing.image_handler import ImageHandler


def test_extract_image_urls_resolves_relative() -> None:
    h = ImageHandler()
    html = '<main><img src="/img/a.png"/><img src="b.jpg"/></main>'
    urls = h.extract_image_urls(html, base_url="https://example.com/docs")
    assert "https://example.com/img/a.png" in urls
    assert "https://example.com/docs/b.jpg" in urls


def test_extract_image_urls_includes_srcset_and_picture_sources() -> None:
    h = ImageHandler()
    html = """
    <main>
      <picture>
        <source srcset="https://media.cyon.ch/x/a.avif 1x, https://media.cyon.ch/x/b.avif 2x" type="image/avif" />
        <source srcset="/rel/c.webp 1x" type="image/webp" />
        <img src="/fallback.png" alt="Example alt" />
      </picture>
    </main>
    """
    urls = h.extract_image_urls(html, base_url="https://www.cyon.ch/support")
    assert "https://www.cyon.ch/fallback.png" in urls
    assert "https://media.cyon.ch/x/a.avif" in urls
    assert "https://www.cyon.ch/rel/c.webp" in urls


def test_rewrite_html_adds_alt_caption_and_url() -> None:
    h = ImageHandler()
    html = '<main><p>Hi</p><img src="/img/a.png" alt="Alt text"/></main>'
    rewritten, embedded, linked, urls = h.rewrite_html(
        html,
        base_url="https://example.com/docs",
        image_handling="link_only",
        downloaded=None,
    )
    assert embedded == 0
    assert linked == 1
    assert "https://example.com/img/a.png" in urls
    assert "ws-img-meta" in rewritten
    assert "Image:" in rewritten
    assert "Alt: Alt text" in rewritten


def test_rewrite_html_caption_prefers_picture_srcset_candidate() -> None:
    h = ImageHandler()
    html = """
    <main>
      <picture>
        <source srcset="https://media.cyon.ch/sup/a/namen-meines-webhostings-andern02.avif 1x" type="image/avif" />
        <img src="/fallback.png" alt="Example alt" />
      </picture>
    </main>
    """
    rewritten, _, _, _ = h.rewrite_html(
        html,
        base_url="https://www.cyon.ch/support",
        image_handling="link_only",
        downloaded=None,
    )
    # Caption always includes the original <img src> URL.
    assert "https://www.cyon.ch/fallback.png" in rewritten
    # And also includes the srcset candidate (if different).
    assert "https://media.cyon.ch/sup/a/namen-meines-webhostings-andern02.avif" in rewritten


