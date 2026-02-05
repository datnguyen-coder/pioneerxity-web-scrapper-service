from __future__ import annotations

from scraper.services.discovery_service import _build_hierarchical_structure


def test_build_hierarchy_from_link_graph() -> None:
    base_url = "https://x.com/support"
    start_url = "https://x.com/support"

    pages = [
        # Root listing page (not expected to be included as a leaf)
        {
            "url": "https://x.com/support",
            "title": "Support",
            "parent_url": "",
            "word_count": 50,
            "text_to_html_ratio": 0.2,
            "internal_links_count": 30,
        },
        # Topic listing page
        {
            "url": "https://x.com/support/e-mail",
            "title": "E-Mail",
            "parent_url": "https://x.com/support",
            "word_count": 120,
            "text_to_html_ratio": 0.2,
            "internal_links_count": 25,
        },
        # Subtopic listing page
        {
            "url": "https://x.com/support/e-mail/e-mail-addresses",
            "title": "E-Mail addresses",
            "parent_url": "https://x.com/support/e-mail",
            "word_count": 200,
            "text_to_html_ratio": 0.2,
            "internal_links_count": 12,
        },
        # Two doc pages under the subtopic
        {
            "url": "https://x.com/support/a/doc-1",
            "title": "Doc 1",
            "parent_url": "https://x.com/support/e-mail/e-mail-addresses",
            "word_count": 900,
            "text_to_html_ratio": 0.35,
            "internal_links_count": 2,
        },
        {
            "url": "https://x.com/support/a/doc-2",
            "title": "Doc 2",
            "parent_url": "https://x.com/support/e-mail/e-mail-addresses",
            "word_count": 850,
            "text_to_html_ratio": 0.33,
            "internal_links_count": 1,
        },
    ]

    structure = _build_hierarchical_structure(pages, base_url=base_url, start_url=start_url)

    assert "E-Mail" in structure
    assert "E-Mail addresses" in structure["E-Mail"]
    assert structure["E-Mail"]["E-Mail addresses"] == ["/a/doc-1", "/a/doc-2"]

    # Ensure we didn't accidentally include section pages as leaf docs
    flat = []
    for topic in structure.values():
        for docs in topic.values():
            flat.extend(docs)
    assert "/e-mail" not in flat
    assert "/e-mail/e-mail-addresses" not in flat


def test_fallback_path_clustering_when_no_sections() -> None:
    base_url = "https://x.com/support"
    start_url = "https://x.com/support"

    pages = [
        {
            "url": "https://x.com/support/domain/dns",
            "title": "DNS",
            "parent_url": "",
            "word_count": 800,
            "text_to_html_ratio": 0.3,
            "internal_links_count": 1,
        },
        {
            "url": "https://x.com/support/domain/transfer",
            "title": "Transfer",
            "parent_url": "",
            "word_count": 700,
            "text_to_html_ratio": 0.3,
            "internal_links_count": 1,
        },
        {
            "url": "https://x.com/support/e-mail/setup",
            "title": "Setup",
            "parent_url": "",
            "word_count": 600,
            "text_to_html_ratio": 0.3,
            "internal_links_count": 1,
        },
    ]

    structure = _build_hierarchical_structure(pages, base_url=base_url, start_url=start_url)

    assert "domain" in structure
    assert "pages" in structure["domain"]
    assert structure["domain"]["pages"] == ["/domain/dns", "/domain/transfer"]

    assert "e-mail" in structure
    assert "pages" in structure["e-mail"]
    assert structure["e-mail"]["pages"] == ["/e-mail/setup"]


