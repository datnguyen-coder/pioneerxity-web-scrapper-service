from __future__ import annotations

from scraper.utils.quality import assess_quality


def test_assess_quality_counts_words_and_ratio() -> None:
    html = "<main><h1>Hello</h1><p>" + ("word " * 120) + "</p></main>"
    q = assess_quality(html)
    assert q.word_count >= 100
    assert q.text_length > 0
    assert q.text_to_html_ratio > 0


