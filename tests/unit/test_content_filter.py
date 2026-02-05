from __future__ import annotations

from scraper.services.content_filter import ContentFilter


def test_content_filter_removes_noise_and_keeps_main() -> None:
    html = """
    <html>
      <nav>Nav</nav>
      <main>
        <h1>Title</h1>
        <p>{"word " * 250}</p>
      </main>
      <footer>Footer</footer>
    </html>
    """
    f = ContentFilter(default_noise_selectors=["nav", "footer"])
    out = f.filter_html(html, noise_selectors=["nav", "footer"])
    assert "Nav" not in out
    assert "Footer" not in out
    assert "Title" in out


