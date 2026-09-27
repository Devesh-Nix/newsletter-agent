"""Article reader tool: download a page and extract its main text."""

from __future__ import annotations

import logging

import httpx
import trafilatura

from newsletter_agent.config import Settings
from newsletter_agent.models import Article, ArticleContent

log = logging.getLogger(__name__)

#: Below this many characters an extraction is probably a cookie wall or paywall stub.
MIN_USEFUL_CHARS = 400


def fetch_page_text(url: str, settings: Settings) -> str | None:
    """Return the main article text of `url`, or None if it can't be read."""
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": settings.user_agent, "Accept": "text/html,*/*;q=0.8"},
            timeout=settings.http_timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        log.info("Could not fetch %s: %s", url, exc)
        return None

    if "html" not in response.headers.get("content-type", "html"):
        return None
    text = trafilatura.extract(
        response.text,
        url=str(response.url),
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not text or len(text) < MIN_USEFUL_CHARS:
        return None
    return text


def read_article(article: Article, settings: Settings) -> ArticleContent:
    """Full article text when available, otherwise the search snippet.

    Long articles are cut at `settings.article_char_limit`: the lede and first sections
    carry what a two-sentence summary needs, and it keeps summarisation calls cheap.
    """
    text = fetch_page_text(article.url, settings)
    if text:
        return ArticleContent(
            url=article.url, text=text[: settings.article_char_limit], from_full_page=True
        )
    fallback = article.snippet or article.title
    return ArticleContent(url=article.url, text=fallback, from_full_page=False)
