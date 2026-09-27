"""Keyless news-search backends (plus optional Tavily).

Each backend takes a query and returns a list of `Article` records. They raise
`SearchError` on transport/parse failures so callers can fall back or report it.
"""

from __future__ import annotations

from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import feedparser
import httpx

from newsletter_agent.config import Settings
from newsletter_agent.models import Article
from newsletter_agent.utils import (
    article_id,
    clean_text,
    domain_of,
    is_within_window,
    parse_date,
    utc_now,
)

BING_NEWS_RSS = "https://www.bing.com/news/search"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
HN_SEARCH_API = "https://hn.algolia.com/api/v1/search"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"
TAVILY_SEARCH_API = "https://api.tavily.com/search"

SNIPPET_CHARS = 400


class SearchError(RuntimeError):
    """A search backend could not return results."""


def _get(url: str, settings: Settings, **params: object) -> httpx.Response:
    try:
        response = httpx.get(
            url,
            params=params,
            headers={"User-Agent": settings.user_agent, "Accept-Language": "en-US,en;q=0.9"},
            timeout=settings.http_timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SearchError(f"{urlsplit(url).netloc}: {exc}") from exc
    return response


def _bing_interval(days: int) -> str:
    # Bing's freshness filter: 7 = past 24h, 8 = past week, 9 = past month.
    return "7" if days <= 1 else "8" if days <= 7 else "9"


def _unwrap_bing_link(link: str) -> str:
    """Bing wraps article links in an `apiclick.aspx?...&url=<real url>` redirect."""
    real = parse_qs(urlsplit(link).query).get("url")
    return real[0] if real else link


def bing_news(query: str, days: int, max_results: int, settings: Settings) -> list[Article]:
    response = _get(
        BING_NEWS_RSS,
        settings,
        q=query,
        format="rss",
        qft=f'interval="{_bing_interval(days)}"',
        setlang="en-us",
        cc="us",
    )
    feed = feedparser.parse(response.content)
    if feed.bozo and not feed.entries:
        raise SearchError(f"Bing News returned an unreadable feed for {query!r}")

    articles: list[Article] = []
    for entry in feed.entries:
        url = _unwrap_bing_link(entry.get("link", ""))
        if not url.startswith("http"):
            continue
        articles.append(
            Article(
                id=article_id(url),
                title=clean_text(entry.get("title")),
                url=url,
                # Bing labels syndicated copies "Fortune on MSN"; credit the publisher.
                source=clean_text(entry.get("news_source")).removesuffix(" on MSN")
                or domain_of(url),
                published_at=parse_date(entry.get("published")),
                snippet=clean_text(entry.get("summary"), SNIPPET_CHARS),
                origin="bing_news",
                query=query,
            )
        )
    return _recent(articles, days)[:max_results]


def google_news(query: str, days: int, max_results: int, settings: Settings) -> list[Article]:
    response = _get(
        GOOGLE_NEWS_RSS, settings, q=f"{query} when:{days}d", hl="en-US", gl="US", ceid="US:en"
    )
    feed = feedparser.parse(response.content)
    if feed.bozo and not feed.entries:
        raise SearchError(f"Google News returned an unreadable feed for {query!r}")

    articles: list[Article] = []
    for entry in feed.entries:
        source = clean_text(entry.get("source", {}).get("title")) or "Google News"
        title = clean_text(entry.get("title")).removesuffix(f" - {source}")
        url = entry.get("link", "")
        articles.append(
            Article(
                id=article_id(url),
                title=title,
                url=url,
                source=source,
                published_at=parse_date(entry.get("published")),
                # Google's description only repeats the title, so leave the snippet empty.
                snippet="",
                origin="google_news",
                query=query,
            )
        )
    return _recent(articles, days)[:max_results]


def hacker_news(query: str, days: int, max_results: int, settings: Settings) -> list[Article]:
    since = int((utc_now() - timedelta(days=days)).timestamp())
    response = _get(
        HN_SEARCH_API,
        settings,
        query=query,
        tags="story",
        numericFilters=f"created_at_i>{since}",
        hitsPerPage=max_results * 2,
    )
    articles: list[Article] = []
    for hit in response.json().get("hits", []):
        url = hit.get("url") or HN_ITEM_URL.format(hit["objectID"])
        articles.append(
            Article(
                id=article_id(url),
                title=clean_text(hit.get("title")),
                url=url,
                source=domain_of(url) if hit.get("url") else "Hacker News",
                published_at=parse_date(hit.get("created_at")),
                snippet=clean_text(hit.get("story_text"), SNIPPET_CHARS),
                origin="hacker_news",
                points=hit.get("points"),
                comments=hit.get("num_comments"),
                query=query,
            )
        )
    # Algolia ranks by text relevance; surface the stories the community engaged with.
    articles.sort(key=lambda a: a.points or 0, reverse=True)
    return articles[:max_results]


def tavily(query: str, days: int, max_results: int, settings: Settings) -> list[Article]:
    if not settings.tavily_api_key:
        raise SearchError("TAVILY_API_KEY is not configured")
    try:
        response = httpx.post(
            TAVILY_SEARCH_API,
            json={"query": query, "topic": "news", "days": days, "max_results": max_results},
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            timeout=settings.http_timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SearchError(f"Tavily: {exc}") from exc

    articles = [
        Article(
            id=article_id(item["url"]),
            title=clean_text(item.get("title")),
            url=item["url"],
            source=domain_of(item["url"]),
            published_at=parse_date(item.get("published_date")),
            snippet=clean_text(item.get("content"), SNIPPET_CHARS),
            origin="tavily",
            query=query,
        )
        for item in response.json().get("results", [])
        if item.get("url")
    ]
    return _recent(articles, days)[:max_results]


def _recent(articles: list[Article], days: int) -> list[Article]:
    return [a for a in articles if a.title and is_within_window(a.published_at, days)]
