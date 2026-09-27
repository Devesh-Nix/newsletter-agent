"""Small, pure helpers shared by tools and nodes (no network, no LLM)."""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from newsletter_agent.models import Article

_TRACKING_PARAMS = re.compile(r"^(utm_|fbclid$|gclid$|mc_|ref$|ref_src$|cmpid$|ocid$|sr_share$)")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its new of on or the to with "
    "how why what after over into says said will this that".split()
)


def normalize_url(url: str) -> str:
    """Canonicalise a URL so the same story from different links dedupes cleanly."""
    parts = urlsplit(url.strip())
    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k.lower()))
    )
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("https", host, path, query, ""))


def article_id(url: str) -> str:
    """Short, stable id for an article, derived from its normalised URL."""
    return hashlib.sha1(normalize_url(url).encode()).hexdigest()[:8]


def domain_of(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def clean_text(text: str | None, max_chars: int | None = None) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    if not text:
        return ""
    cleaned = _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text))).strip()
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return cleaned


def parse_date(value: str | None) -> datetime | None:
    """Parse RFC-822 (RSS) or ISO-8601 dates into timezone-aware UTC datetimes."""
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC)


def is_within_window(published_at: datetime | None, days: int, now: datetime | None = None) -> bool:
    """Undated items are kept (benefit of the doubt); dated ones must be recent."""
    if published_at is None:
        return True
    now = now or utc_now()
    # One day of slack absorbs timezone differences between sources.
    return now - timedelta(days=days + 1) <= published_at <= now + timedelta(days=1)


def title_tokens(title: str) -> set[str]:
    return {w for w in _WORD_RE.findall(title.lower()) if w not in _STOPWORDS and len(w) > 2}


def title_similarity(a: str, b: str) -> float:
    """Jaccard similarity of meaningful title words (0..1)."""
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# Direct links we can read beat aggregator redirects; richer metadata breaks ties.
_ORIGIN_PRIORITY = {"tavily": 3, "bing_news": 3, "hacker_news": 2, "google_news": 1}


def _richness(article: Article) -> tuple[int, int, int]:
    return (
        _ORIGIN_PRIORITY.get(article.origin, 0),
        int(article.published_at is not None),
        len(article.snippet),
    )


def dedupe_articles(articles: Iterable[Article], threshold: float = 0.6) -> list[Article]:
    """Collapse exact-URL duplicates and near-identical headlines, keeping the richest copy."""
    by_id: dict[str, Article] = {}
    for article in articles:
        existing = by_id.get(article.id)
        if existing is None or _richness(article) > _richness(existing):
            by_id[article.id] = article

    kept: list[Article] = []
    for article in sorted(by_id.values(), key=_richness, reverse=True):
        if all(title_similarity(article.title, other.title) < threshold for other in kept):
            kept.append(article)
    return kept


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + "…"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "newsletter"
