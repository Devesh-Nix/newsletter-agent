"""Newsletter generator tool: turn a structured draft into HTML, Markdown and plain text.

Rendering is deterministic on purpose. The LLM decides *what* to say, the templates
decide *how it looks*, so the output is always well-formed, email-client-safe HTML and
every link comes from the research data, never from generated text.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from newsletter_agent.models import Article, NewsletterDraft, RenderedNewsletter
from newsletter_agent.utils import utc_now

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
UNSUBSCRIBE_URL = "https://example.com/unsubscribe"


def _safe_url(url: str) -> str:
    """Only allow http(s) links; anything else (javascript:, data:) becomes inert."""
    return url if urlsplit(url).scheme in {"http", "https"} else "#"


def _md_link_text(text: str) -> str:
    return text.replace("[", r"\[").replace("]", r"\]")


def _environment(autoescape: bool) -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html.j2"]) if autoescape else False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["md_link_text"] = _md_link_text
    return env


_HTML_ENV = _environment(autoescape=True)
_TEXT_ENV = _environment(autoescape=False)


def period_label(days: int, end: datetime | None = None) -> str:
    """E.g. 'Sep 20 – Sep 27, 2026'."""
    end = end or utc_now()
    start = end - timedelta(days=days)
    return f"{start:%b} {start.day} – {end:%b} {end.day}, {end.year}"


def render_newsletter(
    draft: NewsletterDraft,
    articles: Mapping[str, Article],
    *,
    newsletter_name: str,
    period_days: int,
    issue_date: datetime | None = None,
) -> RenderedNewsletter:
    """Render the draft in all three formats. Items whose article is unknown are dropped."""
    items = []
    for item in draft.items:
        article = articles.get(item.article_id)
        if article is None:
            continue
        items.append(
            {
                "headline": item.headline.strip().rstrip("."),
                "summary": item.summary.strip(),
                "why_it_matters": item.why_it_matters.strip(),
                "category": item.category.strip() or "News",
                "url": _safe_url(article.url),
                "source": article.source,
                "date_label": article.date_label,
            }
        )

    context = {
        "newsletter_name": newsletter_name,
        "subject": draft.subject.strip(),
        "preheader": draft.preheader.strip(),
        "period_label": period_label(period_days, issue_date),
        "intro": draft.intro.strip(),
        "items": items,
        "big_picture": draft.big_picture.strip(),
        "sign_off": draft.sign_off.strip(),
        "unsubscribe_url": UNSUBSCRIBE_URL,
    }
    return RenderedNewsletter(
        subject=context["subject"],
        preheader=context["preheader"],
        html=_HTML_ENV.get_template("newsletter.html.j2").render(context),
        markdown=_TEXT_ENV.get_template("newsletter.md.j2").render(context),
        text=_TEXT_ENV.get_template("newsletter.txt.j2").render(context),
    )
