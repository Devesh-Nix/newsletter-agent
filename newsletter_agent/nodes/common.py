"""Formatting helpers shared by several nodes."""

from __future__ import annotations

from newsletter_agent.models import NewsletterDraft
from newsletter_agent.state import AgentState
from newsletter_agent.utils import utc_now


def today() -> str:
    """E.g. 'Sunday, September 27, 2026' — models need the date to judge recency."""
    now = utc_now()
    return f"{now:%A, %B} {now.day}, {now.year}"


def bullet_list(items: list[str], empty: str = "None") -> str:
    return "\n".join(f"- {item}" for item in items) if items else empty


def format_briefs(state: AgentState) -> str:
    """The selected stories with their summaries: the writer's input and the critic's
    ground truth."""
    categories = {pick.article_id: pick.category for pick in state.get("picks", [])}
    blocks = []
    for article in state["selected"]:
        summary = state["summaries"][article.id]
        blocks.append(
            f"[{article.id}] category: {categories.get(article.id, 'News')}\n"
            f"  source: {article.source}, {article.date_label}\n"
            f"  headline: {summary.headline}\n"
            f"  summary: {summary.summary}\n"
            f"  why it matters: {summary.why_it_matters or '(not provided)'}"
        )
    return "\n\n".join(blocks)


def format_draft(draft: NewsletterDraft) -> str:
    """A readable rendering of the draft for review prompts."""
    items = "\n\n".join(
        f"{i}. [{item.article_id}] ({item.category}) {item.headline}\n"
        f"   {item.summary}\n"
        f"   Why it matters: {item.why_it_matters}"
        for i, item in enumerate(draft.items, start=1)
    )
    return (
        f"SUBJECT: {draft.subject}\n"
        f"PREHEADER: {draft.preheader}\n\n"
        f"INTRO: {draft.intro}\n\n"
        f"STORIES:\n{items}\n\n"
        f"THE BIG PICTURE: {draft.big_picture}\n\n"
        f"SIGN-OFF: {draft.sign_off}"
    )
