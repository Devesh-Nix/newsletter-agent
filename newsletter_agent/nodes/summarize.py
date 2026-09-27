"""Step 4 — Summarisation: read each selected article and brief it."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from langgraph.runtime import Runtime

from newsletter_agent.state import AgentContext, AgentState, emit
from newsletter_agent.tools import summarize_articles
from newsletter_agent.tools.summarizer import is_fallback
from newsletter_agent.utils import domain_of


def summarize(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    selected = state["selected"]

    emit(runtime, "summarize", f"Reading {len(selected)} selected articles")
    with ThreadPoolExecutor(max_workers=6) as pool:
        contents = list(pool.map(ctx.read_article, selected))
    for article, content in zip(selected, contents, strict=True):
        detail = (
            f"{len(content.text):,} characters of article text"
            if content.from_full_page
            else "page unreadable, using the search snippet"
        )
        emit(
            runtime,
            "summarize",
            f"read_article({domain_of(article.url)}) → {detail}",
            kind="tool",
            url=article.url,
        )

    # Under a request-rate cap, one call for all articles beats one call each.
    batched = ctx.settings.requests_per_minute is not None
    emit(
        runtime,
        "summarize",
        f"Summarising all {len(selected)} articles in one call (rate-limited provider)"
        if batched
        else "Summarising each article for the audience",
    )
    summaries = summarize_articles(
        ctx.llm, selected, contents, state["plan"], ctx.settings.newsletter_name, batched=batched
    )
    for summary in summaries:
        emit(runtime, "summarize", summary.headline, kind="result")
    if failed := sum(is_fallback(s) for s in summaries):
        emit(
            runtime,
            "summarize",
            f"{failed} article(s) could not be summarised by the model; using their snippets",
            kind="warning",
        )
    return {"summaries": {a.id: s for a, s in zip(selected, summaries, strict=True)}}
