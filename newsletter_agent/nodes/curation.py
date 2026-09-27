"""Step 3 — Curation: dedupe the research pool and let the editor pick the top stories."""

from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
from uuid import uuid4

from langgraph.runtime import Runtime

from newsletter_agent.llm import structured_output
from newsletter_agent.models import Article, CurationResult, NewsletterPlan, RankedPick
from newsletter_agent.nodes.common import bullet_list, today
from newsletter_agent.prompts import CURATOR_PROMPT
from newsletter_agent.state import AgentContext, AgentState, emit
from newsletter_agent.utils import dedupe_articles, is_within_window, title_similarity


class ResearchError(RuntimeError):
    """Research produced nothing usable."""


def curate(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    plan = state["plan"]
    candidates = list(state.get("candidates", {}).values())

    if not candidates:
        emit(
            runtime,
            "curate",
            "The model finished without searching; running the planned queries directly",
            kind="warning",
        )
        candidates = _direct_search(plan, runtime)

    pool = [
        a
        for a in dedupe_articles(candidates)
        if is_within_window(a.published_at, plan.time_window_days)
    ]
    if not pool:
        raise ResearchError(
            f"No stories about {plan.topic!r} from the last {plan.time_window_days} days were "
            "found. Check your network connection or broaden the goal."
        )
    pool = shortlist(pool, ctx.settings.max_candidates_for_curation)
    target = min(plan.target_article_count, len(pool))
    emit(
        runtime,
        "curate",
        f"Ranking {len(pool)} unique candidates (from {len(candidates)} raw results) "
        f"to pick the top {target}",
    )

    curator = CURATOR_PROMPT | structured_output(ctx.llm, CurationResult)
    result = curator.invoke(
        {
            "newsletter_name": ctx.settings.newsletter_name,
            "topic": plan.topic,
            "audience": plan.audience,
            "today": today(),
            "target": target,
            "criteria": bullet_list(plan.selection_criteria),
            "research_notes": state.get("research_notes") or "(none)",
            "count": len(pool),
            "candidates": format_candidates(pool),
        }
    )
    picks = validate_picks(result.picks, pool, target)
    by_id = {a.id: a for a in pool}
    selected = [by_id[p.article_id] for p in picks]

    emit(runtime, "curate", result.coverage_notes, kind="thought")
    for pick, article in zip(picks, selected, strict=True):
        emit(
            runtime,
            "curate",
            f"[{pick.category}] {article.title} ({article.source}) — {pick.rationale}",
            kind="result",
        )
    return {"selected": selected, "picks": picks}


def shortlist(pool: list[Article], limit: int) -> list[Article]:
    """Cap the pool for the LLM while keeping every source represented.

    Items are grouped by the tool that found them (news by recency, Hacker News by
    points) and interleaved, so no single source crowds out the others.
    """
    groups: dict[str, list[Article]] = defaultdict(list)
    for article in pool:
        groups[article.origin].append(article)
    for origin, items in groups.items():
        if origin == "hacker_news":
            items.sort(key=lambda a: a.points or 0, reverse=True)
        else:
            items.sort(
                key=lambda a: a.published_at.timestamp() if a.published_at else 0, reverse=True
            )
    interleaved = [a for row in zip_longest(*groups.values()) for a in row if a is not None]
    return interleaved[:limit]


def format_candidates(pool: list[Article]) -> str:
    lines = []
    for a in pool:
        meta = f"{a.source} · {a.date_label}"
        if a.points is not None:
            meta += f" · HN {a.points} pts, {a.comments or 0} comments"
        line = f"[{a.id}] {a.title} — {meta}"
        if a.snippet:
            line += f"\n    {a.snippet[:240]}"
        lines.append(line)
    return "\n".join(lines)


def validate_picks(picks: list[RankedPick], pool: list[Article], target: int) -> list[RankedPick]:
    """Never trust ids blindly: drop unknown/duplicate/same-story picks, then top up."""
    by_id = {a.id: a for a in pool}
    valid: list[RankedPick] = []
    for pick in picks:
        article = by_id.get(pick.article_id.strip("[] "))
        if article is None or _same_story(article, valid, by_id):
            continue
        valid.append(pick.model_copy(update={"article_id": article.id}))

    # If the model returned too few usable picks, fill from the shortlist order.
    for article in pool:
        if len(valid) >= target:
            break
        if not _same_story(article, valid, by_id):
            valid.append(
                RankedPick(
                    article_id=article.id,
                    category="News",
                    relevance=5,
                    rationale="Added to reach the target story count.",
                )
            )
    return valid[:target]


def _same_story(article: Article, picks: list[RankedPick], by_id: dict[str, Article]) -> bool:
    return any(
        p.article_id == article.id
        or title_similarity(article.title, by_id[p.article_id].title) >= 0.6
        for p in picks
    )


def _direct_search(plan: NewsletterPlan, runtime: Runtime[AgentContext]) -> list[Article]:
    """Fallback for models that won't call tools: run every planned query on every tool."""
    found: list[Article] = []
    for tool in runtime.context.research_tools:
        for query in plan.search_queries:
            emit(runtime, "research", f"{tool.name}({query!r})", kind="tool", tool=tool.name)
            call = {
                "type": "tool_call",
                "id": uuid4().hex,
                "name": tool.name,
                "args": {"query": query, "days": plan.time_window_days},
            }
            try:
                found.extend(tool.invoke(call).artifact or [])
            except Exception as exc:
                emit(runtime, "research", f"{tool.name} failed: {exc}", kind="warning")
    return found
