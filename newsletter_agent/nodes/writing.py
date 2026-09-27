"""Step 5 — Writing: draft (or revise) the newsletter from the story briefs."""

from __future__ import annotations

from langgraph.runtime import Runtime

from newsletter_agent.llm import structured_output
from newsletter_agent.models import DraftItem, NewsletterDraft
from newsletter_agent.nodes.common import bullet_list, format_briefs, today
from newsletter_agent.prompts import REVISION_REQUEST, WRITER_PROMPT
from newsletter_agent.state import AgentContext, AgentState, emit
from newsletter_agent.tools.renderer import period_label


def write(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    plan = state["plan"]
    draft_number = state.get("drafts_written", 0) + 1
    feedback = revision_feedback(state)

    if draft_number == 1:
        emit(runtime, "write", "Writing the first draft")
        revision_request = ""
    else:
        emit(runtime, "write", f"Revising the draft (draft {draft_number})")
        revision_request = REVISION_REQUEST.format(
            previous_draft=state["draft"].model_dump_json(indent=2),
            feedback=bullet_list(feedback),
        )

    writer = WRITER_PROMPT | structured_output(ctx.llm, NewsletterDraft)
    draft = writer.invoke(
        {
            "newsletter_name": ctx.settings.newsletter_name,
            "topic": plan.topic,
            "audience": plan.audience,
            "tone": plan.tone,
            "today": today(),
            "period": period_label(plan.time_window_days),
            "briefs": format_briefs(state),
            "revision_request": revision_request,
        }
    )
    draft, restored = reconcile_items(draft, state)
    if restored:
        emit(
            runtime,
            "write",
            f"The draft skipped {len(restored)} selected stories; restored them from the briefs",
            kind="warning",
        )

    emit(
        runtime,
        "write",
        f'Draft {draft_number}: "{draft.subject}" ({len(draft.items)} stories)',
        kind="result",
    )
    return {"draft": draft, "drafts_written": draft_number, "human_feedback": ""}


def revision_feedback(state: AgentState) -> list[str]:
    """Human feedback takes priority; otherwise use the latest self-critique."""
    if state.get("human_feedback"):
        return [f"From the human editor: {state['human_feedback']}"]
    reviews = state.get("reviews", [])
    if not reviews:
        return []
    last = reviews[-1]
    return [*last.automated_issues, *last.critique.revision_instructions] or last.critique.issues


def reconcile_items(draft: NewsletterDraft, state: AgentState) -> tuple[NewsletterDraft, list[str]]:
    """Keep items tied to real selected articles, and restore any the model dropped.

    Returns the repaired draft and the ids of restored items.
    """
    selected_ids = [a.id for a in state["selected"]]
    categories = {p.article_id: p.category for p in state.get("picks", [])}

    items: list[DraftItem] = []
    seen: set[str] = set()
    for item in draft.items:
        article_id = item.article_id.strip("[] ")
        if article_id in selected_ids and article_id not in seen:
            seen.add(article_id)
            items.append(item.model_copy(update={"article_id": article_id}))

    restored = [article_id for article_id in selected_ids if article_id not in seen]
    for article_id in restored:
        summary = state["summaries"][article_id]
        items.append(
            DraftItem(
                article_id=article_id,
                headline=summary.headline,
                summary=summary.summary,
                why_it_matters=summary.why_it_matters,
                category=categories.get(article_id, "News"),
            )
        )
    return draft.model_copy(update={"items": items}), restored
