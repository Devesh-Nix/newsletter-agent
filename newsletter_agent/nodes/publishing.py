"""Steps 7-9 — Output: render the newsletter, optional human sign-off, then send."""

from __future__ import annotations

from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from newsletter_agent.models import HumanDecision
from newsletter_agent.state import AgentContext, AgentMode, AgentState, emit
from newsletter_agent.tools import render_newsletter, send_newsletter


def render(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    articles = {a.id: a for a in state["selected"]}
    rendered = render_newsletter(
        state["draft"],
        articles,
        newsletter_name=ctx.settings.newsletter_name,
        period_days=state["plan"].time_window_days,
    )
    emit(
        runtime,
        "render",
        f"render_newsletter → HTML ({len(rendered.html) // 1024} KB), Markdown and plain text",
        kind="tool",
    )
    return {"rendered": rendered}


def route_after_render(state: AgentState) -> Literal["review_draft", "publish"]:
    """The autonomy toggle: a human signs off only in human-in-the-loop mode."""
    return "review_draft" if state["mode"] == AgentMode.HUMAN_IN_THE_LOOP else "publish"


def review_draft(state: AgentState) -> Command[Literal["publish", "write", "__end__"]]:
    """Human-in-the-loop checkpoint: approve, request changes or reject before sending."""
    reviews = state.get("reviews", [])
    decision = HumanDecision.model_validate(
        interrupt(
            {
                "checkpoint": "draft_review",
                "title": "Approve the newsletter before it is sent",
                "subject": state["rendered"].subject,
                "preheader": state["rendered"].preheader,
                "markdown": state["rendered"].markdown,
                "html": state["rendered"].html,
                "review": reviews[-1].model_dump() if reviews else None,
            }
        )
    )
    if decision.action == "approve":
        return Command(goto="publish")
    if decision.action == "revise" and decision.feedback.strip():
        return Command(goto="write", update={"human_feedback": decision.feedback.strip()})
    if decision.action == "revise":  # nothing concrete to change
        return Command(goto="publish")
    return Command(goto="__end__", update={"status": "rejected"})


def publish(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    receipt = send_newsletter(state["rendered"], runtime.context.settings)
    emit(
        runtime,
        "publish",
        f"send_newsletter → '{receipt.subject}' delivered to {len(receipt.recipients)} "
        f"subscribers (simulated; saved to {receipt.output_dir})",
        kind="tool",
        receipt=receipt.model_dump(mode="json"),
    )
    return {"delivery": receipt, "status": "sent"}
