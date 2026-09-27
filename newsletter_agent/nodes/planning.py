"""Step 1 — Planning: turn the plain-English goal into an editorial and research plan."""

from __future__ import annotations

from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from newsletter_agent.config import Settings
from newsletter_agent.llm import structured_output
from newsletter_agent.models import HumanDecision, NewsletterPlan
from newsletter_agent.nodes.common import today
from newsletter_agent.prompts import PLANNER_PROMPT
from newsletter_agent.state import AgentContext, AgentMode, AgentState, emit

MAX_QUERIES = 6


def plan(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    settings = runtime.context.settings
    emit(runtime, "plan", "Interpreting the goal and drafting a research plan")

    feedback = ""
    if state.get("plan_feedback") and state.get("plan"):
        feedback = (
            f"\n\nPrevious plan:\n{state['plan'].model_dump_json(indent=2)}\n\n"
            f"The human reviewer asked for these changes: {state['plan_feedback']}"
        )

    chain = PLANNER_PROMPT | structured_output(runtime.context.llm, NewsletterPlan)
    raw_plan = chain.invoke(
        {
            "today": today(),
            "goal": state["goal"],
            "min_articles": settings.min_articles,
            "max_articles": settings.max_articles,
            "window_rule": (
                f"The issue must cover the last {settings.lookback_days} days."
                if settings.lookback_days
                else "Derive the time window from the cadence: weekly means the last 7 days, "
                "daily 1, monthly 30. Default to 7 days."
            ),
            "feedback": feedback,
        }
    )
    new_plan = normalize_plan(raw_plan, settings)

    emit(runtime, "plan", new_plan.reasoning, kind="thought")
    emit(
        runtime,
        "plan",
        f"Plan ready: {new_plan.target_article_count} stories on '{new_plan.topic}' from the "
        f"last {new_plan.time_window_days} days, {len(new_plan.search_queries)} search queries",
        kind="result",
        plan=new_plan.model_dump(),
    )
    return {"plan": new_plan, "plan_feedback": ""}


def normalize_plan(plan: NewsletterPlan, settings: Settings) -> NewsletterPlan:
    """Clamp LLM-proposed numbers to configured bounds and clean up the query list."""
    queries: list[str] = []
    for query in plan.search_queries:
        query = query.strip().strip('"')
        if query and query.lower() not in {q.lower() for q in queries}:
            queries.append(query)
    if not queries:
        queries = [plan.topic, f"{plan.topic} launch", f"{plan.topic} research"]

    return plan.model_copy(
        update={
            "search_queries": queries[:MAX_QUERIES],
            "target_article_count": max(
                settings.min_articles, min(plan.target_article_count, settings.max_articles)
            ),
            # An explicitly configured window wins over the one inferred from the goal.
            "time_window_days": settings.lookback_days
            or max(1, min(plan.time_window_days or settings.default_window_days, 31)),
        }
    )


def route_after_plan(state: AgentState) -> Literal["review_plan", "research"]:
    """The autonomy toggle: a human vets the plan only in human-in-the-loop mode."""
    return "review_plan" if state["mode"] == AgentMode.HUMAN_IN_THE_LOOP else "research"


def review_plan(state: AgentState) -> Command[Literal["research", "plan", "__end__"]]:
    """Human-in-the-loop checkpoint: approve, amend or reject the plan before research."""
    decision = HumanDecision.model_validate(
        interrupt(
            {
                "checkpoint": "plan_review",
                "title": "Review the research plan",
                "plan": state["plan"].model_dump(),
            }
        )
    )
    if decision.action == "approve":
        return Command(goto="research")
    if decision.action == "revise" and decision.feedback.strip():
        return Command(goto="plan", update={"plan_feedback": decision.feedback.strip()})
    if decision.action == "revise":  # "revise" without instructions: nothing to change
        return Command(goto="research")
    return Command(goto="__end__", update={"status": "rejected"})
