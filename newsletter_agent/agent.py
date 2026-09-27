"""Public API: `run_newsletter_agent(goal)` and the lower-level `NewsletterAgent`."""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.types import Command

from newsletter_agent.config import DEFAULT_GOAL, Settings, load_settings
from newsletter_agent.graph import build_graph
from newsletter_agent.llm import create_chat_model, describe_model
from newsletter_agent.models import (
    Article,
    ArticleContent,
    DeliveryReceipt,
    HumanDecision,
    NewsletterDraft,
    NewsletterPlan,
    RenderedNewsletter,
    ReviewRecord,
)
from newsletter_agent.state import AgentContext, AgentEvent, AgentMode
from newsletter_agent.tools import build_research_tools, read_article

Reviewer = Callable[[dict[str, Any]], HumanDecision | dict[str, Any]]
EventHandler = Callable[[AgentEvent], None]


@dataclass
class NewsletterResult:
    """Everything a run produced, for display or further processing."""

    thread_id: str
    status: str
    plan: NewsletterPlan | None = None
    candidates_found: int = 0
    selected: list[Article] = field(default_factory=list)
    draft: NewsletterDraft | None = None
    reviews: list[ReviewRecord] = field(default_factory=list)
    rendered: RenderedNewsletter | None = None
    delivery: DeliveryReceipt | None = None

    @property
    def sent(self) -> bool:
        return self.status == "sent"

    @property
    def subject(self) -> str | None:
        return self.rendered.subject if self.rendered else None


class NewsletterAgent:
    """Owns the compiled graph, its dependencies and its checkpoints.

    Use `start()` / `resume()` to drive a run step by step (the UI does this), or call
    `run_newsletter_agent()` to do everything in one go.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        research_tools: list[BaseTool] | None = None,
        reader: Callable[[Article], ArticleContent] | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.llm = llm or create_chat_model(self.settings)
        self.context = AgentContext(
            llm=self.llm,
            settings=self.settings,
            research_tools=research_tools or build_research_tools(self.settings),
            read_article=reader or partial(read_article, settings=self.settings),
        )
        self.graph = build_graph()

    @property
    def model_label(self) -> str:
        return describe_model(self.llm)

    def start(
        self,
        goal: str = DEFAULT_GOAL,
        mode: AgentMode | str = AgentMode.AUTONOMOUS,
        thread_id: str | None = None,
    ) -> tuple[str, Iterator[AgentEvent]]:
        """Begin a run. Returns its thread id and a live stream of progress events."""
        thread_id = thread_id or uuid4().hex
        initial_state = {"goal": goal.strip(), "mode": AgentMode(mode).value, "status": "running"}
        return thread_id, self._stream(initial_state, thread_id)

    def resume(
        self, thread_id: str, decision: HumanDecision | dict[str, Any]
    ) -> Iterator[AgentEvent]:
        """Continue a run paused at a human-in-the-loop checkpoint."""
        decision = HumanDecision.model_validate(decision)
        return self._stream(Command(resume=decision.model_dump()), thread_id)

    def pending_review(self, thread_id: str) -> dict[str, Any] | None:
        """The payload of the checkpoint the run is paused at, if any."""
        snapshot = self.graph.get_state(self._config(thread_id))
        return snapshot.interrupts[0].value if snapshot.interrupts else None

    def result(self, thread_id: str) -> NewsletterResult:
        values = self.graph.get_state(self._config(thread_id)).values
        return NewsletterResult(
            thread_id=thread_id,
            status=values.get("status", "running"),
            plan=values.get("plan"),
            candidates_found=len(values.get("candidates", {})),
            selected=values.get("selected", []),
            draft=values.get("draft"),
            reviews=values.get("reviews", []),
            rendered=values.get("rendered"),
            delivery=values.get("delivery"),
        )

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}

    def _stream(self, payload: dict | Command, thread_id: str) -> Iterator[AgentEvent]:
        for chunk in self.graph.stream(
            payload, self._config(thread_id), context=self.context, stream_mode="custom"
        ):
            yield AgentEvent(**chunk)


def run_newsletter_agent(
    goal: str = DEFAULT_GOAL,
    mode: AgentMode | str = AgentMode.AUTONOMOUS,
    *,
    reviewer: Reviewer | None = None,
    on_event: EventHandler | None = None,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    research_tools: list[BaseTool] | None = None,
    reader: Callable[[Article], ArticleContent] | None = None,
) -> NewsletterResult:
    """Research, write, self-critique and send a newsletter from a plain-English goal.

    Args:
        goal: What to produce, e.g. "Create a weekly newsletter on latest AI agent news
            and send it to our subscribers."
        mode: ``"autonomous"`` runs end to end. ``"human_in_the_loop"`` pauses for approval
            of the plan and of the final draft.
        reviewer: Called at each human checkpoint with its payload; returns a
            `HumanDecision` (approve / revise with feedback / reject). Defaults to an
            interactive console prompt.
        on_event: Receives live progress events. Defaults to printing them.
        settings: Configuration overrides (defaults come from env / `.env`).
        llm: Any LangChain chat model; defaults to the configured provider.
        research_tools: Replace the default search tools (e.g. with offline fakes).
        reader: Replace the default article reader.
    """
    agent = NewsletterAgent(
        settings=settings, llm=llm, research_tools=research_tools, reader=reader
    )
    reviewer = reviewer or console_reviewer
    on_event = on_event or print_event

    thread_id, events = agent.start(goal, mode)
    while True:
        for event in events:
            on_event(event)
        checkpoint = agent.pending_review(thread_id)
        if checkpoint is None:
            break
        events = agent.resume(thread_id, reviewer(checkpoint))
    return agent.result(thread_id)


# ---------------------------------------------------------------------------
# Minimal console defaults (the CLI and web UI provide richer ones)
# ---------------------------------------------------------------------------


def print_event(event: AgentEvent) -> None:
    print(f"[{event.stage:>9}] {event.message}")


def console_reviewer(checkpoint: dict[str, Any]) -> HumanDecision:
    print(f"\n=== {checkpoint['title']} ===")
    if checkpoint["checkpoint"] == "plan_review":
        plan = checkpoint["plan"]
        print(f"Topic: {plan['topic']}  |  Audience: {plan['audience']}")
        print("Queries: " + "; ".join(plan["search_queries"]))
    else:
        print(f"Subject: {checkpoint['subject']}\n")
        print(textwrap.shorten(checkpoint["markdown"], 1500, placeholder=" …"))

    choice = input("\n[a]pprove, [r]evise, or [x] reject? ").strip().lower()[:1]
    if choice == "r":
        return HumanDecision(action="revise", feedback=input("What should change? "))
    if choice == "x":
        return HumanDecision(action="reject")
    return HumanDecision(action="approve")
