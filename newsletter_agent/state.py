"""Graph state, run-time context and progress events."""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from newsletter_agent.config import Settings
from newsletter_agent.models import (
    Article,
    ArticleContent,
    ArticleSummary,
    DeliveryReceipt,
    NewsletterDraft,
    NewsletterPlan,
    RankedPick,
    RenderedNewsletter,
    ReviewRecord,
)

log = logging.getLogger("newsletter_agent")


class AgentMode(StrEnum):
    """The autonomy toggle."""

    AUTONOMOUS = "autonomous"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"

    @classmethod
    def _missing_(cls, value: object) -> AgentMode | None:
        """Accept friendly spellings such as "hitl", "Human-in-the-loop" or "auto"."""
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "hitl": cls.HUMAN_IN_THE_LOOP,
            "human": cls.HUMAN_IN_THE_LOOP,
            "auto": cls.AUTONOMOUS,
        }
        return aliases.get(key) or next((m for m in cls if m.value == key), None)


def merge_articles(left: dict[str, Article], right: dict[str, Article]) -> dict[str, Article]:
    """Reducer: accumulate research results across rounds, keeping the first copy seen."""
    return {**right, **(left or {})}


class AgentState(TypedDict, total=False):
    """Everything the agent knows, checkpointed after every step."""

    # Input
    goal: str
    mode: str  # an AgentMode value
    # Planning
    plan: NewsletterPlan
    plan_feedback: str
    # Research: a tool-calling conversation plus the structured results it produced
    research_messages: Annotated[list[AnyMessage], add_messages]
    research_rounds: int
    research_notes: str
    candidates: Annotated[dict[str, Article], merge_articles]
    # Curation and summarisation
    selected: list[Article]
    picks: list[RankedPick]
    summaries: dict[str, ArticleSummary]
    # Writing and self-reflection
    draft: NewsletterDraft
    drafts_written: int
    reviews: Annotated[list[ReviewRecord], operator.add]
    human_feedback: str
    # Output
    rendered: RenderedNewsletter
    delivery: DeliveryReceipt
    status: Literal["running", "sent", "rejected"]


@dataclass
class AgentContext:
    """Run-scoped dependencies injected into every node (not checkpointed).

    Injecting them, rather than importing globals, keeps nodes pure functions of
    (state, context) and lets tests swap in a fake LLM and offline tools.
    """

    llm: BaseChatModel
    settings: Settings
    research_tools: list[BaseTool]
    read_article: Callable[[Article], ArticleContent]


Stage = Literal[
    "plan", "research", "curate", "summarize", "write", "critique", "render", "review", "publish"
]
EventKind = Literal["step", "tool", "thought", "result", "warning"]


@dataclass
class AgentEvent:
    """A progress update streamed to the CLI / UI while the agent works."""

    stage: Stage
    kind: EventKind
    message: str
    data: dict[str, Any] = field(default_factory=dict)


def emit(
    runtime: Runtime[AgentContext],
    stage: Stage,
    message: str,
    kind: EventKind = "step",
    **data: Any,
) -> None:
    """Stream an `AgentEvent` (LangGraph "custom" stream mode) and log it."""
    log.info("[%s] %s", stage, message)
    runtime.stream_writer(asdict(AgentEvent(stage=stage, kind=kind, message=message, data=data)))
