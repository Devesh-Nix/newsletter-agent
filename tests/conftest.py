"""Offline test doubles: a scripted chat model and fake research tools.

They let the full LangGraph workflow run deterministically, with no network access and
no API keys, while still exercising real tool calling, structured output and routing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from newsletter_agent.config import Settings
from newsletter_agent.models import (
    Article,
    ArticleContent,
    ArticleSummary,
    CriterionScore,
    Critique,
    CurationResult,
    DraftItem,
    NewsletterDraft,
    NewsletterPlan,
    RankedPick,
)
from newsletter_agent.utils import article_id, utc_now

Responder = Callable[[list[BaseMessage]], BaseModel]

STORY_TOPICS = [
    "OpenAI ships a computer-use agent for enterprises",
    "Anthropic releases an open agent evaluation benchmark",
    "LangGraph adds durable execution for long-running agents",
    "Startup raises $80M to build coding agents",
    "EU publishes guidance on autonomous AI agent liability",
    "Google DeepMind agent solves open maths problems",
    "Security researchers find prompt-injection flaw in browser agents",
    "Microsoft open-sources a multi-agent orchestration framework",
]


# ---------------------------------------------------------------------------
# Scripted chat model
# ---------------------------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """A fake LLM: tool-calling turns are replayed from a script and structured-output
    requests are answered by per-schema responder functions."""

    tool_turns: list[AIMessage] = Field(default_factory=list)
    responders: dict[type, Responder] = Field(default_factory=dict)
    structured_calls: list[str] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        if self.tool_turns:
            message = self.tool_turns.pop(0)
        else:
            message = AIMessage(content="Coverage is strong across launches, research and policy.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any):
        self.bound_tools = [t.name for t in tools]
        return self

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any):
        def respond(prompt_value: Any) -> BaseModel:
            self.structured_calls.append(schema.__name__)
            return self.responders[schema](prompt_value.to_messages())

        return RunnableLambda(respond)


def prompt_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages)


def ids_in(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[([0-9a-f]{8})\]", text)))


def default_plan(_: list[BaseMessage]) -> NewsletterPlan:
    return NewsletterPlan(
        topic="AI agents",
        audience="AI engineers and product leaders",
        tone="sharp and informed",
        time_window_days=7,
        target_article_count=6,
        search_queries=["AI agents", "agent framework", "AI agents", "  "],
        selection_criteria=["Significance", "Novelty"],
        reasoning="Weekly cadence, so cover the last seven days across several angles.",
    )


def default_curation(messages: list[BaseMessage]) -> CurationResult:
    ids = ids_in(prompt_text(messages))
    return CurationResult(
        picks=[
            RankedPick(article_id=i, category="Launch", relevance=9, rationale="Big news.")
            for i in ids[:6]
        ],
        coverage_notes="A balanced mix of launches, research and policy.",
    )


def default_summary(messages: list[BaseMessage]) -> ArticleSummary:
    title = re.search(r"Title: (.+)", prompt_text(messages)).group(1)
    return ArticleSummary(
        headline=title,
        summary=f"{title}. The release gives teams new capabilities and signals how quickly "
        "the agent ecosystem is maturing across vendors and open-source projects.",
        why_it_matters="It changes what builders can ship this quarter.",
    )


def default_draft(messages: list[BaseMessage]) -> NewsletterDraft:
    ids = ids_in(prompt_text(messages))
    return NewsletterDraft(
        subject="Agents go enterprise: this week's biggest moves",
        preheader="Computer use, new benchmarks and a funding surge.",
        intro="Agents crossed from demo to deployment this week.",
        items=[
            DraftItem(
                article_id=i,
                headline=f"Story {n}",
                summary="A concise, factual summary of the development that runs to a healthy "
                "length so the automated checks consider it substantive enough.",
                why_it_matters="Builders get new leverage.",
                category="Launch",
            )
            for n, i in enumerate(ids, start=1)
        ],
        big_picture="Agents are becoming infrastructure.",
        sign_off="See you next week.",
    )


def critique_sequence(*approvals: bool) -> Responder:
    """Critic that returns the given approvals in order (last one repeats)."""
    state = {"calls": 0}

    def respond(_: list[BaseMessage]) -> Critique:
        approved = approvals[min(state["calls"], len(approvals) - 1)]
        state["calls"] += 1
        score = 9 if approved else 6
        return Critique(
            scores=[CriterionScore(criterion="Accuracy", score=score, comment="ok")],
            overall_score=score,
            strengths=["Clear structure"],
            issues=[] if approved else ["The intro is generic."],
            revision_instructions=[] if approved else ["Make the intro more specific."],
            approved=approved,
        )

    return respond


def search_turn(*queries: str) -> AIMessage:
    """A model turn that calls the news and HN tools for each query."""
    calls = []
    for n, query in enumerate(queries):
        calls.append({"name": "search_news", "args": {"query": query, "days": 7}, "id": f"n{n}"})
        calls.append(
            {"name": "search_hacker_news", "args": {"query": query, "days": 7}, "id": f"h{n}"}
        )
    return AIMessage(content="", tool_calls=calls)


# ---------------------------------------------------------------------------
# Fake tools
# ---------------------------------------------------------------------------


def make_article(title: str, origin: str = "bing_news", days_ago: int = 1, **kw: Any) -> Article:
    url = f"https://news.example.com/{re.sub(r'[^a-z0-9]+', '-', title.lower())}"
    return Article(
        id=article_id(url),
        title=title,
        url=url,
        source="Example News",
        published_at=utc_now() - timedelta(days=days_ago),
        snippet=f"Snippet about {title}.",
        origin=origin,
        **kw,
    )


def fake_search_tool(name: str, articles: list[Article]) -> BaseTool:
    def run(query: str, days: int = 7, max_results: int = 10):
        return f"{name}({query!r}) returned {len(articles)} results", articles

    return StructuredTool.from_function(
        func=run, name=name, description=f"Fake {name}.", response_format="content_and_artifact"
    )


def fake_reader(article: Article) -> ArticleContent:
    return ArticleContent(
        url=article.url, text=f"Full text of {article.title}.", from_full_page=True
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    subscribers = tmp_path / "subscribers.json"
    subscribers.write_text('[{"name": "Ada", "email": "ada@example.com"}]', encoding="utf-8")
    return Settings(
        outbox_dir=tmp_path / "outbox",
        subscribers_file=subscribers,
        max_research_rounds=2,
        max_revisions=2,
    )


@pytest.fixture
def articles() -> list[Article]:
    news = [make_article(t) for t in STORY_TOPICS[:5]]
    hn = [
        make_article(t, origin="hacker_news", points=300 - i)
        for i, t in enumerate(STORY_TOPICS[5:])
    ]
    stale = make_article("Old news about agents from last month", days_ago=30)
    return [*news, *hn, stale]


@pytest.fixture
def research_tools(articles) -> list[BaseTool]:
    news = [a for a in articles if a.origin == "bing_news"]
    hn = [a for a in articles if a.origin == "hacker_news"]
    return [fake_search_tool("search_news", news), fake_search_tool("search_hacker_news", hn)]


@pytest.fixture
def make_llm() -> Callable[..., ScriptedChatModel]:
    def factory(
        tool_turns: list[AIMessage] | None = None, critic: Responder | None = None
    ) -> ScriptedChatModel:
        return ScriptedChatModel(
            tool_turns=list(tool_turns if tool_turns is not None else [search_turn("AI agents")]),
            responders={
                NewsletterPlan: default_plan,
                CurationResult: default_curation,
                ArticleSummary: default_summary,
                NewsletterDraft: default_draft,
                Critique: critic or critique_sequence(True),
            },
        )

    return factory
