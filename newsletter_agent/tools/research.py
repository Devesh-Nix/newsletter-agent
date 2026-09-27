"""LLM-callable research tools.

These are real LangChain tools bound to the research model, which decides *which* tool
to call, *with what query* and *when to stop*. Each tool uses the
``content_and_artifact`` response format: the model sees a compact text listing (cheap on
context), while the graph receives the structured `Article` records as the artifact.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from newsletter_agent.config import Settings
from newsletter_agent.models import Article
from newsletter_agent.tools import search

SearchBackend = Callable[[str, int, int, Settings], list[Article]]


def _search_args_schema(default_days: int) -> type[BaseModel]:
    return create_model(
        "SearchArgs",
        query=(str, Field(description="Short keyword query, e.g. 'AI agent framework launch'.")),
        days=(
            int,
            Field(default=default_days, description="Only return stories from the last N days."),
        ),
        max_results=(int, Field(default=10, description="Maximum results to return (1-15).")),
    )


def format_results(tool_name: str, query: str, articles: list[Article]) -> str:
    """Render search hits as a compact list the LLM can reason over."""
    if not articles:
        return f"{tool_name}({query!r}): no results. Try a broader or different query."
    lines = [f"{tool_name}({query!r}) returned {len(articles)} results:"]
    for a in articles:
        extras = f" · {a.points} pts" if a.points is not None else ""
        lines.append(f"- [{a.id}] {a.title} — {a.source} · {a.date_label}{extras}")
    return "\n".join(lines)


def _make_tool(
    name: str,
    description: str,
    backends: list[SearchBackend],
    settings: Settings,
) -> BaseTool:
    """Wrap one or more backends (tried in order until one returns results) as a tool."""

    def run(query: str, days: int, max_results: int = 10) -> tuple[str, list[Article]]:
        days = max(1, min(days, 31))
        max_results = max(1, min(max_results, 15))
        errors: list[str] = []
        for backend in backends:
            try:
                articles = backend(query, days, max_results, settings)
            except search.SearchError as exc:
                errors.append(str(exc))
                continue
            if articles:
                return format_results(name, query, articles), articles
        if errors:
            return f"{name}({query!r}) failed: {'; '.join(errors)}", []
        return format_results(name, query, []), []

    return StructuredTool.from_function(
        func=run,
        name=name,
        description=description,
        args_schema=_search_args_schema(settings.default_window_days),
        response_format="content_and_artifact",
    )


def build_research_tools(settings: Settings) -> list[BaseTool]:
    """The tool belt handed to the research model."""
    tools = [
        _make_tool(
            "search_news",
            "Search recent news coverage from mainstream and tech publications. Best for "
            "product launches, funding, enterprise adoption, policy and security stories.",
            [search.bing_news, search.google_news],
            settings,
        ),
        _make_tool(
            "search_hacker_news",
            "Search Hacker News stories, ranked by community points. Best for developer "
            "tools, open-source agent frameworks, research papers and technical deep dives.",
            [search.hacker_news],
            settings,
        ),
    ]
    if settings.tavily_api_key:
        tools.append(
            _make_tool(
                "search_web",
                "General web search (Tavily) for recent news. Use to fill gaps the other "
                "tools miss, e.g. company blogs and research announcements.",
                [search.tavily],
                settings,
            )
        )
    return tools
