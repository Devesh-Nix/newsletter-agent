"""Step 2 — Research: an LLM-driven tool-calling loop.

The research model is given the plan and a belt of search tools. Each round it decides
which tools to call with which queries (in parallel); `run_research_tools` executes them
and feeds compact results back. The model stops when it judges coverage sufficient, or
when the round budget runs out.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.runtime import Runtime

from newsletter_agent.models import Article
from newsletter_agent.nodes.common import bullet_list, today
from newsletter_agent.prompts import RESEARCH_BRIEF, RESEARCH_SYSTEM
from newsletter_agent.state import AgentContext, AgentState, emit

#: How many distinct candidates to aim for before curating down to 5-7.
CANDIDATE_POOL_TARGET = 25


def research(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    plan = state["plan"]
    history = state.get("research_messages", [])
    new_messages = []

    if not history:
        emit(runtime, "research", "Researching the latest news with search tools")
        new_messages = [
            SystemMessage(
                RESEARCH_SYSTEM.format(
                    today=today(),
                    topic=plan.topic,
                    days=plan.time_window_days,
                    audience=plan.audience,
                    target_pool=CANDIDATE_POOL_TARGET,
                    max_rounds=ctx.settings.max_research_rounds,
                )
            ),
            HumanMessage(
                RESEARCH_BRIEF.format(
                    queries=bullet_list(plan.search_queries),
                    criteria=bullet_list(plan.selection_criteria),
                    days=plan.time_window_days,
                )
            ),
        ]

    model = ctx.llm.bind_tools(ctx.research_tools)
    response: AIMessage = model.invoke([*history, *new_messages])

    if response.tool_calls:
        queries = ", ".join(_describe_call(call) for call in response.tool_calls)
        emit(
            runtime,
            "research",
            f"Round {state.get('research_rounds', 0) + 1}: decided to run "
            f"{len(response.tool_calls)} searches — {queries}",
            kind="thought",
        )
        return {"research_messages": [*new_messages, response]}

    notes = response.text.strip()
    if notes:
        emit(runtime, "research", notes, kind="thought")
    return {"research_messages": [*new_messages, response], "research_notes": notes}


def run_research_tools(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """Execute every tool call from the last model turn concurrently."""
    last = state["research_messages"][-1]
    tools = {tool.name: tool for tool in runtime.context.research_tools}
    known = state.get("candidates", {})

    def execute(call: ToolCall) -> ToolMessage:
        tool = tools.get(call["name"])
        if tool is None:
            return ToolMessage(
                f"Unknown tool {call['name']!r}. Available: {', '.join(tools)}",
                tool_call_id=call["id"],
                status="error",
            )
        try:
            return tool.invoke(call)
        except Exception as exc:  # a failing tool must not crash the agent
            return ToolMessage(f"Tool error: {exc}", tool_call_id=call["id"], status="error")

    for call in last.tool_calls:
        emit(runtime, "research", _describe_call(call), kind="tool", tool=call["name"])
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(execute, last.tool_calls))

    found: dict[str, Article] = {}
    for message in results:
        for article in message.artifact or []:
            found.setdefault(article.id, article)
    new = [aid for aid in found if aid not in known]
    rounds = state.get("research_rounds", 0) + 1
    emit(
        runtime,
        "research",
        f"Round {rounds}: {len(found)} results, {len(new)} new "
        f"({len(known) + len(new)} unique candidates so far)",
        kind="result",
    )
    return {"research_messages": results, "candidates": found, "research_rounds": rounds}


def route_research(
    state: AgentState, runtime: Runtime[AgentContext]
) -> Literal["run_research_tools", "curate"]:
    """Keep searching while the model asks for tools and budget remains."""
    last = state["research_messages"][-1]
    wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
    budget_left = state.get("research_rounds", 0) < runtime.context.settings.max_research_rounds
    return "run_research_tools" if wants_tools and budget_left else "curate"


def route_after_tools(
    state: AgentState, runtime: Runtime[AgentContext]
) -> Literal["research", "curate"]:
    """Let the model reflect on results unless the round budget is spent."""
    budget_left = state.get("research_rounds", 0) < runtime.context.settings.max_research_rounds
    return "research" if budget_left else "curate"


def _describe_call(call: ToolCall) -> str:
    return f"{call['name']}({call['args'].get('query', '')!r})"
