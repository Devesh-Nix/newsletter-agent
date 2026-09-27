"""Wire the nodes into a LangGraph state machine.

    plan ─▶ [review_plan] ─▶ research ⇄ run_research_tools ─▶ curate ─▶ summarize
                                                                           │
    publish ◀─ [review_draft] ◀─ render ◀─ critique ⇄ write ◀──────────────┘

Bracketed nodes are human-in-the-loop checkpoints, visited only in that mode.
"""

from __future__ import annotations

import inspect

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from newsletter_agent import models
from newsletter_agent.nodes import (
    critique,
    curate,
    plan,
    publish,
    render,
    research,
    review_draft,
    review_plan,
    route_after_plan,
    route_after_render,
    route_after_tools,
    route_research,
    run_research_tools,
    summarize,
    write,
)
from newsletter_agent.state import AgentContext, AgentState


def make_checkpointer() -> InMemorySaver:
    """In-memory checkpointer that may deserialize this package's models.

    Checkpoints are what let a human-in-the-loop run pause at an interrupt and resume
    later. The serializer runs in strict mode: besides LangGraph's built-in safe types
    (messages, datetimes, ...) it only revives this package's own Pydantic models.
    """
    own_models = [
        (models.__name__, name)
        for name, obj in inspect.getmembers(models, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == models.__name__
    ]
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=own_models))


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    builder = StateGraph(AgentState, context_schema=AgentContext)

    builder.add_node("plan", plan)
    builder.add_node("review_plan", review_plan)
    builder.add_node("research", research)
    builder.add_node("run_research_tools", run_research_tools)
    builder.add_node("curate", curate)
    builder.add_node("summarize", summarize)
    builder.add_node("write", write)
    builder.add_node("critique", critique)
    builder.add_node("render", render)
    builder.add_node("review_draft", review_draft)
    builder.add_node("publish", publish)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route_after_plan)
    # review_plan routes itself (Command) to research, back to plan, or END.
    builder.add_conditional_edges("research", route_research)
    builder.add_conditional_edges("run_research_tools", route_after_tools)
    builder.add_edge("curate", "summarize")
    builder.add_edge("summarize", "write")
    builder.add_edge("write", "critique")
    # critique routes itself (Command) back to write or on to render.
    builder.add_conditional_edges("render", route_after_render)
    # review_draft routes itself (Command) to publish, back to write, or END.
    builder.add_edge("publish", END)

    return builder.compile(checkpointer=checkpointer or make_checkpointer())
