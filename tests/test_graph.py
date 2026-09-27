"""End-to-end tests of the agent workflow, run offline with a scripted LLM."""

from __future__ import annotations

from email import message_from_bytes
from pathlib import Path

import pytest

from newsletter_agent import AgentMode, HumanDecision, NewsletterAgent, run_newsletter_agent
from newsletter_agent.nodes.curation import ResearchError
from tests.conftest import critique_sequence, fake_reader, fake_search_tool, search_turn

GOAL = "Create a weekly newsletter on latest AI agent news and send it to our subscribers."


def run(settings, llm, research_tools, **kwargs):
    events = []
    result = run_newsletter_agent(
        GOAL,
        settings=settings,
        llm=llm,
        research_tools=research_tools,
        reader=fake_reader,
        on_event=events.append,
        **kwargs,
    )
    return result, events


def test_autonomous_run_goes_from_goal_to_sent_newsletter(settings, make_llm, research_tools):
    llm = make_llm()
    result, events = run(settings, llm, research_tools)

    assert result.sent
    assert llm.bound_tools == ["search_news", "search_hacker_news"]
    # Planning -> research -> writing -> review -> output, in that order.
    stages = list(dict.fromkeys(e.stage for e in events))
    assert stages == [
        "plan",
        "research",
        "curate",
        "summarize",
        "write",
        "critique",
        "render",
        "publish",
    ]
    # The planner's duplicate/blank queries were cleaned up.
    assert result.plan.search_queries == ["AI agents", "agent framework"]
    # 6 stories selected, and the month-old article never made it past curation.
    assert len(result.selected) == 6
    assert all("Old news" not in a.title for a in result.selected)
    assert [i.article_id for i in result.draft.items] == [a.id for a in result.selected]

    files = result.delivery.files
    assert set(files) == {"html", "markdown", "eml", "receipt"}
    email = message_from_bytes(Path(files["eml"]).read_bytes())
    assert email["Subject"] == result.subject
    assert email.get_content_type() == "multipart/alternative"
    assert result.delivery.recipients == ["ada@example.com"]


def test_self_critique_sends_weak_draft_back_for_revision(settings, make_llm, research_tools):
    llm = make_llm(critic=critique_sequence(False, True))
    result, events = run(settings, llm, research_tools)

    assert result.sent
    assert [r.passed for r in result.reviews] == [False, True]
    assert llm.structured_calls.count("NewsletterDraft") == 2
    assert any("sending it back for revision" in e.message for e in events)


def test_revision_loop_stops_at_budget(settings, make_llm, research_tools):
    llm = make_llm(critic=critique_sequence(False))
    result, events = run(settings, llm, research_tools)

    assert result.sent  # best effort still ships
    assert len(result.reviews) == settings.max_revisions + 1
    assert any("Revision budget" in e.message for e in events)


def test_model_that_skips_tools_falls_back_to_planned_queries(settings, make_llm, research_tools):
    llm = make_llm(tool_turns=[])
    result, events = run(settings, llm, research_tools)

    assert result.sent
    assert len(result.selected) == 6
    assert any(e.kind == "warning" and "without searching" in e.message for e in events)


def test_research_loop_respects_round_budget(settings, make_llm, research_tools):
    turns = [search_turn("agents"), search_turn("agent funding"), search_turn("agent policy")]
    llm = make_llm(tool_turns=turns)
    agent = NewsletterAgent(
        settings=settings, llm=llm, research_tools=research_tools, reader=fake_reader
    )
    thread_id, events = agent.start(GOAL)
    list(events)

    state = agent.graph.get_state(agent._config(thread_id)).values
    assert state["research_rounds"] == settings.max_research_rounds == 2
    assert len(llm.tool_turns) == 1  # the third scripted turn was never requested


def test_empty_research_raises_clear_error(settings, make_llm):
    tools = [fake_search_tool("search_news", [])]
    with pytest.raises(ResearchError, match="No stories"):
        run(settings, make_llm(), tools)


# ---------------------------------------------------------------------------
# Human-in-the-loop mode
# ---------------------------------------------------------------------------


def hitl_agent(settings, llm, research_tools) -> NewsletterAgent:
    return NewsletterAgent(
        settings=settings, llm=llm, research_tools=research_tools, reader=fake_reader
    )


def test_hitl_pauses_at_plan_and_draft_then_sends(settings, make_llm, research_tools):
    agent = hitl_agent(settings, make_llm(), research_tools)
    thread_id, events = agent.start(GOAL, AgentMode.HUMAN_IN_THE_LOOP)
    list(events)

    checkpoint = agent.pending_review(thread_id)
    assert checkpoint["checkpoint"] == "plan_review"
    assert checkpoint["plan"]["topic"] == "AI agents"
    assert agent.result(thread_id).status == "running"

    list(agent.resume(thread_id, HumanDecision(action="approve")))
    checkpoint = agent.pending_review(thread_id)
    assert checkpoint["checkpoint"] == "draft_review"
    assert checkpoint["subject"] and "<html" in checkpoint["html"]
    assert not (settings.outbox_dir).exists()  # nothing sent before approval

    list(agent.resume(thread_id, {"action": "approve"}))
    assert agent.pending_review(thread_id) is None
    assert agent.result(thread_id).sent


def test_hitl_feedback_on_plan_triggers_replanning(settings, make_llm, research_tools):
    llm = make_llm()
    agent = hitl_agent(settings, llm, research_tools)
    thread_id, events = agent.start(GOAL, AgentMode.HUMAN_IN_THE_LOOP)
    list(events)

    list(agent.resume(thread_id, HumanDecision(action="revise", feedback="Focus on open source")))
    assert agent.pending_review(thread_id)["checkpoint"] == "plan_review"
    assert llm.structured_calls.count("NewsletterPlan") == 2


def test_hitl_feedback_on_draft_triggers_rewrite(settings, make_llm, research_tools):
    llm = make_llm()
    agent = hitl_agent(settings, llm, research_tools)
    thread_id, events = agent.start(GOAL, AgentMode.HUMAN_IN_THE_LOOP)
    list(events)
    list(agent.resume(thread_id, HumanDecision(action="approve")))

    list(agent.resume(thread_id, HumanDecision(action="revise", feedback="Shorter intro")))
    assert agent.pending_review(thread_id)["checkpoint"] == "draft_review"
    assert llm.structured_calls.count("NewsletterDraft") == 2

    list(agent.resume(thread_id, HumanDecision(action="approve")))
    assert agent.result(thread_id).sent


def test_hitl_reject_stops_without_sending(settings, make_llm, research_tools):
    agent = hitl_agent(settings, make_llm(), research_tools)
    thread_id, events = agent.start(GOAL, AgentMode.HUMAN_IN_THE_LOOP)
    list(events)
    list(agent.resume(thread_id, HumanDecision(action="approve")))
    list(agent.resume(thread_id, HumanDecision(action="reject")))

    result = agent.result(thread_id)
    assert result.status == "rejected"
    assert result.delivery is None
    assert not settings.outbox_dir.exists()


def test_run_newsletter_agent_uses_reviewer_callback(settings, make_llm, research_tools):
    seen = []

    def reviewer(checkpoint):
        seen.append(checkpoint["checkpoint"])
        return HumanDecision(action="approve")

    result, _ = run(
        settings, make_llm(), research_tools, mode="human_in_the_loop", reviewer=reviewer
    )
    assert result.sent
    assert seen == ["plan_review", "draft_review"]
