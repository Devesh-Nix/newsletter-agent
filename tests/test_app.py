"""UI tests for the Streamlit front end (Streamlit's AppTest, offline fakes)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import newsletter_agent
from newsletter_agent.agent import NewsletterAgent
from tests.conftest import fake_reader

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(monkeypatch, settings, make_llm, research_tools) -> AppTest:
    """The app with its agent wired to the scripted LLM and fake tools."""

    def fake_agent(settings: newsletter_agent.Settings) -> NewsletterAgent:
        # Keep the UI's choices but write to the test's temporary outbox.
        return NewsletterAgent(
            settings=settings.model_copy(update=sandbox),
            llm=make_llm(),
            research_tools=research_tools,
            reader=fake_reader,
        )

    sandbox = {"outbox_dir": settings.outbox_dir, "subscribers_file": settings.subscribers_file}
    monkeypatch.setattr(newsletter_agent, "NewsletterAgent", fake_agent)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def click(at: AppTest, label: str) -> None:
    next(b for b in at.button if b.label == label).click().run()


def test_initial_page_offers_goal_and_mode_toggle(app):
    assert not app.exception
    assert "weekly newsletter" in app.text_area[0].value
    assert app.session_state["mode"] == newsletter_agent.AgentMode.AUTONOMOUS


def test_autonomous_run_shows_results(app):
    click(app, "Run agent")

    assert not app.exception
    assert "Sent" in app.success[0].value
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Stories selected"] == "6"
    assert metrics["Drafts written"] == "1"


def test_human_in_the_loop_run_pauses_for_both_reviews(app):
    app.session_state["mode"] = newsletter_agent.AgentMode.HUMAN_IN_THE_LOOP
    app.run()
    click(app, "Run agent")
    assert "Review the research plan" in app.subheader[0].value

    click(app, "Approve plan")
    assert "Approve the newsletter" in app.subheader[0].value
    assert not app.success  # nothing sent yet

    click(app, "Approve & send")
    assert not app.exception
    assert "Sent" in app.success[0].value


def test_request_changes_requires_feedback(app):
    app.session_state["mode"] = newsletter_agent.AgentMode.HUMAN_IN_THE_LOOP
    app.run()
    click(app, "Run agent")
    click(app, "Request changes")
    assert "Describe what should change" in app.warning[0].value
