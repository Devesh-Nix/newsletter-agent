"""UI tests for the Streamlit front end (Streamlit's AppTest, offline fakes)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool
from streamlit.testing.v1 import AppTest

import newsletter_agent
from newsletter_agent.agent import NewsletterAgent
from tests.conftest import fake_reader, fake_search_tool, make_article

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def launch(monkeypatch, settings, make_llm):
    """Start the app with its agent wired to the scripted LLM and the given tools."""
    # Registered first so monkeypatch restores them even if the app sets them.
    for name in ("APP_PASSWORD", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    received: list[newsletter_agent.Settings] = []

    def start(research_tools: list[BaseTool], secrets: dict | None = None) -> AppTest:
        def fake_agent(settings: newsletter_agent.Settings) -> NewsletterAgent:
            received.append(settings)
            # Keep the UI's choices but write to the test's temporary outbox.
            return NewsletterAgent(
                settings=settings.model_copy(update=sandbox),
                llm=make_llm(),
                research_tools=research_tools,
                reader=fake_reader,
            )

        monkeypatch.setattr(newsletter_agent, "NewsletterAgent", fake_agent)
        at = AppTest.from_file(APP, default_timeout=60)
        for name, value in (secrets or {}).items():
            at.secrets[name] = value
        at.run()
        return at

    sandbox = {"outbox_dir": settings.outbox_dir, "subscribers_file": settings.subscribers_file}
    start.received = received
    return start


@pytest.fixture
def app(launch, research_tools) -> AppTest:
    return launch(research_tools)


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


def test_dollar_amounts_are_not_rendered_as_latex(launch):
    stories = [make_article(f"Startup {n} raises ${n}0M for agents") for n in range(1, 8)]
    app = launch([fake_search_tool("search_news", stories)])
    click(app, "Run agent")

    lines = [m.value for m in app.markdown if "raises" in m.value]
    assert lines
    assert all(re.search(r"(?<!\\)\$", line) is None for line in lines)


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


def test_key_typed_in_sidebar_stays_in_the_session(launch, research_tools):
    app = launch(research_tools)
    app.sidebar.selectbox[0].set_value("anthropic").run()
    next(t for t in app.sidebar.text_input if t.label == "ANTHROPIC_API_KEY").set_value(
        "sk-ant-visitor"
    ).run()
    click(app, "Run agent")

    assert launch.received[-1].llm_api_key.get_secret_value() == "sk-ant-visitor"
    assert "ANTHROPIC_API_KEY" not in os.environ  # never shared with other visitors


def test_password_gate_protects_hosted_app(launch, research_tools):
    app = launch(research_tools, secrets={"APP_PASSWORD": "open-sesame"})
    assert not app.button or all(b.label != "Run agent" for b in app.button)

    app.text_input[0].set_value("wrong")
    app.button[0].click().run()
    assert "Incorrect password" in app.error[0].value

    app.text_input[0].set_value("open-sesame")
    app.button[0].click().run()
    assert any(b.label == "Run agent" for b in app.button)


def test_request_changes_requires_feedback(app):
    app.session_state["mode"] = newsletter_agent.AgentMode.HUMAN_IN_THE_LOOP
    app.run()
    click(app, "Run agent")
    click(app, "Request changes")
    assert "Describe what should change" in app.warning[0].value
