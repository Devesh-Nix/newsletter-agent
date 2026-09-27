"""CLI tests: exit codes and the interactive review flow, with offline fakes."""

from __future__ import annotations

import pytest

from newsletter_agent import cli
from newsletter_agent.agent import NewsletterAgent
from newsletter_agent.llm import LLMConfigurationError
from tests.conftest import fake_reader


@pytest.fixture
def fake_cli(monkeypatch, settings, make_llm, research_tools):
    """Route the CLI's agent construction to the scripted LLM and fake tools."""

    def use(llm=None, answers: list[str] | None = None):
        def factory(settings):
            return NewsletterAgent(
                settings=settings.model_copy(update=sandbox),
                llm=llm or make_llm(),
                research_tools=research_tools,
                reader=fake_reader,
            )

        monkeypatch.setattr(cli, "NewsletterAgent", factory)
        replies = iter(answers or [])
        monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: next(replies))

    sandbox = {"outbox_dir": settings.outbox_dir, "subscribers_file": settings.subscribers_file}
    return use


def test_autonomous_run_exits_cleanly_and_writes_outbox(fake_cli, settings, capsys):
    fake_cli()
    assert cli.main([]) == 0
    assert "Newsletter sent" in capsys.readouterr().out
    assert any(settings.outbox_dir.iterdir())


def test_hitl_run_takes_reviewer_input(fake_cli, settings):
    fake_cli(answers=["a", "r", "Punchier intro please", "approve"])
    assert cli.main(["--mode", "hitl"]) == 0
    assert any(settings.outbox_dir.iterdir())


def test_rejection_sends_nothing(fake_cli, settings, capsys):
    fake_cli(answers=["approve", "x"])
    assert cli.main(["--mode", "hitl"]) == 0
    assert "rejected" in capsys.readouterr().out
    assert not settings.outbox_dir.exists()


def test_configuration_error_exits_with_2(monkeypatch, capsys):
    def broken(settings):
        raise LLMConfigurationError("no key")

    monkeypatch.setattr(cli, "NewsletterAgent", broken)
    assert cli.main([]) == 2
    assert "no key" in capsys.readouterr().out


def test_provider_failure_is_reported_without_traceback(fake_cli, make_llm, capsys):
    llm = make_llm()
    llm.responders.clear()  # every structured call now fails with KeyError
    fake_cli(llm=llm)
    assert cli.main([]) == 1
    out = capsys.readouterr().out
    assert "The agent stopped" in out and "Traceback" not in out
