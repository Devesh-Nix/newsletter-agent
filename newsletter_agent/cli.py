"""Command-line interface.

python -m newsletter_agent                         # autonomous, default goal
python -m newsletter_agent "Weekly AI agents digest for CTOs" --mode hitl
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from newsletter_agent.agent import NewsletterAgent, NewsletterResult
from newsletter_agent.config import DEFAULT_GOAL, load_settings
from newsletter_agent.llm import LLMConfigurationError
from newsletter_agent.models import HumanDecision
from newsletter_agent.nodes.curation import ResearchError
from newsletter_agent.state import AgentEvent, AgentMode

console = Console()

STAGE_STYLE = {
    "plan": "magenta",
    "research": "cyan",
    "curate": "blue",
    "summarize": "green",
    "write": "yellow",
    "critique": "red",
    "render": "bright_black",
    "review": "bright_magenta",
    "publish": "bright_green",
}
KIND_ICON = {"step": "•", "tool": "⚙", "thought": "›", "result": "✓", "warning": "!"}
KIND_STYLE = {"thought": "italic", "warning": "yellow", "tool": "dim", "result": "", "step": "bold"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="newsletter-agent",
        description="Autonomous AI agent that researches, writes, self-critiques and sends "
        "a newsletter from a plain-English goal.",
    )
    parser.add_argument("goal", nargs="?", default=DEFAULT_GOAL, help="What to produce.")
    parser.add_argument(
        "--mode",
        choices=["autonomous", "hitl"],
        default="autonomous",
        help="'hitl' pauses for human approval of the plan and the final draft.",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "google_genai", "xai", "ollama"]
    )
    parser.add_argument("--model", help="Model name override, e.g. claude-opus-5.")
    parser.add_argument("--days", type=int, help="Look-back window in days (default 7).")
    parser.add_argument("--max-revisions", type=int, help="Self-critique revision budget.")
    parser.add_argument("--name", help="Newsletter name.")
    parser.add_argument("--show", action="store_true", help="Print the final newsletter.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)


def print_event(event: AgentEvent) -> None:
    stage = f"[{STAGE_STYLE.get(event.stage, 'white')}]{event.stage:>9}[/]"
    icon = KIND_ICON.get(event.kind, "•")
    style = KIND_STYLE.get(event.kind, "")
    message = f"[{style}]{event.message}[/]" if style else event.message
    console.print(f"{stage} {icon} {message}", highlight=False)


def review_in_terminal(checkpoint: dict[str, Any]) -> HumanDecision:
    console.rule(f"[bold bright_magenta]Human review · {checkpoint['title']}")
    if checkpoint["checkpoint"] == "plan_review":
        plan = checkpoint["plan"]
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("[bold]Topic", plan["topic"])
        table.add_row("[bold]Audience", plan["audience"])
        table.add_row("[bold]Tone", plan["tone"])
        table.add_row("[bold]Window", f"last {plan['time_window_days']} days")
        table.add_row("[bold]Stories", str(plan["target_article_count"]))
        table.add_row("[bold]Queries", "\n".join(plan["search_queries"]))
        table.add_row("[bold]Criteria", "\n".join(plan["selection_criteria"]))
        console.print(table)
    else:
        console.print(Panel(Markdown(checkpoint["markdown"]), title=checkpoint["subject"]))
        if review := checkpoint.get("review"):
            critique = review["critique"]
            console.print(
                f"Self-review: [bold]{critique['overall_score']}/10[/] · "
                f"{'approved' if review['passed'] else 'not approved'}"
            )

    choice = Prompt.ask(
        "[bold]Approve[/], [bold]revise[/] or [bold]reject[/]?",
        choices=["approve", "revise", "reject", "a", "r", "x"],
        default="approve",
        show_choices=False,
    )
    action = {"a": "approve", "r": "revise", "x": "reject"}.get(choice, choice)
    feedback = Prompt.ask("What should change?") if action == "revise" else ""
    console.rule()
    return HumanDecision(action=action, feedback=feedback)


def print_summary(result: NewsletterResult, show: bool) -> None:
    if result.reviews:
        table = Table(title="Self-critique history", title_justify="left")
        table.add_column("Draft", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Verdict")
        table.add_column("Top issue")
        for review in result.reviews:
            issues = [*review.automated_issues, *review.critique.issues]
            table.add_row(
                str(review.draft_number),
                f"{review.critique.overall_score}/10",
                "[green]approved[/]" if review.passed else "[yellow]revise[/]",
                issues[0] if issues else "—",
            )
        console.print(table)

    if not result.sent:
        console.print(Panel("The newsletter was rejected and not sent.", style="yellow"))
        return
    if show and result.rendered:
        console.print(Panel(Markdown(result.rendered.markdown), title="Newsletter"))

    delivery = result.delivery
    console.print(
        Panel(
            f"[bold]Subject:[/] {delivery.subject}\n"
            f"[bold]Preheader:[/] {result.rendered.preheader}\n"
            f"[bold]Recipients:[/] {len(delivery.recipients)} subscribers "
            f"({', '.join(delivery.recipients[:3])}{'…' if len(delivery.recipients) > 3 else ''})\n"
            f"[bold]Stories:[/] {len(result.selected)} selected from "
            f"{result.candidates_found} candidates\n"
            f"[bold]Saved to:[/] {delivery.output_dir}\n"
            f"  newsletter.html · newsletter.md · newsletter.eml · delivery.json",
            title="[bold green]Newsletter sent (simulated)",
            border_style="green",
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    mode = AgentMode.HUMAN_IN_THE_LOOP if args.mode == "hitl" else AgentMode.AUTONOMOUS

    try:
        settings = load_settings(
            llm_provider=args.provider,
            llm_model=args.model,
            lookback_days=args.days,
            max_revisions=args.max_revisions,
            newsletter_name=args.name,
            sender_name=args.name,
        )
        agent = NewsletterAgent(settings=settings)
    except LLMConfigurationError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        return 2

    console.print(
        Panel(
            f"[bold]Goal:[/] {args.goal}\n"
            f"[bold]Model:[/] {agent.model_label}   [bold]Mode:[/] {mode.value.replace('_', '-')}",
            title="[bold]Newsletter Agent",
            border_style="cyan",
        )
    )
    try:
        thread_id, events = agent.start(args.goal, mode)
        while True:
            for event in events:
                print_event(event)
            checkpoint = agent.pending_review(thread_id)
            if checkpoint is None:
                break
            events = agent.resume(thread_id, review_in_terminal(checkpoint))
    except ResearchError as exc:
        console.print(f"[red]Research failed:[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        return 130

    print_summary(agent.result(thread_id), show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
