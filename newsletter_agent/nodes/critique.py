"""Step 6 — Self-reflection: the agent critiques its own draft and decides whether to revise.

Two complementary reviewers run on every draft:

* **Automated checks**: deterministic rules (length limits, completeness, stray markup)
  that an LLM might overlook.
* **LLM critic**: a demanding-editor rubric (relevance, accuracy against the source
  briefs, clarity, engagement, structure, insight) that yields scores, issues and precise
  revision instructions.

The draft moves on only when both are satisfied, or when the revision budget is spent.
"""

from __future__ import annotations

import re
from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command

from newsletter_agent.config import Settings
from newsletter_agent.llm import structured_output
from newsletter_agent.models import Critique, NewsletterDraft, ReviewRecord
from newsletter_agent.nodes.common import bullet_list, format_briefs, format_draft, today
from newsletter_agent.prompts import CRITIC_PROMPT
from newsletter_agent.state import AgentContext, AgentState, emit

_MARKUP_RE = re.compile(r"\*\*|__|</?[a-z][^>]*>|^#{1,6}\s|\[[^\]]+\]\(", re.IGNORECASE | re.M)


def critique(
    state: AgentState, runtime: Runtime[AgentContext]
) -> Command[Literal["write", "render"]]:
    ctx = runtime.context
    settings = ctx.settings
    plan = state["plan"]
    draft = state["draft"]
    draft_number = state["drafts_written"]

    emit(runtime, "critique", f"Critiquing draft {draft_number} against the editorial rubric")
    issues = automated_checks(draft, expected_items=len(state["selected"]), settings=settings)

    critic = CRITIC_PROMPT | structured_output(ctx.llm, Critique)
    verdict = critic.invoke(
        {
            "today": today(),
            "newsletter_name": settings.newsletter_name,
            "topic": plan.topic,
            "audience": plan.audience,
            "tone": plan.tone,
            "goal": state["goal"],
            "threshold": settings.quality_threshold,
            "briefs": format_briefs(state),
            "check_issues": bullet_list(issues),
            "draft": format_draft(draft),
        }
    )
    verdict = verdict.model_copy(
        update={"overall_score": round(max(1.0, min(verdict.overall_score, 10.0)), 1)}
    )
    passed = verdict.approved and verdict.overall_score >= settings.quality_threshold and not issues
    record = ReviewRecord(
        draft_number=draft_number, critique=verdict, automated_issues=issues, passed=passed
    )

    scores = ", ".join(f"{s.criterion.lower()} {s.score}" for s in verdict.scores)
    emit(
        runtime,
        "critique",
        f"Draft {draft_number} scored {verdict.overall_score}/10 ({scores})",
        kind="thought",
        review=record.model_dump(),
    )
    for issue in [*issues, *verdict.issues]:
        emit(runtime, "critique", issue, kind="warning")

    revisions_done = draft_number - 1
    if passed:
        emit(runtime, "critique", "Draft approved by the self-review", kind="result")
        goto = "render"
    elif revisions_done < settings.max_revisions:
        emit(runtime, "critique", "Not good enough yet; sending it back for revision")
        goto = "write"
    else:
        emit(
            runtime,
            "critique",
            f"Revision budget ({settings.max_revisions}) spent; moving on with the best effort",
            kind="warning",
        )
        goto = "render"
    return Command(goto=goto, update={"reviews": [record]})


def automated_checks(draft: NewsletterDraft, expected_items: int, settings: Settings) -> list[str]:
    """Deterministic editorial rules. Returns human-readable issues (empty = all good)."""
    issues: list[str] = []

    subject_len = len(draft.subject.strip())
    if subject_len > 70:
        issues.append(f"Subject line is {subject_len} characters; keep it under 60.")
    elif subject_len < 15:
        issues.append("Subject line is too short to tell readers what's inside.")
    if len(draft.preheader.strip()) > 110:
        issues.append(f"Preheader is {len(draft.preheader)} characters; keep it under 100.")
    if draft.preheader.strip().lower() == draft.subject.strip().lower():
        issues.append("Preheader repeats the subject line; make it add information.")

    minimum = min(settings.min_articles, expected_items)
    if len(draft.items) < minimum:
        issues.append(f"Only {len(draft.items)} stories; the issue needs at least {minimum}.")

    intro_words = len(draft.intro.split())
    if intro_words > 90:
        issues.append(f"Intro is {intro_words} words; keep it to 2-3 sentences.")
    if not draft.big_picture.strip():
        issues.append("The big-picture section is missing.")

    headlines = [item.headline.strip().lower() for item in draft.items]
    if len(set(headlines)) < len(headlines):
        issues.append("Two stories share the same headline.")

    for i, item in enumerate(draft.items, start=1):
        words = len(item.summary.split())
        if words > 100:
            issues.append(f"Story {i} summary is {words} words; tighten it to under 80.")
        elif words < 15:
            issues.append(f"Story {i} summary is only {words} words; add the key detail.")
        if not item.why_it_matters.strip():
            issues.append(f"Story {i} is missing its 'why it matters' line.")

    fields = [draft.subject, draft.preheader, draft.intro, draft.big_picture, draft.sign_off]
    fields += [text for item in draft.items for text in (item.headline, item.summary)]
    if any(_MARKUP_RE.search(text) for text in fields):
        issues.append("Remove Markdown/HTML markup; every field must be plain text.")
    return issues
