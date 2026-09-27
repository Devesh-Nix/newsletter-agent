"""Streamlit front end for the Newsletter Agent.

streamlit run app.py
"""

from __future__ import annotations

import base64
import hmac
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from newsletter_agent import DEFAULT_GOAL, AgentMode, HumanDecision, NewsletterAgent, load_settings
from newsletter_agent.agent import NewsletterResult
from newsletter_agent.config import DEFAULT_MODELS, PROVIDER_KEY_ENV
from newsletter_agent.llm import LLMConfigurationError
from newsletter_agent.state import AgentEvent

st.set_page_config(page_title="Newsletter Agent", page_icon="📰", layout="wide")

PROVIDERS = {
    "auto": "Auto-detect from API keys",
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI",
    "google_genai": "Google Gemini",
    "xai": "xAI Grok",
    "ollama": "Ollama (local)",
}
STAGE_COLOR = {
    "plan": "violet",
    "research": "blue",
    "curate": "blue",
    "summarize": "green",
    "write": "orange",
    "critique": "red",
    "render": "gray",
    "review": "violet",
    "publish": "green",
}
KIND_ICON = {
    "step": ":material/arrow_right:",
    "tool": ":material/build:",
    "thought": ":material/psychology:",
    "result": ":material/check_circle:",
    "warning": ":material/warning:",
}
MODES = {
    AgentMode.AUTONOMOUS: ":material/bolt: Fully autonomous",
    AgentMode.HUMAN_IN_THE_LOOP: ":material/person_check: Human-in-the-loop",
}

state = st.session_state
state.setdefault("phase", "idle")  # idle | awaiting_review | done | error
state.setdefault("events", [])


# ---------------------------------------------------------------------------
# Hosting: secrets and access control
# ---------------------------------------------------------------------------


def load_secrets_into_env() -> None:
    """On Streamlit Community Cloud, server configuration (provider keys, LLM_PROVIDER,
    APP_PASSWORD, ...) lives in st.secrets. Copy it into the environment, where the
    agent's settings loader looks. Locally there is no secrets file and this is a no-op."""
    try:
        secrets = {name: st.secrets[name] for name in st.secrets}
    except Exception:  # no secrets.toml
        return
    for name, value in secrets.items():
        if name.isupper() and isinstance(value, str | int | float):
            os.environ.setdefault(name, str(value))


def require_password() -> None:
    """Optional gate for public deployments: with APP_PASSWORD set, only people you give
    the password to can run the agent on the server's API key."""
    expected = os.getenv("APP_PASSWORD")
    if not expected or state.get("unlocked"):
        return
    st.title("📰 Newsletter Agent")
    with st.form("unlock"):
        attempt = st.text_input("Access password", type="password")
        if st.form_submit_button("Unlock", type="primary"):
            if hmac.compare_digest(attempt.encode(), expected.encode()):
                state.unlocked = True
                st.rerun()
            st.error("Incorrect password.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------


def sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Mode",
            options=list(MODES),
            format_func=MODES.get,
            captions=[
                "Plans, researches, writes, self-reviews and sends on its own.",
                "Pauses for your approval of the plan and of the final draft.",
            ],
            key="mode",
        )

        st.subheader("Model", divider="gray")
        provider = st.selectbox("Provider", list(PROVIDERS), format_func=PROVIDERS.get)
        placeholder = DEFAULT_MODELS.get(provider, "provider default")
        model = st.text_input("Model", placeholder=placeholder) or None
        key_vars = PROVIDER_KEY_ENV.get(provider, ())
        api_key = None
        if key_vars:
            api_key = (
                st.text_input(
                    f"{key_vars[0]}",
                    type="password",
                    placeholder="Using the server's key" if os.getenv(key_vars[0]) else "",
                    help="Kept in your browser session only: never stored, logged or shared "
                    "with other visitors.",
                )
                or None
            )

        with st.expander("Agent behaviour"):
            name = st.text_input("Newsletter name", value="The Agentic Brief")
            days = st.selectbox(
                "News window",
                [None, 1, 3, 7, 14, 30],
                format_func=lambda d: "Inferred from the goal" if d is None else f"Last {d} days",
            )
            max_articles = st.slider("Maximum stories", 5, 7, 7)
            rounds = st.slider("Research rounds", 1, 5, 3)
            revisions = st.slider("Self-critique revisions", 0, 4, 2)
            threshold = st.slider("Quality bar (critic score)", 6.0, 10.0, 8.0, 0.5)

        with st.expander("How it works"):
            st.markdown(
                "1. **Plan**: interpret the goal, choose queries and criteria\n"
                "2. **Research**: the LLM calls search tools in a loop\n"
                "3. **Curate**: dedupe and rank the top stories\n"
                "4. **Summarize**: read each article, write a faithful brief\n"
                "5. **Write**: draft the issue\n"
                "6. **Critique**: score the draft, revise until it passes\n"
                "7. **Render** HTML/Markdown, then **send** (simulated)"
            )
    return {
        "mode": mode,
        "llm_provider": provider,
        "llm_model": model,
        "llm_api_key": api_key,
        "newsletter_name": name,
        "sender_name": name,
        "lookback_days": days,
        "max_articles": max_articles,
        "max_research_rounds": rounds,
        "max_revisions": revisions,
        "quality_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Running the agent
# ---------------------------------------------------------------------------


def md(text: str) -> str:
    """Escape text for Streamlit Markdown, where a pair of ``$`` signs starts LaTeX math
    (think "raises $80M ... $20M")."""
    return text.replace("$", r"\$")


def email_preview(html: str, height: int) -> None:
    """Show the email in an isolated iframe. A data: URL gets an opaque origin, so the
    preview can never reach into the app (defence in depth on top of autoescaping)."""
    encoded = base64.b64encode(html.encode("utf-8")).decode("ascii")
    st.iframe(f"data:text/html;charset=utf-8;base64,{encoded}", height=height)


def badges(labels: list[str], color: str = "blue") -> str:
    return " ".join(f":{color}-badge[{md(label).replace(']', '')}]" for label in labels)


def event_line(event: AgentEvent) -> str:
    color = STAGE_COLOR.get(event.stage, "gray")
    text = f"*{md(event.message)}*" if event.kind == "thought" else md(event.message)
    return f":{color}-badge[{event.stage}] {KIND_ICON.get(event.kind, '')} {text}"


def stream_events(events, label: str) -> None:
    """Consume an event stream, rendering it live inside a status box."""
    with st.status(label, expanded=True) as status:
        try:
            for event in events:
                state.events.append(event)
                st.markdown(event_line(event))
        except Exception as exc:  # surface provider/network errors in the UI
            state.phase, state.error = "error", f"{type(exc).__name__}: {exc}"
            status.update(label="The agent stopped with an error", state="error")
            return
        checkpoint = state.agent.pending_review(state.thread_id)
        if checkpoint:
            state.phase, state.checkpoint = "awaiting_review", checkpoint
            status.update(label="Paused for your review", state="complete", expanded=False)
        else:
            state.phase = "done"
            status.update(label="Finished", state="complete", expanded=False)
    st.rerun()


def start_run(goal: str, config: dict) -> None:
    mode = config.pop("mode")
    try:
        settings = load_settings(**config)
        state.agent = NewsletterAgent(settings=settings)
    except LLMConfigurationError as exc:
        state.phase, state.error = "error", str(exc)
        return
    state.events, state.checkpoint = [], None
    state.thread_id, events = state.agent.start(goal, mode)
    stream_events(events, f"Working with {state.agent.model_label}…")


def submit(action: str, feedback: str = "") -> None:
    state.decision = HumanDecision(action=action, feedback=feedback)


# ---------------------------------------------------------------------------
# Human review panels
# ---------------------------------------------------------------------------


def plan_review(checkpoint: dict) -> None:
    plan = checkpoint["plan"]
    with st.container(border=True):
        st.subheader(":material/person_check: Review the research plan")
        left, right = st.columns(2)
        left.markdown(f"**Topic:** {md(plan['topic'])}  \n**Audience:** {md(plan['audience'])}")
        left.markdown(
            f"**Tone:** {md(plan['tone'])}  \n**Window:** last {plan['time_window_days']} days  \n"
            f"**Stories:** {plan['target_article_count']}"
        )
        right.markdown("**Search queries**")
        right.markdown(badges(plan["search_queries"]))
        right.markdown("**Selection criteria**")
        right.markdown("\n".join(f"- {md(c)}" for c in plan["selection_criteria"]))
        st.caption(md(plan["reasoning"]))
        review_controls("plan", approve_label="Approve plan")


def draft_review(checkpoint: dict) -> None:
    with st.container(border=True):
        st.subheader(":material/person_check: Approve the newsletter before it is sent")
        st.markdown(
            f"**Subject:** {md(checkpoint['subject'])}  \n"
            f"**Preheader:** {md(checkpoint['preheader'])}"
        )
        if review := checkpoint.get("review"):
            verdict = "approved" if review["passed"] else "flagged issues"
            st.caption(
                f"Self-review of draft {review['draft_number']}: "
                f"{review['critique']['overall_score']}/10, {verdict}"
            )
        preview, markdown = st.tabs(["Preview", "Markdown"])
        with preview:
            email_preview(checkpoint["html"], height=720)
        with markdown:
            st.code(checkpoint["markdown"], language="markdown")
        review_controls("draft", approve_label="Approve & send")


def review_controls(key: str, approve_label: str) -> None:
    feedback = st.text_area(
        "Feedback (for 'Request changes')",
        key=f"feedback_{key}",
        placeholder="e.g. Focus more on open-source frameworks and less on funding news.",
    )
    approve, revise, reject, _ = st.columns([1.2, 1.2, 1, 3])
    approve.button(
        approve_label, type="primary", icon=":material/check:", on_click=submit, args=("approve",)
    )
    if revise.button("Request changes", icon=":material/edit:"):
        if feedback.strip():
            submit("revise", feedback)
            st.rerun()
        else:
            st.warning("Describe what should change first.")
    reject.button("Reject", icon=":material/close:", on_click=submit, args=("reject",))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def results(result: NewsletterResult) -> None:
    if not result.sent:
        st.warning("The newsletter was rejected and not sent.", icon=":material/block:")
        return

    delivery = result.delivery
    st.success(
        f"**Sent** “{md(delivery.subject)}” to {len(delivery.recipients)} subscribers "
        "(simulated: saved to the outbox).",
        icon=":material/mark_email_read:",
    )
    first, last = result.reviews[0].critique, result.reviews[-1].critique
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Candidates found", result.candidates_found, border=True)
    k2.metric("Stories selected", len(result.selected), border=True)
    k3.metric("Drafts written", len(result.reviews), border=True)
    k4.metric(
        "Final critic score",
        f"{last.overall_score}/10",
        delta=round(last.overall_score - first.overall_score, 1)
        if len(result.reviews) > 1
        else None,
        help="Change is measured against the first draft.",
        border=True,
    )

    tabs = st.tabs(
        [
            ":material/mail: Newsletter",
            ":material/psychology: Self-critique",
            ":material/travel_explore: Research",
            ":material/send: Delivery",
        ]
    )
    with tabs[0]:
        st.markdown(
            f"**Subject:** {md(result.rendered.subject)}  \n"
            f"**Preheader:** {md(result.rendered.preheader)}"
        )
        email_preview(result.rendered.html, height=900)
    with tabs[1]:
        critique_history(result)
    with tabs[2]:
        research_view(result)
    with tabs[3]:
        delivery_view(result)


def critique_history(result: NewsletterResult) -> None:
    for review in reversed(result.reviews):
        critique = review.critique
        verdict = ":green-badge[approved]" if review.passed else ":orange-badge[revise]"
        with st.expander(
            f"Draft {review.draft_number} · {critique.overall_score}/10",
            expanded=review is result.reviews[-1],
        ):
            st.markdown(verdict)
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Criterion": s.criterion, "Score": s.score, "Comment": s.comment}
                        for s in critique.scores
                    ]
                ),
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=10, format="%d / 10", width="small"
                    ),
                    "Comment": st.column_config.TextColumn(width="large"),
                },
                hide_index=True,
            )
            left, right = st.columns(2)
            with left:
                st.markdown("**Strengths**")
                st.markdown("\n".join(f"- {md(s)}" for s in critique.strengths) or "—")
            with right:
                st.markdown("**Issues**")
                issues = [*review.automated_issues, *critique.issues]
                st.markdown("\n".join(f"- {md(i)}" for i in issues) or "None")
            if critique.revision_instructions and not review.passed:
                st.markdown("**Revision instructions sent to the writer**")
                st.markdown("\n".join(f"1. {md(i)}" for i in critique.revision_instructions))


def research_view(result: NewsletterResult) -> None:
    plan = result.plan
    st.markdown(f"**Plan:** {md(plan.reasoning)}")
    st.markdown(badges(plan.search_queries))
    categories = {item.article_id: item.category for item in result.draft.items}
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Story": a.title,
                    "Category": categories.get(a.id, ""),
                    "Source": a.source,
                    "Published": a.date_label,
                    "Found via": a.origin.replace("_", " "),
                    "Link": a.url,
                }
                for a in result.selected
            ]
        ),
        column_config={"Link": st.column_config.LinkColumn(display_text="Open")},
        hide_index=True,
    )


def delivery_view(result: NewsletterResult) -> None:
    delivery = result.delivery
    st.markdown(
        f"**From:** {delivery.sender}  \n**Message-ID:** `{delivery.message_id}`  \n"
        f"**Recipients:** {', '.join(delivery.recipients)}  \n**Saved to:** `{delivery.output_dir}`"
    )
    files = delivery.files
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download HTML",
        Path(files["html"]).read_bytes(),
        "newsletter.html",
        "text/html",
        icon=":material/download:",
    )
    c2.download_button(
        "Download Markdown",
        Path(files["markdown"]).read_bytes(),
        "newsletter.md",
        "text/markdown",
        icon=":material/download:",
    )
    c3.download_button(
        "Download .eml",
        Path(files["eml"]).read_bytes(),
        "newsletter.eml",
        "message/rfc822",
        icon=":material/download:",
    )
    st.code(result.rendered.text, language="text")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

load_secrets_into_env()
require_password()

config = sidebar()
st.title("📰 Newsletter Agent")
st.caption(
    "An autonomous LangGraph agent that plans, researches the web, writes, critiques its "
    "own draft and sends a newsletter, all from one plain-English goal."
)

goal = st.text_area("Goal", value=DEFAULT_GOAL, height=80)
busy = state.phase == "awaiting_review"
if st.button("Run agent", type="primary", icon=":material/play_arrow:", disabled=busy):
    start_run(goal, config)

# A pending human decision resumes the paused run.
if decision := state.pop("decision", None):
    stream_events(state.agent.resume(state.thread_id, decision), "Resuming…")

if state.events and state.phase != "idle":
    with st.expander(f"Agent timeline ({len(state.events)} steps)", expanded=False):
        for event in state.events:
            st.markdown(event_line(event))

if state.phase == "error":
    st.error(state.error, icon=":material/error:")
elif state.phase == "awaiting_review":
    if state.checkpoint["checkpoint"] == "plan_review":
        plan_review(state.checkpoint)
    else:
        draft_review(state.checkpoint)
elif state.phase == "done":
    results(state.agent.result(state.thread_id))
