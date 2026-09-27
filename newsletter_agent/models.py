"""Typed data contracts passed between the agent's steps.

Two kinds of models live here:

* **Domain records** (`Article`, `ArticleContent`, `RenderedNewsletter`, `DeliveryReceipt`)
  are produced by deterministic tools.
* **LLM output schemas** (`NewsletterPlan`, `CurationResult`, `ArticleSummary`,
  `NewsletterDraft`, `Critique`) are requested through structured output. They avoid
  numeric/length constraints and exotic types so that every supported provider can
  honour them; limits are stated in field descriptions and enforced in code instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Domain records
# ---------------------------------------------------------------------------


class Article(BaseModel):
    """A candidate news item discovered by a research tool."""

    id: str = Field(description="Stable short hash of the normalised URL.")
    title: str
    url: str
    source: str = Field(description="Publisher or domain, e.g. 'TechCrunch'.")
    published_at: datetime | None = None
    snippet: str = ""
    origin: str = Field(description="Tool that found it: bing_news, google_news, hacker_news, ...")
    points: int | None = Field(default=None, description="Hacker News points, if applicable.")
    comments: int | None = None
    query: str = Field(default="", description="The search query that surfaced this item.")

    @property
    def date_label(self) -> str:
        return self.published_at.strftime("%b %d, %Y") if self.published_at else "Recent"


class ArticleContent(BaseModel):
    """Full text extracted from an article page (or its snippet as a fallback)."""

    url: str
    text: str
    from_full_page: bool = Field(description="False when only the search snippet was available.")


class RenderedNewsletter(BaseModel):
    subject: str
    preheader: str
    html: str
    markdown: str
    text: str


class DeliveryReceipt(BaseModel):
    """Proof of the simulated send."""

    message_id: str
    subject: str
    sender: str
    recipients: list[str]
    sent_at: datetime
    output_dir: str
    files: dict[str, str] = Field(description="Kind of file -> path on disk.")


# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------


class NewsletterPlan(BaseModel):
    """The planner's interpretation of the goal and its research strategy."""

    topic: str = Field(description="Precise topic of the newsletter, e.g. 'AI agents'.")
    audience: str = Field(description="Who the newsletter is for, inferred from the goal.")
    tone: str = Field(description="Editorial voice, e.g. 'sharp, informed, lightly witty'.")
    time_window_days: int = Field(description="How many days back the news should cover.")
    target_article_count: int = Field(description="How many stories to feature (5-7).")
    search_queries: list[str] = Field(
        description="4-6 short, diverse, keyword-style search queries covering different "
        "angles of the topic. No dates or operators."
    )
    selection_criteria: list[str] = Field(
        description="3-5 criteria used to decide which stories make the cut."
    )
    reasoning: str = Field(description="2-3 sentences explaining the plan.")


class RankedPick(BaseModel):
    article_id: str = Field(description="The id shown in brackets in the candidate list.")
    category: str = Field(
        description="One short label, e.g. Launch, Research, Open Source, Funding, "
        "Enterprise, Policy, Security."
    )
    relevance: int = Field(description="1-10 score of relevance and importance.")
    rationale: str = Field(description="One sentence on why this story made the cut.")


class CurationResult(BaseModel):
    picks: list[RankedPick] = Field(description="Selected stories, most important first.")
    coverage_notes: str = Field(description="One or two sentences on the overall selection.")


class ArticleSummary(BaseModel):
    headline: str = Field(description="Informative headline, at most 12 words, no clickbait.")
    summary: str = Field(description="2-3 sentence factual summary (45-80 words).")
    why_it_matters: str = Field(description="One sentence on why the audience should care.")


class DraftItem(BaseModel):
    article_id: str = Field(description="Id of the source article this item covers.")
    headline: str
    summary: str
    why_it_matters: str
    category: str


class NewsletterDraft(BaseModel):
    subject: str = Field(description="Email subject line, specific and under 60 characters.")
    preheader: str = Field(description="Inbox preview text, under 100 characters.")
    intro: str = Field(description="2-3 sentence opening that ties the week's themes together.")
    items: list[DraftItem] = Field(description="The featured stories, most important first.")
    big_picture: str = Field(description="2-4 sentences synthesising the trend across stories.")
    sign_off: str = Field(description="One or two friendly closing sentences.")


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(description="1-10.")
    comment: str


class Critique(BaseModel):
    """The self-reflection step's verdict on a draft."""

    scores: list[CriterionScore]
    overall_score: float = Field(description="1-10 overall publish-readiness.")
    strengths: list[str]
    issues: list[str] = Field(description="Concrete problems, most severe first. Empty if none.")
    revision_instructions: list[str] = Field(
        description="Actionable edits the writer should make. Empty if publish-ready."
    )
    approved: bool = Field(description="True only if the draft is ready to send as-is.")


# ---------------------------------------------------------------------------
# Review bookkeeping
# ---------------------------------------------------------------------------


class ReviewRecord(BaseModel):
    """One pass of the self-reflection loop: LLM critique plus automated checks."""

    draft_number: int
    critique: Critique
    automated_issues: list[str]
    passed: bool


class HumanDecision(BaseModel):
    """A reviewer's answer at a human-in-the-loop checkpoint."""

    action: Literal["approve", "revise", "reject"]
    feedback: str = ""
