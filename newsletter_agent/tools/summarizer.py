"""Summarizer tool: condense each selected article into a faithful, audience-aware brief.

Two strategies with the same output:

* **Per article** (default): one focused call per article, run concurrently, with
  failures isolated to the article that failed.
* **Batched**: every article in a single call. Used when model calls are rate-limited
  (e.g. the Gemini free tier), where it cuts a run's request count roughly in half.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from newsletter_agent.llm import structured_output
from newsletter_agent.models import (
    Article,
    ArticleContent,
    ArticleSummary,
    NewsletterPlan,
    SummaryBatch,
)
from newsletter_agent.prompts import BATCH_SUMMARIZER_PROMPT, SUMMARIZER_PROMPT
from newsletter_agent.utils import truncate_words

log = logging.getLogger(__name__)


def is_fallback(summary: ArticleSummary) -> bool:
    """True for snippet-based stand-ins produced when the model call failed."""
    return not summary.why_it_matters


def _fallback_summary(article: Article) -> ArticleSummary:
    """Used when the LLM call fails, so one bad article never sinks the issue."""
    return ArticleSummary(
        headline=article.title,
        summary=truncate_words(article.snippet or article.title, 70),
        why_it_matters="",
    )


def _text_kind(content: ArticleContent) -> str:
    return "full article" if content.from_full_page else "search snippet only"


def summarize_articles(
    llm: BaseChatModel,
    articles: list[Article],
    contents: list[ArticleContent],
    plan: NewsletterPlan,
    newsletter_name: str,
    *,
    batched: bool = False,
    max_concurrency: int = 4,
) -> list[ArticleSummary]:
    """Summarise all articles; returns summaries in input order."""
    audience = {"newsletter_name": newsletter_name, "topic": plan.topic, "audience": plan.audience}
    if batched:
        return _summarize_in_one_call(llm, articles, contents, audience)

    chain = SUMMARIZER_PROMPT | structured_output(llm, ArticleSummary)
    inputs = [
        audience
        | {
            "title": article.title,
            "source": article.source,
            "date": article.date_label,
            "url": article.url,
            "text_kind": _text_kind(content),
            "text": content.text,
        }
        for article, content in zip(articles, contents, strict=True)
    ]
    results = chain.batch(
        inputs, config={"max_concurrency": max_concurrency}, return_exceptions=True
    )

    summaries: list[ArticleSummary] = []
    for article, result in zip(articles, results, strict=True):
        if isinstance(result, ArticleSummary):
            summaries.append(result)
        else:
            log.warning("Summarising %s failed: %s", article.url, result)
            summaries.append(_fallback_summary(article))
    return summaries


def _summarize_in_one_call(
    llm: BaseChatModel,
    articles: list[Article],
    contents: list[ArticleContent],
    audience: dict[str, str],
) -> list[ArticleSummary]:
    blocks = [
        f"[{article.id}] {article.title}\n"
        f"Source: {article.source} ({article.date_label}) · {article.url}\n"
        f'Text ({_text_kind(content)}):\n"""\n{content.text}\n"""'
        for article, content in zip(articles, contents, strict=True)
    ]
    chain = BATCH_SUMMARIZER_PROMPT | structured_output(llm, SummaryBatch)
    try:
        batch = chain.invoke(audience | {"articles": "\n\n".join(blocks)})
        by_id = {s.article_id.strip("[] "): s for s in batch.summaries}
    except Exception as exc:  # degrade to snippets rather than abort the issue
        log.warning("Batched summarisation failed: %s", exc)
        by_id = {}

    return [
        ArticleSummary(
            headline=found.headline,
            summary=found.summary,
            why_it_matters=found.why_it_matters,
        )
        if (found := by_id.get(article.id))
        else _fallback_summary(article)
        for article in articles
    ]
