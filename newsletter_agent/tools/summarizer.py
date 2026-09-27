"""Summarizer tool: condense each selected article into a faithful, audience-aware brief."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from newsletter_agent.llm import structured_output
from newsletter_agent.models import Article, ArticleContent, ArticleSummary, NewsletterPlan
from newsletter_agent.prompts import SUMMARIZER_PROMPT
from newsletter_agent.utils import truncate_words

log = logging.getLogger(__name__)


def _fallback_summary(article: Article) -> ArticleSummary:
    """Used when the LLM call fails, so one bad article never sinks the issue."""
    return ArticleSummary(
        headline=article.title,
        summary=truncate_words(article.snippet or article.title, 70),
        why_it_matters="",
    )


def summarize_articles(
    llm: BaseChatModel,
    articles: list[Article],
    contents: list[ArticleContent],
    plan: NewsletterPlan,
    newsletter_name: str,
    max_concurrency: int = 4,
) -> list[ArticleSummary]:
    """Summarise all articles concurrently; returns summaries in input order."""
    chain = SUMMARIZER_PROMPT | structured_output(llm, ArticleSummary)
    inputs = [
        {
            "newsletter_name": newsletter_name,
            "topic": plan.topic,
            "audience": plan.audience,
            "title": article.title,
            "source": article.source,
            "date": article.date_label,
            "url": article.url,
            "text_kind": "full article" if content.from_full_page else "search snippet only",
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
