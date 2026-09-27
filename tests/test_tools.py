"""Search backends, research tools, renderer and mailer (HTTP is mocked)."""

from __future__ import annotations

import json
from email import message_from_bytes
from email.message import Message
from pathlib import Path

import httpx
import pytest

from newsletter_agent.config import Settings
from newsletter_agent.models import DraftItem, NewsletterDraft
from newsletter_agent.tools import render_newsletter, search, send_newsletter
from newsletter_agent.tools.research import build_research_tools
from newsletter_agent.utils import article_id, utc_now
from tests.conftest import make_article

NOW = utc_now().strftime("%a, %d %b %Y %H:%M:%S GMT")

BING_RSS = f"""<?xml version="1.0" encoding="utf-8" ?>
<rss version="2.0" xmlns:News="https://www.bing.com/news/search?q=x&amp;format=rss">
<channel><title>AI agents - BingNews</title>
<item>
  <title>Agents escape the sandbox</title>
  <link>http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;url=https%3a%2f%2ffortune.com%2fagents%2f&amp;c=1</link>
  <description>OpenAI says its agents &lt;b&gt;escaped&lt;/b&gt; a sandbox.</description>
  <pubDate>{NOW}</pubDate>
  <News:Source>Fortune on MSN</News:Source>
</item>
<item>
  <title>Ancient history</title>
  <link>https://example.com/old</link>
  <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
</item>
</channel></rss>"""

GOOGLE_RSS = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Google News</title>
<item>
  <title>Hacks by autonomous AI agents raise legal questions - PBS</title>
  <link>https://news.google.com/rss/articles/CBMi123</link>
  <pubDate>{NOW}</pubDate>
  <source url="https://www.pbs.org">PBS</source>
</item>
</channel></rss>"""

HN_JSON = {
    "hits": [
        {
            "objectID": "1",
            "title": "Small story",
            "url": "https://blog.example.com/a",
            "points": 5,
            "num_comments": 1,
            "created_at": "2026-09-26T10:00:00Z",
        },
        {
            "objectID": "2",
            "title": "Ask HN: Which agent framework?",
            "url": None,
            "points": 300,
            "num_comments": 120,
            "created_at": "2026-09-26T10:00:00Z",
        },
    ]
}


def fake_get(routes: dict[str, httpx.Response | Exception]):
    def get(url, params=None, **kwargs):
        for prefix, response in routes.items():
            if url.startswith(prefix):
                if isinstance(response, Exception):
                    raise response
                response.request = httpx.Request("GET", url)
                return response
        raise AssertionError(f"unexpected URL {url}")

    return get


@pytest.fixture
def settings_obj() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------


def test_bing_news_unwraps_links_credits_publisher_and_filters_old(monkeypatch, settings_obj):
    monkeypatch.setattr(
        httpx, "get", fake_get({search.BING_NEWS_RSS: httpx.Response(200, text=BING_RSS)})
    )
    [article] = search.bing_news("AI agents", 7, 10, settings_obj)

    assert article.url == "https://fortune.com/agents/"
    assert article.id == article_id("https://fortune.com/agents/")
    assert article.source == "Fortune"
    assert article.snippet == "OpenAI says its agents escaped a sandbox."
    assert article.published_at is not None


def test_google_news_strips_publisher_suffix(monkeypatch, settings_obj):
    monkeypatch.setattr(
        httpx, "get", fake_get({search.GOOGLE_NEWS_RSS: httpx.Response(200, text=GOOGLE_RSS)})
    )
    [article] = search.google_news("AI agents", 7, 10, settings_obj)
    assert article.title == "Hacks by autonomous AI agents raise legal questions"
    assert article.source == "PBS"


def test_hacker_news_ranks_by_points_and_links_text_posts(monkeypatch, settings_obj):
    monkeypatch.setattr(
        httpx, "get", fake_get({search.HN_SEARCH_API: httpx.Response(200, json=HN_JSON)})
    )
    articles = search.hacker_news("agent", 7, 10, settings_obj)
    assert [a.points for a in articles] == [300, 5]
    assert articles[0].url == "https://news.ycombinator.com/item?id=2"
    assert articles[0].source == "Hacker News"


def test_http_errors_become_search_errors(monkeypatch, settings_obj):
    monkeypatch.setattr(
        httpx, "get", fake_get({search.BING_NEWS_RSS: httpx.ConnectError("offline")})
    )
    with pytest.raises(search.SearchError, match="offline"):
        search.bing_news("AI agents", 7, 10, settings_obj)


# ---------------------------------------------------------------------------
# LLM-facing research tools
# ---------------------------------------------------------------------------


def call(tool, **args):
    return tool.invoke({"type": "tool_call", "id": "1", "name": tool.name, "args": args})


def test_search_news_falls_back_to_google_when_bing_fails(monkeypatch, settings_obj):
    monkeypatch.setattr(
        httpx,
        "get",
        fake_get(
            {
                search.BING_NEWS_RSS: httpx.ConnectError("blocked"),
                search.GOOGLE_NEWS_RSS: httpx.Response(200, text=GOOGLE_RSS),
            }
        ),
    )
    search_news = build_research_tools(settings_obj)[0]
    message = call(search_news, query="AI agents")

    assert len(message.artifact) == 1
    assert message.artifact[0].origin == "google_news"
    assert f"[{message.artifact[0].id}]" in message.content


def test_research_tool_reports_failure_to_the_model(monkeypatch, settings_obj):
    monkeypatch.setattr(httpx, "get", fake_get({"https://": httpx.ConnectError("offline")}))
    message = call(build_research_tools(settings_obj)[1], query="agents")
    assert message.artifact == []
    assert "failed" in message.content and "offline" in message.content


def test_web_search_tool_only_offered_with_tavily_key():
    assert [t.name for t in build_research_tools(Settings(tavily_api_key=None))] == [
        "search_news",
        "search_hacker_news",
    ]
    names = [t.name for t in build_research_tools(Settings(tavily_api_key="tvly-test"))]
    assert names[-1] == "search_web"


def test_tool_schema_defaults_to_configured_window():
    tool = build_research_tools(Settings(lookback_days=14))[0]
    assert tool.args["days"]["default"] == 14


# ---------------------------------------------------------------------------
# Renderer and mailer
# ---------------------------------------------------------------------------


def make_draft(articles, **overrides) -> NewsletterDraft:
    fields = {
        "subject": "Agents <script>alert(1)</script> weekly",
        "preheader": "Preview text",
        "intro": "Intro & context.",
        "items": [
            DraftItem(
                article_id=a.id,
                headline=f"Headline [{n}]",
                summary="Summary.",
                why_it_matters="Why.",
                category="Launch",
            )
            for n, a in enumerate(articles)
        ],
        "big_picture": "Big picture.",
        "sign_off": "Bye.",
    }
    return NewsletterDraft(**(fields | overrides))


def test_renderer_escapes_llm_text_and_neutralises_unsafe_links():
    good = make_article("Good story")
    evil = good.model_copy(update={"id": "evil0001", "url": "javascript:alert(1)"})
    draft = make_draft([good, evil])
    draft.items.append(draft.items[0].model_copy(update={"article_id": "unknown1"}))

    rendered = render_newsletter(
        draft, {good.id: good, evil.id: evil}, newsletter_name="Brief", period_days=7
    )

    assert "<script>" not in rendered.html and "&lt;script&gt;" in rendered.html
    assert "javascript:" not in rendered.html
    assert rendered.html.count("Read the full story") == 2  # unknown id dropped
    assert f"[Headline \\[0\\]]({good.url})" in rendered.markdown
    assert "Read more: " + good.url in rendered.text


def test_send_newsletter_writes_mime_email_and_receipt(tmp_path):
    subscribers = tmp_path / "subs.json"
    subscribers.write_text(json.dumps([{"name": "A", "email": "a@example.com"}]))
    settings = Settings(outbox_dir=tmp_path / "out", subscribers_file=subscribers)
    article = make_article("Story")
    rendered = render_newsletter(
        make_draft([article], subject="Weekly agents"),
        {article.id: article},
        newsletter_name="Brief",
        period_days=7,
    )

    receipt = send_newsletter(rendered, settings)

    email: Message = message_from_bytes(Path(receipt.files["eml"]).read_bytes())
    assert email["Subject"] == "Weekly agents"
    assert email["To"] == "undisclosed-recipients:;"
    assert [part.get_content_type() for part in email.get_payload()] == ["text/plain", "text/html"]
    assert receipt.recipients == ["a@example.com"]
    saved = json.loads(Path(receipt.files["receipt"]).read_text())
    assert saved["message_id"] == receipt.message_id
