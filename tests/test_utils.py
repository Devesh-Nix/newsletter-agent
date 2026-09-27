from datetime import UTC, datetime, timedelta

from newsletter_agent.utils import (
    article_id,
    clean_text,
    dedupe_articles,
    is_within_window,
    normalize_url,
    parse_date,
    title_similarity,
)
from tests.conftest import make_article


def test_normalize_url_removes_tracking_and_cosmetic_differences():
    variants = [
        "https://www.example.com/story/?utm_source=x&utm_medium=y",
        "http://example.com/story#comments",
        "https://example.com/story?fbclid=abc",
    ]
    assert {normalize_url(v) for v in variants} == {"https://example.com/story"}
    assert len({article_id(v) for v in variants}) == 1


def test_normalize_url_keeps_meaningful_query_params():
    assert normalize_url("https://example.com/a?id=2&page=1") == "https://example.com/a?id=2&page=1"


def test_parse_date_handles_rss_and_iso_formats():
    rss = parse_date("Sun, 27 Sep 2026 01:54:00 GMT")
    iso = parse_date("2026-09-27T01:54:00Z")
    assert rss == iso == datetime(2026, 9, 27, 1, 54, tzinfo=UTC)
    assert parse_date("not a date") is None
    assert parse_date(None) is None


def test_is_within_window():
    now = datetime(2026, 9, 27, tzinfo=UTC)
    assert is_within_window(None, 7, now)
    assert is_within_window(now - timedelta(days=3), 7, now)
    assert not is_within_window(now - timedelta(days=20), 7, now)


def test_clean_text_strips_markup_and_truncates():
    assert clean_text("<p>Agents &amp; tools</p>\n\n<b>ship</b>") == "Agents & tools ship"
    assert clean_text("one two three four five", max_chars=12) == "one two…"


def test_title_similarity_detects_rewrites_of_the_same_story():
    a = "OpenAI launches computer-use agent for enterprise customers"
    b = "OpenAI launches enterprise computer-use agent"
    assert title_similarity(a, b) >= 0.6
    assert title_similarity(a, "EU publishes AI liability guidance") < 0.2


def test_dedupe_prefers_richer_copy_and_collapses_near_duplicates():
    google = make_article("OpenAI launches computer-use agent", origin="google_news")
    bing = google.model_copy(update={"origin": "bing_news"})  # same URL, better origin
    rewrite = make_article("OpenAI launches a computer-use agent", origin="hacker_news")
    other = make_article("EU publishes AI liability guidance")

    result = dedupe_articles([google, bing, rewrite, other])

    assert len(result) == 2
    assert {a.origin for a in result} == {"bing_news"}
