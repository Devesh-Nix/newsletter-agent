"""Pure decision logic inside the nodes: plan normalisation, pick validation,
shortlisting, draft reconciliation and the automated editorial checks."""

from newsletter_agent.config import Settings
from newsletter_agent.models import (
    ArticleSummary,
    DraftItem,
    NewsletterDraft,
    NewsletterPlan,
    RankedPick,
)
from newsletter_agent.nodes.critique import automated_checks
from newsletter_agent.nodes.curation import shortlist, validate_picks
from newsletter_agent.nodes.planning import normalize_plan
from newsletter_agent.nodes.writing import reconcile_items
from tests.conftest import STORY_TOPICS, make_article

SUMMARY = "A factual summary of the development that is long enough to pass the checks easily."


def pick(article_id: str) -> RankedPick:
    return RankedPick(article_id=article_id, category="Launch", relevance=8, rationale="Big.")


def item(article_id: str, **overrides) -> DraftItem:
    fields = {
        "article_id": article_id,
        "headline": f"Headline {article_id}",
        "summary": SUMMARY,
        "why_it_matters": "It matters.",
        "category": "Launch",
    }
    return DraftItem(**(fields | overrides))


def draft(items, **overrides) -> NewsletterDraft:
    fields = {
        "subject": "Agents go enterprise this week",
        "preheader": "Plus a new benchmark",
        "intro": "A short intro.",
        "items": items,
        "big_picture": "The trend.",
        "sign_off": "Bye.",
    }
    return NewsletterDraft(**(fields | overrides))


def test_normalize_plan_clamps_and_cleans():
    plan = NewsletterPlan(
        topic="AI agents",
        audience="builders",
        tone="crisp",
        time_window_days=90,
        target_article_count=12,
        search_queries=['"AI agents"', "ai agents", "", *[f"q{i}" for i in range(10)]],
        selection_criteria=["impact"],
        reasoning="r",
    )
    result = normalize_plan(plan, Settings())
    assert result.target_article_count == 7
    assert result.time_window_days == 31
    assert result.search_queries[:2] == ["AI agents", "q0"]
    assert len(result.search_queries) == 6


def test_configured_window_overrides_the_inferred_one():
    plan = NewsletterPlan(
        topic="AI agents",
        audience="builders",
        tone="crisp",
        time_window_days=7,
        target_article_count=6,
        search_queries=["AI agents"],
        selection_criteria=["impact"],
        reasoning="r",
    )
    assert normalize_plan(plan, Settings()).time_window_days == 7
    assert normalize_plan(plan, Settings(lookback_days=14)).time_window_days == 14


def test_validate_picks_drops_bad_ids_and_duplicates_then_tops_up():
    pool = [make_article(t) for t in STORY_TOPICS[:6]]
    rewrite = make_article(STORY_TOPICS[0] + " today")  # same story, different URL
    pool.insert(1, rewrite)
    picks = [pick(pool[0].id), pick("deadbeef"), pick(pool[0].id), pick(rewrite.id)]

    result = validate_picks(picks, pool, target=5)

    ids = [p.article_id for p in result]
    assert len(ids) == 5 and len(set(ids)) == 5
    assert ids[0] == pool[0].id
    assert "deadbeef" not in ids and rewrite.id not in ids


def test_validate_picks_accepts_bracketed_ids():
    pool = [make_article("Story")]
    assert validate_picks([pick(f"[{pool[0].id}]")], pool, 1)[0].article_id == pool[0].id


def test_shortlist_interleaves_sources_and_ranks_hn_by_points():
    news = [make_article(f"News {i}", days_ago=i) for i in range(5)]
    hn = [make_article(f"HN {i}", origin="hacker_news", points=i * 10) for i in range(5)]
    result = shortlist(news + hn, limit=4)
    assert [a.origin for a in result] == ["bing_news", "hacker_news"] * 2
    assert result[0].title == "News 0" and result[1].title == "HN 4"


def test_reconcile_items_restores_dropped_stories_and_drops_unknown_ids():
    selected = [make_article(t) for t in STORY_TOPICS[:3]]
    summaries = {
        a.id: ArticleSummary(headline=a.title, summary=SUMMARY, why_it_matters="w")
        for a in selected
    }
    state = {"selected": selected, "summaries": summaries, "picks": [pick(selected[2].id)]}
    written = draft([item(selected[0].id), item("unknown1"), item(selected[0].id)])

    repaired, restored = reconcile_items(written, state)

    assert [i.article_id for i in repaired.items] == [a.id for a in selected]
    assert restored == [selected[1].id, selected[2].id]
    assert repaired.items[2].category == "Launch"


def test_automated_checks_pass_a_clean_draft():
    ids = [f"id{i:06d}" for i in range(5)]
    assert automated_checks(draft([item(i) for i in ids]), 5, Settings()) == []


def test_automated_checks_flag_editorial_problems():
    ids = [f"id{i:06d}" for i in range(3)]
    bad = draft(
        [
            item(ids[0], summary="Too short."),
            item(ids[1], why_it_matters=""),
            item(ids[2], summary="**Bold** " + SUMMARY),
        ],
        subject="x" * 80,
        preheader="x" * 80,
    )
    issues = " | ".join(automated_checks(bad, expected_items=5, settings=Settings()))

    assert "Subject line is 80 characters" in issues
    assert "Preheader repeats the subject" in issues
    assert "Only 3 stories" in issues
    assert "Story 1 summary is only 2 words" in issues
    assert "Story 2 is missing" in issues
    assert "Markdown/HTML" in issues
