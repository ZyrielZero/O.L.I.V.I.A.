"""Real-backend tests for the Phase 0 memory correctness fixes.

These run against a real temporary ChromaDB — deliberately no mocks. The bugs
they guard against (wrong ordering, never-invoked TTL pruning, tier-dropping
truncation) were invisible to the mock-based suite.
"""

from datetime import datetime, timedelta

import pytest

from src.core.memory.smart_memory import SmartMemoryDB


@pytest.fixture(scope="module")
def _shared_db(tmp_path_factory):
    """One real ChromaDB per module (model load is expensive)."""
    db = SmartMemoryDB(persist_directory=str(tmp_path_factory.mktemp("memdb")))
    yield db
    db.close()


@pytest.fixture
def mem_db(_shared_db):
    """Fresh collections for every test."""
    _shared_db.clear_all()
    return _shared_db


def _add_with_ts(collection, doc: str, ts: datetime, idx: int, **meta) -> None:
    """Insert a document with an explicit timestamp."""
    collection.add(
        documents=[doc],
        metadatas=[{"timestamp": ts.isoformat(), **meta}],
        ids=[f"t_{idx}_{ts.strftime('%Y%m%d%H%M%S')}"],
    )


# ===== 0.1: TTL pruning actually deletes =====


@pytest.mark.integration
def test_prune_expired_deletes_old_conversations(mem_db):
    now = datetime.now()
    for i in range(3):
        _add_with_ts(mem_db.conversations, f"old conversation {i}", now - timedelta(days=40), i)
    for i in range(2):
        _add_with_ts(mem_db.conversations, f"fresh conversation {i}", now, 100 + i)

    pruned = mem_db.prune_expired(conv_days=30)

    assert pruned["conversations"] == 3
    assert mem_db.conversations.count() == 2
    remaining = mem_db.conversations.get()["documents"]
    assert all("fresh" in d for d in remaining)


@pytest.mark.integration
def test_prune_expired_deletes_old_summaries(mem_db):
    now = datetime.now()
    _add_with_ts(mem_db.summaries, "ancient summary", now - timedelta(days=400), 0)
    _add_with_ts(mem_db.summaries, "recent summary", now - timedelta(days=10), 1)

    pruned = mem_db.prune_expired(summary_days=365)

    assert pruned["summaries"] == 1
    assert mem_db.summaries.count() == 1
    assert "recent" in mem_db.summaries.get()["documents"][0]


# ===== 0.3: get_summaries returns the NEWEST n =====


@pytest.mark.integration
def test_get_summaries_returns_newest_first(mem_db):
    now = datetime.now()
    # Insert in chronological order so insertion order == oldest-first;
    # the old `.get(limit=n)` bug would return these oldest entries
    for i in range(10):
        _add_with_ts(mem_db.summaries, f"summary day {i}", now - timedelta(days=9 - i), i)

    result = mem_db.get_summaries(5)

    assert len(result) == 5
    # The five newest are days 5..9, newest (day 9) first
    assert result == [f"summary day {i}" for i in range(9, 4, -1)]


# ===== 0.2: get_recent_conversations ordering + limit =====


@pytest.mark.integration
def test_get_recent_conversations_newest_first(mem_db):
    now = datetime.now()
    for i in range(8):
        _add_with_ts(
            mem_db.conversations, f"conversation {i}", now - timedelta(hours=7 - i), i
        )

    result = mem_db.get_recent_conversations(3)

    assert result == ["conversation 7", "conversation 6", "conversation 5"]


@pytest.mark.integration
def test_get_recent_conversations_respects_limit(mem_db):
    now = datetime.now()
    for i in range(5):
        _add_with_ts(mem_db.conversations, f"conversation {i}", now - timedelta(minutes=i), i)

    assert len(mem_db.get_recent_conversations(2)) == 2
    assert len(mem_db.get_recent_conversations(50)) == 5


# ===== Phase 2: browse / delete management surface =====


@pytest.mark.integration
def test_browse_add_delete_roundtrip(mem_db):
    fact_id = mem_db.add_fact("User plays guitar", "hobby")
    assert fact_id

    entries = mem_db.browse_entries("facts")
    match = [e for e in entries if e["id"] == fact_id]
    assert match and match[0]["document"] == "User plays guitar"
    assert match[0]["metadata"]["category"] == "hobby"

    searched = mem_db.browse_entries("facts", query="guitar")
    assert any(e["id"] == fact_id for e in searched)

    assert mem_db.delete_entry(fact_id) is True
    assert mem_db.delete_entry(fact_id) is False  # already gone
    assert mem_db.delete_entry("bogus_id") is False
    assert mem_db.db_size_bytes() > 0


# ===== 0.8: search_all merges tiers deterministically =====


@pytest.mark.integration
def test_search_all_prioritizes_facts_over_other_tiers(mem_db):
    """Facts must survive truncation regardless of tier completion order."""
    mem_db.facts.add(
        documents=["User's favorite color is teal"],
        metadatas=[{"category": "preference", "timestamp": datetime.now().isoformat()}],
        ids=["fact_1"],
    )
    now = datetime.now()
    for i in range(4):
        _add_with_ts(
            mem_db.conversations,
            f"User: what colors do you like?\nAssistant: reply {i}",
            now,
            i,
        )
    _add_with_ts(mem_db.summaries, "Talked about favorite colors", now, 50, period="session")

    # Run several times: with the old as_completed() merge the surviving
    # tier depended on thread completion order, so the fact could be
    # truncated away entirely
    for _ in range(5):
        result = mem_db.search_all("favorite color", n_results=2)
        parts = result.split("\n---\n")
        assert parts[0] == "User's favorite color is teal"
        assert len(parts) == 2
        # The second slot must never displace the fact; it belongs to a
        # lower-priority tier (summary or conversation, by recall)
        assert parts[1] != parts[0]
