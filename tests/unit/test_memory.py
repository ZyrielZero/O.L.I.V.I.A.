"""
Memory persistence and retrieval tests.
Tests for ChromaDB persistence, semantic search, and HNSW parameters.
"""

import shutil
import tempfile

import pytest

from src.core.memory.smart_memory import SmartMemoryDB


@pytest.fixture
def persistent_db_dir():
    """Create a persistent temp directory for memory tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ===== Test 1: Memory Fact Persistence =====

@pytest.mark.memory
@pytest.mark.slow
def test_memory_fact_persistence(persistent_db_dir):
    """Facts persist across database restarts."""
    # Create first instance and add facts
    db1 = SmartMemoryDB(persist_directory=persistent_db_dir)
    db1.add_fact("User's favorite food is pizza", "preference")
    db1.add_fact("User lives in New York", "personal")

    # Verify facts exist
    stats1 = db1.get_stats()
    assert stats1["facts"] == 2

    # Delete reference (simulate shutdown)
    del db1

    # Create new instance from same directory
    db2 = SmartMemoryDB(persist_directory=persistent_db_dir)

    # Facts should still exist
    stats2 = db2.get_stats()
    assert stats2["facts"] == 2

    all_facts = db2.get_all_facts()
    assert "pizza" in all_facts.lower()
    assert "New York" in all_facts


# ===== Test 2: Memory Conversation Storage =====

@pytest.mark.memory
def test_memory_conversation_storage(persistent_db_dir):
    """Conversations are stored with correct metadata."""
    db = SmartMemoryDB(persist_directory=persistent_db_dir)

    # Add conversation
    user_msg = "Tell me a joke"
    ai_msg = "Why did the programmer quit? Because he didn't get arrays."

    db.add_conversation(user_msg, ai_msg, auto_extract_facts=False)

    # Get recent conversations
    recent = db.get_recent_conversations(n=1)

    assert len(recent) == 1
    assert "Tell me a joke" in recent[0]
    assert "arrays" in recent[0]


# ===== Test 3: Memory Summary Creation =====

@pytest.mark.memory
def test_memory_summary_creation(persistent_db_dir):
    """Summaries can be added and retrieved."""
    db = SmartMemoryDB(persist_directory=persistent_db_dir)

    # Add summaries
    db.add_summary("User discussed work projects", "session")
    db.add_summary("Conversation about hobbies", "session")

    # Retrieve summaries
    summaries = db.get_summaries(n=5)

    assert len(summaries) == 2
    assert any("work" in s.lower() for s in summaries)
    assert any("hobbies" in s.lower() for s in summaries)


# ===== Test 4: Semantic Search Relevance =====

@pytest.mark.memory
def test_memory_semantic_search_relevance(persistent_db_dir):
    """Relevant results are ranked higher in semantic search."""
    db = SmartMemoryDB(persist_directory=persistent_db_dir)

    # Add diverse conversations
    db.add_conversation(
        "What's the weather forecast?",
        "It's going to be sunny tomorrow, around 75 degrees.",
        auto_extract_facts=False
    )
    db.add_conversation(
        "Tell me about Python",
        "Python is a versatile programming language used for web development and AI.",
        auto_extract_facts=False
    )
    db.add_conversation(
        "Will it rain this weekend?",
        "The forecast shows no rain expected for the next few days.",
        auto_extract_facts=False
    )

    # Search for weather
    results = db.search_conversations("weather forecast rain", n_results=3)

    # Weather-related conversations should be found first
    assert "sunny" in results.lower() or "rain" in results.lower() or "forecast" in results.lower()
    # Python shouldn't be the primary result for weather query
    lines = results.split("\n")
    first_result = lines[0] if lines else ""
    assert "Python" not in first_result


# ===== Test 5: Memory Clear All =====

@pytest.mark.memory
def test_memory_clear_all(persistent_db_dir):
    """clear_all removes all data from all collections."""
    db = SmartMemoryDB(persist_directory=persistent_db_dir)

    # Add data to all tiers
    db.add_fact("Test fact", "general")
    db.add_conversation("Hello", "Hi there")
    db.add_summary("Test summary")

    # Verify data exists
    stats_before = db.get_stats()
    assert stats_before["total"] >= 3

    # Clear all
    db.clear_all()

    # Verify all empty
    stats_after = db.get_stats()
    assert stats_after["facts"] == 0
    assert stats_after["conversations"] == 0
    assert stats_after["summaries"] == 0
    assert stats_after["total"] == 0


# ===== Test 6: Memory Stats Accuracy =====

@pytest.mark.memory
def test_memory_stats_accuracy(persistent_db_dir):
    """get_stats returns accurate counts for all collections."""
    db = SmartMemoryDB(persist_directory=persistent_db_dir)

    # Add known quantities
    for i in range(3):
        db.add_fact(f"Fact {i}", "general")

    for i in range(5):
        db.add_conversation(f"User message {i}", f"AI response {i}", auto_extract_facts=False)

    for i in range(2):
        db.add_summary(f"Summary {i}")

    # Check stats
    stats = db.get_stats()

    assert stats["facts"] == 3
    assert stats["conversations"] == 5
    assert stats["summaries"] == 2
    assert stats["total"] == 10


# ===== Test 7: HNSW Parameters =====

@pytest.mark.memory
def test_memory_hnsw_parameters(persistent_db_dir):
    """HNSW index parameters are set correctly."""
    db = SmartMemoryDB(persist_directory=persistent_db_dir)

    # Verify HNSW metadata is defined
    expected_params = {
        "hnsw:search_ef": 100,
        "hnsw:num_threads": 4,
    }

    for key, value in expected_params.items():
        assert key in db.HNSW_METADATA
        assert db.HNSW_METADATA[key] == value


# ===== Additional Memory Tests =====

@pytest.mark.memory
def test_memory_get_relevant_context(persistent_db_dir):
    """get_relevant_context returns appropriate context for queries."""
    db = SmartMemoryDB(persist_directory=persistent_db_dir)

    # Add data
    db.add_fact("User is a Python developer", "personal")
    db.add_conversation(
        "How do I use async in Python?",
        "You can use async/await keywords for asynchronous programming.",
        auto_extract_facts=False
    )

    # Get relevant context
    context = db.get_relevant_context("Python programming")

    assert len(context) > 0
    assert "Python" in context or "python" in context.lower()


@pytest.mark.memory
def test_memory_empty_search(persistent_db_dir):
    """Searching empty database returns empty string."""
    db = SmartMemoryDB(persist_directory=persistent_db_dir)

    # Search empty database
    results = db.search_all("test query")

    assert results == ""


@pytest.mark.memory
def test_memory_backward_compatible_wrapper(persistent_db_dir):
    """MemoryDB backward compatible wrapper works correctly."""
    from src.core.memory.smart_memory import MemoryDB

    db = MemoryDB(persist_directory=persistent_db_dir)

    # Test add_memory
    db.add_memory("User: Hello\nAI: Hi there!", source="test")

    # Test search_memory
    results = db.search_memory("hello")
    assert "Hello" in results or "hello" in results.lower()

    # Test get_memory_count
    count = db.get_memory_count()
    assert count >= 1
