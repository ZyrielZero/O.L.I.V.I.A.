"""
Unit tests for SmartMemoryDB.
Tests fact extraction, duplicate detection, and context generation.
"""

import shutil
import tempfile

import pytest

from src.core.memory.smart_memory import SmartMemoryDB


@pytest.fixture
def temp_db():
    """Create a temporary memory database for testing."""
    temp_dir = tempfile.mkdtemp()
    db = SmartMemoryDB(persist_directory=temp_dir)
    yield db
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


# ===== Test 1: Fact Extraction Patterns =====

@pytest.mark.unit
@pytest.mark.memory
def test_fact_extraction_patterns(temp_db):
    """Regex patterns correctly extract user info from conversation."""
    db = temp_db

    # Test name extraction
    results = db.extract_facts_from_conversation(
        "my name is John",
        "Nice to meet you, John!"
    )
    assert len(results) >= 1
    name_facts = [f for f, c in results if "John" in f]
    assert len(name_facts) >= 1

    # Test preference extraction
    results = db.extract_facts_from_conversation(
        "I really like pizza and programming",
        "That's great!"
    )
    pref_facts = [f for f, c in results if "like" in f.lower()]
    assert len(pref_facts) >= 1

    # Test dislike extraction
    results = db.extract_facts_from_conversation(
        "I hate cold weather",
        "I understand."
    )
    dislike_facts = [f for f, c in results if "dislike" in f.lower()]
    assert len(dislike_facts) >= 1

    # Test work info extraction
    results = db.extract_facts_from_conversation(
        "I work as a software engineer",
        "That's interesting!"
    )
    work_facts = [f for f, c in results if "software" in f.lower() or "engineer" in f.lower()]
    assert len(work_facts) >= 1


# ===== Test 2: Duplicate Fact Detection =====

@pytest.mark.unit
@pytest.mark.memory
def test_duplicate_fact_detection(temp_db):
    """Similar facts are detected as duplicates."""
    db = temp_db

    # Add first fact
    db.add_fact("User's name is Alex", "name")

    # Check for duplicate (same fact)
    is_dup = db.is_duplicate_fact("User's name is Alex")
    assert is_dup is True

    # Check for semantic duplicate (slightly different wording)
    is_dup = db.is_duplicate_fact("The user's name is Alex")
    assert is_dup is True

    # Check for non-duplicate
    is_dup = db.is_duplicate_fact("User likes ice cream")
    assert is_dup is False


# ===== Test 3: Search Across Tiers =====

@pytest.mark.unit
@pytest.mark.memory
def test_search_across_tiers(temp_db):
    """search_all queries facts, conversations, and summaries."""
    db = temp_db

    # Add data to all tiers
    db.add_fact("User loves Python programming", "preference")
    db.add_conversation(
        "Tell me about databases",
        "Databases store structured data efficiently."
    )
    db.add_summary("Session covered programming topics")

    # Search should find results from multiple tiers
    results = db.search_all("programming", n_results=5)

    assert len(results) > 0
    assert "Python" in results or "programming" in results.lower()


# ===== Test 4: Startup Context Generation =====

@pytest.mark.unit
@pytest.mark.memory
def test_startup_context_generation(temp_db):
    """get_startup_context builds formatted context output."""
    db = temp_db

    # Add some data
    db.add_fact("User's favorite color is blue", "preference")
    db.add_conversation("Hello!", "Hi there! How can I help?")
    db.add_summary("Previous session was casual conversation")

    # Get startup context
    context = db.get_startup_context(recent_conversations=5, include_summaries=3)

    # Should contain section headers
    assert "KNOWN FACTS" in context
    assert "RECENT CONVERSATIONS" in context or "PREVIOUS SESSION" in context

    # Should contain actual data
    assert "blue" in context.lower() or "favorite" in context.lower()


# ===== Additional Memory Tests =====

@pytest.mark.unit
@pytest.mark.memory
def test_add_and_retrieve_fact(temp_db):
    """Facts can be added and retrieved."""
    db = temp_db

    db.add_fact("Test fact content", "test_category")

    all_facts = db.get_all_facts()
    assert "Test fact content" in all_facts
    assert "TEST_CATEGORY" in all_facts  # Category should be uppercase


@pytest.mark.unit
@pytest.mark.memory
def test_add_and_search_conversation(temp_db):
    """Conversations can be added and searched."""
    db = temp_db

    db.add_conversation(
        "What's the weather like?",
        "It's sunny and 75 degrees today."
    )

    results = db.search_conversations("weather", n_results=1)

    assert "weather" in results.lower()
    assert "sunny" in results.lower()


@pytest.mark.unit
@pytest.mark.memory
def test_get_recent_conversations(temp_db):
    """Recent conversations are retrieved in order."""
    db = temp_db

    # Add multiple conversations
    for i in range(5):
        db.add_conversation(f"Message {i}", f"Response {i}")

    recent = db.get_recent_conversations(n=3)

    assert len(recent) == 3
    # Should contain the most recent ones
    assert any("4" in r for r in recent)  # Last message


@pytest.mark.unit
@pytest.mark.memory
def test_get_stats(temp_db):
    """get_stats returns accurate counts."""
    db = temp_db

    # Add some data
    db.add_fact("Fact 1", "general")
    db.add_fact("Fact 2", "general")
    db.add_conversation("User msg", "AI msg")
    db.add_summary("Summary 1")

    stats = db.get_stats()

    assert stats["facts"] == 2
    assert stats["conversations"] == 1
    assert stats["summaries"] == 1
    assert stats["total"] == 4


@pytest.mark.unit
@pytest.mark.memory
def test_clear_all(temp_db):
    """clear_all removes all data from all collections."""
    db = temp_db

    # Add data
    db.add_fact("Test fact", "general")
    db.add_conversation("Hello", "Hi")
    db.add_summary("Test summary")

    # Verify data exists
    stats = db.get_stats()
    assert stats["total"] > 0

    # Clear all
    db.clear_all()

    # Verify empty
    stats = db.get_stats()
    assert stats["facts"] == 0
    assert stats["conversations"] == 0
    assert stats["summaries"] == 0


@pytest.mark.unit
@pytest.mark.memory
def test_empty_input_handling(temp_db):
    """Empty inputs are handled gracefully."""
    db = temp_db

    # These should not raise errors
    db.add_fact("", "general")  # Empty fact
    db.add_fact("   ", "general")  # Whitespace only
    db.add_conversation("", "")  # Empty conversation

    # Should have no data
    stats = db.get_stats()
    assert stats["total"] == 0


@pytest.mark.unit
@pytest.mark.memory
def test_auto_extract_facts_on_conversation(temp_db):
    """Facts are automatically extracted when adding conversations."""
    db = temp_db

    # Add conversation with extractable fact
    db.add_conversation(
        "My name is Sarah and I love music",
        "Nice to meet you, Sarah!"
    )

    # Should have extracted facts
    all_facts = db.get_all_facts()
    assert "Sarah" in all_facts or "name" in all_facts.lower()
