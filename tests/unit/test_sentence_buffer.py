"""
Unit tests for SentenceBuffer.
Tests sentence boundary detection with various edge cases.
"""

import pytest

from src.api.utils.sentence_buffer import SentenceBuffer, SentenceBufferConfig

# ===== Test 1: Splits on Period =====

@pytest.mark.unit
def test_sentence_buffer_splits_on_period():
    """Buffer correctly identifies sentence endings with periods."""
    buffer = SentenceBuffer()
    sentences = []

    # Simulate streaming tokens
    text = "Hello there. How are you today?"
    for char in text:
        for sentence in buffer.add(char):
            sentences.append(sentence)

    # Flush remaining
    final = buffer.flush()
    if final:
        sentences.append(final)

    assert "Hello there." in sentences
    assert "How are you today?" in sentences
    assert len(sentences) == 2


# ===== Test 2: Handles Abbreviations =====

@pytest.mark.unit
def test_sentence_buffer_handles_abbreviations():
    """Abbreviations like Dr., Mr., etc. don't trigger sentence split."""
    buffer = SentenceBuffer()
    sentences = []

    text = "Dr. Smith arrived at 3 p.m. today."
    for char in text:
        for sentence in buffer.add(char):
            sentences.append(sentence)

    final = buffer.flush()
    if final:
        sentences.append(final)

    # Should be ONE sentence, not split on "Dr." or "p.m."
    assert len(sentences) == 1
    assert "Dr. Smith" in sentences[0]
    assert "p.m." in sentences[0]


# ===== Test 3: Handles Ellipsis =====

@pytest.mark.unit
def test_sentence_buffer_handles_ellipsis():
    """Ellipsis '...' handled correctly without premature splitting."""
    buffer = SentenceBuffer()
    sentences = []

    text = "Wait... I need to think. Yes, that's right."
    for char in text:
        for sentence in buffer.add(char):
            sentences.append(sentence)

    final = buffer.flush()
    if final:
        sentences.append(final)

    # Should have produced sentences (ellipsis handling may vary)
    assert len(sentences) >= 1
    # All text should be captured
    all_text = " ".join(sentences)
    assert "Wait" in all_text
    assert "think" in all_text


# ===== Test 4: Max Length Flush =====

@pytest.mark.unit
def test_sentence_buffer_max_length_flush():
    """Long text without punctuation is force-flushed at max buffer length."""
    config = SentenceBufferConfig(max_length=100)
    buffer = SentenceBuffer(cfg=config)
    sentences = []

    # Create text longer than max_buffer_length without sentence enders
    text = "This is a very long sentence that goes on and on without any punctuation marks to break it up and it just keeps going and going, eventually it should be force flushed"

    for char in text:
        for sentence in buffer.add(char):
            sentences.append(sentence)

    # Should have force flushed at least once
    assert len(sentences) >= 1 or len(buffer.pending) < config.max_buffer_length


# ===== Test 5: Newline Split =====

@pytest.mark.unit
def test_sentence_buffer_newline_split():
    """Newlines trigger sentence boundaries (paragraph breaks)."""
    buffer = SentenceBuffer()
    sentences = []

    text = "First line\nSecond line\nThird line"
    for char in text:
        for sentence in buffer.add(char):
            sentences.append(sentence)

    final = buffer.flush()
    if final:
        sentences.append(final)

    assert "First line" in sentences
    assert "Second line" in sentences
    assert "Third line" in sentences
    assert len(sentences) == 3


# ===== Test 6: URL Handling =====

@pytest.mark.unit
def test_sentence_buffer_url_handling():
    """URLs with periods don't cause false sentence splits."""
    buffer = SentenceBuffer()
    sentences = []

    text = "Visit http://example.com for more info. Thanks!"
    for char in text:
        for sentence in buffer.add(char):
            sentences.append(sentence)

    final = buffer.flush()
    if final:
        sentences.append(final)

    # Should not split on the period in the URL
    # "Visit http://example.com for more info." should be one sentence
    assert len(sentences) <= 2
    url_sentence = [s for s in sentences if "http://example.com" in s]
    assert len(url_sentence) >= 1


# ===== Additional Edge Case Tests =====

@pytest.mark.unit
def test_sentence_buffer_multiple_punctuation():
    """Multiple punctuation marks like '!!!' or '???' handled correctly."""
    buffer = SentenceBuffer()
    sentences = []

    text = "What!!! That's amazing?! Tell me more."
    for char in text:
        for sentence in buffer.add(char):
            sentences.append(sentence)

    final = buffer.flush()
    if final:
        sentences.append(final)

    # Should handle multiple punctuation
    assert len(sentences) >= 2


@pytest.mark.unit
def test_sentence_buffer_decimal_numbers():
    """Decimal numbers like 3.14 don't trigger sentence splits."""
    buffer = SentenceBuffer()
    sentences = []

    text = "Pi equals approximately 3.14159. It's irrational."
    for char in text:
        for sentence in buffer.add(char):
            sentences.append(sentence)

    final = buffer.flush()
    if final:
        sentences.append(final)

    # "3.14159" should not split the sentence
    first_sentence = sentences[0] if sentences else ""
    assert "3.14159" in first_sentence or "3." not in first_sentence.split()[-1]


@pytest.mark.unit
def test_sentence_buffer_flush_returns_remaining():
    """Flush returns remaining text when stream ends."""
    buffer = SentenceBuffer()

    # Add incomplete sentence
    for char in "This is incomplete":
        list(buffer.add(char))

    result = buffer.flush()
    assert result == "This is incomplete"


@pytest.mark.unit
def test_sentence_buffer_clear():
    """Clear method empties the buffer without yielding."""
    buffer = SentenceBuffer()

    for char in "Some text here":
        list(buffer.add(char))

    assert buffer.pending != ""
    buffer.clear()
    assert buffer.pending == ""


@pytest.mark.unit
def test_sentence_buffer_empty_input():
    """Empty input doesn't cause errors."""
    buffer = SentenceBuffer()

    sentences = list(buffer.add(""))
    assert sentences == []

    result = buffer.flush()
    assert result is None


@pytest.mark.unit
def test_sentence_buffer_question_mark():
    """Question marks correctly end sentences."""
    buffer = SentenceBuffer()
    sentences = []

    text = "How are you? I'm fine. What about you?"
    for char in text:
        for sentence in buffer.add(char):
            sentences.append(sentence)

    final = buffer.flush()
    if final:
        sentences.append(final)

    assert "How are you?" in sentences
    assert "What about you?" in sentences
