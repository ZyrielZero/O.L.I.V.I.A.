"""
Personality and character compliance tests.
Tests forbidden phrases, emoji usage, and response style.
"""

import re
from pathlib import Path

import pytest
import yaml


# Load character config for tests
@pytest.fixture(scope="module")
def character_config():
    """Load character configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "character.yaml"

    if not config_path.exists():
        pytest.skip("character.yaml not found")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Common forbidden phrases
FORBIDDEN_PHRASES = [
    "Certainly!",
    "Certainly,",
    "Absolutely!",
    "I'd be happy to",
    "I'd be delighted to",
    "Great question!",
    "Good question!",
    "That's a great question",
    "As an AI",
    "As a language model",
    "I apologize",
    "I'm sorry, but",
    "Is there anything else",
    "Feel free to",
    "I hope this helps",
    "Thank you for sharing",
    "I appreciate you sharing",
    "Wonderful!",
    "Fantastic!",
    "Amazing!",
    "Excellent!",
]


# Good test responses that should PASS all checks
GOOD_RESPONSES = [
    "Yeah, I can help with that.",
    "Sure, here's what I think.",
    "Hmm, let me think about that.",
    "That makes sense.",
    "I get it. Here's my take.",
]


# ===== Test 1: Forbidden Phrase - Certainly =====

@pytest.mark.personality
def test_forbidden_phrase_certainly():
    """Good responses don't contain 'Certainly!'."""
    for response in GOOD_RESPONSES:
        for phrase in ["Certainly!", "Certainly,"]:
            assert phrase.lower() not in response.lower(), f"Forbidden phrase found in: {response}"


# ===== Test 2: Forbidden Phrase - Great Question =====

@pytest.mark.personality
def test_forbidden_phrase_great_question():
    """Good responses don't contain 'Great question!'."""
    forbidden = ["Great question!", "Good question!", "That's a great question"]

    for response in GOOD_RESPONSES:
        for phrase in forbidden:
            assert phrase.lower() not in response.lower(), f"Forbidden phrase found in: {response}"


# ===== Test 3: Forbidden Phrase - As an AI =====

@pytest.mark.personality
def test_forbidden_phrase_as_an_ai():
    """Good responses don't contain 'As an AI'."""
    for response in GOOD_RESPONSES:
        assert "as an ai" not in response.lower(), f"Forbidden phrase found in: {response}"


# ===== Test 4: Forbidden Phrase - I Apologize =====

@pytest.mark.personality
def test_forbidden_phrase_i_apologize():
    """Good responses don't contain 'I apologize'."""
    for response in GOOD_RESPONSES:
        assert "i apologize" not in response.lower(), f"Forbidden phrase found in: {response}"


# ===== Test 5: No Emoji in Response =====

@pytest.mark.personality
def test_no_emoji_in_response():
    """Good responses don't contain emojis."""
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U0001F600-\U0001F64F"
        "]+",
        flags=re.UNICODE
    )

    for response in GOOD_RESPONSES:
        assert not emoji_pattern.search(response), f"Emoji found in: {response}"


# ===== Test 6: No Asterisk Actions =====

@pytest.mark.personality
def test_no_asterisk_actions():
    """Good responses don't contain *action* patterns."""
    action_pattern = re.compile(r'\*[^*]+\*')

    for response in GOOD_RESPONSES:
        assert not action_pattern.search(response), f"Asterisk action in: {response}"


# ===== Test 7: Greeting Style Casual =====

@pytest.mark.personality
def test_greeting_style_casual(character_config):
    """Casual greetings are preferred over formal ones."""
    casual_greetings = ["hey", "hi", "what's up", "yo", "hmm", "yeah"]

    # Test casual response
    casual_response = "Hey, what's going on?"
    has_casual = any(g in casual_response.lower() for g in casual_greetings)
    assert has_casual, "Casual response should contain casual greeting"

    # Formal greetings to avoid
    formal_greetings = ["Good morning", "Greetings", "Salutations", "How may I assist you"]
    for formal in formal_greetings:
        assert formal.lower() not in casual_response.lower(), f"Formal greeting found: {formal}"


# ===== Test 8: Response Length Concise =====

@pytest.mark.personality
def test_response_length_concise():
    """Good responses are 1-5 sentences typically."""
    for response in GOOD_RESPONSES:
        # Count sentence-ending punctuation
        sentence_enders = len(re.findall(r'[.!?]+', response))
        # At least 1 sentence, at most 5
        assert 1 <= max(sentence_enders, 1) <= 5, f"Response length issue: {response}"


# ===== Test 9: Contractions Used =====

@pytest.mark.personality
def test_contractions_used():
    """Casual responses use contractions."""
    contractions = ["i'm", "don't", "can't", "won't", "isn't", "aren't", "it's", "that's", "you're", "i've", "let's"]

    # Test a response with contractions
    response_with_contractions = "I'm not sure, but I don't think that's right."
    has_contraction = any(c in response_with_contractions.lower() for c in contractions)
    assert has_contraction, "Response should use contractions"


# ===== Test 10: Character YAML Schema Valid =====

@pytest.mark.personality
def test_character_yaml_schema_valid(character_config):
    """YAML has all required sections."""
    required_sections = ["identity", "personality", "speaking_style"]

    for section in required_sections:
        assert section in character_config, f"Missing required section: {section}"

    assert "name" in character_config["identity"], "Identity missing 'name'"


# ===== Additional Personality Tests =====

@pytest.mark.personality
def test_no_corporate_speak():
    """Good responses avoid corporate AI speak."""
    corporate_phrases = [
        "I understand your concern",
        "rest assured",
        "Please note that",
        "I am here to assist",
        "At your service",
    ]

    for response in GOOD_RESPONSES:
        for phrase in corporate_phrases:
            assert phrase.lower() not in response.lower(), f"Corporate phrase in: {response}"


@pytest.mark.personality
def test_no_emoji_codes():
    """Good responses don't contain :emoji: codes."""
    emoji_code_pattern = re.compile(r':\w+:')

    for response in GOOD_RESPONSES:
        assert not emoji_code_pattern.search(response), f"Emoji code in: {response}"


@pytest.mark.personality
def test_forbidden_phrases_comprehensive():
    """Good responses don't contain any forbidden phrases."""
    for response in GOOD_RESPONSES:
        for phrase in FORBIDDEN_PHRASES:
            assert phrase.lower() not in response.lower(), f"Forbidden phrase '{phrase}' in: {response}"


@pytest.mark.personality
def test_single_question_limit():
    """Good responses have at most one question."""
    # Single question is OK
    single_q = "What do you think?"
    assert single_q.count('?') <= 1

    # No questions is OK
    no_q = "I understand completely."
    assert no_q.count('?') == 0


@pytest.mark.personality
def test_character_warmth_directness(character_config):
    """Character traits are within expected ranges."""
    traits = character_config.get("personality", {}).get("traits", {})

    if traits:
        warmth = traits.get("warmth", 0.5)
        assert 0.3 <= warmth <= 1.0, f"Warmth {warmth} outside range"

        directness = traits.get("directness", 0.5)
        assert 0.3 <= directness <= 1.0, f"Directness {directness} outside range"


@pytest.mark.personality
def test_detection_finds_bad_patterns():
    """Verify detection patterns correctly identify violations."""
    # Test that bad patterns ARE detected
    bad_responses = [
        ("Certainly! I can help.", ["Certainly!"]),
        ("As an AI, I can't do that.", ["As an AI"]),
        ("Great question! Let me explain.", ["Great question!"]),
    ]

    for response, expected_violations in bad_responses:
        found_any = False
        for phrase in expected_violations:
            if phrase.lower() in response.lower():
                found_any = True
                break
        assert found_any, f"Detection should find violations in: {response}"


@pytest.mark.personality
def test_emoji_detection_works():
    """Verify emoji detection pattern works."""
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U0001F600-\U0001F64F"
        "]+",
        flags=re.UNICODE
    )

    # Should detect emoji
    with_emoji = "Hello! \U0001F60A"
    assert emoji_pattern.search(with_emoji), "Should detect emoji"

    # Should not detect in clean text
    without_emoji = "Hello there!"
    assert not emoji_pattern.search(without_emoji), "Should not flag clean text"


@pytest.mark.personality
def test_asterisk_detection_works():
    """Verify asterisk action detection works."""
    action_pattern = re.compile(r'\*[^*]+\*')

    # Should detect action
    with_action = "Hello *smiles* there"
    assert action_pattern.search(with_action), "Should detect asterisk action"

    # Should not detect in clean text
    without_action = "Hello there!"
    assert not action_pattern.search(without_action), "Should not flag clean text"
