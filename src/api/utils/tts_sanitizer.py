"""TTS text sanitizer - strips formatting/control text before speaking."""

import re

# Pre-compiled patterns for text that should not be spoken
_MEMORY_TAG = re.compile(r"\[MEMORY\].*?(?:\n|$)", re.IGNORECASE)
_MARKDOWN_HEADERS = re.compile(r"#{1,6}\s*")
_ASTERISK_ACTIONS = re.compile(r"\*[^*]+\*")
_EMOJI_COLONS = re.compile(r":[a-z_]+:")
_BRACKET_TAGS = re.compile(r"\[[A-Z_]+\]")
_EXTRA_WHITESPACE = re.compile(r"\s{2,}")


def sanitize_for_tts(text: str) -> str:
    """Strip formatting and control text before TTS synthesis.

    Removes: [MEMORY] tags, ### headers, *asterisk actions*, :emoji: codes,
    [BRACKET_TAGS], and normalizes whitespace.
    """
    if not text:
        return ""

    text = _MEMORY_TAG.sub("", text)
    text = _MARKDOWN_HEADERS.sub("", text)
    text = _ASTERISK_ACTIONS.sub("", text)
    text = _EMOJI_COLONS.sub("", text)
    text = _BRACKET_TAGS.sub("", text)
    text = _EXTRA_WHITESPACE.sub(" ", text)
    return text.strip()
