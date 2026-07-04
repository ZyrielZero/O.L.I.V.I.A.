"""Sentence boundary detection for streaming TTS."""

import logging
import re
from dataclasses import dataclass, field
from typing import Generator, Optional

log = logging.getLogger("api.sentence_buffer")

ABBREVIATIONS = (
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Jr.",
    "Sr.",
    "Inc.",
    "Ltd.",
    "Corp.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "U.S.",
    "U.K.",
    "a.m.",
    "p.m.",
    "Ph.D.",
    "M.D.",
    "B.A.",
    "St.",
    "Ave.",
    "Blvd.",
    "Rd.",
    "No.",
    "Vol.",
    "Fig.",
)

# Optimization: frozenset for O(1) character lookup instead of O(n) string iteration.
# Used in hot path for checking sentence terminators.
_SENTENCE_TERMINATORS = frozenset(".!?\n")

# Optimization: Pre-compiled regex at module level for split point detection.
# Single pattern matches all split characters, avoiding multiple rfind() calls.
# Captures the split character to preserve it during splitting.
_SPLIT_POINT_PATTERN = re.compile(r"(,|;|:| - | -- )")

# Optimization: Pre-compiled regex for decimal detection at module level.
_DECIMAL_END_PATTERN = re.compile(r"\d\.$")


@dataclass
class SentenceBufferConfig:
    """Sentence buffer config."""

    abbreviations: tuple = field(default_factory=lambda: ABBREVIATIONS)
    min_length: int = 1  # allow single-word responses
    max_length: int = 500  # force flush for run-on text


class SentenceBuffer:
    """Buffers tokens, yields complete sentences.

    Handles abbreviations, ellipsis, URLs, decimals.
    """

    def __init__(self, cfg: Optional[SentenceBufferConfig] = None):
        self.cfg = cfg or SentenceBufferConfig()
        self._buf = ""
        self._has_terminator = False
        self._end_pat = re.compile(r"(?<![A-Z])(?<!\d)([.!?]+)(?=\s|$)")
        self._url_pat = re.compile(r"https?://\S+")
        self._abbr_pat = self._build_abbr_pat()

    def _build_abbr_pat(self) -> re.Pattern:
        escaped = [re.escape(a) for a in self.cfg.abbreviations]
        return re.compile(r"(" + "|".join(escaped) + r")$", re.IGNORECASE)

    def add(self, tok: str) -> Generator[str, None, None]:
        """Add token, yield complete sentences."""
        self._buf += tok

        # Only scan the new token for terminators (O(len(tok)) vs O(len(buf)))
        if not self._has_terminator:
            if any(c in _SENTENCE_TERMINATORS for c in tok):
                self._has_terminator = True
            else:
                if len(self._buf) >= self.cfg.max_length:
                    yield from self._force_flush()
                return

        if "\n" in tok or "\n" in self._buf:
            yield from self._split_newlines()
            return

        yield from self._extract_sentences()

    def _split_newlines(self) -> Generator[str, None, None]:
        parts = self._buf.split("\n")
        for p in parts[:-1]:
            p = p.strip()
            if p:
                yield p
        self._buf = parts[-1]
        self._has_terminator = any(c in _SENTENCE_TERMINATORS for c in self._buf)

    def _extract_sentences(self) -> Generator[str, None, None]:
        while True:
            m = self._end_pat.search(self._buf)
            if not m:
                break

            end = m.end()
            cand = self._buf[:end].strip()

            if self._is_false_pos(cand):
                break

            if len(cand) < self.cfg.min_length:
                break

            yield cand
            self._buf = self._buf[end:].lstrip()

        self._has_terminator = any(c in _SENTENCE_TERMINATORS for c in self._buf)

    def _is_false_pos(self, text: str) -> bool:
        """Check for abbreviation, URL, decimal false positives."""
        if self._abbr_pat.search(text):
            return True

        if self._url_pat.search(text):
            for url in self._url_pat.findall(text):
                if text.endswith(url):
                    return True

        # Optimization: Use pre-compiled module-level pattern instead of re.search()
        # which compiles the pattern on every call.
        if _DECIMAL_END_PATTERN.search(text):
            return True

        if text.endswith(".") and not text.endswith("..."):
            if self._buf.startswith("."):
                return True

        return False

    def _force_flush(self) -> Generator[str, None, None]:
        """Force flush on long buffer - find best split."""
        # Optimization: Single regex finditer() instead of multiple rfind() calls.
        # Complexity: O(n) single pass vs O(n*k) for k split characters.
        # Find all split points in one pass, then select the best one.
        buf_len = len(self._buf)
        half_len = buf_len // 2
        best = -1
        best_end = -1

        for match in _SPLIT_POINT_PATTERN.finditer(self._buf):
            pos = match.start()
            if pos > half_len:
                end_pos = match.end()
                if pos > best:
                    best = pos
                    best_end = end_pos

        if best_end > 0:
            sent = self._buf[:best_end].strip()
            self._buf = self._buf[best_end:].lstrip()
            if sent:
                yield sent
        else:
            yield self._buf.strip()
            self._buf = ""

    def flush(self) -> Optional[str]:
        """Flush remaining buffer."""
        if self._buf.strip():
            r = self._buf.strip()
            self._buf = ""
            self._has_terminator = False
            return r
        return None

    def clear(self) -> None:
        """Discard the buffer and reset terminator state."""
        self._buf = ""
        self._has_terminator = False

    @property
    def pending(self) -> str:
        """Unflushed buffer contents."""
        return self._buf
