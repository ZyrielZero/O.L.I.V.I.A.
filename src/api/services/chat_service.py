"""Chat service - extracted business logic for chat endpoint.

Provides a testable, reusable service class for chat operations.
Can be used by REST endpoints, WebSocket handlers, CLI, etc.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, AsyncGenerator, Optional

from src.api.constants import (
    DEEP_SEARCH_WORDS,
    GREETING_PATTERNS,
    QUICK_SEARCH_WORDS,
    SEARCH_PATTERNS_COMPILED,
    SSE_DONE,
    SSE_ERROR_TEMPLATE,
    SSE_TOKEN_TEMPLATE,
)

if TYPE_CHECKING:
    from src.api.services.llm_service import LLMService
    from src.api.services.memory_service import MemoryService
    from src.api.services.tts_service import TTSService

log = logging.getLogger("api.chat_service")


@dataclass
class SearchResult:
    """Result of search query detection."""

    query: Optional[str]
    mode: str  # "", "quick", "standard", "deep"


@dataclass
class ChatContext:
    """Assembled context for LLM."""

    search_context: str = ""
    memory_context: str = ""
    provided_context: str = ""

    @property
    def combined(self) -> str:
        """Combine all context sources."""
        parts = [p for p in [self.provided_context, self.search_context, self.memory_context] if p]
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        """Check if context is empty."""
        return not self.combined


class ChatService:
    """Chat service handling message processing, search, and context assembly.

    Extracted from chat route to enable:
    - Unit testing without HTTP layer
    - Reuse in WebSocket endpoints
    - Clearer separation of concerns
    """

    def __init__(
        self,
        llm: "LLMService",
        memory: "MemoryService",
        tts: Optional["TTSService"] = None,
    ):
        self.llm = llm
        self.memory = memory
        self.tts = tts

    @staticmethod
    @lru_cache(maxsize=64)
    def _is_simple_greeting(text_lower: str) -> bool:
        """Check if text is a simple greeting (cached)."""
        return text_lower.strip().rstrip("!?.") in GREETING_PATTERNS

    def detect_search(self, message: str) -> SearchResult:
        """Detect if message is a search query.

        Returns SearchResult with query and mode ("", "quick", "standard", "deep").
        """
        txt = message.lower()

        # Skip greetings
        if self._is_simple_greeting(txt):
            return SearchResult(query=None, mode="")

        # Check search patterns
        for pat in SEARCH_PATTERNS_COMPILED:
            m = pat.search(txt)
            if m:
                query = m.group(1).strip()
                txt_words = set(txt.split())

                # Handle multi-word phrase separately, then single-word set
                if "in depth" in txt or txt_words & DEEP_SEARCH_WORDS:
                    return SearchResult(query=query, mode="deep")
                if txt_words & QUICK_SEARCH_WORDS:
                    return SearchResult(query=query, mode="quick")
                return SearchResult(query=query, mode="standard")

        return SearchResult(query=None, mode="")

    async def fetch_memory_context(self, message: str, n_results: int = 3) -> str:
        """Fetch relevant memory context for message."""
        # Skip for short messages (likely greetings)
        if len(message.split()) <= 6:
            return ""

        try:
            return await self.memory.get_relevant_context(message, n_results=n_results)
        except Exception as e:
            log.warning(f"Memory prefetch failed: {e}")
            return ""

    async def execute_search(self, query: str, mode: str, max_results: int = 5) -> str:
        """Execute web search with specified mode."""
        from src.core.tools.web_search import (
            web_search,
            web_search_deep,
            web_search_quick,
        )

        try:
            if mode == "deep":
                return web_search_deep(query)
            elif mode == "quick":
                return web_search_quick(query)
            else:
                return web_search(query, max_results=max_results)
        except Exception as e:
            log.error(f"Search failed: {e}")
            return f"[Search failed: {e}]"

    async def build_context(
        self,
        message: str,
        provided_context: Optional[str] = None,
    ) -> ChatContext:
        """Build complete context for LLM including search and memory.

        Runs search and memory fetch in parallel for efficiency.
        """
        # OPT: Check word count first to skip search entirely for short messages
        word_count = len(message.split())
        if word_count < 5:
            search_result = SearchResult(query=None, mode="")
        else:
            search_result = self.detect_search(message)

        # Parallel fetch: memory task
        memory_task = asyncio.create_task(self.fetch_memory_context(message))

        # Execute search if needed
        search_ctx = ""
        if search_result.query:
            log.info(f"Search [{search_result.mode}]: {search_result.query}")
            search_ctx = await self.execute_search(search_result.query, search_result.mode)

        # Await memory task
        try:
            memory_ctx = await memory_task
        except Exception as e:
            log.warning(f"Memory fetch failed: {e}")
            memory_ctx = ""

        return ChatContext(
            search_context=search_ctx,
            memory_context=memory_ctx,
            provided_context=provided_context or "",
        )

    async def generate_stream_sse(
        self,
        message: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate LLM response stream with SSE formatting.

        Yields SSE-formatted JSON strings ready for StreamingResponse.
        """
        resp_chunks: list[str] = []

        try:
            async for tok in self.llm.chat_stream(
                message=message,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                resp_chunks.append(tok)
                yield f"data: {SSE_TOKEN_TEMPLATE.format(json.dumps(tok))}\n\n"

            full_response = "".join(resp_chunks)

            # Store conversation
            try:
                await self.memory.add_conversation(message, full_response, auto_extract=True)
            except Exception as e:
                log.warning(f"Failed to store conversation: {e}")

            yield f"data: {SSE_DONE}\n\n"

        except Exception as e:
            log.error(f"Stream error: {e}")
            yield f"data: {SSE_ERROR_TEMPLATE.format(json.dumps(str(e)))}\n\n"

    async def generate_response(
        self,
        message: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate complete LLM response (non-streaming)."""
        resp_chunks: list[str] = []

        async for tok in self.llm.chat_stream(
            message=message,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            resp_chunks.append(tok)

        full_response = "".join(resp_chunks)

        # Store conversation
        try:
            await self.memory.add_conversation(message, full_response, auto_extract=True)
        except Exception as e:
            log.warning(f"Failed to store conversation: {e}")

        return full_response

    async def speak_response(self, text: str) -> None:
        """Speak response via TTS if available."""
        if not self.tts or not self.tts.is_initialized():
            return

        if not text or not text.strip():
            return

        try:
            await self.tts.speak(text)
        except Exception as e:
            log.warning(f"TTS failed: {e}")
