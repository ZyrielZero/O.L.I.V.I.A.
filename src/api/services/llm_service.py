"""LLM service wrapper."""

import logging
from typing import AsyncGenerator, Optional

from src.api.utils.exceptions import LLMError, ServiceUnavailableError
from src.core.llm.ollama_client import ConversationManager, check_ollama_connection_async

log = logging.getLogger("api.llm")


class LLMService:
    """Async wrapper for Ollama conversation manager."""

    def __init__(self, model: str, system_prompt: str, host: str = "http://localhost:11434"):
        self.model = model
        self.system_prompt = system_prompt
        self.host = host
        self._manager: Optional[ConversationManager] = None

    async def initialize(self):
        """Init Ollama connection."""
        if not await check_ollama_connection_async(self.host):
            raise ServiceUnavailableError(f"Ollama not running at {self.host}")

        self._manager = ConversationManager(
            model=self.model, system_prompt=self.system_prompt, host=self.host
        )
        log.info(f"LLM ready: {self.model}")

    async def chat_stream(
        self,
        message: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM tokens (native async via httpx)."""
        if not self._manager:
            raise LLMError("LLM not initialized")

        try:
            async for tok in self._manager.chat_stream_async(
                user_input=message, context=context, temperature=temperature, max_tokens=max_tokens
            ):
                yield tok
        except Exception as e:
            log.error(f"LLM error: {e}")
            raise LLMError(str(e))

    async def clear_history(self):
        if self._manager:
            self._manager.clear_history()

    async def update_system_prompt(self, prompt: str):
        if self._manager:
            self._manager.update_system_prompt(prompt)
            self.system_prompt = prompt

    async def health_check(self) -> bool:
        try:
            return await check_ollama_connection_async(self.host)
        except Exception:
            return False

    def is_initialized(self) -> bool:
        return self._manager is not None

    async def cleanup(self):
        if self._manager:
            await self._manager.close()
            log.info("LLM cleaned up")
