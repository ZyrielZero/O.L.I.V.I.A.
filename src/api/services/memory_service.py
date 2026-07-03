"""Memory service wrapper."""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from src.api.utils.exceptions import MemoryServiceError
from src.core.memory.smart_memory import SmartMemoryDB

log = logging.getLogger("api.memory")

MEM_TIMEOUT = 10.0
_INIT_TIMEOUT = 30.0
_HEALTH_TIMEOUT = 2.0


class MemoryService:
    """Async wrapper for ChromaDB memory."""

    def __init__(self, persist_directory: str = "memory_db"):
        self.persist_directory = persist_directory
        self._db: Optional[SmartMemoryDB] = None

    async def _run_timeout(
        self, fn: Callable[[], Any], op: str, timeout: float = MEM_TIMEOUT
    ) -> Any:
        """Run blocking fn in executor with timeout."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=timeout)
        except asyncio.TimeoutError:
            log.error(f"Memory timeout: {op}")
            raise MemoryServiceError(f"Timeout: {op}")

    async def initialize(self):
        """Init ChromaDB."""

        def _load():
            try:
                return SmartMemoryDB(persist_directory=self.persist_directory)
            except Exception as e:
                log.error(f"Memory init failed: {e}")
                raise MemoryServiceError(f"Init failed: {e}")

        try:
            self._db = await self._run_timeout(_load, "init", timeout=_INIT_TIMEOUT)
            log.info(f"Memory ready: {self.persist_directory}")
        except Exception as e:
            raise MemoryServiceError(f"Memory init failed: {e}")

    async def add_conversation(
        self, user_msg: str, ai_msg: str, auto_extract: bool = True
    ) -> Optional[List[str]]:
        """Store conversation, optionally extract facts."""
        if not self._db:
            raise MemoryServiceError("Not initialized")

        def _add():
            try:
                return self._db.add_conversation(user_msg, ai_msg, auto_extract)
            except Exception as e:
                log.error(f"Add conversation failed: {e}")
                raise MemoryServiceError(f"Store failed: {e}")

        return await self._run_timeout(_add, "add_conversation")

    async def get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """Get relevant memory context."""
        if not self._db:
            raise MemoryServiceError("Not initialized")

        db = self._db

        def _get():
            try:
                return db.get_relevant_context(query, n_results)
            except Exception as e:
                log.error(f"Get context failed: {e}")
                raise MemoryServiceError(f"Context retrieval failed: {e}")

        return await self._run_timeout(_get, "get_relevant_context")

    async def browse_memory(
        self, mem_type: str = "facts", query: Optional[str] = None, n_results: int = 10
    ) -> List[Dict]:
        """List or search entries in one tier (id + document + metadata)."""
        if not self._db:
            raise MemoryServiceError("Not initialized")
        db = self._db
        return await self._run_timeout(
            lambda: db.browse_entries(mem_type, query, n_results), "browse_memory"
        )

    async def add_fact(self, fact: str, category: str = "general") -> Optional[str]:
        """Store a fact; returns the new entry id."""
        if not self._db:
            raise MemoryServiceError("Not initialized")
        db = self._db
        return await self._run_timeout(lambda: db.add_fact(fact, category), "add_fact")

    async def delete_entry(self, entry_id: str) -> bool:
        """Delete one entry by id; False if it doesn't exist."""
        if not self._db:
            raise MemoryServiceError("Not initialized")
        db = self._db
        return await self._run_timeout(lambda: db.delete_entry(entry_id), "delete_entry")

    async def db_size_bytes(self) -> int:
        """On-disk size of the memory database."""
        if not self._db:
            raise MemoryServiceError("Not initialized")
        db = self._db
        return await self._run_timeout(db.db_size_bytes, "db_size")

    async def prune_expired(self, conv_days: int = 30, summary_days: int = 365) -> Dict[str, int]:
        """Remove expired conversations and summaries by TTL."""
        if not self._db:
            raise MemoryServiceError("Not initialized")

        db = self._db

        def _prune():
            try:
                return db.prune_expired(conv_days=conv_days, summary_days=summary_days)
            except Exception as e:
                log.error(f"TTL pruning failed: {e}")
                raise MemoryServiceError(f"Pruning failed: {e}")

        # Full-collection scan — allow more headroom than regular queries
        return await self._run_timeout(_prune, "prune_expired", timeout=60.0)

    async def query_memory(
        self, query: str, n_results: int = 3, mem_type: str = "all"
    ) -> List[str]:
        """Query memory db; facts + conversations run in parallel for "all"."""
        if not self._db:
            raise MemoryServiceError("Not initialized")

        loop = asyncio.get_running_loop()
        db = self._db

        def _query_facts():
            try:
                r = db.facts.query(query_texts=[query], n_results=n_results)
                return r["documents"][0] if r and r.get("documents") else []
            except Exception as e:
                log.error(f"Facts query failed: {e}")
                return []

        def _query_conversations():
            try:
                r = db.conversations.query(query_texts=[query], n_results=n_results)
                return r["documents"][0] if r and r.get("documents") else []
            except Exception as e:
                log.error(f"Conversations query failed: {e}")
                return []

        try:
            if mem_type == "facts":
                return await loop.run_in_executor(None, _query_facts)
            elif mem_type == "conversations":
                return await loop.run_in_executor(None, _query_conversations)
            else:
                facts_task = loop.run_in_executor(None, _query_facts)
                convs_task = loop.run_in_executor(None, _query_conversations)
                facts, convs = await asyncio.gather(facts_task, convs_task)
                return facts + convs
        except Exception as e:
            log.error(f"Query failed: {e}")
            raise MemoryServiceError(f"Query failed: {e}")

    async def get_stats(self) -> Dict[str, int]:
        """Get memory stats."""
        if not self._db:
            raise MemoryServiceError("Not initialized")

        db = self._db

        def _stats():
            try:
                return db.get_stats()
            except Exception as e:
                raise MemoryServiceError(f"Stats failed: {e}")

        return await asyncio.get_running_loop().run_in_executor(None, _stats)

    async def health_check(self) -> bool:
        """Fast heartbeat check."""
        try:
            if not self._db or not self._db.client:
                return False
            client = self._db.client
            return await self._run_timeout(
                lambda: client.heartbeat() > 0, "heartbeat", timeout=_HEALTH_TIMEOUT
            )
        except Exception:
            return False

    def is_initialized(self) -> bool:
        return self._db is not None
