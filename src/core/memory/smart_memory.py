"""Smart Memory System (Option C)
Hybrid approach with Facts, Recent, and Relevant tiers.

Performance Optimizations:
- Pre-compiled regex patterns at module level (avoid re-compilation per call)
- Parallel tier searches using ThreadPoolExecutor
- Batch duplicate checking for multiple facts
"""

import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions

# --- LOGGING INTEGRATION ---
from src.utils.logger import get_logger

# ---------------------------

# =============================================================================
# PRE-COMPILED REGEX PATTERNS (module-level, compiled once)
# O(n) compile per call -> O(1) reuse
# =============================================================================
_FACT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?:my name is|i'm called|call me) ([a-zA-Z]+)", re.IGNORECASE), "name"),
    (
        re.compile(
            r"(?:i (?:really )?(?:like|love|enjoy|prefer)) (.+?)(?:\.|$|,|!)", re.IGNORECASE
        ),
        "preference",
    ),
    (
        re.compile(
            r"(?:i (?:hate|dislike|don't like|can't stand)) (.+?)(?:\.|$|,|!)", re.IGNORECASE
        ),
        "dislike",
    ),
    (re.compile(r"(?:my favorite .+? is) (.+?)(?:\.|$|,|!)", re.IGNORECASE), "preference"),
    (re.compile(r"(?:i work (?:at|as|for)) (.+?)(?:\.|$|,|!)", re.IGNORECASE), "personal"),
    (re.compile(r"(?:i live in|i'm from) (.+?)(?:\.|$|,|!)", re.IGNORECASE), "personal"),
    (
        re.compile(r"(?:remember that|don't forget|keep in mind) (.+?)(?:\.|$|!)", re.IGNORECASE),
        "instruction",
    ),
]


class SmartMemoryDB:
    """Intelligent memory system with tiered storage."""

    # HNSW Index Parameters (validated via web research)
    # search_ef=100 improves recall vs default=10, with acceptable latency
    # Reference: https://cookbook.chromadb.dev/core/collections/
    HNSW_METADATA = {
        "hnsw:search_ef": 100,  # Default 10 is too low for quality recall
        "hnsw:num_threads": 4,  # Parallel search threads
    }

    def __init__(self, persist_directory: str = "data/memory_db"):
        self.log = get_logger("memory")  # Initialize logger
        self._persist_dir = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # TIER 1: Facts collection
        self.facts = self.client.get_or_create_collection(
            name="olivia_facts",
            embedding_function=self.embedding_fn,  # type: ignore
            metadata=self.HNSW_METADATA,
        )

        # TIER 2 & 3: Conversations collection
        self.conversations = self.client.get_or_create_collection(
            name="olivia_conversations",
            embedding_function=self.embedding_fn,  # type: ignore
            metadata=self.HNSW_METADATA,
        )

        # TIER 4: Summaries collection
        self.summaries = self.client.get_or_create_collection(
            name="olivia_summaries",
            embedding_function=self.embedding_fn,  # type: ignore
            metadata=self.HNSW_METADATA,
        )

        # Thread pool for parallel searches
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="memory_search")

        self.log.info(
            f"SmartMemoryDB loaded: "
            f"Facts={self.facts.count()}, "
            f"Convos={self.conversations.count()}, "
            f"Summaries={self.summaries.count()}"
        )

    @staticmethod
    def _gen_id(prefix: str) -> str:
        """Generate unique ID with timestamp + short UUID."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"{prefix}_{ts}_{short_uuid}"

    # =========================================================================
    # TIER 1: FACTS
    # =========================================================================

    def add_fact(self, fact: str, category: str = "general") -> None:
        """Store a key fact about the user."""
        if not fact.strip():
            return

        fact_id = self._gen_id("fact")

        self.facts.add(
            documents=[fact],
            metadatas=[{"category": category, "timestamp": datetime.now().isoformat()}],
            ids=[fact_id],
        )
        self.log.info(f"Stored fact: {fact[:50]}...")

    def get_all_facts(self) -> str:
        """Get all stored facts as formatted string."""
        if self.facts.count() == 0:
            return ""

        results = self.facts.get()

        if not results:
            return ""

        documents = results.get("documents")
        metadatas = results.get("metadatas")

        if not documents:
            return ""

        facts_by_category: Dict[str, List[str]] = {}

        for i in range(len(documents)):
            doc = documents[i]
            if doc is None:
                continue

            # Safely get category from metadata
            category = "general"
            if metadatas and i < len(metadatas):
                meta = metadatas[i]
                if meta and isinstance(meta, dict):
                    category = str(meta.get("category", "general"))

            if category not in facts_by_category:
                facts_by_category[category] = []
            facts_by_category[category].append(str(doc))

        # O(n) string concat via list + join instead of += in loop
        output: List[str] = []
        for category, facts_list in facts_by_category.items():
            output.append(f"[{category.upper()}]")
            for fact in facts_list:
                output.append(f"  - {fact}")

        return "\n".join(output)

    def extract_facts_from_conversation(self, user_msg: str, ai_msg: str) -> List[Tuple[str, str]]:
        """Extract potential facts from a conversation exchange.

        Uses pre-compiled regex patterns for O(1) pattern access.
        """
        extracted: List[Tuple[str, str]] = []
        user_msg_lower = user_msg.lower()

        # Use pre-compiled patterns (O(1) access vs O(n) compile per call)
        for pattern, category in _FACT_PATTERNS:
            matches = pattern.findall(user_msg_lower)
            for match in matches:
                if len(match) > 2 and len(match) < 100:
                    if category == "name":
                        fact = f"User's name is {match.strip().title()}"
                    elif category == "preference":
                        fact = f"User likes {match.strip()}"
                    elif category == "dislike":
                        fact = f"User dislikes {match.strip()}"
                    elif category == "personal":
                        fact = f"User {match.strip()}"
                    elif category == "instruction":
                        fact = f"User requested: {match.strip()}"
                    else:
                        fact = match.strip()

                    extracted.append((fact, category))

        return extracted

    def is_duplicate_fact(self, new_fact: str) -> bool:
        """Check if a similar fact already exists."""
        if self.facts.count() == 0:
            return False

        results = self.facts.query(query_texts=[new_fact], n_results=1)

        if results and results.get("distances"):
            distances = results["distances"]
            if distances and len(distances) > 0 and len(distances[0]) > 0:
                if distances[0][0] < 0.3:
                    return True

        return False

    def batch_check_duplicates(self, facts: List[str], threshold: float = 0.3) -> List[bool]:
        """Batch duplicate check for multiple facts.

        O(n) individual queries -> O(1) batch query
        Uses ChromaDB's batch query capability.
        """
        if not facts or self.facts.count() == 0:
            return [False] * len(facts)

        # Single batch query instead of n individual queries
        results = self.facts.query(query_texts=facts, n_results=1)

        duplicates = []
        if results and results.get("distances"):
            for i, fact_distances in enumerate(results["distances"]):
                if fact_distances and len(fact_distances) > 0:
                    duplicates.append(fact_distances[0] < threshold)
                else:
                    duplicates.append(False)
        else:
            duplicates = [False] * len(facts)

        return duplicates

    # =========================================================================
    # TIER 2: CONVERSATIONS
    # =========================================================================

    def add_conversation(self, user_msg: str, ai_msg: str, auto_extract_facts: bool = True) -> None:
        """Store a conversation exchange."""
        if not user_msg.strip() or not ai_msg.strip():
            return

        conversation = f"User: {user_msg}\nAssistant: {ai_msg}"
        conv_id = self._gen_id("conv")

        self.conversations.add(
            documents=[conversation],
            metadatas=[
                {
                    "timestamp": datetime.now().isoformat(),
                    "user_msg_length": len(user_msg),
                    "ai_msg_length": len(ai_msg),
                }
            ],
            ids=[conv_id],
        )

        if auto_extract_facts:
            extracted = self.extract_facts_from_conversation(user_msg, ai_msg)
            if extracted:
                # Batch duplicate check instead of per-fact queries
                # O(n) queries -> O(1) batch query
                facts_only = [fact for fact, _ in extracted]
                is_duplicate = self.batch_check_duplicates(facts_only)

                for i, (fact, category) in enumerate(extracted):
                    if not is_duplicate[i]:
                        self.add_fact(fact, category)

    def get_recent_conversations(self, n: int = 10) -> List[str]:
        """Get the N most recent conversations."""
        total = self.conversations.count()
        if total == 0:
            return []

        try:
            results = self.conversations.get()
            if results:
                docs = results.get("documents")
                metas = results.get("metadatas")
                if docs and metas:
                    # Sort by timestamp descending, return most recent n
                    paired = list(zip(docs, metas))
                    paired.sort(
                        key=lambda x: x[1].get("timestamp", "") if x[1] else "",
                        reverse=True,
                    )
                    return [str(d) for d, _ in paired[:n] if d is not None]
                elif docs:
                    return [str(d) for d in docs if d is not None]
        except Exception:
            pass

        return []

    def search_conversations(self, query: str, n_results: int = 3) -> str:
        """Search conversations for relevant context."""
        if self.conversations.count() == 0:
            return ""

        results = self.conversations.query(query_texts=[query], n_results=n_results)

        if not results:
            return ""

        documents = results.get("documents")
        if not documents or not documents[0]:
            return ""

        docs_str = [str(d) for d in documents[0] if d is not None]
        return "\n---\n".join(docs_str)

    # =========================================================================
    # TIER 3: SUMMARIES
    # =========================================================================

    def add_summary(self, summary: str, period: str = "session") -> None:
        """Store a conversation summary."""
        if not summary.strip():
            return

        summary_id = self._gen_id("summary")

        self.summaries.add(
            documents=[summary],
            metadatas=[{"period": period, "timestamp": datetime.now().isoformat()}],
            ids=[summary_id],
        )

    def get_summaries(self, n: int = 5) -> List[str]:
        """Get recent summaries."""
        total = self.summaries.count()
        if total == 0:
            return []

        try:
            results = self.summaries.get(limit=n)
            if results:
                docs = results.get("documents")
                metas = results.get("metadatas")
                if docs and metas:
                    paired = list(zip(docs, metas))
                    paired.sort(
                        key=lambda x: x[1].get("timestamp", "") if x[1] else "",
                        reverse=True,
                    )
                    return [str(d) for d, _ in paired[:n] if d is not None]
                elif docs:
                    return [str(d) for d in docs if d is not None]
        except Exception:
            pass

        return []

    # =========================================================================
    # STARTUP CONTEXT
    # =========================================================================

    def get_startup_context(self, recent_conversations: int = 5, include_summaries: int = 3) -> str:
        """Generate intelligent startup context."""
        context_parts: List[str] = []

        # TIER 1: All facts
        facts = self.get_all_facts()
        if facts:
            context_parts.append("=== KNOWN FACTS ABOUT USER ===")
            context_parts.append(facts)

        # TIER 2: Recent conversations
        recent = self.get_recent_conversations(recent_conversations)
        if recent:
            context_parts.append("\n=== RECENT CONVERSATIONS ===")
            for conv in recent[-3:]:
                if len(conv) > 300:
                    conv = conv[:300] + "..."
                context_parts.append(conv)

        # TIER 3: Summaries
        summaries = self.get_summaries(include_summaries)
        if summaries:
            context_parts.append("\n=== PREVIOUS SESSION SUMMARIES ===")
            for summary in summaries:
                context_parts.append(f"- {summary}")

        if not context_parts:
            return ""

        return "\n".join(context_parts)

    def get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """Get relevant context for a query (alias for search_all)."""
        return self.search_all(query, n_results)

    def search_all(self, query: str, n_results: int = 3) -> str:
        """Search across all tiers for relevant information.

        OPTIMIZED: Parallel tier searches using ThreadPoolExecutor.
        O(3n) sequential -> O(n) parallel (where n = query time)
        """
        results_list: List[str] = []

        # Define search tasks for parallel execution
        def search_facts():
            if self.facts.count() > 0:
                fact_results = self.facts.query(query_texts=[query], n_results=2)
                if fact_results:
                    docs = fact_results.get("documents")
                    if docs and docs[0]:
                        return [str(d) for d in docs[0] if d is not None]
            return []

        def search_conversations():
            if self.conversations.count() > 0:
                conv_results = self.conversations.query(query_texts=[query], n_results=n_results)
                if conv_results:
                    docs = conv_results.get("documents")
                    if docs and docs[0]:
                        return [str(d) for d in docs[0] if d is not None]
            return []

        def search_summaries():
            if self.summaries.count() > 0:
                summary_results = self.summaries.query(query_texts=[query], n_results=2)
                if summary_results:
                    docs = summary_results.get("documents")
                    if docs and docs[0]:
                        return [str(d) for d in docs[0] if d is not None]
            return []

        # Execute all searches in parallel
        # O(3 * query_time) sequential -> O(query_time) parallel
        futures = [
            self._executor.submit(search_facts),
            self._executor.submit(search_conversations),
            self._executor.submit(search_summaries),
        ]

        for future in as_completed(futures):
            try:
                tier_results = future.result(timeout=5.0)
                results_list.extend(tier_results)
            except Exception:
                pass

        if not results_list:
            return ""

        return "\n---\n".join(results_list[:n_results])

    # =========================================================================
    # MAINTENANCE
    # =========================================================================

    def backup(self, max_backups: int = 3) -> Optional[str]:
        """Backup memory database. Returns backup path or None on failure."""
        try:
            src = Path(self._persist_dir) if hasattr(self, "_persist_dir") else None
            if not src or not src.exists():
                self.log.warning("Cannot determine persist directory for backup")
                return None

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = src.parent / f"{src.name}_backup_{ts}"
            shutil.copytree(src, backup_dir)
            self.log.info(f"Memory backup: {backup_dir}")

            # Prune old backups, keep max_backups
            backups = sorted(
                src.parent.glob(f"{src.name}_backup_*"),
                key=lambda p: p.stat().st_mtime,
            )
            for old in backups[:-max_backups]:
                shutil.rmtree(old, ignore_errors=True)
                self.log.info(f"Pruned old backup: {old.name}")

            return str(backup_dir)
        except Exception as e:
            self.log.error(f"Backup failed: {e}")
            return None

    def close(self):
        """Explicit cleanup of resources."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
        self.log.info("SmartMemoryDB closed")

    def prune_expired(self, conv_days: int = 30, summary_days: int = 365) -> dict:
        """Remove expired conversations and summaries by TTL.

        Args:
            conv_days: Max age for conversations (default 30 for safety net)
            summary_days: Max age for summaries (default 365)

        Returns:
            Dict with counts of pruned items.
        """
        pruned = {"conversations": 0, "summaries": 0}
        now = datetime.now()

        # Prune conversations
        try:
            all_convos = self.conversations.get()
            if all_convos and all_convos.get("ids"):
                expired_ids = []
                for i, meta in enumerate(all_convos.get("metadatas", [])):
                    if not meta:
                        continue
                    ts_str = meta.get("timestamp", "")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if (now - ts).days > conv_days:
                            expired_ids.append(all_convos["ids"][i])
                    except (ValueError, IndexError):
                        continue
                if expired_ids:
                    for i in range(0, len(expired_ids), 5000):
                        batch = expired_ids[i : i + 5000]
                        self.conversations.delete(ids=batch)
                    pruned["conversations"] = len(expired_ids)
                    self.log.info(f"Pruned {len(expired_ids)} expired conversations")
        except Exception as e:
            self.log.error(f"Conversation pruning failed: {e}")

        # Prune summaries
        try:
            all_sums = self.summaries.get()
            if all_sums and all_sums.get("ids"):
                expired_ids = []
                for i, meta in enumerate(all_sums.get("metadatas", [])):
                    if not meta:
                        continue
                    ts_str = meta.get("timestamp", "")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if (now - ts).days > summary_days:
                            expired_ids.append(all_sums["ids"][i])
                    except (ValueError, IndexError):
                        continue
                if expired_ids:
                    for i in range(0, len(expired_ids), 5000):
                        batch = expired_ids[i : i + 5000]
                        self.summaries.delete(ids=batch)
                    pruned["summaries"] = len(expired_ids)
                    self.log.info(f"Pruned {len(expired_ids)} expired summaries")
        except Exception as e:
            self.log.error(f"Summary pruning failed: {e}")

        return pruned

    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        return {
            "facts": self.facts.count(),
            "conversations": self.conversations.count(),
            "summaries": self.summaries.count(),
            "total": self.facts.count() + self.conversations.count() + self.summaries.count(),
        }

    def clear_all(self) -> None:
        """Clear all memory."""
        self.log.warning("Clearing all memory...")
        self.client.delete_collection("olivia_facts")
        self.client.delete_collection("olivia_conversations")
        self.client.delete_collection("olivia_summaries")

        self.facts = self.client.get_or_create_collection(
            name="olivia_facts",
            embedding_function=self.embedding_fn,  # type: ignore
            metadata=self.HNSW_METADATA,
        )
        self.conversations = self.client.get_or_create_collection(
            name="olivia_conversations",
            embedding_function=self.embedding_fn,  # type: ignore
            metadata=self.HNSW_METADATA,
        )
        self.summaries = self.client.get_or_create_collection(
            name="olivia_summaries",
            embedding_function=self.embedding_fn,  # type: ignore
            metadata=self.HNSW_METADATA,
        )

        self.log.info("Memory clear complete.")

    def __del__(self):
        """Cleanup thread pool on destruction."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


class MemoryDB(SmartMemoryDB):
    """Backward compatible wrapper."""

    def add_memory(
        self, text: str, source: str = "user_chat", metadata: Optional[dict] = None
    ) -> None:
        if "User:" in text and "AI:" in text:
            parts = text.split("AI:")
            if len(parts) == 2:
                user_part = parts[0].replace("User:", "").strip()
                ai_part = parts[1].strip()
                self.add_conversation(user_part, ai_part)
                return
        self.add_conversation(text, "", auto_extract_facts=False)

    def search_memory(self, query: str, n_results: int = 3) -> str:
        return self.search_all(query, n_results)

    def get_memory_count(self) -> int:
        return self.get_stats()["total"]
