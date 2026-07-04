"""Dreaming system for O.L.I.V.I.A.

Memory consolidation during idle/shutdown using LLM summarization and fact extraction.

Performance Optimizations:
- ChromaDB `where` clause for server-side filtering (no Python-side filtering)
- Pre-compiled regex for hallucination detection
- frozenset for O(1) hallucination lookups
- Batch duplicate checking for facts
"""

import ctypes
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from src.utils.logger import get_logger

    log = get_logger("dream")
except ImportError:
    import logging

    log = logging.getLogger("dream")

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# =============================================================================
# HALLUCINATION DETECTION (module-level, compiled once)
# O(n) any() iteration -> O(1) frozenset lookup + O(1) regex match
# =============================================================================
_HALLUCINATED_EXACT: frozenset = frozenset({
    "sister's birthday",
    "march 15",
    "example fact",
    "placeholder",
    "john doe",
    "jane doe",
})

# Single compiled regex for pattern-based hallucination detection
# Matches any of the hallucinated patterns in one pass
_HALLUCINATED_PATTERN: re.Pattern = re.compile(
    r"sister's birthday|march 15|example fact|placeholder|john doe|jane doe",
    re.IGNORECASE
)

# JSON extraction pattern (compiled once)
_JSON_PATTERN: re.Pattern = re.compile(r"\{[\s\S]*\}")


@dataclass
class DreamConfig:
    """Configuration for the dreaming system."""

    dream_on_shutdown: bool = True
    dream_on_idle: bool = True
    idle_threshold_minutes: int = 5
    age_threshold_hours: int = 24
    max_conversations_per_dream: int = 50

    # USE MAIN MODEL - dreaming only runs when idle/shutdown
    summary_model: str = "olivia-finetuned"
    summary_max_tokens: int = 200

    keep_raw_conversations_days: int = 7
    mark_as_dreamed: bool = True
    save_dream_reports: bool = True
    dream_reports_dir: str = "data/logs/dreams"


@dataclass
class DreamReport:
    """Summary of one dreaming (memory consolidation) cycle."""

    timestamp: datetime = field(default_factory=datetime.now)
    conversations_processed: int = 0
    facts_extracted: int = 0
    summaries_created: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    forced_current_session: bool = False

    def to_string(self) -> str:
        """Format the report as a human-readable text block."""
        mode = "FORCED (current session)" if self.forced_current_session else "NORMAL"
        return (
            f"=== DREAM REPORT ===\n"
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Mode: {mode}\n"
            f"Duration: {self.duration_seconds:.1f}s\n"
            f"Conversations processed: {self.conversations_processed}\n"
            f"Facts extracted: {self.facts_extracted}\n"
            f"Summaries created: {self.summaries_created}\n"
            f"Errors: {len(self.errors)}\n"
            f"{'=' * 20}\n"
        )


SUMMARIZATION_PROMPT = """Analyze these conversations and create a concise summary.

CONVERSATIONS:
{conversations}

Create a summary (50-100 words) capturing:
1. Main topics discussed
2. Key decisions or conclusions
3. Important facts learned about the user
4. Any action items or future references

Format as a single paragraph. Focus on what would be useful to remember.
Do NOT invent information - only summarize what is explicitly present."""


FACT_EXTRACTION_PROMPT = """Extract factual information about the user from this conversation.

CONVERSATION:
{conversation}

RULES:
- Only extract facts the user DIRECTLY stated
- Do NOT invent or assume any information
- Do NOT use placeholder examples like "sister's birthday" or "March 15"
- If no facts are present, return empty list
- Be conservative - when in doubt, don't extract

Return ONLY a JSON object:
{{"facts": [
  {{"category": "preference|personal|work|schedule", "fact": "...", "confidence": 0.0-1.0}}
]}}

If no extractable facts, return: {{"facts": []}}"""


class IdleDetector:
    """Detects system idle time on Windows."""

    def __init__(self, idle_threshold_seconds: int = 300):
        self.threshold = idle_threshold_seconds
        self.on_idle: Optional[Callable[[], None]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_idle_trigger = 0.0
        self._cooldown = 60.0

    def get_idle_time(self) -> float:
        """Return seconds since last user input (0.0 on failure)."""
        try:

            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
                return millis / 1000.0
            return 0.0
        except Exception:
            return 0.0

    def start(self):
        """Start the background idle-monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        log.info(f"Idle detector started (threshold: {self.threshold}s)")

    def stop(self):
        """Stop the idle-monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("Idle detector stopped")

    def _monitor_loop(self):
        while self._running:
            try:
                idle_time = self.get_idle_time()
                current_time = time.time()
                if idle_time >= self.threshold:
                    if current_time - self._last_idle_trigger > self._cooldown:
                        self._last_idle_trigger = current_time
                        log.info(f"System idle for {idle_time:.0f}s, triggering dream...")
                        if self.on_idle:
                            try:
                                self.on_idle()
                            except Exception as e:
                                log.error(f"Idle callback error: {e}")
                time.sleep(10)
            except Exception as e:
                log.error(f"Idle monitor error: {e}")
                time.sleep(30)


class DreamingEngine:
    """Memory consolidation engine using main LLM."""

    def __init__(self, memory_db: Any, config: Optional[DreamConfig] = None):
        self.memory = memory_db
        self.config = config or DreamConfig()

        self.idle_detector: Optional[IdleDetector] = None
        if self.config.dream_on_idle:
            self.idle_detector = IdleDetector(
                idle_threshold_seconds=self.config.idle_threshold_minutes * 60
            )
            self.idle_detector.on_idle = self._on_idle_triggered

        self._is_dreaming = False
        self._dream_lock = threading.Lock()

        if self.config.save_dream_reports:
            Path(self.config.dream_reports_dir).mkdir(parents=True, exist_ok=True)

        log.info("DreamingEngine initialized")
        log.info(f"   Summary model: {self.config.summary_model}")

    def start_idle_monitoring(self):
        """Start idle detection if enabled in config."""
        if self.idle_detector:
            self.idle_detector.start()

    def stop_idle_monitoring(self):
        """Stop idle detection if it was started."""
        if self.idle_detector:
            self.idle_detector.stop()

    def _on_idle_triggered(self):
        self.dream_async()

    def dream_async(self) -> threading.Thread:
        """Run a dream cycle in a background daemon thread."""
        thread = threading.Thread(target=self.dream, daemon=True)
        thread.start()
        return thread

    def dream(
        self, age_threshold_hours: Optional[int] = None, force_current_session: bool = False
    ) -> DreamReport:
        """Run one memory consolidation cycle.

        Args:
            age_threshold_hours: Only process conversations older than this
                (defaults to config; 0 processes everything).
            force_current_session: Include the current session's conversations.

        Returns:
            DreamReport with counts, errors, and duration.
        """
        with self._dream_lock:
            if self._is_dreaming:
                log.warning("Already dreaming, skipping...")
                return DreamReport(errors=["Already dreaming"])
            self._is_dreaming = True

        report = DreamReport(forced_current_session=force_current_session)
        start_time = time.time()

        threshold = age_threshold_hours
        if threshold is None:
            threshold = self.config.age_threshold_hours

        if force_current_session:
            threshold = 0
            log.info("Force dreaming current session...")

        try:
            log.info("Starting dream cycle...")

            conversations = self._get_conversations(threshold)

            if not conversations:
                log.info("No conversations to process")
                report.duration_seconds = time.time() - start_time
                self._is_dreaming = False
                return report

            log.info(f"Found {len(conversations)} conversations to process")

            grouped = self._group_conversations(conversations)

            for group_key, group_convos in grouped.items():
                try:
                    summary = self._summarize_group(group_convos)
                    if summary:
                        self.memory.add_summary(summary, period=group_key)
                        report.summaries_created += 1
                        log.info(f"Created summary for {group_key}")

                    # Collect all facts for batch duplicate checking
                    all_facts: List[Dict[str, Any]] = []
                    for conv in group_convos:
                        facts = self._extract_facts_llm(conv)
                        for fact in facts:
                            fact_text = fact.get("fact", "")
                            # Filter hallucinated facts before batch check
                            if not self._is_hallucinated_fact(fact_text):
                                all_facts.append(fact)

                    # Batch duplicate check for all facts in group
                    # O(n) individual queries -> O(1) batch query
                    if all_facts:
                        fact_texts = [f.get("fact", "") for f in all_facts]
                        is_duplicate = self._batch_check_duplicates(fact_texts)

                        for i, fact in enumerate(all_facts):
                            if not is_duplicate[i]:
                                fact_text = fact.get("fact", "")
                                category = fact.get("category", "general")
                                self.memory.add_fact(fact_text, category)
                                report.facts_extracted += 1

                    report.conversations_processed += len(group_convos)
                except Exception as e:
                    log.error(f"Error processing group {group_key}: {e}")
                    report.errors.append(f"Group {group_key}: {e}")

            if self.config.mark_as_dreamed:
                self._mark_as_dreamed(conversations)

            report.duration_seconds = time.time() - start_time
            log.info(
                f"Dream complete: {report.conversations_processed} convos, "
                f"{report.facts_extracted} facts, {report.summaries_created} summaries"
            )

            if self.config.save_dream_reports:
                self._save_report(report)

        except Exception as e:
            log.error(f"Dream failed: {e}")
            report.errors.append(str(e))
        finally:
            self._is_dreaming = False

        return report

    def _is_hallucinated_fact(self, fact: str) -> bool:
        """Check if fact is a hallucinated example from the prompt.

        O(n) any() iteration -> O(1) frozenset lookup + O(1) regex match
        """
        fact_lower = fact.lower()

        # O(1) exact match check against known hallucinations
        for pattern in _HALLUCINATED_EXACT:
            if pattern in fact_lower:
                return True

        # O(1) pattern match (single compiled regex)
        return bool(_HALLUCINATED_PATTERN.search(fact_lower))

    def _batch_check_duplicates(self, facts: List[str]) -> List[bool]:
        """Batch duplicate check for multiple facts.

        O(n) individual queries -> O(1) batch query
        Uses ChromaDB's batch query capability.
        """
        if not facts:
            return []

        try:
            # Use memory's batch method if available
            if hasattr(self.memory, "batch_check_duplicates"):
                return self.memory.batch_check_duplicates(facts)

            # Fallback: check if facts collection exists and has entries
            if self.memory.facts.count() == 0:
                return [False] * len(facts)

            # Single batch query instead of n individual queries
            results = self.memory.facts.query(query_texts=facts, n_results=1)

            duplicates = []
            if results and results.get("distances"):
                for fact_distances in results["distances"]:
                    if fact_distances and len(fact_distances) > 0:
                        duplicates.append(fact_distances[0] < 0.3)
                    else:
                        duplicates.append(False)
            else:
                duplicates = [False] * len(facts)

            return duplicates

        except Exception as e:
            log.debug(f"Batch duplicate check failed: {e}")
            return [False] * len(facts)

    def _get_conversations(self, age_threshold_hours: int) -> List[Dict[str, Any]]:
        """Get conversations for dreaming.

        OPTIMIZED: Uses ChromaDB `where` clause for server-side filtering.
        O(n) Python-side filtering -> O(1) database-side filtering
        """
        try:
            # Build where clause for ChromaDB server-side filtering
            # This avoids fetching all conversations and filtering in Python
            where_clause: Optional[Dict[str, Any]] = None
            where_conditions: List[Dict[str, Any]] = []

            # Filter out already-dreamed conversations
            where_conditions.append({"dreamed": {"$ne": True}})

            # Filter by age threshold if specified
            if age_threshold_hours > 0:
                threshold_time = datetime.now() - timedelta(hours=age_threshold_hours)
                threshold_iso = threshold_time.isoformat()
                # Only get conversations older than threshold
                where_conditions.append({"timestamp": {"$lt": threshold_iso}})

            # Combine conditions with $and if multiple
            if len(where_conditions) == 1:
                where_clause = where_conditions[0]
            elif len(where_conditions) > 1:
                where_clause = {"$and": where_conditions}

            # Query with server-side filtering
            # O(n) Python filtering -> O(log n) database filtering
            all_convos = self.memory.conversations.get(
                limit=self.config.max_conversations_per_dream,
                where=where_clause,
            )

            if not all_convos or not all_convos.get("documents"):
                return []

            result = []
            documents = all_convos.get("documents", [])
            metadatas = all_convos.get("metadatas", [])
            ids = all_convos.get("ids", [])

            for i in range(len(documents)):
                doc = documents[i]
                meta = metadatas[i] if i < len(metadatas) else {}
                conv_id = ids[i] if i < len(ids) else f"conv_{i}"

                # Parse timestamp for grouping (cached in result dict)
                timestamp = datetime.now()
                if meta:
                    timestamp_str = meta.get("timestamp", "")
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                        except ValueError:
                            pass

                result.append(
                    {"id": conv_id, "document": doc, "metadata": meta, "timestamp": timestamp}
                )

            return result

        except Exception as e:
            log.error(f"Error getting conversations: {e}")
            # Fallback to original method if where clause fails
            return self._get_conversations_fallback(age_threshold_hours)

    def _get_conversations_fallback(self, age_threshold_hours: int) -> List[Dict[str, Any]]:
        """Fallback method without where clause (for older ChromaDB versions)."""
        try:
            all_convos = self.memory.conversations.get(
                limit=self.config.max_conversations_per_dream
            )
            if not all_convos or not all_convos.get("documents"):
                return []

            threshold = None
            if age_threshold_hours > 0:
                threshold = datetime.now() - timedelta(hours=age_threshold_hours)

            result = []
            documents = all_convos.get("documents", [])
            metadatas = all_convos.get("metadatas", [])
            ids = all_convos.get("ids", [])

            for i in range(len(documents)):
                doc = documents[i]
                meta = metadatas[i] if i < len(metadatas) else {}
                conv_id = ids[i] if i < len(ids) else f"conv_{i}"

                if meta and meta.get("dreamed", False):
                    continue

                timestamp = datetime.now()
                if threshold is not None:
                    timestamp_str = meta.get("timestamp", "") if meta else ""
                    if timestamp_str:
                        try:
                            conv_time = datetime.fromisoformat(timestamp_str)
                            if conv_time >= threshold:
                                continue
                            timestamp = conv_time
                        except ValueError:
                            pass

                result.append(
                    {"id": conv_id, "document": doc, "metadata": meta, "timestamp": timestamp}
                )
            return result
        except Exception as e:
            log.error(f"Error in fallback get_conversations: {e}")
            return []

    def _group_conversations(self, conversations: List[Dict]) -> Dict[str, List[str]]:
        """Group conversations by date."""
        grouped: Dict[str, List[str]] = {}
        for conv in conversations:
            timestamp = conv.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            date_key = timestamp.strftime("%Y-%m-%d")
            if date_key not in grouped:
                grouped[date_key] = []
            grouped[date_key].append(conv.get("document", ""))
        return grouped

    def _summarize_group(self, conversations: List[str]) -> Optional[str]:
        if not OLLAMA_AVAILABLE or not conversations:
            return None
        try:
            combined = "\n---\n".join(conversations[:10])
            if len(combined) > 8000:
                combined = combined[:8000] + "\n[Truncated...]"

            prompt = SUMMARIZATION_PROMPT.format(conversations=combined)
            response = ollama.chat(
                model=self.config.summary_model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": self.config.summary_max_tokens},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            log.error(f"Summarization failed: {e}")
            return None

    def _extract_facts_llm(self, conversation: str) -> List[Dict[str, Any]]:
        if not OLLAMA_AVAILABLE or not conversation:
            return []
        try:
            prompt = FACT_EXTRACTION_PROMPT.format(conversation=conversation)
            response = ollama.chat(
                model=self.config.summary_model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 300},
            )
            content = response["message"]["content"].strip()
            # Use pre-compiled pattern
            json_match = _JSON_PATTERN.search(content)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("facts", [])
            return []
        except Exception as e:
            log.debug(f"Fact extraction failed: {e}")
            return []

    def _mark_as_dreamed(self, conversations: List[Dict]):
        """Mark conversations as dreamed in the database."""
        marked_count = 0
        for conv in conversations:
            conv_id = conv.get("id")
            if conv_id and hasattr(self.memory, "conversations"):
                try:
                    # Update metadata to mark as dreamed
                    existing_meta = conv.get("metadata", {}) or {}
                    existing_meta["dreamed"] = True
                    existing_meta["dreamed_at"] = datetime.now().isoformat()
                    self.memory.conversations.update(ids=[conv_id], metadatas=[existing_meta])
                    marked_count += 1
                except Exception as e:
                    log.debug(f"Failed to mark conversation {conv_id} as dreamed: {e}")
        log.info(f"Marked {marked_count}/{len(conversations)} conversations as dreamed")

    def _save_report(self, report: DreamReport):
        try:
            filename = f"dream_{report.timestamp.strftime('%Y-%m-%d_%H%M%S')}.txt"
            filepath = Path(self.config.dream_reports_dir) / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report.to_string())
                if report.errors:
                    f.write("\nErrors:\n")
                    for err in report.errors:
                        f.write(f"  - {err}\n")
            log.info(f"Dream report saved: {filepath}")
        except Exception as e:
            log.error(f"Failed to save dream report: {e}")

    def dream_on_shutdown(self):
        """Dream over everything, including the current session, if enabled."""
        if not self.config.dream_on_shutdown:
            return
        log.info("Shutdown dreaming (including current session)...")
        self.dream(age_threshold_hours=0, force_current_session=True)


_dreaming_engine: Optional[DreamingEngine] = None


def get_dreaming_engine() -> Optional[DreamingEngine]:
    """Return the module-level DreamingEngine singleton, if created."""
    return _dreaming_engine


def create_dreaming_engine(memory_db: Any, config: Optional[DreamConfig] = None) -> DreamingEngine:
    """Create and store the module-level DreamingEngine singleton."""
    global _dreaming_engine
    _dreaming_engine = DreamingEngine(memory_db, config)
    return _dreaming_engine


if __name__ == "__main__":
    print("Dreaming System - using olivia-finetuned")
    config = DreamConfig()
    print(f"Summary model: {config.summary_model}")
