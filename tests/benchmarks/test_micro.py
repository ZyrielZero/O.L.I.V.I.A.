"""CPU-safe micro-benchmarks (pytest-benchmark) — the CI benchmarks job.

These guard logic-level hot paths only. GPU voice-pipeline numbers can never
come from CI runners; those come from tools/bench.py on the dev machine.
"""

from datetime import datetime

import pytest

from src.api.utils.sentence_buffer import SentenceBuffer
from src.api.utils.tts_sanitizer import sanitize_for_tts
from src.core.llm.ollama_client import ConversationManager

pytestmark = pytest.mark.benchmark

_TOKENS = (
    "The weather is looking really nice today . I hope you get outside and enjoy it ! "
    "Let me know if you want ideas for something fun to do this afternoon . "
    "There is a park nearby with a lake and some walking trails you might like ."
).split(" ")

_DIRTY_TEXT = (
    "Hello there! [MEMORY]user likes tea[/MEMORY] *smiles warmly* "
    "Here's my answer: ### Section\n- item one\n- item two\n"
    "I think you'll *nods* find it useful..."
)


def test_sentence_buffer_stream(benchmark):
    """Token stream through the sentence buffer (chat hot path)."""

    def run():
        buf = SentenceBuffer()
        sentences = []
        for tok in _TOKENS:
            sentences.extend(buf.add(tok + " "))
        tail = buf.flush()
        if tail:
            sentences.append(tail)
        return sentences

    result = benchmark(run)
    assert len(result) >= 2


def test_tts_sanitizer(benchmark):
    """Sanitizer pass over a representative dirty response."""
    result = benchmark(sanitize_for_tts, _DIRTY_TEXT)
    assert "[MEMORY]" not in result
    assert "*" not in result


def test_llm_payload_build(benchmark):
    """Payload construction with history and injected context."""
    manager = ConversationManager(model="bench-model", system_prompt="Benchmark prompt")
    for i in range(10):
        manager.history.append({"role": "user", "content": f"user message {i}"})
        manager.history.append({"role": "assistant", "content": f"assistant reply {i}"})

    payload = benchmark(
        manager._build_payload, "hello there", "some background context", 0.7, 128
    )
    assert payload["messages"][-1]["content"] == "hello there"


@pytest.fixture(scope="module")
def bench_db(tmp_path_factory):
    """Small real ChromaDB for query benchmarks."""
    from src.core.memory.smart_memory import SmartMemoryDB

    db = SmartMemoryDB(persist_directory=str(tmp_path_factory.mktemp("benchdb")))
    now = datetime.now().isoformat()
    for i in range(50):
        db.conversations.add(
            documents=[f"User: tell me about topic {i}\nAssistant: topic {i} is interesting"],
            metadatas=[{"timestamp": now}],
            ids=[f"bench_conv_{i}"],
        )
    db.facts.add(
        documents=["User's favorite topic is 25"],
        metadatas=[{"category": "preference", "timestamp": now}],
        ids=["bench_fact_1"],
    )
    yield db
    db.close()


def test_memory_search_all(benchmark, bench_db):
    """Three-tier parallel memory search against a real temp ChromaDB."""
    result = benchmark(bench_db.search_all, "tell me about topic 25", 3)
    assert result  # non-empty context comes back
