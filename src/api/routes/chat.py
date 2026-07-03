"""Chat API endpoints."""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dependencies import LLMServiceDep, MemoryServiceDep, get_service
from src.api.models.chat import ChatRequest, ChatResponse
from src.api.services.metrics import get_metrics
from src.api.utils.sentence_buffer import SentenceBuffer
from src.api.utils.tts_sanitizer import sanitize_for_tts

log = logging.getLogger("api.chat")

_JSON_TOKEN_TEMPLATE = '{{"token": {}, "done": false}}'
_JSON_DONE = '{"token": "", "done": true}'
_JSON_ERROR_TEMPLATE = '{{"error": {}, "done": true}}'


# Hard cap on the pre-chat memory lookup: a slow Chroma query must not
# stall chat startup — on timeout we proceed with no memory context
_MEMORY_FETCH_TIMEOUT = 1.5

# Strong references to fire-and-forget tasks: asyncio only keeps weak refs,
# so a bare create_task() can be garbage-collected mid-flight and memory
# writes silently vanish
_bg_tasks: set[asyncio.Task] = set()


def _create_bg_task(coro) -> asyncio.Task:
    """Create a background task and keep a reference until it completes."""
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


async def _fetch_memory_context(memory, msg: str, n=3) -> str:
    """Fetch memory context for a message; empty string on failure/timeout."""
    if len(msg.split()) <= 6:  # skip greetings
        return ""
    try:
        return await asyncio.wait_for(
            memory.get_relevant_context(msg, n_results=n),
            timeout=_MEMORY_FETCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning(f"Memory lookup exceeded {_MEMORY_FETCH_TIMEOUT}s; continuing without it")
        return ""
    except Exception as e:
        log.warning(f"Memory fetch failed: {e}")
        return ""


async def _speak_bg(tts, text: str) -> None:
    """Fire-and-forget TTS playback."""
    try:
        await tts.speak(text)
    except Exception as e:
        log.warning(f"TTS failed: {e}")


async def _store_conversation(memory, user_msg: str, ai_msg: str) -> None:
    """Fire-and-forget conversation storage + LLM fact extraction."""
    try:
        await memory.add_conversation(user_msg, ai_msg, auto_extract=True)
        # Queue for deeper LLM-based fact extraction (background worker)
        from src.api.container import get_container

        extractor = get_container().fact_extractor
        if extractor:
            extractor.llm_extractor.queue_extraction(user_msg, ai_msg)
    except Exception as e:
        log.warning(f"Failed to store conversation: {e}")


router = APIRouter(prefix="/api", tags=["chat"])


@router.delete("/history")
async def clear_history(llm: LLMServiceDep):
    """Clear conversation history."""
    await llm.clear_history()
    return {"status": "ok"}


@router.post("/chat")
async def chat(request: ChatRequest, llm: LLMServiceDep, memory: MemoryServiceDep):
    """Chat endpoint. stream=True returns SSE, else JSON."""
    try:
        ctx = request.context or ""

        mem_ctx = await _fetch_memory_context(memory, request.message)
        if mem_ctx:
            ctx = f"{ctx}\n\n{mem_ctx}" if ctx else mem_ctx

        if request.stream:

            async def gen_sse() -> AsyncGenerator[str, None]:
                """SSE stream with concurrent TTS synthesis.

                Sentences are fed into the SESSION-scoped TTS queue (owned by
                the service layer, not this request) as they complete — the
                first sentence plays while the LLM is still generating, and a
                client closing the SSE stream cannot kill playback (Phase 1.1).
                """
                resp_chunks: list[str] = []
                sent_buf = SentenceBuffer()
                tts = get_service("tts")

                tts_q = None
                if tts and tts.is_initialized():
                    # Lazy import: tts_service pulls torch/ChatterBox
                    from src.api.services.tts_service import get_session_tts_queue

                    tts_q = await get_session_tts_queue(tts)

                llm_start = time.perf_counter()
                first_token_at = None
                try:
                    async for tok in llm.chat_stream(
                        message=request.message,
                        context=ctx or None,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    ):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            get_metrics().record(
                                "llm_ttft_ms", (first_token_at - llm_start) * 1000
                            )
                        resp_chunks.append(tok)
                        yield f"data: {_JSON_TOKEN_TEMPLATE.format(json.dumps(tok))}\n\n"

                        # Feed completed sentences to TTS immediately
                        if tts_q:
                            for sent in sent_buf.add(tok):
                                try:
                                    await tts_q.queue_sentence(sanitize_for_tts(sent))
                                except Exception as e:
                                    log.error(f"TTS queue error: {e}")
                                    await tts_q.stop()
                                    tts_q = None
                                    break
                        else:
                            # Still consume from buffer to avoid stale state
                            for _ in sent_buf.add(tok):
                                pass

                    # Flush remaining text
                    if final := sent_buf.flush():
                        if tts_q:
                            try:
                                await tts_q.queue_sentence(sanitize_for_tts(final))
                            except Exception as e:
                                log.error(f"TTS queue error: {e}")
                                await tts_q.stop()
                                tts_q = None

                    full_resp = "".join(resp_chunks)
                    get_metrics().record("llm_total_ms", (time.perf_counter() - llm_start) * 1000)

                    _create_bg_task(_store_conversation(memory, request.message, full_resp))

                    # Done signal goes out immediately; the session TTS queue
                    # keeps playing on its own — no finish() barrier here
                    yield f"data: {_JSON_DONE}\n\n"

                except Exception as e:
                    log.error(f"Stream error: {e}")
                    yield f"data: {_JSON_ERROR_TEMPLATE.format(json.dumps(str(e)))}\n\n"
                    if tts_q:
                        await tts_q.stop()

            return StreamingResponse(
                gen_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # non-streaming
        else:
            # OPT: List append + join pattern - O(n) vs O(n^2) for string +=
            resp_chunks: list[str] = []
            async for tok in llm.chat_stream(
                message=request.message,
                context=ctx or None,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                resp_chunks.append(tok)
            full_resp = "".join(resp_chunks)

            _create_bg_task(_store_conversation(memory, request.message, full_resp))

            if full_resp:
                tts = get_service("tts")
                if tts and tts.is_initialized():
                    _create_bg_task(_speak_bg(tts, sanitize_for_tts(full_resp)))

            return ChatResponse(
                message=full_resp,
                timestamp=datetime.now(),
                metadata={
                    "context_length": len(ctx) if ctx else 0,
                },
            )

    except Exception as e:
        log.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
