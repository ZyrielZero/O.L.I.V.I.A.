"""Chat API endpoints."""

import asyncio
import json
import logging
from datetime import datetime
from functools import lru_cache
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dependencies import LLMServiceDep, MemoryServiceDep, get_service
from src.api.models.chat import ChatRequest, ChatResponse
from src.api.services.tts_queue import SentenceTTSQueue
from src.api.utils.sentence_buffer import SentenceBuffer
from src.api.utils.tts_sanitizer import sanitize_for_tts

log = logging.getLogger("api.chat")

_GREETING_PATTERNS = frozenset(["hi", "hello", "hey", "thanks", "thank you"])


@lru_cache(maxsize=64)
def _is_simple_greeting(text_lower: str) -> bool:
    """Check if text is a simple greeting.

    Complexity: O(1) frozenset lookup after stripping.
    Uses LRU cache to avoid repeated string operations on common inputs.
    """
    return text_lower.strip().rstrip("!?.") in _GREETING_PATTERNS


# OPT: Pre-built JSON templates to avoid json.dumps() overhead per token
# json.dumps({'token': '', 'done': True}) is ~1.5us vs template format ~0.3us
_JSON_TOKEN_TEMPLATE = '{{"token": {}, "done": false}}'
_JSON_DONE = '{"token": "", "done": true}'
_JSON_ERROR_TEMPLATE = '{{"error": {}, "done": true}}'


async def _prefetch_memory(memory, msg: str, n=3) -> str:
    """Prefetch memory context in background (runs parallel with search)."""
    if len(msg.split()) <= 6:  # skip greetings
        return ""
    try:
        return await memory.get_relevant_context(msg, n_results=n)
    except Exception as e:
        log.warning(f"Memory prefetch failed: {e}")
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

        # prefetch memory
        mem_task = asyncio.create_task(_prefetch_memory(memory, request.message))

        # get memory context
        try:
            mem_ctx = await mem_task
            if mem_ctx:
                ctx = f"{ctx}\n\n{mem_ctx}" if ctx else mem_ctx
        except Exception as e:
            log.warning(f"Memory fetch failed: {e}")

        if request.stream:

            async def gen_sse() -> AsyncGenerator[str, None]:
                """SSE stream with concurrent TTS synthesis.

                TTS queue starts before the LLM loop — sentences are fed to TTS
                as they complete, so the first sentence plays while the LLM is
                still generating the rest.
                """
                resp_chunks: list[str] = []
                sent_buf = SentenceBuffer()
                tts = get_service("tts")

                # Start TTS queue upfront so sentences stream in during LLM generation
                tts_q = None
                if tts and tts.is_initialized():
                    tts_q = SentenceTTSQueue(tts.synthesize_f32, tts.play_f32)
                    await tts_q.start()

                try:
                    async for tok in llm.chat_stream(
                        message=request.message,
                        context=ctx or None,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                    ):
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

                    asyncio.create_task(_store_conversation(memory, request.message, full_resp))

                    # UI gets done signal immediately — TTS continues playing in background
                    yield f"data: {_JSON_DONE}\n\n"

                    # Wait for TTS to finish (keeps StreamingResponse alive)
                    if tts_q:
                        try:
                            await tts_q.finish()
                        except Exception as e:
                            log.error(f"TTS finish error: {e}")
                            await tts_q.stop()

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

            asyncio.create_task(_store_conversation(memory, request.message, full_resp))

            if full_resp:
                tts = get_service("tts")
                if tts and tts.is_initialized():
                    asyncio.create_task(_speak_bg(tts, sanitize_for_tts(full_resp)))

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
