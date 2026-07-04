"""/ws/voice — full-duplex voice pipeline over WebSocket (Phase 1.3/1.4).

Protocol
--------
Client -> server:
  binary frames : raw audio, 16 kHz mono s16le PCM (any chunk size)
  text frames   : JSON control events
    {"type": "start"}            begin a capture session
    {"type": "stop"}             end the capture session
    {"type": "ping"}             keepalive; answered with {"type": "pong"}

Server -> client:
  text frames   : JSON control events
    {"type": "ready"}                              session accepted
    {"type": "speech_start"}                       VAD detected user speech
    {"type": "transcript_final", "text": ...}      utterance transcribed
    {"type": "token", "text": ...}                 LLM token
    {"type": "audio_start", "sample_rate": N, "format": "s16le"}
    {"type": "audio_end"}
    {"type": "done"}                               response complete
    {"type": "barge_in"}                           user interrupted playback
    {"type": "error", "message": ...}
  binary frames : TTS audio, mono s16le PCM at the announced sample_rate
    (only between audio_start and audio_end)

transcript_partial is reserved in the protocol but not emitted yet —
faster-whisper is not a streaming decoder.

Audio playback happens CLIENT-side: the server never touches the speaker
for WS sessions, which is what makes the backend containerizable later.
"""

import asyncio
import json
import logging
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.container import get_container
from src.api.services.metrics import get_metrics
from src.api.utils.sentence_buffer import SentenceBuffer
from src.api.utils.tts_sanitizer import sanitize_for_tts

log = logging.getLogger("api.voice")

router = APIRouter(tags=["voice"])

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # Silero VAD v5 operates on 512-sample windows @16k

# Defaults for the tunable values live in RuntimeSettings (settings_service);
# these constants remain as fixed limits / documented defaults
SILENCE_END_S = 0.5                       # default; runtime value from settings
MIN_SPEECH_S = 0.3                        # shorter bursts are discarded
BARGE_IN_CONFIRM_CHUNKS = 5               # default; runtime value from settings
MAX_UTTERANCE_S = 30.0

_INT16_MAX = np.float32(32767.0)


class VoiceSession:
    """One WebSocket voice session: VAD -> STT -> LLM -> TTS -> client audio."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self._vad_model = None
        self._buf = np.empty(0, dtype=np.float32)      # unprocessed samples
        self._speech: list[np.ndarray] = []            # current utterance
        self._speaking = False
        self._silence_chunks = 0
        self._speech_confirm = 0
        self._response_task: Optional[asyncio.Task] = None
        self._tts_playing = False

    # -- lifecycle -------------------------------------------------------------

    async def run(self) -> None:
        """Load VAD, then route incoming frames until the client disconnects."""
        from src.core.speech.stt import _get_silero_vad

        loop = asyncio.get_running_loop()
        self._vad_model = await loop.run_in_executor(None, _get_silero_vad)
        await self._send({"type": "ready"})

        while True:
            message = await self.ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await self._on_audio(data)
            elif (text := message.get("text")) is not None:
                await self._on_control(text)

    async def close(self) -> None:
        """Cancel any in-flight response task."""
        await self._cancel_response()

    # -- inbound ---------------------------------------------------------------

    async def _on_control(self, text: str) -> None:
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            await self._send({"type": "error", "message": "malformed control frame"})
            return

        kind = event.get("type")
        if kind == "ping":
            await self._send({"type": "pong"})
        elif kind == "stop":
            await self._cancel_response()
            self._reset_capture()
        elif kind == "start":
            self._reset_capture()

    async def _on_audio(self, data: bytes) -> None:
        """Feed incoming PCM through VAD; dispatch utterances downstream."""
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / _INT16_MAX
        self._buf = np.concatenate([self._buf, pcm])

        while len(self._buf) >= CHUNK_SAMPLES:
            chunk, self._buf = self._buf[:CHUNK_SAMPLES], self._buf[CHUNK_SAMPLES:]
            await self._process_chunk(chunk)

    async def _process_chunk(self, chunk: np.ndarray) -> None:
        import torch

        from src.api.services.settings_service import get_settings_service

        settings = get_settings_service().get()
        loop = asyncio.get_running_loop()
        threshold = (
            min(settings.vad_threshold + 0.2, 0.95)
            if self._tts_playing
            else settings.vad_threshold
        )

        def _vad() -> float:
            with torch.no_grad():
                return float(self._vad_model(torch.from_numpy(chunk.copy()), SAMPLE_RATE))

        speech_prob = await loop.run_in_executor(None, _vad)
        is_speech = speech_prob > threshold

        if self._tts_playing:
            # Barge-in guard: require N consecutive confident chunks so
            # residual echo can't interrupt playback (Phase 1.4)
            if is_speech:
                self._speech_confirm += 1
                if self._speech_confirm >= settings.barge_in_confirm_chunks:
                    await self._barge_in()
            else:
                self._speech_confirm = 0
            if self._tts_playing:
                return  # don't capture while the assistant is audible

        chunk_s = CHUNK_SAMPLES / SAMPLE_RATE
        if is_speech:
            if not self._speaking:
                self._speaking = True
                self._speech = []
                await self._send({"type": "speech_start"})
            self._silence_chunks = 0
            self._speech.append(chunk)
            if len(self._speech) * chunk_s > MAX_UTTERANCE_S:
                await self._end_utterance()
        elif self._speaking:
            self._silence_chunks += 1
            self._speech.append(chunk)  # keep trailing context for STT
            if self._silence_chunks * chunk_s >= settings.silence_end_s:
                await self._end_utterance()

    # -- pipeline ---------------------------------------------------------------

    async def _end_utterance(self) -> None:
        # Speech duration excludes the trailing silence chunks kept for STT context
        speech_chunks = len(self._speech) - self._silence_chunks
        audio = np.concatenate(self._speech) if self._speech else np.empty(0, np.float32)
        self._reset_capture()

        if speech_chunks * (CHUNK_SAMPLES / SAMPLE_RATE) < MIN_SPEECH_S:
            return

        await self._cancel_response()
        self._response_task = asyncio.create_task(self._respond(audio))

    async def _respond(self, audio: np.ndarray) -> None:
        """STT -> LLM -> TTS for one utterance; streams everything to the client."""
        container = get_container()
        stt, llm, tts = container.stt, container.llm, container.tts
        utterance_end = time.perf_counter()

        try:
            if not (stt and stt.is_initialized()):
                await self._send({"type": "error", "message": "STT not ready"})
                return

            text = await stt.transcribe_numpy(audio)
            get_metrics().record("stt_ms", (time.perf_counter() - utterance_end) * 1000)
            if not text.strip():
                return
            await self._send({"type": "transcript_final", "text": text})

            if not llm:
                await self._send({"type": "error", "message": "LLM not ready"})
                return

            sent_buf = SentenceBuffer()
            first_audio_sent = False
            llm_start = time.perf_counter()
            first_token = None
            chunks: list[str] = []

            async def speak(sentence: str) -> None:
                nonlocal first_audio_sent
                clean = sanitize_for_tts(sentence)
                if not clean or not (tts and tts.is_initialized()):
                    return
                pcm = await tts.synthesize_f32(clean)
                if pcm is None or len(pcm) == 0:
                    return
                if not first_audio_sent:
                    first_audio_sent = True
                    get_metrics().record(
                        "voice_to_voice_ms", (time.perf_counter() - utterance_end) * 1000
                    )
                    await self._send(
                        {
                            "type": "audio_start",
                            "sample_rate": tts.config.sample_rate,
                            "format": "s16le",
                        }
                    )
                    self._tts_playing = True
                    self._speech_confirm = 0
                await self.ws.send_bytes(
                    (np.clip(pcm, -1.0, 1.0) * _INT16_MAX).astype(np.int16).tobytes()
                )

            async for tok in llm.chat_stream(message=text):
                if first_token is None:
                    first_token = time.perf_counter()
                    get_metrics().record("llm_ttft_ms", (first_token - llm_start) * 1000)
                chunks.append(tok)
                await self._send({"type": "token", "text": tok})
                for sentence in sent_buf.add(tok):
                    await speak(sentence)

            get_metrics().record("llm_total_ms", (time.perf_counter() - llm_start) * 1000)
            if tail := sent_buf.flush():
                await speak(tail)

            if first_audio_sent:
                await self._send({"type": "audio_end"})
                self._tts_playing = False
            await self._send({"type": "done"})

            if container.memory and chunks:
                from src.api.routes.chat import _create_bg_task, _store_conversation

                _create_bg_task(_store_conversation(container.memory, text, "".join(chunks)))

        except asyncio.CancelledError:
            self._tts_playing = False
            raise
        except Exception as e:
            log.error(f"Voice pipeline error: {e}", exc_info=True)
            self._tts_playing = False
            try:
                await self._send({"type": "error", "message": str(e)})
            except Exception:
                pass

    async def _barge_in(self) -> None:
        """User spoke over playback: kill the in-flight response (Phase 1.4)."""
        log.info("Barge-in: user speech during playback")
        self._tts_playing = False
        self._speech_confirm = 0
        await self._cancel_response()
        await self._send({"type": "barge_in"})
        # Client drops its own buffered audio on barge_in; the assistant
        # starts listening again immediately
        self._speaking = True
        self._speech = []
        self._silence_chunks = 0
        await self._send({"type": "speech_start"})

    # -- helpers -----------------------------------------------------------------

    async def _cancel_response(self) -> None:
        task, self._response_task = self._response_task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _reset_capture(self) -> None:
        self._speaking = False
        self._speech = []
        self._silence_chunks = 0
        self._speech_confirm = 0

    async def _send(self, event: dict) -> None:
        await self.ws.send_text(json.dumps(event))


@router.websocket("/ws/voice")
async def voice_ws(ws: WebSocket) -> None:
    """Full-duplex voice session."""
    await ws.accept()
    session = VoiceSession(ws)
    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"Voice session error: {e}", exc_info=True)
    finally:
        await session.close()
        log.info("Voice session closed")
