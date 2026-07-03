"""WebSocket voice client: mic -> /ws/voice -> speaker (Phase 1.5).

Captures 16 kHz mono s16le mic audio with sounddevice, streams it to the
backend's /ws/voice endpoint, and plays returned TTS audio through the shared
persistent AudioOutput. UI concerns stay in the app; this class only reports
protocol events through the on_event callback.
"""

import asyncio
import json
import logging
from typing import Awaitable, Callable, Optional

import numpy as np
import sounddevice as sd
import websockets

from src.api.services.audio_output import AudioOutput

log = logging.getLogger("flet.voice")

SAMPLE_RATE = 16000
BLOCK_SAMPLES = 1600  # 100ms per mic block


class VoiceClient:
    """One live voice session against /ws/voice."""

    def __init__(self, url: str, on_event: Callable[[dict], Awaitable[None]]):
        self.url = url
        self.on_event = on_event
        self._ws = None
        self._stream: Optional[sd.InputStream] = None
        self._send_q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._tasks: list[asyncio.Task] = []
        self._out: Optional[AudioOutput] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Connect, start mic capture, and begin the session."""
        self._ws = await websockets.connect(self.url, max_size=None)
        await self._ws.send(json.dumps({"type": "start"}))

        loop = asyncio.get_running_loop()

        def _mic_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                log.debug(f"Mic status: {status}")
            pcm = (np.clip(indata[:, 0], -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            try:
                loop.call_soon_threadsafe(self._send_q.put_nowait, pcm)
            except (asyncio.QueueFull, RuntimeError):
                pass  # drop the block rather than stall the audio thread

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=BLOCK_SAMPLES,
            callback=_mic_callback,
        )
        self._stream.start()
        self._running = True
        self._tasks = [
            asyncio.create_task(self._sender()),
            asyncio.create_task(self._receiver()),
        ]
        log.info("Voice session started")

    async def stop(self) -> None:
        """End the session and release mic + speaker + socket."""
        self._running = False

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                log.warning("Mic close failed", exc_info=True)
            self._stream = None

        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "stop"}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks = []

        if self._out:
            self._out.flush()
            self._out.close()
            self._out = None
        log.info("Voice session stopped")

    # -- workers -----------------------------------------------------------------

    async def _sender(self) -> None:
        try:
            while self._running:
                data = await self._send_q.get()
                await self._ws.send(data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._running:
                log.error(f"Voice sender error: {e}")
                await self._emit({"type": "error", "message": f"connection lost: {e}"})

    async def _receiver(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, (bytes, bytearray)):
                    if self._out is not None:
                        arr = np.frombuffer(message, dtype=np.int16).astype(np.float32)
                        arr /= 32767.0
                        self._out.write(arr)
                    continue

                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    continue

                kind = event.get("type")
                if kind == "audio_start":
                    sr = int(event.get("sample_rate", 24000))
                    if self._out is None or self._out.sample_rate != sr:
                        if self._out:
                            self._out.close()
                        self._out = AudioOutput(sr)
                elif kind == "barge_in" and self._out:
                    self._out.flush()  # user interrupted: drop buffered speech

                await self._emit(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._running:
                log.error(f"Voice receiver error: {e}")
                await self._emit({"type": "error", "message": f"connection lost: {e}"})

    async def _emit(self, event: dict) -> None:
        try:
            await self.on_event(event)
        except Exception:
            log.warning("Voice event handler failed", exc_info=True)
