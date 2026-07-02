"""TTS queue with overlapping synthesis and playback."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Tuple

log = logging.getLogger("api.tts_queue")

# OPT: Pre-computed constants avoid repeated calculations
_QUEUE_WAIT_TIMEOUT = 0.1  # Reduced from 0.5s for faster responsiveness
_SENTENCE_QUEUE_TIMEOUT = 2.0
_WORDS_PER_SECOND_FACTOR = 0.3  # seconds per word for dynamic timeout


@dataclass
class TTSQueueConfig:
    """TTS queue config."""

    max_sent_q: int = 20
    max_audio_q: int = 5
    synth_timeout: float = 20.0  # base, dynamic adds more
    playback_timeout: float = 60.0


class SentenceTTSQueue:
    """Two-stage queue: synthesis -> playback (overlapped).

    Sentence Queue -> [Synth Worker] -> Audio Queue -> [Playback Worker] -> Speakers.
    """

    def __init__(
        self,
        synthesize_fn: Callable[[str], Awaitable[Any]],
        playback_fn: Callable[[Any], Awaitable[None]],
        cfg: Optional[TTSQueueConfig] = None,
    ):
        self.synthesize_fn = synthesize_fn
        self.playback_fn = playback_fn
        self.cfg = cfg or TTSQueueConfig()

        # OPT: Cache config values to avoid repeated attribute lookups
        self._synth_timeout_base = self.cfg.synth_timeout
        self._playback_timeout = self.cfg.playback_timeout

        self._sent_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=self.cfg.max_sent_q)
        self._audio_q: asyncio.Queue[Optional[Tuple[str, Any]]] = asyncio.Queue(
            maxsize=self.cfg.max_audio_q
        )

        self._synth_task: Optional[asyncio.Task] = None
        self._play_task: Optional[asyncio.Task] = None

        self._stop = asyncio.Event()
        self._synth_done = asyncio.Event()
        self._play_done = asyncio.Event()
        # OPT: Event for data availability - avoids polling timeout loops
        self._sent_available = asyncio.Event()
        self._audio_available = asyncio.Event()
        self._err: Optional[Exception] = None

        self._n_queued = 0
        self._n_synth = 0
        self._n_played = 0

    async def start(self) -> None:
        """Start workers."""
        self._stop.clear()
        self._synth_done.clear()
        self._play_done.clear()
        self._sent_available.clear()
        self._audio_available.clear()

        self._synth_task = asyncio.create_task(self._synth_loop())
        self._play_task = asyncio.create_task(self._play_loop())

    async def stop(self) -> None:
        """Stop and cancel pending work."""
        self._stop.set()

        for q in [self._sent_q, self._audio_q]:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

        for t in [self._synth_task, self._play_task]:
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        log.debug(f"TTS queue stopped: {self._n_played}/{self._n_queued} played")

    async def queue_sentence(self, sent: str) -> None:
        """Queue sentence for synthesis (non-blocking).

        OPT: Signals event after queueing to wake blocked consumers immediately.
        """
        if self._err:
            raise RuntimeError(f"TTS queue failed: {self._err}")

        if self._stop.is_set():
            return

        try:
            await asyncio.wait_for(self._sent_q.put(sent), timeout=_SENTENCE_QUEUE_TIMEOUT)
            self._n_queued += 1
            # OPT: Signal availability - wakes waiting consumer immediately
            self._sent_available.set()
        except asyncio.TimeoutError:
            log.warning(f"Sentence queue full ({self._sent_q.qsize()}), waiting...")
            await self._sent_q.put(sent)
            self._n_queued += 1
            self._sent_available.set()

    async def finish(self) -> None:
        """Wait for all audio to play."""
        await self._sent_q.put(None)  # sentinel
        self._sent_available.set()  # OPT: Wake consumer for sentinel
        await self._synth_done.wait()
        await self._play_done.wait()
        log.info(f"TTS queue done: synth={self._n_synth}, played={self._n_played}/{self._n_queued}")

    async def _synth_loop(self) -> None:
        """Synthesis worker.

        OPT: Uses event-based waiting with short timeout fallback.
        Event signaling reduces average wait time from timeout/2 to near-zero.
        """
        # OPT: Cache method references and values for hot loop
        synth_fn = self.synthesize_fn
        synth_timeout_base = self._synth_timeout_base
        stop_is_set = self._stop.is_set
        sent_q_get_nowait = self._sent_q.get_nowait
        audio_q_put = self._audio_q.put

        try:
            while not stop_is_set():
                # OPT: Wait for event signal OR short timeout (for stop check)
                try:
                    await asyncio.wait_for(self._sent_available.wait(), timeout=_QUEUE_WAIT_TIMEOUT)
                except asyncio.TimeoutError:
                    continue

                # Process all available items
                while not self._sent_q.empty() and not stop_is_set():
                    try:
                        # OPT: get_nowait() avoids await overhead when we know item exists
                        sent = sent_q_get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    if sent is None:  # sentinel
                        self._sent_available.clear()
                        # Signal audio queue and exit
                        await audio_q_put(None)
                        self._audio_available.set()
                        self._synth_done.set()
                        return

                    try:
                        # OPT: Pre-compute timeout with cached base value
                        word_count = sent.count(" ") + 1  # faster than len(split())
                        timeout = synth_timeout_base + word_count * _WORDS_PER_SECOND_FACTOR
                        audio = await asyncio.wait_for(synth_fn(sent), timeout=timeout)
                        self._n_synth += 1

                        if audio:
                            await audio_q_put((sent, audio))
                            # OPT: Signal audio availability
                            self._audio_available.set()

                    except asyncio.TimeoutError:
                        log.error(f"Synth timeout: {sent[:30]}...")
                    except Exception as e:
                        log.error(f"Synth error: {e}")

                # Clear event if queue is empty (will be re-set on next queue)
                if self._sent_q.empty():
                    self._sent_available.clear()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Synth worker error: {e}")
            self._err = e
        finally:
            await self._audio_q.put(None)
            self._audio_available.set()
            self._synth_done.set()

    async def _play_loop(self) -> None:
        """Playback worker.

        OPT: Uses event-based waiting with short timeout fallback.
        """
        # OPT: Cache method references and values for hot loop
        playback_fn = self.playback_fn
        playback_timeout = self._playback_timeout
        stop_is_set = self._stop.is_set
        audio_q_get_nowait = self._audio_q.get_nowait

        try:
            while not stop_is_set():
                # OPT: Wait for event signal OR short timeout (for stop check)
                try:
                    await asyncio.wait_for(
                        self._audio_available.wait(), timeout=_QUEUE_WAIT_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    continue

                # Process all available items
                while not self._audio_q.empty() and not stop_is_set():
                    try:
                        # OPT: get_nowait() avoids await overhead when we know item exists
                        item = audio_q_get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    if item is None:  # sentinel
                        self._audio_available.clear()
                        self._play_done.set()
                        return

                    # OPT: Direct tuple unpacking
                    sent, audio = item

                    try:
                        await asyncio.wait_for(playback_fn(audio), timeout=playback_timeout)
                        self._n_played += 1
                    except asyncio.TimeoutError:
                        log.error(f"Playback timeout: {sent[:30]}...")
                    except Exception as e:
                        log.error(f"Playback error: {e}")

                # Clear event if queue is empty
                if self._audio_q.empty():
                    self._audio_available.clear()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Playback worker error: {e}")
            self._err = e
        finally:
            self._play_done.set()

    async def __aenter__(self) -> "SentenceTTSQueue":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            await self.stop()
        elif not self._play_done.is_set():
            await self.finish()

    @property
    def pending_sentences(self) -> int:
        return self._sent_q.qsize()

    @property
    def pending_audio(self) -> int:
        return self._audio_q.qsize()
