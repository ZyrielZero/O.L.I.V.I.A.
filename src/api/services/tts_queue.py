"""TTS queue with overlapping synthesis and playback."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Tuple

log = logging.getLogger("api.tts_queue")

_SENTENCE_QUEUE_TIMEOUT = 2.0
_WORDS_PER_SECOND_FACTOR = 0.3  # extra synthesis timeout per word


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

    Workers block on `await queue.get()`. A None sentinel flows through both
    stages to signal end-of-input; stop() cancels the worker tasks outright.
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

        self._sent_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=self.cfg.max_sent_q)
        self._audio_q: asyncio.Queue[Optional[Tuple[str, Any]]] = asyncio.Queue(
            maxsize=self.cfg.max_audio_q
        )

        self._synth_task: Optional[asyncio.Task] = None
        self._play_task: Optional[asyncio.Task] = None
        self._stopped = False
        self._err: Optional[Exception] = None

        self._n_queued = 0
        self._n_synth = 0
        self._n_played = 0

    async def start(self) -> None:
        """Start workers."""
        self._stopped = False
        self._synth_task = asyncio.create_task(self._synth_loop())
        self._play_task = asyncio.create_task(self._play_loop())

    async def stop(self) -> None:
        """Stop immediately, cancelling workers and discarding pending work."""
        self._stopped = True

        for task in (self._synth_task, self._play_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        for q in (self._sent_q, self._audio_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

        log.debug(f"TTS queue stopped: {self._n_played}/{self._n_queued} played")

    async def queue_sentence(self, sent: str) -> None:
        """Queue a sentence for synthesis."""
        if self._err:
            raise RuntimeError(f"TTS queue failed: {self._err}")
        if self._stopped:
            return

        try:
            await asyncio.wait_for(self._sent_q.put(sent), timeout=_SENTENCE_QUEUE_TIMEOUT)
        except asyncio.TimeoutError:
            log.warning(f"Sentence queue full ({self._sent_q.qsize()}), waiting...")
            await self._sent_q.put(sent)
        self._n_queued += 1

    async def finish(self) -> None:
        """Signal end of input and wait for all audio to play."""
        await self._sent_q.put(None)  # sentinel
        if self._synth_task:
            await self._synth_task
        if self._play_task:
            await self._play_task
        log.info(f"TTS queue done: synth={self._n_synth}, played={self._n_played}/{self._n_queued}")

    async def _synth_loop(self) -> None:
        """Synthesis worker: sentences in, audio out."""
        try:
            while True:
                sent = await self._sent_q.get()
                if sent is None:  # sentinel: forward downstream and exit
                    await self._audio_q.put(None)
                    return

                try:
                    word_count = sent.count(" ") + 1
                    timeout = self.cfg.synth_timeout + word_count * _WORDS_PER_SECOND_FACTOR
                    audio = await asyncio.wait_for(self.synthesize_fn(sent), timeout=timeout)
                    self._n_synth += 1
                    if audio:
                        await self._audio_q.put((sent, audio))
                except asyncio.TimeoutError:
                    log.error(f"Synth timeout: {sent[:30]}...")
                except Exception as e:
                    log.error(f"Synth error: {e}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"Synth worker error: {e}")
            self._err = e
            await self._audio_q.put(None)

    async def _play_loop(self) -> None:
        """Playback worker: audio in, speakers out."""
        try:
            while True:
                item = await self._audio_q.get()
                if item is None:  # sentinel
                    return

                sent, audio = item
                try:
                    await asyncio.wait_for(
                        self.playback_fn(audio), timeout=self.cfg.playback_timeout
                    )
                    self._n_played += 1
                except asyncio.TimeoutError:
                    log.error(f"Playback timeout: {sent[:30]}...")
                except Exception as e:
                    log.error(f"Playback error: {e}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"Playback worker error: {e}")
            self._err = e

    async def __aenter__(self) -> "SentenceTTSQueue":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            await self.stop()
        elif self._play_task and not self._play_task.done():
            await self.finish()

    @property
    def pending_sentences(self) -> int:
        return self._sent_q.qsize()

    @property
    def pending_audio(self) -> int:
        return self._audio_q.qsize()
