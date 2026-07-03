"""Unit tests for the Phase 1 voice pipeline pieces.

Covers: latency metrics (1.6), the persistent AudioOutput ring buffer (1.2),
the session-scoped TTS queue (1.1), and VoiceSession VAD/barge-in/pipeline
logic (1.3/1.4) with a deterministic fake VAD and WebSocket.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

# ===== 1.6: metrics =====


@pytest.mark.unit
def test_metrics_record_and_averages():
    from src.api.services.metrics import LatencyMetrics

    m = LatencyMetrics()
    for v in (100.0, 200.0, 300.0):
        m.record("llm_ttft_ms", v)
    m.record("nonexistent_stage", 50.0)  # ignored
    m.record("stt_ms", -5.0)  # negative ignored

    out = m.averages()
    assert out["llm_ttft_ms"]["avg_ms"] == 200.0
    assert out["llm_ttft_ms"]["n"] == 3
    assert "nonexistent_stage" not in out
    assert "stt_ms" not in out

    m.reset()
    assert m.averages() == {}


# ===== 1.2: AudioOutput ring buffer (no real device) =====


class _FakeStream:
    def __init__(self, **kw):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


@pytest.mark.unit
def test_audio_output_ring_buffer_order_and_flush():
    with patch("src.api.services.audio_output.sd.OutputStream", _FakeStream):
        from src.api.services.audio_output import AudioOutput

        out = AudioOutput(sample_rate=16000)
        out.write(np.full(300, 0.1, dtype=np.float32))
        out.write(np.full(300, 0.2, dtype=np.float32))
        assert out.pending_samples == 600
        assert not out.wait_drained(timeout=0)

        # Drive the device callback manually: 256-frame blocks
        block = np.zeros((256, 1), dtype=np.float32)
        out._callback(block, 256, None, None)
        assert np.allclose(block[:, 0], 0.1)  # first chunk plays first

        out._callback(block, 256, None, None)
        # crosses the chunk boundary: 44 samples of 0.1 then 0.2
        assert np.allclose(block[:44, 0], 0.1)
        assert np.allclose(block[44:, 0], 0.2)

        # flush drops the rest (barge-in)
        out.flush()
        assert out.pending_samples == 0
        assert out.wait_drained(timeout=0)

        out._callback(block, 256, None, None)
        assert np.allclose(block, 0.0)  # silence while idle
        out.close()


# ===== 1.1: session-scoped TTS queue =====


def _mock_tts():
    async def synthesize_f32(text):
        return np.zeros(10, dtype=np.float32)

    async def play_f32(audio):
        return None

    return SimpleNamespace(synthesize_f32=synthesize_f32, play_f32=play_f32)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_tts_queue_survives_across_requests():
    from src.api.services.tts_service import get_session_tts_queue, set_active_tts_queue

    set_active_tts_queue(None)
    tts = _mock_tts()
    try:
        q1 = await get_session_tts_queue(tts)
        q2 = await get_session_tts_queue(tts)
        assert q1 is q2, "queue must be shared across requests, not per-request"
        assert q1.is_running()

        # After a stop (barge-in/shutdown) a fresh queue is created
        await q1.stop()
        assert not q1.is_running()
        q3 = await get_session_tts_queue(tts)
        assert q3 is not q1
        await q3.stop()
    finally:
        set_active_tts_queue(None)


# ===== 1.3/1.4: VoiceSession =====


class _FakeWS:
    """Captures everything the session sends."""

    def __init__(self):
        self.events: list[dict] = []
        self.binary: list[bytes] = []

    async def send_text(self, text: str):
        self.events.append(json.loads(text))

    async def send_bytes(self, data: bytes):
        self.binary.append(data)

    def event_types(self):
        return [e["type"] for e in self.events]


def _fake_vad(tensor, sr):
    """Deterministic VAD: loud chunk => speech."""
    return 0.95 if float(np.abs(tensor.numpy()).mean()) > 0.05 else 0.01


def _loud_chunk():
    return np.full(512, 0.5, dtype=np.float32)


def _quiet_chunk():
    return np.zeros(512, dtype=np.float32)


def _stub_container():
    from src.api.container import get_container

    async def transcribe_numpy(arr):
        return "hello there"

    async def chat_stream(message, **kw):
        for tok in ["Well", " hey", " yourself", ",", " nice", " to", " hear", " you", "."]:
            yield tok

    async def synthesize_f32(text):
        return np.full(240, 0.1, dtype=np.float32)

    c = get_container()
    c.stt = SimpleNamespace(transcribe_numpy=transcribe_numpy, is_initialized=lambda: True)
    c.llm = SimpleNamespace(chat_stream=chat_stream)
    c.tts = SimpleNamespace(
        synthesize_f32=synthesize_f32,
        is_initialized=lambda: True,
        config=SimpleNamespace(sample_rate=24000),
    )
    return c


@pytest.mark.unit
@pytest.mark.asyncio
async def test_voice_session_full_utterance_flow():
    """Speech in -> transcript, tokens, audio frames, done."""
    from src.api.container import reset_container
    from src.api.routes.voice import SILENCE_END_S, VoiceSession

    reset_container()
    _stub_container()
    try:
        ws = _FakeWS()
        session = VoiceSession(ws)
        session._vad_model = _fake_vad

        for _ in range(15):  # ~0.5s of speech
            await session._process_chunk(_loud_chunk())
        assert "speech_start" in ws.event_types()

        silence_chunks = int(SILENCE_END_S / (512 / 16000)) + 2
        for _ in range(silence_chunks):
            await session._process_chunk(_quiet_chunk())

        assert session._response_task is not None
        await asyncio.wait_for(session._response_task, timeout=5)

        types = ws.event_types()
        assert "transcript_final" in types
        assert "token" in types
        assert "audio_start" in types
        assert "audio_end" in types
        assert "done" in types
        assert types.index("audio_start") < types.index("audio_end") < types.index("done")
        assert ws.binary, "TTS audio frames must be sent as binary"
        # audio is s16le at the announced rate
        start_evt = next(e for e in ws.events if e["type"] == "audio_start")
        assert start_evt["sample_rate"] == 24000
        assert start_evt["format"] == "s16le"
    finally:
        reset_container()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_voice_session_short_noise_is_discarded():
    """Sub-threshold blips must not reach STT."""
    from src.api.container import reset_container
    from src.api.routes.voice import SILENCE_END_S, VoiceSession

    reset_container()
    container = _stub_container()

    calls = []
    real = container.stt.transcribe_numpy

    async def counting(arr):
        calls.append(len(arr))
        return await real(arr)

    container.stt.transcribe_numpy = counting
    try:
        ws = _FakeWS()
        session = VoiceSession(ws)
        session._vad_model = _fake_vad

        for _ in range(3):  # ~0.1s — under MIN_SPEECH_S
            await session._process_chunk(_loud_chunk())
        for _ in range(int(SILENCE_END_S / (512 / 16000)) + 2):
            await session._process_chunk(_quiet_chunk())

        assert session._response_task is None
        assert not calls
    finally:
        reset_container()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_voice_session_barge_in_requires_confirmation():
    """During playback, one loud chunk must NOT interrupt; N in a row must."""
    from src.api.container import reset_container
    from src.api.routes.voice import BARGE_IN_CONFIRM_CHUNKS, VoiceSession

    reset_container()
    _stub_container()
    try:
        ws = _FakeWS()
        session = VoiceSession(ws)
        session._vad_model = _fake_vad
        session._tts_playing = True

        # A single confident chunk (residual echo) does not interrupt
        await session._process_chunk(_loud_chunk())
        await session._process_chunk(_quiet_chunk())  # confirmation resets
        assert "barge_in" not in ws.event_types()
        assert session._tts_playing

        # Sustained speech does
        for _ in range(BARGE_IN_CONFIRM_CHUNKS):
            await session._process_chunk(_loud_chunk())
        assert "barge_in" in ws.event_types()
        assert not session._tts_playing
        # capture resumes immediately for the interrupting utterance
        assert session._speaking
    finally:
        reset_container()
