"""Audio utils for websocket transport."""

import base64

import numpy as np


def encode_audio_to_base64(audio: bytes) -> str:
    """Encode PCM bytes to base64 for WS transport."""
    return base64.b64encode(audio).decode("utf-8")


def decode_base64_to_audio(b64: str) -> bytes:
    """Decode base64 to PCM bytes."""
    return base64.b64decode(b64)


def bytes_to_numpy_audio(audio: bytes, dtype=np.int16) -> np.ndarray:
    """PCM bytes to numpy array."""
    return np.frombuffer(audio, dtype=dtype)


def numpy_audio_to_float(arr: np.ndarray) -> np.ndarray:
    """int16 samples to float32 [-1, 1] for faster-whisper."""
    return arr.astype(np.float32) / 32768.0
