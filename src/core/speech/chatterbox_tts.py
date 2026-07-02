"""ChatterBox Turbo TTS Engine for O.L.I.V.I.A.
Zero-shot voice cloning with streaming support and sub-500ms latency.
"""

import logging
import queue
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import torch

log = logging.getLogger("ChatterBox")


@dataclass
class TTSMetrics:
    """Performance metrics for a single TTS generation."""

    ttfb_ms: float = 0.0
    total_generation_ms: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0
    chunks_generated: int = 0
    model_inference_ms: float = 0.0
    preprocessing_ms: float = 0.0
    text_length: int = 0
    # Optimization: Use perf_counter for higher precision timing (nanosecond resolution)
    # time.time() has ~1ms resolution, perf_counter has ~100ns resolution
    timestamp: float = field(default_factory=time.perf_counter)

    def __str__(self) -> str:
        return (
            f"TTSMetrics(TTFB={self.ttfb_ms:.0f}ms, "
            f"Total={self.total_generation_ms:.0f}ms, "
            f"RTF={self.rtf:.2f}x, "
            f"Chunks={self.chunks_generated}, "
            f"AudioDur={self.audio_duration_s:.1f}s)"
        )


@dataclass
class ChatterBoxConfig:
    """Configuration for ChatterBox TTS."""

    device: str = "cuda"
    voice_reference: str = "assets/voice/reference.wav"
    sample_rate: int = 24000
    chunk_size: int = 50  # Speech tokens per chunk
    # Voice cloning parameters (optimized defaults from research)
    exaggeration: float = 0.5  # Emotion intensity (0.25-2.0)
    cfg_weight: float = 0.5  # Guidance weight (0.2-1.0), lower for fast speakers
    # Quality settings
    seed: Optional[int] = None  # For reproducible output (None = random)
    preprocess_reference: bool = True  # Validate and preprocess reference audio
    enable_post_processing: bool = True  # Apply crossfades between chunks
    crossfade_ms: int = 10  # Crossfade duration in milliseconds
    # Audio callback for API streaming (Phase 1 optimization)
    audio_callback: Optional[Callable[[np.ndarray, int], None]] = (
        None  # (audio_chunk, sample_rate) -> None
    )
    # PyTorch optimizations (Phase 2)
    use_torch_compile: bool = True  # Apply torch.compile() for model acceleration
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", or "max-autotune"
    torch_dtype: str = "float32"  # Keep float32 for quality (no mixed precision)
    enable_inference_mode: bool = True  # Disable gradient computation for faster inference
    enable_cudnn_benchmark: bool = True  # Enable cuDNN autotuner
    enable_tf32: bool = True  # Enable TF32 for Ampere+ GPUs
    # Performance optimizations (Phase 4)
    adaptive_chunking: bool = True  # Use smaller first chunk for lower TTFB
    first_chunk_tokens: int = 30  # First chunk size for faster TTFB
    subsequent_chunk_tokens: int = 50  # Subsequent chunks for better quality
    # Monitoring (Phase 5)
    enable_metrics: bool = True  # Track performance metrics
    log_metrics: bool = False  # Log metrics to console
    memory_cleanup_interval: int = 50  # Clean CUDA memory every N generations


class AudioPlayer:
    """Non-blocking audio playback with stop capability and optional callback mode."""

    def __init__(
        self, sample_rate: int = 24000, callback: Optional[Callable[[np.ndarray, int], None]] = None
    ):
        self.sample_rate = sample_rate
        self.callback = callback  # If set, invoke callback instead of playing to speaker
        self._stop_flag = threading.Event()
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._playback_thread: Optional[threading.Thread] = None
        self._playing = False

    def start(self):
        """Start the playback thread."""
        # Join previous thread before starting a new one
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

        self._stop_flag.clear()
        self._playing = True

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True, name="ChatterBox-AudioPlayer"
        )
        self._playback_thread.start()

    def _playback_loop(self):
        """Background playback loop."""
        try:
            while not self._stop_flag.is_set():
                try:
                    chunk = self._audio_queue.get(timeout=0.1)
                    if chunk is None:  # Sentinel
                        break
                    if self._stop_flag.is_set():
                        break

                    # Callback mode: invoke callback instead of playing to speaker
                    if self.callback is not None:
                        try:
                            self.callback(chunk, self.sample_rate)
                        except Exception:
                            log.error("Audio callback error", exc_info=True)
                    else:
                        # Speaker mode: play chunk using sounddevice
                        sd.play(chunk, self.sample_rate)
                        sd.wait()

                except queue.Empty:
                    continue
        finally:
            self._playing = False

    def play_chunk(self, audio: np.ndarray):
        """Queue a chunk for playback."""
        if not self._stop_flag.is_set():
            try:
                self._audio_queue.put(audio, timeout=1.0)
            except queue.Full:
                log.warning("Audio queue full, dropping chunk")

    def stop(self):
        """Stop playback immediately."""
        self._stop_flag.set()
        # Only call sd.stop() in speaker mode — callback-mode players
        # should never touch sounddevice globally
        if self.callback is None:
            sd.stop()

        # Drain queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Send sentinel
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass

        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

        self._playing = False

    def finish(self):
        """Wait for playback to complete."""
        try:
            self._audio_queue.put(None, timeout=1.0)
        except queue.Full:
            pass

        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=60.0)

        self._playing = False

    def is_playing(self) -> bool:
        return self._playing


class ChatterBoxEngine:
    """ChatterBox Turbo TTS engine for O.L.I.V.I.A.

    Features:
    - Zero-shot voice cloning from reference audio
    - Streaming synthesis with sub-500ms latency
    - Barge-in support via stop()
    """

    def __init__(self, config: Optional[ChatterBoxConfig] = None, **kwargs):
        self.config = config or ChatterBoxConfig()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._model = None
        self._loaded = False
        self._player = AudioPlayer(
            sample_rate=self.config.sample_rate, callback=self.config.audio_callback
        )
        self._stop_flag = threading.Event()
        self._speak_thread: Optional[threading.Thread] = None
        self._speaking = False
        self._generation_count = 0  # For memory cleanup tracking (Phase 5)
        self._last_metrics: Optional[TTSMetrics] = None  # Last generation metrics
        self._compiled = False  # Whether torch.compile() actually succeeded
        self._last_error: Optional[Exception] = None

    def load_model(self) -> None:
        """Load the ChatterBox Turbo model with Phase 2 optimizations."""
        if self._loaded:
            log.info("ChatterBox already loaded")
            return

        try:
            # Suppress known deprecation warnings from chatterbox/diffusers/transformers
            warnings.filterwarnings(
                "ignore", message=r".*LoRACompatibleLinear.*deprecated.*", category=FutureWarning
            )
            warnings.filterwarnings(
                "ignore", message=r".*torch\.backends\.cuda\.sdp_kernel.*deprecated.*"
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*output_attentions=True.*not supported.*sdpa.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore", message=r".*Skipping import of cpp extensions.*incompatible.*"
            )
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            raise ImportError(
                "ChatterBox not installed. Install with:\n  pip install chatterbox-tts"
            )

        log.info(f"Loading ChatterBox Turbo on {self.config.device}...")

        self._enable_cuda_optimizations()
        self._model = ChatterboxTTS.from_pretrained(device=self.config.device)

        if hasattr(self._model, "sr"):
            self.config.sample_rate = self._model.sr
            self._player.sample_rate = self._model.sr

        self._apply_torch_compile()

        if hasattr(self._model, "eval"):
            self._model.eval()

        self._setup_voice_reference()
        self._run_warmup()
        self._loaded = True
        log.info(f"ChatterBox Turbo ready! Voice: {Path(self.config.voice_reference).name}")

    def _enable_cuda_optimizations(self) -> None:
        """Enable CUDA-specific optimizations for Ampere+ GPUs."""
        if not (torch.cuda.is_available() and self.config.device == "cuda"):
            return
        log.info("Enabling CUDA optimizations...")
        if self.config.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if self.config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _apply_torch_compile(self) -> None:
        """Apply torch.compile() for model acceleration.

        Auto-downgrades compile mode if VRAM is tight (>75% used after load).
        """
        if not (self.config.use_torch_compile and hasattr(torch, "compile")):
            return

        # Fall back to eager mode if compilation fails at runtime (e.g. no MSVC on Windows)
        torch._dynamo.config.suppress_errors = True

        mode = self.config.compile_mode
        # Check VRAM pressure — reduce-overhead uses extra memory for caching
        if torch.cuda.is_available() and self.config.device == "cuda":
            try:
                total = torch.cuda.get_device_properties(0).total_mem
                reserved = torch.cuda.memory_reserved()
                if reserved / total > 0.75 and mode == "reduce-overhead":
                    mode = "default"
                    log.info("VRAM pressure high, downgrading to compile mode: default")
            except Exception:
                pass

        try:
            log.info(f"Compiling TTS sub-components with mode: {mode}...")
            compile_start = time.perf_counter()
            compiled = 0
            if hasattr(self._model, "t3") and hasattr(self._model.t3, "inference"):
                self._model.t3.inference = torch.compile(
                    self._model.t3.inference, mode=mode, fullgraph=False, dynamic=True
                )
                compiled += 1
            if hasattr(self._model, "s3gen") and hasattr(self._model.s3gen, "mel2wav"):
                self._model.s3gen.mel2wav.inference = torch.compile(
                    self._model.s3gen.mel2wav.inference, mode=mode, fullgraph=False, dynamic=True
                )
                compiled += 1
            if compiled == 0:
                raise RuntimeError("No compilable sub-components found")
            self._compiled = True
            log.info(
                f"Compiled {compiled} sub-components in "
                f"{time.perf_counter() - compile_start:.1f}s"
            )
        except Exception as e:
            self._compiled = False
            log.warning(f"torch.compile() failed, using standard mode: {e}")

    def _run_warmup(self) -> None:
        """Run warmup pass to warm CUDA kernels (and torch.compile if active)."""
        if not (torch.cuda.is_available() and self.config.device == "cuda"):
            return
        try:
            log.info("Running warmup pass...")
            warmup_start = time.perf_counter()
            ref_path = getattr(self, "_processed_reference", str(self.config.voice_reference))
            with torch.inference_mode(mode=self.config.enable_inference_mode):
                if hasattr(self._model, "generate"):
                    try:
                        self._model.generate(
                            "This is a warmup sentence to prepare the voice engine for speaking.",
                            audio_prompt_path=ref_path,
                            exaggeration=self.config.exaggeration,
                            cfg_weight=self.config.cfg_weight,
                        )
                    except Exception:
                        pass
            # Only warmup streaming path if torch.compile actually succeeded
            if self._compiled and hasattr(self._model, "generate_stream"):
                try:
                    for _ in self._model.generate_stream(
                        "Warmup stream for compiled model.",
                        chunk_size=30,
                        audio_prompt_path=ref_path,
                        exaggeration=self.config.exaggeration,
                        cfg_weight=self.config.cfg_weight,
                    ):
                        break
                except Exception:
                    pass

            log.info(f"Warmup completed in {time.perf_counter() - warmup_start:.1f}s")
        except Exception as e:
            log.warning(f"Warmup failed (not critical): {e}")

    def _setup_voice_reference(self) -> None:
        """Validate and preprocess voice reference audio."""
        ref_path = Path(self.config.voice_reference)
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Voice reference not found: {ref_path}\nPlease provide a 5-15 second WAV file."
            )

        self._processed_reference = str(ref_path)
        if not self.config.preprocess_reference:
            return

        try:
            try:
                from .audio_processing import AudioQualityConfig, ReferenceAudioValidator
            except ImportError:
                from audio_processing import AudioQualityConfig, ReferenceAudioValidator

            quality_cfg = AudioQualityConfig(
                target_sample_rate=self.config.sample_rate,
                crossfade_duration_ms=self.config.crossfade_ms,
            )
            validator = ReferenceAudioValidator(quality_cfg)

            is_valid, warnings = validator.validate(str(ref_path))
            for warning in warnings:
                log.warning(f"Reference audio: {warning}")

            if not is_valid:
                raise ValueError(f"Reference audio validation failed: {ref_path}")

            self._processed_reference = validator.preprocess(str(ref_path))
            log.info(f"Using preprocessed reference: {self._processed_reference}")
        except ImportError:
            log.warning("audio_processing module not available, skipping preprocessing")
        except Exception as e:
            log.warning(f"Reference preprocessing failed, using original: {e}")

    def speak(self, text: str) -> None:
        """Synthesize and play speech with streaming.
        Non-blocking - returns immediately.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not text or not text.strip():
            return

        self.stop()

        self._stop_flag.clear()
        self._speaking = True
        self._speak_thread = threading.Thread(
            target=self._speak_streaming, args=(text.strip(),), daemon=True, name="ChatterBox-Speak"
        )
        self._speak_thread.start()

    def _speak_streaming(self, text: str) -> None:
        """Internal streaming synthesis with metrics tracking."""
        metrics = TTSMetrics(text_length=len(text))
        # Optimization: Use perf_counter for all timing measurements
        # perf_counter has nanosecond resolution vs time.time()'s millisecond resolution
        # Also cache the start time to avoid repeated time.perf_counter() calls
        generation_start = time.perf_counter()
        total_audio_samples = 0

        try:
            log.info(f"Synthesizing: '{text[:50]}...'")
            self._player.start()

            post_processor = self._create_post_processor()
            gen_kwargs = self._build_gen_kwargs()

            inference_start = time.perf_counter()
            with torch.inference_mode(mode=self.config.enable_inference_mode):
                if hasattr(self._model, "generate_stream"):
                    total_audio_samples = self._generate_streaming(
                        text, gen_kwargs, post_processor, metrics, generation_start
                    )
                else:
                    total_audio_samples = self._generate_non_streaming(
                        text, gen_kwargs, post_processor, metrics, generation_start
                    )

            self._finalize_metrics(metrics, inference_start, generation_start, total_audio_samples)

            if not self._stop_flag.is_set():
                self._player.finish()
            else:
                self._player.stop()

        except Exception as e:
            log.error("Streaming synthesis error", exc_info=True)
            self._last_error = e
            self._player.stop()
        finally:
            self._speaking = False

    def _create_post_processor(self):
        """Create post-processor for audio chunk processing."""
        if not self.config.enable_post_processing:
            return None
        try:
            try:
                from .audio_processing import AudioQualityConfig, OutputAudioProcessor
            except ImportError:
                from audio_processing import AudioQualityConfig, OutputAudioProcessor
            return OutputAudioProcessor(
                AudioQualityConfig(crossfade_duration_ms=self.config.crossfade_ms)
            )
        except ImportError:
            log.debug("audio_processing not available, skipping post-processing")
            return None

    def _build_gen_kwargs(self) -> dict:
        """Build generation kwargs for model inference."""
        kwargs = {
            "audio_prompt_path": getattr(
                self, "_processed_reference", str(self.config.voice_reference)
            ),
            "exaggeration": self.config.exaggeration,
            "cfg_weight": self.config.cfg_weight,
        }
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        return kwargs

    def _to_numpy(self, audio_chunk) -> np.ndarray:
        """Convert audio chunk to numpy array."""
        if isinstance(audio_chunk, torch.Tensor):
            return audio_chunk.squeeze().cpu().numpy()
        return np.array(audio_chunk)

    def _generate_streaming(
        self,
        text: str,
        gen_kwargs: dict,
        post_processor,
        metrics: TTSMetrics,
        generation_start: float,
    ) -> int:
        """Generate audio with streaming, return total samples."""
        total_samples = 0
        ttfb_recorded = False
        chunk_size = (
            self.config.first_chunk_tokens
            if self.config.adaptive_chunking
            else self.config.chunk_size
        )

        for chunk_idx, (audio_chunk, _) in enumerate(
            self._model.generate_stream(text, chunk_size=chunk_size, **gen_kwargs)
        ):
            if self._stop_flag.is_set():
                break

            if not ttfb_recorded:
                # Optimization: Use perf_counter for consistent high-precision timing
                metrics.ttfb_ms = (time.perf_counter() - generation_start) * 1000
                ttfb_recorded = True

            audio_np = self._to_numpy(audio_chunk)
            total_samples += len(audio_np)
            metrics.chunks_generated += 1

            if post_processor is not None:
                audio_np = post_processor.process_chunk(
                    audio_np, self.config.sample_rate, is_first=(chunk_idx == 0), is_final=False
                )

            if len(audio_np) > 0:
                self._player.play_chunk(audio_np)

            if self.config.adaptive_chunking and chunk_idx == 0:
                chunk_size = self.config.subsequent_chunk_tokens

        # Flush post-processor buffer
        if post_processor is not None:
            final_audio = post_processor.process_chunk(
                np.array([], dtype=np.float32),
                self.config.sample_rate,
                is_first=False,
                is_final=True,
            )
            if len(final_audio) > 0:
                self._player.play_chunk(final_audio)

        return total_samples

    def _generate_non_streaming(
        self,
        text: str,
        gen_kwargs: dict,
        post_processor,
        metrics: TTSMetrics,
        generation_start: float,
    ) -> int:
        """Generate audio without streaming, return total samples."""
        log.debug("Using non-streaming synthesis")
        wav = self._model.generate(text, **gen_kwargs)
        # Optimization: Use perf_counter for consistent high-precision timing
        metrics.ttfb_ms = (time.perf_counter() - generation_start) * 1000

        if wav is None:
            return 0

        audio_np = self._to_numpy(wav)
        metrics.chunks_generated = 1

        if post_processor is not None:
            audio_np = post_processor.process_chunk(
                audio_np, self.config.sample_rate, is_first=True, is_final=True
            )

        self._player.play_chunk(audio_np)
        return len(audio_np)

    def synthesize_to_numpy(self, text: str) -> tuple[np.ndarray, int]:
        """Pure synthesis — returns (audio_array, sample_rate) without playback.

        Used by CLI's overlapped TTS pipeline where synth and play run on separate threads.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not text or not text.strip():
            return np.array([], dtype=np.float32), self.config.sample_rate

        text = text.strip()
        metrics = TTSMetrics(text_length=len(text))
        gen_start = time.perf_counter()
        chunks: list[np.ndarray] = []

        gen_kwargs = self._build_gen_kwargs()
        inference_start = time.perf_counter()

        with torch.inference_mode(mode=self.config.enable_inference_mode):
            if hasattr(self._model, "generate_stream"):
                ttfb_recorded = False
                chunk_size = (
                    self.config.first_chunk_tokens
                    if self.config.adaptive_chunking
                    else self.config.chunk_size
                )
                for idx, (audio_chunk, _) in enumerate(
                    self._model.generate_stream(text, chunk_size=chunk_size, **gen_kwargs)
                ):
                    if not ttfb_recorded:
                        metrics.ttfb_ms = (time.perf_counter() - gen_start) * 1000
                        ttfb_recorded = True
                    chunks.append(self._to_numpy(audio_chunk))
                    metrics.chunks_generated += 1
                    if self.config.adaptive_chunking and idx == 0:
                        chunk_size = self.config.subsequent_chunk_tokens
            else:
                wav = self._model.generate(text, **gen_kwargs)
                metrics.ttfb_ms = (time.perf_counter() - gen_start) * 1000
                if wav is not None:
                    chunks.append(self._to_numpy(wav))
                    metrics.chunks_generated = 1

        total_samples = sum(len(c) for c in chunks)
        self._finalize_metrics(metrics, inference_start, gen_start, total_samples)

        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        return audio, self.config.sample_rate

    def _finalize_metrics(
        self,
        metrics: TTSMetrics,
        inference_start: float,
        generation_start: float,
        total_samples: int,
    ) -> None:
        """Calculate and store final metrics."""
        # Optimization: Cache current time to avoid two perf_counter() calls
        current_time = time.perf_counter()
        metrics.model_inference_ms = (current_time - inference_start) * 1000
        metrics.total_generation_ms = (current_time - generation_start) * 1000
        metrics.audio_duration_s = total_samples / self.config.sample_rate
        metrics.rtf = (
            (metrics.total_generation_ms / 1000) / metrics.audio_duration_s
            if metrics.audio_duration_s > 0
            else 0.0
        )

        self._last_metrics = metrics
        if self.config.enable_metrics and self.config.log_metrics:
            log.info(f"Performance: {metrics}")

        self._generation_count += 1
        if self._generation_count % self.config.memory_cleanup_interval == 0:
            # Only cleanup when VRAM pressure is high (>80% reserved)
            if torch.cuda.is_available() and self.config.device == "cuda":
                try:
                    total = torch.cuda.get_device_properties(0).total_mem
                    reserved = torch.cuda.memory_reserved()
                    if reserved / total > 0.8:
                        self._cleanup_cuda_memory()
                except Exception:
                    self._cleanup_cuda_memory()

    def speak_blocking(self, text: str) -> None:
        """Synthesize and play, blocking until complete."""
        self._last_error = None
        self.speak(text)
        while self.is_speaking():
            if self._speak_thread:
                self._speak_thread.join(timeout=0.1)
        if self._last_error:
            raise self._last_error

    def stop(self) -> None:
        """Stop speech immediately (barge-in)."""
        self._stop_flag.set()
        self._player.stop()

        if self._speak_thread and self._speak_thread.is_alive():
            self._speak_thread.join(timeout=1.0)

        self._speaking = False

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._speaking or self._player.is_playing()

    def get_metrics(self) -> Optional[TTSMetrics]:
        """Get metrics from the last generation (Phase 5)."""
        return self._last_metrics

    def _cleanup_cuda_memory(self) -> None:
        """Clean up CUDA memory (Phase 5)."""
        if torch.cuda.is_available() and self.config.device == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if self.config.log_metrics:
                    allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                    log.debug(
                        f"CUDA memory: {allocated_mb:.0f}MB allocated, {reserved_mb:.0f}MB reserved"
                    )
            except Exception as e:
                log.warning(f"CUDA memory cleanup failed: {e}")

    def close(self) -> None:
        """Clean up resources."""
        self.stop()
        self._cleanup_cuda_memory()
        self._loaded = False
        log.info("ChatterBox engine closed")


class TTSEngine(ChatterBoxEngine):
    """Compatibility wrapper matching the interface expected by gui_app.py.

    Usage:
        tts = TTSEngine(voice_reference="voice.wav")
        tts.load_model()
        tts.speak("Hello world")  # Non-blocking
        tts.speak_blocking("Hello")  # Blocking
        tts.stop()  # Interrupt
        tts.close()  # Cleanup
    """

    def __init__(
        self,
        voice_reference: Optional[str] = None,
        reference_text: Optional[str] = None,  # noqa: ARG002 - Ignored, ChatterBox doesn't need it
        device: str = "cuda",
        # Legacy params (ignored for compatibility)
        service_host: str = "localhost",  # noqa: ARG002
        service_port: int = 8080,  # noqa: ARG002
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
        stream_mode: str = "http",  # noqa: ARG002
        **kwargs,
    ):
        config = ChatterBoxConfig(
            device=device,
            voice_reference=voice_reference or "assets/voice/reference.wav",
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
        )
        super().__init__(config=config, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChatterBox Turbo Test")
    parser.add_argument("--text", default="Hello! This is a test of the ChatterBox Turbo engine.")
    parser.add_argument("--voice", default="voice_reference.wav")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Testing ChatterBox Turbo with voice: {args.voice}")
    tts = TTSEngine(voice_reference=args.voice, device=args.device)

    try:
        print("Loading model...")
        tts.load_model()
        print(f"Speaking: {args.text}")
        start = time.time()
        tts.speak_blocking(args.text)
        print(f"Done in {time.time() - start:.2f}s")
    except KeyboardInterrupt:
        print("\nInterrupted!")
        tts.stop()
    finally:
        tts.close()
