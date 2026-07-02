"""
O.L.I.V.I.A. Unified Launcher
Starts both FastAPI backend and Flet desktop UI.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description="Launch O.L.I.V.I.A. components")
    parser.add_argument("--api-only", action="store_true", help="Start only FastAPI backend")
    parser.add_argument("--ui-only", action="store_true", help="Start only Flet UI")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload for FastAPI")
    args = parser.parse_args()

    # Ollama performance optimizations
    os.environ.setdefault(
        "OLLAMA_FLASH_ATTENTION", "1"
    )  # Enable flash attention for faster inference
    os.environ.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")  # 8-bit KV cache for memory efficiency
    os.environ.setdefault("OLLAMA_KEEP_ALIVE", "-1")  # Keep model loaded indefinitely
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")  # Prevent model competition for VRAM

    processes = []
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def wait_for_backend(max_wait: float = 10.0, url: str = "http://localhost:8000/health") -> bool:
        """Poll health endpoint with exponential backoff.

        OPT: O(log n) attempts where n = max_wait * 2, exponential backoff
        reduces unnecessary requests while ensuring fast startup detection.

        Args:
            max_wait: Maximum wait time in seconds
            url: Health endpoint URL

        Returns:
            True if backend is healthy, False if timeout
        """
        max_attempts = int(max_wait * 2)  # ~0.5s initial interval
        for i in range(max_attempts):
            try:
                with urlopen(url, timeout=1.0) as resp:
                    if resp.status == 200:
                        return True
            except (URLError, OSError):
                pass
            # OPT: Exponential backoff capped at 1.5s - O(log n) convergence
            # Initial: 0.5s, then 0.6s, 0.72s, ... caps at 1.5s
            sleep_time = min(0.5 * (1.2 ** min(i, 5)), 1.5)
            time.sleep(sleep_time)
        return False

    def shutdown_all():
        """Gracefully shutdown all processes."""
        print("\n")
        print("=" * 70)
        print("  [STOP] Shutting down O.L.I.V.I.A...")
        print("=" * 70)
        for name, process in processes:
            if process.poll() is None:  # Still running
                print(f"  Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force killing {name}...")
                    process.kill()
                    process.wait()
        print("  [OK] All services stopped")
        print()

    print("=" * 70)
    print("  O.L.I.V.I.A. - Offline Local Intelligent Voice Interactive Assistant")
    print("=" * 70)
    print()

    try:
        # Start FastAPI backend
        if not args.ui_only:
            print("[*] Starting FastAPI backend...")
            api_cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ]
            if not args.no_reload:
                api_cmd.append("--reload")

            api_process = subprocess.Popen(api_cmd)
            processes.append(("FastAPI", api_process))
            print("[OK] FastAPI backend starting on http://localhost:8000")
            print("  API Docs: http://localhost:8000/docs")
            print()

            # Wait for backend to start
            if not args.api_only:
                print("[...] Waiting for backend to initialize...")
                if wait_for_backend(max_wait=30.0):
                    print("[OK] Backend is healthy")
                else:
                    print("[!] Backend health check timed out, continuing anyway...")
                print()

        # Start Flet UI
        if not args.api_only:
            print("[*] Starting Flet desktop UI...")
            ui_process = subprocess.Popen([sys.executable, "-m", "src.flet_app.main"])
            processes.append(("Flet UI", ui_process))
            print("[OK] Flet desktop application launched")
            print()

        print("=" * 70)
        print("  O.L.I.V.I.A. is running")
        print("  Press Ctrl+C or close the window to stop all services")
        print("=" * 70)
        print()

        # Monitor processes - exit when either terminates or shutdown signal received
        while not shutdown_event.is_set():
            for name, process in processes:
                if process.poll() is not None:
                    # A process exited, trigger shutdown
                    print(f"\n  {name} exited, shutting down other services...")
                    shutdown_event.set()
                    break
            time.sleep(0.5)

    finally:
        shutdown_all()


if __name__ == "__main__":
    main()
