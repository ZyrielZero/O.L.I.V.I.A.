"""REST API client for FastAPI backend."""

import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

log = logging.getLogger("flet.api_client")


class OliviaAPIClient:
    """Async HTTP client for backend API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        # Read timeout 90s allows for slow LLM responses (Ollama backend uses 60s)
        # Prevents premature timeout during complex reasoning or GPU contention
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=30.0)
        )

    async def check_connection(self, max_retries: int = 30, retry_delay: float = 1.0) -> bool:
        """Check if backend is reachable with retry logic.

        Args:
            max_retries: Maximum number of connection attempts (default 30 = 30 seconds)
            retry_delay: Delay between retries in seconds

        Returns:
            True if backend is accessible, False otherwise
        """
        import asyncio

        for attempt in range(max_retries):
            try:
                response = await self.client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    log.info(f"Backend connected after {attempt + 1} attempt(s)")
                    return True
            except Exception as e:
                if attempt < max_retries - 1:
                    log.debug(f"Connection attempt {attempt + 1}/{max_retries} failed, retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    log.error(f"Connection check failed after {max_retries} attempts: {e}")

        return False

    async def get_health(self) -> Optional[Dict[str, Any]]:
        """Get health status of backend services.

        Returns:
            Health check response or None if failed
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Health check failed: {e}")
            return None

    async def send_message_stream(
        self, message: str, context: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Send text message and stream response tokens.

        Args:
            message: User message
            context: Optional context

        Yields:
            Response tokens
        """
        try:
            payload = {"message": message, "stream": True}
            if context:
                payload["context"] = context

            async with self.client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        try:
                            data = json.loads(data_str)
                            if data.get("done"):
                                break
                            if "token" in data:
                                yield data["token"]
                        except json.JSONDecodeError:
                            continue

        except httpx.TimeoutException as e:
            log.error(f"Message streaming timed out: {type(e).__name__}")
            yield "[Connection timed out]"
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error during streaming: {e.response.status_code}")
            yield f"[HTTP Error: {e.response.status_code}]"
        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            log.error(f"Message streaming failed: {error_msg}")
            yield f"[Error: {error_msg}]"

    async def send_message(self, message: str, context: Optional[str] = None) -> Optional[str]:
        """Send text message and get complete response (non-streaming).

        Args:
            message: User message
            context: Optional context

        Returns:
            Complete response or None if failed
        """
        try:
            payload = {"message": message, "stream": False}
            if context:
                payload["context"] = context

            response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message")

        except Exception as e:
            log.error(f"Message send failed: {e}")
            return None

    async def clear_history(self) -> bool:
        """Clear conversation history.

        Returns:
            True if successful, False otherwise
        """
        try:
            response = await self.client.delete(f"{self.base_url}/api/history")
            response.raise_for_status()
            return True
        except Exception as e:
            log.error(f"Clear history failed: {e}")
            return False

    async def close(self):
        """Close client connection."""
        await self.client.aclose()
