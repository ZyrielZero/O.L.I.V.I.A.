"""Ollama LLM client - sync/async streaming."""

import json
from typing import AsyncGenerator, Dict, List, Optional

import httpx

try:
    from src.utils.logger import get_logger

    log = get_logger("llm")
except ImportError:
    import logging

    log = logging.getLogger("llm")


class OllamaConnectionError(Exception):
    """Ollama connection failed."""

    pass


# generation defaults
GEN_PARAMS = {
    "temperature": 0.3,
    "top_p": 0.7,
    "top_k": 15,
    "repeat_penalty": 1.3,
    "num_ctx": 4096,
    "num_predict": 100,
}

STOP_TOKENS = [
    # Llama 3.1 format
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|reserved_special_token",
    # ChatML format (olivia-finetuned)
    "<|im_start|>",
    "<|im_end|>",
    # General
    "\n\n\n",
]


class ConversationManager:
    """Manages conversation history and Ollama API."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful AI assistant.",
        model: str = "olivia-finetuned",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.host = host
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []

        # OPT: Scaled connection limits prevent request queuing - O(1) connection lookup
        # Higher keepalive count reduces TCP handshake overhead for repeated requests
        # max_connections scales with expected concurrent LLM streams
        self._client = httpx.AsyncClient(
            base_url=host,
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
        self._reset_history()

    def _reset_history(self):
        self.history = [{"role": "system", "content": self.system_prompt}]

    def clear_history(self):
        self._reset_history()

    def update_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        if self.history and self.history[0]["role"] == "system":
            self.history[0]["content"] = prompt

    def trim_history(self, keep: int = 20):
        """Trim history, preserve system prompt."""
        if len(self.history) > keep + 1:
            self.history = [self.history[0]] + self.history[-keep:]

    def _build_payload(
        self,
        user_input: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        self.history.append({"role": "user", "content": user_input})
        # Optimization: Build message list directly instead of .copy()
        # list() shallow copy is slightly faster than .copy() method call
        # and we construct with context injection inline when needed
        n_tok = max_tokens if max_tokens else (250 if context else GEN_PARAMS["num_predict"])

        if context:
            # Build list with context message inserted before the last (user) message
            msgs = self.history[:-1] + [
                {"role": "system", "content": f"[Background: {context}] Reply briefly."},
                self.history[-1],
            ]
        else:
            msgs = list(self.history)

        return {
            "model": self.model,
            "messages": msgs,
            "stream": True,
            "options": {
                "temperature": temperature if temperature else GEN_PARAMS["temperature"],
                "top_p": GEN_PARAMS["top_p"],
                "top_k": GEN_PARAMS["top_k"],
                "repeat_penalty": GEN_PARAMS["repeat_penalty"],
                "num_ctx": GEN_PARAMS["num_ctx"],
                "num_predict": n_tok,
                "stop": STOP_TOKENS,
            },
        }

    async def chat_stream_async(
        self,
        user_input: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama (native async, no thread overhead)."""
        payload = self._build_payload(user_input, context, temperature, max_tokens)
        # Optimization: Use list append + join instead of string += concatenation
        # String += is O(n) per append due to immutability, list append is O(1) amortized
        resp_chunks: List[str] = []

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "message" in data:
                            tok = data["message"].get("content", "")
                            if tok:
                                yield tok
                                resp_chunks.append(tok)
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

            if resp_chunks:
                self.history.append({"role": "assistant", "content": "".join(resp_chunks)})
                self.trim_history()

        except httpx.ConnectError:
            raise OllamaConnectionError("Ollama not reachable")
        except httpx.TimeoutException as e:
            raise OllamaConnectionError(f"Ollama timeout: {e}")
        except Exception as e:
            log.error(f"Chat stream error: {e}")
            raise

    async def close(self):
        await self._client.aclose()


# Module-level reusable sync client for health checks
_sync_health_client: Optional[httpx.Client] = None


async def check_ollama_connection_async(host: str = "http://localhost:11434") -> bool:
    """Async check if Ollama running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{host}/api/tags")
            return r.status_code == 200
    except (httpx.RequestError, OSError):
        return False


def check_ollama_connection(host: str = "http://localhost:11434") -> bool:
    """Sync check if Ollama running (reuses httpx client)."""
    global _sync_health_client
    try:
        if _sync_health_client is None:
            _sync_health_client = httpx.Client(timeout=5.0)
        r = _sync_health_client.get(f"{host}/api/tags")
        return r.status_code == 200
    except (httpx.RequestError, OSError):
        return False


async def chat_simple_async(
    msg: str,
    system: str = "You are a helpful assistant.",
    model: str = "olivia-finetuned",
    host: str = "http://localhost:11434",
    max_tok: int = 150,
) -> str:
    """One-off chat (non-streaming)."""
    try:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": msg}],
            "stream": False,
            "options": {
                "temperature": GEN_PARAMS["temperature"],
                "top_p": GEN_PARAMS["top_p"],
                "top_k": GEN_PARAMS["top_k"],
                "repeat_penalty": GEN_PARAMS["repeat_penalty"],
                "num_predict": max_tok,
                "stop": STOP_TOKENS,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(f"{host}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
    except Exception as e:
        log.error(f"chat_simple_async failed: {e}")
        return ""
