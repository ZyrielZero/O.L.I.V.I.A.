"""Pure ASGI middleware — avoids BaseHTTPMiddleware streaming issues."""

import logging
import time

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger("api.middleware")


class LoggingMiddleware:
    """Log requests with timing (pure ASGI)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Log method/path/status with timing around the wrapped app."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        t0 = time.perf_counter()
        log.info(f"-> {method} {path}")

        status_code = 0

        async def send_wrapper(msg):
            nonlocal status_code
            if msg["type"] == "http.response.start":
                status_code = msg.get("status", 0)
            await send(msg)

        await self.app(scope, receive, send_wrapper)

        ms = (time.perf_counter() - t0) * 1000
        log.info(f"<- {method} {path} {status_code} {ms:.1f}ms")


class ErrorHandlingMiddleware:
    """Catch unhandled exceptions (pure ASGI)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Run the wrapped app, returning a 500 JSON response on unhandled errors."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers_sent = False

        async def send_wrapper(msg):
            nonlocal headers_sent
            if msg["type"] == "http.response.start":
                headers_sent = True
            await send(msg)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            log.error(f"Unhandled: {e}", exc_info=True)
            if not headers_sent:
                resp = JSONResponse(
                    status_code=500,
                    content={"error": "Internal server error"},
                )
                await resp(scope, receive, send)
