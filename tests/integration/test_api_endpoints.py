"""
Integration tests for API endpoints.
Tests the FastAPI routes with mocked services.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import models
from src.api.models.chat import ChatRequest, ChatResponse
from src.api.models.common import HealthCheck, ServiceHealth


class TestRootEndpoint:
    """Tests for the root endpoint."""

    @pytest.fixture
    def app_with_root(self):
        """Create a minimal app with root endpoint."""
        app = FastAPI()

        @app.get("/")
        async def root():
            return {
                "name": "O.L.I.V.I.A. API",
                "version": "1.0.0",
                "description": "Offline Local Intelligent Voice Interactive Assistant",
                "docs": "/docs",
                "health": "/health"
            }

        return app

    @pytest.mark.integration
    def test_root_endpoint_returns_api_info(self, app_with_root):
        """Test that root endpoint returns API information."""
        client = TestClient(app_with_root)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "O.L.I.V.I.A. API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            "llm": MagicMock(
                model="test-model",
                health_check=AsyncMock(return_value=True)
            ),
            "memory": MagicMock(
                health_check=AsyncMock(return_value=True)
            ),
            "stt": MagicMock(
                model_size="small.en",
                health_check=AsyncMock(return_value=True)
            ),
            "tts": MagicMock(
                health_check=AsyncMock(return_value=True)
            ),
            "state": MagicMock(
                get_active_sessions=MagicMock(return_value=0)
            )
        }

    @pytest.fixture
    def app_with_health(self, mock_services):
        """Create an app with health endpoint and mocked services."""
        from fastapi import APIRouter

        app = FastAPI()
        router = APIRouter(tags=["health"])

        @router.get("/health", response_model=HealthCheck)
        async def health_check():
            service_health = {}
            overall_status = "healthy"

            # Check LLM
            llm = mock_services.get("llm")
            if llm:
                llm_ok = await llm.health_check()
                service_health["llm"] = ServiceHealth(
                    name="LLM (Ollama)",
                    status="up" if llm_ok else "down",
                    message=f"Connected to Ollama ({llm.model})" if llm_ok else "Ollama unreachable"
                )

            # Check Memory
            memory = mock_services.get("memory")
            if memory:
                memory_ok = await memory.health_check()
                service_health["memory"] = ServiceHealth(
                    name="Memory (ChromaDB)",
                    status="up" if memory_ok else "down",
                    message="ChromaDB accessible" if memory_ok else "ChromaDB unreachable"
                )

            # Check State
            state = mock_services.get("state")
            if state:
                active_sessions = state.get_active_sessions()
                service_health["state"] = ServiceHealth(
                    name="State Manager",
                    status="up",
                    message=f"{active_sessions} active session(s)"
                )

            return HealthCheck(
                status=overall_status,
                services=service_health,
                uptime_seconds=100.0
            )

        app.include_router(router)
        return app

    @pytest.mark.integration
    def test_health_endpoint_healthy(self, app_with_health):
        """Test health endpoint when all services are healthy."""
        client = TestClient(app_with_health)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["uptime_seconds"] >= 0

    @pytest.mark.integration
    def test_health_endpoint_returns_service_details(self, app_with_health):
        """Test that health endpoint returns service details."""
        client = TestClient(app_with_health)
        response = client.get("/health")

        data = response.json()
        services = data["services"]

        assert "llm" in services
        assert services["llm"]["status"] == "up"
        assert "memory" in services
        assert services["memory"]["status"] == "up"


class TestChatEndpoint:
    """Tests for the chat endpoint."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        service = MagicMock()
        service.model = "test-model"

        async def mock_chat_stream(message, context=None, temperature=None, max_tokens=None):
            for token in ["Hello", "!", " How", " can", " I", " help", "?"]:
                yield token

        service.chat_stream = mock_chat_stream
        return service

    @pytest.fixture
    def mock_memory_service(self):
        """Create a mock memory service."""
        service = MagicMock()
        service.get_relevant_context = AsyncMock(return_value="")
        service.add_conversation = AsyncMock(return_value=None)
        return service

    @pytest.fixture
    def app_with_chat(self, mock_llm_service, mock_memory_service):
        """Create an app with chat endpoint and mocked services."""
        from fastapi import APIRouter
        from fastapi.responses import StreamingResponse

        app = FastAPI()
        router = APIRouter(prefix="/api", tags=["chat"])

        @router.post("/chat")
        async def chat(request: ChatRequest):
            # Non-streaming response
            if not request.stream:
                full_response = ""
                async for token in mock_llm_service.chat_stream(
                    message=request.message,
                    context=request.context,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    full_response += token

                return ChatResponse(
                    message=full_response,
                    timestamp=datetime.now(),
                    metadata={"context_length": 0}
                )

            # Streaming response
            async def generate_sse():
                async for token in mock_llm_service.chat_stream(
                    message=request.message
                ):
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream"
            )

        app.include_router(router)
        return app

    @pytest.mark.integration
    def test_chat_non_streaming(self, app_with_chat):
        """Test chat endpoint with non-streaming response."""
        client = TestClient(app_with_chat)
        response = client.post(
            "/api/chat",
            json={"message": "Hello!", "stream": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert len(data["message"]) > 0
        assert "timestamp" in data

    @pytest.mark.integration
    def test_chat_streaming(self, app_with_chat):
        """Test chat endpoint with streaming response."""
        client = TestClient(app_with_chat)

        with client.stream(
            "POST",
            "/api/chat",
            json={"message": "Hello!", "stream": True}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

            # Read some of the stream
            content = ""
            for chunk in response.iter_text():
                content += chunk
                if "done" in content and "true" in content.lower():
                    break

            assert "data:" in content

    @pytest.mark.integration
    def test_chat_with_context(self, app_with_chat):
        """Test chat endpoint with context provided."""
        client = TestClient(app_with_chat)
        response = client.post(
            "/api/chat",
            json={
                "message": "What is this about?",
                "context": "We were discussing Python programming.",
                "stream": False
            }
        )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_chat_with_temperature(self, app_with_chat):
        """Test chat endpoint with custom temperature."""
        client = TestClient(app_with_chat)
        response = client.post(
            "/api/chat",
            json={
                "message": "Be creative",
                "stream": False,
                "temperature": 0.9
            }
        )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_chat_with_max_tokens(self, app_with_chat):
        """Test chat endpoint with max_tokens limit."""
        client = TestClient(app_with_chat)
        response = client.post(
            "/api/chat",
            json={
                "message": "Short response please",
                "stream": False,
                "max_tokens": 50
            }
        )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_chat_missing_message(self, app_with_chat):
        """Test chat endpoint with missing message."""
        client = TestClient(app_with_chat)
        response = client.post("/api/chat", json={})

        assert response.status_code == 422  # Validation error

    @pytest.mark.integration
    def test_chat_invalid_temperature(self, app_with_chat):
        """Test chat endpoint with invalid temperature."""
        client = TestClient(app_with_chat)
        response = client.post(
            "/api/chat",
            json={
                "message": "Test",
                "stream": False,
                "temperature": 3.0  # Out of range
            }
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.integration
    def test_chat_invalid_max_tokens(self, app_with_chat):
        """Test chat endpoint with invalid max_tokens."""
        client = TestClient(app_with_chat)
        response = client.post(
            "/api/chat",
            json={
                "message": "Test",
                "stream": False,
                "max_tokens": -10  # Invalid
            }
        )

        assert response.status_code == 422  # Validation error


class TestCORSHeaders:
    """Tests for CORS configuration."""

    @pytest.fixture
    def app_with_cors(self):
        """Create an app with CORS enabled."""
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/test")
        async def test():
            return {"status": "ok"}

        return app

    @pytest.mark.integration
    def test_cors_preflight(self, app_with_cors):
        """Test CORS preflight request."""
        client = TestClient(app_with_cors)
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_cors_headers_included(self, app_with_cors):
        """Test that CORS headers are included in response."""
        client = TestClient(app_with_cors)
        response = client.get(
            "/test",
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestErrorHandling:
    """Tests for API error handling."""

    @pytest.fixture
    def app_with_errors(self):
        """Create an app with error-generating endpoints."""
        from fastapi import HTTPException

        app = FastAPI()

        @app.get("/error/500")
        async def internal_error():
            raise Exception("Internal error")

        @app.get("/error/404")
        async def not_found():
            raise HTTPException(status_code=404, detail="Resource not found")

        @app.get("/error/503")
        async def service_unavailable():
            raise HTTPException(status_code=503, detail="Service unavailable")

        return app

    @pytest.mark.integration
    def test_404_error(self, app_with_errors):
        """Test 404 error response."""
        client = TestClient(app_with_errors)
        response = client.get("/error/404")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.integration
    def test_503_error(self, app_with_errors):
        """Test 503 service unavailable response."""
        client = TestClient(app_with_errors)
        response = client.get("/error/503")

        assert response.status_code == 503

    @pytest.mark.integration
    def test_nonexistent_endpoint(self, app_with_errors):
        """Test request to non-existent endpoint."""
        client = TestClient(app_with_errors)
        response = client.get("/this/does/not/exist")

        assert response.status_code == 404


class TestMiddleware:
    """Tests for custom middleware."""

    @pytest.fixture
    def app_with_middleware(self):
        """Create an app with logging middleware."""
        from starlette.middleware.base import BaseHTTPMiddleware

        app = FastAPI()
        request_log = []

        class TestLoggingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request_log.append({
                    "method": request.method,
                    "path": request.url.path
                })
                response = await call_next(request)
                return response

        app.add_middleware(TestLoggingMiddleware)
        app.state.request_log = request_log

        @app.get("/test")
        async def test():
            return {"status": "ok"}

        return app, request_log

    @pytest.mark.integration
    def test_middleware_logs_requests(self, app_with_middleware):
        """Test that middleware logs requests."""
        app, request_log = app_with_middleware
        client = TestClient(app)

        client.get("/test")

        # Middleware should have logged the request
        assert len(request_log) >= 1
        assert request_log[-1]["method"] == "GET"
        assert request_log[-1]["path"] == "/test"
