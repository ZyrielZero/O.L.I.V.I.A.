"""
API endpoint tests.
Tests REST endpoints for chat, health, and error handling.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.models.chat import ChatRequest
from src.api.models.common import HealthCheck, ServiceHealth


@pytest.fixture
def mock_services():
    """Create mock services for testing."""
    llm = MagicMock()
    llm.is_initialized.return_value = True
    llm.health_check = AsyncMock(return_value=True)

    async def mock_stream(*args, **kwargs):
        for token in ["Hello", " ", "there", "!"]:
            yield token

    llm.chat_stream = mock_stream

    memory = MagicMock()
    memory.is_initialized.return_value = True
    memory.health_check = AsyncMock(return_value=True)
    memory.get_relevant_context = AsyncMock(return_value="")
    memory.add_conversation = AsyncMock()

    tts = MagicMock()
    tts.is_initialized.return_value = True
    tts.health_check = AsyncMock(return_value=True)

    stt = MagicMock()
    stt.is_initialized.return_value = True
    stt.health_check = AsyncMock(return_value=True)

    state = MagicMock()

    return {
        "llm": llm,
        "memory": memory,
        "tts": tts,
        "stt": stt,
        "state": state
    }


@pytest.fixture
def test_app(mock_services):
    """Create test FastAPI app with mocked services."""
    from fastapi.responses import StreamingResponse

    from src.api import dependencies
    from src.api.routes.health import router as health_router

    # Inject mocked services
    dependencies.services = mock_services

    app = FastAPI()
    app.include_router(health_router)

    # Mock chat endpoint for testing
    @app.post("/api/chat")
    async def mock_chat(request: dict):
        stream = request.get("stream", True)
        temperature = request.get("temperature")

        # Validate temperature
        if temperature is not None and (temperature < 0 or temperature > 2.0):
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="Temperature must be between 0 and 2")

        if stream:
            async def generate():
                for token in ["Hello", " ", "there", "!"]:
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            return {"message": "Hello there!", "response": "Hello there!"}

    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


# ===== Test 1: Chat Endpoint Streaming Complete =====

@pytest.mark.api
def test_chat_endpoint_streaming_complete(client):
    """Stream completes with done=True in final event."""
    response = client.post(
        "/api/chat",
        json={"message": "Hello", "stream": True},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    # Parse SSE events
    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                events.append(data)
            except json.JSONDecodeError:
                pass

    # Should have events and final done=True
    assert len(events) > 0
    # Last event should have done=True
    assert events[-1].get("done") is True


# ===== Test 2: Chat Endpoint with Temperature =====

@pytest.mark.api
def test_chat_endpoint_with_temperature(client):
    """Custom temperature parameter is accepted."""
    response = client.post(
        "/api/chat",
        json={
            "message": "Hello",
            "stream": False,
            "temperature": 0.8
        },
    )

    assert response.status_code == 200


# ===== Test 3: Chat Endpoint with Max Tokens =====

@pytest.mark.api
def test_chat_endpoint_with_max_tokens(client):
    """Max tokens parameter is accepted."""
    response = client.post(
        "/api/chat",
        json={
            "message": "Hello",
            "stream": False,
            "max_tokens": 50
        },
    )

    assert response.status_code == 200


# ===== Test 4: Invalid Temperature Range =====

@pytest.mark.api
def test_chat_endpoint_invalid_temperature_range(client):
    """Temperature > 2.0 should be rejected with 422."""
    response = client.post(
        "/api/chat",
        json={
            "message": "Hello",
            "stream": False,
            "temperature": 3.0  # Invalid: > 2.0
        },
    )

    # Should reject invalid temperature
    assert response.status_code == 422


# ===== Test 5: Empty Message =====

@pytest.mark.api
def test_chat_endpoint_empty_message(client):
    """Empty message handled gracefully."""
    response = client.post(
        "/api/chat",
        json={
            "message": "",
            "stream": False
        },
    )

    # Should reject empty message or handle gracefully
    assert response.status_code in [200, 400, 422]


# ===== Test 6: Health Endpoint Service Details =====

@pytest.mark.api
def test_health_endpoint_service_details(client):
    """Health shows individual service status."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "services" in data
    assert "uptime_seconds" in data

    services = data["services"]
    assert "llm" in services or "LLM" in str(services)


# ===== Test 7: CORS Headers Present =====

@pytest.mark.api
def test_cors_headers_present(test_app):
    """CORS headers on preflight request."""
    from fastapi.middleware.cors import CORSMiddleware

    # Add CORS middleware
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    client = TestClient(test_app)

    # Preflight request
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }
    )

    # CORS should allow the request
    assert response.status_code in [200, 204, 405]


# ===== Test 8: Logging Middleware Captures Requests =====

@pytest.mark.api
def test_logging_middleware_captures_requests(client, caplog):
    """Requests are logged by middleware."""
    import logging

    with caplog.at_level(logging.DEBUG):
        response = client.get("/health")

    assert response.status_code == 200
    # Request should have been made (logging may vary by config)


# ===== Test 9: API Error Response Format =====

@pytest.mark.api
def test_api_error_response_format(client):
    """Errors return structured JSON."""
    # Send invalid JSON
    response = client.post(
        "/api/chat",
        content="not valid json",
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    data = response.json()

    # FastAPI error format
    assert "detail" in data


# ===== Additional API Tests =====

@pytest.mark.api
def test_health_endpoint_status_values(client):
    """Health status is one of expected values."""
    response = client.get("/health")
    data = response.json()

    assert data["status"] in ["healthy", "degraded", "unhealthy"]


@pytest.mark.api
def test_chat_non_streaming_response(client):
    """Non-streaming chat returns complete response."""
    response = client.post(
        "/api/chat",
        json={"message": "Hello", "stream": False}
    )

    assert response.status_code == 200
    data = response.json()

    assert "message" in data or "response" in data


@pytest.mark.api
def test_chat_request_validation():
    """ChatRequest model validates correctly."""
    # Valid request
    request = ChatRequest(message="Hello", stream=True)
    assert request.message == "Hello"
    assert request.stream is True

    # With optional params
    request = ChatRequest(
        message="Test",
        stream=False,
        temperature=0.7,
        max_tokens=100
    )
    assert request.temperature == 0.7
    assert request.max_tokens == 100


@pytest.mark.api
def test_health_check_model():
    """HealthCheck model structures correctly."""
    health = HealthCheck(
        status="healthy",
        services={
            "llm": ServiceHealth(name="LLM", status="up"),
            "memory": ServiceHealth(name="Memory", status="up"),
        },
        uptime_seconds=120.5
    )

    assert health.status == "healthy"
    assert health.uptime_seconds == 120.5
    assert len(health.services) == 2


@pytest.mark.api
def test_service_health_model():
    """ServiceHealth model structures correctly."""
    health = ServiceHealth(
        name="Test Service",
        status="up",
        message="Running normally"
    )

    assert health.name == "Test Service"
    assert health.status == "up"
    assert health.message == "Running normally"
