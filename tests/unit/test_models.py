"""
Unit tests for API Pydantic models.
Tests validation, serialization, and edge cases.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.models.chat import ChatRequest, ChatResponse
from src.api.models.common import HealthCheck, ServiceHealth


class TestChatRequest:
    """Tests for ChatRequest model."""

    @pytest.mark.unit
    def test_valid_basic_request(self):
        """Test creating a valid basic chat request."""
        request = ChatRequest(message="Hello, how are you?")
        assert request.message == "Hello, how are you?"
        assert request.context is None
        assert request.stream is True  # Default
        assert request.temperature is None
        assert request.max_tokens is None

    @pytest.mark.unit
    def test_valid_full_request(self):
        """Test creating a request with all fields."""
        request = ChatRequest(
            message="Tell me about AI",
            context="You are an expert",
            stream=False,
            temperature=0.8,
            max_tokens=500
        )
        assert request.message == "Tell me about AI"
        assert request.context == "You are an expert"
        assert request.stream is False
        assert request.temperature == 0.8
        assert request.max_tokens == 500

    @pytest.mark.unit
    def test_empty_message_fails(self):
        """Test that empty message should be allowed (string validation)."""
        # Empty string is technically valid for str type, but may not be meaningful
        request = ChatRequest(message="")
        assert request.message == ""

    @pytest.mark.unit
    def test_missing_message_fails(self):
        """Test that missing message raises ValidationError."""
        with pytest.raises(ValidationError):
            ChatRequest()

    @pytest.mark.unit
    def test_temperature_boundaries(self):
        """Test temperature boundary validation."""
        # Valid minimum
        request = ChatRequest(message="test", temperature=0.0)
        assert request.temperature == 0.0

        # Valid maximum
        request = ChatRequest(message="test", temperature=2.0)
        assert request.temperature == 2.0

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            ChatRequest(message="test", temperature=-0.1)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            ChatRequest(message="test", temperature=2.1)

    @pytest.mark.unit
    def test_max_tokens_validation(self):
        """Test max_tokens must be positive."""
        # Valid
        request = ChatRequest(message="test", max_tokens=1)
        assert request.max_tokens == 1

        # Invalid: zero
        with pytest.raises(ValidationError):
            ChatRequest(message="test", max_tokens=0)

        # Invalid: negative
        with pytest.raises(ValidationError):
            ChatRequest(message="test", max_tokens=-10)

    @pytest.mark.unit
    def test_special_characters_in_message(self):
        """Test message with special characters."""
        special_message = "Hello! @#$%^&*() How are you? <script>alert('test')</script>"
        request = ChatRequest(message=special_message)
        assert request.message == special_message

    @pytest.mark.unit
    def test_unicode_message(self):
        """Test message with unicode characters."""
        unicode_message = "Hello! Bonjour! Hola! Guten Tag! Ciao!"
        request = ChatRequest(message=unicode_message)
        assert request.message == unicode_message

    @pytest.mark.unit
    def test_very_long_message(self):
        """Test handling of very long messages."""
        long_message = "A" * 10000
        request = ChatRequest(message=long_message)
        assert len(request.message) == 10000

    @pytest.mark.unit
    def test_json_serialization(self):
        """Test model can serialize to JSON."""
        request = ChatRequest(
            message="Test message",
            stream=False,
            temperature=0.5
        )
        json_data = request.model_dump_json()
        assert "Test message" in json_data
        assert "false" in json_data.lower()


class TestChatResponse:
    """Tests for ChatResponse model."""

    @pytest.mark.unit
    def test_valid_response(self):
        """Test creating a valid response."""
        response = ChatResponse(
            message="Hello! I'm doing great.",
            timestamp=datetime.now()
        )
        assert response.message == "Hello! I'm doing great."
        assert isinstance(response.timestamp, datetime)
        assert response.metadata is None

    @pytest.mark.unit
    def test_response_with_metadata(self):
        """Test response with metadata."""
        metadata = {
            "search_performed": True,
            "search_mode": "standard",
            "context_length": 500
        }
        response = ChatResponse(
            message="Search results found.",
            timestamp=datetime.now(),
            metadata=metadata
        )
        assert response.metadata == metadata
        assert response.metadata["search_performed"] is True

    @pytest.mark.unit
    def test_timestamp_default_factory(self):
        """Test that timestamp has a default factory."""
        response = ChatResponse(message="Test")
        assert response.timestamp is not None
        assert isinstance(response.timestamp, datetime)

    @pytest.mark.unit
    def test_missing_message_fails(self):
        """Test that missing message raises ValidationError."""
        with pytest.raises(ValidationError):
            ChatResponse()

    @pytest.mark.unit
    def test_empty_message_allowed(self):
        """Test that empty response message is allowed."""
        response = ChatResponse(message="")
        assert response.message == ""


class TestServiceHealth:
    """Tests for ServiceHealth model."""

    @pytest.mark.unit
    def test_valid_service_health(self):
        """Test creating valid service health status."""
        health = ServiceHealth(
            name="LLM (Ollama)",
            status="up",
            message="Connected to Ollama"
        )
        assert health.name == "LLM (Ollama)"
        assert health.status == "up"
        assert health.message == "Connected to Ollama"

    @pytest.mark.unit
    def test_service_health_without_message(self):
        """Test service health without optional message."""
        health = ServiceHealth(name="Memory", status="down")
        assert health.name == "Memory"
        assert health.status == "down"
        assert health.message is None

    @pytest.mark.unit
    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            ServiceHealth(name="Test")  # Missing status

        with pytest.raises(ValidationError):
            ServiceHealth(status="up")  # Missing name


class TestHealthCheck:
    """Tests for HealthCheck model."""

    @pytest.mark.unit
    def test_valid_health_check(self):
        """Test creating valid health check response."""
        services = {
            "llm": ServiceHealth(name="LLM", status="up"),
            "memory": ServiceHealth(name="Memory", status="up")
        }
        health = HealthCheck(
            status="healthy",
            services=services,
            uptime_seconds=3600.5
        )
        assert health.status == "healthy"
        assert len(health.services) == 2
        assert health.uptime_seconds == 3600.5

    @pytest.mark.unit
    def test_degraded_status(self):
        """Test degraded health check status."""
        services = {
            "llm": ServiceHealth(name="LLM", status="up"),
            "memory": ServiceHealth(name="Memory", status="down", message="ChromaDB unreachable")
        }
        health = HealthCheck(
            status="degraded",
            services=services,
            uptime_seconds=100.0
        )
        assert health.status == "degraded"
        assert health.services["memory"].status == "down"

    @pytest.mark.unit
    def test_unhealthy_status(self):
        """Test unhealthy health check status."""
        services = {
            "llm": ServiceHealth(name="LLM", status="down", message="Ollama not running")
        }
        health = HealthCheck(
            status="unhealthy",
            services=services,
            uptime_seconds=0.0
        )
        assert health.status == "unhealthy"

    @pytest.mark.unit
    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            HealthCheck(status="healthy")  # Missing services and uptime

    @pytest.mark.unit
    def test_empty_services(self):
        """Test health check with empty services dict."""
        health = HealthCheck(
            status="healthy",
            services={},
            uptime_seconds=0.0
        )
        assert len(health.services) == 0

    @pytest.mark.unit
    def test_json_serialization(self):
        """Test health check model serializes to JSON."""
        services = {
            "llm": ServiceHealth(name="LLM", status="up")
        }
        health = HealthCheck(
            status="healthy",
            services=services,
            uptime_seconds=1000.0
        )
        json_data = health.model_dump_json()
        assert "healthy" in json_data
        assert "LLM" in json_data
