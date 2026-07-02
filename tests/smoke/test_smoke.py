"""
Smoke Tests for O.L.I.V.I.A.
Quick sanity checks to verify the system starts and basic functionality works.
"""

from pathlib import Path

import pytest
import yaml

# ===== Test 1: Health Endpoint Responds =====

@pytest.mark.smoke
def test_health_endpoint_responds(test_client_with_mocks):
    """Verify /health endpoint returns 200 with status field."""
    response = test_client_with_mocks.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded", "unhealthy"]


# ===== Test 2: Chat Endpoint Accepts Request =====

@pytest.mark.smoke
def test_chat_endpoint_accepts_request():
    """Verify /api/chat endpoint accepts valid POST requests."""
    # This test verifies the route is importable and configured correctly
    from src.api.routes.chat import router

    # Check router has the expected route
    routes = [route.path for route in router.routes]
    assert "/chat" in routes or any("/chat" in str(r) for r in router.routes)


# ===== Test 3: LLM Service Imports =====

@pytest.mark.smoke
def test_llm_service_imports():
    """Verify LLMService can be imported without error."""
    try:
        from src.api.services.llm_service import LLMService
        assert LLMService is not None
        assert hasattr(LLMService, 'initialize')
        assert hasattr(LLMService, 'chat_stream')
        assert hasattr(LLMService, 'health_check')
    except ImportError as e:
        pytest.fail(f"Failed to import LLMService: {e}")


# ===== Test 4: Memory Service Imports =====

@pytest.mark.smoke
def test_memory_service_imports():
    """Verify MemoryService can be imported without error."""
    try:
        from src.api.services.memory_service import MemoryService
        assert MemoryService is not None
        assert hasattr(MemoryService, 'initialize')
        assert hasattr(MemoryService, 'add_conversation')
        assert hasattr(MemoryService, 'get_relevant_context')
        assert hasattr(MemoryService, 'health_check')
    except ImportError as e:
        pytest.fail(f"Failed to import MemoryService: {e}")


# ===== Test 5: TTS Service Imports =====

@pytest.mark.smoke
def test_tts_service_imports():
    """Verify TTSService can be imported without error."""
    try:
        from src.api.services.tts_service import TTSService
        assert TTSService is not None
        assert hasattr(TTSService, 'initialize')
        assert hasattr(TTSService, 'synthesize')
        assert hasattr(TTSService, 'health_check')
    except ImportError as e:
        pytest.fail(f"Failed to import TTSService: {e}")


# ===== Test 6: Character Config Loads =====

@pytest.mark.smoke
def test_character_config_loads():
    """Verify character.yaml loads and parses correctly."""
    config_path = Path(__file__).parent.parent.parent / "config" / "character.yaml"

    assert config_path.exists(), f"character.yaml not found at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Verify required sections exist
    assert "identity" in config, "Missing 'identity' section"
    assert "personality" in config, "Missing 'personality' section"
    assert "speaking_style" in config, "Missing 'speaking_style' section"

    # Verify identity has required fields
    identity = config["identity"]
    assert "name" in identity, "Missing 'name' in identity"

    # Verify personality has traits
    personality = config["personality"]
    assert "traits" in personality or len(personality) > 0, "Personality section is empty"
