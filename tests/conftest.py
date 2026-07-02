"""
Pytest configuration and shared fixtures for O.L.I.V.I.A. tests.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ===== Pytest Configuration =====

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ===== Mock Services =====

@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = MagicMock()
    service.model = "test-model"
    service.host = "http://localhost:11434"
    service.is_initialized.return_value = True
    service.health_check = AsyncMock(return_value=True)

    async def mock_chat_stream(message, context=None, temperature=None, max_tokens=None):
        for token in ["Hello", " ", "there", "!"]:
            yield token

    service.chat_stream = mock_chat_stream
    service.clear_history = AsyncMock()
    service.update_system_prompt = AsyncMock()

    return service


@pytest.fixture
def mock_memory_service():
    """Create a mock Memory service."""
    service = MagicMock()
    service.persist_directory = "test_memory_db"
    service.is_initialized.return_value = True
    service.health_check = AsyncMock(return_value=True)
    service.add_conversation = AsyncMock(return_value=["Test fact"])
    service.get_relevant_context = AsyncMock(return_value="Test context")
    service.query_memory = AsyncMock(return_value=["Memory 1", "Memory 2"])
    service.get_stats = AsyncMock(return_value={
        "total_facts": 10,
        "total_conversations": 5,
        "total_entries": 15
    })

    return service


@pytest.fixture
def mock_stt_service():
    """Create a mock STT service."""
    service = MagicMock()
    service.model_size = "small.en"
    service.device = "cpu"
    service.is_initialized.return_value = True
    service.health_check = AsyncMock(return_value=True)
    service.transcribe = AsyncMock(return_value="Hello, this is a test.")
    service.transcribe_numpy = AsyncMock(return_value="Hello, this is a test.")

    return service


@pytest.fixture
def mock_tts_service():
    """Create a mock TTS service."""
    service = MagicMock()
    service.is_initialized.return_value = True
    service.health_check = AsyncMock(return_value=True)
    service.synthesize = AsyncMock(return_value=b'\x00\x00' * 1000)

    async def mock_synthesize_stream(text):
        for _ in range(3):
            yield b'\x00\x00' * 100

    service.synthesize_stream = mock_synthesize_stream
    service.stop = AsyncMock()
    service.get_status = AsyncMock(return_value={
        "initialized": True,
        "model_loaded": True,
        "device": "cpu"
    })

    return service


@pytest.fixture
def mock_state_manager():
    """Create a mock State Manager."""
    from src.api.services.state_manager import StateManager
    return StateManager()


# ===== Test Data Fixtures =====

@pytest.fixture
def sample_chat_request():
    """Sample chat request data."""
    return {
        "message": "Hello, how are you?",
        "context": None,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 100
    }


@pytest.fixture
def sample_audio_bytes():
    """Sample audio bytes for testing."""
    import numpy as np
    # Generate 1 second of silence at 16kHz, 16-bit
    samples = np.zeros(16000, dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def sample_config_data():
    """Sample configuration data."""
    return {
        "name": "TestBot",
        "identity": {
            "name": "TestBot",
            "display_name": "Test Bot"
        },
        "voice": {
            "cfg_weight": 0.5,
            "exaggeration": 0.5
        },
        "system_prompt_template": "You are TestBot, a helpful assistant."
    }


# ===== FastAPI Test Client =====

@pytest.fixture
def test_app():
    """Create a test FastAPI application without real services."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/")
    async def root():
        return {"status": "test"}

    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


# ===== Live Service Fixtures =====

@pytest.fixture(scope="module")
def temp_memory_db(tmp_path_factory):
    """Create a temporary ChromaDB instance for testing."""
    temp_dir = tmp_path_factory.mktemp("memory_db")
    from src.core.memory.smart_memory import SmartMemoryDB
    db = SmartMemoryDB(persist_directory=str(temp_dir))
    yield db
    # Cleanup happens automatically when temp_dir is removed


@pytest.fixture(scope="session")
def live_llm_service():
    """
    Create a live LLM service connected to Ollama.
    Requires Ollama to be running with olivia-finetuned model.
    """
    import asyncio

    from src.api.config import APIConfig
    from src.api.services.llm_service import LLMService

    config = APIConfig()
    service = LLMService(
        model=config.OLLAMA_MODEL,
        system_prompt="You are Olivia, a helpful assistant.",
        host=config.OLLAMA_HOST
    )

    # Initialize synchronously for fixture
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.initialize())

        # Probe with a minimal generation: initialize() only checks daemon
        # connectivity, not that the configured model actually exists
        async def _probe():
            async for _ in service.chat_stream("ping", max_tokens=1):
                break

        loop.run_until_complete(_probe())
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")

    yield service

    # Cleanup
    loop.run_until_complete(service.cleanup())
    loop.close()


@pytest.fixture(scope="session")
def live_memory_service(tmp_path_factory):
    """Create a live memory service with ChromaDB."""
    import asyncio

    from src.api.services.memory_service import MemoryService

    temp_dir = tmp_path_factory.mktemp("live_memory_db")
    service = MemoryService(persist_directory=str(temp_dir))

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.initialize())
    except Exception as e:
        pytest.skip(f"ChromaDB not available: {e}")

    yield service
    loop.close()


@pytest.fixture(scope="session")
def live_tts_service():
    """
    Create a live TTS service with ChatterBox.
    Requires CUDA GPU.
    """
    import asyncio

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for TTS tests")

    from src.api.services.tts_service import TTSService

    service = TTSService(device="cuda")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.initialize())
    except Exception as e:
        pytest.skip(f"TTS service not available: {e}")

    yield service

    loop.run_until_complete(service.cleanup())
    loop.close()


@pytest.fixture
def character_config():
    """Load the character configuration for personality tests."""
    import yaml
    config_path = project_root / "config" / "character.yaml"

    if not config_path.exists():
        pytest.skip("character.yaml not found")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def forbidden_phrases(character_config):
    """Extract forbidden phrases from character config."""
    return character_config.get("forbidden_phrases", [])


@pytest.fixture
def sentence_buffer():
    """Create a SentenceBuffer instance for testing."""
    from src.api.utils.sentence_buffer import SentenceBuffer
    return SentenceBuffer()


@pytest.fixture
def conversation_manager():
    """Create a ConversationManager for testing."""
    from src.core.llm.ollama_client import ConversationManager
    return ConversationManager(
        model="test-model",
        system_prompt="Test system prompt"
    )


# ===== Test App with Mocked Services =====

@pytest.fixture
def test_app_with_mocks(mock_llm_service, mock_memory_service, mock_tts_service, mock_state_manager):
    """Create a FastAPI test app with mocked services injected."""
    from fastapi import FastAPI

    from src.api import dependencies
    from src.api.routes.health import router as health_router

    app = FastAPI()
    app.include_router(health_router)

    # Inject mocked services
    dependencies.services = {
        "llm": mock_llm_service,
        "memory": mock_memory_service,
        "stt": MagicMock(),
        "tts": mock_tts_service,
        "state": mock_state_manager
    }

    return app


@pytest.fixture
def test_client_with_mocks(test_app_with_mocks):
    """Create a test client with mocked services."""
    from fastapi.testclient import TestClient
    return TestClient(test_app_with_mocks)
