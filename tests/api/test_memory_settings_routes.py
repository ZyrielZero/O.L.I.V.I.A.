"""Route tests for the Phase 2 memory and settings APIs."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.memory import router as memory_router
from src.api.routes.settings import router as settings_router
from src.api.services.settings_service import SettingsService


class FakeMemoryService:
    """In-memory stand-in with the browse/add/delete/stats surface."""

    def __init__(self):
        self.entries = {
            "fact_1": {"id": "fact_1", "document": "User likes tea", "metadata": {}, "type": "facts"},
        }

    async def browse_memory(self, mem_type, query=None, n_results=10):
        found = [e for e in self.entries.values() if e["type"] == mem_type]
        if query:
            found = [e for e in found if query.lower() in e["document"].lower()]
        return found[:n_results]

    async def add_fact(self, fact, category="general"):
        entry_id = f"fact_{len(self.entries) + 1}"
        self.entries[entry_id] = {
            "id": entry_id, "document": fact, "metadata": {"category": category}, "type": "facts",
        }
        return entry_id

    async def delete_entry(self, entry_id):
        return self.entries.pop(entry_id, None) is not None

    async def get_stats(self):
        return {"facts": len(self.entries), "conversations": 0, "summaries": 0, "total": len(self.entries)}

    async def db_size_bytes(self):
        return 4096


@pytest.fixture
def client(monkeypatch, tmp_path):
    from src.api.dependencies import get_memory_service
    from src.api.services import settings_service as ss

    app = FastAPI()
    app.include_router(memory_router)
    app.include_router(settings_router)

    fake_memory = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: fake_memory

    # Isolated settings store per test
    monkeypatch.setattr(ss, "_service", SettingsService(tmp_path / "settings.json"))

    with TestClient(app) as c:
        c.fake_memory = fake_memory
        yield c


# ===== /api/memory =====


def test_browse_memory_lists_entries(client):
    resp = client.get("/api/memory", params={"type": "facts"})
    assert resp.status_code == 200
    entries = resp.json()
    assert entries and entries[0]["document"] == "User likes tea"


def test_browse_memory_rejects_bad_type(client):
    assert client.get("/api/memory", params={"type": "nonsense"}).status_code == 422


def test_add_and_delete_fact(client):
    resp = client.post("/api/memory", json={"fact": "User plays guitar"})
    assert resp.status_code == 201
    entry_id = resp.json()["id"]

    assert client.delete(f"/api/memory/{entry_id}").status_code == 200
    assert client.delete(f"/api/memory/{entry_id}").status_code == 404


def test_delete_missing_entry_404(client):
    assert client.delete("/api/memory/fact_does_not_exist").status_code == 404


def test_memory_stats(client):
    resp = client.get("/api/memory/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert body["db_size_bytes"] == 4096


# ===== /api/settings =====


def test_get_settings_defaults(client):
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["vad_threshold"] == 0.5
    assert body["wake_word_enabled"] is False


def test_put_settings_partial_update_persists(client):
    resp = client.put("/api/settings", json={"vad_threshold": 0.7, "wake_word_enabled": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["vad_threshold"] == 0.7
    assert body["wake_word_enabled"] is True
    assert body["silence_end_s"] == 0.5  # untouched fields keep defaults

    # And the change survives a fresh GET
    again = client.get("/api/settings").json()
    assert again["vad_threshold"] == 0.7


def test_put_settings_rejects_out_of_range(client):
    resp = client.put("/api/settings", json={"vad_threshold": 5.0})
    assert resp.status_code == 422
    # Value unchanged
    assert client.get("/api/settings").json()["vad_threshold"] == 0.5


def test_put_settings_applies_tts_to_live_engine(client, monkeypatch):
    from src.api.container import get_container, reset_container

    reset_container()
    try:
        container = get_container()
        container.tts = SimpleNamespace(config=SimpleNamespace(exaggeration=0.5, cfg_weight=0.5))

        resp = client.put("/api/settings", json={"tts_exaggeration": 0.9, "tts_cfg_weight": 0.3})
        assert resp.status_code == 200
        assert container.tts.config.exaggeration == 0.9
        assert container.tts.config.cfg_weight == 0.3
    finally:
        reset_container()
