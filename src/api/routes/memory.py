"""Memory management API (Phase 2)."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import MemoryServiceDep

log = logging.getLogger("api.memory_routes")

router = APIRouter(prefix="/api/memory", tags=["memory"])

_VALID_TYPES = ("facts", "conversations", "summaries")


class MemoryEntry(BaseModel):
    """One stored memory entry."""

    id: str
    document: str
    metadata: Dict[str, Any] = {}
    type: str


class AddFactRequest(BaseModel):
    """Manually store a fact."""

    fact: str = Field(min_length=1, max_length=2000)
    category: str = Field(default="general", max_length=50)


@router.get("", response_model=List[MemoryEntry])
async def browse_memory(
    memory: MemoryServiceDep,
    mem_type: str = Query("facts", alias="type"),
    query: Optional[str] = Query(None, max_length=500),
    n_results: int = Query(10, ge=1, le=100),
):
    """List (newest first) or semantically search one memory tier."""
    if mem_type not in _VALID_TYPES:
        raise HTTPException(status_code=422, detail=f"type must be one of {_VALID_TYPES}")
    try:
        return await memory.browse_memory(mem_type, query, n_results)
    except Exception as e:
        log.error(f"Memory browse failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", status_code=201)
async def add_fact(request: AddFactRequest, memory: MemoryServiceDep):
    """Manually store a fact."""
    try:
        entry_id = await memory.add_fact(request.fact, request.category)
    except Exception as e:
        log.error(f"Add fact failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    if not entry_id:
        raise HTTPException(status_code=422, detail="Empty fact")
    return {"id": entry_id}


@router.delete("/{entry_id}")
async def delete_entry(entry_id: str, memory: MemoryServiceDep):
    """Delete one entry by id."""
    try:
        deleted = await memory.delete_entry(entry_id)
    except Exception as e:
        log.error(f"Delete entry failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No entry with id {entry_id}")
    return {"status": "deleted", "id": entry_id}


@router.get("/stats")
async def memory_stats(memory: MemoryServiceDep):
    """Collection counts and on-disk size."""
    try:
        stats = await memory.get_stats()
        stats["db_size_bytes"] = await memory.db_size_bytes()
        return stats
    except Exception as e:
        log.error(f"Memory stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
