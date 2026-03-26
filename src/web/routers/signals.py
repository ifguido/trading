"""Signals API router."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.web.dependencies import get_engine_manager
from src.web.engine_manager import EngineState
from src.web.schemas import SignalRow

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("", response_model=list[SignalRow])
async def get_signals(limit: int = Query(50, ge=1, le=500)) -> list[SignalRow]:
    """Return recent signals from the database."""
    mgr = get_engine_manager()

    if mgr.state != EngineState.RUNNING or mgr.engine is None:
        return []

    repo = mgr.engine._repository
    if repo is None:
        return []

    signals = await repo.get_recent_signals(limit=limit)
    return [
        SignalRow(
            id=s.id,
            symbol=s.symbol,
            direction=s.direction,
            strategy=s.strategy_name,
            confidence=float(s.confidence),
            created_at=s.created_at.isoformat() if s.created_at else "",
        )
        for s in signals
    ]
