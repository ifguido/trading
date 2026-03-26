"""Trades API router."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.web.dependencies import get_engine_manager
from src.web.engine_manager import EngineState
from src.web.schemas import TradeRow

router = APIRouter(prefix="/api/trades", tags=["trades"])


@router.get("", response_model=list[TradeRow])
async def get_trades(limit: int = Query(50, ge=1, le=500)) -> list[TradeRow]:
    """Return recent trades from the database."""
    mgr = get_engine_manager()

    if mgr.state != EngineState.RUNNING or mgr.engine is None:
        return []

    repo = mgr.engine._repository
    if repo is None:
        return []

    trades = await repo.get_all_trades(limit=limit)
    return [
        TradeRow(
            id=t.id,
            symbol=t.symbol,
            side=t.side,
            quantity=float(t.quantity),
            price=float(t.price),
            fee=float(t.fee),
            strategy=t.strategy_name,
            executed_at=t.executed_at.isoformat() if t.executed_at else "",
        )
        for t in trades
    ]
