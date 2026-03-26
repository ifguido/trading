"""Public read-only API — no authentication required."""

from __future__ import annotations

from fastapi import APIRouter

from src.web.dependencies import get_engine_manager
from src.web.engine_manager import EngineState
from src.web.schemas import PortfolioResponse, TradeRow

router = APIRouter(prefix="/api/public", tags=["public"])


@router.get("/status")
async def public_status() -> dict:
    mgr = get_engine_manager()
    return {
        "state": mgr.state.value,
        "uptime_seconds": mgr.uptime,
    }


@router.get("/portfolio", response_model=PortfolioResponse)
async def public_portfolio() -> PortfolioResponse:
    mgr = get_engine_manager()
    if mgr.state != EngineState.RUNNING or mgr.engine is None:
        return PortfolioResponse()

    tracker = mgr.engine._portfolio_tracker
    if tracker is None:
        return PortfolioResponse()

    positions = []
    for symbol, pos in tracker.positions.items():
        positions.append({
            "symbol": pos.symbol,
            "side": pos.side.value,
            "qty": float(pos.qty),
            "entry_price": float(pos.entry_price),
            "current_price": float(pos.current_price),
            "unrealized_pnl": float(pos.unrealized_pnl),
        })

    total_unrealized = sum(
        float(pos.unrealized_pnl) for pos in tracker.positions.values()
    )

    return PortfolioResponse(
        equity=float(tracker.get_total_equity()),
        realized_pnl=float(tracker.realized_pnl),
        unrealized_pnl=total_unrealized,
        exposure=float(tracker.get_total_exposure()),
        positions=positions,
    )


@router.get("/trades", response_model=list[TradeRow])
async def public_trades() -> list[TradeRow]:
    mgr = get_engine_manager()
    if mgr.state != EngineState.RUNNING or mgr.engine is None:
        return []

    repo = mgr.engine._repository
    if repo is None:
        return []

    trades = await repo.get_all_trades(limit=10)
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
