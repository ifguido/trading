"""Portfolio API router."""

from __future__ import annotations

from fastapi import APIRouter

from src.web.dependencies import get_engine_manager
from src.web.engine_manager import EngineState
from src.web.schemas import PortfolioResponse

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.get("", response_model=PortfolioResponse)
async def get_portfolio() -> PortfolioResponse:
    """Return current portfolio state."""
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
