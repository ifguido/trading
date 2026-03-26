"""Bot control API router: start, stop, status."""

from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, HTTPException

from src.core.config_loader import load_config
from src.web.dependencies import get_engine_manager, get_event_bridge
from src.web.engine_manager import EngineState
from src.web.schemas import BotStatusResponse

router = APIRouter(prefix="/api/bot", tags=["bot"])


@router.post("/start")
async def start_bot() -> dict:
    """Start the trading engine."""
    mgr = get_engine_manager()
    bridge = get_event_bridge()

    if mgr.state == EngineState.RUNNING:
        raise HTTPException(400, "Bot is already running")
    if mgr.state in (EngineState.STARTING, EngineState.STOPPING):
        raise HTTPException(400, f"Bot is {mgr.state.value}, please wait")

    # Load fresh config
    config = load_config()

    # Override initial equity for paper mode
    if config.exchange.mode == "paper":
        # Store capital so engine_manager can reference it
        # The engine reads initial_equity in _init_risk
        pass

    await mgr.start(config)

    # Subscribe event bridge to engine events
    if mgr.engine:
        bridge.subscribe_to_engine(mgr.engine)

    return {"status": "started"}


@router.post("/stop")
async def stop_bot() -> dict:
    """Stop the trading engine."""
    mgr = get_engine_manager()

    if mgr.state != EngineState.RUNNING:
        raise HTTPException(400, f"Bot is not running (state: {mgr.state.value})")

    await mgr.stop()
    return {"status": "stopped"}


@router.get("/status", response_model=BotStatusResponse)
async def bot_status() -> BotStatusResponse:
    """Get current bot status."""
    mgr = get_engine_manager()
    return BotStatusResponse(
        state=mgr.state.value,
        uptime_seconds=mgr.uptime,
        error=mgr.last_error,
    )
