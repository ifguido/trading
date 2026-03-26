"""Configuration API router."""

from __future__ import annotations

import yaml
from fastapi import APIRouter, HTTPException

from src.web.dependencies import (
    AVAILABLE_PAIRS,
    PROJECT_ROOT,
    get_engine_manager,
    read_env_file,
    write_env_file,
)
from src.web.engine_manager import EngineState
from src.web.schemas import ConfigPayload, ConfigResponse

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Return current config with masked API keys."""
    env = read_env_file()
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"

    # Read settings.yaml for pairs and mode
    raw: dict = {}
    if settings_path.exists():
        with open(settings_path) as f:
            raw = yaml.safe_load(f) or {}

    pairs = [p["symbol"] for p in raw.get("pairs", [])]
    mode = raw.get("exchange", {}).get("mode", "paper")

    # Read initial capital from engine manager
    mgr = get_engine_manager()

    return ConfigResponse(
        api_key_set=bool(env.get("BINANCE_API_KEY")),
        api_secret_set=bool(env.get("BINANCE_API_SECRET")),
        mode=mode,
        initial_capital=mgr._initial_capital,
        pairs=pairs or ["BTC/USDC", "ETH/USDC"],
    )


@router.post("")
async def update_config(payload: ConfigPayload) -> dict:
    """Update config. Only allowed when bot is stopped."""
    mgr = get_engine_manager()
    if mgr.state != EngineState.IDLE:
        raise HTTPException(400, "Cannot update config while bot is running")

    # Update .env with API keys if provided
    env_updates: dict[str, str] = {}
    if payload.api_key:
        env_updates["BINANCE_API_KEY"] = payload.api_key
    if payload.api_secret:
        env_updates["BINANCE_API_SECRET"] = payload.api_secret
    if env_updates:
        write_env_file(env_updates)

    # Store initial capital
    mgr._initial_capital = payload.initial_capital

    # Update settings.yaml
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    raw: dict = {}
    if settings_path.exists():
        with open(settings_path) as f:
            raw = yaml.safe_load(f) or {}

    # Update mode
    if "exchange" not in raw:
        raw["exchange"] = {}
    raw["exchange"]["mode"] = payload.mode
    if payload.mode == "sandbox":
        raw["exchange"]["sandbox"] = True
    elif payload.mode == "paper":
        raw["exchange"]["sandbox"] = False

    # Update pairs
    if payload.pairs:
        # Validate pairs
        valid = [p for p in payload.pairs if p in AVAILABLE_PAIRS]
        if valid:
            existing_pairs = {p["symbol"]: p for p in raw.get("pairs", [])}
            new_pairs = []
            for symbol in valid:
                if symbol in existing_pairs:
                    new_pairs.append(existing_pairs[symbol])
                else:
                    new_pairs.append({
                        "symbol": symbol,
                        "timeframes": ["1m", "5m", "15m", "1h"],
                        "strategy": "swing",
                    })
            raw["pairs"] = new_pairs

    with open(settings_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    return {"status": "ok"}
