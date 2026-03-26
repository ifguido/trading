"""Singletons and helpers for FastAPI dependency injection."""

from __future__ import annotations

import os
from pathlib import Path

from .engine_manager import EngineManager
from .event_bridge import EventBridge

# Module-level singletons
_engine_manager: EngineManager | None = None
_event_bridge: EventBridge | None = None

# Available trading pairs for the UI
AVAILABLE_PAIRS = [
    "BTC/USDC",
    "ETH/USDC",
    "SOL/USDC",
    "BNB/USDC",
    "XRP/USDC",
    "ADA/USDC",
    "AVAX/USDC",
    "DOGE/USDC",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_engine_manager() -> EngineManager:
    global _engine_manager
    if _engine_manager is None:
        _engine_manager = EngineManager()
    return _engine_manager


def get_event_bridge() -> EventBridge:
    global _event_bridge
    if _event_bridge is None:
        _event_bridge = EventBridge()
    return _event_bridge


def read_env_file() -> dict[str, str]:
    """Read .env file and return key-value pairs."""
    env_path = PROJECT_ROOT / ".env"
    result: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip().strip("\"'")
    return result


def write_env_file(updates: dict[str, str]) -> None:
    """Update .env file with new values (preserving existing entries)."""
    env_path = PROJECT_ROOT / ".env"
    existing = read_env_file()
    existing.update(updates)

    lines = [f'{k}="{v}"' for k, v in existing.items() if v]
    env_path.write_text("\n".join(lines) + "\n")

    # Also update os.environ for current process
    for k, v in updates.items():
        if v:
            os.environ[k] = v
