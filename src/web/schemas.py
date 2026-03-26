"""Pydantic models for web API request/response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigPayload(BaseModel):
    """POST /api/config request body."""

    api_key: str = ""
    api_secret: str = ""
    mode: str = "paper"  # paper | sandbox | live
    initial_capital: float = 10000.0
    pairs: list[str] = Field(default_factory=list)


class ConfigResponse(BaseModel):
    """GET /api/config response."""

    api_key_set: bool = False
    api_secret_set: bool = False
    mode: str = "paper"
    initial_capital: float = 10000.0
    pairs: list[str] = Field(default_factory=list)


class BotStatusResponse(BaseModel):
    """GET /api/bot/status response."""

    state: str = "IDLE"
    uptime_seconds: float = 0.0
    error: str | None = None


class PortfolioResponse(BaseModel):
    """GET /api/portfolio response."""

    equity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    exposure: float = 0.0
    positions: list[dict] = Field(default_factory=list)


class TradeRow(BaseModel):
    """Single trade in the trades list."""

    id: int
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    strategy: str
    executed_at: str


class SignalRow(BaseModel):
    """Single signal in the signals list."""

    id: int
    symbol: str
    direction: str
    strategy: str
    confidence: float
    created_at: str
