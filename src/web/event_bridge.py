"""Bridge between EventBus and WebSocket clients.

Subscribes to engine events and broadcasts JSON to connected WS clients.
Throttles tick events to max 1 per second per symbol.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from decimal import Decimal
from typing import Any

from starlette.websockets import WebSocket, WebSocketState

logger = logging.getLogger(__name__)


def _default_serializer(obj: Any) -> Any:
    """JSON serializer for Decimal and other non-standard types."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class EventBridge:
    """Bridges EventBus events to WebSocket clients."""

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._last_tick: dict[str, float] = {}  # symbol -> last broadcast time
        self._tick_throttle = 3.0  # seconds

    async def add_client(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.append(ws)
        logger.debug("WS client connected (%d total)", len(self._clients))

    async def remove_client(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients = [c for c in self._clients if c is not ws]
        logger.debug("WS client disconnected (%d total)", len(self._clients))

    async def broadcast(self, message: dict) -> None:
        """Send JSON message to all connected clients."""
        if not self._clients:
            return

        text = json.dumps(message, default=_default_serializer)
        disconnected: list[WebSocket] = []

        async with self._lock:
            for ws in self._clients:
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_text(text)
                except Exception:
                    disconnected.append(ws)

            # Clean up disconnected clients
            if disconnected:
                self._clients = [c for c in self._clients if c not in disconnected]

    def subscribe_to_engine(self, engine: Any) -> None:
        """Subscribe to relevant events on the engine's EventBus."""
        from src.core.events import CandleEvent, FillEvent, SignalEvent, TickEvent

        bus = engine.event_bus
        bus.subscribe(TickEvent, self._on_tick, name="EventBridge.tick")
        bus.subscribe(FillEvent, self._on_fill, name="EventBridge.fill")
        bus.subscribe(SignalEvent, self._on_signal, name="EventBridge.signal")
        bus.subscribe(CandleEvent, self._on_candle, name="EventBridge.candle")

    async def _on_tick(self, event: Any) -> None:
        """Handle tick events with throttling."""
        now = time.time()
        last = self._last_tick.get(event.symbol, 0.0)
        if now - last < self._tick_throttle:
            return
        self._last_tick[event.symbol] = now

        asyncio.create_task(self.broadcast({
            "type": "tick",
            "symbol": event.symbol,
            "bid": event.bid,
            "ask": event.ask,
            "last": event.last,
            "volume_24h": event.volume_24h,
            "timestamp": event.timestamp,
        }))

    async def _on_fill(self, event: Any) -> None:
        asyncio.create_task(self.broadcast({
            "type": "fill",
            "symbol": event.symbol,
            "side": event.side.value,
            "quantity": event.quantity,
            "price": event.price,
            "fee": event.fee,
            "strategy": event.strategy_name,
            "timestamp": event.timestamp,
        }))

    async def _on_signal(self, event: Any) -> None:
        asyncio.create_task(self.broadcast({
            "type": "signal",
            "symbol": event.symbol,
            "direction": event.direction.value,
            "strategy": event.strategy_name,
            "confidence": event.confidence,
            "timestamp": event.timestamp,
        }))

    async def _on_candle(self, event: Any) -> None:
        if not event.closed:
            return
        asyncio.create_task(self.broadcast({
            "type": "candle",
            "symbol": event.symbol,
            "timeframe": event.timeframe,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
            "timestamp": event.timestamp,
        }))
