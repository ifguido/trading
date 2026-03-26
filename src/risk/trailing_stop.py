"""Trailing stop manager.

Monitors open positions via TickEvents and adjusts the stop-loss upward
(for longs) or downward (for shorts) as the price moves in favor.
When the price retraces by the trailing percentage from the peak,
it emits a CLOSE SignalEvent to exit the position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from src.core.event_bus import EventBus
from src.core.events import SignalDirection, SignalEvent, TickEvent

logger = logging.getLogger(__name__)


@dataclass
class TrailingState:
    """Tracks the peak price and trailing stop level for one position."""
    symbol: str
    side: str  # "buy" or "sell"
    entry_price: Decimal
    peak_price: Decimal  # highest price since entry (for longs)
    trailing_stop: Decimal  # current stop level
    trailing_pct: Decimal  # trailing distance as fraction (e.g. 0.03 = 3%)


class TrailingStopManager:
    """Manages trailing stops for all open positions.

    Parameters
    ----------
    event_bus : EventBus
        For subscribing to ticks and publishing close signals.
    trailing_pct : Decimal
        Trailing distance as a fraction (e.g. 0.03 = 3%).
    """

    def __init__(self, event_bus: EventBus, trailing_pct: Decimal = Decimal("0.03")) -> None:
        self._bus = event_bus
        self._trailing_pct = trailing_pct
        self._positions: dict[str, TrailingState] = {}

        self._bus.subscribe(TickEvent, self._on_tick, name="TrailingStop.tick")

    def track(self, symbol: str, side: str, entry_price: Decimal) -> None:
        """Start tracking a new position with trailing stop."""
        if side == "buy":
            stop = entry_price * (Decimal(1) - self._trailing_pct)
        else:
            stop = entry_price * (Decimal(1) + self._trailing_pct)

        self._positions[symbol] = TrailingState(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            peak_price=entry_price,
            trailing_stop=stop,
            trailing_pct=self._trailing_pct,
        )
        logger.info(
            "Trailing stop started for %s: entry=%s stop=%s (%.1f%%)",
            symbol, entry_price, stop, float(self._trailing_pct * 100),
        )

    def untrack(self, symbol: str) -> None:
        """Stop tracking a position."""
        if symbol in self._positions:
            del self._positions[symbol]

    async def _on_tick(self, event: TickEvent) -> None:
        """Update trailing stop on each tick."""
        state = self._positions.get(event.symbol)
        if state is None:
            return

        price = event.last

        if state.side == "buy":
            # Long position: trail upward
            if price > state.peak_price:
                state.peak_price = price
                state.trailing_stop = price * (Decimal(1) - state.trailing_pct)
                logger.debug(
                    "Trailing stop updated %s: peak=%s stop=%s",
                    state.symbol, price, state.trailing_stop,
                )

            # Check if stop hit
            if price <= state.trailing_stop:
                logger.info(
                    "TRAILING STOP HIT %s: price=%s stop=%s peak=%s (%.1f%% from peak)",
                    state.symbol, price, state.trailing_stop, state.peak_price,
                    float((state.peak_price - price) / state.peak_price * 100),
                )
                await self._close_position(state)

        else:
            # Short position: trail downward
            if price < state.peak_price:
                state.peak_price = price
                state.trailing_stop = price * (Decimal(1) + state.trailing_pct)

            if price >= state.trailing_stop:
                logger.info(
                    "TRAILING STOP HIT %s (short): price=%s stop=%s",
                    state.symbol, price, state.trailing_stop,
                )
                await self._close_position(state)

    async def _close_position(self, state: TrailingState) -> None:
        """Emit a CLOSE signal for the position."""
        signal = SignalEvent(
            symbol=state.symbol,
            direction=SignalDirection.CLOSE,
            strategy_name="trailing_stop",
            confidence=1.0,
        )
        self.untrack(state.symbol)
        await self._bus.publish(signal)
