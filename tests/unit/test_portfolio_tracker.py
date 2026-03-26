"""Comprehensive unit tests for PortfolioTracker.

Covers: add_position, close_position (full & partial), get_total_equity,
get_total_exposure, mark_to_market, has_position / get_position,
open_position_count, opposite-side netting, and same-side averaging.
"""

from __future__ import annotations

import pytest
from decimal import Decimal

from src.core.event_bus import EventBus
from src.core.events import FillEvent, Side, TickEvent
from src.risk.portfolio_tracker import PortfolioTracker, Position


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def tracker(event_bus: EventBus) -> PortfolioTracker:
    return PortfolioTracker(event_bus, initial_equity=Decimal("10000"))


# ---------------------------------------------------------------------------
# 1. add_position creates a new position
# ---------------------------------------------------------------------------

class TestAddPosition:
    def test_creates_new_long_position(self, tracker: PortfolioTracker) -> None:
        pos = tracker.add_position("BTCUSDT", Side.BUY, Decimal("0.5"), Decimal("40000"))
        assert pos.symbol == "BTCUSDT"
        assert pos.side == Side.BUY
        assert pos.qty == Decimal("0.5")
        assert pos.entry_price == Decimal("40000")
        assert pos.current_price == Decimal("40000")
        assert tracker.has_position("BTCUSDT")

    def test_creates_new_short_position(self, tracker: PortfolioTracker) -> None:
        pos = tracker.add_position("ETHUSDT", Side.SELL, Decimal("10"), Decimal("2500"))
        assert pos.symbol == "ETHUSDT"
        assert pos.side == Side.SELL
        assert pos.qty == Decimal("10")
        assert pos.entry_price == Decimal("2500")

    def test_new_position_appears_in_positions_dict(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        positions = tracker.positions
        assert "BTCUSDT" in positions
        assert positions["BTCUSDT"].qty == Decimal("1")

    def test_add_multiple_different_symbols(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        tracker.add_position("ETHUSDT", Side.BUY, Decimal("10"), Decimal("2500"))
        assert tracker.open_position_count == 2


# ---------------------------------------------------------------------------
# 2. close_position fully closes and returns realized P&L
# ---------------------------------------------------------------------------

class TestClosePositionFull:
    def test_full_close_long_profit(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        pnl = tracker.close_position("BTCUSDT", Decimal("42000"))
        assert pnl == Decimal("2000")
        assert not tracker.has_position("BTCUSDT")
        assert tracker.realized_pnl == Decimal("2000")

    def test_full_close_long_loss(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        pnl = tracker.close_position("BTCUSDT", Decimal("38000"))
        assert pnl == Decimal("-2000")
        assert tracker.realized_pnl == Decimal("-2000")

    def test_full_close_short_profit(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("10"), Decimal("2500"))
        pnl = tracker.close_position("ETHUSDT", Decimal("2300"))
        # (2500 - 2300) * 10 = 2000
        assert pnl == Decimal("2000")
        assert not tracker.has_position("ETHUSDT")

    def test_full_close_short_loss(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("10"), Decimal("2500"))
        pnl = tracker.close_position("ETHUSDT", Decimal("2700"))
        assert pnl == Decimal("-2000")

    def test_close_nonexistent_position_returns_zero(self, tracker: PortfolioTracker) -> None:
        pnl = tracker.close_position("XYZUSDT", Decimal("100"))
        assert pnl == Decimal("0")

    def test_close_removes_position_from_dict(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        tracker.close_position("BTCUSDT", Decimal("50000"))
        assert tracker.open_position_count == 0
        assert "BTCUSDT" not in tracker.positions

    def test_close_qty_exceeding_position_caps_to_position_qty(self, tracker: PortfolioTracker) -> None:
        """Closing more than owned should close the full position, not error."""
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        pnl = tracker.close_position("BTCUSDT", Decimal("41000"), qty=Decimal("5"))
        assert pnl == Decimal("1000")  # (41000-40000) * 1
        assert not tracker.has_position("BTCUSDT")


# ---------------------------------------------------------------------------
# 3. close_position partial close
# ---------------------------------------------------------------------------

class TestClosePositionPartial:
    def test_partial_close_reduces_qty(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("2"), Decimal("40000"))
        pnl = tracker.close_position("BTCUSDT", Decimal("41000"), qty=Decimal("1"))
        assert pnl == Decimal("1000")
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert pos.qty == Decimal("1")

    def test_partial_close_accumulates_realized_pnl(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("4"), Decimal("40000"))
        tracker.close_position("BTCUSDT", Decimal("41000"), qty=Decimal("1"))
        tracker.close_position("BTCUSDT", Decimal("42000"), qty=Decimal("1"))
        # 1000 + 2000 = 3000
        assert tracker.realized_pnl == Decimal("3000")
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert pos.qty == Decimal("2")

    def test_partial_close_short(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("10"), Decimal("2500"))
        pnl = tracker.close_position("ETHUSDT", Decimal("2400"), qty=Decimal("3"))
        # (2500 - 2400) * 3 = 300
        assert pnl == Decimal("300")
        assert tracker.get_position("ETHUSDT").qty == Decimal("7")

    def test_partial_close_updates_unrealized_pnl(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("2"), Decimal("40000"))
        tracker.close_position("BTCUSDT", Decimal("41000"), qty=Decimal("1"))
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        # After partial close, mark_to_market is called with exit_price (41000)
        # entry_price is still 40000, qty is 1
        # unrealized = (41000 - 40000) * 1 = 1000
        assert pos.unrealized_pnl == Decimal("1000")


# ---------------------------------------------------------------------------
# 4. get_total_equity = initial + realized + unrealized
# ---------------------------------------------------------------------------

class TestGetTotalEquity:
    def test_initial_equity_only(self, tracker: PortfolioTracker) -> None:
        assert tracker.get_total_equity() == Decimal("10000")

    def test_equity_with_realized_gain(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        tracker.close_position("BTCUSDT", Decimal("42000"))
        assert tracker.get_total_equity() == Decimal("12000")

    def test_equity_with_unrealized_gain(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        tracker.update_prices("BTCUSDT", Decimal("43000"))
        # initial 10000 + unrealized (43000-40000)*1 = 13000
        assert tracker.get_total_equity() == Decimal("13000")

    def test_equity_with_both_realized_and_unrealized(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("2"), Decimal("40000"))
        # Close half for +1000
        tracker.close_position("BTCUSDT", Decimal("41000"), qty=Decimal("1"))
        # Remaining position marked at 41000: unrealized = (41000-40000)*1 = 1000
        # Total: 10000 + 1000 (realized) + 1000 (unrealized) = 12000
        assert tracker.get_total_equity() == Decimal("12000")

    def test_equity_with_realized_loss(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        tracker.close_position("BTCUSDT", Decimal("39000"))
        assert tracker.get_total_equity() == Decimal("9000")

    def test_equity_with_multiple_positions(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("10"), Decimal("2500"))
        tracker.update_prices("BTCUSDT", Decimal("41000"))  # unrealized +1000
        tracker.update_prices("ETHUSDT", Decimal("2400"))    # unrealized +1000
        # 10000 + 0 + 1000 + 1000 = 12000
        assert tracker.get_total_equity() == Decimal("12000")


# ---------------------------------------------------------------------------
# 5. get_total_exposure = sum(qty * price)
# ---------------------------------------------------------------------------

class TestGetTotalExposure:
    def test_no_positions_zero_exposure(self, tracker: PortfolioTracker) -> None:
        assert tracker.get_total_exposure() == Decimal("0")

    def test_single_position_exposure(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("0.5"), Decimal("40000"))
        assert tracker.get_total_exposure() == Decimal("20000")

    def test_multiple_positions_exposure(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("0.5"), Decimal("40000"))
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("10"), Decimal("2500"))
        # 0.5 * 40000 + 10 * 2500 = 20000 + 25000 = 45000
        assert tracker.get_total_exposure() == Decimal("45000")

    def test_exposure_updates_after_mark_to_market(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        assert tracker.get_total_exposure() == Decimal("40000")
        tracker.update_prices("BTCUSDT", Decimal("42000"))
        assert tracker.get_total_exposure() == Decimal("42000")


# ---------------------------------------------------------------------------
# 6. mark_to_market updates unrealized P&L
# ---------------------------------------------------------------------------

class TestMarkToMarket:
    def test_update_prices_long(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("2"), Decimal("40000"))
        tracker.update_prices("BTCUSDT", Decimal("41000"))
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert pos.current_price == Decimal("41000")
        assert pos.unrealized_pnl == Decimal("2000")

    def test_update_prices_short(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("5"), Decimal("2500"))
        tracker.update_prices("ETHUSDT", Decimal("2400"))
        pos = tracker.get_position("ETHUSDT")
        assert pos is not None
        assert pos.unrealized_pnl == Decimal("500")

    def test_update_prices_negative_unrealized(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        tracker.update_prices("BTCUSDT", Decimal("38000"))
        pos = tracker.get_position("BTCUSDT")
        assert pos.unrealized_pnl == Decimal("-2000")

    def test_update_prices_nonexistent_symbol_no_error(self, tracker: PortfolioTracker) -> None:
        """Updating price for an unknown symbol should be a no-op."""
        tracker.update_prices("XYZUSDT", Decimal("100"))  # Should not raise

    def test_position_mark_to_market_directly(self) -> None:
        """Test the Position.mark_to_market method directly."""
        pos = Position(
            symbol="BTCUSDT",
            side=Side.BUY,
            qty=Decimal("3"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
        )
        pos.mark_to_market(Decimal("52000"))
        assert pos.current_price == Decimal("52000")
        assert pos.unrealized_pnl == Decimal("6000")

    def test_position_mark_to_market_short_directly(self) -> None:
        pos = Position(
            symbol="ETHUSDT",
            side=Side.SELL,
            qty=Decimal("10"),
            entry_price=Decimal("2000"),
            current_price=Decimal("2000"),
        )
        pos.mark_to_market(Decimal("1800"))
        assert pos.unrealized_pnl == Decimal("2000")


# ---------------------------------------------------------------------------
# 7. has_position / get_position
# ---------------------------------------------------------------------------

class TestHasAndGetPosition:
    def test_has_position_false_when_empty(self, tracker: PortfolioTracker) -> None:
        assert tracker.has_position("BTCUSDT") is False

    def test_has_position_true_after_add(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        assert tracker.has_position("BTCUSDT") is True

    def test_has_position_false_after_full_close(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        tracker.close_position("BTCUSDT", Decimal("50000"))
        assert tracker.has_position("BTCUSDT") is False

    def test_get_position_returns_none_when_missing(self, tracker: PortfolioTracker) -> None:
        assert tracker.get_position("BTCUSDT") is None

    def test_get_position_returns_position_when_present(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert pos.symbol == "BTCUSDT"
        assert pos.qty == Decimal("1")

    def test_get_position_returns_none_after_full_close(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        tracker.close_position("BTCUSDT", Decimal("51000"))
        assert tracker.get_position("BTCUSDT") is None


# ---------------------------------------------------------------------------
# 8. open_position_count
# ---------------------------------------------------------------------------

class TestOpenPositionCount:
    def test_zero_when_empty(self, tracker: PortfolioTracker) -> None:
        assert tracker.open_position_count == 0

    def test_increments_on_add(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        assert tracker.open_position_count == 1
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("5"), Decimal("2500"))
        assert tracker.open_position_count == 2

    def test_decrements_on_full_close(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("5"), Decimal("2500"))
        tracker.close_position("BTCUSDT", Decimal("50000"))
        assert tracker.open_position_count == 1

    def test_unchanged_on_partial_close(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("2"), Decimal("50000"))
        tracker.close_position("BTCUSDT", Decimal("50000"), qty=Decimal("1"))
        assert tracker.open_position_count == 1

    def test_same_symbol_add_does_not_increase_count(self, tracker: PortfolioTracker) -> None:
        """Adding to the same symbol averages in; it should not create a new entry."""
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("52000"))
        assert tracker.open_position_count == 1


# ---------------------------------------------------------------------------
# 9. Opposite-side fill nets against existing position
# ---------------------------------------------------------------------------

class TestOpposideSideNetting:
    def test_full_close_via_opposite_side(self, tracker: PortfolioTracker) -> None:
        """Selling exactly the same qty as an existing long should close it."""
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        result = tracker.add_position("BTCUSDT", Side.SELL, Decimal("1"), Decimal("42000"))
        # Position should be closed
        assert not tracker.has_position("BTCUSDT")
        assert tracker.realized_pnl == Decimal("2000")
        # Result should be a zero-qty sentinel
        assert result.qty == Decimal("0")

    def test_partial_close_via_opposite_side(self, tracker: PortfolioTracker) -> None:
        """Selling less qty than the existing long should partially close."""
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("2"), Decimal("40000"))
        result = tracker.add_position("BTCUSDT", Side.SELL, Decimal("1"), Decimal("42000"))
        assert tracker.has_position("BTCUSDT")
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert pos.qty == Decimal("1")
        assert pos.side == Side.BUY
        assert tracker.realized_pnl == Decimal("2000")

    def test_flip_position_via_opposite_side(self, tracker: PortfolioTracker) -> None:
        """Selling more than the long qty should close long and open a short remainder."""
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        result = tracker.add_position("BTCUSDT", Side.SELL, Decimal("3"), Decimal("42000"))
        # The original long (1 unit) is closed with pnl = (42000-40000)*1 = 2000
        assert tracker.realized_pnl == Decimal("2000")
        # A new short position of 2 units should exist
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == Side.SELL
        assert pos.qty == Decimal("2")
        assert pos.entry_price == Decimal("42000")

    def test_opposite_side_short_closed_by_buy(self, tracker: PortfolioTracker) -> None:
        """Buying against an existing short closes the short."""
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("5"), Decimal("2500"))
        tracker.add_position("ETHUSDT", Side.BUY, Decimal("5"), Decimal("2300"))
        assert not tracker.has_position("ETHUSDT")
        # (2500 - 2300) * 5 = 1000
        assert tracker.realized_pnl == Decimal("1000")


# ---------------------------------------------------------------------------
# 10. Average-in on same side
# ---------------------------------------------------------------------------

class TestAverageIn:
    def test_average_in_long(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("42000"))
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert pos.qty == Decimal("2")
        # avg price = (40000*1 + 42000*1) / 2 = 41000
        assert pos.entry_price == Decimal("41000")
        assert pos.side == Side.BUY

    def test_average_in_short(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("5"), Decimal("2500"))
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("5"), Decimal("2700"))
        pos = tracker.get_position("ETHUSDT")
        assert pos is not None
        assert pos.qty == Decimal("10")
        # avg price = (2500*5 + 2700*5) / 10 = 2600
        assert pos.entry_price == Decimal("2600")

    def test_average_in_weighted(self, tracker: PortfolioTracker) -> None:
        """Unequal quantities should produce a weighted average."""
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("3"), Decimal("40000"))
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("44000"))
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert pos.qty == Decimal("4")
        # (40000*3 + 44000*1) / 4 = 164000 / 4 = 41000
        assert pos.entry_price == Decimal("41000")

    def test_average_in_preserves_count(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("40000"))
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("42000"))
        assert tracker.open_position_count == 1


# ---------------------------------------------------------------------------
# Event handler integration tests (async)
# ---------------------------------------------------------------------------

class TestEventHandlers:
    @pytest.mark.asyncio
    async def test_on_fill_opens_new_position(self, tracker: PortfolioTracker, event_bus: EventBus) -> None:
        fill = FillEvent(
            symbol="BTCUSDT",
            side=Side.BUY,
            quantity=Decimal("1"),
            price=Decimal("50000"),
        )
        await event_bus.publish(fill)
        assert tracker.has_position("BTCUSDT")
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert pos.qty == Decimal("1")
        assert pos.entry_price == Decimal("50000")

    @pytest.mark.asyncio
    async def test_on_fill_closes_long_on_sell(self, tracker: PortfolioTracker, event_bus: EventBus) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        fill = FillEvent(
            symbol="BTCUSDT",
            side=Side.SELL,
            quantity=Decimal("1"),
            price=Decimal("52000"),
        )
        await event_bus.publish(fill)
        assert not tracker.has_position("BTCUSDT")
        assert tracker.realized_pnl == Decimal("2000")

    @pytest.mark.asyncio
    async def test_on_fill_closes_short_on_buy(self, tracker: PortfolioTracker, event_bus: EventBus) -> None:
        tracker.add_position("ETHUSDT", Side.SELL, Decimal("5"), Decimal("2500"))
        fill = FillEvent(
            symbol="ETHUSDT",
            side=Side.BUY,
            quantity=Decimal("5"),
            price=Decimal("2300"),
        )
        await event_bus.publish(fill)
        assert not tracker.has_position("ETHUSDT")
        assert tracker.realized_pnl == Decimal("1000")

    @pytest.mark.asyncio
    async def test_on_fill_partial_close(self, tracker: PortfolioTracker, event_bus: EventBus) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("3"), Decimal("50000"))
        fill = FillEvent(
            symbol="BTCUSDT",
            side=Side.SELL,
            quantity=Decimal("1"),
            price=Decimal("51000"),
        )
        await event_bus.publish(fill)
        assert tracker.has_position("BTCUSDT")
        pos = tracker.get_position("BTCUSDT")
        assert pos.qty == Decimal("2")
        assert tracker.realized_pnl == Decimal("1000")

    @pytest.mark.asyncio
    async def test_on_tick_updates_mark_to_market(self, tracker: PortfolioTracker, event_bus: EventBus) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        tick = TickEvent(symbol="BTCUSDT", last=Decimal("55000"))
        await event_bus.publish(tick)
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert pos.current_price == Decimal("55000")
        assert pos.unrealized_pnl == Decimal("5000")

    @pytest.mark.asyncio
    async def test_on_tick_no_position_is_noop(self, tracker: PortfolioTracker, event_bus: EventBus) -> None:
        """TickEvent for an unknown symbol should not raise."""
        tick = TickEvent(symbol="XYZUSDT", last=Decimal("100"))
        await event_bus.publish(tick)  # Should not raise


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_positions_property_returns_copy(self, tracker: PortfolioTracker) -> None:
        """Mutating the returned dict should not affect internal state."""
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        positions_copy = tracker.positions
        positions_copy.pop("BTCUSDT")
        assert tracker.has_position("BTCUSDT")

    def test_zero_pnl_on_breakeven_close(self, tracker: PortfolioTracker) -> None:
        tracker.add_position("BTCUSDT", Side.BUY, Decimal("1"), Decimal("50000"))
        pnl = tracker.close_position("BTCUSDT", Decimal("50000"))
        assert pnl == Decimal("0")
        assert tracker.realized_pnl == Decimal("0")

    def test_initial_equity_zero(self, event_bus: EventBus) -> None:
        tracker = PortfolioTracker(event_bus, initial_equity=Decimal("0"))
        assert tracker.get_total_equity() == Decimal("0")

    def test_subscribes_to_event_bus(self, event_bus: EventBus) -> None:
        """PortfolioTracker should subscribe to FillEvent and TickEvent on init."""
        tracker = PortfolioTracker(event_bus)
        assert event_bus.subscriber_count(FillEvent) >= 1
        assert event_bus.subscriber_count(TickEvent) >= 1
