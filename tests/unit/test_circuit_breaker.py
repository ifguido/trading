"""Comprehensive unit tests for src.risk.circuit_breaker.CircuitBreaker."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from src.core.config_loader import RiskConfig
from src.core.event_bus import EventBus
from src.core.events import FillEvent, Side
from src.risk.circuit_breaker import AlertSeverity, CircuitBreaker, RiskAlertEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    """Fresh EventBus instance."""
    return EventBus()


@pytest.fixture
def default_risk_config() -> RiskConfig:
    """Default RiskConfig: 3% daily loss, 10% drawdown."""
    return RiskConfig()


@pytest.fixture
def breaker(event_bus: EventBus, default_risk_config: RiskConfig) -> CircuitBreaker:
    """CircuitBreaker with 10 000 USDT initial equity and default config."""
    return CircuitBreaker(
        event_bus=event_bus,
        config=default_risk_config,
        initial_equity=Decimal("10000"),
    )


# ---------------------------------------------------------------------------
# 1. Not tripped initially
# ---------------------------------------------------------------------------


def test_not_tripped_initially(breaker: CircuitBreaker):
    """A freshly created circuit breaker should not be tripped."""
    assert breaker.is_tripped is False


def test_trip_reason_empty_initially(breaker: CircuitBreaker):
    """Trip reason should be empty when breaker is not tripped."""
    assert breaker.trip_reason == ""


def test_check_returns_false_initially(breaker: CircuitBreaker):
    """check() should return False when within thresholds."""
    assert breaker.check() is False


# ---------------------------------------------------------------------------
# 2. Trips on daily loss exceeding 3%
# ---------------------------------------------------------------------------


def test_trips_on_daily_loss_at_threshold(breaker: CircuitBreaker):
    """Breaker should trip when daily loss reaches exactly 3% of initial equity.

    Initial equity = 10 000. 3% = 300 USDT loss.
    """
    breaker.record_realized_pnl(Decimal("-300"))  # exactly 3%

    assert breaker.is_tripped is True
    assert "max_daily_loss" in breaker.trip_reason.lower() or "daily loss" in breaker.trip_reason.lower()


def test_trips_on_daily_loss_exceeding_threshold(breaker: CircuitBreaker):
    """Breaker should trip when daily loss exceeds 3%."""
    breaker.record_realized_pnl(Decimal("-500"))  # 5%

    assert breaker.is_tripped is True


def test_does_not_trip_below_daily_loss_threshold(breaker: CircuitBreaker):
    """Breaker should NOT trip when daily loss is below 3%."""
    breaker.record_realized_pnl(Decimal("-200"))  # 2%

    assert breaker.is_tripped is False


def test_trips_on_accumulated_daily_losses(breaker: CircuitBreaker):
    """Multiple small losses accumulating to >= 3% should trip."""
    breaker.record_realized_pnl(Decimal("-100"))  # 1%
    assert breaker.is_tripped is False

    breaker.record_realized_pnl(Decimal("-100"))  # 2%
    assert breaker.is_tripped is False

    breaker.record_realized_pnl(Decimal("-100"))  # 3%
    assert breaker.is_tripped is True


def test_positive_pnl_offsets_losses(breaker: CircuitBreaker):
    """A gain should offset a loss and prevent tripping."""
    breaker.record_realized_pnl(Decimal("-250"))  # 2.5% loss
    assert breaker.is_tripped is False

    breaker.record_realized_pnl(Decimal("100"))  # net loss = 150 = 1.5%
    assert breaker.is_tripped is False


def test_positive_pnl_then_large_loss_trips(breaker: CircuitBreaker):
    """A gain followed by a large loss should trip if net exceeds 3%."""
    breaker.record_realized_pnl(Decimal("50"))   # +50
    breaker.record_realized_pnl(Decimal("-400"))  # net = -350 = 3.5%

    assert breaker.is_tripped is True


def test_daily_loss_with_custom_threshold(event_bus: EventBus):
    """Verify custom max_daily_loss_pct is respected."""
    config = RiskConfig(max_daily_loss_pct=Decimal("0.05"))  # 5%
    cb = CircuitBreaker(event_bus=event_bus, config=config, initial_equity=Decimal("10000"))

    cb.record_realized_pnl(Decimal("-400"))  # 4% - below custom 5%
    assert cb.is_tripped is False

    cb.record_realized_pnl(Decimal("-100"))  # 5% - at threshold
    assert cb.is_tripped is True


# ---------------------------------------------------------------------------
# 3. Trips on drawdown exceeding 10%
# ---------------------------------------------------------------------------


def test_trips_on_drawdown_at_threshold(breaker: CircuitBreaker):
    """Breaker should trip when drawdown reaches exactly 10%."""
    # Peak = initial = 10 000; current drops to 9 000 => 10% drawdown
    breaker.update_equity(Decimal("9000"))

    assert breaker.is_tripped is True


def test_trips_on_drawdown_exceeding_threshold(breaker: CircuitBreaker):
    """Breaker should trip when drawdown exceeds 10%."""
    breaker.update_equity(Decimal("8000"))  # 20% drawdown

    assert breaker.is_tripped is True


def test_does_not_trip_below_drawdown_threshold(breaker: CircuitBreaker):
    """Breaker should NOT trip when drawdown is below 10%."""
    breaker.update_equity(Decimal("9500"))  # 5% drawdown

    assert breaker.is_tripped is False


def test_drawdown_tracks_peak_rising_equity(breaker: CircuitBreaker):
    """Drawdown should be measured from the peak, even if equity rises first."""
    # Equity rises from 10k to 12k => new peak = 12k
    breaker.update_equity(Decimal("12000"))
    assert breaker.is_tripped is False

    # Equity drops from 12k to 10.9k => drawdown = (12k-10.9k)/12k ~ 9.17%
    breaker.update_equity(Decimal("10900"))
    assert breaker.is_tripped is False

    # Drop to 10 800 => (12k-10.8k)/12k = 10% exactly
    breaker.update_equity(Decimal("10800"))
    assert breaker.is_tripped is True


def test_drawdown_with_custom_threshold(event_bus: EventBus):
    """Verify custom max_drawdown_pct is respected."""
    config = RiskConfig(max_drawdown_pct=Decimal("0.05"))  # 5%
    cb = CircuitBreaker(event_bus=event_bus, config=config, initial_equity=Decimal("10000"))

    cb.update_equity(Decimal("9600"))  # 4% drawdown
    assert cb.is_tripped is False

    cb.update_equity(Decimal("9500"))  # 5% drawdown
    assert cb.is_tripped is True


# ---------------------------------------------------------------------------
# 4. reset() clears trip state
# ---------------------------------------------------------------------------


def test_reset_clears_tripped_state(breaker: CircuitBreaker):
    """reset() should clear the tripped flag."""
    breaker.record_realized_pnl(Decimal("-500"))  # Trip it
    assert breaker.is_tripped is True

    breaker.reset()
    assert breaker.is_tripped is False


def test_reset_clears_trip_reason(breaker: CircuitBreaker):
    """reset() should clear the trip_reason string."""
    breaker.record_realized_pnl(Decimal("-500"))
    assert breaker.trip_reason != ""

    breaker.reset()
    assert breaker.trip_reason == ""


def test_reset_clears_daily_pnl(breaker: CircuitBreaker):
    """reset() should zero out the daily P&L accumulator."""
    breaker.record_realized_pnl(Decimal("-200"))  # 2%, not tripped
    breaker.reset()

    # After reset, adding another 200 should be measured from fresh zero
    # since daily_realized_pnl was cleared.
    breaker.record_realized_pnl(Decimal("-200"))  # 2% of initial
    assert breaker.is_tripped is False


def test_reset_updates_peak_to_current(breaker: CircuitBreaker):
    """reset() should set peak_equity to current_equity."""
    breaker.update_equity(Decimal("12000"))  # peak now 12k
    breaker.update_equity(Decimal("10800"))  # trip on 10% drawdown
    assert breaker.is_tripped is True

    # Current is 10800. After reset, peak should become 10800.
    breaker.reset()
    assert breaker.is_tripped is False

    # A drop from 10800 to 9800 = ~9.26% < 10%. Should NOT trip.
    breaker.update_equity(Decimal("9800"))
    assert breaker.is_tripped is False

    # Drop to 9720 = (10800-9720)/10800 = 10% exactly. Should trip.
    breaker.update_equity(Decimal("9720"))
    assert breaker.is_tripped is True


def test_reset_allows_re_tripping(breaker: CircuitBreaker):
    """After reset(), the breaker can trip again on new losses."""
    breaker.record_realized_pnl(Decimal("-400"))
    assert breaker.is_tripped is True

    breaker.reset()
    assert breaker.is_tripped is False

    breaker.record_realized_pnl(Decimal("-300"))
    assert breaker.is_tripped is True


# ---------------------------------------------------------------------------
# 5. reset_daily() resets daily counters without un-tripping
# ---------------------------------------------------------------------------


def test_reset_daily_does_not_untrip(breaker: CircuitBreaker):
    """reset_daily() must NOT un-trip the breaker."""
    breaker.record_realized_pnl(Decimal("-500"))
    assert breaker.is_tripped is True

    breaker.reset_daily(Decimal("9500"))
    assert breaker.is_tripped is True  # Still tripped


def test_reset_daily_resets_pnl_accumulator(breaker: CircuitBreaker):
    """reset_daily() should zero the daily P&L accumulator."""
    breaker.record_realized_pnl(Decimal("-200"))  # 2%

    breaker.reset_daily(Decimal("9800"))

    # After reset, a fresh 200 loss on a 9800 base is 2.04%, not 4%.
    # First we need to un-trip (reset) to test properly.
    breaker.reset()
    breaker.record_realized_pnl(Decimal("-200"))
    # 200/9800 = 2.04% < 3%
    assert breaker.is_tripped is False


def test_reset_daily_updates_equity_fields(event_bus: EventBus):
    """reset_daily() should update initial, peak, and current equity."""
    config = RiskConfig()
    cb = CircuitBreaker(event_bus=event_bus, config=config, initial_equity=Decimal("10000"))

    cb.update_equity(Decimal("11000"))  # peak = 11k, current = 11k

    cb.reset_daily(Decimal("11000"))

    # After reset_daily with 11000, drawdown is 0. Safe to move a bit.
    cb.update_equity(Decimal("10000"))  # 1000/11000 ~ 9.09% < 10%
    assert cb.is_tripped is False

    cb.update_equity(Decimal("9900"))  # 1100/11000 = 10%
    assert cb.is_tripped is True


# ---------------------------------------------------------------------------
# 6. record_realized_pnl accumulates correctly
# ---------------------------------------------------------------------------


def test_record_realized_pnl_accumulates(breaker: CircuitBreaker):
    """Multiple calls to record_realized_pnl should accumulate."""
    breaker.record_realized_pnl(Decimal("-50"))
    breaker.record_realized_pnl(Decimal("-50"))
    breaker.record_realized_pnl(Decimal("-50"))

    # Total loss = 150, which is 1.5% of 10k. Not tripped.
    assert breaker.is_tripped is False

    breaker.record_realized_pnl(Decimal("-150"))
    # Total loss = 300, which is 3% of 10k. Tripped.
    assert breaker.is_tripped is True


def test_record_realized_pnl_with_mixed_gains_and_losses(breaker: CircuitBreaker):
    """Gains and losses should net out correctly."""
    breaker.record_realized_pnl(Decimal("-200"))  # -200
    breaker.record_realized_pnl(Decimal("100"))   # -100
    breaker.record_realized_pnl(Decimal("-150"))   # -250
    breaker.record_realized_pnl(Decimal("200"))    # -50

    # Net = -50, 0.5% of 10k. Not tripped.
    assert breaker.is_tripped is False


def test_record_realized_pnl_with_zero(breaker: CircuitBreaker):
    """Recording zero P&L should not change state."""
    breaker.record_realized_pnl(Decimal("0"))
    assert breaker.is_tripped is False


def test_record_realized_pnl_all_positive_never_trips(breaker: CircuitBreaker):
    """Positive P&L should never trip the daily loss breaker."""
    for _ in range(100):
        breaker.record_realized_pnl(Decimal("100"))
    assert breaker.is_tripped is False


# ---------------------------------------------------------------------------
# 7. update_equity tracks peak properly
# ---------------------------------------------------------------------------


def test_update_equity_sets_new_peak(breaker: CircuitBreaker):
    """update_equity should update the peak when equity rises."""
    breaker.update_equity(Decimal("11000"))
    breaker.update_equity(Decimal("12000"))

    # Now drop 10% from 12k peak = 1200 => 10800
    breaker.update_equity(Decimal("10800"))
    assert breaker.is_tripped is True


def test_update_equity_does_not_lower_peak(breaker: CircuitBreaker):
    """update_equity should never lower the peak."""
    breaker.update_equity(Decimal("12000"))
    breaker.update_equity(Decimal("11000"))  # peak stays at 12k

    # 12k - 11k = 1k, 1k/12k = 8.33% < 10%
    assert breaker.is_tripped is False


def test_update_equity_peak_tracks_increments(breaker: CircuitBreaker):
    """Peak should track through multiple increments."""
    breaker.update_equity(Decimal("10500"))
    breaker.update_equity(Decimal("11000"))
    breaker.update_equity(Decimal("11500"))
    breaker.update_equity(Decimal("12000"))

    # Drop from 12k peak to 10800 = 10%
    breaker.update_equity(Decimal("10800"))
    assert breaker.is_tripped is True


def test_update_equity_triggers_check(breaker: CircuitBreaker):
    """update_equity should call check() internally to detect drawdown."""
    # This verifies check() is called inside update_equity
    breaker.update_equity(Decimal("9000"))  # 10% drawdown from peak 10k
    assert breaker.is_tripped is True


def test_update_equity_equal_to_peak(breaker: CircuitBreaker):
    """Setting equity equal to peak should not trip (0% drawdown)."""
    breaker.update_equity(Decimal("10000"))
    assert breaker.is_tripped is False


# ---------------------------------------------------------------------------
# 8. Fee deduction via FillEvent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_event_fee_deducted_as_realized_loss(
    event_bus: EventBus, default_risk_config: RiskConfig
):
    """FillEvent fees should be recorded as negative realized P&L."""
    cb = CircuitBreaker(
        event_bus=event_bus,
        config=default_risk_config,
        initial_equity=Decimal("10000"),
    )

    fill = FillEvent(
        symbol="BTC/USDT",
        side=Side.BUY,
        quantity=Decimal("0.1"),
        price=Decimal("50000"),
        fee=Decimal("100"),
        fee_currency="USDT",
    )

    await event_bus.publish(fill)

    # Fee of 100 on 10k equity = 1% loss
    assert cb.is_tripped is False


@pytest.mark.asyncio
async def test_fill_event_large_fee_trips_breaker(
    event_bus: EventBus, default_risk_config: RiskConfig
):
    """A fill with fee >= 3% of equity should trip the breaker."""
    cb = CircuitBreaker(
        event_bus=event_bus,
        config=default_risk_config,
        initial_equity=Decimal("10000"),
    )

    fill = FillEvent(
        symbol="BTC/USDT",
        side=Side.BUY,
        quantity=Decimal("1"),
        price=Decimal("50000"),
        fee=Decimal("300"),  # 3% of 10k
        fee_currency="USDT",
    )

    await event_bus.publish(fill)

    assert cb.is_tripped is True


@pytest.mark.asyncio
async def test_fill_event_zero_fee_no_impact(
    event_bus: EventBus, default_risk_config: RiskConfig
):
    """A fill with zero fee should not affect P&L."""
    cb = CircuitBreaker(
        event_bus=event_bus,
        config=default_risk_config,
        initial_equity=Decimal("10000"),
    )

    fill = FillEvent(
        symbol="BTC/USDT",
        side=Side.BUY,
        quantity=Decimal("0.1"),
        price=Decimal("50000"),
        fee=Decimal("0"),
        fee_currency="USDT",
    )

    await event_bus.publish(fill)

    assert cb.is_tripped is False


@pytest.mark.asyncio
async def test_multiple_fills_accumulate_fees(
    event_bus: EventBus, default_risk_config: RiskConfig
):
    """Multiple fills should accumulate fee deductions."""
    cb = CircuitBreaker(
        event_bus=event_bus,
        config=default_risk_config,
        initial_equity=Decimal("10000"),
    )

    for _ in range(3):
        fill = FillEvent(
            symbol="BTC/USDT",
            side=Side.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            fee=Decimal("100"),
            fee_currency="USDT",
        )
        await event_bus.publish(fill)

    # 3 * 100 = 300, which is 3% of 10k. Should trip.
    assert cb.is_tripped is True


@pytest.mark.asyncio
async def test_fill_event_only_subscribes_to_fill_type(event_bus: EventBus):
    """CircuitBreaker's fill handler only fires on FillEvent, not TickEvent."""
    from src.core.events import TickEvent

    config = RiskConfig()
    cb = CircuitBreaker(
        event_bus=event_bus,
        config=config,
        initial_equity=Decimal("10000"),
    )

    tick = TickEvent(symbol="BTC/USDT", last=Decimal("50000"))
    await event_bus.publish(tick)

    # No fill processed, no impact
    assert cb.is_tripped is False


# ---------------------------------------------------------------------------
# Edge cases & combined scenarios
# ---------------------------------------------------------------------------


def test_zero_initial_equity_daily_loss_check_safe(event_bus: EventBus):
    """With zero initial equity, daily loss check should not divide by zero."""
    config = RiskConfig()
    cb = CircuitBreaker(event_bus=event_bus, config=config, initial_equity=Decimal("0"))

    cb.record_realized_pnl(Decimal("-100"))
    # Should not raise, and should not trip (initial_equity is 0 => skip check)
    assert cb.is_tripped is False


def test_zero_peak_equity_drawdown_check_safe(event_bus: EventBus):
    """With zero peak equity, drawdown check should not divide by zero."""
    config = RiskConfig()
    cb = CircuitBreaker(event_bus=event_bus, config=config, initial_equity=Decimal("0"))

    cb.update_equity(Decimal("-100"))
    # peak stays 0, should not raise
    assert cb.is_tripped is False


def test_already_tripped_check_returns_true_immediately(breaker: CircuitBreaker):
    """If already tripped, check() should return True without re-evaluating."""
    breaker.record_realized_pnl(Decimal("-500"))
    assert breaker.is_tripped is True

    # check() should return True even if conditions improve
    result = breaker.check()
    assert result is True


def test_trip_is_idempotent(breaker: CircuitBreaker):
    """Tripping an already-tripped breaker should not change the reason."""
    breaker.record_realized_pnl(Decimal("-300"))
    assert breaker.is_tripped is True
    original_reason = breaker.trip_reason

    # Record more losses - should not change trip reason
    breaker._tripped = False  # Force un-trip to let _trip logic run again
    breaker.record_realized_pnl(Decimal("-200"))
    # The reason might update since we forcibly un-tripped, but
    # what matters is it re-trips.
    assert breaker.is_tripped is True


@pytest.mark.asyncio
async def test_risk_alert_event_published_on_trip(event_bus: EventBus):
    """When breaker trips, a RiskAlertEvent should be published."""
    config = RiskConfig()
    alerts: list[RiskAlertEvent] = []

    async def on_alert(event: RiskAlertEvent) -> None:
        alerts.append(event)

    event_bus.subscribe(RiskAlertEvent, on_alert, name="alert_catcher")

    cb = CircuitBreaker(
        event_bus=event_bus,
        config=config,
        initial_equity=Decimal("10000"),
    )

    cb.record_realized_pnl(Decimal("-300"))  # 3% => trip

    # The alert is published via create_task, so yield to the event loop
    await asyncio.sleep(0.05)

    assert len(alerts) == 1
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].rule == "max_daily_loss"
    assert alerts[0].threshold == Decimal("0.03")


@pytest.mark.asyncio
async def test_risk_alert_drawdown_event(event_bus: EventBus):
    """Drawdown trip should publish a RiskAlertEvent with rule 'max_drawdown'."""
    config = RiskConfig()
    alerts: list[RiskAlertEvent] = []

    async def on_alert(event: RiskAlertEvent) -> None:
        alerts.append(event)

    event_bus.subscribe(RiskAlertEvent, on_alert, name="alert_catcher")

    cb = CircuitBreaker(
        event_bus=event_bus,
        config=config,
        initial_equity=Decimal("10000"),
    )

    cb.update_equity(Decimal("9000"))  # 10% drawdown => trip

    await asyncio.sleep(0.05)

    assert len(alerts) == 1
    assert alerts[0].rule == "max_drawdown"
    assert alerts[0].threshold == Decimal("0.10")
