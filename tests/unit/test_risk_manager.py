"""Unit tests for RiskManager.

Covers:
1. HOLD signal is ignored (no OrderEvent published)
2. LONG signal with stop_loss produces OrderEvent
3. Signal rejected when circuit breaker is tripped
4. Signal rejected when stop_loss missing (mandatory_stop_loss=True)
5. Signal rejected when max concurrent positions reached
6. Signal rejected when total exposure exceeds limit
7. CLOSE signal produces closing order for existing position
8. CLOSE signal ignored when no open position
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
import pytest_asyncio

from src.core.config_loader import RiskConfig
from src.core.event_bus import EventBus
from src.core.events import (
    OrderEvent,
    OrderType,
    Side,
    SignalDirection,
    SignalEvent,
)
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.portfolio_tracker import PortfolioTracker, Position
from src.risk.position_sizer import PositionSizer, SizingMode
from src.risk.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OrderCapture:
    """Subscribe to OrderEvent and collect all published orders."""

    def __init__(self) -> None:
        self.orders: list[OrderEvent] = []

    async def handler(self, event: OrderEvent) -> None:
        self.orders.append(event)


def _make_signal(
    symbol: str = "BTC/USDT",
    direction: SignalDirection = SignalDirection.LONG,
    stop_loss: Decimal | None = Decimal("29000"),
    take_profit: Decimal | None = Decimal("32000"),
    entry_price: Decimal | None = Decimal("30000"),
    strategy_name: str = "test_strategy",
    confidence: float = 0.8,
) -> SignalEvent:
    """Build a SignalEvent with sensible defaults."""
    metadata: dict = {}
    if entry_price is not None:
        metadata["entry_price"] = str(entry_price)
    return SignalEvent(
        symbol=symbol,
        direction=direction,
        strategy_name=strategy_name,
        confidence=confidence,
        stop_loss=stop_loss,
        take_profit=take_profit,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def bus() -> EventBus:
    return EventBus()


@pytest_asyncio.fixture
async def risk_config() -> RiskConfig:
    return RiskConfig(
        max_position_pct=Decimal("0.10"),
        max_total_exposure_pct=Decimal("0.50"),
        max_concurrent_positions=3,
        max_daily_loss_pct=Decimal("0.02"),
        max_drawdown_pct=Decimal("0.10"),
        mandatory_stop_loss=True,
    )


@pytest_asyncio.fixture
async def portfolio(bus: EventBus) -> PortfolioTracker:
    return PortfolioTracker(bus, initial_equity=Decimal("100000"))


@pytest_asyncio.fixture
async def sizer() -> PositionSizer:
    return PositionSizer(
        mode=SizingMode.FIXED_FRACTION,
        max_position_pct=Decimal("0.10"),
        min_order_size=Decimal("0.00001"),
    )


@pytest_asyncio.fixture
async def circuit_breaker(bus: EventBus, risk_config: RiskConfig) -> CircuitBreaker:
    return CircuitBreaker(bus, risk_config, initial_equity=Decimal("100000"))


@pytest_asyncio.fixture
async def capture(bus: EventBus) -> OrderCapture:
    """Register an OrderEvent listener *before* the RiskManager subscribes."""
    cap = OrderCapture()
    bus.subscribe(OrderEvent, cap.handler, name="test_capture")
    return cap


@pytest_asyncio.fixture
async def risk_manager(
    bus: EventBus,
    risk_config: RiskConfig,
    portfolio: PortfolioTracker,
    sizer: PositionSizer,
    circuit_breaker: CircuitBreaker,
    capture: OrderCapture,  # ensure capture is subscribed first
) -> RiskManager:
    return RiskManager(
        event_bus=bus,
        config=risk_config,
        portfolio=portfolio,
        sizer=sizer,
        circuit_breaker=circuit_breaker,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hold_signal_ignored(
    bus: EventBus,
    risk_manager: RiskManager,
    capture: OrderCapture,
) -> None:
    """A HOLD signal must not produce any OrderEvent."""
    signal = _make_signal(direction=SignalDirection.HOLD)
    await bus.publish(signal)

    assert len(capture.orders) == 0, (
        "Expected no OrderEvent for a HOLD signal"
    )


@pytest.mark.asyncio
async def test_long_signal_produces_order(
    bus: EventBus,
    risk_manager: RiskManager,
    capture: OrderCapture,
) -> None:
    """A valid LONG signal with stop_loss should produce a BUY OrderEvent."""
    signal = _make_signal(
        direction=SignalDirection.LONG,
        stop_loss=Decimal("29000"),
        take_profit=Decimal("32000"),
        entry_price=Decimal("30000"),
    )
    await bus.publish(signal)

    assert len(capture.orders) == 1, (
        f"Expected exactly 1 OrderEvent, got {len(capture.orders)}"
    )

    order = capture.orders[0]
    assert order.symbol == "BTC/USDT"
    assert order.side == Side.BUY
    assert order.order_type == OrderType.MARKET
    assert order.quantity > Decimal(0)
    assert order.stop_loss == Decimal("29000")
    assert order.take_profit == Decimal("32000")
    assert order.strategy_name == "test_strategy"


@pytest.mark.asyncio
async def test_short_signal_produces_sell_order(
    bus: EventBus,
    risk_manager: RiskManager,
    capture: OrderCapture,
) -> None:
    """A valid SHORT signal should produce a SELL OrderEvent."""
    signal = _make_signal(
        direction=SignalDirection.SHORT,
        stop_loss=Decimal("31000"),
        take_profit=Decimal("28000"),
        entry_price=Decimal("30000"),
    )
    await bus.publish(signal)

    assert len(capture.orders) == 1
    order = capture.orders[0]
    assert order.side == Side.SELL


@pytest.mark.asyncio
async def test_signal_rejected_circuit_breaker_tripped(
    bus: EventBus,
    risk_manager: RiskManager,
    circuit_breaker: CircuitBreaker,
    capture: OrderCapture,
) -> None:
    """When the circuit breaker is tripped, all new entry signals must be blocked."""
    # Trip the circuit breaker by recording a huge realized loss
    circuit_breaker.record_realized_pnl(Decimal("-50000"))  # 50% loss > 3% limit

    signal = _make_signal(direction=SignalDirection.LONG)
    await bus.publish(signal)

    assert len(capture.orders) == 0, (
        "Expected no OrderEvent when circuit breaker is tripped"
    )


@pytest.mark.asyncio
async def test_signal_rejected_missing_stop_loss(
    bus: EventBus,
    risk_manager: RiskManager,
    capture: OrderCapture,
) -> None:
    """With mandatory_stop_loss=True, a signal without stop_loss must be rejected."""
    signal = _make_signal(
        direction=SignalDirection.LONG,
        stop_loss=None,
        entry_price=Decimal("30000"),
    )
    await bus.publish(signal)

    assert len(capture.orders) == 0, (
        "Expected no OrderEvent when stop_loss is missing and mandatory"
    )


@pytest.mark.asyncio
async def test_signal_rejected_max_concurrent_positions(
    bus: EventBus,
    risk_manager: RiskManager,
    portfolio: PortfolioTracker,
    capture: OrderCapture,
) -> None:
    """Signal must be rejected when open_position_count >= max_concurrent_positions (3)."""
    # Fill up 3 positions (the configured max)
    portfolio.add_position("ETH/USDT", Side.BUY, Decimal("10"), Decimal("2000"))
    portfolio.add_position("SOL/USDT", Side.BUY, Decimal("100"), Decimal("100"))
    portfolio.add_position("DOGE/USDT", Side.BUY, Decimal("50000"), Decimal("0.10"))

    assert portfolio.open_position_count == 3

    # Try to open a 4th position on a new symbol
    signal = _make_signal(
        symbol="BTC/USDT",
        direction=SignalDirection.LONG,
    )
    await bus.publish(signal)

    assert len(capture.orders) == 0, (
        "Expected no OrderEvent when max concurrent positions reached"
    )


@pytest.mark.asyncio
async def test_signal_allowed_for_existing_position_symbol(
    bus: EventBus,
    risk_manager: RiskManager,
    portfolio: PortfolioTracker,
    capture: OrderCapture,
) -> None:
    """When max positions reached, a signal for an already-held symbol should still pass.

    Use tiny position sizes so the total-exposure check (50% of 100k equity)
    is not triggered.
    """
    # Fill up 3 positions including BTC/USDT -- keep exposure well under 50%
    portfolio.add_position("BTC/USDT", Side.BUY, Decimal("0.1"), Decimal("30000"))
    portfolio.add_position("ETH/USDT", Side.BUY, Decimal("1"), Decimal("2000"))
    portfolio.add_position("SOL/USDT", Side.BUY, Decimal("10"), Decimal("100"))

    assert portfolio.open_position_count == 3
    # Exposure = 0.1*30000 + 1*2000 + 10*100 = 3000 + 2000 + 1000 = 6000
    # 6000 / 100000 = 6% -- well under the 50% limit

    # Signal for BTC/USDT should pass since we already hold it
    signal = _make_signal(
        symbol="BTC/USDT",
        direction=SignalDirection.LONG,
    )
    await bus.publish(signal)

    assert len(capture.orders) == 1, (
        "Expected an OrderEvent for existing position symbol even at max positions"
    )


@pytest.mark.asyncio
async def test_signal_rejected_total_exposure_exceeded(
    bus: EventBus,
    risk_manager: RiskManager,
    portfolio: PortfolioTracker,
    capture: OrderCapture,
) -> None:
    """Signal must be rejected when total exposure / equity >= max_total_exposure_pct (50%)."""
    # Create a position with huge exposure relative to the 100k equity
    # 10 BTC at 30000 = 300000 exposure, which is 300% of 100k equity
    portfolio.add_position("ETH/USDT", Side.BUY, Decimal("100"), Decimal("2000"))
    portfolio.update_prices("ETH/USDT", Decimal("2000"))
    # Exposure = 100 * 2000 = 200,000. Equity = 100,000. Ratio = 200%.

    signal = _make_signal(
        symbol="BTC/USDT",
        direction=SignalDirection.LONG,
    )
    await bus.publish(signal)

    assert len(capture.orders) == 0, (
        "Expected no OrderEvent when total exposure exceeds limit"
    )


@pytest.mark.asyncio
async def test_close_signal_produces_closing_order(
    bus: EventBus,
    risk_manager: RiskManager,
    portfolio: PortfolioTracker,
    capture: OrderCapture,
) -> None:
    """A CLOSE signal for an open BUY position should produce a SELL order with matching qty."""
    position_qty = Decimal("2.5")
    portfolio.add_position("BTC/USDT", Side.BUY, position_qty, Decimal("30000"))

    signal = _make_signal(
        symbol="BTC/USDT",
        direction=SignalDirection.CLOSE,
    )
    await bus.publish(signal)

    assert len(capture.orders) == 1, (
        f"Expected 1 closing OrderEvent, got {len(capture.orders)}"
    )

    order = capture.orders[0]
    assert order.symbol == "BTC/USDT"
    assert order.side == Side.SELL, "Close of BUY position should be SELL"
    assert order.quantity == position_qty
    assert order.order_type == OrderType.MARKET


@pytest.mark.asyncio
async def test_close_signal_for_short_produces_buy(
    bus: EventBus,
    risk_manager: RiskManager,
    portfolio: PortfolioTracker,
    capture: OrderCapture,
) -> None:
    """A CLOSE signal for an open SELL (short) position should produce a BUY order."""
    position_qty = Decimal("5")
    portfolio.add_position("BTC/USDT", Side.SELL, position_qty, Decimal("30000"))

    signal = _make_signal(
        symbol="BTC/USDT",
        direction=SignalDirection.CLOSE,
    )
    await bus.publish(signal)

    assert len(capture.orders) == 1
    order = capture.orders[0]
    assert order.side == Side.BUY, "Close of SELL position should be BUY"
    assert order.quantity == position_qty


@pytest.mark.asyncio
async def test_close_signal_ignored_when_no_position(
    bus: EventBus,
    risk_manager: RiskManager,
    capture: OrderCapture,
) -> None:
    """A CLOSE signal when there is no open position should be silently dropped."""
    signal = _make_signal(
        symbol="BTC/USDT",
        direction=SignalDirection.CLOSE,
    )
    await bus.publish(signal)

    assert len(capture.orders) == 0, (
        "Expected no OrderEvent for CLOSE signal with no open position"
    )
