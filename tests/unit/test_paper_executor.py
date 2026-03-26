"""Unit tests for src.execution.paper_executor.PaperOrderExecutor."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.core.event_bus import EventBus
from src.core.events import (
    FillEvent,
    OrderEvent,
    OrderType,
    Side,
    TickEvent,
)
from src.data.data_store import DataStore
from src.execution.paper_executor import PaperOrderExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_store(event_bus: EventBus, symbols: list[str] | None = None) -> DataStore:
    """Create a DataStore wired to the EventBus."""
    return DataStore(event_bus=event_bus, symbols=symbols or ["BTC/USDT"])


def _make_order_event(
    *,
    symbol: str = "BTC/USDT",
    side: Side = Side.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: Decimal = Decimal("0.5"),
    price: Decimal | None = None,
    stop_loss: Decimal | None = None,
    strategy_name: str = "test_strategy",
) -> OrderEvent:
    return OrderEvent(
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        stop_loss=stop_loss,
        strategy_name=strategy_name,
    )


# ---------------------------------------------------------------------------
# 1. Receiving an OrderEvent produces a FillEvent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_order_event_produces_fill_event():
    """When a PaperOrderExecutor receives an OrderEvent it should publish
    a FillEvent with the correct symbol, side, quantity, and strategy."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    fills: list[FillEvent] = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(FillEvent, capture_fill, name="test_fill_capture")
    await executor.start()

    order = _make_order_event(price=Decimal("50000"))
    await bus.publish(order)

    assert len(fills) == 1
    fill = fills[0]
    assert fill.symbol == "BTC/USDT"
    assert fill.side == Side.BUY
    assert fill.quantity == Decimal("0.5")
    assert fill.strategy_name == "test_strategy"
    assert fill.exchange_order_id.startswith("paper-")
    assert fill.client_order_id == order.client_order_id

    await executor.stop()


@pytest.mark.asyncio
async def test_fill_event_has_correct_price_from_order():
    """When DataStore has no tick, the fill price should fall back to the
    order's own price."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    fills: list[FillEvent] = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(FillEvent, capture_fill, name="test_fill_capture")
    await executor.start()

    order = _make_order_event(
        order_type=OrderType.LIMIT,
        price=Decimal("49999.99"),
    )
    await bus.publish(order)

    assert len(fills) == 1
    assert fills[0].price == Decimal("49999.99")

    await executor.stop()


@pytest.mark.asyncio
async def test_fill_event_uses_stop_loss_as_fallback_price():
    """When there is no tick and no order price, the fill should use
    the stop_loss price as a last resort."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    fills: list[FillEvent] = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(FillEvent, capture_fill, name="test_fill_capture")
    await executor.start()

    order = _make_order_event(
        order_type=OrderType.MARKET,
        price=None,
        stop_loss=Decimal("48000"),
    )
    await bus.publish(order)

    assert len(fills) == 1
    assert fills[0].price == Decimal("48000")

    await executor.stop()


# ---------------------------------------------------------------------------
# 2. Paper orders are tracked in memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_orders_are_tracked():
    """Every paper order should be recorded in the executor's orders list."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)
    await executor.start()

    # Publish three orders
    for _ in range(3):
        await bus.publish(_make_order_event(price=Decimal("50000")))

    assert len(executor.orders) == 3

    for order in executor.orders:
        assert order.order_id.startswith("paper-")
        assert order.symbol == "BTC/USDT"
        assert order.side == Side.BUY
        assert order.status == "filled"

    await executor.stop()


@pytest.mark.asyncio
async def test_tracked_order_has_correct_details():
    """Tracked paper order should carry all relevant fields."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)
    await executor.start()

    order_event = _make_order_event(
        symbol="ETH/USDT",
        side=Side.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("10"),
        price=Decimal("3200"),
    )
    await bus.publish(order_event)

    assert len(executor.orders) == 1
    po = executor.orders[0]
    assert po.symbol == "ETH/USDT"
    assert po.side == Side.SELL
    assert po.order_type == OrderType.LIMIT
    assert po.quantity == Decimal("10")
    assert po.requested_price == Decimal("3200")
    assert po.fill_price == Decimal("3200")

    await executor.stop()


# ---------------------------------------------------------------------------
# 3. cancel_order is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_order_is_noop():
    """cancel_order should return a synthetic response without error."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    result = await executor.cancel_order("paper-abc123", "BTC/USDT")

    assert result["id"] == "paper-abc123"
    assert result["status"] == "canceled"


@pytest.mark.asyncio
async def test_fetch_order_status_is_noop():
    """fetch_order_status should return a synthetic 'closed' status."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    result = await executor.fetch_order_status("paper-xyz789", "ETH/USDT")

    assert result["id"] == "paper-xyz789"
    assert result["status"] == "closed"


# ---------------------------------------------------------------------------
# 4. Fill price from DataStore when available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_price_from_data_store():
    """When the DataStore has a recent tick, the fill price should come
    from the tick's ``last`` field, not from the order price."""
    bus = EventBus()
    ds = _make_data_store(bus)
    await ds.start()

    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    fills: list[FillEvent] = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(FillEvent, capture_fill, name="test_fill_capture")
    await executor.start()

    # Inject a tick into the DataStore
    tick = TickEvent(
        symbol="BTC/USDT",
        bid=Decimal("51000"),
        ask=Decimal("51050"),
        last=Decimal("51025"),
        volume_24h=Decimal("999"),
    )
    await bus.publish(tick)

    # Now send an order — the fill should use the tick price, not the limit price
    order = _make_order_event(
        order_type=OrderType.LIMIT,
        price=Decimal("50000"),
    )
    await bus.publish(order)

    assert len(fills) == 1
    assert fills[0].price == Decimal("51025")  # from tick, not from order

    await ds.stop()
    await executor.stop()


@pytest.mark.asyncio
async def test_fill_price_falls_back_when_no_tick():
    """Without a tick in DataStore, the fill price should fall back to
    the order price."""
    bus = EventBus()
    ds = _make_data_store(bus)
    # DataStore started but no ticks published
    await ds.start()

    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    fills: list[FillEvent] = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(FillEvent, capture_fill, name="test_fill_capture")
    await executor.start()

    order = _make_order_event(
        order_type=OrderType.LIMIT,
        price=Decimal("49000"),
    )
    await bus.publish(order)

    assert len(fills) == 1
    assert fills[0].price == Decimal("49000")

    await ds.stop()
    await executor.stop()


# ---------------------------------------------------------------------------
# 5. Simulated fee calculation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_fee_by_default():
    """By default the simulated fee should be zero."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    fills: list[FillEvent] = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(FillEvent, capture_fill, name="test_fill_capture")
    await executor.start()

    await bus.publish(_make_order_event(price=Decimal("50000")))

    assert len(fills) == 1
    assert fills[0].fee == Decimal(0)

    await executor.stop()


@pytest.mark.asyncio
async def test_configurable_simulated_fee():
    """When simulated_fee_rate is set, the fee should be price * qty * rate."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(
        event_bus=bus,
        data_store=ds,
        simulated_fee_rate=Decimal("0.001"),  # 0.1%
        fee_currency="BNB",
    )

    fills: list[FillEvent] = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(FillEvent, capture_fill, name="test_fill_capture")
    await executor.start()

    await bus.publish(_make_order_event(
        price=Decimal("50000"),
        quantity=Decimal("2"),
    ))

    assert len(fills) == 1
    # fee = 50000 * 2 * 0.001 = 100
    assert fills[0].fee == Decimal("100")
    assert fills[0].fee_currency == "BNB"

    await executor.stop()


# ---------------------------------------------------------------------------
# 6. Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_and_stop_idempotent():
    """Calling start() twice should warn but not error. Same for stop()."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    await executor.start()
    assert executor.is_running is True

    # Second start should be safe
    await executor.start()
    assert executor.is_running is True

    await executor.stop()
    assert executor.is_running is False

    # Second stop should be safe
    await executor.stop()
    assert executor.is_running is False


@pytest.mark.asyncio
async def test_ignores_events_when_not_running():
    """After stop(), incoming OrderEvents should be ignored."""
    bus = EventBus()
    ds = _make_data_store(bus)
    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    await executor.start()
    await executor.stop()

    fills: list[FillEvent] = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(FillEvent, capture_fill, name="test_fill_capture")

    # Manually call handler (simulating a late delivery)
    await executor._handle_order_event(_make_order_event(price=Decimal("50000")))

    # No fill should have been published
    assert len(fills) == 0
    assert len(executor.orders) == 0


# ---------------------------------------------------------------------------
# 7. Multiple symbols
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_symbols():
    """Paper executor should handle orders for different symbols."""
    bus = EventBus()
    ds = _make_data_store(bus, symbols=["BTC/USDT", "ETH/USDT"])
    await ds.start()

    executor = PaperOrderExecutor(event_bus=bus, data_store=ds)

    fills: list[FillEvent] = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(FillEvent, capture_fill, name="test_fill_capture")
    await executor.start()

    # Inject ticks for both symbols
    await bus.publish(TickEvent(symbol="BTC/USDT", last=Decimal("60000"), bid=Decimal("59999"), ask=Decimal("60001")))
    await bus.publish(TickEvent(symbol="ETH/USDT", last=Decimal("4000"), bid=Decimal("3999"), ask=Decimal("4001")))

    # Orders for both
    await bus.publish(_make_order_event(symbol="BTC/USDT", price=Decimal("50000")))
    await bus.publish(_make_order_event(symbol="ETH/USDT", side=Side.SELL, price=Decimal("3500")))

    assert len(fills) == 2
    btc_fill = next(f for f in fills if f.symbol == "BTC/USDT")
    eth_fill = next(f for f in fills if f.symbol == "ETH/USDT")

    assert btc_fill.price == Decimal("60000")  # from tick
    assert eth_fill.price == Decimal("4000")    # from tick
    assert eth_fill.side == Side.SELL

    await ds.stop()
    await executor.stop()
