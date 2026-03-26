"""Integration tests for the storage layer.

Tests the full Repository <-> SQLAlchemy <-> SQLite (in-memory) stack,
verifying that every CRUD operation produces the expected database state.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.core.events import (
    FillEvent,
    OrderEvent,
    OrderType,
    Side,
    SignalDirection,
    SignalEvent,
)
from src.storage.db import async_session_factory, close_db, init_db
from src.storage.models import DailyPnL, Order, SignalLog, Trade
from src.storage.repository import Repository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def repo():
    """Create an in-memory SQLite DB, yield a Repository, then tear down."""
    await init_db("sqlite+aiosqlite:///:memory:")
    factory = async_session_factory()
    repository = Repository(factory)
    yield repository
    await close_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS_BASE = int(datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def _make_fill(**overrides) -> FillEvent:
    """Build a FillEvent with sensible defaults; override any field."""
    defaults = dict(
        symbol="BTC/USDT",
        side=Side.BUY,
        quantity=Decimal("0.5"),
        price=Decimal("60000"),
        fee=Decimal("30"),
        fee_currency="USDT",
        exchange_order_id="exch-001",
        client_order_id="cli-001",
        strategy_name="momentum",
        timestamp=_TS_BASE,
    )
    defaults.update(overrides)
    return FillEvent(**defaults)


def _make_order(**overrides) -> OrderEvent:
    defaults = dict(
        symbol="BTC/USDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("59000"),
        stop_loss=Decimal("57000"),
        take_profit=Decimal("63000"),
        client_order_id="ord-001",
        strategy_name="momentum",
    )
    defaults.update(overrides)
    return OrderEvent(**defaults)


def _make_signal(**overrides) -> SignalEvent:
    defaults = dict(
        symbol="BTC/USDT",
        direction=SignalDirection.LONG,
        strategy_name="momentum",
        confidence=0.85,
        stop_loss=Decimal("57000"),
        take_profit=Decimal("63000"),
        metadata={"rsi": 72, "reason": "breakout"},
    )
    defaults.update(overrides)
    return SignalEvent(**defaults)


# ---------------------------------------------------------------------------
# 1. Trades
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_get_trade(repo: Repository):
    """Save a trade from a FillEvent and retrieve it by symbol."""
    fill = _make_fill()
    saved = await repo.save_trade(fill)

    assert isinstance(saved, Trade)
    assert saved.id is not None

    trades = await repo.get_trades_by_symbol("BTC/USDT")
    assert len(trades) == 1

    trade = trades[0]
    assert trade.symbol == "BTC/USDT"
    assert trade.side == "buy"
    assert trade.quantity == Decimal("0.5")
    assert trade.price == Decimal("60000")
    assert trade.fee == Decimal("30")
    assert trade.fee_currency == "USDT"
    assert trade.strategy_name == "momentum"
    assert trade.exchange_order_id == "exch-001"
    assert trade.client_order_id == "cli-001"


# ---------------------------------------------------------------------------
# 2. Orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_order(repo: Repository):
    """Save an order from an OrderEvent and verify its fields."""
    order_evt = _make_order()
    saved = await repo.save_order(order_evt)

    assert isinstance(saved, Order)
    assert saved.id is not None
    assert saved.symbol == "BTC/USDT"
    assert saved.side == "buy"
    assert saved.order_type == "limit"
    assert saved.quantity == Decimal("1.0")
    assert saved.price == Decimal("59000")
    assert saved.stop_loss == Decimal("57000")
    assert saved.take_profit == Decimal("63000")
    assert saved.status == "pending"
    assert saved.strategy_name == "momentum"
    assert saved.client_order_id == "ord-001"


@pytest.mark.asyncio
async def test_update_order_status(repo: Repository):
    """Save an order, update its status to filled, and verify."""
    order_evt = _make_order()
    saved = await repo.save_order(order_evt)
    assert saved.status == "pending"

    await repo.update_order(
        "ord-001",
        status="filled",
        exchange_order_id="exch-fill-001",
    )

    # Retrieve open orders -- should be empty since the order is now filled.
    open_orders = await repo.get_open_orders(symbol="BTC/USDT")
    assert len(open_orders) == 0


@pytest.mark.asyncio
async def test_get_open_orders(repo: Repository):
    """Only orders with status pending or open are returned."""
    # Create several orders with varying statuses.
    await repo.save_order(_make_order(client_order_id="o-pending"))
    await repo.save_order(_make_order(client_order_id="o-open"))
    await repo.save_order(_make_order(client_order_id="o-filled"))
    await repo.save_order(_make_order(client_order_id="o-cancelled"))

    # Move some to non-open states.
    await repo.update_order("o-open", status="open")
    await repo.update_order("o-filled", status="filled")
    await repo.update_order("o-cancelled", status="cancelled")

    open_orders = await repo.get_open_orders()
    client_ids = {o.client_order_id for o in open_orders}
    assert client_ids == {"o-pending", "o-open"}


# ---------------------------------------------------------------------------
# 3. Signal Logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_signal_log(repo: Repository):
    """Save a SignalEvent and verify the stored log including metadata_json."""
    signal = _make_signal()
    saved = await repo.save_signal_log(signal)

    assert isinstance(saved, SignalLog)
    assert saved.id is not None
    assert saved.symbol == "BTC/USDT"
    assert saved.direction == "long"
    assert saved.strategy_name == "momentum"
    assert saved.confidence == Decimal("0.85")
    assert saved.stop_loss == Decimal("57000")
    assert saved.take_profit == Decimal("63000")
    assert saved.event_id == signal.event_id

    # Verify the JSON metadata round-trips correctly.
    assert saved.metadata_json is not None
    meta = json.loads(saved.metadata_json)
    assert meta["rsi"] == 72
    assert meta["reason"] == "breakout"


# ---------------------------------------------------------------------------
# 4. Daily PnL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_daily_pnl_insert(repo: Repository):
    """Insert a new DailyPnL row and retrieve it."""
    row = await repo.save_daily_pnl(
        trade_date=date(2025, 6, 1),
        symbol="BTC/USDT",
        realized_pnl=Decimal("150.50"),
        unrealized_pnl=Decimal("-20.00"),
        fees=Decimal("5.25"),
        trade_count=7,
    )

    assert isinstance(row, DailyPnL)
    assert row.id is not None
    assert row.trade_date == date(2025, 6, 1)
    assert row.symbol == "BTC/USDT"
    assert row.realized_pnl == Decimal("150.50")
    assert row.unrealized_pnl == Decimal("-20.00")
    assert row.fees == Decimal("5.25")
    assert row.trade_count == 7

    # Also verify via get_daily_pnl.
    rows = await repo.get_daily_pnl(symbol="BTC/USDT")
    assert len(rows) == 1
    assert rows[0].id == row.id


@pytest.mark.asyncio
async def test_save_daily_pnl_upsert(repo: Repository):
    """Saving the same date+symbol twice should update, not duplicate."""
    await repo.save_daily_pnl(
        trade_date=date(2025, 6, 1),
        symbol="ETH/USDT",
        realized_pnl=Decimal("100"),
        trade_count=3,
    )

    # Save again with different values.
    updated = await repo.save_daily_pnl(
        trade_date=date(2025, 6, 1),
        symbol="ETH/USDT",
        realized_pnl=Decimal("250"),
        unrealized_pnl=Decimal("10"),
        fees=Decimal("2"),
        trade_count=5,
    )

    rows = await repo.get_daily_pnl(symbol="ETH/USDT")
    assert len(rows) == 1, "Upsert should not create a duplicate row"
    assert rows[0].realized_pnl == Decimal("250")
    assert rows[0].unrealized_pnl == Decimal("10")
    assert rows[0].fees == Decimal("2")
    assert rows[0].trade_count == 5
    assert rows[0].id == updated.id


# ---------------------------------------------------------------------------
# 5. Date/filter queries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_trades_with_date_filter(repo: Repository):
    """The `since` parameter on get_trades_by_symbol filters by executed_at."""
    ts_early = int(datetime(2025, 5, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    ts_late = int(datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    await repo.save_trade(_make_fill(client_order_id="t-early", exchange_order_id="e-1", timestamp=ts_early))
    await repo.save_trade(_make_fill(client_order_id="t-late", exchange_order_id="e-2", timestamp=ts_late))

    # Fetch all -- should be 2.
    all_trades = await repo.get_trades_by_symbol("BTC/USDT")
    assert len(all_trades) == 2

    # Fetch with since=June 1 -- only the late trade should come back.
    cutoff = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    filtered = await repo.get_trades_by_symbol("BTC/USDT", since=cutoff)
    assert len(filtered) == 1
    assert filtered[0].client_order_id == "t-late"


@pytest.mark.asyncio
async def test_get_daily_pnl_with_filters(repo: Repository):
    """Test symbol, since, and until filters on get_daily_pnl."""
    # Seed data: 3 dates x 2 symbols = 6 rows.
    dates = [date(2025, 6, 1), date(2025, 6, 2), date(2025, 6, 3)]
    symbols = ["BTC/USDT", "ETH/USDT"]

    for d in dates:
        for sym in symbols:
            await repo.save_daily_pnl(
                trade_date=d,
                symbol=sym,
                realized_pnl=Decimal("100"),
                trade_count=1,
            )

    # No filters -- all 6 rows.
    all_rows = await repo.get_daily_pnl()
    assert len(all_rows) == 6

    # Filter by symbol.
    btc_rows = await repo.get_daily_pnl(symbol="BTC/USDT")
    assert len(btc_rows) == 3
    assert all(r.symbol == "BTC/USDT" for r in btc_rows)

    # Filter by since.
    since_rows = await repo.get_daily_pnl(since=date(2025, 6, 2))
    assert len(since_rows) == 4  # June 2 + June 3, 2 symbols each

    # Filter by until.
    until_rows = await repo.get_daily_pnl(until=date(2025, 6, 2))
    assert len(until_rows) == 4  # June 1 + June 2, 2 symbols each

    # Combined: symbol + since + until.
    combined = await repo.get_daily_pnl(
        symbol="ETH/USDT",
        since=date(2025, 6, 2),
        until=date(2025, 6, 2),
    )
    assert len(combined) == 1
    assert combined[0].trade_date == date(2025, 6, 2)
    assert combined[0].symbol == "ETH/USDT"
