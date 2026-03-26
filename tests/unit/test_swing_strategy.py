"""Unit tests for SwingStrategy.

Covers:
1. Does not emit signals before min_candles reached
2. Ignores non-closed candles
3. Ignores symbols it's not tracking
4. After enough candles with uptrend data, emits LONG signal
5. After enough candles with downtrend data, emits SHORT signal
6. Does not emit duplicate signals (same direction twice)
"""

from __future__ import annotations

import random
from decimal import Decimal

import pytest
import pytest_asyncio

from src.core.event_bus import EventBus
from src.core.events import CandleEvent, SignalDirection, SignalEvent
from src.strategy.swing.swing_strategy import SwingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SignalCapture:
    """Subscribe to SignalEvent and collect all published signals."""

    def __init__(self) -> None:
        self.signals: list[SignalEvent] = []

    async def handler(self, event: SignalEvent) -> None:
        self.signals.append(event)


def _make_candle(
    symbol: str = "BTC/USDT",
    open_: Decimal = Decimal("30000"),
    high: Decimal = Decimal("30100"),
    low: Decimal = Decimal("29900"),
    close: Decimal = Decimal("30050"),
    volume: Decimal = Decimal("100"),
    closed: bool = True,
    timestamp: int = 0,
) -> CandleEvent:
    """Build a CandleEvent with sensible defaults."""
    return CandleEvent(
        symbol=symbol,
        timeframe="1h",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        closed=closed,
        timestamp=timestamp,
    )


def _generate_reversal_up_candles(symbol: str, count: int) -> list[CandleEvent]:
    """Generate candles that produce a LONG signal via indicator confluence.

    The data has three phases:
    1. Slow decline with noise (drags slow MA down, builds bearish baseline)
    2. Near-flat stabilisation (RSI recovers from oversold)
    3. Gradual ascent with pullbacks (fast MA crosses above slow MA while RSI
       is moderate, MACD histogram turns positive, and price sits near the
       lower Bollinger Band)

    Uses a fixed random seed for deterministic results.
    """
    rng = random.Random(42)
    candles: list[CandleEvent] = []
    prices: list[float] = []
    base = 30000.0

    # Phase 1: sideways-to-down (40% of candles)
    phase1 = int(count * 0.4)
    for _ in range(phase1):
        base += rng.gauss(-20, 50)
        prices.append(base)

    # Phase 2: flat bottom (10% of candles)
    phase2 = int(count * 0.1)
    for _ in range(phase2):
        base += rng.gauss(5, 40)
        prices.append(base)

    # Phase 3: gradual rise with pullbacks (50% of candles)
    phase3 = count - phase1 - phase2
    for i in range(phase3):
        if i % 4 == 3:
            base += rng.gauss(-40, 20)  # pullback keeps RSI moderate
        else:
            base += rng.gauss(50, 30)
        prices.append(base)

    for i, p in enumerate(prices):
        noise = rng.uniform(30, 80)
        candles.append(
            _make_candle(
                symbol=symbol,
                open_=Decimal(str(round(p - rng.uniform(-30, 30), 2))),
                high=Decimal(str(round(p + noise, 2))),
                low=Decimal(str(round(p - noise, 2))),
                close=Decimal(str(round(p, 2))),
                volume=Decimal("1000"),
                closed=True,
                timestamp=1_700_000_000_000 + i * 3_600_000,
            )
        )
    return candles


def _generate_reversal_down_candles(symbol: str, count: int) -> list[CandleEvent]:
    """Generate candles that produce a SHORT signal via indicator confluence.

    Mirror of ``_generate_reversal_up_candles``:
    1. Slow rise with noise (drags slow MA up, builds bullish baseline)
    2. Near-flat stabilisation at top
    3. Gradual descent with bounces (fast MA crosses below slow MA while RSI
       is moderate-high, MACD histogram turns negative)

    Uses a fixed random seed for deterministic results.
    """
    rng = random.Random(99)
    candles: list[CandleEvent] = []
    prices: list[float] = []
    base = 30000.0

    # Phase 1: sideways-to-up (40% of candles)
    phase1 = int(count * 0.4)
    for _ in range(phase1):
        base += rng.gauss(20, 50)
        prices.append(base)

    # Phase 2: flat top (10% of candles)
    phase2 = int(count * 0.1)
    for _ in range(phase2):
        base += rng.gauss(-5, 40)
        prices.append(base)

    # Phase 3: gradual decline with bounces (50% of candles)
    phase3 = count - phase1 - phase2
    for i in range(phase3):
        if i % 4 == 3:
            base += rng.gauss(40, 20)  # bounce keeps RSI moderate
        else:
            base += rng.gauss(-50, 30)
        prices.append(base)

    for i, p in enumerate(prices):
        noise = rng.uniform(30, 80)
        candles.append(
            _make_candle(
                symbol=symbol,
                open_=Decimal(str(round(p - rng.uniform(-30, 30), 2))),
                high=Decimal(str(round(p + noise, 2))),
                low=Decimal(str(round(p - noise, 2))),
                close=Decimal(str(round(p, 2))),
                volume=Decimal("1000"),
                closed=True,
                timestamp=1_700_000_000_000 + i * 3_600_000,
            )
        )
    return candles


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Use a small min_candles value so tests run faster while still being
# large enough for all indicator computations (slow_ma=30, macd_slow=26).
_TEST_MIN_CANDLES = 50


@pytest_asyncio.fixture
async def bus() -> EventBus:
    return EventBus()


@pytest_asyncio.fixture
async def capture(bus: EventBus) -> SignalCapture:
    cap = SignalCapture()
    bus.subscribe(SignalEvent, cap.handler, name="test_signal_capture")
    return cap


@pytest_asyncio.fixture
async def strategy(bus: EventBus, capture: SignalCapture) -> SwingStrategy:
    """Create a SwingStrategy with relaxed confidence so indicators can fire."""
    strat = SwingStrategy(
        name="test_swing",
        symbols=["BTC/USDT"],
        event_bus=bus,
        params={
            "min_candles": _TEST_MIN_CANDLES,
            "min_confidence": 0.25,  # lower threshold so trends trigger signals
            "fast_ma": 10,
            "slow_ma": 30,
        },
    )
    await strat.initialize()
    return strat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_signal_before_min_candles(
    strategy: SwingStrategy,
    capture: SignalCapture,
) -> None:
    """The strategy must not emit any signal until min_candles have been received."""
    # Feed fewer than min_candles (49 < 50) using simple ascending candles
    for i in range(_TEST_MIN_CANDLES - 1):
        price = 20000 + i * 100
        candle = _make_candle(
            symbol="BTC/USDT",
            close=Decimal(str(price)),
            closed=True,
            timestamp=1_700_000_000_000 + i * 3_600_000,
        )
        await strategy.on_candle(candle)

    assert len(capture.signals) == 0, (
        f"Expected no signals before min_candles, got {len(capture.signals)}"
    )


@pytest.mark.asyncio
async def test_non_closed_candles_ignored(
    strategy: SwingStrategy,
    capture: SignalCapture,
) -> None:
    """Candles with closed=False must be silently skipped."""
    # Feed more than enough candles but all marked as not-closed
    for i in range(80):
        candle = _make_candle(
            symbol="BTC/USDT",
            close=Decimal(str(30000 + i * 50)),
            closed=False,
            timestamp=1_700_000_000_000 + i * 3_600_000,
        )
        await strategy.on_candle(candle)

    assert len(capture.signals) == 0, (
        "Expected no signals from non-closed candles"
    )


@pytest.mark.asyncio
async def test_untracked_symbol_ignored(
    strategy: SwingStrategy,
    capture: SignalCapture,
) -> None:
    """Candles for a symbol not in the strategy's symbol list must be ignored."""
    candles = _generate_reversal_up_candles("ETH/USDT", count=100)
    for candle in candles:
        await strategy.on_candle(candle)

    assert len(capture.signals) == 0, (
        "Expected no signals for untracked symbol"
    )


@pytest.mark.asyncio
async def test_uptrend_emits_long_signal(
    strategy: SwingStrategy,
    capture: SignalCapture,
) -> None:
    """After enough candles with a reversal-up pattern, a LONG signal should be emitted."""
    candles = _generate_reversal_up_candles("BTC/USDT", count=100)
    for candle in candles:
        await strategy.on_candle(candle)

    # There should be at least one LONG signal emitted
    long_signals = [
        s for s in capture.signals if s.direction == SignalDirection.LONG
    ]
    assert len(long_signals) >= 1, (
        f"Expected at least one LONG signal from uptrend data, "
        f"got signals: {[(s.direction.value, s.confidence) for s in capture.signals]}"
    )

    sig = long_signals[0]
    assert sig.symbol == "BTC/USDT"
    assert sig.strategy_name == "test_swing"
    assert sig.stop_loss is not None
    assert sig.take_profit is not None
    assert sig.confidence > 0


@pytest.mark.asyncio
async def test_downtrend_emits_short_signal(
    strategy: SwingStrategy,
    capture: SignalCapture,
) -> None:
    """After enough candles with a reversal-down pattern, a SHORT signal should be emitted."""
    candles = _generate_reversal_down_candles("BTC/USDT", count=100)
    for candle in candles:
        await strategy.on_candle(candle)

    short_signals = [
        s for s in capture.signals if s.direction == SignalDirection.SHORT
    ]
    assert len(short_signals) >= 1, (
        f"Expected at least one SHORT signal from downtrend data, "
        f"got signals: {[(s.direction.value, s.confidence) for s in capture.signals]}"
    )

    sig = short_signals[0]
    assert sig.symbol == "BTC/USDT"
    assert sig.strategy_name == "test_swing"
    assert sig.stop_loss is not None
    assert sig.take_profit is not None
    assert sig.confidence > 0


@pytest.mark.asyncio
async def test_no_duplicate_signals(
    strategy: SwingStrategy,
    capture: SignalCapture,
) -> None:
    """The strategy must not emit the same signal direction twice in a row.

    Once a LONG is emitted, subsequent evaluations that still produce LONG
    must be suppressed until the direction changes.
    """
    candles = _generate_reversal_up_candles("BTC/USDT", count=120)
    for candle in candles:
        await strategy.on_candle(candle)

    # The dedup logic in SwingStrategy._evaluate prevents emitting the same
    # direction consecutively.  Verify that no two adjacent signals share
    # a direction.
    for idx in range(1, len(capture.signals)):
        prev_dir = capture.signals[idx - 1].direction
        curr_dir = capture.signals[idx].direction
        assert prev_dir != curr_dir, (
            f"Duplicate consecutive signal direction {curr_dir.value} "
            f"at index {idx}"
        )
