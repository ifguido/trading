"""Comprehensive unit tests for src.core.event_bus.EventBus."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from src.core.event_bus import EventBus
from src.core.events import Event, FillEvent, Side, TickEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyEvent(Event):
    """Minimal concrete event for testing."""
    pass


class _OtherEvent(Event):
    """A second distinct event type."""
    pass


# ---------------------------------------------------------------------------
# 1. Subscribe and receive events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_and_receive_event():
    """A single subscriber should receive the published event."""
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(_DummyEvent, handler, name="test_handler")

    evt = _DummyEvent()
    await bus.publish(evt)

    assert len(received) == 1
    assert received[0] is evt


@pytest.mark.asyncio
async def test_subscribe_receives_only_matching_event_type():
    """Subscribers should only get events of the type they subscribed to."""
    bus = EventBus()
    dummy_received: list[Event] = []
    other_received: list[Event] = []

    async def dummy_handler(event: Event) -> None:
        dummy_received.append(event)

    async def other_handler(event: Event) -> None:
        other_received.append(event)

    bus.subscribe(_DummyEvent, dummy_handler)
    bus.subscribe(_OtherEvent, other_handler)

    await bus.publish(_DummyEvent())
    await bus.publish(_OtherEvent())
    await bus.publish(_DummyEvent())

    assert len(dummy_received) == 2
    assert len(other_received) == 1


@pytest.mark.asyncio
async def test_publish_with_no_subscribers_does_not_raise():
    """Publishing an event with zero subscribers should be a no-op."""
    bus = EventBus()
    await bus.publish(_DummyEvent())  # should not raise


# ---------------------------------------------------------------------------
# 2. Multiple subscribers to same event type
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_subscribers_all_receive_event():
    """All subscribers to the same event type should receive the event."""
    bus = EventBus()
    results: list[str] = []

    async def handler_a(event: Event) -> None:
        results.append("a")

    async def handler_b(event: Event) -> None:
        results.append("b")

    async def handler_c(event: Event) -> None:
        results.append("c")

    bus.subscribe(_DummyEvent, handler_a, name="a")
    bus.subscribe(_DummyEvent, handler_b, name="b")
    bus.subscribe(_DummyEvent, handler_c, name="c")

    await bus.publish(_DummyEvent())

    assert sorted(results) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_multiple_subscribers_each_gets_same_event_instance():
    """Every handler should receive the exact same event object."""
    bus = EventBus()
    received: list[Event] = []

    async def handler_1(event: Event) -> None:
        received.append(event)

    async def handler_2(event: Event) -> None:
        received.append(event)

    bus.subscribe(_DummyEvent, handler_1)
    bus.subscribe(_DummyEvent, handler_2)

    evt = _DummyEvent()
    await bus.publish(evt)

    assert len(received) == 2
    assert all(e is evt for e in received)


# ---------------------------------------------------------------------------
# 3. Unsubscribe stops receiving
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsubscribe_stops_receiving_events():
    """After unsubscribing, a handler should no longer receive events."""
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(_DummyEvent, handler, name="removable")

    await bus.publish(_DummyEvent())
    assert len(received) == 1

    bus.unsubscribe(_DummyEvent, handler)

    await bus.publish(_DummyEvent())
    assert len(received) == 1  # no new events


@pytest.mark.asyncio
async def test_unsubscribe_only_removes_specific_handler():
    """Unsubscribing one handler should not affect other handlers."""
    bus = EventBus()
    results_a: list[str] = []
    results_b: list[str] = []

    async def handler_a(event: Event) -> None:
        results_a.append("a")

    async def handler_b(event: Event) -> None:
        results_b.append("b")

    bus.subscribe(_DummyEvent, handler_a, name="a")
    bus.subscribe(_DummyEvent, handler_b, name="b")

    bus.unsubscribe(_DummyEvent, handler_a)

    await bus.publish(_DummyEvent())

    assert len(results_a) == 0
    assert len(results_b) == 1


@pytest.mark.asyncio
async def test_unsubscribe_nonexistent_handler_is_safe():
    """Unsubscribing a handler that was never subscribed should not raise."""
    bus = EventBus()

    async def handler(event: Event) -> None:
        pass

    # Should not raise even though handler was never subscribed
    bus.unsubscribe(_DummyEvent, handler)


# ---------------------------------------------------------------------------
# 4. Error in one handler doesn't affect others
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_in_handler_does_not_affect_others():
    """If one handler raises, the remaining handlers still execute."""
    bus = EventBus()
    results: list[str] = []

    async def failing_handler(event: Event) -> None:
        raise RuntimeError("boom")

    async def healthy_handler(event: Event) -> None:
        results.append("ok")

    bus.subscribe(_DummyEvent, failing_handler, name="failing")
    bus.subscribe(_DummyEvent, healthy_handler, name="healthy")

    await bus.publish(_DummyEvent())

    assert results == ["ok"]


@pytest.mark.asyncio
async def test_multiple_failing_handlers_dont_propagate():
    """Even if multiple handlers fail, no exception propagates to caller."""
    bus = EventBus()
    results: list[str] = []

    async def fail_1(event: Event) -> None:
        raise ValueError("fail_1")

    async def fail_2(event: Event) -> None:
        raise TypeError("fail_2")

    async def success(event: Event) -> None:
        results.append("success")

    bus.subscribe(_DummyEvent, fail_1)
    bus.subscribe(_DummyEvent, fail_2)
    bus.subscribe(_DummyEvent, success)

    await bus.publish(_DummyEvent())

    assert results == ["success"]


# ---------------------------------------------------------------------------
# 5. publish_nowait works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_nowait_fires_handlers():
    """publish_nowait should fire handlers as background tasks."""
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(_DummyEvent, handler, name="nowait_handler")

    evt = _DummyEvent()
    await bus.publish_nowait(evt)

    # Give tasks a moment to complete
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0] is evt


@pytest.mark.asyncio
async def test_publish_nowait_creates_tasks():
    """publish_nowait should create asyncio tasks that are tracked."""
    bus = EventBus()

    async def slow_handler(event: Event) -> None:
        await asyncio.sleep(10)

    bus.subscribe(_DummyEvent, slow_handler, name="slow")

    await bus.publish_nowait(_DummyEvent())

    # There should be at least one tracked task
    assert len(bus._tasks) >= 1

    # Clean up
    await bus.shutdown()


@pytest.mark.asyncio
async def test_publish_nowait_error_in_handler_does_not_propagate():
    """Errors in handlers launched by publish_nowait are swallowed."""
    bus = EventBus()
    results: list[str] = []

    async def failing(event: Event) -> None:
        raise RuntimeError("nowait boom")

    async def working(event: Event) -> None:
        results.append("done")

    bus.subscribe(_DummyEvent, failing, name="failing")
    bus.subscribe(_DummyEvent, working, name="working")

    await bus.publish_nowait(_DummyEvent())
    await asyncio.sleep(0.05)

    assert results == ["done"]


@pytest.mark.asyncio
async def test_publish_nowait_with_no_subscribers():
    """publish_nowait with no subscribers should be a no-op."""
    bus = EventBus()
    await bus.publish_nowait(_DummyEvent())  # should not raise
    assert len(bus._tasks) == 0


# ---------------------------------------------------------------------------
# 6. Subscriber count
# ---------------------------------------------------------------------------


def test_subscriber_count_zero_initially():
    """subscriber_count should return 0 for unseen event types."""
    bus = EventBus()
    assert bus.subscriber_count(_DummyEvent) == 0


def test_subscriber_count_after_subscribe():
    """subscriber_count should reflect the number of registered handlers."""
    bus = EventBus()

    async def h1(event: Event) -> None:
        pass

    async def h2(event: Event) -> None:
        pass

    bus.subscribe(_DummyEvent, h1, name="h1")
    assert bus.subscriber_count(_DummyEvent) == 1

    bus.subscribe(_DummyEvent, h2, name="h2")
    assert bus.subscriber_count(_DummyEvent) == 2


def test_subscriber_count_after_unsubscribe():
    """subscriber_count should decrease after unsubscribing."""
    bus = EventBus()

    async def h1(event: Event) -> None:
        pass

    async def h2(event: Event) -> None:
        pass

    bus.subscribe(_DummyEvent, h1)
    bus.subscribe(_DummyEvent, h2)
    assert bus.subscriber_count(_DummyEvent) == 2

    bus.unsubscribe(_DummyEvent, h1)
    assert bus.subscriber_count(_DummyEvent) == 1


def test_subscriber_count_per_event_type():
    """subscriber_count should be tracked independently per event type."""
    bus = EventBus()

    async def h(event: Event) -> None:
        pass

    bus.subscribe(_DummyEvent, h)
    bus.subscribe(_DummyEvent, h)
    bus.subscribe(_OtherEvent, h)

    assert bus.subscriber_count(_DummyEvent) == 2
    assert bus.subscriber_count(_OtherEvent) == 1


# ---------------------------------------------------------------------------
# 7. Shutdown cancels tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_tasks():
    """shutdown() should cancel all pending handler tasks."""
    bus = EventBus()

    cancelled = False

    async def long_running(event: Event) -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            cancelled = True
            raise

    bus.subscribe(_DummyEvent, long_running, name="long")

    await bus.publish_nowait(_DummyEvent())
    assert len(bus._tasks) >= 1

    # Yield to the event loop so the task actually starts executing
    # (enters the asyncio.sleep(100) call inside the handler).
    await asyncio.sleep(0)

    await bus.shutdown()

    assert cancelled is True
    assert len(bus._tasks) == 0


@pytest.mark.asyncio
async def test_shutdown_when_no_tasks():
    """shutdown() with no pending tasks should complete cleanly."""
    bus = EventBus()
    await bus.shutdown()  # should not raise
    assert len(bus._tasks) == 0


@pytest.mark.asyncio
async def test_shutdown_handles_already_completed_tasks():
    """shutdown() should handle tasks that completed before shutdown."""
    bus = EventBus()

    async def fast_handler(event: Event) -> None:
        pass  # finishes instantly

    bus.subscribe(_DummyEvent, fast_handler, name="fast")
    await bus.publish_nowait(_DummyEvent())

    # Let the task finish
    await asyncio.sleep(0.05)

    # shutdown should still work even if tasks are done
    await bus.shutdown()
    assert len(bus._tasks) == 0


# ---------------------------------------------------------------------------
# Integration-ish: using real event types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_tick_event():
    """EventBus works with real TickEvent instances."""
    bus = EventBus()
    received: list[TickEvent] = []

    async def on_tick(event: TickEvent) -> None:
        received.append(event)

    bus.subscribe(TickEvent, on_tick, name="tick_listener")

    tick = TickEvent(
        symbol="BTC/USDT",
        bid=Decimal("50000.00"),
        ask=Decimal("50001.00"),
        last=Decimal("50000.50"),
        volume_24h=Decimal("1234.56"),
    )
    await bus.publish(tick)

    assert len(received) == 1
    assert received[0].symbol == "BTC/USDT"
    assert received[0].bid == Decimal("50000.00")


@pytest.mark.asyncio
async def test_with_fill_event():
    """EventBus works with real FillEvent instances."""
    bus = EventBus()
    received: list[FillEvent] = []

    async def on_fill(event: FillEvent) -> None:
        received.append(event)

    bus.subscribe(FillEvent, on_fill, name="fill_listener")

    fill = FillEvent(
        symbol="ETH/USDT",
        side=Side.BUY,
        quantity=Decimal("1.5"),
        price=Decimal("3000.00"),
        fee=Decimal("4.50"),
        fee_currency="USDT",
    )
    await bus.publish(fill)

    assert len(received) == 1
    assert received[0].symbol == "ETH/USDT"
    assert received[0].fee == Decimal("4.50")
