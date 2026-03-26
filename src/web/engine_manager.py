"""State machine that wraps the Engine for web control.

States: IDLE -> STARTING -> RUNNING -> STOPPING -> IDLE  (+ ERROR)
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EngineState(str, Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class EngineManager:
    """Manages the Engine lifecycle from the web UI."""

    def __init__(self) -> None:
        self.state: EngineState = EngineState.IDLE
        self.engine: Any = None
        self._task: asyncio.Task | None = None
        self._started_at: float = 0.0
        self.last_error: str | None = None
        self._initial_capital: float = 10000.0

    @property
    def uptime(self) -> float:
        if self.state == EngineState.RUNNING and self._started_at:
            return time.time() - self._started_at
        return 0.0

    async def start(self, config: Any) -> None:
        """Create and start the Engine as an asyncio task."""
        if self.state not in (EngineState.IDLE, EngineState.ERROR):
            raise RuntimeError(f"Cannot start engine in state {self.state}")

        self.state = EngineState.STARTING
        self.last_error = None

        try:
            from src.core.engine import Engine

            self.engine = Engine(config)
            # Tell engine not to register OS signal handlers (web process owns them)
            self.engine._skip_signal_handlers = True

            self._task = asyncio.create_task(self._run_engine())
            self.state = EngineState.RUNNING
            self._started_at = time.time()
            logger.info("Engine started via web UI")
        except Exception as exc:
            self.state = EngineState.ERROR
            self.last_error = str(exc)
            logger.exception("Failed to start engine")
            raise

    async def _run_engine(self) -> None:
        """Wrapper that runs engine.start() and catches crashes."""
        try:
            await self.engine.start()
        except Exception as exc:
            logger.exception("Engine crashed")
            self.state = EngineState.ERROR
            self.last_error = str(exc)
        finally:
            if self.state == EngineState.RUNNING:
                self.state = EngineState.IDLE

    async def stop(self) -> None:
        """Gracefully stop the engine."""
        if self.state != EngineState.RUNNING:
            raise RuntimeError(f"Cannot stop engine in state {self.state}")

        self.state = EngineState.STOPPING
        try:
            # Trigger the engine's shutdown event
            if self.engine and hasattr(self.engine, "_shutdown_event"):
                self.engine._shutdown_event.set()
            # Wait for the task to finish
            if self._task and not self._task.done():
                await asyncio.wait_for(self._task, timeout=30)
        except asyncio.TimeoutError:
            logger.warning("Engine stop timed out, cancelling task")
            if self._task:
                self._task.cancel()
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("Error stopping engine")
        finally:
            self.state = EngineState.IDLE
            self.engine = None
            self._task = None
            self._started_at = 0.0
            logger.info("Engine stopped via web UI")
