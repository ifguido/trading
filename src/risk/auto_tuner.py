"""Auto-tuner: analyzes recent trades and adjusts strategy parameters.

Runs every TUNING_INTERVAL hours, reads the trade history from the DB,
computes performance metrics, and adjusts parameters within safe bounds.
All changes are logged to the tuning_logs table.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

# Safe parameter bounds
_BOUNDS = {
    "min_confidence": (0.15, 0.45),
    "trailing_pct": (0.02, 0.08),
    "stop_loss_pct": (0.01, 0.05),
    "max_position_pct": (0.15, 0.50),
}

TUNING_INTERVAL_HOURS = 6


@dataclass
class TradeMetrics:
    """Performance metrics computed from recent trades."""
    total_trades: int = 0
    buys: int = 0
    sells: int = 0
    completed_rounds: int = 0  # buy+sell pairs
    winning_rounds: int = 0
    losing_rounds: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    hours_since_last_trade: float = 0.0


def _compute_metrics(db_path: str) -> TradeMetrics:
    """Compute trading metrics from the database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM trades")
    total = c.fetchone()[0]

    if total == 0:
        conn.close()
        return TradeMetrics()

    c.execute("SELECT COUNT(*) FROM trades WHERE side='buy'")
    buys = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trades WHERE side='sell'")
    sells = c.fetchone()[0]

    # Get all trades ordered by time
    c.execute("SELECT symbol, side, quantity, price, executed_at FROM trades ORDER BY executed_at")
    trades = c.fetchall()

    # Match buy/sell rounds per symbol
    open_buys: dict[str, list] = {}
    rounds: list[dict] = []

    for symbol, side, qty, price, exec_at in trades:
        if side == "buy":
            if symbol not in open_buys:
                open_buys[symbol] = []
            open_buys[symbol].append({"qty": qty, "price": price, "at": exec_at})
        elif side == "sell" and symbol in open_buys and open_buys[symbol]:
            buy = open_buys[symbol].pop(0)
            pnl = (price - buy["price"]) * min(qty, buy["qty"])
            rounds.append({"symbol": symbol, "pnl": pnl, "buy_price": buy["price"], "sell_price": price})

    completed = len(rounds)
    wins = [r for r in rounds if r["pnl"] > 0]
    losses = [r for r in rounds if r["pnl"] <= 0]

    # Hours since last trade
    c.execute("SELECT executed_at FROM trades ORDER BY executed_at DESC LIMIT 1")
    last = c.fetchone()
    hours_since = 0.0
    if last and last[0]:
        try:
            last_dt = datetime.fromisoformat(last[0])
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            hours_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
        except Exception:
            pass

    conn.close()

    return TradeMetrics(
        total_trades=total,
        buys=buys,
        sells=sells,
        completed_rounds=completed,
        winning_rounds=len(wins),
        losing_rounds=len(losses),
        total_pnl=sum(r["pnl"] for r in rounds),
        avg_win=sum(r["pnl"] for r in wins) / len(wins) if wins else 0.0,
        avg_loss=sum(r["pnl"] for r in losses) / len(losses) if losses else 0.0,
        win_rate=len(wins) / completed if completed > 0 else 0.0,
        hours_since_last_trade=hours_since,
    )


def _decide_adjustments(
    metrics: TradeMetrics, current: dict[str, float]
) -> tuple[dict[str, float], list[str]]:
    """Decide parameter adjustments based on metrics. Returns new params and reasons."""
    new = dict(current)
    reasons: list[str] = []

    # Rule 1: No trades in 24h+ -> lower confidence to generate more signals
    if metrics.hours_since_last_trade > 24 and metrics.total_trades < 3:
        old = new["min_confidence"]
        new["min_confidence"] = max(_BOUNDS["min_confidence"][0], old - 0.03)
        if new["min_confidence"] != old:
            reasons.append(f"No trades in {metrics.hours_since_last_trade:.0f}h -> lowered confidence {old:.2f} -> {new['min_confidence']:.2f}")

    # Rule 2: Win rate below 35% -> increase confidence (be more selective)
    if metrics.completed_rounds >= 3 and metrics.win_rate < 0.35:
        old = new["min_confidence"]
        new["min_confidence"] = min(_BOUNDS["min_confidence"][1], old + 0.03)
        if new["min_confidence"] != old:
            reasons.append(f"Low win rate {metrics.win_rate:.0%} -> raised confidence {old:.2f} -> {new['min_confidence']:.2f}")

    # Rule 3: Win rate above 65% -> decrease confidence (trade more)
    if metrics.completed_rounds >= 3 and metrics.win_rate > 0.65:
        old = new["min_confidence"]
        new["min_confidence"] = max(_BOUNDS["min_confidence"][0], old - 0.02)
        if new["min_confidence"] != old:
            reasons.append(f"High win rate {metrics.win_rate:.0%} -> lowered confidence {old:.2f} -> {new['min_confidence']:.2f}")

    # Rule 4: Average loss bigger than average win -> tighten stop loss
    if metrics.completed_rounds >= 3 and metrics.avg_loss != 0:
        if abs(metrics.avg_loss) > abs(metrics.avg_win) * 1.5:
            old = new["stop_loss_pct"]
            new["stop_loss_pct"] = max(_BOUNDS["stop_loss_pct"][0], old - 0.005)
            if new["stop_loss_pct"] != old:
                reasons.append(f"Avg loss ${metrics.avg_loss:.2f} > avg win ${metrics.avg_win:.2f} -> tightened SL {old:.3f} -> {new['stop_loss_pct']:.3f}")

    # Rule 5: Winning trades with small gains -> widen trailing stop
    if metrics.completed_rounds >= 3 and metrics.win_rate > 0.5:
        if metrics.avg_win < 1.0:  # Winning less than $1 avg
            old = new["trailing_pct"]
            new["trailing_pct"] = min(_BOUNDS["trailing_pct"][1], old + 0.005)
            if new["trailing_pct"] != old:
                reasons.append(f"Small avg wins ${metrics.avg_win:.2f} -> widened trailing {old:.3f} -> {new['trailing_pct']:.3f}")

    # Rule 6: Losing money overall -> reduce position size
    if metrics.completed_rounds >= 3 and metrics.total_pnl < -5:
        old = new["max_position_pct"]
        new["max_position_pct"] = max(_BOUNDS["max_position_pct"][0], old - 0.05)
        if new["max_position_pct"] != old:
            reasons.append(f"Negative PnL ${metrics.total_pnl:.2f} -> reduced position size {old:.2f} -> {new['max_position_pct']:.2f}")

    # Rule 7: Making money -> can increase position size slightly
    if metrics.completed_rounds >= 5 and metrics.total_pnl > 5 and metrics.win_rate > 0.55:
        old = new["max_position_pct"]
        new["max_position_pct"] = min(_BOUNDS["max_position_pct"][1], old + 0.03)
        if new["max_position_pct"] != old:
            reasons.append(f"Profitable ${metrics.total_pnl:.2f} with {metrics.win_rate:.0%} WR -> increased position size {old:.2f} -> {new['max_position_pct']:.2f}")

    if not reasons:
        reasons.append("No adjustments needed — parameters are within optimal range for current performance")

    return new, reasons


def _log_tuning(db_path: str, summary: str, before: dict, after: dict, metrics: TradeMetrics) -> None:
    """Write tuning log to database."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tuning_logs (id INTEGER PRIMARY KEY, summary TEXT, parameters_before TEXT, parameters_after TEXT, metrics TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO tuning_logs (summary, parameters_before, parameters_after, metrics, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            summary,
            json.dumps(before),
            json.dumps(after),
            json.dumps({
                "total_trades": metrics.total_trades,
                "completed_rounds": metrics.completed_rounds,
                "win_rate": round(metrics.win_rate, 3),
                "total_pnl": round(metrics.total_pnl, 4),
                "avg_win": round(metrics.avg_win, 4),
                "avg_loss": round(metrics.avg_loss, 4),
                "hours_since_last_trade": round(metrics.hours_since_last_trade, 1),
            }),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


class AutoTuner:
    """Periodically analyzes performance and adjusts strategy parameters.

    Parameters
    ----------
    engine : Any
        Reference to the Engine for accessing strategy and risk config.
    db_path : str
        Path to the SQLite database.
    interval_hours : float
        Hours between tuning cycles.
    """

    def __init__(self, engine: Any, db_path: str = "data/cryptotrader.db", interval_hours: float = TUNING_INTERVAL_HOURS) -> None:
        self._engine = engine
        self._db_path = db_path
        self._interval = interval_hours * 3600
        self._running = False
        self._task: asyncio.Task | None = None

    def _get_current_params(self) -> dict[str, float]:
        """Read current parameters from the strategy and risk config."""
        params: dict[str, float] = {
            "min_confidence": 0.25,
            "trailing_pct": 0.03,
            "stop_loss_pct": 0.02,
            "max_position_pct": 0.40,
        }

        # Read from strategy if available
        if hasattr(self._engine, '_strategy_manager') and self._engine._strategy_manager:
            for s in self._engine._strategy_manager._strategies:
                if hasattr(s, '_min_confidence'):
                    params["min_confidence"] = s._min_confidence
                if hasattr(s, '_stop_loss_pct'):
                    params["stop_loss_pct"] = float(s._stop_loss_pct)
                break

        if hasattr(self._engine, '_trailing_stop') and self._engine._trailing_stop:
            params["trailing_pct"] = float(self._engine._trailing_stop._trailing_pct)

        if hasattr(self._engine, '_position_sizer') and self._engine._position_sizer:
            params["max_position_pct"] = float(self._engine._position_sizer._max_position_pct)

        return params

    def _apply_params(self, params: dict[str, float]) -> None:
        """Apply new parameters to the running strategy."""
        if hasattr(self._engine, '_strategy_manager') and self._engine._strategy_manager:
            for s in self._engine._strategy_manager._strategies:
                if hasattr(s, '_min_confidence'):
                    s._min_confidence = params["min_confidence"]
                if hasattr(s, '_stop_loss_pct'):
                    s._stop_loss_pct = Decimal(str(params["stop_loss_pct"]))
                break

        if hasattr(self._engine, '_trailing_stop') and self._engine._trailing_stop:
            self._engine._trailing_stop._trailing_pct = Decimal(str(params["trailing_pct"]))

        if hasattr(self._engine, '_position_sizer') and self._engine._position_sizer:
            self._engine._position_sizer._max_position_pct = Decimal(str(params["max_position_pct"]))

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("AutoTuner started (interval=%.1fh)", self._interval / 3600)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        # Wait a bit before first run
        await asyncio.sleep(60)
        while self._running:
            try:
                await self._tune()
            except Exception as e:
                logger.error("AutoTuner error: %s", e)
            await asyncio.sleep(self._interval)

    async def _tune(self) -> None:
        """Run one tuning cycle."""
        metrics = _compute_metrics(self._db_path)
        current = self._get_current_params()
        new_params, reasons = _decide_adjustments(metrics, current)

        summary = " | ".join(reasons)
        changed = any(abs(new_params[k] - current[k]) > 0.0001 for k in current)

        if changed:
            self._apply_params(new_params)
            logger.info("AutoTuner ADJUSTED: %s", summary)
        else:
            logger.info("AutoTuner: no changes needed (trades=%d, WR=%.0f%%, PnL=$%.2f)",
                       metrics.total_trades, metrics.win_rate * 100, metrics.total_pnl)

        _log_tuning(self._db_path, summary, current, new_params, metrics)
