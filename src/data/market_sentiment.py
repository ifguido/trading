"""Market sentiment data: funding rates + whale detection.

Periodically fetches funding rates from Binance Futures (public API)
and monitors recent trades for whale activity. Publishes sentiment
scores that the strategy can use as an additional voting signal.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal

import ccxt.async_support as ccxt

from src.core.event_bus import EventBus
from src.core.events import Event, _utc_ms

logger = logging.getLogger(__name__)

# Mapping from spot symbols to futures symbols for funding rate lookup
_FUTURES_MAP = {
    "BTC/USDC": "BTC/USDT:USDT",
    "ETH/USDC": "ETH/USDT:USDT",
    "DOGE/USDC": "DOGE/USDT:USDT",
    "SOL/USDC": "SOL/USDT:USDT",
    "BNB/USDC": "BNB/USDT:USDT",
    "XRP/USDC": "XRP/USDT:USDT",
    "ADA/USDC": "ADA/USDT:USDT",
    "AVAX/USDC": "AVAX/USDT:USDT",
}

# Whale threshold in USD — trades above this are considered whale activity
_WHALE_THRESHOLD_USD = Decimal("50000")

# Number of recent trades to analyze for whale detection
_WHALE_TRADES_LIMIT = 200


@dataclass(frozen=True, slots=True)
class SentimentEvent(Event):
    """Market sentiment data for a symbol."""
    symbol: str = ""
    funding_rate: float = 0.0          # Current funding rate (positive = longs pay shorts)
    whale_buy_volume: Decimal = Decimal(0)   # Large buy volume in USD
    whale_sell_volume: Decimal = Decimal(0)  # Large sell volume in USD
    whale_bias: float = 0.0            # -1.0 (all sells) to +1.0 (all buys)
    sentiment_score: float = 0.0       # Combined score: -1.0 bearish to +1.0 bullish
    timestamp: int = field(default_factory=_utc_ms)


class MarketSentimentFeed:
    """Fetches funding rates and whale activity periodically.

    Parameters
    ----------
    event_bus : EventBus
        Bus to publish SentimentEvents.
    symbols : list[str]
        Spot trading symbols to monitor.
    poll_interval : int
        Seconds between each poll (default: 300 = 5 minutes).
    """

    def __init__(
        self,
        event_bus: EventBus,
        symbols: list[str],
        poll_interval: int = 300,
    ) -> None:
        self._bus = event_bus
        self._symbols = symbols
        self._poll_interval = poll_interval
        self._exchange: ccxt.binance | None = None
        self._running = False
        self._task: asyncio.Task | None = None

        # Cache last sentiment per symbol
        self.latest: dict[str, SentimentEvent] = {}

    async def start(self) -> None:
        """Start the sentiment feed."""
        self._exchange = ccxt.binance({"enableRateLimit": True})
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "MarketSentimentFeed started for %s (interval=%ds)",
            ", ".join(self._symbols), self._poll_interval,
        )

    async def stop(self) -> None:
        """Stop the sentiment feed."""
        self._running = False
        if self._task:
            self._task.cancel()
        if self._exchange:
            await self._exchange.close()
        logger.info("MarketSentimentFeed stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._fetch_all()
            except Exception as e:
                logger.error("Sentiment fetch error: %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _fetch_all(self) -> None:
        """Fetch sentiment for all symbols."""
        for symbol in self._symbols:
            try:
                event = await self._fetch_symbol(symbol)
                self.latest[symbol] = event
                await self._bus.publish(event)
            except Exception as e:
                logger.warning("Sentiment fetch failed for %s: %s", symbol, e)

    async def _fetch_symbol(self, symbol: str) -> SentimentEvent:
        """Fetch funding rate and whale activity for one symbol."""
        assert self._exchange is not None

        # 1. Funding rate from futures
        funding_rate = 0.0
        futures_symbol = _FUTURES_MAP.get(symbol)
        if futures_symbol:
            try:
                fr = await self._exchange.fetch_funding_rate(futures_symbol)
                funding_rate = float(fr.get("fundingRate", 0) or 0)
            except Exception as e:
                logger.debug("Funding rate unavailable for %s: %s", symbol, e)

        # 2. Whale detection from recent trades
        whale_buy = Decimal(0)
        whale_sell = Decimal(0)
        try:
            trades = await self._exchange.fetch_trades(symbol, limit=_WHALE_TRADES_LIMIT)
            for t in trades:
                volume_usd = Decimal(str(t["amount"])) * Decimal(str(t["price"]))
                if volume_usd >= _WHALE_THRESHOLD_USD:
                    if t["side"] == "buy":
                        whale_buy += volume_usd
                    else:
                        whale_sell += volume_usd
        except Exception as e:
            logger.debug("Trade fetch failed for %s: %s", symbol, e)

        # 3. Calculate whale bias (-1 to +1)
        total_whale = whale_buy + whale_sell
        if total_whale > 0:
            whale_bias = float((whale_buy - whale_sell) / total_whale)
        else:
            whale_bias = 0.0

        # 4. Combined sentiment score
        # Funding rate: positive = too many longs = contrarian bearish
        # We invert it: high positive funding = bearish signal
        funding_signal = -funding_rate * 1000  # Scale up, invert
        funding_signal = max(-1.0, min(1.0, funding_signal))

        # Combine: 40% whale bias + 60% funding contrarian signal
        sentiment_score = 0.4 * whale_bias + 0.6 * funding_signal
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        logger.info(
            "Sentiment %s: funding=%.6f whale_buy=$%.0f whale_sell=$%.0f bias=%.2f score=%.2f",
            symbol, funding_rate, whale_buy, whale_sell, whale_bias, sentiment_score,
        )

        return SentimentEvent(
            symbol=symbol,
            funding_rate=funding_rate,
            whale_buy_volume=whale_buy,
            whale_sell_volume=whale_sell,
            whale_bias=whale_bias,
            sentiment_score=sentiment_score,
        )
