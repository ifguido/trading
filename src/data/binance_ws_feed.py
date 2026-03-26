"""Feed de datos en tiempo real de Binance mediante WebSockets de CCXT Pro.

Se suscribe a tickers, libro de ordenes y operaciones para los simbolos
configurados, y publica TickEvent y OrderBookEvent en el EventBus.
Incluye logica de reconexion automatica con retroceso exponencial
y soporte para apagado controlado (graceful shutdown).
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal, InvalidOperation
from typing import Any

import ccxt.pro as ccxtpro

from src.core.config_loader import ExchangeConfig, PairConfig
from src.core.event_bus import EventBus
from src.core.events import OrderBookEvent, TickEvent
from src.core.exceptions import ExchangeError

logger = logging.getLogger(__name__)

# Parametros de retroceso exponencial para reconexion
_INITIAL_RECONNECT_DELAY = 1.0  # Retraso inicial en segundos antes de reconectar
_MAX_RECONNECT_DELAY = 60.0  # Retraso maximo entre intentos de reconexion
_RECONNECT_BACKOFF_FACTOR = 2.0  # Factor multiplicador del retroceso exponencial

# Profundidad por defecto del libro de ordenes a solicitar
_ORDER_BOOK_DEPTH = 10


def _to_decimal(value: Any) -> Decimal:
    """Convierte un valor a Decimal de forma segura, retornando cero en caso de fallo.

    Parametros
    ----------
    value:
        Cualquier valor a convertir (puede ser None, str, int, float, etc.).

    Retorna
    -------
    Decimal
        El valor convertido, o Decimal(0) si la conversion falla.
    """
    if value is None:
        return Decimal(0)  # Valor nulo se convierte a cero
    try:
        # Convertir a string primero para evitar imprecision de punto flotante
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0)  # Ante cualquier error de conversion, retornar cero


class BinanceWebSocketFeed:
    """Feed de datos de mercado en tiempo real desde Binance via WebSockets de CCXT Pro.

    Esta clase gestiona la conexion WebSocket con Binance, se suscribe a
    actualizaciones de tickers y libros de ordenes para los pares configurados,
    y publica los eventos correspondientes en el EventBus central.
    Implementa reconexion automatica con retroceso exponencial ante fallos.

    Parametros
    ----------
    event_bus:
        EventBus central para publicar eventos de datos de mercado.
    exchange_config:
        Configuracion de conexion al exchange (API keys, sandbox, etc.).
    pairs:
        Lista de configuraciones de pares de trading a los que suscribirse.
    order_book_depth:
        Numero de niveles del libro de ordenes a solicitar (por defecto 10).
    """

    def __init__(
        self,
        event_bus: EventBus,
        exchange_config: ExchangeConfig,
        pairs: list[PairConfig],
        order_book_depth: int = _ORDER_BOOK_DEPTH,
    ) -> None:
        self._event_bus = event_bus  # Bus de eventos central para publicar datos
        self._exchange_config = exchange_config  # Configuracion del exchange
        self._pairs = pairs  # Configuraciones de pares de trading
        self._symbols = [p.symbol for p in pairs]  # Lista de simbolos extraidos de los pares
        self._order_book_depth = order_book_depth  # Profundidad del libro de ordenes

        self._exchange: ccxtpro.binance | None = None  # Instancia del cliente CCXT Pro
        self._tasks: list[asyncio.Task] = []  # Tareas asyncio de suscripcion activas
        self._running = False  # Bandera de estado de ejecucion
        self._reconnect_delay = _INITIAL_RECONNECT_DELAY  # Retraso actual de reconexion

    # ── Ciclo de Vida ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Conecta con Binance e inicia todos los bucles de suscripcion WebSocket.

        Crea una instancia del exchange y lanza una tarea asincrona por cada
        simbolo tanto para el ticker como para el libro de ordenes.
        """
        if self._running:
            logger.warning("BinanceWebSocketFeed is already running")
            return

        self._running = True
        self._exchange = self._create_exchange()

        logger.info(
            "Starting BinanceWebSocketFeed for symbols: %s",
            ", ".join(self._symbols),
        )

        # Lanzar un bucle de ticker y uno de libro de ordenes por cada simbolo
        for symbol in self._symbols:
            self._tasks.append(
                asyncio.create_task(
                    self._watch_ticker_loop(symbol),
                    name=f"ws-ticker-{symbol}",
                )
            )
            self._tasks.append(
                asyncio.create_task(
                    self._watch_order_book_loop(symbol),
                    name=f"ws-orderbook-{symbol}",
                )
            )

    async def stop(self) -> None:
        """Detiene de forma controlada todas las suscripciones y cierra la conexion al exchange.

        Cancela todas las tareas de suscripcion activas, espera su finalizacion
        y cierra la conexion CCXT con el exchange.
        """
        if not self._running:
            return

        logger.info("Stopping BinanceWebSocketFeed")
        self._running = False

        # Cancelar todas las tareas de suscripcion
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Cerrar la conexion CCXT con el exchange
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception:
                logger.exception("Error closing exchange connection")
            self._exchange = None

        logger.info("BinanceWebSocketFeed stopped")

    # ── Fabrica del Exchange ────────────────────────────────────────────

    def _create_exchange(self) -> ccxtpro.binance:
        """Crea e instancia un cliente CCXT Pro de Binance.

        Configura las claves API, el modo sandbox y las opciones del exchange
        segun la configuracion proporcionada.

        Retorna
        -------
        ccxtpro.binance
            Instancia configurada del cliente Binance de CCXT Pro.
        """
        config: dict[str, Any] = {
            "apiKey": self._exchange_config.api_key or None,
            "secret": self._exchange_config.api_secret or None,
            "enableRateLimit": self._exchange_config.rate_limit,
            "options": {
                **self._exchange_config.options,
                "defaultType": "spot",
            },
        }
        if self._exchange_config.sandbox:
            config["sandbox"] = True

        exchange = ccxtpro.binance(config)
        logger.debug("Created CCXT Pro Binance exchange (sandbox=%s)", self._exchange_config.sandbox)
        return exchange

    # ── Bucles de Suscripcion ──────────────────────────────────────────

    async def _watch_ticker_loop(self, symbol: str) -> None:
        """Observa actualizaciones del ticker para un simbolo con reconexion automatica.

        Mantiene un bucle continuo que recibe datos del ticker via WebSocket.
        Ante un error, espera un tiempo con retroceso exponencial antes de reintentar.

        Parametros
        ----------
        symbol:
            Simbolo del par de trading (ej: 'BTC/USDT').
        """
        delay = _INITIAL_RECONNECT_DELAY

        while self._running:
            try:
                ticker = await self._exchange.watch_ticker(symbol)
                delay = _INITIAL_RECONNECT_DELAY  # Reiniciar retraso tras exito

                # Construir evento de tick con los datos recibidos del WebSocket
                event = TickEvent(
                    symbol=symbol,
                    bid=_to_decimal(ticker.get("bid")),  # Mejor precio de compra
                    ask=_to_decimal(ticker.get("ask")),  # Mejor precio de venta
                    last=_to_decimal(ticker.get("last")),  # Ultimo precio operado
                    volume_24h=_to_decimal(ticker.get("baseVolume")),  # Volumen 24h en moneda base
                )
                await self._event_bus.publish(event)  # Publicar evento en el bus

            except asyncio.CancelledError:
                logger.debug("Ticker loop cancelled for %s", symbol)
                return
            except Exception as exc:
                if not self._running:
                    return
                logger.error(
                    "Ticker WS error for %s: %s — reconnecting in %.1fs",
                    symbol,
                    exc,
                    delay,
                )
                # Esperar antes de reintentar con retroceso exponencial
                await self._safe_sleep(delay)
                delay = min(delay * _RECONNECT_BACKOFF_FACTOR, _MAX_RECONNECT_DELAY)

    async def _watch_order_book_loop(self, symbol: str) -> None:
        """Observa actualizaciones del libro de ordenes para un simbolo con reconexion automatica.

        Mantiene un bucle continuo que recibe snapshots del libro de ordenes
        via WebSocket. Calcula el spread y publica OrderBookEvent.

        Parametros
        ----------
        symbol:
            Simbolo del par de trading (ej: 'BTC/USDT').
        """
        delay = _INITIAL_RECONNECT_DELAY

        while self._running:
            try:
                ob = await self._exchange.watch_order_book(symbol, self._order_book_depth)
                delay = _INITIAL_RECONNECT_DELAY  # Reiniciar retraso tras exito

                # Convertir ofertas de compra (bids) a tuplas de Decimal (precio, cantidad)
                bids = tuple(
                    (Decimal(str(price)), Decimal(str(qty)))
                    for price, qty in (ob.get("bids") or [])[:self._order_book_depth]
                )
                # Convertir ofertas de venta (asks) a tuplas de Decimal (precio, cantidad)
                asks = tuple(
                    (Decimal(str(price)), Decimal(str(qty)))
                    for price, qty in (ob.get("asks") or [])[:self._order_book_depth]
                )

                # Calcular el spread (diferencia entre mejor ask y mejor bid)
                spread = Decimal(0)
                if bids and asks:
                    spread = asks[0][0] - bids[0][0]

                # Publicar evento del libro de ordenes en el bus
                event = OrderBookEvent(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    spread=spread,
                )
                await self._event_bus.publish(event)

            except asyncio.CancelledError:
                logger.debug("Order book loop cancelled for %s", symbol)
                return
            except Exception as exc:
                if not self._running:
                    return
                logger.error(
                    "Order book WS error for %s: %s — reconnecting in %.1fs",
                    symbol,
                    exc,
                    delay,
                )
                # Esperar antes de reintentar con retroceso exponencial
                await self._safe_sleep(delay)
                delay = min(delay * _RECONNECT_BACKOFF_FACTOR, _MAX_RECONNECT_DELAY)

    # ── Utilidades ────────────────────────────────────────────────────

    async def _safe_sleep(self, seconds: float) -> None:
        """Pausa que respeta la bandera de ejecucion y maneja cancelaciones.

        Parametros
        ----------
        seconds:
            Duracion de la pausa en segundos.
        """
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            pass

    @property
    def is_running(self) -> bool:
        """Indica si el feed esta actualmente en ejecucion."""
        return self._running

    @property
    def symbols(self) -> list[str]:
        """Retorna una copia de la lista de simbolos suscritos."""
        return list(self._symbols)
