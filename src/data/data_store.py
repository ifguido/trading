"""Almacen de datos en memoria con buffer circular para datos de mercado recientes.

Almacena ticks, velas (candles) y snapshots del libro de ordenes por simbolo
en buffers de tamano fijo usando ``collections.deque``. Las lecturas son
seguras entre hilos (thread-safe) gracias a ``threading.Lock``, permitiendo
que consumidores sincronos (ej: paneles de monitoreo) accedan a los datos
de forma segura mientras el bucle de eventos asincrono escribe en ellos.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Sequence

from src.core.event_bus import EventBus
from src.core.events import (
    CandleEvent,
    Event,
    OrderBookEvent,
    TickEvent,
)

logger = logging.getLogger(__name__)

# Tamanos por defecto de los buffers circulares
_DEFAULT_TICK_BUFFER = 1000  # Maximo de ticks por simbolo
_DEFAULT_CANDLE_BUFFER = 500  # Maximo de velas por (simbolo, temporalidad)
_DEFAULT_ORDERBOOK_BUFFER = 200  # Maximo de snapshots del libro de ordenes por simbolo


class DataStore:
    """Almacen en memoria con buffer circular para eventos de datos de mercado recientes.

    Se suscribe a ``TickEvent``, ``CandleEvent`` y ``OrderBookEvent``
    en el EventBus y retiene las N entradas mas recientes por simbolo
    en objetos ``collections.deque`` de tamano limitado.
    Proporciona una API de lectura segura entre hilos para consumidores sincronos.

    Parametros
    ----------
    event_bus:
        EventBus central al que suscribirse para recibir eventos.
    symbols:
        Simbolos de pares de trading a rastrear.
    tick_buffer_size:
        Numero maximo de ticks a retener por simbolo.
    candle_buffer_size:
        Numero maximo de velas a retener por (simbolo, temporalidad).
    orderbook_buffer_size:
        Numero maximo de snapshots del libro de ordenes a retener por simbolo.
    """

    def __init__(
        self,
        event_bus: EventBus,
        symbols: Sequence[str],
        tick_buffer_size: int = _DEFAULT_TICK_BUFFER,
        candle_buffer_size: int = _DEFAULT_CANDLE_BUFFER,
        orderbook_buffer_size: int = _DEFAULT_ORDERBOOK_BUFFER,
    ) -> None:
        self._event_bus = event_bus  # Bus de eventos central
        self._symbols = list(symbols)  # Simbolos de pares a rastrear
        self._tick_buffer_size = tick_buffer_size  # Tamano maximo del buffer de ticks
        self._candle_buffer_size = candle_buffer_size  # Tamano maximo del buffer de velas
        self._orderbook_buffer_size = orderbook_buffer_size  # Tamano maximo del buffer del libro
        self._running = False  # Bandera de estado de ejecucion

        # El lock protege las lecturas de deque contra escrituras asincronas concurrentes.
        # asyncio es de un solo hilo, asi que las escrituras no compiten entre si,
        # pero los lectores sincronos en otros hilos necesitan proteccion.
        self._lock = threading.Lock()

        # simbolo -> deque[TickEvent] — Buffer circular de ticks por simbolo
        self._ticks: dict[str, deque[TickEvent]] = {
            s: deque(maxlen=tick_buffer_size) for s in symbols
        }

        # (simbolo, temporalidad) -> deque[CandleEvent] — Buffer de velas por par clave
        self._candles: dict[tuple[str, str], deque[CandleEvent]] = {}

        # simbolo -> deque[OrderBookEvent] — Buffer de snapshots del libro de ordenes
        self._orderbooks: dict[str, deque[OrderBookEvent]] = {
            s: deque(maxlen=orderbook_buffer_size) for s in symbols
        }

        # simbolo -> ultimo TickEvent (acceso rapido de conveniencia)
        self._latest_tick: dict[str, TickEvent] = {}

        # simbolo -> ultimo OrderBookEvent (acceso rapido de conveniencia)
        self._latest_orderbook: dict[str, OrderBookEvent] = {}

    # ── Ciclo de Vida ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Suscribe a los eventos de datos de mercado en el EventBus.

        Registra manejadores para TickEvent, CandleEvent y OrderBookEvent.
        """
        if self._running:
            logger.warning("DataStore is already running")
            return

        self._running = True
        self._event_bus.subscribe(TickEvent, self._on_tick, name="DataStore.tick")
        self._event_bus.subscribe(CandleEvent, self._on_candle, name="DataStore.candle")
        self._event_bus.subscribe(OrderBookEvent, self._on_orderbook, name="DataStore.orderbook")
        logger.info("DataStore started for symbols: %s", ", ".join(self._symbols))

    async def stop(self) -> None:
        """Cancela las suscripciones a eventos del EventBus."""
        if not self._running:
            return

        self._running = False
        self._event_bus.unsubscribe(TickEvent, self._on_tick)
        self._event_bus.unsubscribe(CandleEvent, self._on_candle)
        self._event_bus.unsubscribe(OrderBookEvent, self._on_orderbook)
        logger.info("DataStore stopped")

    # ── Manejadores de Eventos ──────────────────────────────────────────

    async def _on_tick(self, event: Event) -> None:
        """Maneja un evento de tick entrante y lo almacena en el buffer.

        Parametros
        ----------
        event:
            Evento recibido; solo se procesan instancias de TickEvent.
        """
        if not isinstance(event, TickEvent):
            return
        symbol = event.symbol
        with self._lock:  # Proteger escritura con lock para seguridad entre hilos
            if symbol in self._ticks:
                self._ticks[symbol].append(event)  # Agregar al buffer existente
            else:
                # Crear nuevo buffer para simbolo no registrado previamente
                buf: deque[TickEvent] = deque(maxlen=self._tick_buffer_size)
                buf.append(event)
                self._ticks[symbol] = buf
            self._latest_tick[symbol] = event  # Actualizar referencia rapida al ultimo tick

    async def _on_candle(self, event: Event) -> None:
        """Maneja un evento de vela entrante y lo almacena en el buffer.

        Parametros
        ----------
        event:
            Evento recibido; solo se procesan instancias de CandleEvent.
        """
        if not isinstance(event, CandleEvent):
            return
        key = (event.symbol, event.timeframe)  # Clave compuesta: (simbolo, temporalidad)
        with self._lock:  # Proteger escritura con lock
            if key not in self._candles:
                # Crear nuevo buffer de velas para esta combinacion
                self._candles[key] = deque(maxlen=self._candle_buffer_size)
            self._candles[key].append(event)

    async def _on_orderbook(self, event: Event) -> None:
        """Maneja un evento de libro de ordenes entrante y lo almacena en el buffer.

        Parametros
        ----------
        event:
            Evento recibido; solo se procesan instancias de OrderBookEvent.
        """
        if not isinstance(event, OrderBookEvent):
            return
        symbol = event.symbol
        with self._lock:  # Proteger escritura con lock
            if symbol in self._orderbooks:
                self._orderbooks[symbol].append(event)  # Agregar al buffer existente
            else:
                # Crear nuevo buffer para simbolo no registrado previamente
                buf: deque[OrderBookEvent] = deque(maxlen=self._orderbook_buffer_size)
                buf.append(event)
                self._orderbooks[symbol] = buf
            self._latest_orderbook[symbol] = event  # Actualizar referencia rapida

    # ── API de Lectura Thread-Safe ──────────────────────────────────────

    def get_recent_ticks(self, symbol: str, n: int | None = None) -> list[TickEvent]:
        """Retorna hasta *n* ticks mas recientes para el *simbolo* (mas nuevo al final).

        Si *n* es ``None``, retorna todos los ticks en el buffer.

        Parametros
        ----------
        symbol:
            Simbolo del par de trading.
        n:
            Cantidad maxima de ticks a retornar. None para todos.

        Retorna
        -------
        list[TickEvent]
            Lista de ticks ordenados del mas antiguo al mas reciente.
        """
        with self._lock:
            buf = self._ticks.get(symbol)
            if buf is None:
                return []
            items = list(buf)
        if n is not None:
            items = items[-n:]
        return items

    def get_latest_tick(self, symbol: str) -> TickEvent | None:
        """Retorna el tick mas reciente para el *simbolo*, o ``None`` si no hay datos.

        Parametros
        ----------
        symbol:
            Simbolo del par de trading.

        Retorna
        -------
        TickEvent | None
            El ultimo tick recibido, o None si no se ha recibido ninguno.
        """
        with self._lock:
            return self._latest_tick.get(symbol)

    def get_recent_candles(
        self,
        symbol: str,
        timeframe: str,
        n: int | None = None,
    ) -> list[CandleEvent]:
        """Retorna hasta *n* velas mas recientes para *(simbolo, temporalidad)* (mas nueva al final).

        Parametros
        ----------
        symbol:
            Simbolo del par de trading.
        timeframe:
            Temporalidad de las velas (ej: '1m', '5m', '1h').
        n:
            Cantidad maxima de velas a retornar. None para todas.

        Retorna
        -------
        list[CandleEvent]
            Lista de velas ordenadas de la mas antigua a la mas reciente.
        """
        with self._lock:
            buf = self._candles.get((symbol, timeframe))
            if buf is None:
                return []
            items = list(buf)
        if n is not None:
            items = items[-n:]
        return items

    def get_recent_orderbooks(self, symbol: str, n: int | None = None) -> list[OrderBookEvent]:
        """Retorna hasta *n* snapshots mas recientes del libro de ordenes (mas nuevo al final).

        Parametros
        ----------
        symbol:
            Simbolo del par de trading.
        n:
            Cantidad maxima de snapshots a retornar. None para todos.

        Retorna
        -------
        list[OrderBookEvent]
            Lista de snapshots del libro de ordenes ordenados cronologicamente.
        """
        with self._lock:
            buf = self._orderbooks.get(symbol)
            if buf is None:
                return []
            items = list(buf)
        if n is not None:
            items = items[-n:]
        return items

    def get_latest_orderbook(self, symbol: str) -> OrderBookEvent | None:
        """Retorna el snapshot mas reciente del libro de ordenes, o ``None`` si no hay datos.

        Parametros
        ----------
        symbol:
            Simbolo del par de trading.

        Retorna
        -------
        OrderBookEvent | None
            El ultimo snapshot del libro de ordenes, o None.
        """
        with self._lock:
            return self._latest_orderbook.get(symbol)

    # ── Utilidades ─────────────────────────────────────────────────────

    def clear(self, symbol: str | None = None) -> None:
        """Limpia los datos almacenados. Si se proporciona *symbol*, limpia solo ese simbolo.

        Parametros
        ----------
        symbol:
            Simbolo especifico a limpiar. Si es None, limpia todos los datos.
        """
        with self._lock:
            if symbol is None:
                for buf in self._ticks.values():
                    buf.clear()
                for buf in self._candles.values():
                    buf.clear()
                for buf in self._orderbooks.values():
                    buf.clear()
                self._latest_tick.clear()
                self._latest_orderbook.clear()
            else:
                if symbol in self._ticks:
                    self._ticks[symbol].clear()
                if symbol in self._orderbooks:
                    self._orderbooks[symbol].clear()
                # Limpiar buffers de velas para todas las temporalidades de este simbolo
                for key in list(self._candles.keys()):
                    if key[0] == symbol:
                        self._candles[key].clear()
                self._latest_tick.pop(symbol, None)
                self._latest_orderbook.pop(symbol, None)

    def buffer_stats(self) -> dict[str, dict[str, int]]:
        """Retorna los tamanos actuales de los buffers para monitoreo.

        Util para diagnosticar el uso de memoria y verificar que los datos
        estan fluyendo correctamente.

        Retorna
        -------
        dict[str, dict[str, int]]
            Diccionario con las categorias 'ticks', 'candles' y 'orderbooks',
            cada una mapeando identificadores a conteos. Ejemplo::

                {
                    "ticks": {"BTC/USDT": 342, "ETH/USDT": 128},
                    "candles": {"BTC/USDT:1m": 60, "BTC/USDT:5m": 12},
                    "orderbooks": {"BTC/USDT": 200},
                }
        """
        with self._lock:
            return {
                "ticks": {s: len(buf) for s, buf in self._ticks.items()},
                "candles": {
                    f"{sym}:{tf}": len(buf)
                    for (sym, tf), buf in self._candles.items()
                },
                "orderbooks": {s: len(buf) for s, buf in self._orderbooks.items()},
            }

    @property
    def is_running(self) -> bool:
        """Indica si el almacen de datos esta actualmente en ejecucion."""
        return self._running

    @property
    def symbols(self) -> list[str]:
        """Retorna una copia de la lista de simbolos rastreados."""
        return list(self._symbols)
