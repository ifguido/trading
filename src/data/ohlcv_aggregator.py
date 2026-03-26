"""Agrega TickEvents en CandleEvents para multiples temporalidades.

Se suscribe a TickEvent en el EventBus y mantiene el estado parcial de las
velas en construccion. Cuando un periodo de vela se cierra, publica un
CandleEvent finalizado e inicia una nueva vela. Todos los valores OHLCV
(apertura, maximo, minimo, cierre, volumen) utilizan Decimal para precision.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from src.core.event_bus import EventBus
from src.core.events import CandleEvent, Event, TickEvent

logger = logging.getLogger(__name__)

# Mapeo de cadena de temporalidad -> duracion en segundos
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,       # 1 minuto
    "5m": 300,      # 5 minutos
    "15m": 900,     # 15 minutos
    "1h": 3600,     # 1 hora
    "4h": 14400,    # 4 horas
    "1d": 86400,    # 1 dia
}

# Tupla de temporalidades soportadas para validacion
SUPPORTED_TIMEFRAMES = tuple(TIMEFRAME_SECONDS.keys())


@dataclass
class _PartialCandle:
    """Acumulador mutable de vela en construccion.

    Mantiene el estado OHLCV parcial mientras se reciben ticks dentro
    de un periodo de tiempo. Se reinicia al comenzar un nuevo periodo.
    """

    symbol: str  # Simbolo del par de trading
    timeframe: str  # Temporalidad de la vela (ej: '1m', '5m')
    open: Decimal = Decimal(0)  # Precio de apertura del periodo
    high: Decimal = Decimal(0)  # Precio maximo del periodo
    low: Decimal = Decimal(0)  # Precio minimo del periodo
    close: Decimal = Decimal(0)  # Ultimo precio del periodo (cierre)
    volume: Decimal = Decimal(0)  # Volumen acumulado (snapshot 24h)
    period_start_ms: int = 0  # Inicio del periodo en milisegundos UTC
    tick_count: int = 0  # Cantidad de ticks procesados en este periodo


def _period_start(timestamp_ms: int, period_seconds: int) -> int:
    """Calcula el inicio del periodo actual como milisegundos UTC epoch.

    Alinea el timestamp hacia abajo al limite de periodo mas cercano.
    Esto asegura que todos los ticks dentro del mismo periodo se agrupen
    bajo el mismo timestamp de inicio.

    Parametros
    ----------
    timestamp_ms:
        Timestamp en milisegundos UTC.
    period_seconds:
        Duracion del periodo en segundos.

    Retorna
    -------
    int
        Timestamp del inicio del periodo en milisegundos UTC.
    """
    period_ms = period_seconds * 1000  # Convertir duracion del periodo a milisegundos
    return (timestamp_ms // period_ms) * period_ms  # Division entera para alinear al limite


class OHLCVAggregator:
    """Agrega ticks en tiempo real en velas OHLCV para las temporalidades configuradas.

    Escucha eventos TickEvent del EventBus, mantiene velas parciales en memoria
    y publica CandleEvent cuando un periodo se cierra o cuando se detiene el
    agregador (publicando las velas parciales como no cerradas).

    Parametros
    ----------
    event_bus:
        EventBus central para suscribirse y publicar eventos.
    symbols:
        Simbolos de pares de trading para los que agregar velas.
    timeframes:
        Temporalidades de velas a producir (ej: ``["1m", "5m", "1h"]``).
        Por defecto ``["1m", "5m"]`` si no se especifica.
    """

    def __init__(
        self,
        event_bus: EventBus,
        symbols: Sequence[str],
        timeframes: Sequence[str] | None = None,
    ) -> None:
        self._event_bus = event_bus  # Bus de eventos central
        self._symbols = list(symbols)  # Simbolos de pares a agregar
        self._timeframes = list(timeframes or ["1m", "5m"])  # Temporalidades a producir
        self._running = False  # Bandera de estado de ejecucion

        # Validar que todas las temporalidades sean soportadas
        for tf in self._timeframes:
            if tf not in TIMEFRAME_SECONDS:
                raise ValueError(
                    f"Unsupported timeframe '{tf}'. "
                    f"Supported: {', '.join(SUPPORTED_TIMEFRAMES)}"
                )

        # Estado de velas parciales: (simbolo, temporalidad) -> _PartialCandle
        self._candles: dict[tuple[str, str], _PartialCandle] = {}

    # ── Ciclo de Vida ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Suscribe al TickEvent en el EventBus para comenzar a agregar velas."""
        if self._running:
            logger.warning("OHLCVAggregator is already running")
            return

        self._running = True
        self._event_bus.subscribe(
            TickEvent,
            self._on_tick,
            name="OHLCVAggregator",
        )
        logger.info(
            "OHLCVAggregator started — symbols=%s, timeframes=%s",
            self._symbols,
            self._timeframes,
        )

    async def stop(self) -> None:
        """Cancela la suscripcion a eventos y publica las velas parciales abiertas.

        Las velas que aun no han completado su periodo se publican con
        closed=False para indicar que son datos incompletos.
        """
        if not self._running:
            return

        self._running = False
        self._event_bus.unsubscribe(TickEvent, self._on_tick)

        # Publicar velas parcialmente llenas como no cerradas
        for key, candle in self._candles.items():
            if candle.tick_count > 0:
                await self._publish_candle(candle, closed=False)

        self._candles.clear()
        logger.info("OHLCVAggregator stopped")

    # ── Manejador de Eventos ──────────────────────────────────────────

    async def _on_tick(self, event: Event) -> None:
        """Procesa un TickEvent entrante y actualiza todas las velas relevantes.

        Filtra eventos que no son TickEvent o que no pertenecen a los simbolos
        configurados. Para cada temporalidad, actualiza la vela parcial
        correspondiente.

        Parametros
        ----------
        event:
            Evento recibido del EventBus; solo se procesan TickEvent.
        """
        if not isinstance(event, TickEvent):
            return

        tick: TickEvent = event
        if tick.symbol not in self._symbols:
            return  # Ignorar simbolos no configurados

        price = tick.last  # Usar el ultimo precio como referencia OHLCV
        if price <= 0:
            return  # Ignorar precios invalidos (cero o negativos)

        # Actualizar la vela parcial para cada temporalidad configurada
        for tf in self._timeframes:
            await self._update_candle(tick.symbol, tf, price, tick.volume_24h, tick.timestamp)

    # ── Logica de Velas ────────────────────────────────────────────────

    async def _update_candle(
        self,
        symbol: str,
        timeframe: str,
        price: Decimal,
        volume: Decimal,
        timestamp_ms: int,
    ) -> None:
        """Actualiza la vela parcial para (simbolo, temporalidad), publicando si el periodo cambio.

        Si el tick pertenece a un nuevo periodo, la vela anterior se cierra y
        publica, y se inicia una nueva. Si pertenece al mismo periodo, se
        actualizan los valores de maximo, minimo, cierre y volumen.

        Parametros
        ----------
        symbol:
            Simbolo del par de trading.
        timeframe:
            Temporalidad de la vela.
        price:
            Ultimo precio del tick.
        volume:
            Volumen 24h del tick (snapshot).
        timestamp_ms:
            Timestamp del tick en milisegundos UTC.
        """
        key = (symbol, timeframe)  # Clave compuesta para buscar la vela parcial
        period_secs = TIMEFRAME_SECONDS[timeframe]  # Duracion del periodo en segundos
        current_period = _period_start(timestamp_ms, period_secs)  # Inicio del periodo actual

        candle = self._candles.get(key)  # Buscar vela parcial existente

        if candle is None:
            # Primer tick para esta combinacion (simbolo, temporalidad)
            self._candles[key] = _PartialCandle(
                symbol=symbol,
                timeframe=timeframe,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                period_start_ms=current_period,
                tick_count=1,
            )
            return

        if current_period != candle.period_start_ms:
            # El periodo cambio — cerrar la vela anterior y publicarla
            await self._publish_candle(candle, closed=True)

            # Iniciar una nueva vela con el tick actual
            self._candles[key] = _PartialCandle(
                symbol=symbol,
                timeframe=timeframe,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                period_start_ms=current_period,
                tick_count=1,
            )
            return

        # Mismo periodo — actualizar valores OHLC de la vela parcial
        candle.high = max(candle.high, price)  # Actualizar maximo si el precio es mayor
        candle.low = min(candle.low, price)  # Actualizar minimo si el precio es menor
        candle.close = price  # El cierre siempre es el ultimo precio recibido
        candle.volume = volume  # Snapshot mas reciente del volumen 24h
        candle.tick_count += 1  # Incrementar contador de ticks procesados

    async def _publish_candle(self, candle: _PartialCandle, *, closed: bool) -> None:
        """Publica un CandleEvent a partir del estado de la vela parcial.

        Parametros
        ----------
        candle:
            Vela parcial con los datos OHLCV acumulados.
        closed:
            True si el periodo de la vela se cerro completamente,
            False si es una publicacion parcial (ej: al detener el agregador).
        """
        event = CandleEvent(
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            closed=closed,
            timestamp=candle.period_start_ms,
        )
        logger.debug(
            "Candle %s [%s] O=%s H=%s L=%s C=%s V=%s closed=%s (%d ticks)",
            candle.symbol,
            candle.timeframe,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            closed,
            candle.tick_count,
        )
        await self._event_bus.publish(event)

    # ── Introspeccion ─────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """Indica si el agregador esta actualmente en ejecucion."""
        return self._running

    @property
    def symbols(self) -> list[str]:
        """Retorna una copia de la lista de simbolos configurados."""
        return list(self._symbols)

    @property
    def timeframes(self) -> list[str]:
        """Retorna una copia de la lista de temporalidades configuradas."""
        return list(self._timeframes)

    def get_partial_candle(self, symbol: str, timeframe: str) -> _PartialCandle | None:
        """Retorna la vela parcial en curso (si existe) para inspeccion.

        Parametros
        ----------
        symbol:
            Simbolo del par de trading.
        timeframe:
            Temporalidad de la vela.

        Retorna
        -------
        _PartialCandle | None
            La vela parcial actual, o None si no hay datos para esa combinacion.
        """
        return self._candles.get((symbol, timeframe))
