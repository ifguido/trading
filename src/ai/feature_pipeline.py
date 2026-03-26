"""Pipeline de features: convierte datos crudos de mercado en diccionarios de features para modelos AI.

Se suscribe a ``TickEvent`` y ``CandleEvent`` en el ``EventBus``,
computa indicadores técnicos por símbolo y los expone para consulta
bajo demanda por la capa de estrategia.

Indicadores calculados:
- Cambio de precio (retorno de 1 vela)
- Ratio de volumen (actual vs promedio histórico)
- RSI (Relative Strength Index)
- MACD (línea, señal, histograma)
- Posición en Bollinger Bands (normalizada)
"""

from __future__ import annotations

import logging
from collections import deque
from decimal import Decimal
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import CandleEvent, TickEvent

logger = logging.getLogger(__name__)

# ── Constantes por defecto para indicadores técnicos ──────────────

_DEFAULT_RSI_PERIOD = 14          # Período del RSI (estándar de Wilder)
_DEFAULT_MACD_FAST = 12           # EMA rápida del MACD
_DEFAULT_MACD_SLOW = 26           # EMA lenta del MACD
_DEFAULT_MACD_SIGNAL = 9          # Línea de señal del MACD
_DEFAULT_BOLLINGER_PERIOD = 20    # Período de las Bollinger Bands
_DEFAULT_MAX_CANDLES = 200        # Máximo de velas almacenadas por símbolo


class FeaturePipeline:
    """Computa y cachea features por símbolo a partir de eventos de mercado en vivo.

    Se suscribe automáticamente al EventBus para recibir ticks y velas.
    Cada vez que llega una vela cerrada, recalcula todos los indicadores
    técnicos para ese símbolo.

    Uso::

        pipeline = FeaturePipeline(event_bus)
        features = pipeline.get_features("BTC/USDT")
    """

    def __init__(
        self,
        event_bus: EventBus,
        rsi_period: int = _DEFAULT_RSI_PERIOD,
        macd_fast: int = _DEFAULT_MACD_FAST,
        macd_slow: int = _DEFAULT_MACD_SLOW,
        macd_signal: int = _DEFAULT_MACD_SIGNAL,
        bollinger_period: int = _DEFAULT_BOLLINGER_PERIOD,
        max_candles: int = _DEFAULT_MAX_CANDLES,
    ) -> None:
        """Inicializa el pipeline de features y se suscribe a eventos.

        Parámetros
        ----------
        event_bus : EventBus
            Bus de eventos para recibir TickEvent y CandleEvent.
        rsi_period : int
            Período para el cálculo del RSI.
        macd_fast : int
            Período de la EMA rápida del MACD.
        macd_slow : int
            Período de la EMA lenta del MACD.
        macd_signal : int
            Período de la línea de señal del MACD.
        bollinger_period : int
            Período para las Bollinger Bands.
        max_candles : int
            Número máximo de velas a almacenar por símbolo.
        """
        self._event_bus = event_bus
        self._rsi_period = rsi_period
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._bollinger_period = bollinger_period
        self._max_candles = max_candles

        # Estado por símbolo — deques con tamaño máximo para limitar memoria
        self._closes: dict[str, deque[Decimal]] = {}    # Precios de cierre por símbolo
        self._volumes: dict[str, deque[Decimal]] = {}    # Volúmenes por símbolo
        self._last_tick: dict[str, TickEvent] = {}       # Último tick recibido por símbolo
        self._features: dict[str, dict[str, Any]] = {}   # Features computados por símbolo

        # Suscripción a eventos del bus
        self._event_bus.subscribe(TickEvent, self._on_tick, name="FeaturePipeline.tick")
        self._event_bus.subscribe(CandleEvent, self._on_candle, name="FeaturePipeline.candle")
        logger.info("FeaturePipeline inicializado")

    # ── API Pública ──────────────────────────────────────────────────

    def get_features(self, symbol: str) -> dict[str, Any]:
        """Devuelve los features más recientes computados para *symbol*.

        Retorna un dict vacío si aún no se han recibido datos para ese símbolo.

        Parámetros
        ----------
        symbol : str
            Símbolo del par de trading (ej: "BTC/USDT").

        Retorna
        -------
        dict[str, Any]
            Copia del diccionario de features (para evitar mutaciones externas).
        """
        return dict(self._features.get(symbol, {}))

    @property
    def symbols(self) -> list[str]:
        """Símbolos para los cuales hay features disponibles."""
        return list(self._features.keys())

    # ── Manejadores de Eventos ───────────────────────────────────────

    async def _on_tick(self, event: TickEvent) -> None:
        """Cachea el último tick y computa features a nivel de tick.

        Calcula el spread (diferencia ask-bid), spread en puntos base,
        último precio y volumen 24h.
        """
        self._last_tick[event.symbol] = event
        features = self._features.setdefault(event.symbol, {})

        # Calcular spread y spread en puntos base (bps) si hay precios válidos
        if event.ask > 0 and event.bid > 0:
            features["spread"] = float(event.ask - event.bid)
            mid = (event.ask + event.bid) / 2  # Precio medio entre ask y bid
            features["spread_bps"] = float((event.ask - event.bid) / mid * Decimal(10_000)) if mid else 0.0
        features["last_price"] = float(event.last)       # Último precio de transacción
        features["volume_24h"] = float(event.volume_24h)  # Volumen acumulado 24 horas

    async def _on_candle(self, event: CandleEvent) -> None:
        """Acumula velas cerradas y recalcula indicadores técnicos.

        Solo procesa velas cerradas (event.closed=True). Las velas
        en progreso se ignoran para evitar señales basadas en datos parciales.
        """
        # Ignorar velas que aún no se han cerrado
        if not event.closed:
            return

        symbol = event.symbol
        # Inicializar deques por símbolo si es la primera vela
        closes = self._closes.setdefault(symbol, deque(maxlen=self._max_candles))
        volumes = self._volumes.setdefault(symbol, deque(maxlen=self._max_candles))

        # Agregar datos de la nueva vela cerrada
        closes.append(event.close)
        volumes.append(event.volume)

        # Recalcular todos los indicadores con los datos actualizados
        self._recompute(symbol)

    # ── Cálculo de Indicadores ────────────────────────────────────────

    def _recompute(self, symbol: str) -> None:
        """Recalcula todos los features técnicos para *symbol*.

        Se ejecuta cada vez que llega una nueva vela cerrada. Requiere
        al menos 2 velas para calcular el cambio de precio básico.
        Los indicadores más complejos (RSI, MACD, Bollinger) necesitan
        más datos y devuelven None si no hay suficientes.
        """
        closes = self._closes.get(symbol)
        volumes = self._volumes.get(symbol)
        # Se necesitan al menos 2 velas para calcular retornos
        if not closes or len(closes) < 2:
            return

        features = self._features.setdefault(symbol, {})

        # Convertir Decimal a float para cálculos numéricos
        close_floats = [float(c) for c in closes]
        volume_floats = [float(v) for v in volumes] if volumes else []

        # Cambio de precio (retorno de 1 vela)
        features["price_change"] = (close_floats[-1] - close_floats[-2]) / close_floats[-2] if close_floats[-2] else 0.0

        # Ratio de volumen (actual vs promedio histórico)
        if len(volume_floats) >= 2:
            avg_vol = sum(volume_floats[:-1]) / len(volume_floats[:-1])  # Promedio excluyendo la vela actual
            features["volume_ratio"] = volume_floats[-1] / avg_vol if avg_vol else 0.0
        else:
            features["volume_ratio"] = 1.0  # Sin historial suficiente, ratio neutro

        # RSI — solo se incluye si hay suficientes datos
        rsi = self._compute_rsi(close_floats, self._rsi_period)
        if rsi is not None:
            features["rsi"] = rsi

        # MACD — línea, señal e histograma
        macd_line, signal_line, histogram = self._compute_macd(
            close_floats, self._macd_fast, self._macd_slow, self._macd_signal
        )
        if macd_line is not None:
            features["macd"] = macd_line
            features["macd_signal"] = signal_line
            features["macd_histogram"] = histogram

        # Posición en Bollinger Bands (normalizada)
        boll_pos = self._compute_bollinger_position(close_floats, self._bollinger_period)
        if boll_pos is not None:
            features["bollinger_position"] = boll_pos

        # Cantidad de velas (útil para que el modelo sepa el estado de precalentamiento)
        features["candle_count"] = len(close_floats)

    # ── Métodos Estáticos Auxiliares ──────────────────────────────────

    @staticmethod
    def _ema(values: list[float], period: int) -> list[float]:
        """Calcula la media móvil exponencial (EMA).

        Usa SMA (media simple) como semilla inicial, luego aplica el
        multiplicador exponencial para los valores subsiguientes.

        Parámetros
        ----------
        values : list[float]
            Serie de valores numéricos.
        period : int
            Período de la EMA.

        Retorna
        -------
        list[float]
            Valores de la EMA. Lista vacía si no hay suficientes datos.
        """
        if len(values) < period:
            return []
        multiplier = 2.0 / (period + 1)  # Factor de suavizado exponencial
        ema_values: list[float] = []
        # Semilla con SMA (media simple de los primeros 'period' valores)
        sma = sum(values[:period]) / period
        ema_values.append(sma)
        # Aplicar fórmula EMA iterativamente
        for v in values[period:]:
            ema_values.append((v - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values

    @staticmethod
    def _compute_rsi(closes: list[float], period: int) -> float | None:
        """Calcula el RSI (Relative Strength Index) a partir de precios de cierre.

        Utiliza suavizado de Wilder: el promedio de ganancias/pérdidas se
        actualiza exponencialmente con factor (period-1)/period.
        RSI > 70 indica sobrecompra, RSI < 30 indica sobreventa.

        Parámetros
        ----------
        closes : list[float]
            Precios de cierre ordenados cronológicamente.
        period : int
            Período del RSI (típicamente 14).

        Retorna
        -------
        float | None
            Valor RSI entre 0 y 100, o None si no hay suficientes datos.
        """
        if len(closes) < period + 1:
            return None

        # Calcular deltas (cambios entre velas consecutivas)
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0.0 for d in deltas]    # Solo ganancias (positivos)
        losses = [-d if d < 0 else 0.0 for d in deltas]   # Solo pérdidas (valor absoluto)

        # Promedios iniciales usando media simple
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Suavizado de Wilder para el resto de la serie
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        # Si no hay pérdidas, RSI = 100 (máximo alcista)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss  # Ratio de fuerza relativa
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_macd(
        closes: list[float],
        fast: int,
        slow: int,
        signal_period: int,
    ) -> tuple[float | None, float | None, float | None]:
        """Calcula la línea MACD, línea de señal e histograma.

        - Línea MACD = EMA(rápida) - EMA(lenta)
        - Señal = EMA de la línea MACD
        - Histograma = MACD - Señal (indica momentum)

        Parámetros
        ----------
        closes : list[float]
            Precios de cierre.
        fast : int
            Período de la EMA rápida.
        slow : int
            Período de la EMA lenta.
        signal_period : int
            Período de la línea de señal.

        Retorna
        -------
        tuple[float | None, float | None, float | None]
            (macd, señal, histograma) o (None, None, None) si datos insuficientes.
        """
        # Verificar datos mínimos necesarios
        if len(closes) < slow + signal_period:
            return None, None, None

        ema_fast = FeaturePipeline._ema(closes, fast)
        ema_slow = FeaturePipeline._ema(closes, slow)

        if not ema_fast or not ema_slow:
            return None, None, None

        # Alinear longitudes: ema_fast es más larga porque su período es menor
        offset = len(ema_fast) - len(ema_slow)
        macd_line_values = [
            ema_fast[offset + i] - ema_slow[i] for i in range(len(ema_slow))
        ]

        if len(macd_line_values) < signal_period:
            return None, None, None

        # Calcular línea de señal como EMA de la línea MACD
        signal_values = FeaturePipeline._ema(macd_line_values, signal_period)
        if not signal_values:
            return None, None, None

        macd_val = macd_line_values[-1]       # Último valor de la línea MACD
        signal_val = signal_values[-1]         # Último valor de la señal
        return macd_val, signal_val, macd_val - signal_val  # Histograma = MACD - señal

    @staticmethod
    def _compute_bollinger_position(
        closes: list[float], period: int, num_std: float = 2.0
    ) -> float | None:
        """Devuelve la posición normalizada dentro de las Bollinger Bands [-1, 1].

        -1 = en la banda inferior, 0 = en la media (SMA), +1 = en la banda superior.
        Valores fuera de [-1, 1] indican que el precio superó las bandas.

        Parámetros
        ----------
        closes : list[float]
            Precios de cierre.
        period : int
            Período de la SMA y desviación estándar.
        num_std : float
            Número de desviaciones estándar para las bandas (default 2.0).

        Retorna
        -------
        float | None
            Posición normalizada o None si datos insuficientes.
        """
        if len(closes) < period:
            return None

        # Tomar la ventana de los últimos 'period' precios
        window = closes[-period:]
        sma = sum(window) / period                                    # Media simple
        variance = sum((x - sma) ** 2 for x in window) / period      # Varianza poblacional
        std = variance**0.5                                           # Desviación estándar

        # Si no hay variación, el precio está exactamente en la media
        if std == 0:
            return 0.0

        upper = sma + num_std * std   # Banda superior
        lower = sma - num_std * std   # Banda inferior
        band_width = upper - lower    # Ancho total de las bandas

        if band_width == 0:
            return 0.0

        # Normalizar el precio actual al rango [-1, 1]
        return (closes[-1] - sma) / (num_std * std)
