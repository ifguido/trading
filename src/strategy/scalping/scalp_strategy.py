"""Estrategia de scalping basada en desbalance del libro de ordenes y momentum de ticks.

Se enfoca en señales de corta duracion derivadas del analisis de microestructura
del mercado: desbalance bid/ask, porcentaje del spread, y momentum de ticks
recientes. Diseñada para ejecucion rapida y rotacion veloz de posiciones.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from decimal import Decimal
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import (
    CandleEvent,
    OrderBookEvent,
    SignalDirection,
    SignalEvent,
    TickEvent,
)
from src.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

# ── Parametros por defecto ────────────────────────────────────────────

_DEFAULT_TICK_WINDOW = 50  # Maximo de ticks recientes a conservar por simbolo
_DEFAULT_TICK_EXPIRY_MS = 5_000  # Tiempo de expiracion de ticks en milisegundos
_DEFAULT_IMBALANCE_THRESHOLD = Decimal("1.5")  # Umbral de ratio bid/ask para desbalance
_DEFAULT_SPREAD_MAX_PCT = Decimal("0.0015")  # Spread maximo aceptable (0.15%)
_DEFAULT_MOMENTUM_THRESHOLD = Decimal("0.0005")  # Cambio minimo de precio para momentum (0.05%)
_DEFAULT_STOP_LOSS_PCT = Decimal("0.005")  # Porcentaje de stop loss (0.5%)
_DEFAULT_TAKE_PROFIT_PCT = Decimal("0.008")  # Porcentaje de toma de ganancias (0.8%)
_DEFAULT_MIN_CONFIDENCE = 0.5  # Confianza minima para emitir señal
_DEFAULT_COOLDOWN_MS = 3_000  # Tiempo minimo entre señales por simbolo (ms)
_DEFAULT_OB_DEPTH_LEVELS = 5  # Niveles del libro de ordenes a analizar


class ScalpStrategy(BaseStrategy):
    """Estrategia de scalping usando desbalance del libro de ordenes y momentum de ticks.

    Analiza la microestructura del mercado en tiempo real para generar señales
    LONG/SHORT de corta duracion. La estrategia puntua tres componentes:
    1. Desbalance de volumen bid/ask en el libro de ordenes
    2. Porcentaje del spread relativo a un umbral maximo
    3. Momentum de precio basado en ticks recientes (direccion y magnitud)

    Parametros
    ----------
    name:
        Identificador unico para esta instancia de estrategia.
    symbols:
        Lista de simbolos de pares de trading que opera esta estrategia.
    event_bus:
        El EventBus central para publicar SignalEvents.
    params:
        Parametros especificos de la estrategia. Claves soportadas:
        - tick_window (int): Max ticks recientes por simbolo. Default 50.
        - tick_expiry_ms (int): Eliminar ticks mas viejos que esto (ms). Default 5000.
        - imbalance_threshold (str/Decimal): Umbral ratio bid/ask. Default "1.5".
        - spread_max_pct (str/Decimal): Spread max pct para entrada. Default "0.0015".
        - momentum_threshold (str/Decimal): Cambio min de precio para momentum. Default "0.0005".
        - stop_loss_pct (str/Decimal): Porcentaje de stop loss. Default "0.005".
        - take_profit_pct (str/Decimal): Porcentaje de toma de ganancias. Default "0.008".
        - min_confidence (float): Confianza minima para emitir señal. Default 0.5.
        - cooldown_ms (int): Minimo ms entre señales por simbolo. Default 3000.
        - ob_depth_levels (int): Niveles del libro de ordenes a analizar. Default 5.
    """

    def __init__(
        self,
        name: str,
        symbols: list[str],
        event_bus: EventBus,
        params: dict[str, Any] | None = None,
        ai_model: Any | None = None,
        feature_pipeline: Any | None = None,
    ) -> None:
        super().__init__(name, symbols, event_bus, params, ai_model, feature_pipeline)

        # Parametros de configuracion extraidos del diccionario de params
        self._tick_window: int = self._params.get("tick_window", _DEFAULT_TICK_WINDOW)
        self._tick_expiry_ms: int = self._params.get("tick_expiry_ms", _DEFAULT_TICK_EXPIRY_MS)
        self._imbalance_threshold: Decimal = Decimal(
            str(self._params.get("imbalance_threshold", _DEFAULT_IMBALANCE_THRESHOLD))
        )
        self._spread_max_pct: Decimal = Decimal(
            str(self._params.get("spread_max_pct", _DEFAULT_SPREAD_MAX_PCT))
        )
        self._momentum_threshold: Decimal = Decimal(
            str(self._params.get("momentum_threshold", _DEFAULT_MOMENTUM_THRESHOLD))
        )
        self._stop_loss_pct: Decimal = Decimal(
            str(self._params.get("stop_loss_pct", _DEFAULT_STOP_LOSS_PCT))
        )
        self._take_profit_pct: Decimal = Decimal(
            str(self._params.get("take_profit_pct", _DEFAULT_TAKE_PROFIT_PCT))
        )
        self._min_confidence: float = self._params.get("min_confidence", _DEFAULT_MIN_CONFIDENCE)
        self._cooldown_ms: int = self._params.get("cooldown_ms", _DEFAULT_COOLDOWN_MS)
        self._ob_depth_levels: int = self._params.get("ob_depth_levels", _DEFAULT_OB_DEPTH_LEVELS)

        # Estado por simbolo
        # Ticks recientes: deque de tuplas (timestamp_ms, precio_decimal)
        self._recent_ticks: dict[str, deque[tuple[int, Decimal]]] = {}
        # Ultimo snapshot del libro de ordenes por simbolo
        self._latest_ob: dict[str, OrderBookEvent] = {}
        # Momento de la ultima emision de señal por simbolo (epoch ms)
        self._last_signal_time: dict[str, int] = {}

    # ── Ciclo de vida ─────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Configura los contenedores de estado por simbolo.

        Inicializa la cola de ticks recientes con tamaño maximo
        y resetea el tiempo de la ultima señal para cada simbolo.
        """
        await super().initialize()
        for symbol in self._symbols:
            # Cola de ticks con tamaño maximo limitado
            self._recent_ticks[symbol] = deque(maxlen=self._tick_window)
            # Sin señales previas emitidas
            self._last_signal_time[symbol] = 0
        logger.info(
            "ScalpStrategy '%s' initialized — imbalance_threshold=%s, "
            "spread_max_pct=%s, momentum_threshold=%s, cooldown=%dms",
            self._name,
            self._imbalance_threshold,
            self._spread_max_pct,
            self._momentum_threshold,
            self._cooldown_ms,
        )

    # ── Manejadores de eventos ────────────────────────────────────────

    async def on_tick(self, event: TickEvent) -> None:
        """Registra datos de tick y evalua oportunidad de scalping.

        Cada tick se almacena en la ventana de ticks recientes. Si tambien
        hay un snapshot del libro de ordenes disponible, se ejecuta una
        evaluacion combinada de los tres componentes.

        Parametros
        ----------
        event:
            Evento de tick con precio bid/ask/last.
        """
        # Ignorar simbolos que no pertenecen a esta estrategia
        if event.symbol not in self._symbols:
            return

        symbol = event.symbol
        # Agregar tick a la cola de recientes (timestamp, precio)
        self._recent_ticks[symbol].append((event.timestamp, event.last))
        # Eliminar ticks vencidos de la ventana
        self._prune_old_ticks(symbol)

        # Solo evaluar si tenemos un snapshot reciente del libro de ordenes
        if symbol in self._latest_ob:
            await self._evaluate(symbol)

    async def on_order_book(self, event: OrderBookEvent) -> None:
        """Almacena el ultimo snapshot del libro de ordenes y evalua.

        Los datos del libro de ordenes se usan para calcular el desbalance
        bid/ask y el spread en el siguiente ciclo de evaluacion.

        Parametros
        ----------
        event:
            Evento de libro de ordenes con listas de bids y asks.
        """
        # Ignorar simbolos que no pertenecen a esta estrategia
        if event.symbol not in self._symbols:
            return

        # Guardar el snapshot mas reciente del libro de ordenes
        self._latest_ob[event.symbol] = event

        # Evaluar inmediatamente si tambien tenemos datos de ticks
        if self._recent_ticks.get(event.symbol):
            await self._evaluate(event.symbol)

    async def on_candle(self, event: CandleEvent) -> None:
        """La estrategia de scalping no utiliza datos de velas."""
        pass

    # ── Interno: Gestion de ventana de ticks ──────────────────────────

    def _prune_old_ticks(self, symbol: str) -> None:
        """Elimina ticks mas antiguos que la ventana de expiracion.

        Recorre la cola desde el inicio y remueve todos los ticks
        cuyo timestamp sea anterior al punto de corte.

        Parametros
        ----------
        symbol:
            Simbolo del par de trading cuyos ticks se limpian.
        """
        now_ms = int(time.time() * 1000)  # Tiempo actual en milisegundos
        cutoff = now_ms - self._tick_expiry_ms  # Punto de corte temporal
        ticks = self._recent_ticks[symbol]
        # Remover ticks vencidos desde el inicio de la cola
        while ticks and ticks[0][0] < cutoff:
            ticks.popleft()

    # ── Interno: Evaluacion de señales ────────────────────────────────

    async def _evaluate(self, symbol: str) -> None:
        """Puntua desbalance del libro, spread y momentum para decidir señal.

        Combina tres componentes de analisis de microestructura:
        1. Desbalance del libro de ordenes (quien domina, compradores o vendedores)
        2. Estrechez del spread (spreads mas apretados = mejor para scalping)
        3. Momentum de precio (tendencia reciente de los ticks)

        Solo emite señal si todos los filtros se cumplen y la confianza
        supera el umbral minimo.

        Parametros
        ----------
        symbol:
            Par de trading a evaluar.
        """
        now_ms = int(time.time() * 1000)

        # Respetar el periodo de enfriamiento entre señales
        if now_ms - self._last_signal_time.get(symbol, 0) < self._cooldown_ms:
            return

        ob = self._latest_ob.get(symbol)  # Ultimo libro de ordenes
        ticks = self._recent_ticks.get(symbol)  # Ticks recientes
        if ob is None or not ticks:
            return

        # ── 1) Desbalance del libro de ordenes ────────────────────────
        imbalance_score, imbalance_direction = self._compute_imbalance(ob)

        # ── 2) Verificacion del spread ────────────────────────────────
        spread_ok, spread_score = self._compute_spread(ob)
        if not spread_ok:
            return  # Spread demasiado amplio para scalping

        # ── 3) Momentum de ticks ──────────────────────────────────────
        momentum_score, momentum_direction = self._compute_momentum(ticks)

        # ── Agregar puntuaciones ──────────────────────────────────────
        # imbalance_direction y momentum_direction: +1 alcista, -1 bajista, 0 neutral
        direction_votes = imbalance_direction + momentum_direction

        if direction_votes == 0:
            return  # Señales conflictivas o neutrales

        # Confianza calculada del promedio de los tres componentes (cada uno en [0, 1])
        confidence = (imbalance_score + spread_score + momentum_score) / 3.0

        # Verificar que la confianza supere el umbral minimo
        if confidence < self._min_confidence:
            return

        # Determinar direccion basada en los votos
        direction = SignalDirection.LONG if direction_votes > 0 else SignalDirection.SHORT

        # Calcular stop-loss y take-profit desde el ultimo precio
        latest_price = ticks[-1][1]  # Ultimo precio (Decimal)
        if direction == SignalDirection.LONG:
            # Posicion larga: SL por debajo, TP por encima
            stop_loss = latest_price * (Decimal(1) - self._stop_loss_pct)
            take_profit = latest_price * (Decimal(1) + self._take_profit_pct)
        else:
            # Posicion corta: SL por encima, TP por debajo
            stop_loss = latest_price * (Decimal(1) + self._stop_loss_pct)
            take_profit = latest_price * (Decimal(1) - self._take_profit_pct)

        # Construir evento de señal con metadata de los componentes
        signal = SignalEvent(
            symbol=symbol,
            direction=direction,
            strategy_name=self._name,
            confidence=round(confidence, 4),
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "imbalance_score": round(imbalance_score, 4),
                "spread_score": round(spread_score, 4),
                "momentum_score": round(momentum_score, 4),
                "direction_votes": direction_votes,
            },
        )

        # Registrar tiempo de la señal para el periodo de enfriamiento
        self._last_signal_time[symbol] = now_ms
        await self._publish_signal(signal)

    # ── Interno: Funciones de puntuacion por componente ───────────────

    def _compute_imbalance(
        self, ob: OrderBookEvent
    ) -> tuple[float, int]:
        """Calcula el desbalance de volumen bid/ask del libro de ordenes.

        Compara el volumen total de bids contra asks en los primeros
        N niveles del libro. Si los bids dominan significativamente,
        indica presion compradora (alcista). Si los asks dominan,
        indica presion vendedora (bajista).

        Parametros
        ----------
        ob:
            Evento de libro de ordenes con listas de bids y asks.

        Retorna
        -------
        score:
            Fuerza del desbalance normalizada en [0, 1].
        direction:
            +1 si bids dominan (alcista), -1 si asks dominan (bajista), 0 neutral.
        """
        # Tomar solo los primeros N niveles de profundidad
        bids = ob.bids[: self._ob_depth_levels]
        asks = ob.asks[: self._ob_depth_levels]

        if not bids or not asks:
            return 0.0, 0

        # Sumar volumen total de bids y asks
        bid_volume = sum(qty for _, qty in bids)
        ask_volume = sum(qty for _, qty in asks)

        if ask_volume == 0 or bid_volume == 0:
            return 0.0, 0

        # Ratio de volumen bid/ask (division Decimal)
        ratio = bid_volume / ask_volume

        if ratio >= self._imbalance_threshold:
            # Bids dominan — señal alcista
            # Normalizar: ratio de 1.5 -> 0.5, ratio de 3.0 -> 1.0 (con tope)
            raw = float((ratio - Decimal(1)) / (self._imbalance_threshold - Decimal(1)))
            score = min(raw / 2.0, 1.0)
            return score, 1
        elif Decimal(1) / ratio >= self._imbalance_threshold:
            # Asks dominan — señal bajista
            inv_ratio = Decimal(1) / ratio  # Ratio invertido para medir dominio de asks
            raw = float((inv_ratio - Decimal(1)) / (self._imbalance_threshold - Decimal(1)))
            score = min(raw / 2.0, 1.0)
            return score, -1
        else:
            return 0.0, 0  # No hay desbalance significativo

    def _compute_spread(self, ob: OrderBookEvent) -> tuple[bool, float]:
        """Verifica si el spread es aceptable y puntua su estrechez.

        Un spread mas estrecho es mejor para scalping ya que reduce el
        costo de entrada/salida. Si el spread supera el maximo permitido,
        la operacion se rechaza.

        Parametros
        ----------
        ob:
            Evento de libro de ordenes con mejor bid y ask.

        Retorna
        -------
        ok:
            True si el spread esta dentro del umbral maximo.
        score:
            Puntuacion de estrechez en [0, 1] — spread mas estrecho = mayor puntuacion.
        """
        if not ob.bids or not ob.asks:
            return False, 0.0

        best_bid = ob.bids[0][0]  # Mejor precio de compra
        best_ask = ob.asks[0][0]  # Mejor precio de venta

        if best_bid == 0:
            return False, 0.0

        # Calcular precio medio entre bid y ask
        mid = (best_bid + best_ask) / Decimal(2)
        if mid == 0:
            return False, 0.0

        # Calcular spread como porcentaje del precio medio
        spread_pct = (best_ask - best_bid) / mid

        # Rechazar si el spread excede el maximo permitido
        if spread_pct > self._spread_max_pct:
            return False, 0.0

        # Spread mas estrecho = puntuacion mas alta
        # Con spread 0 -> score 1.0; con spread maximo -> score 0.0
        score = float(Decimal(1) - spread_pct / self._spread_max_pct)
        return True, score

    def _compute_momentum(
        self, ticks: deque[tuple[int, Decimal]]
    ) -> tuple[float, int]:
        """Calcula el momentum de precio a partir de ticks recientes.

        Compara el precio mas antiguo con el mas reciente en la ventana
        de ticks para determinar la direccion y magnitud del movimiento.

        Parametros
        ----------
        ticks:
            Cola de ticks recientes como tuplas (timestamp_ms, precio).

        Retorna
        -------
        score:
            Fuerza del momentum en [0, 1].
        direction:
            +1 alcista, -1 bajista, 0 neutral.
        """
        # Se necesitan al menos 3 ticks para calcular momentum confiable
        if len(ticks) < 3:
            return 0.0, 0

        oldest_price = ticks[0][1]  # Precio mas antiguo en la ventana
        latest_price = ticks[-1][1]  # Precio mas reciente

        if oldest_price == 0:
            return 0.0, 0

        # Calcular cambio porcentual del precio
        pct_change = (latest_price - oldest_price) / oldest_price

        # Si el cambio no supera el umbral minimo, no hay momentum significativo
        if abs(pct_change) < self._momentum_threshold:
            return 0.0, 0

        # Normalizar: umbral de momentum -> 0.3, 3x umbral -> 1.0 (con tope)
        raw = float(abs(pct_change) / self._momentum_threshold)
        score = min(raw / 3.0, 1.0)

        # Direccion basada en el signo del cambio de precio
        direction = 1 if pct_change > 0 else -1
        return score, direction
