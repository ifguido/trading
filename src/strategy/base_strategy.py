"""Clase base abstracta para todas las estrategias de trading.

Las estrategias reciben eventos de mercado (ticks, candles, order book)
y publican SignalEvents al EventBus cuando detectan oportunidades.

Soporta integración opcional con modelo AI: si se pasa un ai_model
y feature_pipeline, las estrategias pueden consultar predicciones AI
como un factor adicional en su evaluación.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from src.core.event_bus import EventBus
from src.core.events import (
    CandleEvent,
    OrderBookEvent,
    SignalEvent,
    TickEvent,
)

if TYPE_CHECKING:
    from src.ai.feature_pipeline import FeaturePipeline
    from src.ai.signal import AISignal

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Clase base que todas las estrategias de trading deben heredar.

    Define la interfaz comun para recibir eventos de mercado y publicar
    señales de trading. Cada estrategia concreta debe implementar los
    metodos abstractos on_tick, on_candle y on_order_book.

    Parametros
    ----------
    name:
        Identificador unico de esta instancia de estrategia.
    symbols:
        Lista de pares de trading que esta estrategia opera.
    event_bus:
        El EventBus central para publicar SignalEvents.
    params:
        Parametros de configuracion especificos de la estrategia.
    ai_model:
        Modelo AI opcional que implementa ModelInterface (Protocol).
        Si es None, la estrategia opera solo con indicadores tecnicos.
    feature_pipeline:
        Pipeline de features opcional para obtener indicadores en vivo.
        Necesario si ai_model no es None.
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
        self._name = name  # Nombre identificador de la estrategia
        self._symbols = list(symbols)  # Pares de trading a operar
        self._event_bus = event_bus  # Bus de eventos para publicar señales
        self._params = params or {}  # Parametros de configuracion
        self._initialized = False  # Bandera de estado de inicializacion

        # Modelo AI y pipeline de features (opcionales)
        self._ai_model = ai_model
        self._feature_pipeline = feature_pipeline

    # ── Propiedades ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Nombre unico que identifica esta instancia de estrategia."""
        return self._name

    @property
    def symbols(self) -> list[str]:
        """Simbolos de pares de trading a los que esta suscrita la estrategia."""
        return list(self._symbols)

    @property
    def is_initialized(self) -> bool:
        """Indica si la estrategia ha completado su fase de inicializacion."""
        return self._initialized

    # ── Ciclo de vida ─────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Realiza la configuracion necesaria antes de procesar eventos.

        Las subclases pueden sobreescribir este metodo para cargar datos
        historicos, pre-calcular indicadores, o validar parametros.
        """
        logger.info("Initializing strategy: %s (symbols=%s)", self._name, self._symbols)
        self._initialized = True

    async def shutdown(self) -> None:
        """Libera recursos cuando la estrategia es detenida.

        Las subclases pueden sobreescribir este metodo para vaciar estado,
        cerrar conexiones, etc.
        """
        logger.info("Shutting down strategy: %s", self._name)
        self._initialized = False

    # ── Manejadores de eventos ────────────────────────────────────────

    @abstractmethod
    async def on_tick(self, event: TickEvent) -> None:
        """Maneja un tick de precio en tiempo real.

        Parametros
        ----------
        event:
            El evento de tick entrante con precios bid/ask/last.
        """

    @abstractmethod
    async def on_candle(self, event: CandleEvent) -> None:
        """Maneja una vela OHLCV completada.

        Parametros
        ----------
        event:
            El evento de vela entrante con datos OHLCV.
        """

    @abstractmethod
    async def on_order_book(self, event: OrderBookEvent) -> None:
        """Maneja una actualizacion del libro de ordenes.

        Parametros
        ----------
        event:
            El evento de libro de ordenes entrante con bids/asks.
        """

    # ── Publicacion de señales ────────────────────────────────────────

    async def _publish_signal(self, signal: SignalEvent) -> None:
        """Publica una señal de trading al EventBus.

        Registra la direccion, simbolo y confianza de la señal,
        y luego la envia al bus de eventos para que otros componentes
        (como el gestor de riesgo u ordenes) la procesen.

        Parametros
        ----------
        signal:
            El SignalEvent a publicar.
        """
        logger.info(
            "Strategy %s signal: %s %s (confidence=%.2f)",
            self._name,
            signal.direction.value,
            signal.symbol,
            signal.confidence,
        )
        await self._event_bus.publish(signal)

    # ── Ayudante AI ──────────────────────────────────────────────────

    async def _get_ai_signal(self, symbol: str) -> AISignal | None:
        """Consulta al modelo AI para obtener una prediccion.

        Obtiene features del FeaturePipeline para el simbolo dado
        y los pasa al modelo AI para generar una señal.

        Retorna None si:
        - No hay modelo AI configurado
        - No hay feature pipeline
        - No hay features suficientes para el simbolo
        - Ocurre un error durante la prediccion

        Parametros
        ----------
        symbol:
            Par de trading para el cual obtener la prediccion AI.

        Retorna
        -------
        AISignal | None
            La señal AI, o None si no se puede generar.
        """
        # Si no hay modelo AI o pipeline, no se puede generar señal
        if self._ai_model is None or self._feature_pipeline is None:
            return None

        try:
            # Obtener features actuales del pipeline para el simbolo
            features = self._feature_pipeline.get_features(symbol)
            if not features:
                return None

            # Ejecutar prediccion del modelo AI con las features
            ai_signal = await self._ai_model.predict(features)
            return ai_signal

        except Exception:
            logger.warning(
                "Error obteniendo señal AI para %s en strategy '%s'",
                symbol,
                self._name,
                exc_info=True,
            )
            return None
