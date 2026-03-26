"""Gestor de estrategias: crea, configura y enruta eventos a las estrategias.

Se suscribe a TickEvent, CandleEvent y OrderBookEvent en el EventBus
y los despacha a las instancias de estrategia correspondientes segun
el simbolo y la configuracion del tipo de estrategia.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.config_loader import AppConfig, PairConfig
from src.core.event_bus import EventBus
from src.core.events import CandleEvent, OrderBookEvent, TickEvent
from src.core.exceptions import ConfigError, StrategyError
from src.strategy.base_strategy import BaseStrategy
from src.strategy.scalping.scalp_strategy import ScalpStrategy
from src.strategy.swing.swing_strategy import SwingStrategy

logger = logging.getLogger(__name__)

# Registro que mapea nombres de tipo de estrategia a sus clases
_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "swing": SwingStrategy,
    "scalp": ScalpStrategy,
    "scalping": ScalpStrategy,  # Alias alternativo para scalp
}


class StrategyManager:
    """Gestiona el ciclo de vida de estrategias de trading y enruta eventos de mercado.

    Responsabilidades:
    - Crea instancias de estrategia a partir de la configuracion de la aplicacion.
    - Se suscribe a eventos de datos de mercado en el EventBus.
    - Enruta TickEvent, CandleEvent y OrderBookEvent a las instancias de
      estrategia suscritas al simbolo correspondiente.
    - Maneja la inicializacion y apagado de estrategias.

    Parametros
    ----------
    event_bus:
        El EventBus central para recibir eventos y pasarlos a las estrategias.
    config:
        La configuracion raiz de la aplicacion.
    strategy_params:
        Sobrecargas opcionales de parametros por tipo de estrategia.
        Las claves son nombres de tipo de estrategia (ej. "swing", "scalp"),
        los valores son diccionarios de parametros pasados al constructor.
    ai_model:
        Modelo AI opcional que se pasa a cada estrategia creada.
    feature_pipeline:
        Pipeline de features opcional que se pasa a cada estrategia creada.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: AppConfig,
        strategy_params: dict[str, dict[str, Any]] | None = None,
        ai_model: Any | None = None,
        feature_pipeline: Any | None = None,
    ) -> None:
        self._event_bus = event_bus  # Bus de eventos central
        self._config = config  # Configuracion de la aplicacion
        self._strategy_params = strategy_params or {}  # Parametros por tipo de estrategia

        # Modelo AI y pipeline de features — se pasan a cada estrategia
        self._ai_model = ai_model
        self._feature_pipeline = feature_pipeline

        # Todas las instancias de estrategia activas
        self._strategies: list[BaseStrategy] = []

        # Busqueda rapida: simbolo -> lista de estrategias que lo manejan
        self._symbol_strategies: dict[str, list[BaseStrategy]] = {}

        self._running = False  # Bandera de estado de ejecucion

    # ── Propiedades ───────────────────────────────────────────────────

    @property
    def strategies(self) -> list[BaseStrategy]:
        """Todas las instancias de estrategia registradas."""
        return list(self._strategies)

    @property
    def is_running(self) -> bool:
        """Indica si el gestor de estrategias esta en ejecucion."""
        return self._running

    # ── Ciclo de vida ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Crea estrategias desde la config, las inicializa y se suscribe a eventos.

        Este metodo ejecuta la secuencia completa de arranque:
        1. Crear instancias de estrategia segun la configuracion de pares
        2. Inicializar cada estrategia (carga de datos, validacion, etc.)
        3. Suscribirse a eventos de mercado en el EventBus
        """
        if self._running:
            logger.warning("StrategyManager is already running")
            return

        self._create_strategies()  # Paso 1: Crear instancias
        await self._initialize_strategies()  # Paso 2: Inicializar
        self._subscribe_events()  # Paso 3: Suscribirse a eventos
        self._running = True

        logger.info(
            "StrategyManager started with %d strategy instance(s)",
            len(self._strategies),
        )

    async def stop(self) -> None:
        """Apaga todas las estrategias y cancela suscripciones a eventos.

        Ejecuta el apagado ordenado: primero cancela suscripciones,
        luego apaga cada estrategia individualmente capturando errores,
        y finalmente limpia las estructuras internas.
        """
        if not self._running:
            return

        logger.info("Stopping StrategyManager")
        # Cancelar suscripciones a eventos antes de apagar estrategias
        self._unsubscribe_events()

        # Apagar cada estrategia capturando errores individuales
        for strategy in self._strategies:
            try:
                await strategy.shutdown()
            except Exception:
                logger.exception("Error shutting down strategy '%s'", strategy.name)

        # Limpiar todas las estructuras internas
        self._strategies.clear()
        self._symbol_strategies.clear()
        self._running = False
        logger.info("StrategyManager stopped")

    # ── Creacion de estrategias ───────────────────────────────────────

    def _create_strategies(self) -> None:
        """Instancia objetos de estrategia desde la configuracion de pares.

        Agrupa pares por tipo de estrategia y luego crea una instancia
        de estrategia por tipo con todos sus simbolos asignados.
        Tambien construye el indice de busqueda rapida simbolo -> estrategias.

        Lanza
        ------
        ConfigError:
            Si se encuentra un tipo de estrategia desconocido en la configuracion.
        """
        # Agrupar simbolos por tipo de estrategia
        type_to_pairs: dict[str, list[PairConfig]] = {}
        for pair in self._config.pairs:
            strategy_type = pair.strategy.lower()
            type_to_pairs.setdefault(strategy_type, []).append(pair)

        for strategy_type, pairs in type_to_pairs.items():
            # Buscar la clase de estrategia en el registro
            cls = _STRATEGY_REGISTRY.get(strategy_type)
            if cls is None:
                raise ConfigError(
                    f"Unknown strategy type '{strategy_type}'. "
                    f"Available: {', '.join(_STRATEGY_REGISTRY.keys())}"
                )

            # Extraer simbolos y parametros para esta estrategia
            symbols = [p.symbol for p in pairs]
            params = self._strategy_params.get(strategy_type, {})
            # Generar nombre unico combinando tipo y simbolos
            name = f"{strategy_type}_{'-'.join(s.replace('/', '_') for s in symbols)}"

            # Crear instancia de la estrategia con todos los parametros
            strategy = cls(
                name=name,
                symbols=symbols,
                event_bus=self._event_bus,
                params=params,
                ai_model=self._ai_model,
                feature_pipeline=self._feature_pipeline,
            )
            self._strategies.append(strategy)

            # Construir indice de busqueda rapida simbolo -> estrategia
            for symbol in symbols:
                self._symbol_strategies.setdefault(symbol, []).append(strategy)

            logger.info(
                "Created strategy '%s' (%s) for symbols: %s",
                name,
                cls.__name__,
                ", ".join(symbols),
            )

    async def _initialize_strategies(self) -> None:
        """Inicializa todas las estrategias creadas.

        Llama al metodo initialize() de cada estrategia. Si alguna
        falla, lanza un StrategyError con el detalle del error.

        Lanza
        ------
        StrategyError:
            Si la inicializacion de alguna estrategia falla.
        """
        for strategy in self._strategies:
            try:
                await strategy.initialize()
            except Exception as exc:
                raise StrategyError(
                    f"Failed to initialize strategy '{strategy.name}': {exc}"
                ) from exc

    # ── Suscripcion a eventos ─────────────────────────────────────────

    def _subscribe_events(self) -> None:
        """Se suscribe a eventos de datos de mercado en el EventBus.

        Registra callbacks para TickEvent, CandleEvent y OrderBookEvent
        que enrutaran cada evento a las estrategias correspondientes.
        """
        self._event_bus.subscribe(
            TickEvent,
            self._on_tick,
            name="StrategyManager._on_tick",
        )
        self._event_bus.subscribe(
            CandleEvent,
            self._on_candle,
            name="StrategyManager._on_candle",
        )
        self._event_bus.subscribe(
            OrderBookEvent,
            self._on_order_book,
            name="StrategyManager._on_order_book",
        )
        logger.debug("StrategyManager subscribed to TickEvent, CandleEvent, OrderBookEvent")

    def _unsubscribe_events(self) -> None:
        """Cancela la suscripcion a todos los eventos de datos de mercado.

        Remueve los callbacks registrados en el EventBus para dejar de
        recibir eventos de mercado.
        """
        self._event_bus.unsubscribe(TickEvent, self._on_tick)
        self._event_bus.unsubscribe(CandleEvent, self._on_candle)
        self._event_bus.unsubscribe(OrderBookEvent, self._on_order_book)
        logger.debug("StrategyManager unsubscribed from market events")

    # ── Enrutamiento de eventos ───────────────────────────────────────

    async def _on_tick(self, event: TickEvent) -> None:
        """Enruta un TickEvent a todas las estrategias suscritas a su simbolo.

        Itera sobre las estrategias registradas para el simbolo del evento
        y llama a on_tick en cada una, capturando errores individuales.

        Parametros
        ----------
        event:
            Evento de tick con datos de precio en tiempo real.
        """
        # Obtener estrategias registradas para este simbolo
        strategies = self._symbol_strategies.get(event.symbol, [])
        for strategy in strategies:
            try:
                await strategy.on_tick(event)
            except Exception:
                logger.exception(
                    "Strategy '%s' failed on TickEvent for %s",
                    strategy.name,
                    event.symbol,
                )

    async def _on_candle(self, event: CandleEvent) -> None:
        """Enruta un CandleEvent a todas las estrategias suscritas a su simbolo.

        Itera sobre las estrategias registradas para el simbolo del evento
        y llama a on_candle en cada una, capturando errores individuales.

        Parametros
        ----------
        event:
            Evento de vela con datos OHLCV.
        """
        # Obtener estrategias registradas para este simbolo
        strategies = self._symbol_strategies.get(event.symbol, [])
        for strategy in strategies:
            try:
                await strategy.on_candle(event)
            except Exception:
                logger.exception(
                    "Strategy '%s' failed on CandleEvent for %s",
                    strategy.name,
                    event.symbol,
                )

    async def _on_order_book(self, event: OrderBookEvent) -> None:
        """Enruta un OrderBookEvent a todas las estrategias suscritas a su simbolo.

        Itera sobre las estrategias registradas para el simbolo del evento
        y llama a on_order_book en cada una, capturando errores individuales.

        Parametros
        ----------
        event:
            Evento de libro de ordenes con listas de bids y asks.
        """
        # Obtener estrategias registradas para este simbolo
        strategies = self._symbol_strategies.get(event.symbol, [])
        for strategy in strategies:
            try:
                await strategy.on_order_book(event)
            except Exception:
                logger.exception(
                    "Strategy '%s' failed on OrderBookEvent for %s",
                    strategy.name,
                    event.symbol,
                )

    # ── Utilidades ────────────────────────────────────────────────────

    def get_strategy(self, name: str) -> BaseStrategy | None:
        """Busca una estrategia por su nombre unico.

        Parametros
        ----------
        name:
            Nombre identificador de la estrategia a buscar.

        Retorna
        -------
        BaseStrategy | None
            La instancia de estrategia si se encuentra, o None si no existe.
        """
        for strategy in self._strategies:
            if strategy.name == name:
                return strategy
        return None

    def get_strategies_for_symbol(self, symbol: str) -> list[BaseStrategy]:
        """Retorna todas las estrategias que manejan un simbolo dado.

        Parametros
        ----------
        symbol:
            Par de trading para el cual buscar estrategias.

        Retorna
        -------
        list[BaseStrategy]
            Lista de estrategias registradas para ese simbolo.
        """
        return list(self._symbol_strategies.get(symbol, []))

    def register_strategy_class(self, type_name: str, cls: type[BaseStrategy]) -> None:
        """Registra una clase de estrategia personalizada en tiempo de ejecucion.

        Permite que plugins o modulos externos agreguen nuevos tipos de
        estrategia sin modificar el registro incorporado.

        Parametros
        ----------
        type_name:
            El nombre a usar en la configuracion (ej. "mi_estrategia_custom").
        cls:
            La clase de estrategia (debe ser subclase de BaseStrategy).

        Lanza
        ------
        TypeError:
            Si la clase proporcionada no es subclase de BaseStrategy.
        """
        if not issubclass(cls, BaseStrategy):
            raise TypeError(f"{cls.__name__} is not a subclass of BaseStrategy")
        _STRATEGY_REGISTRY[type_name.lower()] = cls
        logger.info("Registered custom strategy class: %s -> %s", type_name, cls.__name__)
