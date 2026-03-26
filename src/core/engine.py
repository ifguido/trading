"""Orquestador principal: conecta todos los componentes y gestiona el ciclo de vida asincrono.

Este modulo contiene el Engine, que es el corazon del sistema CryptoTrader.
Su responsabilidad es:
- Inicializar todos los componentes en el orden correcto de dependencias.
- Gestionar las senales del sistema operativo (SIGINT, SIGTERM) para apagado graceful.
- Coordinar el inicio y la detencion ordenada de todos los subsistemas.

El orden de inicializacion es critico:
1. Logging (para que todos los componentes puedan loggear).
2. Storage (base de datos, necesaria para persistencia).
3. Exchange (conexion al exchange, necesaria para datos y ejecucion).
4. Data (feeds WebSocket, agregador OHLCV, data store).
5. Risk (portfolio tracker, position sizer, circuit breaker, risk manager).
6. Execution (order executor, order manager, fill handler).
7. AI (feature pipeline y modelo de inteligencia artificial).
8. Strategies (strategy manager que conecta datos con senales).
9. Monitoring (Telegram notifier, health check).

El apagado se realiza en orden inverso para garantizar consistencia.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import signal
import sys
from typing import Any

from .config_loader import AppConfig, load_config
from .event_bus import EventBus
from .exceptions import ConfigError

logger = logging.getLogger(__name__)


class Engine:
    """Motor central que inicializa, ejecuta y apaga todos los componentes del sistema.

    El Engine actua como orquestador: no contiene logica de trading,
    sino que conecta los componentes entre si a traves del EventBus
    y gestiona su ciclo de vida.

    Atributos:
        config: Configuracion completa de la aplicacion (AppConfig).
        event_bus: Bus central de eventos para comunicacion entre componentes.
        _running: Indica si el engine esta activo.
        _shutdown_event: Evento asyncio que se activa al recibir senal de apagado.
        _exchange ... _health_check: Referencias a todos los componentes del sistema.
    """

    def __init__(self, config: AppConfig) -> None:
        """Inicializa el engine con la configuracion proporcionada.

        Solo almacena la configuracion y crea el bus de eventos.
        Los componentes reales se inicializan en start() para permitir
        operaciones asincronas durante la inicializacion.

        Parametros:
            config: Configuracion completa validada por Pydantic.
        """
        self.config = config
        self.event_bus = EventBus()
        self._running = False                       # Estado de ejecucion
        self._shutdown_event = asyncio.Event()       # Senal de apagado graceful

        # Referencias a componentes (se inicializan en start)
        # Se declaran como Any porque los tipos reales se importan
        # de forma lazy en cada metodo _init_* para evitar imports circulares.
        self._exchange: Any = None            # Conexion al exchange (ccxt)
        self._ws_feed: Any = None             # Feed WebSocket de datos en tiempo real
        self._ohlcv_aggregator: Any = None    # Agregador de velas OHLCV desde ticks
        self._data_store: Any = None          # Almacen de datos de mercado en memoria
        self._portfolio_tracker: Any = None   # Rastreador del estado del portafolio
        self._position_sizer: Any = None      # Calculador de tamano de posiciones
        self._circuit_breaker: Any = None     # Interruptor de emergencia del trading
        self._risk_manager: Any = None        # Gestor de riesgo (aprueba/rechaza ordenes)
        self._order_executor: Any = None      # Ejecutor de ordenes (real o paper)
        self._order_manager: Any = None       # Gestor de ordenes abiertas
        self._fill_handler: Any = None        # Procesador de ejecuciones confirmadas
        self._strategy_manager: Any = None    # Gestor de estrategias de trading
        self._ai_model: Any = None            # Modelo de inteligencia artificial
        self._feature_pipeline: Any = None    # Pipeline de features para el modelo AI
        self._telegram_notifier: Any = None   # Notificador de alertas por Telegram
        self._health_check: Any = None        # Monitor de salud del sistema
        self._repository: Any = None          # Repositorio de persistencia (DB)

    async def start(self) -> None:
        """Inicializa y arranca todos los componentes en orden de dependencias.

        Este es el metodo principal que pone en marcha el sistema completo.
        Registra handlers de senales del SO para apagado graceful,
        inicializa cada subsistema en orden, y luego espera indefinidamente
        hasta recibir una senal de apagado (SIGINT/SIGTERM).

        Si ocurre un error fatal durante la inicializacion, se loggea
        y se propaga la excepcion tras detener los componentes ya iniciados.
        """
        logger.info("Starting CryptoTrader engine...")
        self._running = True

        # Registrar handlers de senales del SO para apagado graceful
        # Esto permite detener el sistema con Ctrl+C o kill
        # Se omite cuando el Engine corre dentro del proceso web (FastAPI maneja las senales)
        if not getattr(self, '_skip_signal_handlers', False):
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self._signal_shutdown()))

        try:
            # Inicializar cada subsistema en orden de dependencias
            # El orden es crucial: cada paso puede depender de los anteriores
            await self._init_logging()      # 1. Logging primero para poder registrar todo
            await self._init_storage()      # 2. Base de datos para persistencia
            await self._init_exchange()     # 3. Conexion al exchange
            await self._init_data()         # 4. Feeds de datos y agregadores
            await self._init_risk()         # 5. Gestion de riesgo
            await self._init_execution()    # 6. Ejecucion de ordenes
            await self._init_ai()           # 7. Modelo AI y features
            await self._init_strategies()   # 8. Estrategias de trading
            await self._init_monitoring()   # 9. Monitoreo y notificaciones

            logger.info("All components initialized. Starting live feeds...")

            # Iniciar feeds de datos en tiempo real
            if self._ws_feed:
                await self._ws_feed.start()

            # Iniciar monitor de salud del sistema
            if self._health_check:
                await self._health_check.start()

            logger.info("CryptoTrader engine running. Press Ctrl+C to stop.")

            # Esperar indefinidamente hasta recibir senal de apagado
            # Este await se desbloquea cuando _signal_shutdown() activa el evento
            await self._shutdown_event.wait()

        except Exception:
            logger.exception("Fatal error during engine startup")
            raise
        finally:
            # Siempre ejecutar el apagado, incluso si hubo error
            await self.stop()

    async def stop(self) -> None:
        """Apagado graceful: detiene feeds, cancela ordenes, cierra DB y conexiones.

        Ejecuta la secuencia de apagado en orden inverso al de inicializacion
        para garantizar que no se pierdan datos ni queden ordenes huerfanas.
        Cada paso esta envuelto en try/except para que un error en un
        componente no impida el apagado de los demas.
        """
        # Evitar apagar mas de una vez
        if not self._running:
            return
        self._running = False
        logger.info("Shutting down CryptoTrader engine...")

        # Paso 1: Detener feeds de datos (dejar de recibir datos nuevos)
        if self._ws_feed:
            try:
                await self._ws_feed.stop()
                logger.info("WebSocket feed stopped")
            except Exception:
                logger.exception("Error stopping WebSocket feed")

        if self._ohlcv_aggregator:
            try:
                await self._ohlcv_aggregator.stop()
            except Exception:
                logger.exception("Error stopping OHLCV aggregator")

        # Paso 2: Detener estrategias (dejar de generar senales)
        if self._strategy_manager:
            try:
                await self._strategy_manager.stop()
                logger.info("Strategies stopped")
            except Exception:
                logger.exception("Error stopping strategies")

        # Paso 3: Cancelar ordenes abiertas (critico para no dejar ordenes huerfanas)
        if self._order_manager:
            try:
                cancelled = await self._order_manager.shutdown_cancel_open_orders()
                if cancelled:
                    logger.info("Cancelled %d open orders", len(cancelled))
            except Exception:
                logger.exception("Error cancelling open orders")

        # Paso 4: Detener ejecucion de ordenes
        if self._order_executor:
            try:
                await self._order_executor.stop()
            except Exception:
                logger.exception("Error stopping order executor")

        # Paso 5: Detener monitoreo
        if self._health_check:
            try:
                await self._health_check.stop()
            except Exception:
                logger.exception("Error stopping health check")

        if self._telegram_notifier:
            try:
                await self._telegram_notifier.stop()
            except Exception:
                logger.exception("Error stopping Telegram notifier")

        # Paso 6: Cerrar base de datos (flush de datos pendientes)
        try:
            from src.storage.db import close_db
            await close_db()
            logger.info("Database closed")
        except Exception:
            logger.exception("Error closing database")

        # Paso 7: Cerrar conexion con el exchange
        if self._exchange:
            try:
                await self._exchange.close()
                logger.info("Exchange connection closed")
            except Exception:
                logger.exception("Error closing exchange")

        # Paso 8: Apagar el bus de eventos (cancelar tareas pendientes)
        await self.event_bus.shutdown()
        logger.info("CryptoTrader engine stopped")

    async def _signal_shutdown(self) -> None:
        """Handler de senales del SO (SIGINT, SIGTERM).

        Al recibir una senal de terminacion, activa el evento de apagado
        que desbloquea el await en start(), iniciando la secuencia de
        apagado graceful.
        """
        logger.info("Shutdown signal received")
        self._shutdown_event.set()

    # ── Inicializacion de Componentes ──────────────────────────────

    async def _init_logging(self) -> None:
        """Configura el sistema de logging segun la configuracion.

        Establece el nivel de log y el formato. Tambien silencia loggers
        ruidosos de librerias externas (ccxt, asyncio) para mantener
        los logs del sistema limpios y legibles.
        """
        level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"

        if self.config.logging.json_format:
            logging.basicConfig(level=level, format=fmt, stream=sys.stdout)
        else:
            logging.basicConfig(level=level, format=fmt, stream=sys.stdout)

        # Silenciar loggers ruidosos de librerias externas
        logging.getLogger("ccxt").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)

    async def _init_storage(self) -> None:
        """Inicializa la base de datos y el repositorio de persistencia.

        Crea las tablas si no existen, configura la sesion asincrona
        y crea el repositorio que otros componentes usaran para
        guardar y consultar datos.
        """
        from src.storage.db import init_db, async_session_factory
        from src.storage.repository import Repository

        await init_db(self.config.storage.url)
        session_factory = async_session_factory()
        self._repository = Repository(session_factory)
        logger.info("Storage initialized")

    async def _init_exchange(self) -> None:
        """Inicializa la conexion con el exchange de criptomonedas.

        En modo "paper" no se conecta al exchange real, lo que permite
        hacer trading simulado sin riesgo. En modo "live" o "sandbox",
        crea una instancia de ccxt con las credenciales configuradas
        y valida la conectividad cargando los mercados disponibles.

        Lanza:
            ConfigError: Si el nombre del exchange no es valido en ccxt.
        """
        # En modo paper no necesitamos conexion real al exchange
        if self.config.exchange.mode == "paper":
            logger.info("Exchange in PAPER mode — skipping real exchange connection")
            self._exchange = None
            return

        import ccxt.async_support as ccxt

        # Buscar la clase del exchange en ccxt por nombre
        exchange_cls = getattr(ccxt, self.config.exchange.name, None)
        if exchange_cls is None:
            raise ConfigError(f"Unknown exchange: {self.config.exchange.name}")

        # Crear instancia del exchange con credenciales y opciones
        self._exchange = exchange_cls({
            "apiKey": self.config.exchange.api_key,
            "secret": self.config.exchange.api_secret,
            "enableRateLimit": self.config.exchange.rate_limit,
            "options": self.config.exchange.options,
        })

        # Activar modo sandbox si esta configurado
        if self.config.exchange.sandbox:
            self._exchange.set_sandbox_mode(True)
            logger.info("Exchange in SANDBOX mode")

        # Validar conectividad cargando la lista de mercados
        await self._exchange.load_markets()
        logger.info("Exchange connected: %s (%d markets)", self.config.exchange.name, len(self._exchange.markets))

    async def _init_data(self) -> None:
        """Inicializa los componentes de datos de mercado.

        Crea y conecta:
        - BinanceWebSocketFeed: recibe ticks en tiempo real via WebSocket.
        - OHLCVAggregator: convierte ticks en velas OHLCV para las estrategias.
        - DataStore: almacen en memoria con los datos mas recientes.

        Los timeframes se extraen de la configuracion de cada par de trading.
        """
        from src.data.binance_ws_feed import BinanceWebSocketFeed
        from src.data.ohlcv_aggregator import OHLCVAggregator
        from src.data.data_store import DataStore

        # Crear feed WebSocket con los pares configurados
        self._ws_feed = BinanceWebSocketFeed(
            event_bus=self.event_bus,
            exchange_config=self.config.exchange,
            pairs=self.config.pairs,
        )

        # Recopilar todos los timeframes unicos y simbolos de los pares configurados
        all_timeframes: set[str] = set()  # Set para evitar duplicados
        symbols: list[str] = []
        for pair in self.config.pairs:
            symbols.append(pair.symbol)
            all_timeframes.update(pair.timeframes)

        # Crear agregador OHLCV que convierte ticks en velas
        self._ohlcv_aggregator = OHLCVAggregator(
            event_bus=self.event_bus,
            symbols=symbols,
            timeframes=list(all_timeframes),
        )
        await self._ohlcv_aggregator.start()

        # Crear almacen de datos en memoria para acceso rapido
        self._data_store = DataStore(event_bus=self.event_bus, symbols=symbols)
        await self._data_store.start()

        # Market sentiment feed: funding rates + whale detection
        from src.data.market_sentiment import MarketSentimentFeed
        self._sentiment_feed = MarketSentimentFeed(
            event_bus=self.event_bus,
            symbols=symbols,
            poll_interval=300,  # Every 5 minutes
        )
        await self._sentiment_feed.start()

        logger.info("Data components initialized (including sentiment feed)")

    async def _init_risk(self) -> None:
        """Inicializa el sistema de gestion de riesgo.

        Crea todos los componentes de riesgo:
        - PortfolioTracker: rastrea el estado actual del portafolio.
        - PositionSizer: calcula el tamano apropiado de cada posicion.
        - CircuitBreaker: detiene el trading si se exceden limites criticos.
        - RiskManager: orquesta la evaluacion de riesgo de cada senal.

        El equity inicial se obtiene del balance del exchange en modo live,
        o se usa un valor por defecto en modo paper.
        """
        from src.risk.portfolio_tracker import PortfolioTracker
        from src.risk.position_sizer import PositionSizer
        from src.risk.circuit_breaker import CircuitBreaker
        from src.risk.risk_manager import RiskManager
        from decimal import Decimal

        # Determinar la moneda quote a partir de los pares configurados
        quote_currency = "USDC"
        if self.config.pairs:
            parts = self.config.pairs[0].symbol.split("/")
            if len(parts) == 2:
                quote_currency = parts[1]

        # Obtener equity inicial del balance del exchange
        if self._exchange is not None:
            balance = await self._exchange.fetch_balance()
            logger.info("Exchange balance for %s: %s", quote_currency, balance.get(quote_currency))
            quote_balance = balance.get(quote_currency, {}).get("total", 0)
            # Convertir a Decimal para precision; usar 1000 como minimo de seguridad
            initial_equity = Decimal(str(quote_balance)) if quote_balance else Decimal("1000")
        else:
            # Modo paper: usar equity inicial por defecto para simulacion
            initial_equity = Decimal("10000")
            logger.info("Paper mode: using default initial equity of %s %s", initial_equity, quote_currency)

        # Rastreador del portafolio: mantiene equity actual, posiciones, PnL
        self._portfolio_tracker = PortfolioTracker(
            event_bus=self.event_bus,
            initial_equity=initial_equity,
        )

        # Calculador de tamano de posicion basado en porcentaje del portafolio
        self._position_sizer = PositionSizer(
            max_position_pct=self.config.risk.max_position_pct,
        )

        # Circuit breaker: detiene todo el trading si se exceden limites criticos
        self._circuit_breaker = CircuitBreaker(
            event_bus=self.event_bus,
            config=self.config.risk,
            initial_equity=initial_equity,
        )

        # Risk manager: evalua cada senal contra todas las reglas de riesgo
        self._risk_manager = RiskManager(
            event_bus=self.event_bus,
            config=self.config.risk,
            portfolio=self._portfolio_tracker,
            sizer=self._position_sizer,
            circuit_breaker=self._circuit_breaker,
        )

        # Trailing stop: reemplaza take-profit fijo con trailing dinámico (3%)
        from src.risk.trailing_stop import TrailingStopManager
        self._trailing_stop = TrailingStopManager(
            event_bus=self.event_bus,
            trailing_pct=Decimal("0.03"),
        )

        # Conectar fills al trailing stop para trackear nuevas posiciones
        from src.core.events import FillEvent
        async def _on_fill_trailing(event: FillEvent) -> None:
            self._trailing_stop.track(event.symbol, event.side.value, event.price)
        self.event_bus.subscribe(FillEvent, _on_fill_trailing, name="TrailingStop.fill")

        logger.info("Risk management initialized (equity: %s %s)", initial_equity, quote_currency)

    async def _init_execution(self) -> None:
        """Inicializa el sistema de ejecucion de ordenes.

        Selecciona entre PaperOrderExecutor (simulado) y OrderExecutor (real)
        segun el modo configurado. Tambien inicializa el OrderManager (gestor
        de ordenes abiertas) y el FillHandler (procesador de ejecuciones).

        En modo paper, las ordenes se simulan usando datos del DataStore.
        En modo live, las ordenes se envian al exchange real.
        """
        from src.execution.order_manager import OrderManager
        from src.execution.fill_handler import FillHandler

        # Seleccionar el executor segun el modo de operacion
        if self.config.exchange.mode == "paper":
            # Modo paper: simular ejecuciones usando el DataStore
            from src.execution.paper_executor import PaperOrderExecutor

            self._order_executor = PaperOrderExecutor(
                event_bus=self.event_bus,
                data_store=self._data_store,
            )
            logger.info("Using PaperOrderExecutor (paper trading mode)")
        else:
            # Modo live: enviar ordenes reales al exchange
            from src.execution.order_executor import OrderExecutor

            self._order_executor = OrderExecutor(
                event_bus=self.event_bus,
                exchange_config=self.config.exchange,
            )

        await self._order_executor.start()

        # Gestor de ordenes abiertas: rastrea y gestiona ordenes activas
        self._order_manager = OrderManager(
            event_bus=self.event_bus,
            order_executor=self._order_executor,
        )
        await self._order_manager.start()

        # Handler de fills: procesa ejecuciones, actualiza DB y notifica
        self._fill_handler = FillHandler(
            event_bus=self.event_bus,
            repository=self._repository,
            notifier=self._telegram_notifier,  # Puede ser None aqui; se actualiza en _init_monitoring
        )
        await self._fill_handler.start()
        logger.info("Execution components initialized")

    async def _init_ai(self) -> None:
        """Inicializa el pipeline de features y el modelo de inteligencia artificial.

        Crea el FeaturePipeline que prepara los datos de mercado para el modelo.
        Si AI esta habilitado, carga dinamicamente la clase del modelo especificada
        en la configuracion. Si no, usa DummyModel que siempre retorna senales neutras.
        """
        from src.ai.feature_pipeline import FeaturePipeline

        # Pipeline de features: transforma datos de mercado en inputs para el modelo
        self._feature_pipeline = FeaturePipeline(event_bus=self.event_bus)

        if self.config.ai.enabled:
            # Cargar modelo AI real desde la ruta especificada en configuracion
            self._ai_model = self._load_ai_model()
            await self._ai_model.warmup({})  # Precalentar el modelo (cargar pesos, etc.)
            logger.info("AI model loaded: %s", self._ai_model.model_id)
        else:
            # AI desactivado: usar modelo dummy que no afecta las senales
            from src.ai.dummy_model import DummyModel
            self._ai_model = DummyModel()
            logger.info("AI disabled, using DummyModel")

    def _load_ai_model(self) -> Any:
        """Carga dinamicamente la clase del modelo AI desde la ruta de configuracion.

        Usa importlib para importar el modulo y obtener la clase especificada
        en config.ai.model_class (ej. "src.ai.lstm_model.LSTMModel").

        Retorna:
            Instancia del modelo AI configurado.

        Lanza:
            ConfigError: Si la ruta es invalida o la clase no se puede importar.
        """
        class_path = self.config.ai.model_class
        # Separar "src.ai.lstm_model.LSTMModel" en modulo y clase
        parts = class_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ConfigError(f"Invalid model_class path: {class_path}")

        module_path, class_name = parts
        try:
            module = importlib.import_module(module_path)   # Importar el modulo
            model_cls = getattr(module, class_name)          # Obtener la clase del modulo
        except (ImportError, AttributeError) as e:
            raise ConfigError(f"Cannot load AI model {class_path}: {e}") from e

        # Instanciar el modelo con los parametros de configuracion
        return model_cls(config=self.config.ai.config)

    async def _init_strategies(self) -> None:
        """Inicializa el StrategyManager y le pasa el modelo AI y el feature pipeline.

        Carga los parametros de AI del config para inyectar ai_weight a las estrategias.
        El StrategyManager se encarga de crear las instancias de estrategia correctas
        para cada par de trading segun la configuracion.
        """
        from src.strategy.strategy_manager import StrategyManager

        # Cargar parametros AI del config para inyectar ai_weight a las estrategias
        strategy_params: dict[str, dict[str, Any]] = {}
        if self.config.ai.enabled and self.config.ai.config:
            ai_weight = self.config.ai.config.get("ai_weight", 1.5)
            # Inyectar ai_weight como parametro de cada tipo de estrategia
            strategy_params["swing"] = {"ai_weight": ai_weight}

        # Crear el gestor de estrategias con todos los componentes necesarios
        self._strategy_manager = StrategyManager(
            event_bus=self.event_bus,
            config=self.config,
            strategy_params=strategy_params if strategy_params else None,
            ai_model=self._ai_model,
            feature_pipeline=self._feature_pipeline,
        )
        await self._strategy_manager.start()

        # Inject sentiment feed into strategies
        if hasattr(self, '_sentiment_feed') and self._sentiment_feed is not None:
            for strategy in self._strategy_manager._strategies:
                if hasattr(strategy, '_sentiment_feed'):
                    strategy._sentiment_feed = self._sentiment_feed

        logger.info("Strategies initialized")

    async def _init_monitoring(self) -> None:
        """Inicializa los componentes de monitoreo y notificaciones.

        Crea:
        - TelegramNotifier: envia alertas de trades, errores y estado a Telegram.
        - HealthCheck: verifica periodicamente la salud de todos los componentes.

        Tambien actualiza el FillHandler con la referencia al notificador,
        ya que el notificador se crea despues del FillHandler.
        """
        from src.monitoring.telegram_notifier import TelegramNotifier
        from src.monitoring.health_check import HealthCheck

        # Crear notificador de Telegram para alertas
        self._telegram_notifier = TelegramNotifier(
            event_bus=self.event_bus,
            bot_token=self.config.telegram.bot_token,
            chat_id=self.config.telegram.chat_id,
        )
        await self._telegram_notifier.start()

        # Actualizar el FillHandler con el notificador ahora que esta inicializado
        # Esto es necesario porque el FillHandler se crea antes que el notificador
        # debido al orden de dependencias de inicializacion
        if self._fill_handler and hasattr(self._fill_handler, "_notifier"):
            self._fill_handler._notifier = self._telegram_notifier

        # Obtener referencia al engine de la DB para verificaciones de salud
        from src.storage.db import get_engine as get_db_engine
        try:
            db_engine = get_db_engine()
        except RuntimeError:
            db_engine = None  # La DB podria no estar inicializada en algunos modos

        # Crear monitor de salud que verifica periodicamente todos los componentes
        self._health_check = HealthCheck(
            interval_seconds=60,              # Verificar cada 60 segundos
            notifier=self._telegram_notifier,
            exchange=self._exchange,
            db_engine=db_engine,
            ai_model=self._ai_model,
            ws_feed=self._ws_feed,
        )
        logger.info("Monitoring initialized")


async def run(config_path: str = "config/settings.yaml") -> None:
    """Punto de entrada: carga la configuracion y ejecuta el engine.

    Esta funcion es el entry point principal del sistema CryptoTrader.
    Carga la configuracion desde los archivos YAML y variables de entorno,
    crea una instancia del Engine y la ejecuta.

    Parametros:
        config_path: Ruta al archivo YAML principal de configuracion.
    """
    config = load_config(settings_path=config_path)
    engine = Engine(config)
    await engine.start()
