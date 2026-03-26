"""Diagnosticos periodicos de verificacion de salud del sistema.

Se ejecuta en un intervalo configurable, verificando los subsistemas clave:
  - Conectividad del feed WebSocket
  - Accesibilidad de la API REST del exchange
  - Conexion a la base de datos
  - Disponibilidad del modelo de IA

Los resultados se registran en el log y, en caso de fallo, se reenvian
al notificador de Telegram para alertar al operador.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.ai.model_interface import ModelInterface
    from src.monitoring.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class HealthCheck:
    """Verifica periodicamente que todos los subsistemas estan operativos.

    Esta clase ejecuta un bucle en segundo plano que, a intervalos regulares,
    prueba la conectividad y disponibilidad de cada componente critico del
    sistema de trading. Si alguna verificacion falla, se envia una alerta
    a traves del notificador de Telegram.

    Uso::

        hc = HealthCheck(
            interval_seconds=60,
            notifier=telegram_notifier,
            exchange=ccxt_exchange,
            db_engine=sqlalchemy_engine,
            ai_model=model,
            ws_feed=ws_feed,
        )
        await hc.start()   # Inicia el bucle en segundo plano
        await hc.stop()    # Cancela el bucle

    Atributos internos
    ------------------
    _interval : int
        Segundos entre cada verificacion.
    _notifier : TelegramNotifier | None
        Notificador para enviar alertas en caso de fallo.
    _exchange : Any
        Instancia del exchange CCXT para verificar la API REST.
    _db_engine : Any
        Motor de base de datos SQLAlchemy para verificar la conexion.
    _ai_model : ModelInterface | None
        Interfaz del modelo de IA para verificar su disponibilidad.
    _ws_feed : Any
        Feed WebSocket para verificar la conectividad en tiempo real.
    _task : asyncio.Task | None
        Referencia a la tarea del bucle en segundo plano.
    _last_report : dict
        Ultimo reporte de salud generado.
    """

    def __init__(
        self,
        interval_seconds: int = 60,
        notifier: TelegramNotifier | None = None,
        exchange: Any = None,
        db_engine: Any = None,
        ai_model: ModelInterface | None = None,
        ws_feed: Any = None,
    ) -> None:
        self._interval = interval_seconds  # Intervalo entre verificaciones (en segundos)
        self._notifier = notifier  # Notificador de Telegram para alertas
        self._exchange = exchange  # Instancia del exchange (CCXT)
        self._db_engine = db_engine  # Motor asincrono de base de datos
        self._ai_model = ai_model  # Interfaz del modelo de IA
        self._ws_feed = ws_feed  # Feed de datos en tiempo real via WebSocket
        self._task: asyncio.Task | None = None  # Tarea del bucle en segundo plano
        self._last_report: dict[str, Any] = {}  # Almacena el ultimo reporte generado

    # ── Ciclo de Vida ─────────────────────────────────────────────

    async def start(self) -> None:
        """Inicia el bucle periodico de verificacion de salud.

        Si el bucle ya esta en ejecucion, no hace nada (evita duplicados).
        La verificacion se ejecuta como una tarea asincrona en segundo plano.
        """
        if self._task is not None:
            return  # Ya esta en ejecucion; no crear otra tarea.
        self._task = asyncio.create_task(self._loop(), name="HealthCheck")
        logger.info("HealthCheck started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        """Cancela el bucle de verificacion de salud.

        Espera a que la tarea termine correctamente despues de la cancelacion.
        """
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task  # Esperar a que la tarea procese la cancelacion.
            except asyncio.CancelledError:
                pass  # La cancelacion es el comportamiento esperado.
            self._task = None
            logger.info("HealthCheck stopped")

    # ── API Publica ───────────────────────────────────────────────

    @property
    def last_report(self) -> dict[str, Any]:
        """Retorna el reporte de salud mas reciente.

        Retorno
        -------
        dict[str, Any]
            Diccionario con el estado de cada componente: ``True`` si esta
            sano, o un string con el mensaje de error si fallo.
        """
        return dict(self._last_report)

    async def run_once(self) -> dict[str, Any]:
        """Ejecuta una sola ronda de verificacion de salud y retorna el reporte.

        Verifica cada subsistema de forma secuencial y construye un reporte
        con el resultado de cada verificacion. Si alguna falla, se envia
        una notificacion a Telegram.

        Retorno
        -------
        dict[str, Any]
            Reporte con el estado de cada componente.
        """
        report: dict[str, Any] = {}

        # Ejecutar cada verificacion individual y agregar al reporte.
        report["websocket"] = await self._check_websocket()
        report["exchange_rest"] = await self._check_exchange_rest()
        report["database"] = await self._check_database()
        report["ai_model"] = await self._check_ai_model()

        # Guardar el reporte mas reciente para consultas posteriores.
        self._last_report = report

        # Evaluar si todos los componentes estan sanos.
        all_ok = all(v is True for v in report.values())
        if all_ok:
            logger.debug("Health check passed: %s", report)
        else:
            # Al menos un componente fallo; registrar advertencia y notificar.
            logger.warning("Health check failures: %s", report)
            if self._notifier is not None:
                await self._notifier.notify_health(report)

        return report

    # ── Bucle en Segundo Plano ────────────────────────────────────

    async def _loop(self) -> None:
        """Ejecuta verificaciones de salud al intervalo configurado.

        Este bucle se ejecuta indefinidamente hasta que la tarea sea
        cancelada. Los errores inesperados se capturan para evitar que
        el bucle se detenga prematuramente.
        """
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise  # Propagar la cancelacion para detener el bucle limpiamente.
            except Exception:
                logger.exception("Unexpected error in health-check loop")
            await asyncio.sleep(self._interval)  # Esperar hasta la proxima verificacion.

    # ── Verificaciones Individuales ───────────────────────────────

    async def _check_websocket(self) -> bool | str:
        """Verifica la conectividad del feed WebSocket.

        Si el feed no esta configurado, se omite la verificacion y se
        retorna ``True``. Se espera que el feed exponga una propiedad o
        metodo ``is_connected``.

        Retorno
        -------
        bool | str
            ``True`` si esta conectado, o un string con el error.
        """
        if self._ws_feed is None:
            return True  # No esta configurado; omitir verificacion.
        try:
            # Verificar si el feed expone una propiedad o metodo ``is_connected``.
            connected = getattr(self._ws_feed, "is_connected", None)
            if callable(connected):
                # Soportar tanto metodos sincronos como asincronos.
                connected = await connected() if asyncio.iscoroutinefunction(connected) else connected()
            if connected is False:
                return "WebSocket disconnected"
            return True
        except Exception as exc:
            return f"WebSocket check error: {exc}"

    async def _check_exchange_rest(self) -> bool | str:
        """Verifica la accesibilidad de la API REST del exchange mediante ``fetch_time``.

        Utiliza el metodo ``fetch_time`` de CCXT como ping ligero para
        confirmar que la API del exchange responde correctamente.

        Retorno
        -------
        bool | str
            ``True`` si la API responde, o un string con el error.
        """
        if self._exchange is None:
            return True  # No esta configurado; omitir verificacion.
        try:
            await self._exchange.fetch_time()  # Ping ligero al exchange.
            return True
        except Exception as exc:
            return f"REST API error: {exc}"

    async def _check_database(self) -> bool | str:
        """Ejecuta una consulta ligera contra la base de datos para verificar la conexion.

        Utiliza ``SELECT 1`` como consulta de prueba minima.

        Retorno
        -------
        bool | str
            ``True`` si la conexion funciona, o un string con el error.
        """
        if self._db_engine is None:
            return True  # No esta configurado; omitir verificacion.
        try:
            from sqlalchemy import text

            # Ejecutar una consulta minima para verificar la conexion.
            async with self._db_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            return f"DB error: {exc}"

    async def _check_ai_model(self) -> bool | str:
        """Delega la verificacion de salud al metodo ``health_check`` del modelo de IA.

        Retorno
        -------
        bool | str
            ``True`` si el modelo esta sano, o un string con el error.
        """
        if self._ai_model is None:
            return True  # No esta configurado; omitir verificacion.
        try:
            healthy = await self._ai_model.health_check()
            if not healthy:
                return "AI model reports unhealthy"
            return True
        except Exception as exc:
            return f"AI model error: {exc}"
