"""Manejador de ejecuciones: procesa FillEvents para persistencia y notificaciones.

Se suscribe a FillEvent en el EventBus y para cada ejecucion:
1. Persiste el registro de la operacion en almacenamiento via el repositorio.
2. Envia notificaciones (ej. Telegram) sobre la ejecucion.

Nota: El seguimiento del portafolio es manejado directamente por el
PortfolioTracker, que tiene su propia suscripcion al FillEvent en el
EventBus. El FillHandler NO duplica ese trabajo.

Este es el consumidor descendente de ejecuciones -- NO coloca ordenes.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import FillEvent

logger = logging.getLogger(__name__)


class FillHandler:
    """Procesa FillEvents: persiste operaciones y envia notificaciones.

    Las actualizaciones del portafolio se omiten intencionalmente aqui
    porque el :class:`~src.risk.portfolio_tracker.PortfolioTracker` se
    suscribe a ``FillEvent`` en el EventBus de forma independiente.

    El pipeline de procesamiento tiene dos etapas:
    1. Persistencia en base de datos via el repositorio.
    2. Envio de notificaciones via el notificador.

    Parametros
    ----------
    event_bus :
        EventBus central para suscribirse a FillEvent.
    repository :
        Persiste registros de operaciones en la base de datos via
        ``save_trade(fill: FillEvent)``. Puede ser ``None`` si el
        almacenamiento no esta inicializado aun.
    notifier :
        Envia notificaciones de operaciones via
        ``notify_fill(fill: FillEvent)``. Puede ser ``None`` si las
        notificaciones estan deshabilitadas.
    """

    def __init__(
        self,
        event_bus: EventBus,
        repository: Any | None = None,
        notifier: Any | None = None,
    ) -> None:
        self._event_bus = event_bus  # Bus de eventos central
        self._repository = repository  # Repositorio para persistencia de operaciones
        self._notifier = notifier  # Servicio de notificaciones (ej. Telegram)
        self._running = False  # Indicador de estado de ejecucion
        self._fill_count = 0  # Contador de ejecuciones procesadas exitosamente

    # -- Ciclo de vida -------------------------------------------------------

    async def start(self) -> None:
        """Se suscribe a FillEvent en el EventBus.

        Registra el manejador de ejecuciones para comenzar a procesar
        fills entrantes del sistema.
        """
        if self._running:
            logger.warning("FillHandler is already running")
            return

        # Registrar el manejador de fills en el bus de eventos
        self._event_bus.subscribe(
            FillEvent,
            self._handle_fill,
            name="FillHandler",
        )
        self._running = True
        logger.info("FillHandler started")

    async def stop(self) -> None:
        """Cancela la suscripcion a FillEvent.

        Detiene el procesamiento de ejecuciones y registra la cantidad
        total de fills procesados durante la sesion.
        """
        if not self._running:
            return

        self._running = False
        # Desuscribirse del bus de eventos
        self._event_bus.unsubscribe(FillEvent, self._handle_fill)
        logger.info("FillHandler stopped (processed %d fills)", self._fill_count)

    # -- Manejador de Eventos ------------------------------------------------

    async def _handle_fill(self, event: FillEvent) -> None:
        """Procesa un FillEvent individual a traves del pipeline.

        Ejecuta las dos etapas del pipeline en orden:
        1. Persistencia en almacenamiento.
        2. Envio de notificacion.

        Parametros
        ----------
        event :
            Evento de ejecucion con todos los detalles de la operacion
            (simbolo, lado, cantidad, precio, comisiones, etc.).
        """
        if not self._running:
            return

        logger.info(
            "Processing fill: %s %s %s qty=%s @ %s fee=%s %s (exchange_id=%s)",
            event.side.value,
            event.symbol,
            event.strategy_name,
            event.quantity,
            event.price,
            event.fee,
            event.fee_currency,
            event.exchange_order_id,
        )

        # Etapa 1: Persistir la operacion en almacenamiento
        await self._persist_trade(event)

        # Etapa 2: Enviar notificacion de la operacion
        await self._send_notification(event)

        # Incrementar el contador de ejecuciones procesadas exitosamente
        self._fill_count += 1

    # -- Etapa 1: Persistencia -----------------------------------------------

    async def _persist_trade(self, event: FillEvent) -> None:
        """Persiste la ejecucion como registro de operacion en la base de datos.

        Si no hay repositorio configurado, la operacion se omite
        silenciosamente. Los errores de persistencia se registran pero
        no detienen el flujo de procesamiento.

        Parametros
        ----------
        event :
            Evento de ejecucion a persistir.
        """
        if self._repository is None:
            # No hay repositorio configurado; saltar la persistencia
            logger.debug("No repository configured; skipping trade persistence")
            return

        try:
            await self._repository.save_trade(event)
            logger.debug("Trade persisted for fill %s", event.event_id)
        except Exception:
            # El error de persistencia no debe bloquear el procesamiento
            logger.exception(
                "Failed to persist trade for fill %s", event.event_id
            )

    # -- Etapa 2: Notificacion -----------------------------------------------

    async def _send_notification(self, event: FillEvent) -> None:
        """Envia una notificacion sobre la ejecucion.

        Si no hay notificador configurado, se omite silenciosamente.
        Los fallos en la notificacion nunca deben bloquear el trading,
        por lo que los errores se registran pero no se propagan.

        Parametros
        ----------
        event :
            Evento de ejecucion sobre el cual notificar.
        """
        if self._notifier is None:
            return  # Notificaciones deshabilitadas

        try:
            await self._notifier.notify_fill(event)
            logger.debug("Notification sent for fill %s", event.event_id)
        except Exception:
            # El fallo en la notificacion nunca debe bloquear el trading
            logger.exception(
                "Failed to send notification for fill %s", event.event_id
            )

    # -- Diagnosticos --------------------------------------------------------

    @property
    def fill_count(self) -> int:
        """Numero de ejecuciones procesadas exitosamente.

        Retorna
        -------
        int
            Cantidad total de fills que pasaron por el pipeline completo.
        """
        return self._fill_count

    @property
    def is_running(self) -> bool:
        """Indica si el manejador de ejecuciones esta activo y procesando eventos."""
        return self._running
