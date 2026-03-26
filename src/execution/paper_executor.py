"""Ejecutor de ordenes en papel: simula ejecuciones sin tocar un exchange real.

Se suscribe a OrderEvent en el EventBus (misma interfaz que OrderExecutor).
Para cada orden entrante:
1. Registra la orden claramente marcada como PAPER (papel).
2. Determina un precio de ejecucion desde el DataStore (ultimo tick) o
   utiliza como respaldo el precio propio de la orden / stop_loss como estimacion.
3. Genera un FillEvent sintetico con un ID de orden prefijado con ``paper-``
   y lo publica de vuelta al EventBus.

No se requiere conexion real a un exchange. ``cancel_order`` y
``fetch_order_status`` son operaciones sin efecto (no-ops) para que el
OrderManager pueda funcionar sin cambios.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import FillEvent, OrderEvent, OrderType, Side
from src.data.data_store import DataStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass auxiliar para rastrear ordenes en papel
# ---------------------------------------------------------------------------

@dataclass
class PaperOrder:
    """Registro de una orden simulada en papel.

    Almacena todos los detalles de una orden ejecutada en modo paper trading,
    incluyendo el precio solicitado, el precio de ejecucion simulado y las
    comisiones calculadas.

    Atributos
    ---------
    order_id : str
        ID unico generado para la orden en papel (prefijo "paper-").
    client_order_id : str
        ID del cliente asignado por la estrategia.
    symbol : str
        Par de trading (ej. "BTC/USDT").
    side : Side
        Lado de la orden (BUY o SELL).
    order_type : OrderType
        Tipo de orden (MARKET, LIMIT, etc.).
    quantity : Decimal
        Cantidad operada.
    requested_price : Decimal | None
        Precio solicitado originalmente (puede ser None para ordenes de mercado).
    fill_price : Decimal
        Precio de ejecucion simulado.
    fee : Decimal
        Comision simulada aplicada a la operacion.
    fee_currency : str
        Moneda en la que se expresa la comision.
    status : str
        Estado de la orden (siempre "filled" para ordenes en papel).
    timestamp : int
        Marca de tiempo en milisegundos de la creacion de la orden.
    """

    order_id: str
    client_order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    requested_price: Decimal | None
    fill_price: Decimal
    fee: Decimal
    fee_currency: str
    status: str = "filled"  # Las ordenes en papel siempre se ejecutan inmediatamente
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


# ---------------------------------------------------------------------------
# PaperOrderExecutor
# ---------------------------------------------------------------------------

class PaperOrderExecutor:
    """Simula la ejecucion de ordenes para trading en papel (paper trading).

    Proporciona la misma interfaz publica que ``OrderExecutor`` para que el
    resto del sistema (OrderManager, FillHandler, etc.) pueda funcionar sin
    cambios. Las ordenes se "ejecutan" instantaneamente usando precios del
    mercado obtenidos del DataStore.

    Parametros
    ----------
    event_bus :
        EventBus central para suscribirse a OrderEvent y publicar FillEvent.
    data_store :
        DataStore utilizado para obtener el ultimo precio de mercado para
        una simulacion realista de ejecucion.
    simulated_fee_rate :
        Tasa de comision aplicada a cada ejecucion en papel (como fraccion
        decimal, ej. ``Decimal("0.001")`` para 0.1%). Valor predeterminado: cero.
    fee_currency :
        Cadena de moneda asociada a las comisiones simuladas (ej. ``"USDC"``).
    """

    def __init__(
        self,
        event_bus: EventBus,
        data_store: DataStore,
        *,
        simulated_fee_rate: Decimal = Decimal(0),
        fee_currency: str = "USDC",
    ) -> None:
        self._event_bus = event_bus  # Bus de eventos central
        self._data_store = data_store  # Almacen de datos de mercado
        self._simulated_fee_rate = simulated_fee_rate  # Tasa de comision simulada
        self._fee_currency = fee_currency  # Moneda para las comisiones

        self._running = False  # Indicador de estado de ejecucion
        self._orders: list[PaperOrder] = []  # Historial de ordenes en papel ejecutadas

    # -- Ciclo de vida -------------------------------------------------------

    async def start(self) -> None:
        """Se suscribe a OrderEvent en el EventBus.

        Registra el manejador de eventos para comenzar a procesar
        ordenes en modo paper trading.
        """
        if self._running:
            logger.warning("PaperOrderExecutor is already running")
            return

        # Registrar el manejador de ordenes en el bus de eventos
        self._event_bus.subscribe(
            OrderEvent,
            self._handle_order_event,
            name="PaperOrderExecutor",
        )
        self._running = True
        logger.info("PaperOrderExecutor started (paper trading mode)")

    async def stop(self) -> None:
        """Cancela la suscripcion a eventos.

        Detiene el procesamiento de ordenes y registra la cantidad
        total de ordenes ejecutadas durante la sesion.
        """
        if not self._running:
            return

        self._running = False
        # Desuscribirse del bus de eventos
        self._event_bus.unsubscribe(OrderEvent, self._handle_order_event)
        logger.info(
            "PaperOrderExecutor stopped (%d paper orders executed)",
            len(self._orders),
        )

    # -- Manejador de Eventos ------------------------------------------------

    async def _handle_order_event(self, event: OrderEvent) -> None:
        """Simula una ejecucion (fill) para el OrderEvent entrante.

        Flujo completo:
        1. Determina el precio de ejecucion simulado.
        2. Calcula la comision simulada.
        3. Genera un ID unico para la orden en papel.
        4. Registra la orden en el historial interno.
        5. Construye y publica un FillEvent sintetico.

        Parametros
        ----------
        event :
            Evento de orden con los detalles de la operacion a simular.
        """
        if not self._running:
            logger.warning(
                "PaperOrderExecutor not running; ignoring OrderEvent %s",
                event.event_id,
            )
            return

        # Determinar el precio de ejecucion simulado
        fill_price = self._determine_fill_price(event)
        # Calcular la comision basada en precio y cantidad
        fee = self._calculate_fee(fill_price, event.quantity)
        # Generar un ID unico con prefijo "paper-" para identificacion
        paper_order_id = f"paper-{uuid.uuid4().hex[:12]}"

        logger.info(
            "[PAPER] %s %s %s qty=%s price=%s (simulated fill @ %s, fee=%s)",
            event.order_type.value,
            event.side.value,
            event.symbol,
            event.quantity,
            event.price,
            fill_price,
            fee,
        )

        # Registrar la orden en papel en el historial interno
        paper_order = PaperOrder(
            order_id=paper_order_id,
            client_order_id=event.client_order_id,
            symbol=event.symbol,
            side=event.side,
            order_type=event.order_type,
            quantity=event.quantity,
            requested_price=event.price,
            fill_price=fill_price,
            fee=fee,
            fee_currency=self._fee_currency,
        )
        self._orders.append(paper_order)

        # Construir y publicar el FillEvent sintetico al bus de eventos
        fill_event = FillEvent(
            symbol=event.symbol,
            side=event.side,
            quantity=event.quantity,
            price=fill_price,
            fee=fee,
            fee_currency=self._fee_currency,
            exchange_order_id=paper_order_id,
            client_order_id=event.client_order_id,
            strategy_name=event.strategy_name,
        )
        await self._event_bus.publish(fill_event)

        logger.info(
            "[PAPER] Fill published: %s %s %s qty=%s @ %s (id=%s)",
            event.side.value,
            event.symbol,
            event.order_type.value,
            event.quantity,
            fill_price,
            paper_order_id,
        )

    # -- Determinacion de Precio ---------------------------------------------

    def _determine_fill_price(self, event: OrderEvent) -> Decimal:
        """Determina un precio de ejecucion realista para la orden en papel.

        Orden de prioridad:
        1. Precio del ultimo tick desde el DataStore (campo ``last``).
        2. El propio precio de la orden (para ordenes limite).
        3. El precio de ``stop_loss`` de la orden (como estimacion aproximada).
        4. Respaldo a ``Decimal(0)`` (no deberia ocurrir en la practica).

        Parametros
        ----------
        event :
            Evento de orden del cual extraer informacion de precio.

        Retorna
        -------
        Decimal
            Precio de ejecucion simulado.
        """
        # Prioridad 1: Obtener el precio mas reciente del mercado
        tick = self._data_store.get_latest_tick(event.symbol)
        if tick is not None and tick.last > 0:
            return tick.last

        # Prioridad 2: Usar el precio de la orden (ordenes limite)
        if event.price is not None and event.price > 0:
            return event.price

        # Prioridad 3: Usar el precio de stop-loss como estimacion
        if event.stop_loss is not None and event.stop_loss > 0:
            return event.stop_loss

        # Respaldo: no hay precio disponible
        logger.warning(
            "[PAPER] No price available for %s; using Decimal(0)",
            event.symbol,
        )
        return Decimal(0)

    def _calculate_fee(self, price: Decimal, quantity: Decimal) -> Decimal:
        """Calcula la comision simulada para una ejecucion en papel.

        La comision se calcula como: precio * cantidad * tasa_de_comision.

        Parametros
        ----------
        price :
            Precio de ejecucion simulado.
        quantity :
            Cantidad de la operacion.

        Retorna
        -------
        Decimal
            Monto de la comision simulada.
        """
        return price * quantity * self._simulated_fee_rate

    # -- Metodos Sin Efecto (Compatibilidad con OrderManager) ----------------

    async def cancel_order(
        self, exchange_order_id: str, symbol: str
    ) -> dict[str, Any]:
        """Cancelacion sin efecto para ordenes en papel.

        Las ordenes en papel se ejecutan instantaneamente, por lo que no
        hay nada que cancelar. Retorna una respuesta sintetica para
        compatibilidad con el OrderManager.

        Parametros
        ----------
        exchange_order_id :
            ID de la orden en el exchange (en este caso, el ID en papel).
        symbol :
            Par de trading.

        Retorna
        -------
        dict[str, Any]
            Respuesta sintetica indicando estado "canceled".
        """
        logger.info(
            "[PAPER] cancel_order called for %s on %s (no-op)",
            exchange_order_id,
            symbol,
        )
        return {"id": exchange_order_id, "status": "canceled", "info": "paper"}

    async def fetch_order_status(
        self, exchange_order_id: str, symbol: str
    ) -> dict[str, Any]:
        """Consulta de estado sin efecto para ordenes en papel.

        Retorna un estado sintetico de 'filled' (ejecutada) para
        compatibilidad con el OrderManager.

        Parametros
        ----------
        exchange_order_id :
            ID de la orden en el exchange (en este caso, el ID en papel).
        symbol :
            Par de trading.

        Retorna
        -------
        dict[str, Any]
            Respuesta sintetica indicando estado "closed" (ejecutada).
        """
        logger.info(
            "[PAPER] fetch_order_status called for %s on %s (no-op)",
            exchange_order_id,
            symbol,
        )
        return {"id": exchange_order_id, "status": "closed", "info": "paper"}

    # -- API de Inspeccion ---------------------------------------------------

    @property
    def orders(self) -> list[PaperOrder]:
        """Retorna una copia de todas las ordenes en papel registradas.

        Retorna
        -------
        list[PaperOrder]
            Lista con copias de todas las ordenes ejecutadas en modo papel.
        """
        return list(self._orders)

    @property
    def is_running(self) -> bool:
        """Indica si el ejecutor de ordenes en papel esta activo y procesando eventos."""
        return self._running
