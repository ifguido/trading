"""Gestor de ordenes: rastrea las ordenes abiertas y su ciclo de vida.

Mantiene un registro en memoria de todas las ordenes rastreadas, gestiona
ejecuciones parciales, proporciona metodos de consulta y sincroniza
periodicamente el estado de las ordenes con el exchange. Soporta cierre
ordenado cancelando todas las ordenes abiertas.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import FillEvent, OrderEvent, Side, OrderType

logger = logging.getLogger(__name__)

# Intervalo en segundos para sincronizar ordenes abiertas con el exchange
_DEFAULT_SYNC_INTERVAL = 30.0


class OrderStatus(str, Enum):
    """Estados del ciclo de vida de una orden rastreada.

    Define los posibles estados por los que pasa una orden desde su
    creacion hasta su finalizacion.
    """

    PENDING = "pending"          # Enviada al exchange, esperando confirmacion
    OPEN = "open"                # Confirmada por el exchange, en el libro de ordenes
    PARTIALLY_FILLED = "partially_filled"  # Parcialmente ejecutada
    FILLED = "filled"            # Completamente ejecutada
    CANCELLED = "cancelled"      # Cancelada por el usuario o el sistema
    REJECTED = "rejected"        # Rechazada por el exchange
    EXPIRED = "expired"          # Expirada por limite de tiempo


@dataclass
class TrackedOrder:
    """Representacion en memoria de una orden rastreada.

    Almacena todos los detalles de una orden activa o historica,
    incluyendo cantidades ejecutadas, precios promedio y comisiones
    acumuladas.

    Atributos
    ---------
    client_order_id : str
        ID unico asignado por el cliente/estrategia.
    exchange_order_id : str
        ID asignado por el exchange (vacio hasta la confirmacion).
    symbol : str
        Par de trading (ej. "BTC/USDT").
    side : Side
        Lado de la orden (BUY o SELL).
    order_type : OrderType
        Tipo de orden (MARKET, LIMIT, etc.).
    quantity : Decimal
        Cantidad total solicitada.
    price : Decimal | None
        Precio solicitado (None para ordenes de mercado).
    stop_price : Decimal | None
        Precio de activacion para ordenes stop-loss.
    status : OrderStatus
        Estado actual de la orden en su ciclo de vida.
    filled_quantity : Decimal
        Cantidad acumulada ejecutada hasta el momento.
    average_fill_price : Decimal
        Precio promedio de ejecucion ponderado.
    fee : Decimal
        Comisiones acumuladas totales.
    fee_currency : str
        Moneda en la que se expresan las comisiones.
    strategy_name : str
        Nombre de la estrategia que genero la orden.
    created_at : int
        Marca de tiempo de creacion (milisegundos epoch).
    updated_at : int
        Marca de tiempo de la ultima actualizacion (milisegundos epoch).
    """

    client_order_id: str
    exchange_order_id: str = ""
    symbol: str = ""
    side: Side = Side.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = Decimal(0)
    price: Decimal | None = None
    stop_price: Decimal | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal(0)
    average_fill_price: Decimal = Decimal(0)
    fee: Decimal = Decimal(0)
    fee_currency: str = ""
    strategy_name: str = ""
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def remaining_quantity(self) -> Decimal:
        """Cantidad pendiente de ejecucion.

        Retorna
        -------
        Decimal
            Diferencia entre la cantidad total solicitada y la ejecutada.
        """
        return self.quantity - self.filled_quantity

    @property
    def is_terminal(self) -> bool:
        """Indica si la orden esta en un estado final (no puede cambiar mas).

        Retorna
        -------
        bool
            True si la orden esta ejecutada, cancelada, rechazada o expirada.
        """
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def is_open(self) -> bool:
        """Indica si la orden todavia esta activa en el exchange.

        Retorna
        -------
        bool
            True si la orden esta pendiente, abierta o parcialmente ejecutada.
        """
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        )


class OrderManager:
    """Rastrea todas las ordenes y sincroniza su estado con el exchange.

    Este componente es el registro central de todas las ordenes del sistema.
    Se suscribe a eventos de orden y ejecucion para mantener el estado
    actualizado, y ejecuta un bucle periodico para detectar cambios
    externos (ordenes ejecutadas/canceladas desde la interfaz web del exchange).

    Parametros
    ----------
    event_bus :
        EventBus central para suscribirse a FillEvent y OrderEvent.
    order_executor :
        Referencia al OrderExecutor, usado para consultar el estado de
        ordenes y cancelarlas en el exchange.
    sync_interval :
        Segundos entre sincronizaciones periodicas con el exchange.
    """

    def __init__(
        self,
        event_bus: EventBus,
        order_executor: Any,  # OrderExecutor (se usa Any para evitar importacion circular)
        *,
        sync_interval: float = _DEFAULT_SYNC_INTERVAL,
    ) -> None:
        self._event_bus = event_bus  # Bus de eventos central
        self._executor = order_executor  # Ejecutor de ordenes para operaciones en el exchange
        self._sync_interval = sync_interval  # Intervalo de sincronizacion en segundos

        # Diccionario de ordenes rastreadas: client_order_id -> TrackedOrder
        self._orders: dict[str, TrackedOrder] = {}
        self._lock = asyncio.Lock()  # Candado asincrono para acceso concurrente seguro
        self._sync_task: asyncio.Task | None = None  # Tarea de sincronizacion periodica
        self._running = False  # Indicador de estado de ejecucion

    # -- Ciclo de vida -------------------------------------------------------

    async def start(self) -> None:
        """Se suscribe a eventos e inicia el bucle de sincronizacion periodica.

        Registra manejadores para FillEvent (actualizar estado de ordenes)
        y OrderEvent (rastrear nuevas ordenes automaticamente), y lanza
        la tarea de sincronizacion con el exchange.
        """
        if self._running:
            logger.warning("OrderManager is already running")
            return

        self._running = True

        # Suscribirse a eventos de ejecucion para actualizar ordenes rastreadas
        self._event_bus.subscribe(
            FillEvent,
            self._handle_fill_event,
            name="OrderManager.fill",
        )
        # Suscribirse a eventos de orden para rastreo automatico
        self._event_bus.subscribe(
            OrderEvent,
            self._handle_order_event,
            name="OrderManager.order",
        )

        # Iniciar tarea de sincronizacion periodica con el exchange
        self._sync_task = asyncio.create_task(
            self._sync_loop(),
            name="order-manager-sync",
        )
        logger.info("OrderManager started (sync_interval=%.1fs)", self._sync_interval)

    async def stop(self) -> None:
        """Cancela el bucle de sincronizacion y se desuscribe de eventos.

        Detiene la tarea de sincronizacion periodica de forma limpia
        y elimina todas las suscripciones al bus de eventos.
        """
        if not self._running:
            return

        self._running = False

        # Detener el bucle de sincronizacion periodica
        if self._sync_task is not None and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass  # Cancelacion esperada, no es un error
            self._sync_task = None

        # Desuscribirse de todos los eventos
        self._event_bus.unsubscribe(FillEvent, self._handle_fill_event)
        self._event_bus.unsubscribe(OrderEvent, self._handle_order_event)

        logger.info("OrderManager stopped")

    async def shutdown_cancel_open_orders(self) -> list[str]:
        """Cierre ordenado: cancela todas las ordenes abiertas en el exchange.

        Itera sobre todas las ordenes activas e intenta cancelarlas una
        por una en el exchange. Los errores individuales se registran pero
        no detienen el proceso de cancelacion de las demas ordenes.

        Retorna
        -------
        list[str]
            Lista de client_order_ids de las ordenes canceladas exitosamente.
        """
        open_orders = self.get_open_orders()
        if not open_orders:
            logger.info("No open orders to cancel during shutdown")
            return []

        logger.info(
            "Cancelling %d open orders during shutdown", len(open_orders)
        )

        cancelled: list[str] = []  # Acumulador de ordenes canceladas exitosamente
        for order in open_orders:
            try:
                # Solo cancelar si la orden tiene un ID del exchange asignado
                if order.exchange_order_id:
                    await self._executor.cancel_order(
                        order.exchange_order_id, order.symbol
                    )
                # Actualizar el estado local de la orden
                async with self._lock:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = int(time.time() * 1000)
                cancelled.append(order.client_order_id)
                logger.info(
                    "Cancelled order %s (%s) during shutdown",
                    order.client_order_id,
                    order.exchange_order_id,
                )
            except Exception as exc:
                # Registrar el error pero continuar con las demas ordenes
                logger.error(
                    "Failed to cancel order %s during shutdown: %s",
                    order.client_order_id,
                    exc,
                )

        return cancelled

    # -- API Publica ---------------------------------------------------------

    async def track_order(
        self,
        order_event: OrderEvent,
        exchange_order_id: str = "",
    ) -> TrackedOrder:
        """Comienza a rastrear una nueva orden.

        Se llama desde codigo externo justo despues de colocar una orden,
        o internamente desde el manejador de OrderEvent.

        Parametros
        ----------
        order_event :
            Evento de orden con los detalles de la orden a rastrear.
        exchange_order_id :
            ID asignado por el exchange (puede estar vacio si aun no
            se ha recibido confirmacion).

        Retorna
        -------
        TrackedOrder
            La nueva orden rastreada con estado PENDING.
        """
        async with self._lock:
            tracked = TrackedOrder(
                client_order_id=order_event.client_order_id,
                exchange_order_id=exchange_order_id,
                symbol=order_event.symbol,
                side=order_event.side,
                order_type=order_event.order_type,
                quantity=order_event.quantity,
                price=order_event.price,
                stop_price=order_event.stop_loss,
                status=OrderStatus.PENDING,
                strategy_name=order_event.strategy_name,
            )
            # Registrar la orden en el diccionario interno
            self._orders[order_event.client_order_id] = tracked

        logger.debug(
            "Tracking order %s for %s (%s %s qty=%s)",
            tracked.client_order_id,
            tracked.symbol,
            tracked.side.value,
            tracked.order_type.value,
            tracked.quantity,
        )
        return tracked

    async def update_order_status(
        self,
        client_order_id: str,
        *,
        status: OrderStatus | None = None,
        exchange_order_id: str | None = None,
        filled_quantity: Decimal | None = None,
        average_fill_price: Decimal | None = None,
        fee: Decimal | None = None,
        fee_currency: str | None = None,
    ) -> TrackedOrder | None:
        """Actualiza campos de una orden rastreada existente.

        Solo se actualizan los campos que se proporcionan como argumentos
        (los que no son None). La marca de tiempo de actualizacion se
        establece automaticamente.

        Parametros
        ----------
        client_order_id :
            ID del cliente de la orden a actualizar.
        status :
            Nuevo estado de la orden (opcional).
        exchange_order_id :
            ID del exchange a establecer (opcional).
        filled_quantity :
            Nueva cantidad ejecutada acumulada (opcional).
        average_fill_price :
            Nuevo precio promedio de ejecucion (opcional).
        fee :
            Nueva comision acumulada (opcional).
        fee_currency :
            Nueva moneda de comision (opcional).

        Retorna
        -------
        TrackedOrder | None
            La orden actualizada, o None si no se encontro.
        """
        async with self._lock:
            order = self._orders.get(client_order_id)
            if order is None:
                logger.warning(
                    "Cannot update unknown order %s", client_order_id
                )
                return None

            # Actualizar solo los campos proporcionados
            if status is not None:
                order.status = status
            if exchange_order_id is not None:
                order.exchange_order_id = exchange_order_id
            if filled_quantity is not None:
                order.filled_quantity = filled_quantity
            if average_fill_price is not None:
                order.average_fill_price = average_fill_price
            if fee is not None:
                order.fee = fee
            if fee_currency is not None:
                order.fee_currency = fee_currency

            # Actualizar marca de tiempo de la ultima modificacion
            order.updated_at = int(time.time() * 1000)

        logger.debug(
            "Updated order %s: status=%s filled=%s/%s",
            client_order_id,
            order.status.value,
            order.filled_quantity,
            order.quantity,
        )
        return order

    def get_open_orders(self) -> list[TrackedOrder]:
        """Retorna todas las ordenes que aun estan activas (no terminales).

        Retorna
        -------
        list[TrackedOrder]
            Lista de ordenes con estado PENDING, OPEN o PARTIALLY_FILLED.
        """
        return [o for o in self._orders.values() if o.is_open]

    def get_order(self, client_order_id: str) -> TrackedOrder | None:
        """Busca una orden rastreada por su ID de cliente.

        Parametros
        ----------
        client_order_id :
            ID unico del cliente asignado a la orden.

        Retorna
        -------
        TrackedOrder | None
            La orden encontrada, o None si no existe.
        """
        return self._orders.get(client_order_id)

    def get_orders_by_symbol(self, symbol: str) -> list[TrackedOrder]:
        """Retorna todas las ordenes rastreadas (abiertas y cerradas) para un simbolo.

        Parametros
        ----------
        symbol :
            Par de trading (ej. "BTC/USDT").

        Retorna
        -------
        list[TrackedOrder]
            Todas las ordenes asociadas al simbolo especificado.
        """
        return [o for o in self._orders.values() if o.symbol == symbol]

    def get_orders_by_strategy(self, strategy_name: str) -> list[TrackedOrder]:
        """Retorna todas las ordenes rastreadas para una estrategia dada.

        Parametros
        ----------
        strategy_name :
            Nombre de la estrategia que genero las ordenes.

        Retorna
        -------
        list[TrackedOrder]
            Todas las ordenes asociadas a la estrategia especificada.
        """
        return [
            o for o in self._orders.values()
            if o.strategy_name == strategy_name
        ]

    async def cancel_order(self, client_order_id: str) -> TrackedOrder | None:
        """Cancela una orden en el exchange y actualiza su estado rastreado.

        Verifica que la orden exista, no este en estado terminal y tenga
        un ID del exchange antes de intentar la cancelacion.

        Parametros
        ----------
        client_order_id :
            ID del cliente de la orden a cancelar.

        Retorna
        -------
        TrackedOrder | None
            La orden actualizada, o None si no se encontro o no tiene
            ID del exchange.
        """
        order = self._orders.get(client_order_id)
        if order is None:
            logger.warning("Cannot cancel unknown order %s", client_order_id)
            return None

        # Verificar que la orden no este ya en un estado terminal
        if order.is_terminal:
            logger.warning(
                "Cannot cancel order %s — already in terminal state %s",
                client_order_id,
                order.status.value,
            )
            return order

        # Verificar que la orden tenga un ID del exchange asignado
        if not order.exchange_order_id:
            logger.warning(
                "Cannot cancel order %s — no exchange order ID",
                client_order_id,
            )
            return None

        try:
            # Enviar la solicitud de cancelacion al exchange
            await self._executor.cancel_order(
                order.exchange_order_id, order.symbol
            )
            # Actualizar el estado local de la orden
            async with self._lock:
                order.status = OrderStatus.CANCELLED
                order.updated_at = int(time.time() * 1000)

            logger.info(
                "Cancelled order %s (%s)",
                client_order_id,
                order.exchange_order_id,
            )
        except Exception as exc:
            logger.error(
                "Failed to cancel order %s: %s", client_order_id, exc
            )

        return order

    # -- Manejadores de Eventos ----------------------------------------------

    async def _handle_order_event(self, event: OrderEvent) -> None:
        """Rastrea automaticamente las ordenes cuando se crean.

        Si la orden aun no esta siendo rastreada, se agrega al registro.

        Parametros
        ----------
        event :
            Evento de orden con los detalles de la nueva orden.
        """
        if event.client_order_id not in self._orders:
            await self.track_order(event)

    async def _handle_fill_event(self, event: FillEvent) -> None:
        """Actualiza el estado de la orden rastreada cuando se recibe una ejecucion.

        Acumula la cantidad ejecutada, actualiza el precio promedio y las
        comisiones. Determina si la orden esta completamente ejecutada o
        parcialmente ejecutada.

        Parametros
        ----------
        event :
            Evento de ejecucion con los detalles del fill recibido.
        """
        order = self._orders.get(event.client_order_id)
        if order is None:
            logger.warning(
                "Received FillEvent for untracked order %s",
                event.client_order_id,
            )
            return

        async with self._lock:
            # Acumular cantidad ejecutada y comisiones
            order.filled_quantity += event.quantity
            order.average_fill_price = event.price
            order.fee += event.fee
            order.fee_currency = event.fee_currency or order.fee_currency
            order.exchange_order_id = event.exchange_order_id or order.exchange_order_id

            # Determinar si la orden esta completamente o parcialmente ejecutada
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED  # Totalmente ejecutada
            else:
                order.status = OrderStatus.PARTIALLY_FILLED  # Parcialmente ejecutada

            order.updated_at = int(time.time() * 1000)

        logger.info(
            "Fill processed for order %s: filled=%s/%s status=%s",
            event.client_order_id,
            order.filled_quantity,
            order.quantity,
            order.status.value,
        )

    # -- Sincronizacion Periodica --------------------------------------------

    async def _sync_loop(self) -> None:
        """Bucle de sincronizacion periodica de ordenes abiertas con el exchange.

        Se ejecuta continuamente mientras el gestor este activo, consultando
        el estado de las ordenes abiertas en intervalos regulares.
        """
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                # Verificar si se detuvo durante la espera
                if not self._running:
                    break
                await self._sync_open_orders()
            except asyncio.CancelledError:
                return  # Cancelacion limpia del bucle
            except Exception:
                logger.exception("Error during order sync loop")

    async def _sync_open_orders(self) -> None:
        """Obtiene el estado actual de todas las ordenes abiertas desde el exchange.

        Detecta ordenes que fueron ejecutadas, canceladas o expiradas
        externamente (ej. desde la interfaz web de Binance).
        """
        open_orders = self.get_open_orders()
        if not open_orders:
            return  # No hay ordenes abiertas para sincronizar

        logger.debug("Syncing %d open orders with exchange", len(open_orders))

        for order in open_orders:
            # Saltar ordenes sin ID del exchange (aun no confirmadas)
            if not order.exchange_order_id:
                continue

            try:
                # Consultar el estado actual en el exchange
                result = await self._executor.fetch_order_status(
                    order.exchange_order_id, order.symbol
                )
                # Aplicar los cambios de estado detectados
                await self._apply_exchange_status(order, result)

            except Exception as exc:
                logger.warning(
                    "Failed to sync order %s: %s",
                    order.client_order_id,
                    exc,
                )

    async def _apply_exchange_status(
        self, order: TrackedOrder, exchange_result: dict[str, Any]
    ) -> None:
        """Mapea el estado del exchange a nuestro OrderStatus interno.

        Traduce la respuesta del exchange al modelo de estados interno,
        detecta ejecuciones parciales y publica FillEvents para ordenes
        ejecutadas externamente.

        Parametros
        ----------
        order :
            Orden rastreada a actualizar.
        exchange_result :
            Respuesta del exchange con el estado actual de la orden.
        """
        raw_status = (exchange_result.get("status") or "").lower()

        # Mapeo de estados del exchange a estados internos
        status_map = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,  # Variante de escritura britanica
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        new_status = status_map.get(raw_status)

        if new_status is None:
            return  # Estado desconocido, no hacer nada

        filled = exchange_result.get("filled")
        amount = exchange_result.get("amount")

        # Detectar ejecuciones parciales: si esta "abierta" pero tiene cantidad ejecutada > 0
        if (
            new_status == OrderStatus.OPEN
            and filled is not None
            and float(filled) > 0
        ):
            new_status = OrderStatus.PARTIALLY_FILLED

        async with self._lock:
            # Registrar cambio de estado si es diferente al actual
            if order.status != new_status:
                logger.info(
                    "Order %s status changed: %s -> %s (via exchange sync)",
                    order.client_order_id,
                    order.status.value,
                    new_status.value,
                )
                order.status = new_status

            # Actualizar cantidad ejecutada si esta disponible
            if filled is not None:
                order.filled_quantity = Decimal(str(filled))
            # Actualizar precio promedio de ejecucion si esta disponible
            avg = exchange_result.get("average")
            if avg is not None:
                order.average_fill_price = Decimal(str(avg))

            # Actualizar informacion de comisiones si esta disponible
            fee_info = exchange_result.get("fee") or {}
            if fee_info.get("cost") is not None:
                order.fee = Decimal(str(fee_info["cost"]))
            if fee_info.get("currency"):
                order.fee_currency = fee_info["currency"]

            order.updated_at = int(time.time() * 1000)

        # Si la orden fue ejecutada externamente, publicar un FillEvent
        if new_status == OrderStatus.FILLED and order.filled_quantity > 0:
            fill_event = FillEvent(
                symbol=order.symbol,
                side=order.side,
                quantity=order.filled_quantity,
                price=order.average_fill_price,
                fee=order.fee,
                fee_currency=order.fee_currency,
                exchange_order_id=order.exchange_order_id,
                client_order_id=order.client_order_id,
                strategy_name=order.strategy_name,
            )
            await self._event_bus.publish(fill_event)

    # -- Diagnosticos --------------------------------------------------------

    @property
    def total_tracked(self) -> int:
        """Numero total de ordenes rastreadas historicamente.

        Retorna
        -------
        int
            Cantidad total de ordenes que han sido rastreadas (incluyendo terminadas).
        """
        return len(self._orders)

    @property
    def open_count(self) -> int:
        """Numero de ordenes actualmente abiertas.

        Retorna
        -------
        int
            Cantidad de ordenes que aun no estan en estado terminal.
        """
        return sum(1 for o in self._orders.values() if o.is_open)

    @property
    def is_running(self) -> bool:
        """Indica si el gestor de ordenes esta activo y sincronizando."""
        return self._running
