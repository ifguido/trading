"""Repositorio CRUD asincrono para el almacenamiento de CryptoTrader.

Todos los metodos publicos adquieren su propia sesion desde la fabrica,
por lo que los llamadores nunca necesitan gestionar sesiones ni
transacciones directamente.

Este repositorio proporciona una interfaz de alto nivel para las
operaciones de base de datos: guardar/consultar operaciones (trades),
ordenes, registros de senales y resumen diario de PnL.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.events import FillEvent, OrderEvent, SignalEvent
from src.core.exceptions import StorageError

from .models import DailyPnL, Order, SignalLog, Trade

logger = logging.getLogger(__name__)


class Repository:
    """Interfaz CRUD asincrona de alto nivel sobre los modelos de almacenamiento.

    Encapsula todas las operaciones de base de datos necesarias para el
    funcionamiento del bot de trading. Cada metodo gestiona su propia
    sesion y transaccion, simplificando el uso desde las capas superiores.

    Parametros
    ----------
    session_factory:
        Una instancia de ``async_sessionmaker`` (creada por ``init_db``).
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory  # Fabrica de sesiones asincronas

    # ── Trades (Operaciones) ──────────────────────────────────────

    async def save_trade(self, fill: FillEvent) -> Trade:
        """Persiste una operacion a partir de un :class:`FillEvent`.

        Convierte el evento de ejecucion (fill) en un registro de la tabla
        ``trades`` y lo guarda en la base de datos.

        Parametros
        ----------
        fill : FillEvent
            Evento de ejecucion que contiene los detalles de la operacion.

        Retorno
        -------
        Trade
            La instancia ORM del trade guardado con su ID asignado.

        Lanza
        -----
        StorageError
            Si ocurre un error al guardar en la base de datos.
        """
        # Construir el objeto Trade a partir del evento de ejecucion.
        trade = Trade(
            exchange_order_id=fill.exchange_order_id,
            client_order_id=fill.client_order_id,
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=fill.quantity,
            price=fill.price,
            fee=fill.fee,
            fee_currency=fill.fee_currency,
            strategy_name=fill.strategy_name,
            # Convertir timestamp en milisegundos a datetime UTC.
            executed_at=datetime.fromtimestamp(fill.timestamp / 1000, tz=timezone.utc),
        )
        try:
            async with self._sf() as session:
                session.add(trade)
                await session.commit()
                await session.refresh(trade)  # Obtener el ID autogenerado y valores por defecto.
                logger.debug("Saved trade id=%s %s", trade.id, trade.symbol)
                return trade
        except Exception as exc:
            raise StorageError(f"Failed to save trade: {exc}") from exc

    async def get_trades_by_symbol(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Trade]:
        """Obtiene las operaciones para un simbolo, ordenadas por fecha de ejecucion descendente.

        Parametros
        ----------
        symbol : str
            Par de trading, por ejemplo ``"BTC/USDT"``.
        since : datetime | None
            Filtro opcional de fecha minima (UTC). Solo retorna operaciones
            ejecutadas desde esta fecha en adelante.
        limit : int
            Numero maximo de filas a retornar (por defecto 100).

        Retorno
        -------
        list[Trade]
            Lista de operaciones que cumplen los criterios de busqueda.

        Lanza
        -----
        StorageError
            Si ocurre un error al consultar la base de datos.
        """
        try:
            async with self._sf() as session:
                # Consulta base: filtrar por simbolo, ordenar por fecha descendente.
                stmt = (
                    select(Trade)
                    .where(Trade.symbol == symbol)
                    .order_by(Trade.executed_at.desc())
                    .limit(limit)
                )
                # Aplicar filtro de fecha minima si fue proporcionado.
                if since is not None:
                    stmt = stmt.where(Trade.executed_at >= since)

                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as exc:
            raise StorageError(f"Failed to fetch trades: {exc}") from exc

    # ── Orders (Ordenes) ──────────────────────────────────────────

    async def save_order(self, order_event: OrderEvent) -> Order:
        """Persiste una nueva orden a partir de un :class:`OrderEvent`.

        La orden se guarda con estado inicial ``"pending"`` (pendiente).

        Parametros
        ----------
        order_event : OrderEvent
            Evento de orden que contiene los detalles de la orden a crear.

        Retorno
        -------
        Order
            La instancia ORM de la orden guardada con su ID asignado.

        Lanza
        -----
        StorageError
            Si ocurre un error al guardar en la base de datos.
        """
        # Construir el objeto Order a partir del evento de orden.
        order = Order(
            client_order_id=order_event.client_order_id,
            symbol=order_event.symbol,
            side=order_event.side.value,
            order_type=order_event.order_type.value,
            quantity=order_event.quantity,
            price=order_event.price,
            stop_loss=order_event.stop_loss,
            take_profit=order_event.take_profit,
            status="pending",  # Estado inicial de toda orden nueva.
            strategy_name=order_event.strategy_name,
        )
        try:
            async with self._sf() as session:
                session.add(order)
                await session.commit()
                await session.refresh(order)  # Obtener el ID autogenerado.
                logger.debug(
                    "Saved order id=%s client_id=%s %s",
                    order.id,
                    order.client_order_id,
                    order.symbol,
                )
                return order
        except Exception as exc:
            raise StorageError(f"Failed to save order: {exc}") from exc

    async def update_order(
        self,
        client_order_id: str,
        *,
        status: str | None = None,
        exchange_order_id: str | None = None,
    ) -> None:
        """Actualiza el estado o el ID del exchange de una orden existente.

        Solo actualiza los campos proporcionados. Si no se pasa ningun
        campo para actualizar, el metodo retorna sin hacer nada.

        Parametros
        ----------
        client_order_id : str
            Identificador local de la orden (unico).
        status : str | None
            Nuevo estado de la orden (por ejemplo: ``"open"``, ``"filled"``, ``"cancelled"``).
        exchange_order_id : str | None
            Identificador de la orden asignado por el exchange.

        Lanza
        -----
        StorageError
            Si ocurre un error al actualizar en la base de datos.
        """
        # Construir el diccionario de valores a actualizar.
        values: dict = {}
        if status is not None:
            values["status"] = status
        if exchange_order_id is not None:
            values["exchange_order_id"] = exchange_order_id
        if not values:
            return  # No hay nada que actualizar.

        # Siempre actualizar la marca de tiempo de ultima modificacion.
        values["updated_at"] = datetime.now(timezone.utc)

        try:
            async with self._sf() as session:
                # Actualizar la orden identificada por su client_order_id.
                stmt = (
                    update(Order)
                    .where(Order.client_order_id == client_order_id)
                    .values(**values)
                )
                await session.execute(stmt)
                await session.commit()
                logger.debug("Updated order %s -> %s", client_order_id, values)
        except Exception as exc:
            raise StorageError(f"Failed to update order: {exc}") from exc

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Obtiene las ordenes que aun estan abiertas (estado: pending u open).

        Parametros
        ----------
        symbol : str | None
            Filtro opcional por par de trading. Si es ``None``, retorna
            todas las ordenes abiertas de cualquier simbolo.

        Retorno
        -------
        list[Order]
            Lista de ordenes abiertas, ordenadas por fecha de creacion descendente.

        Lanza
        -----
        StorageError
            Si ocurre un error al consultar la base de datos.
        """
        try:
            async with self._sf() as session:
                # Filtrar solo ordenes con estado "pending" u "open".
                stmt = (
                    select(Order)
                    .where(Order.status.in_(["pending", "open"]))
                    .order_by(Order.created_at.desc())
                )
                # Aplicar filtro por simbolo si fue proporcionado.
                if symbol is not None:
                    stmt = stmt.where(Order.symbol == symbol)

                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as exc:
            raise StorageError(f"Failed to fetch open orders: {exc}") from exc

    # ── Signal Logs (Registros de Senales) ────────────────────────

    async def save_signal_log(self, signal: SignalEvent) -> SignalLog:
        """Persiste un registro inmutable de senal a partir de un :class:`SignalEvent`.

        El diccionario ``metadata`` del evento se almacena como texto JSON
        en la columna ``metadata_json``.

        Parametros
        ----------
        signal : SignalEvent
            Evento de senal generado por una estrategia de trading.

        Retorno
        -------
        SignalLog
            La instancia ORM del registro de senal guardado.

        Lanza
        -----
        StorageError
            Si ocurre un error al guardar en la base de datos.
        """
        # Construir el objeto SignalLog a partir del evento de senal.
        log = SignalLog(
            event_id=signal.event_id,
            symbol=signal.symbol,
            direction=signal.direction.value,
            strategy_name=signal.strategy_name,
            confidence=Decimal(str(signal.confidence)),  # Convertir float a Decimal para precision.
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            # Serializar metadatos a JSON si existen.
            metadata_json=json.dumps(signal.metadata) if signal.metadata else None,
        )
        try:
            async with self._sf() as session:
                session.add(log)
                await session.commit()
                await session.refresh(log)  # Obtener el ID autogenerado.
                logger.debug("Saved signal log id=%s %s", log.id, log.symbol)
                return log
        except Exception as exc:
            raise StorageError(f"Failed to save signal log: {exc}") from exc

    # ── All Trades (para la web UI) ───────────────────────────────

    async def get_all_trades(self, *, limit: int = 100) -> list[Trade]:
        """Obtiene las operaciones recientes de todos los simbolos."""
        try:
            async with self._sf() as session:
                stmt = (
                    select(Trade)
                    .order_by(Trade.executed_at.desc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as exc:
            raise StorageError(f"Failed to fetch all trades: {exc}") from exc

    # ── Recent Signals (para la web UI) ─────────────────────────

    async def get_recent_signals(
        self, *, symbol: str | None = None, limit: int = 100
    ) -> list[SignalLog]:
        """Obtiene los signal logs recientes."""
        try:
            async with self._sf() as session:
                stmt = (
                    select(SignalLog)
                    .order_by(SignalLog.created_at.desc())
                    .limit(limit)
                )
                if symbol is not None:
                    stmt = stmt.where(SignalLog.symbol == symbol)
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as exc:
            raise StorageError(f"Failed to fetch recent signals: {exc}") from exc

    # ── Daily PnL (Ganancias y Perdidas Diarias) ─────────────────

    async def save_daily_pnl(
        self,
        trade_date: date,
        symbol: str,
        *,
        realized_pnl: Decimal = Decimal(0),
        unrealized_pnl: Decimal = Decimal(0),
        fees: Decimal = Decimal(0),
        trade_count: int = 0,
    ) -> DailyPnL:
        """Inserta o actualiza el registro de PnL diario para una fecha y simbolo dados.

        Utiliza un patron "buscar-luego-insertar/actualizar" (fetch-then-upsert)
        que es seguro para la arquitectura de escritor unico de CryptoTrader.

        Parametros
        ----------
        trade_date : date
            Fecha del dia del resumen.
        symbol : str
            Par de trading o ``"PORTFOLIO"`` para el agregado general.
        realized_pnl : Decimal
            Ganancias/perdidas realizadas del dia.
        unrealized_pnl : Decimal
            Ganancias/perdidas no realizadas al cierre del dia.
        fees : Decimal
            Comisiones totales pagadas en el dia.
        trade_count : int
            Cantidad de operaciones del dia.

        Retorno
        -------
        DailyPnL
            La instancia ORM del registro de PnL diario (insertado o actualizado).

        Lanza
        -----
        StorageError
            Si ocurre un error al guardar en la base de datos.
        """
        try:
            async with self._sf() as session:
                # Buscar si ya existe un registro para esta fecha y simbolo.
                stmt = select(DailyPnL).where(
                    DailyPnL.trade_date == trade_date,
                    DailyPnL.symbol == symbol,
                )
                result = await session.execute(stmt)
                row = result.scalar_one_or_none()

                if row is not None:
                    # El registro ya existe: actualizar los valores.
                    row.realized_pnl = realized_pnl
                    row.unrealized_pnl = unrealized_pnl
                    row.fees = fees
                    row.trade_count = trade_count
                else:
                    # No existe: crear un nuevo registro.
                    row = DailyPnL(
                        trade_date=trade_date,
                        symbol=symbol,
                        realized_pnl=realized_pnl,
                        unrealized_pnl=unrealized_pnl,
                        fees=fees,
                        trade_count=trade_count,
                    )
                    session.add(row)

                await session.commit()
                await session.refresh(row)  # Obtener valores actualizados de la BD.
                logger.debug(
                    "Saved daily PnL id=%s %s %s realized=%s",
                    row.id,
                    row.trade_date,
                    row.symbol,
                    row.realized_pnl,
                )
                return row
        except Exception as exc:
            raise StorageError(f"Failed to save daily PnL: {exc}") from exc

    async def get_daily_pnl(
        self,
        *,
        symbol: str | None = None,
        since: date | None = None,
        until: date | None = None,
    ) -> list[DailyPnL]:
        """Obtiene los registros de PnL diario con filtros opcionales de rango de fechas y simbolo.

        Los resultados se ordenan por ``trade_date`` de forma ascendente.

        Parametros
        ----------
        symbol : str | None
            Filtro opcional por par de trading o ``"PORTFOLIO"``.
        since : date | None
            Fecha minima (inclusive) del rango de consulta.
        until : date | None
            Fecha maxima (inclusive) del rango de consulta.

        Retorno
        -------
        list[DailyPnL]
            Lista de registros de PnL diario que cumplen los filtros.

        Lanza
        -----
        StorageError
            Si ocurre un error al consultar la base de datos.
        """
        try:
            async with self._sf() as session:
                # Consulta base ordenada por fecha ascendente.
                stmt = select(DailyPnL).order_by(DailyPnL.trade_date.asc())

                # Aplicar filtros opcionales.
                if symbol is not None:
                    stmt = stmt.where(DailyPnL.symbol == symbol)
                if since is not None:
                    stmt = stmt.where(DailyPnL.trade_date >= since)
                if until is not None:
                    stmt = stmt.where(DailyPnL.trade_date <= until)

                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception as exc:
            raise StorageError(f"Failed to fetch daily PnL: {exc}") from exc
