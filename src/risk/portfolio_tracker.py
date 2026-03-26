"""Rastreador de portafolio: posiciones abiertas, P&L realizado / no realizado.

Escucha eventos FillEvent (abrir/cerrar posiciones) y TickEvent (valoracion a mercado).
Todos los valores monetarios utilizan Decimal para precision financiera.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict

from src.core.event_bus import EventBus
from src.core.events import FillEvent, Side, TickEvent

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Representa una posicion abierta individual en el portafolio.

    Almacena el simbolo del activo, la direccion (compra/venta), cantidad,
    precio de entrada y calcula el P&L no realizado al actualizar precios.
    """

    symbol: str  # Simbolo del par de trading (ej. BTC/USDT)
    side: Side  # Direccion de la posicion: BUY (compra) o SELL (venta)
    qty: Decimal  # Cantidad de unidades del activo base
    entry_price: Decimal  # Precio promedio de entrada
    current_price: Decimal = Decimal(0)  # Ultimo precio de mercado conocido
    unrealized_pnl: Decimal = Decimal(0)  # Ganancia/perdida no realizada actual

    def mark_to_market(self, price: Decimal) -> None:
        """Recalcula el P&L no realizado al precio de mercado dado.

        Parametros
        ----------
        price : Decimal
            Precio actual de mercado para valorar la posicion.
        """
        self.current_price = price
        # Para posiciones largas (BUY): ganancia si el precio sube
        if self.side == Side.BUY:
            self.unrealized_pnl = (price - self.entry_price) * self.qty
        # Para posiciones cortas (SELL): ganancia si el precio baja
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.qty


class PortfolioTracker:
    """Rastrea todas las posiciones abiertas y el P&L realizado acumulado.

    Se suscribe automaticamente a FillEvent y TickEvent a traves del EventBus
    para mantener el estado del portafolio actualizado en tiempo real.

    Uso:
        tracker = PortfolioTracker(event_bus, initial_equity=Decimal("10000"))
        # El EventBus alimenta FillEvents y TickEvents automaticamente.
    """

    def __init__(self, event_bus: EventBus, initial_equity: Decimal = Decimal(0)) -> None:
        self._event_bus = event_bus
        self._initial_equity = initial_equity  # Capital inicial de la cuenta
        self._positions: Dict[str, Position] = {}  # Posiciones abiertas indexadas por simbolo
        self._realized_pnl: Decimal = Decimal(0)  # P&L realizado acumulado total

        # Suscribirse a los eventos relevantes del bus de eventos
        self._event_bus.subscribe(FillEvent, self._on_fill, name="PortfolioTracker.fill")
        self._event_bus.subscribe(TickEvent, self._on_tick, name="PortfolioTracker.tick")

    # -- API Publica --------------------------------------------------------

    def add_position(self, symbol: str, side: Side, qty: Decimal, entry_price: Decimal) -> Position:
        """Abre (o agrega a) una posicion para el simbolo dado.

        Si ya existe una posicion en la misma direccion, promedia el precio de entrada.
        Si existe en direccion opuesta, cierra parcial o totalmente y abre el remanente.

        Parametros
        ----------
        symbol : str
            Simbolo del par de trading.
        side : Side
            Direccion de la operacion (BUY o SELL).
        qty : Decimal
            Cantidad de unidades a operar.
        entry_price : Decimal
            Precio de entrada de la operacion.

        Retorna
        -------
        Position
            La posicion resultante (nueva, promediada o remanente).
        """
        if symbol in self._positions:
            existing = self._positions[symbol]
            if existing.side != side:
                # Direccion opuesta: cierre parcial o total, luego abrir remanente
                return self._net_position(existing, side, qty, entry_price)
            # Misma direccion: promediar precio de entrada (average-in)
            total_qty = existing.qty + qty
            avg_price = (
                (existing.entry_price * existing.qty + entry_price * qty) / total_qty
            )
            existing.qty = total_qty
            existing.entry_price = avg_price
            # Revaluar la posicion con el precio actual o el nuevo precio de entrada
            existing.mark_to_market(existing.current_price or entry_price)
            logger.info(
                "Averaged into %s position for %s: qty=%s avg_price=%s",
                side.value, symbol, total_qty, avg_price,
            )
            return existing

        # Crear nueva posicion desde cero
        position = Position(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            current_price=entry_price,
        )
        self._positions[symbol] = position
        logger.info(
            "Opened %s position for %s: qty=%s price=%s",
            side.value, symbol, qty, entry_price,
        )
        return position

    def close_position(self, symbol: str, exit_price: Decimal, qty: Decimal | None = None) -> Decimal:
        """Cierra (total o parcialmente) una posicion. Retorna el P&L realizado de la porcion cerrada.

        Parametros
        ----------
        symbol : str
            Simbolo del par cuya posicion se cierra.
        exit_price : Decimal
            Precio de salida/cierre.
        qty : Decimal | None
            Cantidad a cerrar. Si es None, cierra toda la posicion.

        Retorna
        -------
        Decimal
            P&L realizado de la porcion cerrada.
        """
        if symbol not in self._positions:
            logger.warning("Attempted to close non-existent position for %s", symbol)
            return Decimal(0)

        position = self._positions[symbol]
        # Si no se especifica cantidad, cerrar toda la posicion
        close_qty = qty if qty is not None else position.qty

        # Limitar la cantidad de cierre a la cantidad disponible
        if close_qty > position.qty:
            close_qty = position.qty

        # Calcular P&L segun la direccion de la posicion
        if position.side == Side.BUY:
            pnl = (exit_price - position.entry_price) * close_qty
        else:
            pnl = (position.entry_price - exit_price) * close_qty

        # Acumular el P&L realizado
        self._realized_pnl += pnl

        # Calcular cantidad restante despues del cierre
        remaining = position.qty - close_qty
        if remaining <= Decimal(0):
            # Cierre total: eliminar la posicion del diccionario
            del self._positions[symbol]
            logger.info(
                "Closed %s position for %s: pnl=%s",
                position.side.value, symbol, pnl,
            )
        else:
            # Cierre parcial: actualizar cantidad restante y revaluar
            position.qty = remaining
            position.mark_to_market(exit_price)
            logger.info(
                "Partially closed %s position for %s: closed_qty=%s remaining=%s pnl=%s",
                position.side.value, symbol, close_qty, remaining, pnl,
            )

        return pnl

    def update_prices(self, symbol: str, price: Decimal) -> None:
        """Actualiza la valoracion a mercado (mark-to-market) de una posicion individual.

        Parametros
        ----------
        symbol : str
            Simbolo del par a actualizar.
        price : Decimal
            Nuevo precio de mercado.
        """
        if symbol in self._positions:
            self._positions[symbol].mark_to_market(price)

    def get_total_exposure(self) -> Decimal:
        """Calcula la exposicion total: suma de (cantidad * precio_actual) de todas las posiciones abiertas.

        Retorna
        -------
        Decimal
            Valor total de exposicion en la moneda de cotizacion.
        """
        return sum(
            (pos.qty * pos.current_price for pos in self._positions.values()),
            Decimal(0),
        )

    def get_total_equity(self) -> Decimal:
        """Calcula el capital total: capital_inicial + P&L_realizado + P&L_no_realizado.

        Retorna
        -------
        Decimal
            Capital total actual de la cuenta.
        """
        # Sumar todo el P&L no realizado de las posiciones abiertas
        total_unrealized = sum(
            (pos.unrealized_pnl for pos in self._positions.values()),
            Decimal(0),
        )
        return self._initial_equity + self._realized_pnl + total_unrealized

    @property
    def positions(self) -> Dict[str, Position]:
        """Retorna una copia del diccionario de posiciones abiertas."""
        return dict(self._positions)

    @property
    def open_position_count(self) -> int:
        """Retorna la cantidad de posiciones abiertas actualmente."""
        return len(self._positions)

    @property
    def realized_pnl(self) -> Decimal:
        """Retorna el P&L realizado acumulado total."""
        return self._realized_pnl

    def has_position(self, symbol: str) -> bool:
        """Verifica si existe una posicion abierta para el simbolo dado."""
        return symbol in self._positions

    def get_position(self, symbol: str) -> Position | None:
        """Obtiene la posicion abierta para el simbolo dado, o None si no existe."""
        return self._positions.get(symbol)

    # -- Metodos Privados Auxiliares ----------------------------------------

    def _net_position(
        self, existing: Position, new_side: Side, new_qty: Decimal, new_price: Decimal,
    ) -> Position:
        """Gestiona una ejecucion en la direccion opuesta a una posicion existente.

        Puede resultar en cierre total, cierre parcial o cambio de direccion (flip).

        Parametros
        ----------
        existing : Position
            La posicion existente que se va a compensar.
        new_side : Side
            Direccion de la nueva operacion.
        new_qty : Decimal
            Cantidad de la nueva operacion.
        new_price : Decimal
            Precio de la nueva operacion.

        Retorna
        -------
        Position
            La posicion resultante tras la compensacion.
        """
        if new_qty >= existing.qty:
            # Cierre total + posible cambio de direccion (flip)
            pnl = self.close_position(existing.symbol, new_price)
            remainder = new_qty - existing.qty  # Cantidad sobrante para nueva posicion
            if remainder > Decimal(0):
                # Abrir nueva posicion con el remanente en la direccion opuesta
                return self.add_position(existing.symbol, new_side, remainder, new_price)
            # Posicion cerrada exactamente: retornar centinela con cantidad cero
            return Position(
                symbol=existing.symbol, side=new_side, qty=Decimal(0),
                entry_price=new_price, current_price=new_price,
            )
        else:
            # Cierre parcial: solo reduce la posicion existente
            self.close_position(existing.symbol, new_price, qty=new_qty)
            return self._positions[existing.symbol]

    # -- Manejadores de Eventos ---------------------------------------------

    async def _on_fill(self, event: FillEvent) -> None:
        """Procesa una ejecucion (fill) recibida del exchange.

        Determina si la ejecucion cierra una posicion existente o abre una nueva.

        Parametros
        ----------
        event : FillEvent
            Evento de ejecucion con simbolo, direccion, cantidad y precio.
        """
        symbol = event.symbol
        side = event.side
        qty = event.quantity
        price = event.price

        # Verificar si la venta cierra una posicion larga existente
        if side == Side.SELL and self.has_position(symbol):
            position = self.get_position(symbol)
            if position is not None and position.side == Side.BUY:
                self.close_position(symbol, price, qty)
                return
        # Verificar si la compra cierra una posicion corta existente
        if side == Side.BUY and self.has_position(symbol):
            position = self.get_position(symbol)
            if position is not None and position.side == Side.SELL:
                self.close_position(symbol, price, qty)
                return

        # No hay posicion opuesta que cerrar: abrir nueva posicion
        self.add_position(symbol, side, qty, price)

    async def _on_tick(self, event: TickEvent) -> None:
        """Actualiza la valoracion a mercado en cada tick de precio recibido.

        Parametros
        ----------
        event : TickEvent
            Evento de tick con el ultimo precio del simbolo.
        """
        self.update_prices(event.symbol, event.last)
