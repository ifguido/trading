"""Puerta central de riesgo.

Se suscribe a SignalEvents en el EventBus. Para cada senal, el gestor:
1. Valida contra todas las reglas de riesgo.
2. Si se aprueba, calcula el tamano via PositionSizer y publica un OrderEvent.
3. Si se rechaza, registra la razon en el log (sin excepcion, simplemente descarta la senal).
"""

from __future__ import annotations

import logging
from decimal import Decimal

from src.core.config_loader import RiskConfig
from src.core.event_bus import EventBus
from src.core.events import (
    OrderEvent,
    OrderType,
    Side,
    SignalDirection,
    SignalEvent,
)

from .circuit_breaker import CircuitBreaker
from .portfolio_tracker import PortfolioTracker
from .position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class RiskManager:
    """Evalua cada senal contra las reglas de riesgo antes de convertirla en orden.

    Actua como el filtro final entre las estrategias de trading y la ejecucion,
    asegurando que ninguna operacion viole los limites de riesgo configurados.

    Parametros
    ----------
    event_bus : EventBus
        Bus de eventos para suscripcion y publicacion.
    config : RiskConfig
        Configuracion de riesgo con limites y reglas.
    portfolio : PortfolioTracker
        Rastreador de portafolio para consultar posiciones y capital.
    sizer : PositionSizer
        Motor de dimensionamiento de posiciones.
    circuit_breaker : CircuitBreaker
        Interruptor de circuito para verificar si el trading esta detenido.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: RiskConfig,
        portfolio: PortfolioTracker,
        sizer: PositionSizer,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        self._bus = event_bus  # Bus de eventos para comunicacion
        self._cfg = config  # Configuracion de reglas de riesgo
        self._portfolio = portfolio  # Rastreador de portafolio
        self._sizer = sizer  # Calculador de tamano de posicion
        self._cb = circuit_breaker  # Interruptor de circuito

        # Suscribirse a las senales de las estrategias
        self._bus.subscribe(SignalEvent, self._on_signal, name="RiskManager.signal")

    # -- Manejador de Eventos -----------------------------------------------

    async def _on_signal(self, event: SignalEvent) -> None:
        """Filtra un SignalEvent a traves de todas las verificaciones de riesgo.

        Procesa la senal segun su direccion: HOLD se ignora, CLOSE genera
        una orden de cierre, y LONG/SHORT pasan por validacion y dimensionamiento.

        Parametros
        ----------
        event : SignalEvent
            Senal generada por una estrategia de trading.
        """
        symbol = event.symbol
        direction = event.direction

        # Las senales HOLD (mantener) se ignoran, no requieren accion
        if direction == SignalDirection.HOLD:
            return

        # Las senales CLOSE (cerrar) se procesan directamente sin dimensionamiento
        if direction == SignalDirection.CLOSE:
            await self._handle_close(event)
            return

        # -- Verificaciones de riesgo pre-operacion -------------------------
        rejection = self._validate(event)
        if rejection is not None:
            logger.warning(
                "Signal REJECTED for %s (%s): %s",
                symbol, direction.value, rejection,
            )
            return

        # -- Dimensionar la orden -------------------------------------------
        equity = self._portfolio.get_total_equity()  # Capital total actual
        entry_price = self._estimate_entry_price(event)  # Precio estimado de entrada
        stop_distance = self._stop_distance(event, entry_price)  # Distancia al stop-loss

        # Verificar que la distancia al stop sea valida
        if stop_distance <= Decimal(0):
            logger.warning(
                "Signal REJECTED for %s: could not compute valid stop distance",
                symbol,
            )
            return

        # Calcular la cantidad usando el motor de dimensionamiento
        qty = self._sizer.calculate(
            equity=equity,
            risk_per_trade=self._cfg.max_daily_loss_pct,  # Presupuesto de riesgo por operacion
            entry_price=entry_price,
            stop_distance=stop_distance,
        )

        # Rechazar si la cantidad calculada es cero o negativa
        if qty <= Decimal(0):
            logger.warning("Signal REJECTED for %s: calculated qty is zero", symbol)
            return

        # -- Construir y publicar el OrderEvent -----------------------------
        # Convertir la direccion de la senal al lado correspondiente de la orden
        side = Side.BUY if direction == SignalDirection.LONG else Side.SELL
        order = OrderEvent(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
            stop_loss=event.stop_loss,
            take_profit=event.take_profit,
            strategy_name=event.strategy_name,
        )

        logger.info(
            "Signal APPROVED for %s: side=%s qty=%s stop=%s tp=%s",
            symbol, side.value, qty, event.stop_loss, event.take_profit,
        )
        await self._bus.publish(order)

    # -- Validacion ---------------------------------------------------------

    def _validate(self, event: SignalEvent) -> str | None:
        """Ejecuta todas las verificaciones de riesgo pre-operacion.

        Verifica en orden: interruptor de circuito, stop-loss obligatorio,
        posiciones concurrentes maximas y limite de exposicion total.

        Parametros
        ----------
        event : SignalEvent
            Senal a validar contra las reglas de riesgo.

        Retorna
        -------
        str | None
            Razon de rechazo como cadena de texto, o None si se aprueba.
        """
        # 1. Verificar interruptor de circuito
        if self._cb.is_tripped:
            return "circuit breaker is tripped"

        # 2. Stop-loss obligatorio: rechazar si no se proporciona y es requerido
        if self._cfg.mandatory_stop_loss and event.stop_loss is None:
            return "stop-loss is mandatory but not provided"

        # 3. Limite de posiciones concurrentes: rechazar si ya se alcanzo el maximo
        if (
            not self._portfolio.has_position(event.symbol)
            and self._portfolio.open_position_count >= self._cfg.max_concurrent_positions
        ):
            return (
                f"max concurrent positions reached "
                f"({self._portfolio.open_position_count}/{self._cfg.max_concurrent_positions})"
            )

        # 4. Limite de exposicion total: rechazar si la exposicion supera el maximo permitido
        equity = self._portfolio.get_total_equity()
        if equity > Decimal(0):
            exposure_ratio = self._portfolio.get_total_exposure() / equity
            if exposure_ratio >= self._cfg.max_total_exposure_pct:
                return (
                    f"total exposure {exposure_ratio:.2%} exceeds limit "
                    f"{self._cfg.max_total_exposure_pct:.2%}"
                )

        return None  # Todas las verificaciones pasaron: senal aprobada

    # -- Metodos Auxiliares -------------------------------------------------

    async def _handle_close(self, event: SignalEvent) -> None:
        """Publica un OrderEvent de cierre para una posicion existente.

        Determina el lado opuesto de la posicion actual y genera una orden
        de mercado para cerrar toda la posicion.

        Parametros
        ----------
        event : SignalEvent
            Senal de cierre con el simbolo de la posicion a cerrar.
        """
        position = self._portfolio.get_position(event.symbol)
        if position is None:
            logger.debug("CLOSE signal for %s but no open position", event.symbol)
            return

        # Determinar el lado opuesto para cerrar la posicion
        close_side = Side.SELL if position.side == Side.BUY else Side.BUY
        order = OrderEvent(
            symbol=event.symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=position.qty,
            strategy_name=event.strategy_name,
        )
        logger.info(
            "CLOSE order for %s: side=%s qty=%s",
            event.symbol, close_side.value, position.qty,
        )
        await self._bus.publish(order)

    @staticmethod
    def _estimate_entry_price(event: SignalEvent) -> Decimal:
        """Deriva un precio de entrada aproximado desde los metadatos de la senal.

        Las estrategias deben incluir ``entry_price`` en los metadatos; de lo
        contrario, se usa el stop_loss como respaldo, o cero si no hay datos.

        Parametros
        ----------
        event : SignalEvent
            Senal con metadatos que pueden contener el precio de entrada.

        Retorna
        -------
        Decimal
            Precio de entrada estimado.
        """
        # Prioridad 1: precio de entrada explicito en los metadatos
        if "entry_price" in event.metadata:
            return Decimal(str(event.metadata["entry_price"]))
        # Prioridad 2: si solo hay stop_loss disponible, asumir entrada cercana al stop
        if event.stop_loss is not None:
            return event.stop_loss
        # Sin datos disponibles: retornar cero (la orden sera rechazada por el sizer)
        return Decimal(0)

    @staticmethod
    def _stop_distance(event: SignalEvent, entry_price: Decimal) -> Decimal:
        """Calcula la distancia absoluta en precio desde la entrada hasta el stop-loss.

        Parametros
        ----------
        event : SignalEvent
            Senal que contiene el nivel de stop-loss.
        entry_price : Decimal
            Precio de entrada estimado.

        Retorna
        -------
        Decimal
            Distancia absoluta al stop-loss. Retorna 0 si no se puede calcular.
        """
        # Sin stop-loss o sin precio de entrada valido, no se puede calcular la distancia
        if event.stop_loss is None or entry_price <= Decimal(0):
            return Decimal(0)
        return abs(entry_price - event.stop_loss)
