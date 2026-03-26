"""Interruptor de circuito: detiene el trading cuando las perdidas superan los umbrales de seguridad.

Monitorea:
    - P&L diario (max_daily_loss_pct, por defecto 3%)
    - Caida desde el maximo (drawdown) pico-a-valle (max_drawdown_pct, por defecto 10%)

Cuando se supera cualquiera de los umbrales, el interruptor se **activa** (trip)
y todas las senales nuevas se bloquean hasta que se haga un llamado manual a ``reset()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from src.core.config_loader import RiskConfig
from src.core.event_bus import EventBus
from src.core.events import Event, FillEvent, TickEvent

logger = logging.getLogger(__name__)


# -- Evento de Alerta -------------------------------------------------------


class AlertSeverity(str, Enum):
    """Niveles de severidad para las alertas de riesgo."""
    WARNING = "warning"  # Advertencia: acercandose al limite
    CRITICAL = "critical"  # Critico: limite superado, interruptor activado


@dataclass(frozen=True, slots=True)
class RiskAlertEvent(Event):
    """Evento publicado cuando el interruptor de circuito se activa o se acerca a un limite.

    Atributos
    ---------
    severity : AlertSeverity
        Nivel de severidad de la alerta (advertencia o critico).
    rule : str
        Nombre de la regla que genero la alerta (ej. 'max_daily_loss', 'max_drawdown').
    message : str
        Mensaje descriptivo de la alerta.
    current_value : Decimal
        Valor actual de la metrica monitoreada.
    threshold : Decimal
        Umbral configurado que fue superado.
    """

    severity: AlertSeverity = AlertSeverity.WARNING
    rule: str = ""
    message: str = ""
    current_value: Decimal = Decimal(0)
    threshold: Decimal = Decimal(0)


# -- Interruptor de Circuito ------------------------------------------------


class CircuitBreaker:
    """Monitorea el P&L diario y el drawdown; se activa cuando se exceden los limites.

    Actua como un mecanismo de seguridad que detiene automaticamente toda
    actividad de trading cuando las perdidas superan los umbrales configurados.

    Parametros
    ----------
    event_bus : EventBus
        Bus de eventos para suscripcion y publicacion de alertas.
    config : RiskConfig
        Configuracion de riesgo con los umbrales de perdida y drawdown.
    initial_equity : Decimal
        Capital de la cuenta al inicio del dia (o sesion) de trading.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: RiskConfig,
        initial_equity: Decimal,
    ) -> None:
        self._bus = event_bus
        self._max_daily_loss_pct = config.max_daily_loss_pct  # Porcentaje maximo de perdida diaria
        self._max_drawdown_pct = config.max_drawdown_pct  # Porcentaje maximo de drawdown permitido

        # Seguimiento de capital (equity)
        self._initial_equity = initial_equity  # Capital al inicio del dia
        self._peak_equity = initial_equity  # Capital maximo alcanzado (para calcular drawdown)
        self._current_equity = initial_equity  # Capital actual en tiempo real

        # Acumulador de P&L realizado del dia
        self._daily_realized_pnl: Decimal = Decimal(0)

        # Estado del interruptor
        self._tripped = False  # Indica si el interruptor esta activado
        self._trip_reason: str = ""  # Razon por la cual se activo

        # Suscribirse a ejecuciones para rastrear el P&L realizado
        self._bus.subscribe(FillEvent, self._on_fill, name="CircuitBreaker.fill")

    # -- API Publica --------------------------------------------------------

    @property
    def is_tripped(self) -> bool:
        """Retorna True si el interruptor esta actualmente activado (trading detenido)."""
        return self._tripped

    @property
    def trip_reason(self) -> str:
        """Retorna la razon por la cual el interruptor fue activado."""
        return self._trip_reason

    def check(self) -> bool:
        """Evalua todas las condiciones del interruptor de circuito.

        Verifica la perdida diaria y el drawdown contra los umbrales configurados.

        Retorna
        -------
        bool
            True si el interruptor esta activado (el trading debe detenerse).
        """
        if self._tripped:
            return True

        # 1. Verificacion de perdida diaria
        if self._initial_equity > Decimal(0):
            # Calcular el ratio de perdida diaria respecto al capital inicial
            daily_loss_ratio = -self._daily_realized_pnl / self._initial_equity
            if daily_loss_ratio >= self._max_daily_loss_pct:
                self._trip(
                    rule="max_daily_loss",
                    message=(
                        f"Daily loss {daily_loss_ratio:.2%} exceeds limit "
                        f"{self._max_daily_loss_pct:.2%}"
                    ),
                    current_value=daily_loss_ratio,
                    threshold=self._max_daily_loss_pct,
                )
                return True

        # 2. Verificacion de drawdown (caida desde el maximo)
        if self._peak_equity > Decimal(0):
            # Calcular drawdown como porcentaje de caida desde el pico maximo
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
            if drawdown >= self._max_drawdown_pct:
                self._trip(
                    rule="max_drawdown",
                    message=(
                        f"Drawdown {drawdown:.2%} exceeds limit "
                        f"{self._max_drawdown_pct:.2%}"
                    ),
                    current_value=drawdown,
                    threshold=self._max_drawdown_pct,
                )
                return True

        return False

    def update_equity(self, equity: Decimal) -> None:
        """Actualiza el capital actual y el pico maximo para el seguimiento del drawdown.

        Debe llamarse despues de cada ciclo de valoracion a mercado (mark-to-market).

        Parametros
        ----------
        equity : Decimal
            Capital total actual de la cuenta.
        """
        self._current_equity = equity
        # Actualizar el pico maximo si el capital actual lo supera
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Re-evaluar las condiciones despues de la actualizacion
        self.check()

    def record_realized_pnl(self, pnl: Decimal) -> None:
        """Acumula P&L realizado para el dia actual.

        Parametros
        ----------
        pnl : Decimal
            P&L realizado de la operacion (positivo = ganancia, negativo = perdida).
        """
        self._daily_realized_pnl += pnl
        self.check()

    def reset(self) -> None:
        """Reinicio manual: rearma el interruptor de circuito.

        Tipicamente se invoca al inicio de un nuevo dia de trading o por un operador.
        Reinicia el estado de activacion, la razon, el P&L diario y el pico de capital.
        """
        logger.info(
            "Circuit breaker RESET (was tripped=%s, reason='%s')",
            self._tripped, self._trip_reason,
        )
        self._tripped = False  # Desactivar el interruptor
        self._trip_reason = ""  # Limpiar la razon de activacion
        self._daily_realized_pnl = Decimal(0)  # Reiniciar el P&L diario acumulado
        self._peak_equity = self._current_equity  # Establecer nuevo pico desde el capital actual

    def reset_daily(self, equity: Decimal) -> None:
        """Reinicia los contadores diarios para un nuevo dia de trading.

        NO desactiva el interruptor si esta activado. Para eso, usar ``reset()``.

        Parametros
        ----------
        equity : Decimal
            Capital al inicio del nuevo dia de trading.
        """
        self._daily_realized_pnl = Decimal(0)  # Reiniciar acumulador diario
        self._initial_equity = equity  # Nuevo capital de referencia para el dia
        self._peak_equity = equity  # Reiniciar pico de capital
        self._current_equity = equity  # Establecer capital actual
        logger.info("Circuit breaker daily counters reset, equity=%s", equity)

    # -- Metodos Internos ---------------------------------------------------

    def _trip(
        self,
        rule: str,
        message: str,
        current_value: Decimal,
        threshold: Decimal,
    ) -> None:
        """Activa el interruptor y publica un evento de alerta.

        Parametros
        ----------
        rule : str
            Nombre de la regla violada.
        message : str
            Mensaje descriptivo del motivo de activacion.
        current_value : Decimal
            Valor actual de la metrica que provoco la activacion.
        threshold : Decimal
            Umbral que fue superado.
        """
        if self._tripped:
            return  # Ya esta activado, evitar alertas duplicadas

        self._tripped = True
        self._trip_reason = message
        logger.critical("CIRCUIT BREAKER TRIPPED: %s", message)

        # Construir el evento de alerta critica
        alert = RiskAlertEvent(
            severity=AlertSeverity.CRITICAL,
            rule=rule,
            message=message,
            current_value=current_value,
            threshold=threshold,
        )
        # La publicacion es asincrona; programarla como tarea independiente (fire-and-forget)
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._bus.publish(alert))
        except RuntimeError:
            # No hay loop de eventos activo (ej. en tests); solo registrar en log
            logger.warning("No event loop available to publish RiskAlertEvent")

    # -- Manejadores de Eventos ---------------------------------------------

    async def _on_fill(self, event: FillEvent) -> None:
        """Rastrea el P&L realizado a partir de ejecuciones (fills).

        Por simplicidad, el interruptor de circuito solo usa la comision (fee)
        como costo garantizado. El P&L realizado completo se alimenta a traves
        de ``record_realized_pnl`` por el PortfolioTracker despues de calcular
        el impacto de la ejecucion en el P&L.

        Parametros
        ----------
        event : FillEvent
            Evento de ejecucion con informacion de la comision.
        """
        # Deducir comisiones como costo realizado
        if event.fee > Decimal(0):
            self.record_realized_pnl(-event.fee)
