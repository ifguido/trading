"""Servicio de notificaciones por Telegram para alertas de trading y salud del sistema.

Envia mensajes a traves de la API asincrona de python-telegram-bot v20+.
Si el token del bot no esta configurado, opera de forma silenciosa (no-op)
sin generar errores, permitiendo que el sistema funcione sin Telegram.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import FillEvent

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Envia alertas a un chat de Telegram.

    Se suscribe al evento ``FillEvent`` en el ``EventBus`` para enviar
    notificaciones automaticas de cada operacion ejecutada. Tambien
    proporciona metodos explicitos para notificaciones de circuit-breaker
    y verificacion de salud del sistema.

    Si ``bot_token`` o ``chat_id`` estan vacios, el notificador opera en
    modo no-op: todos los metodos retornan inmediatamente sin error.

    Parametros
    ----------
    event_bus : EventBus
        Bus de eventos del sistema para suscribirse a los fills.
    bot_token : str
        Token del bot de Telegram (obtenido de @BotFather).
    chat_id : str
        Identificador del chat de Telegram donde enviar las notificaciones.
    """

    def __init__(
        self,
        event_bus: EventBus,
        bot_token: str = "",
        chat_id: str = "",
    ) -> None:
        self._event_bus = event_bus  # Bus de eventos del sistema
        self._bot_token = bot_token  # Token del bot de Telegram
        self._chat_id = chat_id  # ID del chat destino
        self._enabled = bool(bot_token and chat_id)  # Flag: True solo si ambos estan configurados
        self._bot: Any = None  # Instancia de telegram.Bot, creada de forma perezosa (lazy)

        if self._enabled:
            # Suscribirse a los eventos de ejecucion para notificar automaticamente.
            self._event_bus.subscribe(FillEvent, self._on_fill, name="TelegramNotifier.fill")
            logger.info("TelegramNotifier enabled (chat_id=%s)", chat_id)
        else:
            logger.info("TelegramNotifier disabled (token or chat_id not configured)")

    # ── Ciclo de Vida ─────────────────────────────────────────────

    async def start(self) -> None:
        """Inicializa de forma perezosa el cliente del bot de Telegram.

        Si la biblioteca python-telegram-bot no esta instalada, se
        deshabilita el notificador automaticamente sin lanzar error.
        """
        if not self._enabled:
            return  # No esta habilitado; no hacer nada.
        try:
            from telegram import Bot

            self._bot = Bot(token=self._bot_token)  # Crear la instancia del bot.
            logger.info("Telegram bot client initialised")
        except ImportError:
            # La dependencia no esta instalada; deshabilitar el notificador.
            logger.warning(
                "python-telegram-bot not installed; TelegramNotifier disabled"
            )
            self._enabled = False

    async def stop(self) -> None:
        """Apaga el cliente del bot de Telegram de forma ordenada.

        Ignora errores durante el shutdown ya que no afectan al sistema.
        """
        if self._bot is not None:
            try:
                await self._bot.shutdown()  # Cerrar la conexion del bot.
            except Exception:
                logger.debug("Telegram bot shutdown error (ignored)")
            self._bot = None

    # ── API Publica ───────────────────────────────────────────────

    async def send_message(self, text: str, parse_mode: str = "HTML") -> None:
        """Envia un mensaje arbitrario al chat configurado.

        No hace nada si el notificador esta deshabilitado.

        Parametros
        ----------
        text : str
            Contenido del mensaje a enviar.
        parse_mode : str
            Modo de parseo de Telegram (por defecto ``"HTML"``).
        """
        if not self._enabled or self._bot is None:
            return  # Notificador deshabilitado o bot no inicializado.
        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=parse_mode,
            )
        except Exception:
            logger.exception("Failed to send Telegram message")

    async def notify_fill(self, fill: FillEvent) -> None:
        """Formatea y envia una notificacion de operacion ejecutada (fill).

        Parametros
        ----------
        fill : FillEvent
            Evento de ejecucion con los detalles de la operacion.
        """
        if not self._enabled:
            return  # Notificador deshabilitado.
        # Formatear los valores decimales para presentacion legible.
        price = _fmt_decimal(fill.price)
        qty = _fmt_decimal(fill.quantity)
        fee = _fmt_decimal(fill.fee)
        # Construir el mensaje con formato HTML para Telegram.
        text = (
            f"<b>Fill</b> {fill.side.value.upper()} {fill.symbol}\n"
            f"Qty: {qty} @ {price}\n"
            f"Fee: {fee} {fill.fee_currency}\n"
            f"Strategy: {fill.strategy_name or 'n/a'}"
        )
        await self.send_message(text)

    async def notify_circuit_breaker(self, reason: str) -> None:
        """Envia una alerta de circuit-breaker (interruptor de circuito).

        Se invoca cuando el sistema detecta condiciones anormales que
        requieren detener las operaciones de trading.

        Parametros
        ----------
        reason : str
            Razon por la que se activo el circuit-breaker.
        """
        if not self._enabled:
            return  # Notificador deshabilitado.
        text = f"<b>CIRCUIT BREAKER</b>\n{reason}"
        await self.send_message(text)

    async def notify_health(self, report: dict[str, Any]) -> None:
        """Envia un reporte de verificacion de salud (tipicamente cuando hay fallos).

        Parametros
        ----------
        report : dict[str, Any]
            Diccionario que mapea nombre del componente a ``True`` (sano)
            o un string con el mensaje de error.
        """
        if not self._enabled:
            return  # Notificador deshabilitado.
        # Construir las lineas del reporte con el estado de cada componente.
        lines = ["<b>Health Check</b>"]
        for component, status in report.items():
            icon = "OK" if status is True else f"FAIL ({status})"
            lines.append(f"  {component}: {icon}")
        await self.send_message("\n".join(lines))

    # ── Manejador de Eventos ──────────────────────────────────────

    async def _on_fill(self, event: FillEvent) -> None:
        """Callback interno que se ejecuta al recibir un FillEvent del bus de eventos."""
        await self.notify_fill(event)


# ── Funciones Auxiliares ──────────────────────────────────────────


def _fmt_decimal(value: Decimal) -> str:
    """Formatea un Decimal, eliminando los ceros finales innecesarios.

    Parametros
    ----------
    value : Decimal
        Valor decimal a formatear.

    Retorno
    -------
    str
        Representacion en texto del valor sin ceros finales.
    """
    return f"{value:f}".rstrip("0").rstrip(".")
