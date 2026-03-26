"""Dataclasses de eventos para la arquitectura dirigida por eventos (event-driven).

Este modulo define todos los tipos de eventos que fluyen a traves del sistema.
Cada componente (feeds de datos, estrategias, gestor de riesgo, ejecucion)
se comunica exclusivamente mediante estos eventos, lo que garantiza un
acoplamiento debil y facilita las pruebas unitarias.

Convenciones:
- Todos los precios usan Decimal para evitar errores de punto flotante.
- Todos los timestamps son milisegundos UTC epoch (int).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


# ── Enumeraciones de Trading ───────────────────────────────────────
# Estas enums representan los valores posibles para lados de orden,
# tipos de orden y direcciones de senal. Heredan de str para
# facilitar la serializacion a JSON.


class Side(str, Enum):
    """Lado de una orden: compra o venta."""
    BUY = "buy"    # Orden de compra
    SELL = "sell"   # Orden de venta


class OrderType(str, Enum):
    """Tipo de orden soportado por el sistema de ejecucion.

    Cada tipo determina como el exchange procesara la orden:
    - MARKET: se ejecuta al mejor precio disponible inmediatamente.
    - LIMIT: se ejecuta solo al precio especificado o mejor.
    - STOP_LOSS: se activa cuando el precio alcanza un umbral.
    - STOP_LOSS_LIMIT: como stop loss, pero con precio limite.
    """
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"


class SignalDirection(str, Enum):
    """Direccion de una senal de trading generada por una estrategia.

    - LONG: abrir posicion larga (comprar).
    - SHORT: abrir posicion corta (vender en corto).
    - CLOSE: cerrar la posicion actual.
    - HOLD: mantener sin accion (estado por defecto).
    """
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"


# ── Funciones auxiliares para valores por defecto ──────────────────

def _utc_ms() -> int:
    """Retorna el timestamp actual en milisegundos UTC.

    Se usa como valor por defecto para el campo 'timestamp' de los eventos,
    garantizando que cada evento tenga una marca temporal precisa.
    """
    return int(time.time() * 1000)


def _uuid() -> str:
    """Genera un identificador unico de 16 caracteres hexadecimales.

    Se usa como valor por defecto para 'event_id' y 'client_order_id',
    proporcionando IDs compactos pero con baja probabilidad de colision.
    """
    return uuid.uuid4().hex[:16]


# ── Evento Base ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Event:
    """Evento base del que heredan todos los eventos del sistema.

    Todos los eventos son inmutables (frozen=True) para garantizar la
    integridad de los datos mientras viajan por el bus de eventos.
    Usan slots=True para optimizar el uso de memoria.

    Atributos:
        event_id: Identificador unico del evento (16 chars hex).
        timestamp: Marca temporal UTC en milisegundos.
    """

    event_id: str = field(default_factory=_uuid)        # ID unico del evento
    timestamp: int = field(default_factory=_utc_ms)      # Timestamp UTC en ms


# ── Eventos de Datos de Mercado ────────────────────────────────────
# Estos eventos transportan informacion de mercado en tiempo real
# desde los feeds de datos hacia las estrategias y otros consumidores.


@dataclass(frozen=True, slots=True)
class TickEvent(Event):
    """Tick de precio en tiempo real recibido del exchange.

    Representa una actualizacion instantanea del mercado para un par
    de trading. Es el evento mas frecuente y de menor latencia.

    Atributos:
        symbol: Par de trading (ej. "BTC/USDT").
        bid: Mejor precio de compra actual.
        ask: Mejor precio de venta actual.
        last: Ultimo precio de ejecucion.
        volume_24h: Volumen acumulado en las ultimas 24 horas.
    """

    symbol: str = ""
    bid: Decimal = Decimal(0)          # Mejor precio de compra (bid)
    ask: Decimal = Decimal(0)          # Mejor precio de venta (ask)
    last: Decimal = Decimal(0)         # Ultimo precio ejecutado
    volume_24h: Decimal = Decimal(0)   # Volumen de 24h


@dataclass(frozen=True, slots=True)
class OrderBookEvent(Event):
    """Instantanea del libro de ordenes (top-of-book).

    Contiene los mejores niveles de oferta y demanda del libro de ordenes.
    El spread (diferencia entre mejor ask y mejor bid) es un indicador
    clave de liquidez.

    Atributos:
        symbol: Par de trading.
        bids: Tupla de tuplas (precio, cantidad) del lado comprador.
        asks: Tupla de tuplas (precio, cantidad) del lado vendedor.
        spread: Diferencia entre el mejor ask y el mejor bid.
    """

    symbol: str = ""
    bids: tuple[tuple[Decimal, Decimal], ...] = ()  # (precio, cantidad) del lado comprador
    asks: tuple[tuple[Decimal, Decimal], ...] = ()   # (precio, cantidad) del lado vendedor
    spread: Decimal = Decimal(0)                      # Diferencia ask - bid


@dataclass(frozen=True, slots=True)
class CandleEvent(Event):
    """Vela OHLCV (Open, High, Low, Close, Volume).

    Puede ser generada por agregacion de ticks o recibida directamente
    del exchange. Las estrategias consumen principalmente este tipo de
    evento para analisis tecnico.

    Atributos:
        symbol: Par de trading.
        timeframe: Intervalo temporal de la vela (ej. "1m", "5m", "1h").
        open: Precio de apertura.
        high: Precio maximo del periodo.
        low: Precio minimo del periodo.
        close: Precio de cierre.
        volume: Volumen total del periodo.
        closed: True si la vela esta finalizada (no es parcial).
    """

    symbol: str = ""
    timeframe: str = "1m"              # Intervalo temporal (ej. "1m", "5m")
    open: Decimal = Decimal(0)         # Precio de apertura
    high: Decimal = Decimal(0)         # Precio maximo del periodo
    low: Decimal = Decimal(0)          # Precio minimo del periodo
    close: Decimal = Decimal(0)        # Precio de cierre
    volume: Decimal = Decimal(0)       # Volumen total del periodo
    closed: bool = True                # True si la vela esta finalizada


# ── Eventos de Trading ─────────────────────────────────────────────
# Estos eventos representan el flujo de una operacion: senal -> orden -> ejecucion.
# Una estrategia emite un SignalEvent, el gestor de riesgo lo evalua y
# genera un OrderEvent, y el exchange responde con un FillEvent.


@dataclass(frozen=True, slots=True)
class SignalEvent(Event):
    """Senal de trading emitida por una estrategia.

    Representa la intencion de una estrategia de abrir, cerrar o mantener
    una posicion. El gestor de riesgo evalua esta senal antes de convertirla
    en una orden real.

    Atributos:
        symbol: Par de trading objetivo.
        direction: Direccion de la senal (LONG, SHORT, CLOSE, HOLD).
        strategy_name: Nombre de la estrategia que genero la senal.
        confidence: Nivel de confianza de la senal (0.0 a 1.0).
        stop_loss: Precio de stop loss sugerido (opcional).
        take_profit: Precio de take profit sugerido (opcional).
        metadata: Datos adicionales de la estrategia (indicadores, etc.).
    """

    symbol: str = ""
    direction: SignalDirection = SignalDirection.HOLD  # Direccion de la senal
    strategy_name: str = ""                            # Estrategia que genero la senal
    confidence: float = 0.0                            # Confianza de 0.0 a 1.0
    stop_loss: Decimal | None = None                   # Stop loss sugerido (opcional)
    take_profit: Decimal | None = None                 # Take profit sugerido (opcional)
    metadata: dict[str, Any] = field(default_factory=dict)  # Datos extra de la estrategia


@dataclass(frozen=True, slots=True)
class OrderEvent(Event):
    """Solicitud de orden aprobada por el gestor de riesgo.

    Este evento se genera despues de que el RiskManager valida una senal.
    Contiene todos los parametros necesarios para enviar la orden al exchange.

    Atributos:
        symbol: Par de trading.
        side: Lado de la orden (BUY o SELL).
        order_type: Tipo de orden (MARKET, LIMIT, etc.).
        quantity: Cantidad a operar.
        price: Precio limite (solo para ordenes LIMIT).
        stop_loss: Nivel de stop loss (obligatorio segun configuracion de riesgo).
        take_profit: Nivel de take profit (opcional).
        client_order_id: ID unico generado por el cliente para rastreo.
        strategy_name: Estrategia que origino la orden.
    """

    symbol: str = ""
    side: Side = Side.BUY                              # Lado: compra o venta
    order_type: OrderType = OrderType.MARKET            # Tipo de orden
    quantity: Decimal = Decimal(0)                      # Cantidad a operar
    price: Decimal | None = None                        # Precio limite (solo para LIMIT)
    stop_loss: Decimal | None = None                    # Nivel de stop loss
    take_profit: Decimal | None = None                  # Nivel de take profit
    client_order_id: str = field(default_factory=_uuid) # ID de rastreo del cliente
    strategy_name: str = ""                             # Estrategia que origino la orden


@dataclass(frozen=True, slots=True)
class FillEvent(Event):
    """Confirmacion de ejecucion de orden recibida del exchange.

    Indica que una orden fue ejecutada (total o parcialmente).
    El FillHandler usa este evento para actualizar el portafolio,
    registrar la operacion en la base de datos y enviar notificaciones.

    Atributos:
        symbol: Par de trading ejecutado.
        side: Lado de la ejecucion (BUY o SELL).
        quantity: Cantidad ejecutada.
        price: Precio de ejecucion.
        fee: Comision cobrada por el exchange.
        fee_currency: Moneda en la que se cobro la comision.
        exchange_order_id: ID asignado por el exchange.
        client_order_id: ID del cliente para correlacionar con OrderEvent.
        strategy_name: Estrategia que origino la operacion.
    """

    symbol: str = ""
    side: Side = Side.BUY                # Lado de la ejecucion
    quantity: Decimal = Decimal(0)       # Cantidad ejecutada
    price: Decimal = Decimal(0)          # Precio de ejecucion
    fee: Decimal = Decimal(0)            # Comision del exchange
    fee_currency: str = ""               # Moneda de la comision (ej. "USDT", "BNB")
    exchange_order_id: str = ""          # ID asignado por el exchange
    client_order_id: str = ""            # ID del cliente para correlacion
    strategy_name: str = ""              # Estrategia que origino la operacion
