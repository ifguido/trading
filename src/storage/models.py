"""Modelos ORM de SQLAlchemy para CryptoTrader.

Todos los precios y valores monetarios se almacenan como ``Numeric`` (que
se mapea a ``Decimal`` en Python) para evitar errores de redondeo por
punto flotante. Todas las marcas de tiempo estan en UTC.

Modelos definidos:
  - Trade: registro de una operacion completada (fill).
  - Order: orden enviada o confirmada por el exchange.
  - DailyPnL: resumen diario de ganancias y perdidas por simbolo.
  - SignalLog: registro inmutable de cada senal generada por las estrategias.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import Date, DateTime, Index, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """Retorna la fecha y hora actual en UTC. Se usa como valor por defecto en columnas de timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base declarativa compartida para todos los modelos ORM del proyecto."""


# ── Trade (Operacion) ─────────────────────────────────────────────


class Trade(Base):
    """Registro de una operacion completada (fill) en el exchange.

    Cada instancia representa una ejecucion real de una orden,
    incluyendo el precio de ejecucion, la cantidad, las comisiones
    y la estrategia que la origino.

    Atributos
    ---------
    id : int
        Clave primaria autoincrementable.
    exchange_order_id : str
        Identificador de la orden asignado por el exchange.
    client_order_id : str
        Identificador local de la orden generado por el sistema.
    symbol : str
        Par de trading, por ejemplo ``"BTC/USDT"``.
    side : str
        Lado de la operacion: ``"buy"`` (compra) o ``"sell"`` (venta).
    quantity : Decimal
        Cantidad ejecutada del activo.
    price : Decimal
        Precio de ejecucion.
    fee : Decimal
        Comision cobrada por el exchange.
    fee_currency : str
        Moneda en la que se cobro la comision.
    strategy_name : str
        Nombre de la estrategia que genero la orden.
    executed_at : datetime
        Momento exacto de la ejecucion (UTC).
    created_at : datetime
        Momento en que se creo el registro en la BD (UTC).
    """

    __tablename__ = "trades"

    # Clave primaria autoincrementable
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # --- Identificadores ---
    exchange_order_id: Mapped[str] = mapped_column(String(64), nullable=False)  # ID asignado por el exchange
    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # ID local del sistema
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)  # Par de trading (ej: BTC/USDT)

    # --- Detalles de la operacion ---
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # Lado: buy (compra) / sell (venta)
    quantity: Mapped[Decimal] = mapped_column(Numeric(precision=28, scale=12), nullable=False)  # Cantidad ejecutada
    price: Mapped[Decimal] = mapped_column(Numeric(precision=28, scale=12), nullable=False)  # Precio de ejecucion
    fee: Mapped[Decimal] = mapped_column(Numeric(precision=28, scale=12), nullable=False, default=Decimal(0))  # Comision
    fee_currency: Mapped[str] = mapped_column(String(16), nullable=False, default="")  # Moneda de la comision

    # --- Contexto ---
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False, default="")  # Estrategia que origino la orden

    # --- Marcas de tiempo (UTC) ---
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )  # Momento de ejecucion en el exchange
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )  # Momento de creacion del registro

    # Indices compuestos para optimizar consultas frecuentes.
    __table_args__ = (
        Index("ix_trades_symbol_executed", "symbol", "executed_at"),  # Busqueda por simbolo y fecha de ejecucion
        Index("ix_trades_strategy", "strategy_name"),  # Busqueda por estrategia
    )

    def __repr__(self) -> str:
        return (
            f"<Trade id={self.id} {self.side} {self.quantity} {self.symbol} "
            f"@ {self.price}>"
        )


# ── Order (Orden) ─────────────────────────────────────────────────


class Order(Base):
    """Orden enviada al exchange o confirmada por este.

    Almacena toda la informacion relevante de una orden, incluyendo su
    estado actual, los niveles de stop-loss y take-profit, y la
    estrategia que la genero.

    Atributos
    ---------
    id : int
        Clave primaria autoincrementable.
    client_order_id : str
        Identificador unico local de la orden.
    exchange_order_id : str | None
        Identificador asignado por el exchange (puede ser None si aun no fue confirmada).
    symbol : str
        Par de trading.
    side : str
        Lado de la orden: ``"buy"`` o ``"sell"``.
    order_type : str
        Tipo de orden (market, limit, stop_limit, etc.).
    quantity : Decimal
        Cantidad solicitada.
    price : Decimal | None
        Precio limite (None para ordenes de mercado).
    stop_loss : Decimal | None
        Nivel de stop-loss configurado.
    take_profit : Decimal | None
        Nivel de take-profit configurado.
    status : str
        Estado actual: pending / open / filled / cancelled / rejected.
    strategy_name : str
        Nombre de la estrategia que genero la orden.
    created_at : datetime
        Momento de creacion (UTC).
    updated_at : datetime
        Ultima actualizacion (UTC), se actualiza automaticamente.
    """

    __tablename__ = "orders"

    # Clave primaria autoincrementable
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # --- Identificadores ---
    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)  # ID unico local
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # ID del exchange (puede ser nulo)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)  # Par de trading

    # --- Detalles de la orden ---
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # Lado: buy / sell
    order_type: Mapped[str] = mapped_column(String(24), nullable=False)  # Tipo: market, limit, etc.
    quantity: Mapped[Decimal] = mapped_column(Numeric(precision=28, scale=12), nullable=False)  # Cantidad solicitada
    price: Mapped[Decimal | None] = mapped_column(Numeric(precision=28, scale=12), nullable=True)  # Precio limite (nulo para market)
    stop_loss: Mapped[Decimal | None] = mapped_column(Numeric(precision=28, scale=12), nullable=True)  # Nivel de stop-loss
    take_profit: Mapped[Decimal | None] = mapped_column(Numeric(precision=28, scale=12), nullable=True)  # Nivel de take-profit

    # --- Seguimiento de estado ---
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )  # Estados posibles: pending / open / filled / cancelled / rejected

    # --- Contexto ---
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False, default="")  # Estrategia que origino la orden

    # --- Marcas de tiempo (UTC) ---
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )  # Momento de creacion
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )  # Se actualiza automaticamente en cada modificacion

    # Indices compuestos para optimizar consultas frecuentes.
    __table_args__ = (
        Index("ix_orders_symbol_status", "symbol", "status"),  # Busqueda por simbolo y estado
        Index("ix_orders_strategy", "strategy_name"),  # Busqueda por estrategia
    )

    def __repr__(self) -> str:
        return (
            f"<Order id={self.id} {self.side} {self.order_type} "
            f"{self.quantity} {self.symbol} status={self.status}>"
        )


# ── DailyPnL (Ganancias y Perdidas Diarias) ──────────────────────


class DailyPnL(Base):
    """Resumen diario de ganancias y perdidas por simbolo (o a nivel de portafolio).

    Almacena las metricas financieras agregadas por dia, incluyendo
    PnL realizado, PnL no realizado, comisiones acumuladas y cantidad
    de operaciones. El valor ``"PORTFOLIO"`` en el campo symbol indica
    un agregado de todo el portafolio.

    Atributos
    ---------
    id : int
        Clave primaria autoincrementable.
    trade_date : date
        Fecha del dia al que corresponde el resumen.
    symbol : str
        Par de trading o ``"PORTFOLIO"`` para el agregado general.
    realized_pnl : Decimal
        Ganancias/perdidas realizadas del dia.
    unrealized_pnl : Decimal
        Ganancias/perdidas no realizadas al cierre del dia.
    fees : Decimal
        Comisiones totales pagadas en el dia.
    trade_count : int
        Cantidad de operaciones realizadas en el dia.
    created_at : datetime
        Momento de creacion del registro (UTC).
    """

    __tablename__ = "daily_pnl"

    # Clave primaria autoincrementable
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    trade_date: Mapped[date] = mapped_column(Date, nullable=False)  # Fecha del resumen
    symbol: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PORTFOLIO"
    )  # 'PORTFOLIO' para el agregado de todo el portafolio

    # --- Metricas financieras ---
    realized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(precision=28, scale=12), nullable=False, default=Decimal(0)
    )  # PnL realizado (ganancias/perdidas cerradas)
    unrealized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(precision=28, scale=12), nullable=False, default=Decimal(0)
    )  # PnL no realizado (posiciones abiertas)
    fees: Mapped[Decimal] = mapped_column(
        Numeric(precision=28, scale=12), nullable=False, default=Decimal(0)
    )  # Comisiones totales del dia
    trade_count: Mapped[int] = mapped_column(nullable=False, default=0)  # Numero de operaciones del dia

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )  # Momento de creacion del registro

    # Indice unico compuesto: solo puede haber un registro por fecha y simbolo.
    __table_args__ = (
        Index("ix_daily_pnl_date_symbol", "trade_date", "symbol", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<DailyPnL {self.trade_date} {self.symbol} "
            f"realized={self.realized_pnl} unrealized={self.unrealized_pnl}>"
        )


# ── SignalLog (Registro de Senales) ───────────────────────────────


class SignalLog(Base):
    """Registro inmutable de cada senal generada por las estrategias de trading.

    Cada senal incluye la direccion (long, short, close, hold), el nivel
    de confianza, los niveles de stop-loss y take-profit sugeridos, y
    metadatos adicionales en formato JSON.

    Este registro es de solo escritura (inmutable) y sirve para auditoria
    y analisis posterior del rendimiento de las estrategias.

    Atributos
    ---------
    id : int
        Clave primaria autoincrementable.
    event_id : str
        Identificador unico del evento que genero la senal.
    symbol : str
        Par de trading al que se refiere la senal.
    direction : str
        Direccion de la senal: long / short / close / hold.
    strategy_name : str
        Nombre de la estrategia que genero la senal.
    confidence : Decimal
        Nivel de confianza de la senal (entre 0 y 1).
    stop_loss : Decimal | None
        Nivel de stop-loss sugerido.
    take_profit : Decimal | None
        Nivel de take-profit sugerido.
    metadata_json : str | None
        Metadatos adicionales en formato JSON.
    created_at : datetime
        Momento de creacion del registro (UTC).
    """

    __tablename__ = "signal_logs"

    # Clave primaria autoincrementable
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # --- Identificacion de la senal ---
    event_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # ID unico del evento
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)  # Par de trading
    direction: Mapped[str] = mapped_column(String(8), nullable=False)  # Direccion: long / short / close / hold
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)  # Estrategia que genero la senal
    confidence: Mapped[Decimal] = mapped_column(
        Numeric(precision=8, scale=6), nullable=False, default=Decimal(0)
    )  # Nivel de confianza (0.0 a 1.0)
    stop_loss: Mapped[Decimal | None] = mapped_column(Numeric(precision=28, scale=12), nullable=True)  # Stop-loss sugerido
    take_profit: Mapped[Decimal | None] = mapped_column(Numeric(precision=28, scale=12), nullable=True)  # Take-profit sugerido
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # Metadatos adicionales en formato JSON

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )  # Momento de creacion del registro

    # Indices compuestos para consultas frecuentes.
    __table_args__ = (
        Index("ix_signal_logs_symbol_created", "symbol", "created_at"),  # Busqueda por simbolo y fecha
        Index("ix_signal_logs_strategy", "strategy_name"),  # Busqueda por estrategia
    )

    def __repr__(self) -> str:
        return (
            f"<SignalLog id={self.id} {self.direction} {self.symbol} "
            f"strategy={self.strategy_name} confidence={self.confidence}>"
        )
