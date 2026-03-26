"""Motor de dimensionamiento de posiciones.

Calcula el numero de unidades a comprar/vender dado el capital, la tolerancia
al riesgo y una estimacion opcional de volatilidad o tasa de acierto.
Todos los valores utilizan Decimal para precision financiera.

Modos soportados:
    - fixed_fraction : arriesgar un porcentaje fijo del capital por operacion.
    - volatility_adjusted : escalar la posicion inversamente con la volatilidad reciente.
    - kelly : criterio de Kelly completo basado en tasa de acierto y ratio de pago.
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal, ROUND_DOWN
from enum import Enum

logger = logging.getLogger(__name__)


class SizingMode(str, Enum):
    """Modos disponibles para el calculo del tamano de posicion."""
    FIXED_FRACTION = "fixed_fraction"  # Fraccion fija del capital
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # Ajustado por volatilidad
    KELLY = "kelly"  # Criterio de Kelly


class PositionSizer:
    """Calcula la cantidad de la orden (en unidades del activo base) para una operacion.

    Aplica el algoritmo de dimensionamiento seleccionado y luego impone limites
    de seguridad como el tope maximo por posicion y el tamano minimo de orden.

    Parametros
    ----------
    mode : SizingMode
        Algoritmo a utilizar (por defecto: fixed_fraction).
    max_position_pct : Decimal
        Tope maximo de cualquier posicion individual como fraccion del capital (ej. 0.10 = 10%).
    min_order_size : Decimal
        Tamano minimo de orden que acepta el exchange (filtra ordenes insignificantes).
    """

    def __init__(
        self,
        mode: SizingMode = SizingMode.FIXED_FRACTION,
        max_position_pct: Decimal = Decimal("0.10"),
        min_order_size: Decimal = Decimal("0.00001"),
    ) -> None:
        self._mode = mode  # Modo de calculo seleccionado
        self._max_position_pct = max_position_pct  # Limite maximo por posicion (fraccion del capital)
        self._min_order_size = min_order_size  # Tamano minimo de orden aceptable

    # -- API Publica --------------------------------------------------------

    def calculate(
        self,
        equity: Decimal,
        risk_per_trade: Decimal,
        entry_price: Decimal,
        stop_distance: Decimal,
        volatility: Decimal | None = None,
        win_rate: Decimal | None = None,
        payoff_ratio: Decimal | None = None,
    ) -> Decimal:
        """Calcula y retorna el tamano de posicion en unidades del activo base.

        Parametros
        ----------
        equity : Decimal
            Capital total actual de la cuenta.
        risk_per_trade : Decimal
            Fraccion del capital que se esta dispuesto a perder si se activa el stop (ej. 0.02 = 2%).
        entry_price : Decimal
            Precio esperado de entrada.
        stop_distance : Decimal
            Distancia absoluta en precio desde la entrada hasta el stop-loss.
        volatility : Decimal | None
            Volatilidad reciente del precio (requerida para el modo volatility_adjusted).
        win_rate : Decimal | None
            Tasa historica de acierto entre 0 y 1 (requerida para el modo kelly).
        payoff_ratio : Decimal | None
            Ratio ganancia_promedio / perdida_promedio (requerido para el modo kelly).

        Retorna
        -------
        Decimal
            Numero de unidades del activo base, redondeado hacia abajo.
            Retorna 0 si el tamano calculado es menor que min_order_size.
        """
        # Validaciones basicas: capital y precio deben ser positivos
        if equity <= Decimal(0) or entry_price <= Decimal(0):
            return Decimal(0)

        # La distancia al stop debe ser positiva para calcular el riesgo
        if stop_distance <= Decimal(0):
            logger.warning("stop_distance must be > 0; returning 0")
            return Decimal(0)

        # Seleccionar el algoritmo de dimensionamiento segun el modo configurado
        if self._mode == SizingMode.FIXED_FRACTION:
            qty = self._fixed_fraction(equity, risk_per_trade, stop_distance)
        elif self._mode == SizingMode.VOLATILITY_ADJUSTED:
            qty = self._volatility_adjusted(
                equity, risk_per_trade, stop_distance, volatility,
            )
        elif self._mode == SizingMode.KELLY:
            qty = self._kelly(
                equity, risk_per_trade, stop_distance, win_rate, payoff_ratio,
            )
        else:
            logger.error("Unknown sizing mode: %s", self._mode)
            return Decimal(0)

        # Imponer limite maximo por posicion (como fraccion del capital total)
        max_qty = (equity * self._max_position_pct) / entry_price
        qty = min(qty, max_qty)

        # Redondear hacia abajo a 8 decimales y aplicar filtro de tamano minimo
        qty = qty.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
        if qty < self._min_order_size:
            logger.debug("Calculated qty %s below min order size %s", qty, self._min_order_size)
            return Decimal(0)

        return qty

    # -- Algoritmos de Dimensionamiento -------------------------------------

    @staticmethod
    def _fixed_fraction(
        equity: Decimal, risk_per_trade: Decimal, stop_distance: Decimal,
    ) -> Decimal:
        """Arriesga una fraccion fija del capital por operacion.

        Formula: cantidad = (capital * riesgo_por_operacion) / distancia_al_stop

        Parametros
        ----------
        equity : Decimal
            Capital total de la cuenta.
        risk_per_trade : Decimal
            Fraccion del capital a arriesgar.
        stop_distance : Decimal
            Distancia absoluta al stop-loss.

        Retorna
        -------
        Decimal
            Cantidad calculada en unidades del activo base.
        """
        risk_amount = equity * risk_per_trade  # Monto monetario en riesgo
        return risk_amount / stop_distance

    @staticmethod
    def _volatility_adjusted(
        equity: Decimal,
        risk_per_trade: Decimal,
        stop_distance: Decimal,
        volatility: Decimal | None,
    ) -> Decimal:
        """Escala la posicion inversamente con la volatilidad.

        Si la volatilidad es alta, reduce el tamano; si es baja, lo aumenta.
        Formula: cantidad = (capital * riesgo) / (distancia_stop * multiplicador_vol)

        El multiplicador de volatilidad se calcula como:
            multiplicador = max(volatilidad / vol_base, 0.5) limitado a 3.0
        donde vol_base es una linea base del 2%.

        Parametros
        ----------
        equity : Decimal
            Capital total de la cuenta.
        risk_per_trade : Decimal
            Fraccion del capital a arriesgar.
        stop_distance : Decimal
            Distancia absoluta al stop-loss.
        volatility : Decimal | None
            Volatilidad reciente del precio. Si es None o <= 0, usa fraccion fija como respaldo.

        Retorna
        -------
        Decimal
            Cantidad ajustada por volatilidad en unidades del activo base.
        """
        base_vol = Decimal("0.02")  # Volatilidad base de referencia (2%)
        if volatility is None or volatility <= Decimal(0):
            # Sin volatilidad valida: usar metodo de fraccion fija como respaldo
            logger.warning("No valid volatility provided; falling back to fixed_fraction")
            risk_amount = equity * risk_per_trade
            return risk_amount / stop_distance

        # Calcular el multiplicador de volatilidad relativo a la base
        vol_multiplier = volatility / base_vol
        # Limitar el multiplicador entre 0.5 (minimo) y 3.0 (maximo)
        vol_multiplier = max(vol_multiplier, Decimal("0.5"))
        vol_multiplier = min(vol_multiplier, Decimal("3.0"))

        risk_amount = equity * risk_per_trade  # Monto monetario en riesgo
        return risk_amount / (stop_distance * vol_multiplier)

    @staticmethod
    def _kelly(
        equity: Decimal,
        risk_per_trade: Decimal,
        stop_distance: Decimal,
        win_rate: Decimal | None,
        payoff_ratio: Decimal | None,
    ) -> Decimal:
        """Dimensionamiento por criterio de Kelly (medio-Kelly por seguridad).

        Formula de Kelly completo:
            fraccion_kelly = tasa_acierto - (1 - tasa_acierto) / ratio_pago

        Se usa medio-Kelly y se limita al rango [0, riesgo_por_operacion].

        Parametros
        ----------
        equity : Decimal
            Capital total de la cuenta.
        risk_per_trade : Decimal
            Fraccion maxima del capital a arriesgar por operacion.
        stop_distance : Decimal
            Distancia absoluta al stop-loss.
        win_rate : Decimal | None
            Tasa historica de acierto (0-1). Requerido; si es None, usa fraccion fija.
        payoff_ratio : Decimal | None
            Ratio ganancia_promedio / perdida_promedio. Requerido; si es None, usa fraccion fija.

        Retorna
        -------
        Decimal
            Cantidad calculada en unidades del activo base usando medio-Kelly.
        """
        if win_rate is None or payoff_ratio is None:
            # Parametros insuficientes: usar fraccion fija como respaldo
            logger.warning("Kelly mode requires win_rate and payoff_ratio; falling back to fixed_fraction")
            risk_amount = equity * risk_per_trade
            return risk_amount / stop_distance

        # Ratio de pago debe ser positivo para que el calculo tenga sentido
        if payoff_ratio <= Decimal(0):
            return Decimal(0)

        # Calcular la fraccion de Kelly completa
        kelly_f = win_rate - (Decimal(1) - win_rate) / payoff_ratio

        # Usar medio-Kelly para mayor seguridad, limitado entre 0 y riesgo maximo
        half_kelly = kelly_f / Decimal(2)
        half_kelly = max(half_kelly, Decimal(0))  # No permitir valores negativos
        half_kelly = min(half_kelly, risk_per_trade)  # No exceder el riesgo maximo por operacion

        risk_amount = equity * half_kelly  # Monto en riesgo segun medio-Kelly
        return risk_amount / stop_distance
