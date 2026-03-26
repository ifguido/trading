"""Funciones auxiliares para normalizacion de tiempo en UTC.

Todas las marcas de tiempo internas en CryptoTrader se representan como
milisegundos epoch UTC (int). Estas funciones auxiliares convierten entre
esa representacion y los objetos ``datetime`` de Python, y tambien
proporcionan formateo legible para humanos.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


def utc_now_ms() -> int:
    """Retorna la hora actual en UTC como milisegundos epoch.

    Utiliza ``time.time()`` para obtener la hora del sistema y la
    convierte a milisegundos truncando a entero.

    Retorno
    -------
    int
        Milisegundos epoch UTC actuales.
    """
    return int(time.time() * 1000)


def ms_to_datetime(epoch_ms: int) -> datetime:
    """Convierte milisegundos epoch UTC a un ``datetime`` con zona horaria.

    Parametros
    ----------
    epoch_ms : int
        Tiempo epoch UTC en milisegundos.

    Retorno
    -------
    datetime
        Un objeto ``datetime`` con ``tzinfo=timezone.utc``.
    """
    # Dividir por 1000 para convertir milisegundos a segundos.
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """Convierte un ``datetime`` a milisegundos epoch UTC.

    Si *dt* es naive (sin tzinfo), se asume que esta en UTC.
    Esto es consistente con la convencion del proyecto de que
    todas las marcas de tiempo internas son UTC.

    Parametros
    ----------
    dt : datetime
        Una instancia de ``datetime`` (con o sin zona horaria).

    Retorno
    -------
    int
        Tiempo epoch UTC en milisegundos.
    """
    if dt.tzinfo is None:
        # Si el datetime no tiene zona horaria, se asume UTC.
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def format_timestamp(epoch_ms: int, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """Formatea milisegundos epoch UTC como una cadena legible para humanos.

    Parametros
    ----------
    epoch_ms : int
        Tiempo epoch UTC en milisegundos.
    fmt : str
        Cadena de formato ``strftime``. Por defecto ``"%Y-%m-%d %H:%M:%S UTC"``.

    Retorno
    -------
    str
        Cadena con la marca de tiempo formateada.
    """
    # Primero convertir milisegundos a datetime, luego formatear.
    dt = ms_to_datetime(epoch_ms)
    return dt.strftime(fmt)
