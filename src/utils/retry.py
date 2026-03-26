"""Decorador asincrono de reintento con backoff exponencial y jitter.

Proporciona un decorador reutilizable que reintenta automaticamente
funciones asincronas fallidas, con tiempos de espera que crecen
exponencialmente y un componente aleatorio (jitter) para evitar
que multiples llamadores reintentan al mismo tiempo (efecto "thundering herd").
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Variable de tipo generica para preservar la firma de la funcion decorada.
F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorador que reintenta una funcion asincrona con backoff exponencial.

    Envuelve una funcion asincrona para que, si falla con una de las
    excepciones especificadas, se reintente automaticamente con un
    tiempo de espera creciente. El jitter aleatorio evita reintentos
    sincronizados entre multiples instancias.

    Parametros
    ----------
    max_retries : int
        Numero maximo de reintentos (0 = sin reintentos, solo la llamada
        inicial). Por defecto 3.
    base_delay : float
        Retardo base en segundos antes del primer reintento. Por defecto 1.0.
    max_delay : float
        Limite superior para el retardo calculado. Por defecto 60.0 segundos.
    exceptions : tuple[type[BaseException], ...]
        Tupla de tipos de excepcion que disparan un reintento.
        Por defecto ``(Exception,)``.

    Lanza
    -----
    Exception
        La ultima excepcion capturada si se agotan todos los reintentos.

    Ejemplo::

        @retry(max_retries=5, base_delay=0.5)
        async def fetch_data():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None  # Almacena la ultima excepcion para re-lanzar si se agotan los reintentos
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    # Verificar si se agotaron todos los reintentos.
                    if attempt == max_retries:
                        logger.error(
                            "Retry exhausted for %s after %d attempts: %s",
                            func.__qualname__,
                            max_retries + 1,
                            exc,
                        )
                        raise  # Re-lanzar la ultima excepcion al agotar los reintentos.

                    # Calcular el retardo con backoff exponencial y jitter completo.
                    delay = min(base_delay * (2**attempt), max_delay)  # Backoff exponencial con tope maximo
                    jittered = random.uniform(0, delay)  # noqa: S311 -- Jitter aleatorio entre 0 y el retardo calculado
                    logger.warning(
                        "Retry %d/%d for %s in %.2fs: %s",
                        attempt + 1,
                        max_retries,
                        func.__qualname__,
                        jittered,
                        exc,
                    )
                    await asyncio.sleep(jittered)  # Esperar antes del siguiente reintento.

            # Este punto nunca deberia alcanzarse, pero satisface al verificador de tipos.
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
