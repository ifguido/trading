"""Limitador de tasa basado en token-bucket para operaciones asincronas.

Complementa el limitador de tasa integrado de CCXT con un limitador
a nivel de aplicacion que puede ser compartido entre multiples
llamadores. Implementa el algoritmo de token-bucket con reposicion
continua de tokens.
"""

from __future__ import annotations

import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Limitador de tasa asincrono basado en el algoritmo token-bucket.

    Los tokens se reponen de forma continua basandose en el tiempo
    transcurrido. Los llamadores deben usar ``await acquire()`` antes
    de realizar una solicitud limitada por tasa; el metodo dormira
    hasta que haya un token disponible.

    Tambien se puede usar como gestor de contexto asincrono (async context manager).

    Parametros
    ----------
    rate : float
        Numero maximo de solicitudes por segundo.
    burst : int | None
        Tamano maximo de rafaga (capacidad del bucket). Si no se
        especifica, se usa el valor de ``rate``, lo que significa
        que no se permite rafaga por encima de la tasa sostenida.

    Ejemplo::

        limiter = RateLimiter(rate=10)       # 10 solicitudes/segundo
        async with limiter:
            await exchange.fetch_ticker(...)

    Atributos internos
    ------------------
    _rate : float
        Tasa configurada (solicitudes por segundo).
    _max_tokens : int
        Capacidad maxima del bucket de tokens.
    _tokens : float
        Cantidad actual de tokens disponibles.
    _last_refill : float
        Marca de tiempo (monotonica) de la ultima reposicion de tokens.
    _lock : asyncio.Lock
        Candado para garantizar acceso exclusivo en operaciones concurrentes.
    """

    def __init__(self, rate: float, burst: int | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = rate  # Solicitudes permitidas por segundo
        self._max_tokens = burst if burst is not None else int(rate)  # Capacidad maxima del bucket
        self._tokens = float(self._max_tokens)  # Iniciar con el bucket lleno
        self._last_refill = time.monotonic()  # Marca de tiempo de la ultima reposicion
        self._lock = asyncio.Lock()  # Candado para concurrencia segura

    @property
    def rate(self) -> float:
        """Retorna la tasa configurada de solicitudes por segundo.

        Retorno
        -------
        float
            Numero maximo de solicitudes por segundo.
        """
        return self._rate

    @property
    def available_tokens(self) -> float:
        """Retorna el numero actual de tokens disponibles (aproximado).

        Realiza una reposicion de tokens antes de retornar el valor
        para que la lectura sea lo mas precisa posible.

        Retorno
        -------
        float
            Cantidad aproximada de tokens disponibles.
        """
        self._refill()
        return self._tokens

    async def acquire(self, tokens: int = 1) -> None:
        """Espera hasta que haya suficientes tokens disponibles y los consume.

        Este metodo es seguro para uso concurrente gracias al candado
        interno. Si no hay suficientes tokens, calcula el tiempo de
        espera necesario y duerme hasta que se repongan.

        Parametros
        ----------
        tokens : int
            Numero de tokens a consumir (por defecto 1).
        """
        async with self._lock:
            while True:
                self._refill()  # Reponer tokens segun el tiempo transcurrido.
                if self._tokens >= tokens:
                    self._tokens -= tokens  # Consumir los tokens solicitados.
                    return
                # Calcular cuanto tiempo esperar para obtener los tokens faltantes.
                deficit = tokens - self._tokens
                wait = deficit / self._rate  # Tiempo en segundos para reponer el deficit.
                await asyncio.sleep(wait)

    # ── Gestor de Contexto Asincrono ──────────────────────────────

    async def __aenter__(self) -> RateLimiter:
        """Al entrar al bloque ``async with``, adquiere un token automaticamente."""
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Al salir del bloque ``async with``, no se requiere limpieza."""
        pass

    # ── Metodos Internos ──────────────────────────────────────────

    def _refill(self) -> None:
        """Repone tokens en el bucket basandose en el tiempo transcurrido.

        Calcula cuantos tokens se deben agregar desde la ultima
        reposicion, sin exceder la capacidad maxima del bucket.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill  # Tiempo transcurrido desde la ultima reposicion
        # Agregar tokens proporcionales al tiempo transcurrido, sin exceder el maximo.
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now  # Actualizar la marca de tiempo de la ultima reposicion
