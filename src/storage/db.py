"""Configuracion del motor asincrono y sesiones de SQLAlchemy v2.

Soporta tanto PostgreSQL (asyncpg) como SQLite (aiosqlite) mediante
la URL de conexion proporcionada en StorageConfig.

Este modulo gestiona los singletons del motor de base de datos y la
fabrica de sesiones, asegurando que solo exista una instancia de cada
uno durante el ciclo de vida de la aplicacion.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .models import Base

logger = logging.getLogger(__name__)

# Singletons a nivel de modulo, inicializados por ``init_db()``.
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Retorna el motor asincrono actual.

    Lanza ``RuntimeError`` si ``init_db()`` no ha sido invocado previamente.

    Retorno
    -------
    AsyncEngine
        La instancia del motor asincrono de SQLAlchemy.
    """
    if _engine is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _engine


def async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Retorna la fabrica de sesiones actual.

    Lanza ``RuntimeError`` si ``init_db()`` no ha sido invocado previamente.

    Retorno
    -------
    async_sessionmaker[AsyncSession]
        La fabrica de sesiones asincronas configurada.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _session_factory


async def init_db(url: str, *, echo: bool = False) -> AsyncEngine:
    """Crea el motor asincrono, la fabrica de sesiones y todas las tablas.

    Esta funcion debe ser llamada una sola vez al inicio de la aplicacion.
    Configura los parametros del pool de conexiones segun el dialecto
    (SQLite vs PostgreSQL) y ejecuta la creacion de tablas si no existen.

    Parametros
    ----------
    url:
        URL de conexion asincrona de SQLAlchemy, por ejemplo:
        ``sqlite+aiosqlite:///data/cryptotrader.db`` o
        ``postgresql+asyncpg://user:pass@host/dbname``.
    echo:
        Si es ``True``, registra en el log todo el SQL emitido (util para depuracion).

    Retorno
    -------
    AsyncEngine
        La instancia del motor recien creada.
    """
    global _engine, _session_factory  # noqa: PLW0603

    # Argumentos de conexion y pool, dependiendo del dialecto de la BD.
    connect_args: dict = {}
    pool_kwargs: dict = {}

    if url.startswith("sqlite"):
        # aiosqlite no soporta pool_size / max_overflow; se deshabilita check_same_thread.
        connect_args = {"check_same_thread": False}
    else:
        # asyncpg / PostgreSQL se beneficia del ajuste del pool de conexiones.
        pool_kwargs = {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_pre_ping": True,
        }

    # Crear el motor asincrono con los argumentos configurados.
    _engine = create_async_engine(
        url,
        echo=echo,
        connect_args=connect_args,
        **pool_kwargs,
    )

    # Configurar la fabrica de sesiones vinculada al motor.
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Evita consultas adicionales al acceder a atributos post-commit.
    )

    # Crear todas las tablas definidas en los modelos si aun no existen.
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialised (%s)", url.split("@")[-1] if "@" in url else url)
    return _engine


async def close_db() -> None:
    """Libera el motor y cierra todas las conexiones del pool.

    Debe invocarse durante el apagado ordenado de la aplicacion para
    liberar correctamente los recursos de base de datos.
    """
    global _engine, _session_factory  # noqa: PLW0603

    if _engine is not None:
        await _engine.dispose()  # Cierra todas las conexiones activas del pool.
        logger.info("Database connections closed")

    # Resetear los singletons a None para permitir una posible reinicializacion.
    _engine = None
    _session_factory = None
