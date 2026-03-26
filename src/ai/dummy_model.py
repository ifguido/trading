"""Modelo AI sin operación (no-op) para ejecutar sin un modelo real.

Siempre devuelve HOLD con confianza cero. Útil para trading basado
puramente en estrategia o para desarrollo/testing donde no hay un modelo
de machine learning disponible. Satisface el protocolo ModelInterface
mediante tipado estructural.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.events import SignalDirection

from .signal import AISignal

logger = logging.getLogger(__name__)


class DummyModel:
    """Implementación dummy (sin operación) de ``ModelInterface``.

    Todos los métodos son no-op para que el sistema pueda funcionar
    sin un modelo AI real. Cumple con el protocolo ``ModelInterface``
    mediante tipado estructural (duck-typing).

    Se usa como fallback cuando:
    - No hay modelo entrenado disponible
    - Se quiere testear el pipeline sin predicciones AI
    - Se está en modo de desarrollo
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Inicializa el modelo dummy con configuración opcional.

        Parámetros
        ----------
        config : dict[str, Any] | None
            Diccionario de configuración. Solo se usa 'model_id'
            para asignar un identificador personalizado.
        """
        self._config = config or {}  # Configuración del modelo (puede estar vacía)
        self._model_id = self._config.get("model_id", "dummy-v0")  # Identificador por defecto
        logger.info("DummyModel inicializado (model_id=%s)", self._model_id)

    @property
    def model_id(self) -> str:
        """Identificador único de esta instancia del modelo."""
        return self._model_id

    async def predict(self, features: dict[str, Any]) -> AISignal:
        """Siempre devuelve HOLD con confianza cero (sin operación).

        Parámetros
        ----------
        features : dict[str, Any]
            Features computados (se ignoran en esta implementación).

        Retorna
        -------
        AISignal
            Señal HOLD con confianza 0.0 y metadata indicando no-op.
        """
        return AISignal(
            direction=SignalDirection.HOLD,
            confidence=0.0,
            metadata={"model": self._model_id, "reason": "dummy_no_op"},
        )

    async def warmup(self, historical_data: dict[str, Any]) -> None:
        """Precalentamiento sin operación — no hace nada.

        Parámetros
        ----------
        historical_data : dict[str, Any]
            Datos históricos (se ignoran en esta implementación).
        """
        logger.debug("DummyModel warmup llamado (sin operación)")

    async def health_check(self) -> bool:
        """Siempre devuelve True — el modelo dummy siempre está saludable."""
        return True
