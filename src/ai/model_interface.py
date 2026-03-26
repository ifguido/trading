"""Protocolo que define la interfaz que todo modelo AI debe cumplir.

Utiliza tipado estructural (Protocol) para que los modelos no necesiten
heredar de una clase base -- solo necesitan implementar los métodos requeridos.
Esto permite duck-typing verificable en tiempo de ejecución gracias a
``@runtime_checkable``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .signal import AISignal


@runtime_checkable
class ModelInterface(Protocol):
    """Interfaz estructural para modelos de predicción AI.

    Cualquier clase que implemente estos métodos y propiedades se considera
    un ``ModelInterface`` válido en tiempo de ejecución (duck-typing
    verificado por ``@runtime_checkable``).

    Métodos requeridos:
    - model_id (propiedad): identificador único del modelo
    - predict(): genera señal de trading a partir de features
    - warmup(): precalentamiento con datos históricos
    - health_check(): verifica si el modelo está listo
    """

    @property
    def model_id(self) -> str:
        """Identificador único para esta instancia del modelo."""
        ...

    async def predict(self, features: dict[str, Any]) -> AISignal:
        """Genera una señal de trading a partir de features computados.

        Parámetros
        ----------
        features : dict[str, Any]
            Diccionario de nombre_feature -> valor producido por el
            FeaturePipeline.

        Retorna
        -------
        AISignal
            Señal con dirección, confianza y metadatos.
        """
        ...

    async def warmup(self, historical_data: dict[str, Any]) -> None:
        """Precalienta el estado del modelo con datos históricos (opcional).

        Se llama una vez al inicio para que el modelo pueda pre-cargar pesos,
        calibrar umbrales o rellenar buffers internos.

        Parámetros
        ----------
        historical_data : dict[str, Any]
            Datos indexados por símbolo, con historial de velas, etc.
        """
        ...

    async def health_check(self) -> bool:
        """Devuelve True si el modelo está listo para generar predicciones."""
        ...
