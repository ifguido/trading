"""Dataclass de señal AI devuelta por las predicciones del modelo.

Define la estructura inmutable que encapsula el resultado de una predicción
del modelo de inteligencia artificial: dirección de trading, nivel de confianza,
metadatos opcionales y marca de tiempo de generación.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.core.events import SignalDirection


@dataclass(frozen=True, slots=True)
class AISignal:
    """Resultado de predicción de un modelo AI.

    Dataclass inmutable (frozen=True) con slots para eficiencia de memoria.
    Representa la salida estándar que todos los modelos deben producir.

    Atributos
    ---------
    direction : SignalDirection
        Dirección de trading predicha (LONG, SHORT, CLOSE, HOLD).
    confidence : float
        Nivel de confianza del modelo en la predicción, acotado a [0, 1].
    metadata : dict[str, Any]
        Metadatos arbitrarios del modelo (importancia de features, etc.).
    timestamp : int
        Milisegundos epoch UTC en el momento de generación de la señal.
    """

    direction: SignalDirection = SignalDirection.HOLD  # Dirección de trading por defecto: mantener posición
    confidence: float = 0.0  # Confianza por defecto: cero (sin predicción)
    metadata: dict[str, Any] = field(default_factory=dict)  # Metadatos opcionales del modelo
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))  # Marca temporal de generación

    def __post_init__(self) -> None:
        """Valida y ajusta los valores tras la inicialización.

        Acota la confianza al rango [0, 1] sin violar la restricción de
        inmutabilidad (frozen) usando object.__setattr__ directamente.
        """
        # Acotar confianza al rango [0, 1] sin violar frozen
        if not 0.0 <= self.confidence <= 1.0:
            object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))
