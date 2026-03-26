"""Modelo AI local basado en GradientBoostingClassifier de scikit-learn.

Implementa ModelInterface (Protocol) para integrarse con el pipeline de trading.
Carga un modelo pre-entrenado y un scaler desde disco (formato joblib).
Si no encuentra el archivo de modelo, degrada graciosamente a comportamiento
tipo DummyModel (siempre HOLD) en vez de crashear.

Flujo de predicción:
1. Recibe dict de features del FeaturePipeline
2. Ordena features según el orden esperado por el modelo
3. Normaliza con StandardScaler
4. Ejecuta predict_proba() del GBM
5. Convierte probabilidades a AISignal (dirección + confianza)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.core.events import SignalDirection

from .feature_engineer import get_feature_names
from .signal import AISignal

logger = logging.getLogger(__name__)


class LocalModel:
    """Modelo GradientBoosting local que satisface ModelInterface (Protocol).

    Parámetros de config:
    - model_path: ruta al archivo .joblib con el modelo entrenado
    - scaler_path: ruta al archivo .joblib con el StandardScaler (opcional,
      se deriva de model_path si no se especifica)
    - ai_weight: peso del voto AI en la estrategia (informativo, no se usa aquí)
    - confidence_threshold: confianza mínima para emitir BUY/SELL (default 0.4)
    """

    # Mapeo de índices de clase a SignalDirection
    # El modelo predice 3 clases: 0=SELL, 1=HOLD, 2=BUY
    _CLASS_MAP = {
        0: SignalDirection.SHORT,
        1: SignalDirection.HOLD,
        2: SignalDirection.LONG,
    }
    _CLASS_NAMES = ["SELL", "HOLD", "BUY"]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Inicializa el modelo local con configuración de rutas y umbrales.

        Parámetros
        ----------
        config : dict[str, Any] | None
            Diccionario de configuración con claves opcionales:
            - model_path: ruta al archivo .joblib del modelo
            - scaler_path: ruta al archivo .joblib del scaler
            - confidence_threshold: umbral mínimo de confianza para señales
        """
        self._config = config or {}

        # Rutas a los archivos del modelo y scaler
        self._model_path = Path(self._config.get("model_path", "models/model.joblib"))
        scaler_default = self._model_path.parent / "scaler.joblib"  # Derivar ruta del scaler si no se especifica
        self._scaler_path = Path(self._config.get("scaler_path", str(scaler_default)))

        # Umbral de confianza: si la probabilidad máxima no supera esto, devuelve HOLD
        self._confidence_threshold = float(self._config.get("confidence_threshold", 0.4))

        # Estado interno — se carga en warmup()
        self._model: Any = None  # GradientBoostingClassifier
        self._scaler: Any = None  # StandardScaler
        self._feature_names: list[str] = get_feature_names()
        self._is_loaded = False  # Flag: True si el modelo se cargó correctamente
        self._model_id = "local-gbm-v1"

        logger.info(
            "LocalModel inicializado (model_path=%s, threshold=%.2f)",
            self._model_path,
            self._confidence_threshold,
        )

    # ── Propiedades (requeridas por ModelInterface) ────────────────────

    @property
    def model_id(self) -> str:
        """Identificador único de este modelo."""
        return self._model_id

    # ── Métodos del Protocol ──────────────────────────────────────────

    async def warmup(self, historical_data: dict[str, Any]) -> None:
        """Carga el modelo y scaler desde disco.

        Si los archivos no existen, registra un warning y opera en modo
        degradado (siempre devuelve HOLD). Esto permite que el bot arranque
        sin modelo entrenado -- útil en desarrollo o primera ejecución.

        Parámetros
        ----------
        historical_data : dict[str, Any]
            Datos históricos por símbolo (no se usan en esta implementación,
            pero son requeridos por el protocolo ModelInterface).
        """
        try:
            import joblib
        except ImportError:
            logger.warning(
                "joblib no instalado — LocalModel operará en modo degradado (HOLD siempre)"
            )
            return

        # Intentar cargar el modelo
        if not self._model_path.exists():
            logger.warning(
                "Archivo de modelo no encontrado: %s — operando en modo degradado",
                self._model_path,
            )
            return

        try:
            self._model = joblib.load(self._model_path)
            logger.info("Modelo cargado desde %s", self._model_path)
        except Exception:
            logger.exception("Error cargando modelo desde %s", self._model_path)
            return

        # Intentar cargar el scaler (opcional pero recomendado)
        if self._scaler_path.exists():
            try:
                self._scaler = joblib.load(self._scaler_path)
                logger.info("Scaler cargado desde %s", self._scaler_path)
            except Exception:
                logger.exception("Error cargando scaler desde %s", self._scaler_path)
                self._scaler = None
        else:
            logger.warning(
                "Archivo de scaler no encontrado: %s — predicciones sin normalizar",
                self._scaler_path,
            )

        self._is_loaded = True
        logger.info("LocalModel listo (model_id=%s)", self._model_id)

    async def predict(self, features: dict[str, Any]) -> AISignal:
        """Genera una señal de trading a partir de features computados.

        Flujo:
        1. Si no hay modelo cargado -> devuelve HOLD (degradacion graciosa)
        2. Extrae features en el orden esperado por el modelo
        3. Normaliza con scaler (si esta disponible)
        4. Ejecuta predict_proba() para obtener probabilidades por clase
        5. Clase con mayor probabilidad -> direccion de la senal
        6. Si la confianza no supera el umbral -> HOLD

        Parámetros
        ----------
        features : dict[str, Any]
            Diccionario de features computados por FeaturePipeline o feature_engineer.

        Retorna
        -------
        AISignal
            Señal con dirección (LONG/SHORT/HOLD), confianza y metadatos
            incluyendo probabilidades por clase.
        """
        # Modo degradado: sin modelo cargado
        if not self._is_loaded or self._model is None:
            return AISignal(
                direction=SignalDirection.HOLD,
                confidence=0.0,
                metadata={"model": self._model_id, "reason": "model_not_loaded"},
            )

        # Si no hay features suficientes
        if not features:
            return AISignal(
                direction=SignalDirection.HOLD,
                confidence=0.0,
                metadata={"model": self._model_id, "reason": "no_features"},
            )

        # Construir vector de features en el orden correcto
        feature_vector = []
        for fname in self._feature_names:
            val = features.get(fname, 0.0)
            # Reemplazar NaN o None por 0.0
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            feature_vector.append(float(val))

        # Reshape para sklearn: (1, n_features)
        X = np.array(feature_vector).reshape(1, -1)

        # Normalizar con scaler si está disponible
        if self._scaler is not None:
            try:
                X = self._scaler.transform(X)
            except Exception:
                logger.warning("Error en scaler.transform — usando features sin normalizar")

        # Predecir probabilidades por clase
        try:
            probas = self._model.predict_proba(X)[0]  # Array de probabilidades [P(SELL), P(HOLD), P(BUY)]
        except Exception:
            logger.exception("Error en predict_proba")
            return AISignal(
                direction=SignalDirection.HOLD,
                confidence=0.0,
                metadata={"model": self._model_id, "reason": "predict_error"},
            )

        # Clase con mayor probabilidad
        best_class_idx = int(np.argmax(probas))
        best_confidence = float(probas[best_class_idx])

        # Si la confianza no supera el umbral → HOLD
        if best_confidence < self._confidence_threshold:
            direction = SignalDirection.HOLD
        else:
            direction = self._CLASS_MAP.get(best_class_idx, SignalDirection.HOLD)

        return AISignal(
            direction=direction,
            confidence=best_confidence,
            metadata={
                "model": self._model_id,
                "probabilities": {
                    self._CLASS_NAMES[i]: round(float(probas[i]), 4)
                    for i in range(len(probas))
                },
                "predicted_class": self._CLASS_NAMES[best_class_idx],
            },
        )

    async def health_check(self) -> bool:
        """Devuelve True si el modelo está cargado y listo para predicciones.

        Retorna
        -------
        bool
            True si el modelo fue cargado exitosamente en warmup(), False en caso contrario.
        """
        return self._is_loaded and self._model is not None
