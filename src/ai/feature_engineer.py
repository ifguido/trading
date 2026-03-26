"""Feature engineer: convierte OHLCV DataFrames en vectores de features para el modelo AI.

Módulo compartido entre entrenamiento (scripts/train_model.py) e inferencia en vivo
(LocalModel). Todas las funciones son puras — reciben datos, devuelven features —
sin estado interno ni efectos secundarios.

Features computadas:
- Retornos: 1, 5, 15, 30 candles
- RSI (14 períodos)
- MACD (12, 26, 9) — línea, señal, histograma
- Bollinger Bands — posición normalizada [-1, 1]
- Ratios de volumen (actual vs promedio)
- Volatilidad (desviación estándar de retornos)
- Ratios de medias móviles (fast/slow)
- Hora del día y día de la semana (si hay timestamps)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Períodos por defecto para indicadores técnicos ─────────────────────
_RSI_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_BB_PERIOD = 20
_BB_STD = 2.0
_FAST_MA = 10
_SLOW_MA = 30

# Ventanas de retorno para capturar momentum a distintas escalas
_RETURN_WINDOWS = [1, 5, 15, 30]

# Mínimo de candles necesarios para calcular todos los features
MIN_CANDLES_FOR_FEATURES = 50


def compute_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Computa features técnicos para cada fila de un DataFrame OHLCV.

    Diseñada para entrenamiento: recibe un DataFrame completo y devuelve
    otro DataFrame con una columna por feature, alineado por índice.
    Las primeras filas tendrán NaN hasta que haya suficientes datos
    para calcular todos los indicadores.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas: open, high, low, close, volume.
        Opcionalmente 'timestamp' (epoch ms) para features temporales.

    Retorna
    -------
    pd.DataFrame
        DataFrame con una columna por feature, mismo índice que df.
    """
    features = pd.DataFrame(index=df.index)  # DataFrame de salida con mismo índice
    # Extraer series OHLCV como float para cálculos numéricos
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # ── Retornos a distintas ventanas (momentum multi-escala) ──────────
    for w in _RETURN_WINDOWS:
        features[f"return_{w}"] = close.pct_change(w)

    # ── RSI (Relative Strength Index) ──────────────────────────────────
    features["rsi"] = _compute_rsi_series(close, _RSI_PERIOD)

    # ── MACD (línea, señal, histograma) ────────────────────────────────
    macd_line, macd_signal, macd_hist = _compute_macd_series(
        close, _MACD_FAST, _MACD_SLOW, _MACD_SIGNAL
    )
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_histogram"] = macd_hist

    # ── Bollinger Bands — posición normalizada ─────────────────────────
    features["bb_position"] = _compute_bb_position_series(close, _BB_PERIOD, _BB_STD)

    # ── Ratios de volumen ──────────────────────────────────────────────
    vol_sma = volume.rolling(window=20).mean()
    features["volume_ratio"] = volume / vol_sma.replace(0, np.nan)

    # ── Volatilidad (desviación estándar de retornos, ventana 20) ──────
    features["volatility"] = close.pct_change().rolling(window=20).std()

    # ── Ratios de medias móviles ───────────────────────────────────────
    fast_ma = close.rolling(window=_FAST_MA).mean()
    slow_ma = close.rolling(window=_SLOW_MA).mean()
    features["ma_ratio"] = fast_ma / slow_ma.replace(0, np.nan)

    # ── Rango del candle (high-low) normalizado por close ──────────────
    features["candle_range"] = (high - low) / close.replace(0, np.nan)

    # ── Features temporales (hora y día de la semana) ──────────────────
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        features["hour"] = ts.dt.hour / 23.0  # Normalizado [0, 1]
        features["day_of_week"] = ts.dt.dayofweek / 6.0  # Normalizado [0, 1]

    return features


def compute_features_dict(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    timestamp_ms: int | None = None,
) -> dict[str, float]:
    """Computa features a partir de listas de precios — para inferencia en vivo.

    Toma los últimos N precios acumulados y devuelve un diccionario
    plano de features listo para pasar al modelo.

    Parámetros
    ----------
    closes : list[float]
        Precios de cierre, del más antiguo al más reciente.
    highs : list[float]
        Precios máximos.
    lows : list[float]
        Precios mínimos.
    volumes : list[float]
        Volúmenes.
    timestamp_ms : int | None
        Timestamp UTC en milisegundos del último candle (para features temporales).

    Retorna
    -------
    dict[str, float]
        Diccionario feature_name → valor. Vacío si no hay suficientes datos.
    """
    n = len(closes)  # Cantidad de velas disponibles
    # Verificar datos mínimos para evitar features con NaN
    if n < MIN_CANDLES_FOR_FEATURES:
        return {}

    # Construir DataFrame temporal para reutilizar la lógica de compute_features_dataframe
    df = pd.DataFrame({
        "open": closes,  # Aproximación: en vivo usamos close como proxy del open
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    if timestamp_ms is not None:
        # Generar timestamps ficticios: 1 candle = 1 minuto de diferencia
        # Solo el último timestamp es real; los anteriores son estimados
        interval_ms = 60_000  # 1 minuto
        timestamps = [timestamp_ms - (n - 1 - i) * interval_ms for i in range(n)]
        df["timestamp"] = timestamps

    feat_df = compute_features_dataframe(df)

    # Tomar la última fila (valores más recientes)
    last_row = feat_df.iloc[-1]

    # Convertir a dict, reemplazando NaN por 0.0
    result: dict[str, float] = {}
    for col in feat_df.columns:
        val = last_row[col]
        result[col] = 0.0 if (isinstance(val, float) and np.isnan(val)) else float(val)

    return result


def get_feature_names() -> list[str]:
    """Devuelve la lista ordenada de nombres de features.

    Útil para verificar que el modelo fue entrenado con los mismos features
    que se están computando en inferencia. El orden debe coincidir exactamente
    con el orden en que compute_features_dataframe genera las columnas.

    Retorna
    -------
    list[str]
        Lista de nombres de features en el orden esperado por el modelo.
    """
    # Construir lista de nombres en el mismo orden que compute_features_dataframe
    base = []
    for w in _RETURN_WINDOWS:
        base.append(f"return_{w}")  # Features de retorno multi-escala
    base.extend([
        "rsi",              # Relative Strength Index
        "macd",             # Línea MACD
        "macd_signal",      # Línea de señal del MACD
        "macd_histogram",   # Histograma MACD (momentum)
        "bb_position",      # Posición normalizada en Bollinger Bands
        "volume_ratio",     # Ratio de volumen actual vs promedio
        "volatility",       # Desviación estándar de retornos
        "ma_ratio",         # Ratio de medias móviles (fast/slow)
        "candle_range",     # Rango del candle normalizado
        "hour",             # Hora del día (normalizada)
        "day_of_week",      # Día de la semana (normalizado)
    ])
    return base


# ── Funciones auxiliares de indicadores (pd.Series → pd.Series) ────────


def _compute_rsi_series(close: pd.Series, period: int) -> pd.Series:
    """Calcula RSI usando suavizado de Wilder sobre toda la serie.

    Wilder's smoothing usa alpha = 1/period en ewm, lo cual da un promedio
    exponencial más suave que el EMA estándar. El RSI oscila entre 0 y 100:
    - > 70: sobrecomprado (señal bajista)
    - < 30: sobrevendido (señal alcista)
    """
    delta = close.diff()                      # Diferencia entre velas consecutivas
    gains = delta.clip(lower=0)                # Solo movimientos positivos (ganancias)
    losses = -delta.clip(upper=0)              # Solo movimientos negativos (pérdidas, valor absoluto)
    # Suavizado de Wilder con alpha=1/period para promedios exponenciales
    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Ratio de fuerza relativa (evitar div/0)
    return 100.0 - (100.0 / (1.0 + rs))         # Fórmula RSI estándar


def _compute_macd_series(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calcula MACD: línea MACD, señal, e histograma.

    - Línea MACD = EMA(fast) - EMA(slow)
    - Señal = EMA(línea MACD, signal)
    - Histograma = línea MACD - señal
    Un histograma positivo creciente indica momentum alcista.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()      # EMA rápida (período corto)
    ema_slow = close.ewm(span=slow, adjust=False).mean()      # EMA lenta (período largo)
    macd_line = ema_fast - ema_slow                             # Línea MACD = diferencia de EMAs
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()  # Señal = EMA de la línea MACD
    histogram = macd_line - signal_line                         # Histograma = diferencia MACD - señal
    return macd_line, signal_line, histogram


def _compute_bb_position_series(
    close: pd.Series, period: int, num_std: float
) -> pd.Series:
    """Calcula la posición normalizada dentro de las Bollinger Bands.

    Devuelve un valor en [-1, 1]:
    - -1 = precio en la banda inferior
    -  0 = precio en la media (SMA)
    - +1 = precio en la banda superior
    Valores fuera de [-1, 1] indican que el precio rompió las bandas.
    """
    sma = close.rolling(window=period).mean()   # Media móvil simple
    std = close.rolling(window=period).std()    # Desviación estándar móvil
    # Normalizar posición: (close - sma) / (num_std * std), evitando división por cero
    return (close - sma) / (num_std * std).replace(0, np.nan)
