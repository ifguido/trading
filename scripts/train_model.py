#!/usr/bin/env python3
"""Script de entrenamiento del modelo AI local (GradientBoosting).

Descarga datos históricos OHLCV de Binance vía ccxt REST (datos públicos,
no necesita API key), computa features técnicos, genera labels basados en
retorno futuro, y entrena un GradientBoostingClassifier.

Uso:
    python scripts/train_model.py
    python scripts/train_model.py --symbol ETH/USDT --timeframe 5m --days 180

El modelo entrenado y el scaler se guardan en models/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Agregar raíz del proyecto al path para imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 365,
    exchange_name: str = "binance",
) -> pd.DataFrame:
    """Descarga datos históricos OHLCV de Binance usando ccxt (síncrono).

    ccxt en modo síncrono usa REST público — no necesita API key para
    datos de mercado. Descarga en batches de 1000 candles (límite de Binance).

    Parámetros
    ----------
    symbol : str
        Par de trading (ej: "BTC/USDT").
    timeframe : str
        Timeframe de los candles (ej: "1h", "5m", "15m").
    days : int
        Cantidad de días de historia a descargar.
    exchange_name : str
        Nombre del exchange en ccxt.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas: timestamp, open, high, low, close, volume.
    """
    import ccxt

    logger.info(
        "Descargando %d días de %s %s desde %s...",
        days, symbol, timeframe, exchange_name,
    )

    exchange_cls = getattr(ccxt, exchange_name)
    exchange = exchange_cls({"enableRateLimit": True})

    # Calcular timestamp de inicio
    since = int((time.time() - days * 86400) * 1000)

    all_candles: list[list] = []
    batch_limit = 1000  # Máximo de Binance por request

    while True:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=batch_limit
            )
        except Exception as e:
            logger.warning("Error descargando batch: %s — reintentando...", e)
            time.sleep(2)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        logger.info("  Descargados %d candles (total: %d)", len(candles), len(all_candles))

        # Avanzar el cursor al timestamp del último candle + 1ms
        since = candles[-1][0] + 1

        # Si recibimos menos del límite, ya no hay más datos
        if len(candles) < batch_limit:
            break

        # Respetar rate limits
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        raise RuntimeError(f"No se pudieron descargar datos para {symbol} {timeframe}")

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    # Eliminar duplicados por timestamp
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Descarga completa: %d candles desde %s hasta %s",
        len(df),
        pd.to_datetime(df["timestamp"].iloc[0], unit="ms"),
        pd.to_datetime(df["timestamp"].iloc[-1], unit="ms"),
    )
    return df


def create_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    buy_threshold: float = 0.005,
    sell_threshold: float = -0.005,
) -> pd.Series:
    """Genera labels de clasificación basados en retorno futuro.

    Mira N candles hacia adelante y clasifica:
    - 2 (BUY) si el retorno futuro > buy_threshold (+0.5%)
    - 0 (SELL) si el retorno futuro < sell_threshold (-0.5%)
    - 1 (HOLD) si está entre ambos umbrales

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columna 'close'.
    horizon : int
        Candles hacia adelante para calcular el retorno.
    buy_threshold : float
        Retorno mínimo para clasificar como BUY (ej: 0.005 = 0.5%).
    sell_threshold : float
        Retorno máximo para clasificar como SELL (ej: -0.005 = -0.5%).

    Retorna
    -------
    pd.Series
        Serie con labels: 0=SELL, 1=HOLD, 2=BUY. Las últimas N filas son NaN.
    """
    future_return = df["close"].pct_change(horizon).shift(-horizon)

    labels = pd.Series(1, index=df.index, dtype=int)  # Default: HOLD
    labels[future_return > buy_threshold] = 2  # BUY
    labels[future_return < sell_threshold] = 0  # SELL
    labels[future_return.isna()] = -1  # Sin datos futuros — se descarta

    return labels


def train(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 365,
    horizon: int = 5,
    buy_threshold: float = 0.005,
    sell_threshold: float = -0.005,
    output_dir: str = "models",
) -> None:
    """Pipeline completo de entrenamiento.

    1. Descarga datos históricos
    2. Computa features con feature_engineer
    3. Genera labels (retorno futuro)
    4. Split temporal 80/20
    5. Entrena GradientBoostingClassifier
    6. Imprime métricas
    7. Guarda modelo + scaler en disco
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import joblib

    from src.ai.feature_engineer import compute_features_dataframe

    # ── 1. Descargar datos ─────────────────────────────────────────────
    df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, days=days)

    # ── 2. Computar features ───────────────────────────────────────────
    logger.info("Computando features...")
    features_df = compute_features_dataframe(df)

    # ── 3. Generar labels ──────────────────────────────────────────────
    logger.info(
        "Generando labels (horizon=%d, buy=%.3f, sell=%.3f)...",
        horizon, buy_threshold, sell_threshold,
    )
    labels = create_labels(
        df, horizon=horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # ── 4. Limpiar filas sin datos ─────────────────────────────────────
    # Combinar features y labels, descartar filas con NaN
    combined = features_df.copy()
    combined["label"] = labels

    # Descartar filas sin label válido y filas con NaN en features
    combined = combined[combined["label"] >= 0].dropna()

    X = combined.drop(columns=["label"]).values
    y = combined["label"].values.astype(int)
    feature_names = [c for c in combined.columns if c != "label"]

    logger.info("Dataset limpio: %d muestras, %d features", len(X), len(feature_names))
    logger.info(
        "Distribución de clases — SELL: %d, HOLD: %d, BUY: %d",
        (y == 0).sum(), (y == 1).sum(), (y == 2).sum(),
    )

    # ── 5. Split temporal 80/20 ────────────────────────────────────────
    # Split temporal (NO aleatorio) para respetar la secuencialidad
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info("Train: %d muestras, Test: %d muestras", len(X_train), len(X_test))

    # ── 6. Normalizar features ─────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 7. Entrenar modelo ─────────────────────────────────────────────
    logger.info("Entrenando GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train_scaled, y_train)
    logger.info("Entrenamiento completado")

    # ── 8. Evaluar en test set ─────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    class_names = ["SELL", "HOLD", "BUY"]
    print("\n" + "=" * 60)
    print("MÉTRICAS DE EVALUACIÓN (Test Set)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Accuracy total
    accuracy = (y_pred == y_test).mean()
    print(f"\nAccuracy total: {accuracy:.4f}")

    # Feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 features más importantes:")
    for i in range(min(10, len(feature_names))):
        idx = sorted_idx[i]
        print(f"  {feature_names[idx]:25s} {importances[idx]:.4f}")

    # ── 9. Guardar modelo y scaler ─────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / "model.joblib"
    scaler_file = output_path / "scaler.joblib"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    logger.info("Modelo guardado en: %s", model_file)
    logger.info("Scaler guardado en: %s", scaler_file)
    logger.info("Feature names: %s", feature_names)

    # Guardar metadata del modelo
    metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "horizon": horizon,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "accuracy": float(accuracy),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "model_type": "GradientBoostingClassifier",
    }
    import json
    meta_file = output_path / "model_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata guardada en: %s", meta_file)

    print(f"\n{'=' * 60}")
    print("¡Entrenamiento exitoso!")
    print(f"Modelo: {model_file}")
    print(f"Scaler: {scaler_file}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo AI local para el bot de trading"
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Par de trading (default: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--days", type=int, default=365, help="Días de historia (default: 365)")
    parser.add_argument("--horizon", type=int, default=5, help="Candles futuros para label (default: 5)")
    parser.add_argument("--buy-threshold", type=float, default=0.005, help="Umbral BUY (default: 0.005)")
    parser.add_argument("--sell-threshold", type=float, default=-0.005, help="Umbral SELL (default: -0.005)")
    parser.add_argument("--output-dir", default="models", help="Directorio de salida (default: models)")

    args = parser.parse_args()

    train(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        horizon=args.horizon,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
