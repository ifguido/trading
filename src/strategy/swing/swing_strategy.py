"""Estrategia de swing trading basada en confluencia de indicadores tecnicos.

Usa cruce de medias moviles (MA crossover), RSI, MACD y Bandas de Bollinger
para generar señales de trading a mediano plazo. Opcionalmente integra un
modelo AI como 5to votante con peso configurable (ai_weight).

Sistema de votacion:
- 4 indicadores tecnicos (peso 1 cada uno)
- 1 voto AI opcional (peso configurable, default 1.5)
- Confianza = |score_neto| / total_indicadores
- Señal solo si confianza >= min_confidence (0.4)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from src.core.event_bus import EventBus
from src.core.events import (
    CandleEvent,
    OrderBookEvent,
    SignalDirection,
    SignalEvent,
    TickEvent,
)
from src.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

# ── Parametros por defecto ────────────────────────────────────────────

_DEFAULT_FAST_MA = 10  # Periodo de la media movil rapida
_DEFAULT_SLOW_MA = 30  # Periodo de la media movil lenta
_DEFAULT_RSI_PERIOD = 14  # Periodo de calculo del RSI
_DEFAULT_RSI_OVERBOUGHT = 70  # Umbral de sobrecompra del RSI
_DEFAULT_RSI_OVERSOLD = 30  # Umbral de sobreventa del RSI
_DEFAULT_MACD_FAST = 12  # Periodo EMA rapida del MACD
_DEFAULT_MACD_SLOW = 26  # Periodo EMA lenta del MACD
_DEFAULT_MACD_SIGNAL = 9  # Periodo de la linea de señal del MACD
_DEFAULT_BB_PERIOD = 20  # Periodo de las Bandas de Bollinger
_DEFAULT_BB_STD = 2.0  # Multiplicador de desviacion estandar para Bollinger
_DEFAULT_MIN_CANDLES = 15  # Velas minimas antes de generar señales
_DEFAULT_STOP_LOSS_PCT = Decimal("0.02")  # Porcentaje de stop loss (2%)
_DEFAULT_TAKE_PROFIT_PCT = Decimal("0.04")  # Porcentaje de toma de ganancias (4%)
_DEFAULT_MIN_CONFIDENCE = 0.25  # Confianza minima para emitir señal


class SwingStrategy(BaseStrategy):
    """Estrategia de swing trading basada en confluencia de indicadores tecnicos.

    Acumula velas OHLCV en un DataFrame de pandas por simbolo y calcula
    los siguientes indicadores:
    - Cruce de medias moviles (rapida vs. lenta)
    - RSI (Indice de Fuerza Relativa)
    - MACD (Convergencia/Divergencia de Medias Moviles)
    - Bandas de Bollinger

    Se genera una señal cuando multiples indicadores coinciden en la direccion.
    La confianza se basa en la cantidad de indicadores que confirman.

    Parametros
    ----------
    name:
        Identificador unico para esta instancia de estrategia.
    symbols:
        Lista de simbolos de pares de trading que opera esta estrategia.
    event_bus:
        El EventBus central para publicar SignalEvents.
    params:
        Parametros especificos de la estrategia. Claves soportadas:
        - fast_ma (int): Periodo de media movil rapida. Default 10.
        - slow_ma (int): Periodo de media movil lenta. Default 30.
        - rsi_period (int): Periodo de retrospeccion del RSI. Default 14.
        - rsi_overbought (int): Umbral de sobrecompra RSI. Default 70.
        - rsi_oversold (int): Umbral de sobreventa RSI. Default 30.
        - macd_fast (int): Periodo EMA rapida del MACD. Default 12.
        - macd_slow (int): Periodo EMA lenta del MACD. Default 26.
        - macd_signal (int): Periodo linea de señal MACD. Default 9.
        - bb_period (int): Periodo Bandas de Bollinger. Default 20.
        - bb_std (float): Multiplicador desv. estandar Bollinger. Default 2.0.
        - min_candles (int): Velas minimas antes de generar señales. Default 50.
        - stop_loss_pct (str/Decimal): Porcentaje de stop loss. Default "0.02".
        - take_profit_pct (str/Decimal): Porcentaje de toma de ganancias. Default "0.04".
        - min_confidence (float): Confianza minima para emitir señal. Default 0.4.
    """

    def __init__(
        self,
        name: str,
        symbols: list[str],
        event_bus: EventBus,
        params: dict[str, Any] | None = None,
        ai_model: Any | None = None,
        feature_pipeline: Any | None = None,
    ) -> None:
        super().__init__(name, symbols, event_bus, params, ai_model, feature_pipeline)

        # Parametros de indicadores tecnicos extraidos de la configuracion
        self._fast_ma: int = self._params.get("fast_ma", _DEFAULT_FAST_MA)
        self._slow_ma: int = self._params.get("slow_ma", _DEFAULT_SLOW_MA)
        self._rsi_period: int = self._params.get("rsi_period", _DEFAULT_RSI_PERIOD)
        self._rsi_overbought: int = self._params.get("rsi_overbought", _DEFAULT_RSI_OVERBOUGHT)
        self._rsi_oversold: int = self._params.get("rsi_oversold", _DEFAULT_RSI_OVERSOLD)
        self._macd_fast: int = self._params.get("macd_fast", _DEFAULT_MACD_FAST)
        self._macd_slow: int = self._params.get("macd_slow", _DEFAULT_MACD_SLOW)
        self._macd_signal: int = self._params.get("macd_signal", _DEFAULT_MACD_SIGNAL)
        self._bb_period: int = self._params.get("bb_period", _DEFAULT_BB_PERIOD)
        self._bb_std: float = self._params.get("bb_std", _DEFAULT_BB_STD)
        self._min_candles: int = self._params.get("min_candles", _DEFAULT_MIN_CANDLES)
        self._stop_loss_pct: Decimal = Decimal(
            str(self._params.get("stop_loss_pct", _DEFAULT_STOP_LOSS_PCT))
        )
        self._take_profit_pct: Decimal = Decimal(
            str(self._params.get("take_profit_pct", _DEFAULT_TAKE_PROFIT_PCT))
        )
        self._min_confidence: float = self._params.get("min_confidence", _DEFAULT_MIN_CONFIDENCE)

        # Peso del voto AI (1.5 = vale mas que un indicador individual)
        self._ai_weight: float = float(self._params.get("ai_weight", 1.5))

        # Peso del voto de sentiment (funding rates + whales)
        self._sentiment_weight: float = float(self._params.get("sentiment_weight", 1.0))

        # Referencia al feed de sentiment (se inyecta desde el engine)
        self._sentiment_feed: Any | None = None

        # DataFrames de velas por simbolo: symbol -> DataFrame
        self._candles: dict[str, pd.DataFrame] = {}

        # Rastreo de la ultima señal por simbolo para evitar emisiones duplicadas
        self._last_signal: dict[str, SignalDirection] = {}

    # ── Ciclo de vida ─────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Configura DataFrames vacios para cada simbolo.

        Crea un DataFrame con columnas OHLCV para cada par de trading
        y establece la direccion inicial de señal como HOLD.
        """
        await super().initialize()
        for symbol in self._symbols:
            # Crear DataFrame vacio con columnas de vela OHLCV
            self._candles[symbol] = pd.DataFrame(
                {col: pd.Series(dtype="float64") for col in ["timestamp", "open", "high", "low", "close", "volume"]}
            )
            # Inicializar la ultima señal como HOLD (sin posicion)
            self._last_signal[symbol] = SignalDirection.HOLD
        logger.info(
            "SwingStrategy '%s' initialized — fast_ma=%d, slow_ma=%d, "
            "rsi=%d, min_candles=%d",
            self._name,
            self._fast_ma,
            self._slow_ma,
            self._rsi_period,
            self._min_candles,
        )

    # ── Manejadores de eventos ────────────────────────────────────────

    async def on_tick(self, event: TickEvent) -> None:
        """La estrategia de swing no actua sobre ticks individuales."""
        pass

    async def on_order_book(self, event: OrderBookEvent) -> None:
        """La estrategia de swing no utiliza datos del libro de ordenes."""
        pass

    async def on_candle(self, event: CandleEvent) -> None:
        """Acumula datos de velas y evalua indicadores.

        Solo procesa velas finalizadas (cerradas). Si el simbolo no
        pertenece a esta estrategia o la vela no esta cerrada, se ignora.
        Una vez que se alcanza el minimo de velas, se ejecuta la evaluacion.

        Parametros
        ----------
        event:
            Evento de vela OHLCV entrante.
        """
        # Solo evaluar en timeframe de 1 minuto para evitar duplicados
        if event.timeframe != "1m":
            return
        # Ignorar simbolos que no pertenecen a esta estrategia
        if event.symbol not in self._symbols:
            return
        # Solo procesar velas cerradas/finalizadas
        if not event.closed:
            return

        self._append_candle(event)

        df = self._candles[event.symbol]
        # Esperar hasta tener suficientes velas para calcular indicadores
        if len(df) < self._min_candles:
            logger.info(
                "Waiting for candles: %d/%d (%s)",
                len(df),
                self._min_candles,
                event.symbol,
            )
            return

        # Evaluar indicadores y posiblemente generar señal
        await self._evaluate(event.symbol)

    # ── Interno: Acumulacion de velas ─────────────────────────────────

    def _append_candle(self, event: CandleEvent) -> None:
        """Agrega un evento de vela al DataFrame del simbolo.

        Convierte los valores de la vela a float y los almacena.
        Mantiene una ventana movil para evitar crecimiento ilimitado
        de memoria.

        Parametros
        ----------
        event:
            Evento de vela con datos OHLCV a agregar.
        """
        # Crear fila con datos de la vela convertidos a float
        new_row = pd.DataFrame(
            [
                {
                    "timestamp": event.timestamp,
                    "open": float(event.open),
                    "high": float(event.high),
                    "low": float(event.low),
                    "close": float(event.close),
                    "volume": float(event.volume),
                }
            ]
        )
        symbol = event.symbol
        self._candles[symbol] = pd.concat(
            [self._candles[symbol], new_row], ignore_index=True
        )

        # Mantener ventana movil para prevenir crecimiento ilimitado de memoria
        max_rows = max(self._slow_ma, self._macd_slow, self._bb_period) * 5
        if len(self._candles[symbol]) > max_rows:
            self._candles[symbol] = (
                self._candles[symbol].iloc[-max_rows:].reset_index(drop=True)
            )

    # ── Interno: Funciones puras de calculo de indicadores ────────────

    @staticmethod
    def _compute_sma(series: pd.Series, period: int) -> pd.Series:
        """Media Movil Simple (SMA).

        Calcula el promedio aritmetico de los ultimos 'period' valores.

        Parametros
        ----------
        series:
            Serie de precios sobre la cual calcular.
        period:
            Numero de periodos para la ventana del promedio.

        Retorna
        -------
        pd.Series
            Serie con la media movil simple calculada.
        """
        return series.rolling(window=period).mean()

    @staticmethod
    def _compute_ema(series: pd.Series, period: int) -> pd.Series:
        """Media Movil Exponencial (EMA) usando pandas ewm.

        Da mayor peso a los valores mas recientes, reaccionando
        mas rapido a cambios de precio que la SMA.

        Parametros
        ----------
        series:
            Serie de precios sobre la cual calcular.
        period:
            Numero de periodos para el span de la EMA.

        Retorna
        -------
        pd.Series
            Serie con la media movil exponencial calculada.
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        """RSI usando el metodo de suavizado de Wilder.

        Calcula el Indice de Fuerza Relativa separando ganancias y perdidas,
        luego aplicando media movil exponencial ponderada con alpha=1/period
        (equivalente al suavizado de Wilder).

        Parametros
        ----------
        series:
            Serie de precios de cierre.
        period:
            Numero de periodos para el calculo del RSI.

        Retorna
        -------
        pd.Series
            Serie con valores del RSI entre 0 y 100.
        """
        # Calcular cambios entre periodos consecutivos
        delta = series.diff()
        # Separar ganancias (valores positivos) y perdidas (valores negativos)
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        # Suavizado de Wilder: ewm con alpha = 1/period
        avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        # Calcular fuerza relativa y convertir a indice (0-100)
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _compute_macd(
        self, series: pd.Series, fast: int, slow: int, signal: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD: EMA rapida - EMA lenta, linea de señal, histograma.

        El MACD mide el momentum comparando dos medias moviles exponenciales.
        El histograma muestra la diferencia entre la linea MACD y la señal.

        Parametros
        ----------
        series:
            Serie de precios de cierre.
        fast:
            Periodo de la EMA rapida.
        slow:
            Periodo de la EMA lenta.
        signal:
            Periodo de la linea de señal (EMA del MACD).

        Retorna
        -------
        macd_line : pd.Series
            Linea MACD (EMA rapida - EMA lenta).
        signal_line : pd.Series
            Linea de señal (EMA de la linea MACD).
        histogram : pd.Series
            Histograma (MACD - señal).
        """
        fast_ema = self._compute_ema(series, fast)
        slow_ema = self._compute_ema(series, slow)
        macd_line = fast_ema - slow_ema  # Diferencia entre EMAs
        signal_line = self._compute_ema(macd_line, signal)  # Señal suavizada
        histogram = macd_line - signal_line  # Divergencia MACD vs señal
        return macd_line, signal_line, histogram

    def _compute_bbands(
        self, series: pd.Series, period: int, std_multiplier: float
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bandas de Bollinger: SMA central +/- multiplicador * desv. estandar movil.

        Las bandas se expanden con alta volatilidad y se contraen con baja
        volatilidad. El precio cerca de la banda inferior sugiere sobreventa,
        y cerca de la banda superior sugiere sobrecompra.

        Parametros
        ----------
        series:
            Serie de precios de cierre.
        period:
            Periodo para la SMA central y la desviacion estandar.
        std_multiplier:
            Multiplicador de la desviacion estandar para las bandas.

        Retorna
        -------
        lower : pd.Series
            Banda inferior de Bollinger.
        middle : pd.Series
            Banda media (SMA).
        upper : pd.Series
            Banda superior de Bollinger.
        """
        middle = self._compute_sma(series, period)  # Banda media = SMA
        rolling_std = series.rolling(window=period).std()  # Desviacion estandar movil
        upper = middle + std_multiplier * rolling_std  # Banda superior
        lower = middle - std_multiplier * rolling_std  # Banda inferior
        return lower, middle, upper

    # ── Interno: Evaluacion de indicadores ────────────────────────────

    async def _evaluate(self, symbol: str) -> None:
        """Calcula indicadores y genera una señal si las condiciones se cumplen.

        Ejecuta el sistema de votacion donde cada indicador tecnico aporta
        un voto alcista o bajista. Si el modelo AI esta activo, agrega
        un voto adicional ponderado. La confianza final determina si
        se emite la señal.

        Parametros
        ----------
        symbol:
            Par de trading a evaluar.
        """
        df = self._candles[symbol]
        close = df["close"]  # Serie de precios de cierre

        # Calcular indicadores usando pandas / numpy puro
        fast_ma = self._compute_sma(close, self._fast_ma)
        slow_ma = self._compute_sma(close, self._slow_ma)
        rsi = self._compute_rsi(close, self._rsi_period)
        macd_line, macd_signal_line, macd_hist = self._compute_macd(
            close, self._macd_fast, self._macd_slow, self._macd_signal
        )
        bb_lower, bb_middle, bb_upper = self._compute_bbands(
            close, self._bb_period, self._bb_std
        )

        # Verificar que los indicadores tengan valores validos (no NaN)
        if fast_ma.iloc[-1] is np.nan or slow_ma.iloc[-1] is np.nan:
            return
        if np.isnan(rsi.iloc[-1]):
            return
        if np.isnan(macd_line.iloc[-1]) or np.isnan(bb_lower.iloc[-1]):
            return

        # Valores mas recientes de cada indicador
        current_fast_ma = fast_ma.iloc[-1]  # MA rapida actual
        current_slow_ma = slow_ma.iloc[-1]  # MA lenta actual
        prev_fast_ma = fast_ma.iloc[-2]  # MA rapida periodo anterior
        prev_slow_ma = slow_ma.iloc[-2]  # MA lenta periodo anterior
        current_rsi = rsi.iloc[-1]  # RSI actual
        current_close = close.iloc[-1]  # Precio de cierre actual

        current_macd = macd_line.iloc[-1]  # Linea MACD actual
        current_macd_hist = macd_hist.iloc[-1]  # Histograma MACD actual

        current_bbl = bb_lower.iloc[-1]  # Banda Bollinger inferior actual
        current_bbu = bb_upper.iloc[-1]  # Banda Bollinger superior actual

        # ── Puntuar cada indicador ─────────────────────────────────────
        # Cada indicador vota: +1 alcista, -1 bajista, 0 neutral
        bullish_votes = 0.0  # Acumulador de votos alcistas
        bearish_votes = 0.0  # Acumulador de votos bajistas
        # total_indicators comienza en 4 (tecnicos) y sube si AI esta activo
        total_indicators = 4.0

        # 1) MA Crossover — cruce de medias moviles
        # Cruce alcista: MA rapida cruza por encima de MA lenta
        ma_crossover_up = prev_fast_ma <= prev_slow_ma and current_fast_ma > current_slow_ma
        # Cruce bajista: MA rapida cruza por debajo de MA lenta
        ma_crossover_down = prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma
        # Posicion relativa sin cruce (señal mas debil)
        ma_bullish = current_fast_ma > current_slow_ma
        ma_bearish = current_fast_ma < current_slow_ma

        if ma_crossover_up:
            bullish_votes += 1  # Cruce alcista confirmado = voto completo
        elif ma_crossover_down:
            bearish_votes += 1  # Cruce bajista confirmado = voto completo
        elif ma_bullish:
            bullish_votes += 0.5  # Posicion alcista sin cruce = medio voto
        elif ma_bearish:
            bearish_votes += 0.5  # Posicion bajista sin cruce = medio voto

        # 2) RSI — sobrecompra/sobreventa
        if current_rsi <= self._rsi_oversold:
            bullish_votes += 1  # Sobrevendido = posible rebote alcista
        elif current_rsi >= self._rsi_overbought:
            bearish_votes += 1  # Sobrecomprado = posible correccion bajista
        elif current_rsi < 45:
            bullish_votes += 0.25  # Zona ligeramente alcista
        elif current_rsi > 55:
            bearish_votes += 0.25  # Zona ligeramente bajista

        # 3) MACD — momentum (impulso del mercado)
        if current_macd_hist > 0 and current_macd > 0:
            bullish_votes += 1  # Histograma y MACD positivos = impulso alcista fuerte
        elif current_macd_hist < 0 and current_macd < 0:
            bearish_votes += 1  # Histograma y MACD negativos = impulso bajista fuerte
        elif current_macd_hist > 0:
            bullish_votes += 0.5  # Solo histograma positivo = impulso alcista debil
        elif current_macd_hist < 0:
            bearish_votes += 0.5  # Solo histograma negativo = impulso bajista debil

        # 4) Bollinger Bands — posicion del precio relativa a las bandas
        if current_close <= current_bbl:
            bullish_votes += 1  # Precio en banda inferior = posible rebote
        elif current_close >= current_bbu:
            bearish_votes += 1  # Precio en banda superior = posible reversa
        elif current_close < (current_bbl + current_bbu) / 2:
            bullish_votes += 0.25  # Precio bajo la media = leve sesgo alcista
        elif current_close > (current_bbl + current_bbu) / 2:
            bearish_votes += 0.25  # Precio sobre la media = leve sesgo bajista

        # 5) AI Vote — prediccion del modelo de machine learning
        ai_prediction = None
        ai_signal = await self._get_ai_signal(symbol)

        if ai_signal is not None and ai_signal.direction != SignalDirection.HOLD:
            # AI tiene opinion -> agregar su voto con peso configurable
            total_indicators += self._ai_weight
            ai_prediction = ai_signal.direction.value

            if ai_signal.direction == SignalDirection.LONG:
                # Voto alcista ponderado por confianza del modelo y peso AI
                bullish_votes += self._ai_weight * ai_signal.confidence
            elif ai_signal.direction == SignalDirection.SHORT:
                # Voto bajista ponderado por confianza del modelo y peso AI
                bearish_votes += self._ai_weight * ai_signal.confidence

            logger.debug(
                "AI vote para %s: %s (confidence=%.2f, weight=%.1f)",
                symbol, ai_signal.direction.value, ai_signal.confidence, self._ai_weight,
            )

        # 6) Sentiment Vote — funding rates + whale activity
        if self._sentiment_feed is not None:
            sentiment = self._sentiment_feed.latest.get(symbol)
            if sentiment is not None and abs(sentiment.sentiment_score) > 0.1:
                total_indicators += self._sentiment_weight
                if sentiment.sentiment_score > 0:
                    bullish_votes += self._sentiment_weight * abs(sentiment.sentiment_score)
                else:
                    bearish_votes += self._sentiment_weight * abs(sentiment.sentiment_score)
                logger.info(
                    "Sentiment vote %s: score=%.2f funding=%.6f whale_bias=%.2f",
                    symbol, sentiment.sentiment_score, sentiment.funding_rate, sentiment.whale_bias,
                )

        # ── Determinar direccion y confianza ───────────────────────────
        # Score neto: positivo = alcista, negativo = bajista
        net_score = bullish_votes - bearish_votes
        # Confianza normalizada entre 0 y 1
        confidence = abs(net_score) / total_indicators

        logger.info(
            "%s | bull=%.1f bear=%.1f net=%.1f conf=%.2f (min=%.2f) rsi=%.1f",
            symbol, bullish_votes, bearish_votes, net_score, confidence,
            self._min_confidence, current_rsi,
        )

        # Si la confianza no alcanza el minimo, no emitir señal
        if confidence < self._min_confidence:
            return

        # Determinar direccion basada en el score neto
        # En spot trading solo se puede ir LONG o CLOSE (no se puede shortear)
        if net_score > 0:
            direction = SignalDirection.LONG
        elif net_score < 0:
            # Bearish en spot = cerrar posición si existe, no shortear
            direction = SignalDirection.CLOSE
        else:
            return

        # No emitir si la direccion no cambio (evitar duplicados)
        if self._last_signal.get(symbol) == direction:
            return
        self._last_signal[symbol] = direction  # Actualizar ultima señal emitida

        # Calcular stop-loss desde el precio de cierre actual
        # Take-profit se maneja via trailing stop (no fijo)
        close_price = Decimal(str(current_close))
        if direction == SignalDirection.LONG:
            stop_loss = close_price * (Decimal(1) - self._stop_loss_pct)
        else:
            stop_loss = close_price * (Decimal(1) + self._stop_loss_pct)
        take_profit = None

        # Metadata incluye detalles de cada indicador + prediccion AI
        metadata: dict[str, Any] = {
            "entry_price": float(close_price),
            "fast_ma": round(current_fast_ma, 6),
            "slow_ma": round(current_slow_ma, 6),
            "rsi": round(current_rsi, 2),
            "macd": round(current_macd, 6),
            "macd_hist": round(current_macd_hist, 6),
            "bb_lower": round(current_bbl, 6),
            "bb_upper": round(current_bbu, 6),
            "bullish_votes": bullish_votes,
            "bearish_votes": bearish_votes,
            "total_indicators": total_indicators,
        }

        # Agregar info del AI al metadata si hubo prediccion
        if ai_signal is not None:
            metadata["ai_prediction"] = ai_prediction
            metadata["ai_confidence"] = round(ai_signal.confidence, 4)
            metadata["ai_metadata"] = ai_signal.metadata

        # Construir y publicar el evento de señal
        signal = SignalEvent(
            symbol=symbol,
            direction=direction,
            strategy_name=self._name,
            confidence=round(confidence, 4),
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
        )
        await self._publish_signal(signal)
