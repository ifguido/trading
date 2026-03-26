"""Carga de configuracion: archivos YAML + .env -> modelos Pydantic.

Este modulo se encarga de cargar, fusionar y validar toda la configuracion
del sistema. El flujo es:
1. Cargar variables de entorno desde .env (dotenv).
2. Leer el archivo principal settings.yaml.
3. Fusionar archivos YAML de estrategias individuales.
4. Sustituir placeholders ${ENV_VAR} con valores reales del entorno.
5. Inyectar secretos (API keys, tokens) desde variables de entorno.
6. Validar todo con modelos Pydantic para garantizar tipos correctos.

Este enfoque permite tener configuracion versionada en YAML y secretos
seguros en variables de entorno, sin comprometer credenciales en el repositorio.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


# ── Modelos de Configuracion Pydantic ──────────────────────────────
# Cada modelo representa una seccion del archivo de configuracion.
# Pydantic valida automaticamente los tipos y aplica valores por defecto.


class ExchangeConfig(BaseModel):
    """Configuracion de conexion al exchange (ej. Binance).

    Define los parametros necesarios para conectarse al exchange,
    incluyendo modo de operacion y credenciales.

    Atributos:
        name: Nombre del exchange (debe coincidir con ccxt).
        mode: Modo de operacion ("live", "paper", "sandbox").
        sandbox: Si True, usar el entorno sandbox del exchange.
        api_key: Clave API del exchange (inyectada desde .env).
        api_secret: Secreto API del exchange (inyectado desde .env).
        rate_limit: Si True, respetar los limites de tasa del exchange.
        options: Opciones adicionales especificas del exchange.
    """
    name: str = "binance"
    mode: str = "live"          # "live" = real, "paper" = simulado, "sandbox" = pruebas
    sandbox: bool = False       # Activar modo sandbox del exchange
    api_key: str = ""           # Clave API (se inyecta desde variables de entorno)
    api_secret: str = ""        # Secreto API (se inyecta desde variables de entorno)
    rate_limit: bool = True     # Respetar rate limits del exchange
    options: dict[str, Any] = Field(default_factory=dict)  # Opciones extra para ccxt


class PairConfig(BaseModel):
    """Configuracion de un par de trading individual.

    Cada par define que simbolo operar, en que timeframes y con que estrategia.

    Atributos:
        symbol: Par de trading (ej. "BTC/USDT").
        timeframes: Lista de intervalos temporales a monitorear.
        strategy: Nombre de la estrategia a usar para este par.
    """
    symbol: str                                                     # Par de trading (obligatorio)
    timeframes: list[str] = Field(default_factory=lambda: ["1m", "5m"])  # Timeframes a monitorear
    strategy: str = "swing"                                         # Estrategia asignada al par


class RiskConfig(BaseModel):
    """Configuracion de gestion de riesgo.

    Define los limites que el RiskManager y el CircuitBreaker
    verifican antes de aprobar cualquier orden.

    Atributos:
        max_position_pct: Porcentaje maximo del portafolio por posicion.
        max_total_exposure_pct: Exposicion total maxima del portafolio.
        max_concurrent_positions: Numero maximo de posiciones abiertas simultaneamente.
        max_daily_loss_pct: Perdida diaria maxima permitida (activa circuit breaker).
        max_drawdown_pct: Drawdown maximo permitido desde el pico.
        mandatory_stop_loss: Si True, toda orden debe incluir stop loss.
    """
    max_position_pct: Decimal = Decimal("0.10")           # Max 10% del portafolio por posicion
    max_total_exposure_pct: Decimal = Decimal("0.50")     # Max 50% de exposicion total
    max_concurrent_positions: int = 6                      # Max 6 posiciones abiertas
    max_daily_loss_pct: Decimal = Decimal("0.03")         # Max 3% de perdida diaria
    max_drawdown_pct: Decimal = Decimal("0.10")           # Max 10% de drawdown
    mandatory_stop_loss: bool = True                       # Stop loss obligatorio en toda orden

    @field_validator("*", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any, info: Any) -> Any:
        """Convierte valores int/float a Decimal para campos de porcentaje.

        Los archivos YAML representan porcentajes como float (ej. 0.10),
        pero el sistema usa Decimal para precision. Este validador convierte
        automaticamente cualquier campo cuyo nombre contenga "pct".

        Parametros:
            v: Valor del campo antes de la validacion.
            info: Informacion del campo (nombre, tipo, etc.).

        Retorna:
            Decimal si el campo es de porcentaje y el valor es numerico,
            o el valor original sin modificar.
        """
        # Solo convertir a Decimal los campos que contengan "pct" en su nombre
        if info.field_name and "pct" in info.field_name and isinstance(v, (int, float)):
            return Decimal(str(v))
        return v


class AIConfig(BaseModel):
    """Configuracion del modulo de inteligencia artificial.

    Permite activar/desactivar el modelo AI y especificar que clase
    de modelo cargar dinamicamente.

    Atributos:
        enabled: Si True, cargar y usar el modelo AI real.
        model_class: Ruta completa de la clase del modelo (para importacion dinamica).
        config: Parametros especificos del modelo (ej. ai_weight, thresholds).
    """
    enabled: bool = False                                      # AI desactivado por defecto
    model_class: str = "src.ai.dummy_model.DummyModel"        # Clase del modelo a cargar
    config: dict[str, Any] = Field(default_factory=dict)       # Parametros del modelo


class TelegramConfig(BaseModel):
    """Configuracion del notificador de Telegram.

    Permite enviar alertas y notificaciones de trading a un chat de Telegram.

    Atributos:
        enabled: Si True, activar las notificaciones por Telegram.
        bot_token: Token del bot de Telegram (inyectado desde .env).
        chat_id: ID del chat al que enviar los mensajes.
    """
    enabled: bool = False    # Notificaciones desactivadas por defecto
    bot_token: str = ""      # Token del bot (se inyecta desde variables de entorno)
    chat_id: str = ""        # ID del chat de Telegram


class StorageConfig(BaseModel):
    """Configuracion de la base de datos.

    Define la URL de conexion a la base de datos donde se almacenan
    trades, ordenes, snapshots del portafolio, etc.

    Atributos:
        url: URL de conexion compatible con SQLAlchemy async.
    """
    url: str = "sqlite+aiosqlite:///data/cryptotrader.db"  # SQLite async por defecto


class LoggingConfig(BaseModel):
    """Configuracion del sistema de logging.

    Atributos:
        level: Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Si True, formatear logs en JSON (util para produccion).
    """
    level: str = "INFO"          # Nivel de logging
    json_format: bool = True     # Formato JSON para logs estructurados


class AppConfig(BaseModel):
    """Configuracion raiz de la aplicacion.

    Agrupa todas las secciones de configuracion en un solo modelo.
    Es el objeto principal que el Engine recibe para inicializar
    todos los componentes del sistema.

    Atributos:
        exchange: Configuracion de conexion al exchange.
        pairs: Lista de pares de trading a operar.
        risk: Limites y reglas de gestion de riesgo.
        ai: Configuracion del modelo de inteligencia artificial.
        telegram: Configuracion de notificaciones por Telegram.
        storage: Configuracion de la base de datos.
        logging: Configuracion del sistema de logging.
    """

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    pairs: list[PairConfig] = Field(default_factory=list)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ── Funciones de Carga ─────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> dict:
    """Fusiona recursivamente dos diccionarios (override sobre base).

    Se usa para combinar el archivo base settings.yaml con los archivos
    de configuracion de estrategias individuales. Los valores en 'override'
    tienen prioridad sobre los de 'base'. Los diccionarios anidados se
    fusionan recursivamente en lugar de reemplazarse.

    Parametros:
        base: Diccionario base con valores por defecto.
        override: Diccionario con valores que sobreescriben al base.

    Retorna:
        Nuevo diccionario con la fusion de ambos.
    """
    result = base.copy()
    for key, value in override.items():
        # Si ambos valores son diccionarios, fusionar recursivamente
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            # De lo contrario, el valor del override reemplaza al base
            result[key] = value
    return result


def _substitute_env(data: Any) -> Any:
    """Reemplaza placeholders ${ENV_VAR} con variables de entorno reales.

    Recorre recursivamente la estructura de datos (dicts y listas) buscando
    strings con formato ${NOMBRE_VARIABLE}. Si la variable de entorno existe
    y tiene valor, se reemplaza el placeholder. Si no existe o esta vacia,
    se reemplaza con None para que los valores por defecto downstream apliquen.

    Esto permite tener configuracion como:
        api_key: ${BINANCE_API_KEY}
    en YAML, y el valor real se inyecta desde el entorno.

    Parametros:
        data: Estructura de datos a procesar (str, dict, list, u otro).

    Retorna:
        La misma estructura con los placeholders sustituidos.
    """
    # Caso: string con placeholder ${...}
    if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_key = data[2:-1]  # Extraer nombre de la variable (sin ${ y })
        value = os.environ.get(env_key, "")
        return value if value else None  # None si la variable no existe o esta vacia
    # Caso: diccionario - procesar recursivamente cada valor
    if isinstance(data, dict):
        return {k: _substitute_env(v) for k, v in data.items()}
    # Caso: lista - procesar recursivamente cada elemento
    if isinstance(data, list):
        return [_substitute_env(item) for item in data]
    # Caso: cualquier otro tipo - devolver sin modificar
    return data


def load_config(
    settings_path: str | Path = "config/settings.yaml",
    strategy_dir: str | Path = "config/strategies",
    env_path: str | Path = ".env",
) -> AppConfig:
    """Carga y fusiona archivos de configuracion YAML con sustitucion de variables de entorno.

    Este es el punto de entrada principal para la configuracion del sistema.
    Ejecuta el siguiente flujo:
    1. Carga variables de entorno desde el archivo .env
    2. Lee settings.yaml como configuracion base
    3. Fusiona archivos YAML de estrategias (config/strategies/*.yaml)
    4. Sustituye placeholders ${ENV_VAR} con valores reales
    5. Inyecta secretos del exchange y Telegram desde variables de entorno
    6. Valida toda la configuracion con el modelo Pydantic AppConfig

    Parametros:
        settings_path: Ruta al archivo YAML principal de configuracion.
        strategy_dir: Directorio con archivos YAML de estrategias individuales.
        env_path: Ruta al archivo .env con variables de entorno.

    Retorna:
        AppConfig: Modelo Pydantic validado con toda la configuracion del sistema.
    """
    # Paso 1: Cargar variables de entorno desde .env
    load_dotenv(env_path)

    settings_path = Path(settings_path)
    strategy_dir = Path(strategy_dir)

    # Paso 2: Cargar configuracion base desde settings.yaml
    raw: dict[str, Any] = {}
    if settings_path.exists():
        with open(settings_path) as f:
            raw = yaml.safe_load(f) or {}

    # Paso 3: Fusionar configuraciones de estrategias individuales
    # Los archivos se procesan en orden alfabetico para determinismo
    if strategy_dir.exists():
        for yaml_file in sorted(strategy_dir.glob("*.yaml")):
            with open(yaml_file) as f:
                strategy_data = yaml.safe_load(f) or {}
            raw = _deep_merge(raw, strategy_data)

    # Paso 4: Sustituir placeholders ${ENV_VAR} en toda la configuracion
    raw = _substitute_env(raw)

    # Paso 5: Inyectar secretos del exchange desde variables de entorno
    # Se usa setdefault para no sobreescribir valores ya presentes en el YAML
    if "exchange" not in raw:
        raw["exchange"] = {}
    raw["exchange"].setdefault("api_key", os.environ.get("BINANCE_API_KEY", ""))
    raw["exchange"].setdefault("api_secret", os.environ.get("BINANCE_API_SECRET", ""))

    # Inyectar credenciales de Telegram desde variables de entorno
    if "telegram" not in raw:
        raw["telegram"] = {}
    raw["telegram"].setdefault("bot_token", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    raw["telegram"].setdefault("chat_id", os.environ.get("TELEGRAM_CHAT_ID", ""))

    # Configurar URL de base de datos (prioridad: YAML > env var > default SQLite)
    raw.setdefault("storage", {})
    default_db = os.environ.get("DATABASE_URL") or "sqlite+aiosqlite:///data/cryptotrader.db"
    if not raw["storage"].get("url"):
        raw["storage"]["url"] = default_db

    # Paso 6: Validar toda la configuracion con Pydantic
    return AppConfig.model_validate(raw)
