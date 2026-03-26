"""Jerarquia de excepciones personalizadas para CryptoTrader.

Este modulo define todas las excepciones especificas del sistema.
Tener una jerarquia propia permite:
- Capturar errores de forma granular (por tipo especifico) o general (CryptoTraderError).
- Distinguir errores del sistema de errores de Python estandar.
- Incluir informacion contextual relevante (ej. regla de riesgo violada).
- Facilitar el logging y las notificaciones automaticas de errores.

Todas las excepciones heredan de CryptoTraderError, lo que permite capturar
cualquier error del sistema con un solo except.
"""


class CryptoTraderError(Exception):
    """Excepcion base para todos los errores de CryptoTrader.

    Todas las excepciones del sistema heredan de esta clase.
    Permite capturar cualquier error especifico del trader con:
        except CryptoTraderError as e: ...
    """


class ConfigError(CryptoTraderError):
    """Configuracion invalida o faltante.

    Se lanza cuando un archivo de configuracion YAML tiene valores
    invalidos, cuando faltan campos obligatorios, o cuando no se puede
    cargar la clase de un modelo AI especificado en la configuracion.
    """


class ExchangeError(CryptoTraderError):
    """Error al comunicarse con el exchange.

    Se lanza cuando hay problemas de conectividad, autenticacion,
    rate limiting, o respuestas inesperadas del exchange (ej. Binance).
    """


class RiskLimitExceeded(CryptoTraderError):
    """Se ha excedido un limite de riesgo.

    Se lanza cuando una operacion propuesta viola alguna regla de gestion
    de riesgo, como el tamano maximo de posicion, la exposicion total,
    o la perdida diaria maxima.

    Atributos:
        rule: Nombre de la regla de riesgo violada (ej. "max_position_pct").
        detail: Descripcion detallada de por que se excedio el limite.
    """

    def __init__(self, rule: str, detail: str = ""):
        self.rule = rule      # Nombre de la regla violada
        self.detail = detail  # Detalle de la violacion
        super().__init__(f"Risk limit exceeded [{rule}]: {detail}")


class CircuitBreakerTripped(CryptoTraderError):
    """El circuit breaker se ha activado, deteniendo todo el trading.

    Este mecanismo de seguridad se dispara cuando se detectan condiciones
    peligrosas como perdidas excesivas o drawdown que supera el limite.
    Mientras esta activo, ninguna orden nueva puede ser enviada al exchange.

    Atributos:
        reason: Razon por la cual se activo el circuit breaker.
    """

    def __init__(self, reason: str):
        self.reason = reason  # Razon de activacion del circuit breaker
        super().__init__(f"Circuit breaker tripped: {reason}")


class OrderError(CryptoTraderError):
    """Error al colocar o gestionar una orden.

    Se lanza cuando hay problemas al enviar una orden al exchange,
    al modificarla o al cancelarla. Puede deberse a parametros invalidos,
    mercado cerrado, o errores internos del exchange.
    """


class InsufficientBalance(CryptoTraderError):
    """Balance insuficiente para ejecutar la operacion.

    Se lanza cuando el saldo disponible en la cuenta del exchange no es
    suficiente para cubrir la cantidad de la orden, incluyendo comisiones.
    """


class StrategyError(CryptoTraderError):
    """Error dentro de una estrategia de trading.

    Se lanza cuando una estrategia falla al procesar datos de mercado,
    calcular indicadores, o generar senales. El sistema captura este error
    para evitar que una estrategia defectuosa afecte al resto.
    """


class StorageError(CryptoTraderError):
    """Fallo en una operacion de base de datos o almacenamiento.

    Se lanza cuando hay problemas al leer o escribir en la base de datos
    (SQLite/PostgreSQL), como errores de conexion, restricciones de integridad,
    o migraciones fallidas.
    """
