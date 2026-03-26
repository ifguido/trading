"""Ejecutor de ordenes: coloca ordenes en Binance a traves de la API REST de CCXT.

Se suscribe a OrderEvent en el EventBus. Para cada orden entrante:
1. Coloca la orden principal (mercado o limite) usando CCXT.
2. Si el OrderEvent incluye un precio de stop_loss, coloca una orden
   de stop-loss complementaria inmediatamente despues de la ejecucion principal.
3. En caso de exito, publica un FillEvent de vuelta al EventBus.

La logica de reintentos utiliza retroceso exponencial con fluctuacion aleatoria
(jitter) para errores transitorios del exchange. Los IDs de orden del cliente
garantizan idempotencia entre reintentos.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import ccxt
import ccxt.async_support as ccxt_async

from src.core.config_loader import ExchangeConfig
from src.core.event_bus import EventBus
from src.core.events import FillEvent, OrderEvent, OrderType, Side
from src.core.exceptions import ExchangeError, OrderError

logger = logging.getLogger(__name__)

# -- Valores predeterminados para reintentos ---------------------------------

_MAX_RETRIES = 4
_INITIAL_DELAY = 0.5  # segundos de espera inicial antes del primer reintento
_MAX_DELAY = 16.0  # tope maximo de espera entre reintentos (segundos)
_BACKOFF_FACTOR = 2.0  # factor multiplicador para el retroceso exponencial
_JITTER_MAX = 0.5  # fluctuacion aleatoria maxima (hasta 0.5 s)

# Familias de excepciones de CCXT que son seguras para reintentar (errores transitorios)
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    ccxt.NetworkError,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
)


def _to_decimal(value: Any) -> Decimal:
    """Convierte de forma segura un valor del exchange a Decimal.

    Parametros
    ----------
    value :
        Valor a convertir (puede ser None, str, int, float, etc.).

    Retorna
    -------
    Decimal
        El valor convertido, o Decimal(0) si la conversion falla o el
        valor es None.
    """
    if value is None:
        return Decimal(0)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0)


class OrderExecutor:
    """Coloca ordenes en Binance y publica FillEvents.

    Esta clase se encarga de la comunicacion directa con el exchange
    a traves de CCXT asincrono. Gestiona el ciclo de vida de las
    ordenes, incluyendo reintentos con retroceso exponencial y
    colocacion de ordenes stop-loss complementarias.

    Parametros
    ----------
    event_bus :
        EventBus central para suscribirse a OrderEvent y publicar
        FillEvent.
    exchange_config :
        Configuracion de conexion al exchange (claves API, modo sandbox, etc.).
    max_retries :
        Numero maximo de intentos de reintento para fallos transitorios.
    """

    def __init__(
        self,
        event_bus: EventBus,
        exchange_config: ExchangeConfig,
        *,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._event_bus = event_bus  # Bus de eventos central
        self._exchange_config = exchange_config  # Configuracion del exchange
        self._max_retries = max_retries  # Maximo de reintentos permitidos

        self._exchange: ccxt_async.binance | None = None  # Cliente CCXT asincrono (se inicializa en start)
        self._running = False  # Indicador de estado de ejecucion

    # -- Ciclo de vida -------------------------------------------------------

    async def start(self) -> None:
        """Crea el cliente CCXT del exchange y se suscribe a OrderEvent.

        Inicializa la conexion asincrona con Binance, carga los mercados
        disponibles y registra el manejador de eventos de ordenes.
        """
        if self._running:
            logger.warning("OrderExecutor is already running")
            return

        # Crear e inicializar el cliente del exchange
        self._exchange = self._create_exchange()
        await self._exchange.load_markets()

        # Suscribirse a eventos de orden en el bus de eventos
        self._event_bus.subscribe(
            OrderEvent,
            self._handle_order_event,
            name="OrderExecutor",
        )
        self._running = True
        logger.info("OrderExecutor started")

    async def stop(self) -> None:
        """Cancela la suscripcion a eventos y cierra la conexion con el exchange.

        Realiza una limpieza ordenada: primero desuscribe del EventBus,
        luego cierra la conexion CCXT de forma segura.
        """
        if not self._running:
            return

        self._running = False
        # Desuscribirse del bus de eventos
        self._event_bus.unsubscribe(OrderEvent, self._handle_order_event)

        # Cerrar la conexion con el exchange de forma segura
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception:
                logger.exception("Error closing CCXT exchange")
            self._exchange = None

        logger.info("OrderExecutor stopped")

    # -- Fabrica del Exchange ------------------------------------------------

    def _create_exchange(self) -> ccxt_async.binance:
        """Instancia un cliente REST asincrono de Binance via CCXT.

        Retorna
        -------
        ccxt_async.binance
            Cliente de Binance configurado con las claves API, modo
            sandbox y opciones proporcionadas en la configuracion.
        """
        # Construir el diccionario de configuracion para CCXT
        config: dict[str, Any] = {
            "apiKey": self._exchange_config.api_key or None,
            "secret": self._exchange_config.api_secret or None,
            "enableRateLimit": self._exchange_config.rate_limit,
            "options": {
                **self._exchange_config.options,
                "defaultType": "spot",  # Tipo de mercado predeterminado: spot
            },
        }
        # Activar modo sandbox si esta configurado
        if self._exchange_config.sandbox:
            config["sandbox"] = True

        exchange = ccxt_async.binance(config)
        logger.debug(
            "Created async CCXT Binance REST client (sandbox=%s)",
            self._exchange_config.sandbox,
        )
        return exchange

    # -- Manejador de Eventos ------------------------------------------------

    async def _handle_order_event(self, event: OrderEvent) -> None:
        """Procesa un OrderEvent entrante: coloca la orden en el exchange.

        Flujo principal:
        1. Coloca la orden con logica de reintentos.
        2. Construye y publica un FillEvent con los detalles de la ejecucion.
        3. Si el evento incluye stop_loss, coloca una orden stop-loss complementaria.

        Parametros
        ----------
        event :
            Evento de orden con los detalles (simbolo, lado, cantidad, precio, etc.).

        Lanza
        -----
        OrderError
            Si la ejecucion de la orden falla despues de agotar los reintentos.
        """
        if not self._running:
            logger.warning("OrderExecutor not running; ignoring OrderEvent %s", event.event_id)
            return

        logger.info(
            "Processing OrderEvent: %s %s %s qty=%s price=%s client_id=%s",
            event.order_type.value,
            event.side.value,
            event.symbol,
            event.quantity,
            event.price,
            event.client_order_id,
        )

        try:
            # Paso 1: Colocar la orden con reintentos
            result = await self._place_order_with_retry(event)
            # Paso 2: Construir y publicar el evento de ejecucion (fill)
            fill_event = self._build_fill_event(event, result)
            await self._event_bus.publish(fill_event)

            logger.info(
                "Order filled: %s %s %s qty=%s @ %s (exchange_id=%s)",
                event.side.value,
                event.symbol,
                event.order_type.value,
                fill_event.quantity,
                fill_event.price,
                fill_event.exchange_order_id,
            )

            # Paso 3: Colocar stop-loss complementario si fue especificado
            if event.stop_loss is not None:
                await self._place_stop_loss(event)

        except OrderError:
            raise
        except Exception as exc:
            logger.exception(
                "Failed to execute order %s: %s",
                event.client_order_id,
                exc,
            )
            raise OrderError(
                f"Order execution failed for {event.client_order_id}: {exc}"
            ) from exc

    # -- Colocacion de Ordenes -----------------------------------------------

    async def _place_order_with_retry(
        self, event: OrderEvent
    ) -> dict[str, Any]:
        """Coloca una orden con retroceso exponencial y jitter en reintentos.

        Utiliza el client_order_id para idempotencia: incluso si un intento
        anterior agoto el tiempo de espera pero realmente tuvo exito en el
        exchange, el reintento sera rechazado como duplicado, y en su lugar
        se obtiene la orden existente.

        Parametros
        ----------
        event :
            Evento de orden con los detalles de la orden a colocar.

        Retorna
        -------
        dict[str, Any]
            Respuesta del exchange con los detalles de la orden ejecutada.

        Lanza
        -----
        OrderError
            Si la orden es invalida (parametros incorrectos, saldo insuficiente).
        ExchangeError
            Si se agotan los reintentos para errores transitorios.
        """
        delay = _INITIAL_DELAY  # Retardo inicial entre reintentos

        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._place_order(event)

            except ccxt.InvalidOrder as exc:
                # Error no transitorio: parametros invalidos, saldo insuficiente, etc.
                logger.error(
                    "Invalid order (attempt %d/%d) %s: %s",
                    attempt,
                    self._max_retries,
                    event.client_order_id,
                    exc,
                )
                raise OrderError(str(exc)) from exc

            except ccxt.OrderNotFound:
                # Caso limite durante reintento idempotente: buscar la orden existente
                logger.warning(
                    "Order not found during retry for %s; "
                    "fetching by client_order_id",
                    event.client_order_id,
                )
                return await self._fetch_order_by_client_id(
                    event.symbol, event.client_order_id
                )

            except _RETRYABLE_ERRORS as exc:
                # Error transitorio: verificar si se agotaron los reintentos
                if attempt == self._max_retries:
                    logger.error(
                        "Max retries (%d) exhausted for order %s: %s",
                        self._max_retries,
                        event.client_order_id,
                        exc,
                    )
                    raise ExchangeError(
                        f"Order placement failed after {self._max_retries} "
                        f"retries: {exc}"
                    ) from exc

                # Calcular tiempo de espera con jitter aleatorio
                jitter = random.uniform(0, _JITTER_MAX)
                sleep_time = delay + jitter
                logger.warning(
                    "Transient error (attempt %d/%d) for order %s: %s "
                    "— retrying in %.2fs",
                    attempt,
                    self._max_retries,
                    event.client_order_id,
                    exc,
                    sleep_time,
                )
                await asyncio.sleep(sleep_time)
                # Incrementar el retardo con retroceso exponencial, hasta el maximo
                delay = min(delay * _BACKOFF_FACTOR, _MAX_DELAY)

        # No deberia llegar aqui nunca, pero por precaucion
        raise OrderError(f"Unexpected retry loop exit for {event.client_order_id}")

    async def _place_order(self, event: OrderEvent) -> dict[str, Any]:
        """Coloca una orden individual en el exchange via CCXT.

        Soporta ordenes de tipo: MARKET, LIMIT, STOP_LOSS y STOP_LOSS_LIMIT.

        Parametros
        ----------
        event :
            Evento de orden con tipo, lado, simbolo, cantidad y precio.

        Retorna
        -------
        dict[str, Any]
            Respuesta cruda del exchange normalizada por CCXT.

        Lanza
        -----
        OrderError
            Si el tipo de orden no es soportado o faltan parametros requeridos.
        """
        assert self._exchange is not None

        # Parametros adicionales con el ID de orden del cliente para idempotencia
        params: dict[str, Any] = {
            "newClientOrderId": event.client_order_id,
        }

        if event.order_type == OrderType.MARKET:
            # Orden de mercado: se ejecuta inmediatamente al mejor precio disponible
            result = await self._exchange.create_order(
                symbol=event.symbol,
                type="market",
                side=event.side.value,
                amount=float(event.quantity),
                params=params,
            )

        elif event.order_type == OrderType.LIMIT:
            # Orden limite: requiere un precio especifico
            if event.price is None:
                raise OrderError("Limit order requires a price")
            result = await self._exchange.create_order(
                symbol=event.symbol,
                type="limit",
                side=event.side.value,
                amount=float(event.quantity),
                price=float(event.price),
                params=params,
            )

        elif event.order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT):
            # Orden stop-loss o stop-loss-limit: delegar a metodo especializado
            result = await self._place_stop_loss_order(
                symbol=event.symbol,
                side=event.side,
                quantity=event.quantity,
                stop_price=event.price,
                limit_price=event.price if event.order_type == OrderType.STOP_LOSS_LIMIT else None,
                client_order_id=event.client_order_id,
            )

        else:
            raise OrderError(f"Unsupported order type: {event.order_type}")

        return result

    async def _place_stop_loss(self, event: OrderEvent) -> None:
        """Coloca una orden stop-loss complementaria despues de una ejecucion principal.

        El lado del stop-loss es opuesto al lado de la orden original
        (si la orden original fue BUY, el stop-loss sera SELL y viceversa).

        Parametros
        ----------
        event :
            Evento de la orden original que contiene el precio de stop_loss.
        """
        assert event.stop_loss is not None

        # El stop-loss siempre es del lado contrario a la orden original
        stop_side = Side.SELL if event.side == Side.BUY else Side.BUY
        # Prefijo "sl-" para identificar ordenes stop-loss complementarias
        stop_client_id = f"sl-{event.client_order_id}"

        logger.info(
            "Placing stop-loss: %s %s qty=%s stop_price=%s client_id=%s",
            stop_side.value,
            event.symbol,
            event.quantity,
            event.stop_loss,
            stop_client_id,
        )

        try:
            result = await self._retry_wrapper(
                self._place_stop_loss_order,
                symbol=event.symbol,
                side=stop_side,
                quantity=event.quantity,
                stop_price=event.stop_loss,
                limit_price=None,
                client_order_id=stop_client_id,
            )
            logger.info(
                "Stop-loss placed: exchange_id=%s",
                result.get("id", "unknown"),
            )
        except Exception as exc:
            # Se registra el error pero NO se relanza: la orden principal ya fue ejecutada.
            # El order manager rastreara y manejara esto por separado.
            logger.error(
                "Failed to place stop-loss for %s: %s",
                event.client_order_id,
                exc,
            )

    async def _place_stop_loss_order(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: Decimal,
        stop_price: Decimal | None,
        limit_price: Decimal | None = None,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Coloca una orden stop-loss (o stop-loss-limit) en el exchange.

        Parametros
        ----------
        symbol :
            Par de trading (ej. "BTC/USDT").
        side :
            Lado de la orden (BUY o SELL).
        quantity :
            Cantidad a operar.
        stop_price :
            Precio de activacion del stop-loss.
        limit_price :
            Precio limite para ordenes stop-loss-limit. Si es None, se
            coloca una orden stop-loss de mercado.
        client_order_id :
            ID unico del cliente para idempotencia.

        Retorna
        -------
        dict[str, Any]
            Respuesta del exchange con los detalles de la orden.

        Lanza
        -----
        OrderError
            Si no se proporciona un precio de stop.
        """
        assert self._exchange is not None

        if stop_price is None:
            raise OrderError("Stop-loss order requires a stop price")

        # Configurar parametros con el precio de activacion
        params: dict[str, Any] = {
            "newClientOrderId": client_order_id,
            "stopPrice": float(stop_price),
        }

        if limit_price is not None:
            # Orden stop-loss con precio limite
            order_type = "STOP_LOSS_LIMIT"
            params["timeInForce"] = "GTC"  # Good-Till-Cancelled: activa hasta que se cancele
            result = await self._exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side.value,
                amount=float(quantity),
                price=float(limit_price),
                params=params,
            )
        else:
            # Orden stop-loss de mercado (se ejecuta al mejor precio disponible al activarse)
            order_type = "STOP_LOSS"
            result = await self._exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side.value,
                amount=float(quantity),
                params=params,
            )

        return result

    async def _retry_wrapper(
        self, func, *args, **kwargs
    ) -> dict[str, Any]:
        """Envoltorio generico de reintentos con retroceso exponencial y jitter.

        Parametros
        ----------
        func :
            Funcion asincrona a ejecutar con reintentos.
        *args :
            Argumentos posicionales para la funcion.
        **kwargs :
            Argumentos con nombre para la funcion.

        Retorna
        -------
        dict[str, Any]
            Resultado de la funcion ejecutada exitosamente.

        Lanza
        -----
        OrderError
            Si la orden es invalida (error no transitorio).
        ExchangeError
            Si se agotan todos los reintentos.
        """
        delay = _INITIAL_DELAY  # Retardo inicial

        for attempt in range(1, self._max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except ccxt.InvalidOrder as exc:
                # Error no transitorio: no reintentar
                raise OrderError(str(exc)) from exc

            except _RETRYABLE_ERRORS as exc:
                # Error transitorio: verificar si quedan reintentos
                if attempt == self._max_retries:
                    raise ExchangeError(
                        f"Operation failed after {self._max_retries} retries: {exc}"
                    ) from exc

                # Calcular tiempo de espera con jitter para evitar tormentas de reintentos
                jitter = random.uniform(0, _JITTER_MAX)
                sleep_time = delay + jitter
                logger.warning(
                    "Transient error (attempt %d/%d): %s — retrying in %.2fs",
                    attempt,
                    self._max_retries,
                    exc,
                    sleep_time,
                )
                await asyncio.sleep(sleep_time)
                # Aplicar retroceso exponencial con tope maximo
                delay = min(delay * _BACKOFF_FACTOR, _MAX_DELAY)

        raise OrderError("Unexpected retry loop exit")

    # -- Metodos Auxiliares --------------------------------------------------

    async def _fetch_order_by_client_id(
        self, symbol: str, client_order_id: str
    ) -> dict[str, Any]:
        """Obtiene una orden del exchange usando el ID de orden del cliente.

        Se utiliza durante reintentos idempotentes cuando un intento previo
        pudo haber tenido exito pero no recibimos la respuesta.

        Parametros
        ----------
        symbol :
            Par de trading (ej. "BTC/USDT").
        client_order_id :
            ID unico del cliente asignado a la orden.

        Retorna
        -------
        dict[str, Any]
            Datos de la orden obtenidos del exchange.

        Lanza
        -----
        OrderError
            Si la orden no se puede encontrar en el exchange.
        """
        assert self._exchange is not None

        try:
            # Buscar la orden en el exchange usando el ID original del cliente
            orders = await self._exchange.fetch_orders(
                symbol=symbol,
                params={"origClientOrderId": client_order_id},
            )
            if orders:
                return orders[0]  # Retornar la primera coincidencia
        except Exception as exc:
            logger.error(
                "Failed to fetch order by client_id %s: %s",
                client_order_id,
                exc,
            )

        raise OrderError(
            f"Could not find order with client_order_id={client_order_id}"
        )

    def _build_fill_event(
        self, order_event: OrderEvent, exchange_result: dict[str, Any]
    ) -> FillEvent:
        """Construye un FillEvent a partir de la respuesta del exchange.

        Extrae los detalles de la ejecucion (cantidad ejecutada, precio
        promedio, comisiones) de la respuesta normalizada de CCXT.

        Parametros
        ----------
        order_event :
            Evento de orden original que genero esta ejecucion.
        exchange_result :
            Respuesta del exchange normalizada por CCXT.

        Retorna
        -------
        FillEvent
            Evento de ejecucion listo para publicar en el EventBus.
        """
        # CCXT normaliza la respuesta; extraer detalles de la ejecucion
        filled_qty = _to_decimal(exchange_result.get("filled"))  # Cantidad ejecutada
        avg_price = _to_decimal(exchange_result.get("average"))  # Precio promedio de ejecucion
        fee_info: dict[str, Any] = exchange_result.get("fee") or {}  # Informacion de comisiones

        return FillEvent(
            symbol=order_event.symbol,
            side=order_event.side,
            quantity=filled_qty if filled_qty > 0 else order_event.quantity,  # Usar cantidad original si no hay datos de ejecucion
            price=avg_price,
            fee=_to_decimal(fee_info.get("cost")),  # Costo de la comision
            fee_currency=fee_info.get("currency", ""),  # Moneda de la comision
            exchange_order_id=exchange_result.get("id", ""),  # ID asignado por el exchange
            client_order_id=order_event.client_order_id,
            strategy_name=order_event.strategy_name,
        )

    async def cancel_order(
        self, exchange_order_id: str, symbol: str
    ) -> dict[str, Any]:
        """Cancela una orden en el exchange usando su ID del exchange.

        Metodo publico expuesto para uso del OrderManager durante el
        cierre ordenado del sistema.

        Parametros
        ----------
        exchange_order_id :
            ID de la orden asignado por el exchange.
        symbol :
            Par de trading (ej. "BTC/USDT").

        Retorna
        -------
        dict[str, Any]
            Resultado de la cancelacion o estado "not_found" si ya fue ejecutada.

        Lanza
        -----
        ExchangeError
            Si la cancelacion falla por un error del exchange.
        """
        assert self._exchange is not None

        try:
            result = await self._retry_wrapper(
                self._exchange.cancel_order, exchange_order_id, symbol
            )
            logger.info("Cancelled order %s on %s", exchange_order_id, symbol)
            return result
        except ccxt.OrderNotFound:
            # La orden no existe: posiblemente ya fue ejecutada o cancelada previamente
            logger.warning(
                "Order %s not found for cancellation (may already be filled)",
                exchange_order_id,
            )
            return {"id": exchange_order_id, "status": "not_found"}
        except Exception as exc:
            logger.exception("Failed to cancel order %s: %s", exchange_order_id, exc)
            raise ExchangeError(
                f"Failed to cancel order {exchange_order_id}: {exc}"
            ) from exc

    async def fetch_order_status(
        self, exchange_order_id: str, symbol: str
    ) -> dict[str, Any]:
        """Obtiene el estado actual de una orden desde el exchange.

        Metodo publico expuesto para uso del OrderManager durante la
        sincronizacion periodica de ordenes.

        Parametros
        ----------
        exchange_order_id :
            ID de la orden asignado por el exchange.
        symbol :
            Par de trading (ej. "BTC/USDT").

        Retorna
        -------
        dict[str, Any]
            Estado actual de la orden segun el exchange.
        """
        assert self._exchange is not None

        return await self._retry_wrapper(
            self._exchange.fetch_order, exchange_order_id, symbol
        )

    @property
    def is_running(self) -> bool:
        """Indica si el ejecutor de ordenes esta activo y procesando eventos."""
        return self._running
