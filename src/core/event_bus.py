"""Bus de eventos asincrono basado en el patron publicar-suscribir (pub-sub).

Este modulo implementa el corazon de la comunicacion entre componentes del sistema.
El EventBus permite que cualquier componente publique eventos y que otros se suscriban
a tipos especificos de eventos, sin conocerse mutuamente. Esto desacopla completamente
los productores de datos (feeds, exchange) de los consumidores (estrategias, riesgo).

Patron utilizado: Pub-Sub asincrono con asyncio.
Cada handler se ejecuta de forma concurrente y un error en un handler
no afecta a los demas, garantizando resiliencia.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

from .events import Event

logger = logging.getLogger(__name__)

# Alias de tipo para los handlers asincronos que procesan eventos.
# Cada handler recibe un Event y retorna None (es una corrutina).
Handler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Bus central de eventos asincrono.

    Es el eje de la arquitectura event-driven del sistema. Todos los componentes
    (feeds de datos, estrategias, gestor de riesgo, ejecucion, monitoreo)
    se comunican exclusivamente a traves de este bus.

    Los componentes se suscriben a tipos de evento especificos y reciben
    notificaciones cuando se publica un evento de ese tipo.
    La publicacion puede ser bloqueante (publish) o fire-and-forget (publish_nowait).

    Atributos:
        _subscribers: Diccionario que mapea tipo de evento a lista de (nombre, handler).
        _max_queue_size: Tamano maximo de cola (reservado para uso futuro con colas).
        _running: Indica si el bus esta activo.
        _tasks: Lista de tareas asyncio pendientes creadas por publish_nowait.
    """

    def __init__(self, max_queue_size: int = 10_000) -> None:
        """Inicializa el bus de eventos.

        Parametros:
            max_queue_size: Tamano maximo de la cola de eventos (reservado para
                            implementacion futura con colas por suscriptor).
        """
        # Mapa de tipo de evento -> lista de tuplas (nombre_handler, handler)
        self._subscribers: dict[type[Event], list[tuple[str, Handler]]] = defaultdict(list)
        self._max_queue_size = max_queue_size  # Limite de cola (para uso futuro)
        self._running = False                   # Estado de ejecucion del bus
        self._tasks: list[asyncio.Task] = []    # Tareas pendientes de publish_nowait

    def subscribe(self, event_type: type[Event], handler: Handler, name: str = "") -> None:
        """Registra un handler para un tipo de evento especifico.

        Cuando se publique un evento de tipo `event_type`, el handler sera
        invocado automaticamente. Multiples handlers pueden suscribirse
        al mismo tipo de evento.

        Parametros:
            event_type: Clase del evento al que suscribirse (ej. TickEvent).
            handler: Corrutina asincrona que procesara el evento.
            name: Nombre descriptivo del handler (para logging). Si no se
                  proporciona, se usa el nombre calificado de la funcion.
        """
        # Si no se proporciona nombre, usar el nombre calificado de la funcion
        handler_name = name or getattr(handler, "__qualname__", str(handler))
        self._subscribers[event_type].append((handler_name, handler))
        logger.debug("Subscribed %s to %s", handler_name, event_type.__name__)

    def unsubscribe(self, event_type: type[Event], handler: Handler) -> None:
        """Elimina un handler de un tipo de evento.

        Se usa cuando un componente ya no necesita recibir cierto tipo de evento,
        por ejemplo, al detener una estrategia en caliente.

        Parametros:
            event_type: Clase del evento del que desuscribirse.
            handler: Referencia al handler que se quiere eliminar.
        """
        # Filtrar la lista, eliminando el handler que coincida por identidad
        self._subscribers[event_type] = [
            (n, h) for n, h in self._subscribers[event_type] if h is not handler
        ]

    async def publish(self, event: Event) -> None:
        """Despacha un evento a todos los suscriptores de su tipo (bloqueante).

        Los handlers se ejecutan secuencialmente. Si un handler lanza una
        excepcion, se registra en el log pero los demas handlers continuan
        ejecutandose normalmente. Esto garantiza que un componente con errores
        no bloquee al resto del sistema.

        Parametros:
            event: Instancia del evento a publicar.
        """
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        # Si no hay suscriptores para este tipo, retornar inmediatamente
        if not handlers:
            return

        # Ejecutar cada handler secuencialmente, capturando errores individuales
        for handler_name, handler in handlers:
            try:
                await handler(event)
            except Exception:
                # Registrar el error pero NO propagar la excepcion,
                # para que los demas handlers puedan ejecutarse
                logger.exception(
                    "Handler %s failed processing %s",
                    handler_name,
                    event_type.__name__,
                )

    async def publish_nowait(self, event: Event) -> None:
        """Publica un evento de forma fire-and-forget (no bloqueante).

        Crea tareas asyncio independientes para cada handler, sin esperar
        a que terminen. Util para eventos de alta frecuencia donde la latencia
        es critica (ej. ticks de precio).

        Parametros:
            event: Instancia del evento a publicar.
        """
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        # Si no hay suscriptores, retornar inmediatamente
        if not handlers:
            return

        for handler_name, handler in handlers:
            # Crear una tarea asincrona independiente para cada handler
            task = asyncio.create_task(
                self._safe_call(handler, handler_name, event),
                name=f"{handler_name}:{event_type.__name__}",
            )
            self._tasks.append(task)
            # Limpiar referencia cuando la tarea termine (si el metodo discard existe)
            task.add_done_callback(self._tasks.discard if hasattr(self._tasks, "discard") else lambda t: None)

    async def _safe_call(self, handler: Handler, name: str, event: Event) -> None:
        """Wrapper seguro que ejecuta un handler capturando cualquier excepcion.

        Este metodo existe para que publish_nowait pueda lanzar tareas
        sin riesgo de excepciones no capturadas que cancelarian el loop.

        Parametros:
            handler: Corrutina del handler a ejecutar.
            name: Nombre del handler (para logging de errores).
            event: Evento a procesar.
        """
        try:
            await handler(event)
        except Exception:
            logger.exception("Handler %s failed processing %s", name, type(event).__name__)

    def subscriber_count(self, event_type: type[Event]) -> int:
        """Retorna el numero de suscriptores registrados para un tipo de evento.

        Util para diagnostico y para verificar que los componentes se
        registraron correctamente durante la inicializacion.

        Parametros:
            event_type: Clase del evento a consultar.

        Retorna:
            Numero entero de handlers suscritos a ese tipo de evento.
        """
        return len(self._subscribers.get(event_type, []))

    async def shutdown(self) -> None:
        """Apaga el bus de eventos, cancelando todas las tareas pendientes.

        Se llama durante el apagado graceful del engine. Cancela todas las
        tareas asyncio creadas por publish_nowait y espera a que terminen,
        asegurandose de que no queden corrutinas huerfanas.
        """
        # Cancelar todas las tareas que aun no han terminado
        for task in self._tasks:
            if not task.done():
                task.cancel()
        # Esperar a que todas las tareas terminen (incluyendo las canceladas)
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()  # Limpiar la lista de tareas
        logger.info("EventBus shut down")
