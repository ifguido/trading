"""WebSocket endpoint for real-time data streaming."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.web.dependencies import get_event_bridge

router = APIRouter()


@router.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """WebSocket endpoint that streams real-time events to the client."""
    bridge = get_event_bridge()
    await ws.accept()
    await bridge.add_client(ws)

    try:
        # Keep connection alive by reading (client can send pings)
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await bridge.remove_client(ws)
