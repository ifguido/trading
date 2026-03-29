"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

from .auth import setup_auth
from .routers import bot, config, portfolio, public, signals, trades, tuning
from .ws import router as ws_router

_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="CryptoTrader", version="0.1.0")

    # Setup authentication (must be before routes)
    setup_auth(app)

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Setup Jinja2 templates
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # Include API routers
    app.include_router(config.router)
    app.include_router(bot.router)
    app.include_router(portfolio.router)
    app.include_router(trades.router)
    app.include_router(signals.router)
    app.include_router(ws_router)
    app.include_router(public.router)
    app.include_router(tuning.router)

    # HTML page routes
    @app.get("/")
    async def dashboard_page(request: Request):
        return templates.TemplateResponse(request, "dashboard.html")

    @app.get("/setup")
    async def setup_page(request: Request):
        return templates.TemplateResponse(request, "setup.html")

    @app.get("/trades")
    async def trades_page(request: Request):
        return templates.TemplateResponse(request, "trades.html")

    @app.get("/signals")
    async def signals_page(request: Request):
        return templates.TemplateResponse(request, "signals.html")

    @app.get("/tuning")
    async def tuning_page(request: Request):
        return templates.TemplateResponse(request, "tuning.html")

    @app.get("/live")
    async def public_page(request: Request):
        return templates.TemplateResponse(request, "public.html")

    # Auto-start bot if AUTO_START=true
    @app.on_event("startup")
    async def auto_start_bot():
        if os.environ.get("AUTO_START", "").lower() in ("true", "1", "yes"):
            from .dependencies import get_engine_manager, get_event_bridge
            from src.core.config_loader import load_config

            logger.info("Auto-starting bot...")
            try:
                config = load_config()
                mgr = get_engine_manager()
                bridge = get_event_bridge()
                await mgr.start(config)
                if mgr.engine:
                    bridge.subscribe_to_engine(mgr.engine)
                logger.info("Bot auto-started successfully")
            except Exception as e:
                logger.error("Auto-start failed: %s", e)

    return app
