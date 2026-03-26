"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .auth import setup_auth
from .routers import bot, config, portfolio, signals, trades
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

    return app
