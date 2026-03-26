"""Authentication middleware with brute-force protection.

Credentials are read from environment variables WEB_USERNAME and
WEB_PASSWORD_HASH. After a failed login attempt, the IP is locked
out for 24 hours. Sessions are cookie-based with a random token.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time

from dotenv import load_dotenv

load_dotenv()
from typing import Any

from fastapi import Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_USERNAME = os.environ.get("WEB_USERNAME", "admin")
_PASSWORD_HASH = os.environ.get("WEB_PASSWORD_HASH", "")

_LOCKOUT_SECONDS = 86400  # 24 hours

# IP -> timestamp of last failed attempt
_failed_attempts: dict[str, float] = {}

# Active session tokens
_sessions: set[str] = set()

# Paths that don't require auth
_PUBLIC_PATHS = {"/login"}


def _check_password(password: str) -> bool:
    return hashlib.sha256(password.encode()).hexdigest() == _PASSWORD_HASH


def _is_locked_out(ip: str) -> bool:
    last_fail = _failed_attempts.get(ip)
    if last_fail is None:
        return False
    return (time.time() - last_fail) < _LOCKOUT_SECONDS


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


LOGIN_PAGE = """
<!DOCTYPE html>
<html lang="en" class="h-full bg-gray-950">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - CryptoTrader</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: { extend: { colors: { brand: { 500: '#6366f1', 600: '#4f46e5' } } } }
        }
    </script>
</head>
<body class="h-full dark text-gray-100 flex items-center justify-center">
    <div class="bg-gray-900 border border-gray-800 rounded-lg p-8 w-full max-w-sm">
        <h1 class="text-xl font-bold text-center mb-6">CryptoTrader</h1>
        <!-- ERROR_PLACEHOLDER -->
        <form method="POST" action="/login" class="space-y-4">
            <div>
                <label class="block text-sm text-gray-400 mb-1">Username</label>
                <input type="text" name="username" required autocomplete="username"
                    class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-brand-500">
            </div>
            <div>
                <label class="block text-sm text-gray-400 mb-1">Password</label>
                <input type="password" name="password" required autocomplete="current-password"
                    class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-brand-500">
            </div>
            <button type="submit"
                class="w-full bg-brand-600 hover:bg-brand-500 text-white py-2 rounded text-sm font-medium transition">
                Login
            </button>
        </form>
    </div>
</body>
</html>
"""


def _render_login(error: str = "") -> HTMLResponse:
    error_html = ""
    if error:
        error_html = f'<p class="text-red-400 text-sm text-center mb-4">{error}</p>'
    return HTMLResponse(LOGIN_PAGE.replace("<!-- ERROR_PLACEHOLDER -->", error_html))


def setup_auth(app: Any) -> None:
    """Add login/logout routes and auth middleware to the FastAPI app."""
    from fastapi import Form

    @app.post("/login")
    async def login(request: Request, username: str = Form(...), password: str = Form(...)):
        ip = _get_client_ip(request)

        if _is_locked_out(ip):
            logger.warning("Locked out IP attempted login: %s", ip)
            return _render_login("Too many failed attempts. Try again in 24 hours.")

        if username == _USERNAME and _check_password(password):
            # Success — clear any failed attempt record
            _failed_attempts.pop(ip, None)
            token = secrets.token_hex(32)
            _sessions.add(token)
            response = RedirectResponse(url="/", status_code=303)
            response.set_cookie(
                key="session",
                value=token,
                httponly=True,
                max_age=86400 * 7,  # 7 days
                samesite="lax",
            )
            logger.info("Successful login from %s", ip)
            return response

        # Failed — lock out this IP
        _failed_attempts[ip] = time.time()
        logger.warning("Failed login attempt from %s", ip)
        return _render_login("Invalid credentials. This IP is now locked for 24 hours.")

    @app.get("/login")
    async def login_page(request: Request):
        ip = _get_client_ip(request)
        if _is_locked_out(ip):
            return _render_login("Too many failed attempts. Try again in 24 hours.")
        return _render_login()

    @app.get("/logout")
    async def logout(request: Request):
        token = request.cookies.get("session")
        if token:
            _sessions.discard(token)
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie("session")
        return response

    # Middleware to protect all routes
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path

        # Allow public paths, static files, and public API
        if path in _PUBLIC_PATHS or path.startswith("/static") or path == "/live" or path.startswith("/api/public"):
            return await call_next(request)

        # Check session cookie
        token = request.cookies.get("session")
        if token and token in _sessions:
            return await call_next(request)

        # Not authenticated — redirect to login for pages, 401 for API/WS
        if path.startswith("/api") or path.startswith("/ws"):
            return Response(status_code=401, content="Unauthorized")

        return RedirectResponse(url="/login", status_code=303)
