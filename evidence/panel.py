"""Isolated development/test HTTP harness. Production uses AstrBot Pages only."""

from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path

from aiohttp import web

from .admin import AdminAPI


def load_token(path: Path) -> str:
    if path.exists():
        token = path.read_text().strip()
        if len(token) < 32:
            raise ValueError("Invalid development token")
        path.chmod(0o600)
        return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(36)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    return token


def create_app(store, engine, token: str, static_dir: Path, providers=None):
    if len(token) < 32:
        raise ValueError("Development token too short")
    admin = AdminAPI(store, engine, providers)

    @web.middleware
    async def guard(request, handler):
        if request.path.startswith("/api/"):
            if not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer " + token):
                return web.json_response({"error": "Unauthorized"}, status=401)
            if request.headers.get("Sec-Fetch-Site") == "cross-site":
                return web.json_response({"error": "Cross-site request denied"}, status=403)
        response = await handler(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
            }
        )
        return response

    app = web.Application(middlewares=[guard], client_max_size=64 * 1024)

    async def operation(request):
        try:
            body = await request.json() if request.can_read_body else {}
        except (ValueError, TypeError):
            return web.json_response({"error": "Invalid JSON"}, status=400)
        envelope = (
            body
            if request.path == "/api/bridge"
            else {
                "path": str(request.rel_url)[5:],
                "method": request.method,
                "body": body,
            }
        )
        data, status = await admin.handle(envelope)
        return web.json_response(data, status=status)

    async def static(request):
        path = (static_dir / (request.match_info.get("path", "") or "index.html")).resolve()
        if not path.is_relative_to(static_dir.resolve()) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    app.router.add_route("*", "/api/{path:.*}", operation)
    app.router.add_get("/{path:.*}", static)
    return app
