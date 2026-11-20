"""Small security helpers shared by the FastAPI application and routes."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request

from config import get_env


def allowed_hosts() -> list[str]:
    """Return configured hostnames, with safe local-development defaults."""
    configured = get_env("ALLOWED_HOSTS")
    if configured:
        return [host.strip() for host in configured.split(",") if host.strip()]
    return ["localhost", "127.0.0.1", "testserver"]


def require_admin(request: Request) -> None:
    """Require a configured bearer token for operational endpoints."""
    expected = get_env("ADMIN_API_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Administrative endpoints are not configured",
        )

    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not supplied or not secrets.compare_digest(
        supplied.strip(), expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Administrative authorization required",
            headers={"WWW-Authenticate": "Bearer"},
        )
