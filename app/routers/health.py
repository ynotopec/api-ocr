"""Health and readiness probes."""
from __future__ import annotations

from fastapi import APIRouter, Response, status

router = APIRouter(tags=["probes"])


@router.get("/healthz", summary="Liveness probe")
async def health() -> Response:
    """Simple liveness endpoint."""

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/readyz", summary="Readiness probe")
async def ready() -> Response:
    """Simple readiness endpoint."""

    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
