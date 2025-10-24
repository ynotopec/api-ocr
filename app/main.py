"""FastAPI application entrypoint."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:  # pragma: no cover - optional dependency
    from prometheus_fastapi_instrumentator import Instrumentator
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Instrumentator = None  # type: ignore

from app.config import get_settings
from app.logging import configure_logging
from app.routers import health, ocr

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(
        title="DeepSeek OCR API",
        version="1.0.0",
        summary="Production ready FastAPI wrapper around the deepseek-ai/DeepSeek-OCR model",
    )

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.cors_allow_origins],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
        )

    app.include_router(health.router)
    app.include_router(ocr.router)

    if settings.metrics_enabled and Instrumentator is not None:
        Instrumentator().instrument(app).expose(app)
    elif settings.metrics_enabled:
        logger.warning("prometheus-fastapi-instrumentator is not installed; /metrics disabled")

    return app


app = create_app()


__all__ = ["app", "create_app"]
