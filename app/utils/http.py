"""HTTP utilities."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx

from app.config import get_settings


async def fetch_bytes(url: str, timeout: Optional[int] = None) -> bytes:
    """Download binary content from an URL."""

    settings = get_settings()
    client_timeout = timeout or settings.request_timeout

    async with httpx.AsyncClient(timeout=client_timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def gather_with_concurrency(semaphore: asyncio.Semaphore, *aws: Any) -> list[Any]:
    """Run asynchronous tasks while respecting a concurrency semaphore."""

    async def _sem_task(aw):
        async with semaphore:
            return await aw

    return await asyncio.gather(*(_sem_task(aw) for aw in aws))


__all__ = ["fetch_bytes", "gather_with_concurrency"]
