from __future__ import annotations

import asyncio

import pytest

from app.utils.http import gather_with_concurrency


@pytest.mark.asyncio
async def test_gather_with_concurrency_limits_parallelism(monkeypatch):
    counter = 0
    max_parallel = 0
    lock = asyncio.Lock()

    async def worker(duration: float):
        nonlocal counter, max_parallel
        async with lock:
            counter += 1
            max_parallel = max(max_parallel, counter)
        await asyncio.sleep(duration)
        async with lock:
            counter -= 1
        return duration

    semaphore = asyncio.Semaphore(2)
    results = await gather_with_concurrency(
        semaphore,
        *(worker(0.01) for _ in range(5)),
    )
    assert results == [0.01] * 5
    assert max_parallel <= 2
