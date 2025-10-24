"""OCR endpoints."""
from __future__ import annotations

import asyncio
from typing import Iterable

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.models.schemas import OCRInput, OCRPdfInput, OCRResponse, PdfPagesOptions
from app.services.backends import BackendNotAvailable, BaseOCRBackend, build_backend
from app.utils.http import fetch_bytes, gather_with_concurrency
from app.utils.pdf import (
    PDFConversionError,
    ensure_pdf_support,
    get_total_pages,
    pdf_bytes_to_images,
    resolve_page_range,
)

router = APIRouter(prefix="/ocr", tags=["ocr"])


async def get_backend(settings: Settings = Depends(get_settings)) -> BaseOCRBackend:
    try:
        backend = build_backend(settings)
    except BackendNotAvailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    await backend.warmup()
    return backend


async def _download_images(urls: Iterable[str], semaphore: asyncio.Semaphore) -> list[bytes]:
    return await gather_with_concurrency(semaphore, *(fetch_bytes(url) for url in urls))


@router.post("", response_model=OCRResponse, summary="Run OCR on images")
async def run_ocr(payload: OCRInput, backend: BaseOCRBackend = Depends(get_backend)) -> OCRResponse:
    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.max_concurrency)

    try:
        images = await _download_images((str(url) for url in payload.urls), semaphore)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    results = await backend.infer(images=images, prompt=payload.prompt, mode=payload.mode)
    return OCRResponse(results=results, backend=settings.backend)


@router.post("/pdf", response_model=OCRResponse, summary="Run OCR on each page of a PDF")
async def run_ocr_pdf(payload: OCRPdfInput, backend: BaseOCRBackend = Depends(get_backend)) -> OCRResponse:
    settings = get_settings()
    if not settings.enable_pdf_support:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="PDF support disabled")

    try:
        ensure_pdf_support()
    except PDFConversionError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    try:
        pdf_bytes = await fetch_bytes(str(payload.pdf_url))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    total_pages = get_total_pages(pdf_bytes)
    options = payload.pdf_pages or PdfPagesOptions()
    dpi = options.dpi or settings.pdf_dpi_default
    page_range = resolve_page_range(
        total_pages=total_pages,
        start=options.page_from or 1,
        end=options.page_to,
        max_pages=settings.pdf_max_pages,
    )

    images = list(pdf_bytes_to_images(pdf_bytes, dpi=dpi, page_range=page_range))
    if not images:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No pages to process")

    prompt = payload.prompt or "<image>\n<|grounding|>Convert the document to markdown."
    results = await backend.infer(images=images, prompt=prompt, mode=payload.mode)
    return OCRResponse(results=results, backend=settings.backend)


__all__ = ["router"]
