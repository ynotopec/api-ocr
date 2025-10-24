"""PDF utilities for rasterisation."""
from __future__ import annotations

import contextlib
import io
from typing import Iterable

try:  # pragma: no cover - optional dependency import guard
    import pypdfium2
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pypdfium2 = None  # type: ignore


class PDFConversionError(RuntimeError):
    """Raised when PDF to image conversion cannot be completed."""


def ensure_pdf_support() -> None:
    """Ensure that the optional PDF dependency is installed."""

    if pypdfium2 is None:
        raise PDFConversionError(
            "pypdfium2 is not installed. Install it or disable PDF support via ENABLE_PDF_SUPPORT=0."
        )


def resolve_page_range(total_pages: int, start: int, end: int | None, max_pages: int) -> range:
    """Compute the range of pages to render while applying safety limits."""

    actual_end = end or total_pages
    if actual_end < start:
        actual_end = start
    if actual_end - start + 1 > max_pages:
        actual_end = start + max_pages - 1
    actual_end = min(actual_end, total_pages)
    return range(start - 1, actual_end)


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int, page_range: range) -> Iterable[bytes]:
    """Convert PDF bytes to an iterator of PNG bytes."""

    ensure_pdf_support()
    assert pypdfium2 is not None

    pdf = pypdfium2.PdfDocument(io.BytesIO(pdf_bytes))
    try:
        for page_index in page_range:
            if page_index >= len(pdf):
                break
            page = pdf.get_page(page_index)
            with contextlib.closing(page) as page_obj:
                pil_image = page_obj.render_topil(scale=dpi / 72)
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                yield buffer.getvalue()
    finally:
        pdf.close()


def get_total_pages(pdf_bytes: bytes) -> int:
    """Return the total number of pages contained in the PDF."""

    ensure_pdf_support()
    assert pypdfium2 is not None
    pdf = pypdfium2.PdfDocument(io.BytesIO(pdf_bytes))
    try:
        return len(pdf)
    finally:
        pdf.close()


__all__ = [
    "PDFConversionError",
    "ensure_pdf_support",
    "resolve_page_range",
    "pdf_bytes_to_images",
    "get_total_pages",
]
