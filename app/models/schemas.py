"""Pydantic models shared across the API."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, Field, conint, validator


class PresetSize(str, Enum):
    tiny = "tiny"
    small = "small"
    base = "base"
    large = "large"
    gundam = "gundam"


class OCRMode(BaseModel):
    """Optional inference tuning knobs."""

    preset: Optional[PresetSize] = Field(
        default=None,
        description="Preset describing an optimized configuration shipped with the DeepSeek model.",
    )
    max_new_tokens: Optional[conint(gt=0)] = Field(
        default=None, description="Maximum number of generated tokens returned by the backend."
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, description="Sampling temperature parameter."
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Top-p nucleus sampling value."
    )
    dtype: Optional[str] = Field(default=None, description="Precision override (bf16, fp16, fp32).")


class OCRInput(BaseModel):
    """Payload for the /ocr endpoint."""

    prompt: str = Field(..., description="Multimodal prompt containing <image> tokens.")
    urls: list[AnyHttpUrl] = Field(..., min_items=1, description="List of image URLs to process.")
    mode: Optional[OCRMode] = None

    @validator("prompt")
    def _validate_prompt(cls, value: str) -> str:
        if "<image>" not in value:
            raise ValueError("prompt must contain at least one <image> placeholder")
        return value


class PdfPagesOptions(BaseModel):
    page_from: Optional[int] = Field(1, ge=1)
    page_to: Optional[int] = Field(None, ge=1)
    dpi: Optional[int] = Field(None, ge=72, le=600)

    @validator("page_to")
    def _ensure_range(cls, value: Optional[int], values: dict[str, int | None]) -> Optional[int]:
        if value is not None and values.get("page_from") and value < values["page_from"]:
            raise ValueError("page_to must be greater than or equal to page_from")
        return value


class OCRPdfInput(BaseModel):
    pdf_url: AnyHttpUrl = Field(..., description="Remote PDF to download and convert to images.")
    pdf_pages: Optional[PdfPagesOptions] = None
    mode: Optional[OCRMode] = None
    prompt: Optional[str] = Field(
        default="<image>\n<|grounding|>Convert the document to markdown.",
        description="Prompt injected when hitting the PDF endpoint.",
    )


class OCRResult(BaseModel):
    text: str = Field(..., description="Transcribed markdown text for the image input.")
    duration_ms: Optional[int] = Field(
        default=None,
        description="Latency in milliseconds for the backend inference call.",
    )
    raw: Optional[dict] = Field(default=None, description="Backend specific payload for debugging.")


class OCRResponse(BaseModel):
    results: list[OCRResult] = Field(..., description="Results in the same order as the input URLs.")
    backend: Literal["transformers", "vllm"]


__all__ = [
    "PresetSize",
    "OCRMode",
    "OCRInput",
    "PdfPagesOptions",
    "OCRPdfInput",
    "OCRResult",
    "OCRResponse",
]
