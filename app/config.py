"""Application configuration settings for the DeepSeek OCR API."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseSettings, Field, PositiveInt, validator


class Settings(BaseSettings):
    """Environment driven configuration."""

    backend: Literal["transformers", "vllm"] = Field(
        "transformers", env="BACKEND", description="Inference backend implementation to use."
    )
    model_id: str = Field(
        "deepseek-ai/DeepSeek-OCR",
        env="MODEL_ID",
        description="Hugging Face model identifier to load.",
    )
    model_revision: Optional[str] = Field(
        default=None,
        env="MODEL_REVISION",
        description="Optional model revision or commit hash.",
    )
    hf_token: Optional[str] = Field(
        default=None,
        env="HF_TOKEN",
        description="Optional Hugging Face access token for private models.",
    )
    request_timeout: PositiveInt = Field(
        60,
        env="REQUEST_TIMEOUT",
        description="Timeout in seconds for outbound HTTP requests when downloading documents.",
    )
    max_concurrency: PositiveInt = Field(
        2,
        env="MAX_CONCURRENCY",
        description="Maximum number of in-flight inference requests handled concurrently.",
    )
    metrics_enabled: bool = Field(
        True,
        env="METRICS_ENABLED",
        description="Toggle Prometheus metrics endpoint instrumentation.",
    )
    torch_dtype: str = Field(
        "bf16",
        env="TORCH_DTYPE",
        description="Torch dtype passed to the backend loader (bf16, fp16, fp32, ...).",
    )
    device_map: str = Field(
        "auto",
        env="DEVICE_MAP",
        description="Device mapping strategy for the transformers backend.",
    )
    trust_remote_code: bool = Field(
        True,
        env="TRUST_REMOTE_CODE",
        description="Allow execution of custom code when loading the model from Hugging Face.",
    )
    vllm_tensor_parallel_size: PositiveInt = Field(
        1,
        env="VLLM_TENSOR_PARALLEL_SIZE",
        description="Tensor parallelism degree when using the vLLM backend.",
    )
    vllm_gpu_memory_utilization: float = Field(
        0.9,
        env="VLLM_GPU_MEMORY_UTILIZATION",
        description="GPU memory utilization ratio for vLLM.",
    )
    enable_pdf_support: bool = Field(
        True,
        env="ENABLE_PDF_SUPPORT",
        description="Enable PDF to image conversion. Requires pypdfium2 runtime dependency.",
    )
    pdf_dpi_default: PositiveInt = Field(
        180,
        env="PDF_DPI_DEFAULT",
        description="Default DPI used for PDF rasterization when not provided by the client.",
    )
    pdf_max_pages: PositiveInt = Field(
        20,
        env="PDF_MAX_PAGES",
        description="Hard limit for number of pages processed per PDF request.",
    )
    cors_allow_origins: list[str] = Field(
        default_factory=list,
        env="CORS_ALLOW_ORIGINS",
        description="List of origins allowed to perform cross-origin requests.",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

        @classmethod
        def parse_env_var(cls, field_name: str, raw_value: str | None):  # type: ignore[override]
            if field_name == "cors_allow_origins" and isinstance(raw_value, str):
                # Defer parsing to validators so comma-separated lists are supported.
                return raw_value
            return BaseSettings.Config.parse_env_var(field_name, raw_value)

    @validator("cors_allow_origins", pre=True)
    def _split_origins(cls, value: str | list[str] | None) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()


__all__ = ["Settings", "get_settings"]
