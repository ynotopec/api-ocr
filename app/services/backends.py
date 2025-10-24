"""Inference backend implementations."""
from __future__ import annotations

import asyncio
import importlib
import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.models.schemas import OCRMode, OCRResult


class BackendNotAvailable(RuntimeError):
    """Raised when a requested backend cannot be used."""


@dataclass
class BackendMetadata:
    name: str
    model_id: str
    revision: str | None


class BaseOCRBackend:
    """Interface implemented by OCR backends."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.metadata = BackendMetadata(
            name=self.__class__.__name__.removesuffix("Backend").lower(),
            model_id=settings.model_id,
            revision=settings.model_revision,
        )

    async def warmup(self) -> None:
        """Load model weights. Default implementation does nothing."""

    async def infer(self, images: list[bytes], prompt: str, mode: OCRMode | None = None) -> list[OCRResult]:
        raise NotImplementedError


class TransformersOCRBackend(BaseOCRBackend):
    """Implementation backed by Hugging Face Transformers."""

    _pipeline: Any | None = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        if importlib.util.find_spec("transformers") is None:  # pragma: no cover - runtime guard
            raise BackendNotAvailable(
                "transformers is not available. Install it or switch BACKEND=vllm."
            )

        from transformers import AutoProcessor, AutoTokenizer, pipeline

        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
            raise BackendNotAvailable("torch is not installed") from exc

        dtype = getattr(torch, self.settings.torch_dtype, None)
        if dtype is None:  # pragma: no cover - user misconfiguration
            raise BackendNotAvailable(f"Unsupported torch dtype: {self.settings.torch_dtype}")

        processor = AutoProcessor.from_pretrained(
            self.settings.model_id,
            revision=self.settings.model_revision,
            token=self.settings.hf_token,
            trust_remote_code=self.settings.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_id,
            revision=self.settings.model_revision,
            token=self.settings.hf_token,
            trust_remote_code=self.settings.trust_remote_code,
        )

        self._pipeline = pipeline(
            "image-to-text",
            model=self.settings.model_id,
            revision=self.settings.model_revision,
            token=self.settings.hf_token,
            torch_dtype=dtype,
            device_map=self.settings.device_map,
            trust_remote_code=self.settings.trust_remote_code,
            processor=processor,
            tokenizer=tokenizer,
        )
        return self._pipeline

    async def warmup(self) -> None:  # pragma: no cover - expensive operation
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_pipeline)

    async def infer(self, images: list[bytes], prompt: str, mode: OCRMode | None = None) -> list[OCRResult]:
        pipeline = self._load_pipeline()
        loop = asyncio.get_running_loop()

        def _run_inference(image_bytes: bytes) -> OCRResult:
            start = time.perf_counter()
            result = pipeline(
                image_bytes,
                prompt=prompt,
                generate_kwargs={
                    key: value
                    for key, value in {
                        "max_new_tokens": mode.max_new_tokens if mode else None,
                        "temperature": mode.temperature if mode else None,
                        "top_p": mode.top_p if mode else None,
                    }.items()
                    if value is not None
                },
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            text = result[0]["generated_text"] if result else ""
            return OCRResult(text=text, duration_ms=elapsed_ms, raw={"backend": "transformers"})

        tasks = [loop.run_in_executor(None, _run_inference, image) for image in images]
        return await asyncio.gather(*tasks)


class VLLMOCRBackend(BaseOCRBackend):
    """Implementation backed by vLLM's AsyncLLMEngine."""

    _engine: Any | None = None

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        if importlib.util.find_spec("vllm") is None:  # pragma: no cover - runtime guard
            raise BackendNotAvailable("vLLM is not available. Install it or use BACKEND=transformers.")

        from vllm import AsyncLLMEngine, SamplingParams

        self._engine = AsyncLLMEngine.from_engine_args(
            model=self.settings.model_id,
            revision=self.settings.model_revision,
            tensor_parallel_size=self.settings.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.settings.vllm_gpu_memory_utilization,
            trust_remote_code=self.settings.trust_remote_code,
            dtype=self.settings.torch_dtype,
        )
        self._sampling_cls = SamplingParams
        return self._engine

    async def warmup(self) -> None:  # pragma: no cover - expensive operation
        await self._load_engine().do_warmup()

    async def infer(self, images: list[bytes], prompt: str, mode: OCRMode | None = None) -> list[OCRResult]:
        engine = self._load_engine()
        sampling_kwargs = {
            key: value
            for key, value in {
                "max_tokens": mode.max_new_tokens if mode else None,
                "temperature": mode.temperature if mode else None,
                "top_p": mode.top_p if mode else None,
            }.items()
            if value is not None
        }
        sampling_params = self._sampling_cls(**sampling_kwargs)

        async def _run_inference(image_bytes: bytes) -> OCRResult:
            start = time.perf_counter()
            request_id = await engine.add_request(
                prompt=prompt,
                multi_modal_data={"image": image_bytes},
                sampling_params=sampling_params,
            )
            result = await engine.get_request(request_id)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            text = "".join(chunk.text for chunk in result.outputs[0].outputs)
            return OCRResult(text=text, duration_ms=elapsed_ms, raw={"backend": "vllm"})

        return await asyncio.gather(*(_run_inference(image) for image in images))


BACKENDS: dict[str, type[BaseOCRBackend]] = {
    "transformers": TransformersOCRBackend,
    "vllm": VLLMOCRBackend,
}


def build_backend(settings: Settings) -> BaseOCRBackend:
    """Instantiate the backend requested in the settings."""

    try:
        backend_cls = BACKENDS[settings.backend]
    except KeyError as exc:  # pragma: no cover - runtime guard
        raise BackendNotAvailable(f"Unsupported backend '{settings.backend}'") from exc
    return backend_cls(settings)


__all__ = [
    "BaseOCRBackend",
    "BackendNotAvailable",
    "TransformersOCRBackend",
    "VLLMOCRBackend",
    "build_backend",
]
