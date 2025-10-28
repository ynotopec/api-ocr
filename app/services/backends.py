"""Inference backend implementations."""
from __future__ import annotations

import asyncio
import importlib
import io
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from app.config import Settings
from app.models.schemas import OCRMode, OCRResult


class BackendNotAvailable(RuntimeError):
    """Raised when a requested backend cannot be used."""


TORCH_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
}


def _normalize_torch_dtype(dtype_name: str) -> str:
    """Return the canonical torch dtype attribute name for ``dtype_name``."""

    key = dtype_name.lower()
    return TORCH_DTYPE_ALIASES.get(key, dtype_name)


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

    _model: Any | None = None
    _processor: Any | None = None
    _tokenizer: Any | None = None
    _torch: Any | None = None

    def _load_components(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._processor is not None and self._tokenizer is not None:
            return self._model, self._processor, self._tokenizer

        if importlib.util.find_spec("transformers") is None:  # pragma: no cover - runtime guard
            raise BackendNotAvailable(
                "transformers is not available. Install it or switch BACKEND=vllm."
            )

        from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
            raise BackendNotAvailable("torch is not installed") from exc

        dtype_name = _normalize_torch_dtype(self.settings.torch_dtype)
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:  # pragma: no cover - user misconfiguration
            raise BackendNotAvailable(f"Unsupported torch dtype: {self.settings.torch_dtype}")

        common_kwargs = {
            "revision": self.settings.model_revision,
            "token": self.settings.hf_token,
            "trust_remote_code": self.settings.trust_remote_code,
        }

        processor = AutoProcessor.from_pretrained(
            self.settings.model_id,
            **common_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_id,
            **common_kwargs,
        )

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        last_error: Exception | None = None
        model = None
        for loader in (AutoModelForVision2Seq, AutoModelForCausalLM):
            try:
                model = loader.from_pretrained(
                    self.settings.model_id,
                    torch_dtype=dtype,
                    device_map=self.settings.device_map,
                    **common_kwargs,
                )
                break
            except Exception as exc:  # pragma: no cover - exercised via fallback
                last_error = exc
                continue

        if model is None:  # pragma: no cover - propagation of underlying error
            assert last_error is not None
            raise BackendNotAvailable(str(last_error)) from last_error

        self._model = model
        self._processor = processor
        self._tokenizer = tokenizer
        self._torch = torch
        return model, processor, tokenizer

    async def warmup(self) -> None:  # pragma: no cover - expensive operation
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_components)

    async def infer(self, images: list[bytes], prompt: str, mode: OCRMode | None = None) -> list[OCRResult]:
        model, processor, tokenizer = self._load_components()
        loop = asyncio.get_running_loop()

        generate_kwargs = {
            key: value
            for key, value in {
                "max_new_tokens": mode.max_new_tokens if mode else None,
                "temperature": mode.temperature if mode else None,
                "top_p": mode.top_p if mode else None,
                "do_sample": (mode.temperature is not None or (mode.top_p is not None and mode.top_p < 1.0))
                if mode
                else None,
            }.items()
            if value is not None
        }

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if pad_token_id is not None:
            generate_kwargs.setdefault("pad_token_id", pad_token_id)

        def _load_image(image_bytes: bytes) -> Image.Image:
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.convert("RGB")

        torch = self._torch
        if torch is None:  # pragma: no cover - defensive fallback
            torch = importlib.import_module("torch")
            self._torch = torch

        def _run_inference(image_bytes: bytes) -> OCRResult:
            start = time.perf_counter()
            image = _load_image(image_bytes)
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )

            try:
                to_device = model.device  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover - some accelerate configs
                to_device = None

            if to_device is not None:
                inputs = {
                    key: value.to(to_device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generate_kwargs)

            if hasattr(generated_ids, "to"):
                generated_ids = generated_ids.to("cpu")

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            text = decoded[0] if decoded else ""
            if text.startswith(prompt):
                text = text[len(prompt) :].lstrip()
            text = text.strip()
            elapsed_ms = int((time.perf_counter() - start) * 1000)
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
            dtype=_normalize_torch_dtype(self.settings.torch_dtype),
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
