"""Tests for backend utilities."""

from app.services.backends import _normalize_torch_dtype


def test_normalize_torch_dtype_aliases():
    assert _normalize_torch_dtype("bf16") == "bfloat16"
    assert _normalize_torch_dtype("BF16") == "bfloat16"
    assert _normalize_torch_dtype("fp16") == "float16"
    assert _normalize_torch_dtype("float32") == "float32"
    # Unknown values should pass through unchanged to preserve error handling downstream.
    assert _normalize_torch_dtype("auto") == "auto"
