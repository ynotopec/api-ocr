"""Tests for backend utilities."""

from app.services.backends import _normalize_torch_dtype, _reorder_loader_for_deepseek


class _FakeLoader:
    def __init__(self, name: str):
        self.__name__ = name


AutoModelForVision2Seq = _FakeLoader("AutoModelForVision2Seq")
AutoModelForCausalLM = _FakeLoader("AutoModelForCausalLM")
AutoModel = _FakeLoader("AutoModel")


def test_normalize_torch_dtype_aliases():
    assert _normalize_torch_dtype("bf16") == "bfloat16"
    assert _normalize_torch_dtype("BF16") == "bfloat16"
    assert _normalize_torch_dtype("fp16") == "float16"
    assert _normalize_torch_dtype("float32") == "float32"
    # Unknown values should pass through unchanged to preserve error handling downstream.
    assert _normalize_torch_dtype("auto") == "auto"


def test_reorder_loader_for_deepseek_prioritises_causal_loader():
    loader_order = [AutoModelForVision2Seq, AutoModelForCausalLM, AutoModel]
    reordered = _reorder_loader_for_deepseek(
        loader_order,
        ["DeepseekOCRForConditionalGeneration"],
        preferred=(AutoModelForCausalLM, AutoModelForVision2Seq),
    )

    assert reordered[:2] == [AutoModelForCausalLM, AutoModelForVision2Seq]
    assert reordered[-1] is AutoModel


def test_reorder_loader_for_deepseek_noop_for_other_architectures():
    loader_order = [AutoModelForVision2Seq, AutoModelForCausalLM, AutoModel]
    reordered = _reorder_loader_for_deepseek(
        loader_order,
        ["SomethingElse"],
        preferred=(AutoModelForCausalLM, AutoModelForVision2Seq),
    )

    assert reordered == loader_order
