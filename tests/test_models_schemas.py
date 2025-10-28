"""Tests for pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.models.schemas import OCRInput


@pytest.mark.parametrize(
    "urls",
    [
        ["${urlLink}"],
        ["image.png", "${ANOTHER}"],
    ],
)
def test_ocr_input_rejects_unexpanded_shell_placeholders(urls):
    with pytest.raises(ValidationError) as exc:
        OCRInput(prompt="<image>", urls=urls)

    assert "unexpanded shell variable" in str(exc.value)


def test_ocr_input_accepts_valid_urls():
    payload = OCRInput(
        prompt="<image>",
        urls=["https://example.com/image.png"],
    )

    assert payload.urls[0] == "https://example.com/image.png"
