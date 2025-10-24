from __future__ import annotations

from app.config import Settings


def test_settings_split_origins(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://a.test, https://b.test")
    settings = Settings()
    assert settings.cors_allow_origins == ["https://a.test", "https://b.test"]


def test_settings_defaults():
    settings = Settings()
    assert settings.backend == "transformers"
    assert settings.max_concurrency == 2
    assert settings.pdf_dpi_default == 180
