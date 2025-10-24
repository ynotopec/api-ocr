"""Command line entrypoint for running the API with uvicorn."""
from __future__ import annotations

import uvicorn


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
