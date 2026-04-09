"""Load local `.env` without failing if python-dotenv is absent."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_if_present(start: Path | None = None) -> None:
    """Populate os.environ from `.env` in cwd or project root (first wins)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    cwd = Path.cwd()
    candidates = [cwd / ".env"]
    if start is not None:
        candidates.insert(0, start / ".env")
    for p in candidates:
        if p.is_file():
            load_dotenv(p, override=False)
            return


def project_root() -> Path:
    """Directory containing pyproject.toml when running from repo."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()
