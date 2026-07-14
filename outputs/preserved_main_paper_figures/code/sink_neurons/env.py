"""Runtime environment helpers."""

from __future__ import annotations

import os
from pathlib import Path


def configure_runtime() -> None:
    """Set lightweight runtime defaults for CLI scripts."""
    mpl_dir = Path("/tmp/matplotlib")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
