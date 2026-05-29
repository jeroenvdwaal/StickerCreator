"""Filesystem locations for user-writable data."""

import os
from pathlib import Path

_APP_DIRNAME = "sticker-creator"


def user_data_dir() -> Path:
    """Per-user writable data directory (XDG_DATA_HOME spec)."""
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / _APP_DIRNAME


def user_model_dir() -> Path:
    """Per-user directory for SAM 2 checkpoints.

    Falls back to a bundled ``models/`` directory next to the source tree if
    one exists with at least one ``.pt`` file (useful in dev), but only when
    that directory is writable. In a Flatpak install /app is read-only, so we
    always end up under XDG_DATA_HOME.
    """
    bundled = Path(__file__).resolve().parent.parent.parent / "models"
    if bundled.is_dir() and any(bundled.glob("*.pt")) and os.access(bundled, os.W_OK):
        return bundled
    return user_data_dir() / "models"
