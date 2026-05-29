"""Single source of truth for persisted settings.

The app identity (``QSettings(ORGANIZATION, APPLICATION)``) and every key string
live here once, so no caller hardcodes the org/app tuple or a bare key literal.
Typed accessors cover the values touched from more than one place (the active
model); UI-only values use :class:`Keys` constants with the shared factory.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings

ORGANIZATION = "KDE"
APPLICATION = "Sticker Creator"


class Keys:
    """Persisted setting keys."""

    ACTIVE_MODEL = "active_model"
    GEOMETRY = "ui/geometry"
    WINDOW_STATE = "ui/window_state"
    LAST_OPEN_DIR = "paths/last_open_dir"
    LAST_SAVE_DIR = "paths/last_save_dir"
    SHOW_WELCOME = "ui/show_welcome"


def get_settings() -> QSettings:
    """A persistent :class:`QSettings` for the application."""
    return QSettings(ORGANIZATION, APPLICATION)


# ── Active model ────────────────────────────────────────────────────────────
# Read/written from both the segmenter worker thread and the GUI thread; the
# typed accessors keep the key and the sync() discipline in one place.

def get_active_model() -> str | None:
    """The persisted active model name, or ``None`` if unset."""
    return get_settings().value(Keys.ACTIVE_MODEL, "") or None


def set_active_model(name: str) -> None:
    """Persist *name* as the active model."""
    s = get_settings()
    s.setValue(Keys.ACTIVE_MODEL, name)
    s.sync()


def clear_active_model() -> None:
    """Forget the active model (e.g. after its checkpoint is deleted)."""
    s = get_settings()
    s.remove(Keys.ACTIVE_MODEL)
    s.sync()
