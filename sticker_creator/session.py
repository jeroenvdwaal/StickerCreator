"""The sticker session — the workflow state machine.

Holds the document state of one editing session (the loaded image, its mask,
the derived raw sticker, the composed sticker, and the user's compose options)
and owns the transitions between them — including the invalidation cascade that
was previously hand-repeated in three ``MainWindow`` slots.

It is a ``QObject`` so it can announce changes via signals, but it touches no
widgets: the window connects the session's signals to widget setters. That lets
the whole image → mask → sticker workflow be exercised without a ``QMainWindow``
or an event loop (the balloon compose step needs a ``QGuiApplication``; the
mask → raw → border path does not).

Prompt points stay in the ``ImageViewer`` (it draws them); the session is told
the resulting mask via :meth:`set_mask` and is told to drop it via
:meth:`clear_derived`. Running segmentation is a service call and stays with the
window.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

from sticker_creator.utils.sticker_border import (
    BACKGROUND_TRANSPARENT,
    apply_background,
)
from sticker_creator.utils.sticker_pipeline import (
    StickerOptions,
    auto_border_width,
    build_raw_sticker,
    compose_sticker,
)
from sticker_creator.utils.balloon_renderer import STYLE_AUTO


class StickerSession(QObject):
    """Workflow state + transitions for one sticker-editing session.

    Signals:
        image_changed(object): the loaded image (ndarray) or ``None``.
        mask_changed(object): the segmentation mask (ndarray) or ``None``.
        border_width_changed(int): a newly auto-derived border width.
        sticker_changed(object): the preview-ready sticker (background applied,
            ndarray) or ``None``. Save/copy operate on :attr:`current_sticker`,
            which is the same sticker *without* the preview background.
    """

    image_changed = Signal(object)
    mask_changed = Signal(object)
    border_width_changed = Signal(int)
    sticker_changed = Signal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._image_path: str | None = None
        self._mask: np.ndarray | None = None
        self._raw_sticker: np.ndarray | None = None
        self._current_sticker: np.ndarray | None = None

        self._border_enabled: bool = True
        self._border_width: int = 7
        self._background: str = BACKGROUND_TRANSPARENT
        self._balloon_text: str = ""
        self._balloon_style: str = STYLE_AUTO

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def current_image(self) -> np.ndarray | None:
        return self._image

    @property
    def image_path(self) -> str | None:
        return self._image_path

    @property
    def current_sticker(self) -> np.ndarray | None:
        """The composed sticker for save/copy — no preview background."""
        return self._current_sticker

    @property
    def balloon_text(self) -> str:
        return self._balloon_text

    @property
    def has_sticker(self) -> bool:
        return self._current_sticker is not None

    # ── Image / mask transitions ─────────────────────────────────────────────

    def set_image(self, image: np.ndarray, path: str | None = None) -> None:
        """Load a new image; invalidate everything derived from the old one."""
        self._image = image
        self._image_path = path
        self._clear_derived_state()
        self.image_changed.emit(image)
        self.mask_changed.emit(None)
        self.sticker_changed.emit(None)

    def set_mask(self, mask: np.ndarray) -> None:
        """Accept a fresh segmentation mask and re-derive the sticker."""
        if self._image is None:
            return
        self._mask = mask
        self._raw_sticker = build_raw_sticker(self._image, mask)
        self._border_width = auto_border_width(self._raw_sticker)
        self.border_width_changed.emit(self._border_width)
        self.mask_changed.emit(mask)
        self._recompose()

    def clear_derived(self) -> None:
        """Drop mask + sticker (e.g. last point undone, or points cleared)."""
        self._clear_derived_state()
        self.mask_changed.emit(None)
        self.sticker_changed.emit(None)

    def _clear_derived_state(self) -> None:
        self._mask = None
        self._raw_sticker = None
        self._current_sticker = None

    # ── Compose options ──────────────────────────────────────────────────────

    def set_border_enabled(self, enabled: bool) -> None:
        self._border_enabled = enabled
        self._recompose()

    def set_border_width(self, width: int) -> None:
        self._border_width = width
        if self._border_enabled:
            self._recompose()

    def set_background(self, color: str) -> None:
        self._background = color
        self._recompose()

    def set_balloon_text(self, text: str) -> None:
        self._balloon_text = text
        self._recompose()

    def set_balloon_style(self, style: str) -> None:
        self._balloon_style = style
        self._recompose()

    # ── Compose ──────────────────────────────────────────────────────────────

    def _recompose(self) -> None:
        """Re-derive the composed sticker and emit the preview-ready image."""
        if self._raw_sticker is None:
            self._current_sticker = None
            self.sticker_changed.emit(None)
            return
        options = StickerOptions(
            border_enabled=self._border_enabled,
            border_width=self._border_width,
            balloon_text=self._balloon_text,
            balloon_style=self._balloon_style,
            background=self._background,
        )
        self._current_sticker = compose_sticker(self._raw_sticker, options)
        self.sticker_changed.emit(apply_background(self._current_sticker, self._background))
