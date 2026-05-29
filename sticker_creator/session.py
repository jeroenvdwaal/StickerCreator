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

Prompt points live here too: the session owns the canonical list and the
orchestration that turns clicks into segmentation runs. The ``ImageViewer`` is a
dumb view — it emits raw clicks and renders whatever points the session pushes
back via :data:`prompt_points_changed`. Running segmentation is still a service
call (the model lives off-thread), so the session does not call the segmenter
directly: it emits :data:`segment_requested` and the window dispatches it. That
keeps the whole click → segment → mask → compose loop drivable headless — a test
listens to :data:`segment_requested` and feeds masks back via :meth:`set_mask`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import QObject, Signal

from sticker_creator.utils.sticker_border import apply_background
from sticker_creator.utils.sticker_pipeline import (
    StickerOptions,
    auto_border_width,
    build_raw_sticker,
    compose_sticker,
)


class StickerSession(QObject):
    """Workflow state + transitions for one sticker-editing session.

    Signals:
        image_changed(object): the loaded image (ndarray) or ``None``.
        mask_changed(object): the segmentation mask (ndarray) or ``None``.
        border_width_changed(int): a newly auto-derived border width.
        sticker_changed(object): the preview-ready sticker (background applied,
            ndarray) or ``None``. Save/copy operate on :attr:`current_sticker`,
            which is the same sticker *without* the preview background.
        prompt_points_changed(object): the current prompt points (a fresh list
            of ``{x, y, label}`` dicts) for the viewer to render.
        segment_requested(object, object): ``(image, points)`` — a request for
            the window to run the segmenter; the resulting mask comes back via
            :meth:`set_mask`.
    """

    # Prompt-point labels (mirror the viewer's POSITIVE/NEGATIVE).
    POSITIVE = 1
    NEGATIVE = 0

    image_changed = Signal(object)
    mask_changed = Signal(object)
    border_width_changed = Signal(int)
    sticker_changed = Signal(object)
    prompt_points_changed = Signal(object)
    segment_requested = Signal(object, object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._image_path: str | None = None
        self._points: list[dict] = []
        self._mask: np.ndarray | None = None
        self._raw_sticker: np.ndarray | None = None
        self._current_sticker: np.ndarray | None = None

        # The single currency for everything the user controls about composition.
        self._options = StickerOptions()

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
        return self._options.balloon_text

    @property
    def has_sticker(self) -> bool:
        return self._current_sticker is not None

    def prompt_points(self) -> list[dict]:
        """The current prompt points as a fresh, independently-mutable list."""
        return [dict(p) for p in self._points]

    @property
    def has_prompt_points(self) -> bool:
        return bool(self._points)

    # ── Prompt points + segmentation orchestration ───────────────────────────

    def add_prompt(self, x: int, y: int, label: int) -> None:
        """Record a prompt click and request a re-segmentation.

        No-op when no image is loaded — a click on an empty canvas means nothing.
        """
        if self._image is None:
            return
        self._points.append({"x": x, "y": y, "label": label})
        self.prompt_points_changed.emit(self.prompt_points())
        self._request_segmentation()

    def undo_prompt(self) -> None:
        """Drop the most recent prompt point.

        Re-segments with what remains, or clears the derived sticker when the
        last point is undone.
        """
        if not self._points:
            return
        self._points.pop()
        self.prompt_points_changed.emit(self.prompt_points())
        if not self._points:
            self.clear_derived()
            return
        self._request_segmentation()

    def clear_prompts(self) -> None:
        """Drop all prompt points and the derived sticker."""
        if not self._points:
            return
        self._points.clear()
        self.prompt_points_changed.emit(self.prompt_points())
        self.clear_derived()

    def _request_segmentation(self) -> None:
        if self._image is None or not self._points:
            return
        self.segment_requested.emit(self._image, self.prompt_points())

    # ── Image / mask transitions ─────────────────────────────────────────────

    def set_image(self, image: np.ndarray, path: str | None = None) -> None:
        """Load a new image; invalidate the prompts and everything derived."""
        self._image = image
        self._image_path = path
        self._points = []
        self._clear_derived_state()
        self.image_changed.emit(image)
        self.prompt_points_changed.emit(self.prompt_points())
        self.mask_changed.emit(None)
        self.sticker_changed.emit(None)

    def set_mask(self, mask: np.ndarray) -> None:
        """Accept a fresh segmentation mask and re-derive the sticker."""
        if self._image is None:
            return
        self._mask = mask
        self._raw_sticker = build_raw_sticker(self._image, mask)
        self._options = replace(
            self._options, border_width=auto_border_width(self._raw_sticker)
        )
        self.border_width_changed.emit(self._options.border_width)
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

    def set_options(self, options: StickerOptions) -> None:
        """Replace the whole compose-option bundle and re-derive the sticker.

        The single entry point the inspector drives — one signal, one currency.
        The granular setters below remain for headless/test convenience.
        """
        self._options = options
        self._recompose()

    def set_border_enabled(self, enabled: bool) -> None:
        self._options = replace(self._options, border_enabled=enabled)
        self._recompose()

    def set_border_width(self, width: int) -> None:
        self._options = replace(self._options, border_width=width)
        if self._options.border_enabled:
            self._recompose()

    def set_background(self, color: str) -> None:
        self._options = replace(self._options, background=color)
        self._recompose()

    def set_balloon_text(self, text: str) -> None:
        self._options = replace(self._options, balloon_text=text)
        self._recompose()

    def set_balloon_style(self, style: str) -> None:
        self._options = replace(self._options, balloon_style=style)
        self._recompose()

    # ── Compose ──────────────────────────────────────────────────────────────

    def _recompose(self) -> None:
        """Re-derive the composed sticker and emit the preview-ready image."""
        if self._raw_sticker is None:
            self._current_sticker = None
            self.sticker_changed.emit(None)
            return
        self._current_sticker = compose_sticker(self._raw_sticker, self._options)
        self.sticker_changed.emit(
            apply_background(self._current_sticker, self._options.background)
        )
