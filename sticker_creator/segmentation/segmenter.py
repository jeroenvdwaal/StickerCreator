"""Qt adapter over the headless SAM 2 core.

The inference itself lives in :class:`sticker_creator.segmentation.sam_segmenter.SamSegmenter`
(torch + numpy, no Qt).  This module only wraps it: a ``QObject`` worker turns
the core's return values and exceptions into Qt signals and runs them in a
background :class:`WorkerThread` to keep the UI responsive, and the public
:class:`Segmenter` manages that thread and the persisted active-model name.

The compatibility patches and ``_CONFIG_MAP`` / ``_MIN_CHECKPOINT_BYTES`` logic
that the old environment requires live with the core in ``sam_segmenter.py`` and
``model_registry.py`` — not here.
"""

import traceback

import numpy as np

from PySide6.QtCore import QObject, Signal, Slot

from sticker_creator.utils import settings as app_settings
from sticker_creator.utils.worker_thread import WorkerThread
from sticker_creator.segmentation.model_registry import (
    KNOWN_MODELS,
    MODEL_SIZES,
)
from sticker_creator.segmentation.sam_segmenter import (
    DEFAULT_MODEL_DIR,
    SamSegmenter,
)

__all__ = ["DEFAULT_MODEL_DIR", "Segmenter", "SegmenterWorker"]


class SegmenterWorker(QObject):
    """Worker object that drives :class:`SamSegmenter` in a background thread.

    Signals:
        model_loaded(str): Emitted with the model name when loaded successfully.
        model_error(str): Emitted on model load failure with error message.
        mask_ready(ndarray): Emitted with the binary mask (HxW, uint8).
        processing_started: Emitted before inference begins.
        processing_finished: Emitted after inference completes.
    """

    model_loaded = Signal(str)
    model_error = Signal(str)
    mask_ready = Signal(object)  # numpy array
    processing_started = Signal()
    processing_finished = Signal()

    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        super().__init__()
        self._core = SamSegmenter(model_dir)

    @property
    def registry(self):
        """The core's :class:`ModelRegistry` (used by the public facade)."""
        return self._core.registry

    @Slot(object)
    def load_model(self, model_name: str | None = None):
        """Load the SAM 2 model in the background thread.

        Resolves ``None`` to the persisted active model, then persists whatever
        the core actually loaded.
        """
        try:
            if model_name is None:
                model_name = app_settings.get_active_model() or None

            loaded = self._core.load_model(model_name)
            app_settings.set_active_model(loaded)
            self.model_loaded.emit(loaded)

        except Exception as e:
            self.model_error.emit(str(e))

    @Slot(object, object)
    def segment(self, image: np.ndarray, points: list[dict]):
        """Run SAM 2 segmentation with point prompts.

        Args:
            image: RGB/RGBA numpy array.
            points: List of dicts with keys: x, y, label (1=positive, 0=negative).
        """
        if len(points) == 0:
            return

        if not self._core.is_loaded:
            self.model_error.emit("Model not loaded yet")
            return

        self.processing_started.emit()
        try:
            mask = self._core.segment(image, points)
            if mask is not None:
                self.mask_ready.emit(mask)
        except Exception as e:
            self.model_error.emit(
                f"Segmentation failed: {e}\n{traceback.format_exc()}"
            )
        finally:
            self.processing_finished.emit()


class Segmenter(QObject):
    """Public interface for the SAM 2 segmentation service.

    Manages the worker thread and exposes signals to the rest of the application.
    Supports dynamic model switching at runtime.
    """

    model_loaded = Signal(str)
    model_error = Signal(str)
    mask_ready = Signal(object)
    processing_started = Signal()
    processing_finished = Signal()

    # Internal dispatch signals that route calls into the background thread
    _load_requested = Signal(object)    # model_name (str | None)
    _segment_requested = Signal(object, object)  # image, points

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = SegmenterWorker()
        self._runner = WorkerThread(self._worker)

        # Forward worker signals to our own
        self._worker.model_loaded.connect(self.model_loaded)
        self._worker.model_error.connect(self.model_error)
        self._worker.mask_ready.connect(self.mask_ready)
        self._worker.processing_started.connect(self.processing_started)
        self._worker.processing_finished.connect(self.processing_finished)

        # Route load/segment calls into the worker thread via queued connections
        self._load_requested.connect(self._worker.load_model)
        self._segment_requested.connect(self._worker.segment)

        # Persistent thread: work is dispatched via the queued signals above.
        self._runner.start()

    # ── Model management ───────────────────────────────────────────────────

    @property
    def active_model_name(self) -> str | None:
        """The currently active model name, or ``None`` if none set."""
        return app_settings.get_active_model()

    @active_model_name.setter
    def active_model_name(self, name: str) -> None:
        app_settings.set_active_model(name)

    def load_model(self, model_name: str | None = None):
        """Queue model loading in the background thread."""
        self._load_requested.emit(model_name)

    def available_models(self) -> list[dict]:
        """Return info about all known model variants and their download status."""
        registry = self._worker.registry
        active = self.active_model_name
        results: list[dict] = []
        for model_name in KNOWN_MODELS:
            ckpt = registry.checkpoint_path(model_name)
            is_downloaded = registry.is_downloaded(model_name)
            results.append({
                "name": model_name,
                "size": MODEL_SIZES.get(model_name, "unknown"),
                "path": ckpt if is_downloaded else None,
                "is_downloaded": is_downloaded,
                "is_active": model_name == active,
            })
        return results

    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded checkpoint. Returns ``True`` on success."""
        ckpt = self._worker.registry.checkpoint_path(model_name)
        if ckpt.exists():
            ckpt.unlink()
            if self.active_model_name == model_name:
                app_settings.clear_active_model()
            return True
        return False

    def set_active_model(self, model_name: str) -> None:
        """Set the active model and trigger a reload."""
        self.active_model_name = model_name
        self.load_model(model_name)

    def segment(self, image: np.ndarray, points: list[dict]):
        """Queue a segmentation task in the background thread."""
        self._segment_requested.emit(image, points)

    def shutdown(self):
        """Stop the background thread and clean up."""
        self._runner.stop()
