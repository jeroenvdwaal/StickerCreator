"""SAM 2 segmentation model wrapper with background thread execution.

Loads the SAM 2 model, caches image embeddings, and runs point-prompted
mask generation in a QThread to keep the UI responsive.

Supports multiple model checkpoints and dynamic model switching at runtime.
The active model name is persisted via QSettings.
"""

import os
import threading
from pathlib import Path

import numpy as np
import torch

from PySide6.QtCore import QObject, Signal, Slot

from sticker_creator.utils.paths import user_model_dir
from sticker_creator.utils import settings as app_settings
from sticker_creator.utils.worker_thread import WorkerThread
from sticker_creator.segmentation.model_registry import (
    KNOWN_MODELS,
    MODEL_SIZES,
    ModelRegistry,
    config_name as _model_config_name,
)

DEFAULT_MODEL_DIR = user_model_dir()


def _patch_sam2_transforms() -> None:
    """Inject a pure-torch replacement for sam2.utils.transforms.

    torchvision (required by sam2.utils.transforms) may be incompatible with
    the installed torch version, causing an import error.  This function
    installs a drop-in SAM2Transforms that uses only torch/numpy so that
    SAM2ImagePredictor can still be imported and used.
    """
    import sys
    import types

    if "sam2.utils.transforms" in sys.modules:
        return  # Already loaded (real or patched)

    try:
        import sam2.utils.transforms  # noqa: F401 — check if real one works
        return
    except Exception:
        pass  # torchvision broken; inject replacement

    import torch.nn as nn
    import torch.nn.functional as F

    class _SAM2Transforms(nn.Module):
        """Pure-torch reimplementation of SAM2Transforms (no torchvision)."""

        def __init__(
            self,
            resolution: int,
            mask_threshold: float,
            max_hole_area: float = 0.0,
            max_sprinkle_area: float = 0.0,
        ):
            super().__init__()
            self.resolution = resolution
            self.mask_threshold = mask_threshold
            self.max_hole_area = max_hole_area
            self.max_sprinkle_area = max_sprinkle_area
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            # Store as plain tensors (CPU-only usage)
            self._mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
            self._std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)

        def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
            """HWC uint8 numpy → CHW float32 tensor in [0, 1]."""
            arr = np.asarray(img)
            return torch.from_numpy(arr.copy()).float().div_(255.0).permute(2, 0, 1)

        def _resize(self, x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                x.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            ).squeeze(0)

        def _normalize(self, x: torch.Tensor) -> torch.Tensor:
            return (x - self._mean) / self._std

        def __call__(self, x):
            x = self._to_tensor(x)
            x = self._resize(x)
            x = self._normalize(x)
            return x

        def forward_batch(self, img_list):
            return torch.stack([self(img) for img in img_list], dim=0)

        def transform_coords(
            self,
            coords: torch.Tensor,
            normalize: bool = False,
            orig_hw=None,
        ) -> torch.Tensor:
            if normalize:
                assert orig_hw is not None
                h, w = orig_hw
                coords = coords.clone()
                coords[..., 0] = coords[..., 0] / w
                coords[..., 1] = coords[..., 1] / h
            return coords * self.resolution

        def transform_boxes(
            self,
            boxes: torch.Tensor,
            normalize: bool = False,
            orig_hw=None,
        ) -> torch.Tensor:
            return self.transform_coords(
                boxes.reshape(-1, 2, 2), normalize, orig_hw
            )

        def postprocess_masks(
            self, masks: torch.Tensor, orig_hw
        ) -> torch.Tensor:
            masks = masks.float()
            return F.interpolate(
                masks, orig_hw, mode="bilinear", align_corners=False
            )

    mod = types.ModuleType("sam2.utils.transforms")
    mod.SAM2Transforms = _SAM2Transforms  # type: ignore[attr-defined]

    # Ensure parent package entries exist
    if "sam2.utils" not in sys.modules:
        sys.modules["sam2.utils"] = types.ModuleType("sam2.utils")

    sys.modules["sam2.utils.transforms"] = mod
    sys.modules["sam2.utils"].transforms = mod  # type: ignore[attr-defined]


def _patch_sam2_checkpoint_loading() -> None:
    """Patch build_sam2's _load_checkpoint to use weights_only=False.

    Newer torch versions default to weights_only=True which rejects the
    pickle opcodes used when the SAM 2 checkpoint was originally saved.
    """
    import torch
    import sam2.build_sam as _build_sam_mod

    original = _build_sam_mod._load_checkpoint

    def _patched_load_checkpoint(model, ckpt_path):  # type: ignore[no-untyped-def]
        if ckpt_path is None:
            return
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys or unexpected_keys:
            import logging
            if missing_keys:
                logging.error(missing_keys)
            if unexpected_keys:
                logging.error(unexpected_keys)
            raise RuntimeError("Checkpoint mismatch — see log for details")

    _build_sam_mod._load_checkpoint = _patched_load_checkpoint


class SegmenterWorker(QObject):
    """Worker object that runs SAM 2 inference in a background thread.

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

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        super().__init__()
        self.registry = ModelRegistry(model_dir)
        self._model = None
        self._predictor = None          # cached SAM2ImagePredictor
        self._predictor_image = None    # last image set on the predictor
        self._lock = threading.Lock()
        self._current_model_name: str | None = None

    @Slot(object)
    def load_model(self, model_name: str | None = None):
        """Load the SAM 2 model in the background thread."""
        try:
            # Resolve which model to load
            if model_name is None:
                model_name = app_settings.get_active_model() or ""
                if not model_name:
                    available = self.registry.list_downloaded()
                    if available:
                        model_name = available[0]
                    else:
                        self.model_error.emit(
                            "No SAM 2 checkpoint found in models/ directory.\n\n"
                            "Go to Models → Manage Models to download one."
                        )
                        return

            checkpoint = self.registry.find_loadable(model_name)
            if checkpoint is None:
                self.model_error.emit(
                    f"Checkpoint {model_name}.pt not found in models/.\n\n"
                    "Go to Models → Manage Models to download it."
                )
                return

            # Apply compatibility patches before any SAM2 imports
            _patch_sam2_transforms()
            from sam2.build_sam import build_sam2  # import before patching _load_checkpoint
            _patch_sam2_checkpoint_loading()
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            cfg = _model_config_name(model_name)
            device = torch.device("cpu")
            torch.set_num_threads(min(4, os.cpu_count() or 4))

            # Unload previous state
            with self._lock:
                self._model = None
                self._predictor = None
                self._predictor_image = None

            self._model = build_sam2(cfg, str(checkpoint), device=device)
            self._model.eval()

            # Create predictor once; it is reused across segment() calls
            with self._lock:
                self._predictor = SAM2ImagePredictor(self._model)
                self._predictor_image = None

            self._current_model_name = model_name

            app_settings.set_active_model(model_name)

            self.model_loaded.emit(model_name)

        except Exception as e:
            self.model_error.emit(str(e))

    @Slot(object, object)
    def segment(self, image: np.ndarray, points: list[dict]):
        """Run SAM 2 segmentation with point prompts.

        Args:
            image: RGB/RGBA numpy array.
            points: List of dicts with keys: x, y, label (1=positive, 0=negative).
        """
        with self._lock:
            predictor = self._predictor

        if predictor is None:
            self.model_error.emit("Model not loaded yet")
            return

        if len(points) == 0:
            return

        self.processing_started.emit()

        try:
            with self._lock:
                # Re-encode only when the image changes (embedding cache)
                if self._predictor_image is not image:
                    # Ensure RGB for the predictor
                    if image.ndim == 3 and image.shape[2] == 4:
                        rgb = image[:, :, :3]
                    else:
                        rgb = image
                    predictor.set_image(rgb)
                    self._predictor_image = image

                # Point prompts in original image pixel coordinates
                coords = np.array(
                    [[pt["x"], pt["y"]] for pt in points], dtype=np.float32
                )  # shape (N, 2)
                labels = np.array(
                    [pt["label"] for pt in points], dtype=np.int32
                )  # shape (N,)

                masks, _scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                    normalize_coords=True,  # predictor normalizes coords internally
                )

            # masks shape: (C, H, W) where C=1 when multimask_output=False
            mask = masks[0]  # (H, W) boolean/float array
            mask_np = (mask > 0.0).astype(np.uint8) * 255

            self.mask_ready.emit(mask_np)

        except Exception as e:
            import traceback
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
