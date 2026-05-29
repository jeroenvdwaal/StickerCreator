"""Pure SAM 2 inference core — torch + numpy only, no Qt, no QSettings.

This is the segmentation engine with every GUI concern stripped out: it loads
a checkpoint, caches the image embedding, and turns point prompts into a binary
mask.  Image in is a numpy array, mask out is a numpy array — nothing here
imports PySide6, so the whole load→mask→sticker pipeline can run headless
(CLI, tests, batch) without a Qt event loop.

:class:`sticker_creator.segmentation.segmenter.SegmenterWorker` wraps this in a
``QObject`` that turns the return values and exceptions below into Qt signals
and runs them in a background thread; settings persistence (the active model
name) lives in that wrapper, not here.

The non-obvious torch/sam2 compatibility patches required by the current
Python 3.14 / torch 2.11 / sam2 1.1 environment live here too — see
:func:`_patch_sam2_transforms` and :func:`_patch_sam2_checkpoint_loading`.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch

from sticker_creator.utils.paths import user_model_dir
from sticker_creator.segmentation.model_registry import (
    ModelRegistry,
    config_name as _model_config_name,
)

DEFAULT_MODEL_DIR = user_model_dir()


class ModelNotFoundError(Exception):
    """Raised when no loadable checkpoint exists for the requested model."""


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


class SamSegmenter:
    """Headless SAM 2 wrapper: load a checkpoint, prompt with points → mask.

    Pure torch + numpy.  Thread-safe (an internal lock guards the cached model,
    predictor, and per-image embedding) so the same instance can be driven from
    a worker thread, but it owns no thread and emits no signals itself.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        self.registry = ModelRegistry(model_dir)
        self._model = None
        self._predictor = None          # cached SAM2ImagePredictor
        self._predictor_image = None    # last image set on the predictor
        self._lock = threading.Lock()
        self._current_model_name: str | None = None

    @property
    def current_model_name(self) -> str | None:
        """Name of the currently loaded model, or ``None`` if none loaded."""
        return self._current_model_name

    @property
    def is_loaded(self) -> bool:
        """True once a model is loaded and ready to :meth:`segment`."""
        with self._lock:
            return self._predictor is not None

    def load_model(self, model_name: str | None = None) -> str:
        """Load a SAM 2 checkpoint into memory and return its resolved name.

        When *model_name* is ``None`` the first downloaded checkpoint is used.

        Raises:
            ModelNotFoundError: no checkpoint available for the request.
        """
        if model_name is None:
            available = self.registry.list_downloaded()
            if not available:
                raise ModelNotFoundError(
                    "No SAM 2 checkpoint found in models/ directory.\n\n"
                    "Go to Models → Manage Models to download one."
                )
            model_name = available[0]

        checkpoint = self.registry.find_loadable(model_name)
        if checkpoint is None:
            raise ModelNotFoundError(
                f"Checkpoint {model_name}.pt not found in models/.\n\n"
                "Go to Models → Manage Models to download it."
            )

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
        return model_name

    def segment(self, image: np.ndarray, points: list[dict]) -> np.ndarray | None:
        """Run point-prompted segmentation; return an ``H×W`` uint8 mask.

        Args:
            image: RGB/RGBA numpy array.
            points: List of dicts with keys ``x``, ``y``, ``label``
                (1=positive, 0=negative).

        Returns:
            ``H×W`` uint8 mask (0/255), or ``None`` when *points* is empty.

        Raises:
            RuntimeError: no model has been loaded yet.
        """
        with self._lock:
            predictor = self._predictor

        if predictor is None:
            raise RuntimeError("Model not loaded yet")

        if len(points) == 0:
            return None

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
        return (mask > 0.0).astype(np.uint8) * 255
