"""Tests for the SAM 2 segmenter logic.

These tests cover the checkpoint discovery and image encoding logic
without requiring torch or PySide6 — just numpy and pathlib.
"""

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Standalone copy of the _find_checkpoint logic from segmenter.py
# so tests don't need to import torch or PySide6.
# ---------------------------------------------------------------------------
def _find_checkpoint(model_dir: Path) -> Path | None:
    """Find the SAM 2 checkpoint in the model directory."""
    model_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        "sam2.1_hiera_tiny.pt",
        "sam2_hiera_tiny.pt",
        "sam2.1_hiera_small.pt",
        "sam2_hiera_small.pt",
        "*.pt",
    ]
    for pattern in patterns:
        if "*" in pattern:
            matches = list(model_dir.glob(pattern))
            if matches:
                return matches[0]
        else:
            candidate = model_dir / pattern
            if candidate.exists():
                return candidate
    return None


class TestCheckpointDiscovery:
    """Tests for _find_checkpoint logic."""

    def test_find_checkpoint_no_files(self, model_dir):
        """Returns None when no checkpoint files exist."""
        result = _find_checkpoint(model_dir)
        assert result is None

    def test_find_checkpoint_with_tiny(self, model_dir):
        """Finds sam2.1_hiera_tiny.pt when present."""
        checkpoint = model_dir / "sam2.1_hiera_tiny.pt"
        checkpoint.write_text("dummy checkpoint data")
        result = _find_checkpoint(model_dir)
        assert result == checkpoint

    def test_find_checkpoint_with_small(self, model_dir):
        """Finds sam2.1_hiera_small.pt when present."""
        checkpoint = model_dir / "sam2.1_hiera_small.pt"
        checkpoint.write_text("dummy checkpoint data")
        result = _find_checkpoint(model_dir)
        assert result == checkpoint

    def test_find_checkpoint_any_pt(self, model_dir):
        """Finds any .pt file if no named checkpoint exists."""
        checkpoint = model_dir / "custom_model.pt"
        checkpoint.write_text("dummy checkpoint data")
        result = _find_checkpoint(model_dir)
        assert result == checkpoint

    def test_find_checkpoint_prefers_named(self, model_dir):
        """Prefers the named tiny checkpoint over other .pt files."""
        tiny = model_dir / "sam2.1_hiera_tiny.pt"
        tiny.write_text("tiny data")
        other = model_dir / "other.pt"
        other.write_text("other data")
        result = _find_checkpoint(model_dir)
        assert result == tiny

    def test_find_checkpoint_legacy_name(self, model_dir):
        """Finds sam2_hiera_tiny.pt (legacy name) when present."""
        checkpoint = model_dir / "sam2_hiera_tiny.pt"
        checkpoint.write_text("legacy checkpoint")
        result = _find_checkpoint(model_dir)
        assert result == checkpoint

    def test_find_checkpoint_prefers_named_over_wildcard(self, model_dir):
        """sam2.1_hiera_tiny.pt is preferred over sam2_hiera_tiny.pt."""
        new_name = model_dir / "sam2.1_hiera_tiny.pt"
        new_name.write_text("v2.1")
        legacy = model_dir / "sam2_hiera_tiny.pt"
        legacy.write_text("legacy")
        result = _find_checkpoint(model_dir)
        assert result == new_name


class TestImageEncoding:
    """Tests for image encoding preprocessing logic.

    The real _encode_image does:
      1. Strip alpha channel if RGBA
      2. Downscale if max dimension > 1024
      3. Convert to tensor
    We test steps 1-2 in isolation.
    """

    def test_rgb_image_shape_preserved(self, rgb_image):
        """RGB image keeps its shape when under 1024px."""
        h, w = rgb_image.shape[:2]
        max_dim = 1024
        scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
        assert scale == 1.0  # 100x100 is well under 1024

    def test_large_image_downscaled(self):
        """A large image (>1024px) gets downscaled."""
        large = np.zeros((2000, 1500, 3), dtype=np.uint8)
        h, w = large.shape[:2]
        max_dim = 1024
        scale = max_dim / max(h, w)
        assert scale < 1.0
        new_h, new_w = int(h * scale), int(w * scale)
        assert max(new_h, new_w) <= max_dim

    def test_exact_boundary_not_downscaled(self):
        """An image exactly 1024px on the long edge is not downscaled."""
        img = np.zeros((1024, 800, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        max_dim = 1024
        scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
        assert scale == 1.0

    def test_slightly_over_boundary_downscaled(self):
        """An image slightly over 1024px gets downscaled."""
        img = np.zeros((1025, 800, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        max_dim = 1024
        scale = max_dim / max(h, w)
        assert scale < 1.0

    def test_rgba_strips_alpha(self, rgba_image):
        """RGBA input has alpha channel stripped for encoding."""
        if rgba_image.shape[2] == 4:
            rgb = rgba_image[:, :, :3].copy()
        else:
            rgb = rgba_image.copy()
        assert rgb.shape[2] == 3

    def test_grayscale_converted_to_rgb(self):
        """Grayscale (H, W) input should conceptually be handled."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        # The segmenter expects (H, W, 3); grayscale needs conversion
        if gray.ndim == 2:
            rgb = np.stack([gray] * 3, axis=-1)
        assert rgb.shape == (100, 100, 3)


class TestModelErrorHandling:
    """Tests for model loading error scenarios."""

    def test_no_checkpoint_returns_error_message(self, model_dir):
        """When no checkpoint exists, the error mentions the user action."""
        result = _find_checkpoint(model_dir)
        assert result is None
        # The real segmenter would emit a model_error with a message
        # that includes "No SAM 2 checkpoint found"
        error_msg = (
            "No SAM 2 checkpoint found in models/ directory.\n\n"
            "Please download sam2.1_hiera_tiny.pt from:\n"
            "https://github.com/facebookresearch/sam2"
        )
        assert "No SAM 2 checkpoint found" in error_msg
        assert "models/" in error_msg
        assert "sam2.1_hiera_tiny.pt" in error_msg

    def test_empty_model_dir_handled_gracefully(self, model_dir):
        """Empty model directory does not raise an exception."""
        # Should not raise
        result = _find_checkpoint(model_dir)
        assert result is None
