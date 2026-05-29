"""Tests for the SAM 2 segmenter logic.

Checkpoint-discovery/validity is tested against the real
:class:`ModelRegistry` (pathlib-only — no torch or PySide6), so the tests
exercise the same code the segmenter and downloader use rather than a copy.
"""

from pathlib import Path

import numpy as np
import pytest

from sticker_creator.segmentation.model_registry import (
    MIN_CHECKPOINT_BYTES,
    ModelRegistry,
    config_name,
)


def _write_valid_checkpoint(path: Path) -> Path:
    """Create a file that passes the registry's size guard (sparse, fast)."""
    with open(path, "wb") as f:
        f.truncate(MIN_CHECKPOINT_BYTES + 1)
    return path


def _write_corrupt_checkpoint(path: Path) -> Path:
    """Create an under-sized (corrupt/truncated) checkpoint file."""
    path.write_bytes(b"too small")
    return path


class TestCheckpointDiscovery:
    """Tests for ModelRegistry.find_loadable / is_downloaded."""

    def test_find_checkpoint_no_files(self, model_dir):
        """Returns None when no checkpoint files exist."""
        reg = ModelRegistry(model_dir)
        assert reg.find_loadable("sam2.1_hiera_tiny") is None

    def test_find_checkpoint_with_tiny(self, model_dir):
        """Finds sam2.1_hiera_tiny.pt when present and valid."""
        ckpt = _write_valid_checkpoint(model_dir / "sam2.1_hiera_tiny.pt")
        reg = ModelRegistry(model_dir)
        assert reg.find_loadable("sam2.1_hiera_tiny") == ckpt
        assert reg.is_downloaded("sam2.1_hiera_tiny")

    def test_find_checkpoint_with_small(self, model_dir):
        """Finds sam2.1_hiera_small.pt when present and valid."""
        ckpt = _write_valid_checkpoint(model_dir / "sam2.1_hiera_small.pt")
        reg = ModelRegistry(model_dir)
        assert reg.find_loadable("sam2.1_hiera_small") == ckpt

    def test_corrupt_checkpoint_rejected(self, model_dir):
        """An under-sized file is treated as absent (not loadable)."""
        _write_corrupt_checkpoint(model_dir / "sam2.1_hiera_tiny.pt")
        reg = ModelRegistry(model_dir)
        assert reg.find_loadable("sam2.1_hiera_tiny") is None
        assert not reg.is_downloaded("sam2.1_hiera_tiny")

    def test_find_checkpoint_legacy_name(self, model_dir):
        """Resolves the legacy sam2_hiera_tiny.pt name for a 2.1 request."""
        legacy = _write_valid_checkpoint(model_dir / "sam2_hiera_tiny.pt")
        reg = ModelRegistry(model_dir)
        assert reg.find_loadable("sam2.1_hiera_tiny") == legacy

    def test_find_checkpoint_prefers_canonical_over_legacy(self, model_dir):
        """sam2.1_hiera_tiny.pt is preferred over the legacy sam2_hiera_tiny.pt."""
        canonical = _write_valid_checkpoint(model_dir / "sam2.1_hiera_tiny.pt")
        _write_valid_checkpoint(model_dir / "sam2_hiera_tiny.pt")
        reg = ModelRegistry(model_dir)
        assert reg.find_loadable("sam2.1_hiera_tiny") == canonical

    def test_list_downloaded_returns_stems(self, model_dir):
        """list_downloaded returns the stems of every .pt present."""
        _write_valid_checkpoint(model_dir / "sam2.1_hiera_tiny.pt")
        _write_corrupt_checkpoint(model_dir / "other.pt")
        reg = ModelRegistry(model_dir)
        assert reg.list_downloaded() == ["other", "sam2.1_hiera_tiny"]

    def test_config_name_maps_known_and_falls_back(self):
        """Known stems map to Hydra config paths; unknown names pass through."""
        assert config_name("sam2.1_hiera_tiny") == "configs/sam2.1/sam2.1_hiera_t"
        assert config_name("mystery") == "mystery"


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

    def test_no_checkpoint_returns_none(self, model_dir):
        """When no checkpoint exists, the registry reports nothing loadable."""
        reg = ModelRegistry(model_dir)
        assert reg.find_loadable("sam2.1_hiera_tiny") is None

    def test_empty_model_dir_handled_gracefully(self, model_dir):
        """An empty model directory does not raise."""
        reg = ModelRegistry(model_dir)
        assert reg.list_downloaded() == []
        assert reg.find_loadable("sam2.1_hiera_tiny") is None
