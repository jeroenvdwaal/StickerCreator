"""Shared fixtures for Sticker Creator tests."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_qt: test needs a Qt application / event loop"
    )


@pytest.fixture
def rgb_image() -> np.ndarray:
    """Create a synthetic 100x100 RGB test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Red square in top-left quadrant
    img[10:40, 10:40, :] = [255, 0, 0]
    # Green circle area in center
    img[40:70, 40:70, :] = [0, 255, 0]
    # Blue area in bottom-right
    img[70:90, 70:90, :] = [0, 0, 255]
    return img


@pytest.fixture
def rgba_image() -> np.ndarray:
    """Create a synthetic 100x100 RGBA test image with alpha."""
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[:, :, :3] = [128, 128, 128]
    img[10:40, 10:40, 3] = 255    # fully opaque
    img[40:70, 40:70, 3] = 128    # semi-transparent
    # rest is transparent (alpha=0)
    return img


@pytest.fixture
def binary_mask() -> np.ndarray:
    """Create a 100x100 binary mask with a rectangular region."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 1
    return mask


@pytest.fixture
def png_file(tmp_path: Path, rgb_image: np.ndarray) -> Path:
    """Create a temporary PNG file from the synthetic RGB image."""
    path = tmp_path / "test.png"
    pil_img = Image.fromarray(rgb_image, "RGB")
    pil_img.save(path, format="PNG")
    return path


@pytest.fixture
def jpg_file(tmp_path: Path, rgb_image: np.ndarray) -> Path:
    """Create a temporary JPEG file from the synthetic RGB image."""
    path = tmp_path / "test.jpg"
    pil_img = Image.fromarray(rgb_image, "RGB")
    pil_img.save(path, format="JPEG")
    return path


@pytest.fixture
def png_bytes(rgb_image: np.ndarray) -> bytes:
    """Return raw PNG bytes of the synthetic RGB image."""
    buf = io.BytesIO()
    pil_img = Image.fromarray(rgb_image, "RGB")
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory."""
    path = tmp_path / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
