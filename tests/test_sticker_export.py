"""Tests for sticker_export — the per-target export policy.

Pure I/O over numpy; no Qt, no dialog. Each format is asserted to land a file
of the right kind and (for WhatsApp) the right canvas size, so the policy
"WhatsApp = 512×512 WebP" is pinned independent of the UI that selects it.
"""

import numpy as np
import pytest
from PIL import Image

from sticker_creator.utils.sticker_export import ExportFormat, export_sticker


@pytest.fixture
def sticker() -> np.ndarray:
    rgba = np.zeros((120, 80, 4), dtype=np.uint8)
    rgba[20:100, 20:60] = [200, 30, 30, 255]
    return rgba


def test_png_writes_lossless_png(tmp_path, sticker):
    path = tmp_path / "out.png"
    assert export_sticker(sticker, path, ExportFormat.PNG) is True
    with Image.open(path) as img:
        assert img.format == "PNG"
        assert img.size == (80, 120)        # unchanged dimensions
        assert img.mode == "RGBA"


def test_webp_writes_webp_unresized(tmp_path, sticker):
    path = tmp_path / "out.webp"
    assert export_sticker(sticker, path, ExportFormat.WEBP) is True
    with Image.open(path) as img:
        assert img.format == "WEBP"
        assert img.size == (80, 120)        # no resize for plain WebP


def test_whatsapp_resizes_to_512_webp(tmp_path, sticker):
    path = tmp_path / "out.webp"
    assert export_sticker(sticker, path, ExportFormat.WHATSAPP_WEBP) is True
    with Image.open(path) as img:
        assert img.format == "WEBP"
        assert img.size == (512, 512)       # the WhatsApp policy: fixed canvas


def test_failure_returns_false(sticker):
    # A non-RGBA array makes ImageSaver fail; the policy propagates that.
    rgb = sticker[:, :, :3]
    assert export_sticker(rgb, "/nonexistent/dir/out.png", ExportFormat.PNG) is False
