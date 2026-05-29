"""Tests for clipboard operations.

These tests require pytest-qt with a QApplication fixture.
The qtbot fixture is provided by pytest-qt automatically.
"""

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QClipboard, QImage


@pytest.mark.requires_qt
class TestClipboardManager:
    """Tests for ClipboardManager requiring a QApplication."""

    def test_copy_image(self, qtbot, rgba_image):
        """Copying an RGBA image to clipboard should succeed."""
        from sticker_creator.utils.clipboard import ClipboardManager

        cm = ClipboardManager()
        success = cm.copy_image(rgba_image)
        assert success is True

    def test_paste_after_copy(self, qtbot, rgba_image):
        """Pasting after a copy should return an RGB array."""
        from sticker_creator.utils.clipboard import ClipboardManager

        cm = ClipboardManager()
        cm.copy_image(rgba_image)

        pasted = cm.paste_image()
        # We should get an RGB image back
        assert pasted is not None
        assert pasted.shape[2] == 3  # RGB
        assert pasted.shape[0] == rgba_image.shape[0]
        assert pasted.shape[1] == rgba_image.shape[1]
        assert pasted.dtype == np.uint8

    def test_roundtrip_preserves_content(self, qtbot):
        """A known RGB image survives copy→paste roundtrip."""
        from sticker_creator.utils.clipboard import ClipboardManager

        # Create a simple RGBA image
        original = np.zeros((50, 50, 4), dtype=np.uint8)
        original[:, :, :3] = [64, 128, 192]  # blue-gray color
        original[:, :, 3] = 255  # fully opaque

        cm = ClipboardManager()
        cm.copy_image(original)

        pasted = cm.paste_image()
        assert pasted is not None
        # Colors may shift slightly through clipboard conversion,
        # but should be approximately the same
        mean_color = pasted.mean(axis=(0, 1))
        assert abs(float(mean_color[0]) - 64) < 5   # R
        assert abs(float(mean_color[1]) - 128) < 5  # G
        assert abs(float(mean_color[2]) - 192) < 5  # B

    def test_clipboard_has_image_png_mime(self, qtbot, rgba_image):
        """Clipboard image can be retrieved via pixmap with alpha preserved."""
        from sticker_creator.utils.clipboard import ClipboardManager

        cm = ClipboardManager()
        cm.copy_image(rgba_image)

        clipboard = QCoreApplication.instance().clipboard()

        # The primary path: retrieve via pixmap (what all real apps use).
        # In headless CI only 'application/x-qt-image' is advertised, but
        # pixmap() still works because Qt holds the image internally.
        pixmap = clipboard.pixmap(QClipboard.Mode.Clipboard)
        assert pixmap is not None and not pixmap.isNull(), (
            "Clipboard should contain a valid pixmap after copy_image"
        )

        # Convert to RGBA numpy array and verify alpha values
        qimage = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        ptr = qimage.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
            (qimage.height(), qimage.width(), 4)
        )

        # Dimensions should match original
        assert arr.shape[:2] == (100, 100)

        # Verify alpha values match original
        opaque_mask = rgba_image[:, :, 3] == 255
        semi_mask = rgba_image[:, :, 3] == 128
        trans_mask = rgba_image[:, :, 3] == 0

        np.testing.assert_array_equal(
            arr[opaque_mask, 3], 255,
            "Opaque pixels should have alpha=255 in clipboard image",
        )
        np.testing.assert_array_equal(
            arr[semi_mask, 3], 128,
            "Semi-transparent pixels should have alpha=128 in clipboard image",
        )
        np.testing.assert_array_equal(
            arr[trans_mask, 3], 0,
            "Transparent pixels should have alpha=0 in clipboard image",
        )
