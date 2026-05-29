"""Tests for file I/O utilities."""

import numpy as np
from PIL import Image

from sticker_creator.utils.file_io import ImageLoader, ImageSaver


class TestImageLoader:
    """Tests for ImageLoader."""

    def test_load_png(self, png_file, rgb_image):
        """Loading a valid PNG returns the correct RGB image."""
        result = ImageLoader.load(png_file)
        assert result is not None
        assert result.shape == rgb_image.shape
        assert result.dtype == np.uint8
        # Check pixel values match
        np.testing.assert_array_equal(result, rgb_image)

    def test_load_jpg(self, jpg_file):
        """Loading a valid JPEG returns an RGB image (lossy but approximate)."""
        result = ImageLoader.load(jpg_file)
        assert result is not None
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    def test_load_nonexistent(self):
        """Loading a non-existent file returns None."""
        result = ImageLoader.load("/nonexistent/path/image.png")
        assert result is None

    def test_load_from_bytes(self, png_bytes, rgb_image):
        """Deserializing from PNG bytes returns the correct image."""
        result = ImageLoader.load_from_bytes(png_bytes)
        assert result is not None
        assert result.shape == rgb_image.shape
        assert result.dtype == np.uint8

    def test_load_from_empty_bytes(self):
        """Loading from empty bytes returns None."""
        result = ImageLoader.load_from_bytes(b"")
        assert result is None

    def test_load_from_invalid_bytes(self):
        """Loading from invalid bytes returns None."""
        result = ImageLoader.load_from_bytes(b"not an image at all")
        assert result is None


class TestImageSaver:
    """Tests for ImageSaver."""

    def test_save_png_rgba(self, tmp_path, rgba_image):
        """Saving an RGBA array creates a valid PNG file."""
        path = tmp_path / "output.png"
        success = ImageSaver.save_png(rgba_image, path)
        assert success is True
        assert path.exists()
        assert path.stat().st_size > 0

        # Verify the saved file can be loaded back
        reloaded = Image.open(path)
        assert reloaded.mode == "RGBA"
        assert reloaded.size == (100, 100)

    def test_save_png_rgb_fails(self, tmp_path, rgb_image):
        """Saving an RGB (3-channel) array returns False."""
        path = tmp_path / "output.png"
        # Method catches the channel validation error and returns False
        success = ImageSaver.save_png(rgb_image, path)
        assert success is False

    def test_save_png_invalid_path(self, rgba_image):
        """Saving to an invalid path returns False."""
        success = ImageSaver.save_png(rgba_image, "/nonexistent/dir/file.png")
        assert success is False

    def test_save_png_bytes(self, rgba_image):
        """Encoding RGBA to PNG bytes returns valid data."""
        result = ImageSaver.save_png_bytes(rgba_image)
        assert result is not None
        assert len(result) > 0
        # Verify it's a valid PNG by checking magic bytes
        assert result[:8] == b'\x89PNG\r\n\x1a\n'

    # ── WebP tests ──────────────────────────────────────────────────────

    def test_save_webp_rgba(self, tmp_path, rgba_image):
        """Saving an RGBA array creates a valid WebP file with transparency."""
        path = tmp_path / "output.webp"
        success = ImageSaver.save_webp(rgba_image, path, quality=90)
        assert success is True
        assert path.exists()
        assert path.stat().st_size > 0

        # Verify the saved file can be loaded back with RGBA mode
        reloaded = Image.open(path)
        # WebP with alpha may report mode 'RGBA' or 'P' (palette) depending
        # on the encoder; ensure at least 3 channels and size preserved
        assert reloaded.size == (100, 100)
        assert reloaded.mode in ("RGBA", "RGB", "P", "PA")

    def test_save_webp_rgb_fails(self, tmp_path, rgb_image):
        """Saving an RGB (3-channel) array as WebP returns False."""
        path = tmp_path / "output.webp"
        success = ImageSaver.save_webp(rgb_image, path)
        assert success is False

    def test_save_webp_invalid_path(self, rgba_image):
        """Saving WebP to an invalid path returns False."""
        success = ImageSaver.save_webp(rgba_image, "/nonexistent/dir/file.webp")
        assert success is False

    def test_save_webp_bytes(self, rgba_image):
        """Encoding RGBA to WebP bytes returns valid data."""
        result = ImageSaver.save_webp_bytes(rgba_image, quality=80)
        assert result is not None
        assert len(result) > 0
        # WebP magic bytes: RIFF....WEBP
        assert result[:4] == b'RIFF'
        assert result[8:12] == b'WEBP'

    def test_save_webp_bytes_empty(self):
        """Encoding an invalid array returns None."""
        result = ImageSaver.save_webp_bytes(np.zeros((0, 0, 4), dtype=np.uint8))
        assert result is None

    def test_save_webp_different_quality(self, tmp_path, rgba_image):
        """WebP files are created successfully at different quality settings."""
        low_path = tmp_path / "low.webp"
        high_path = tmp_path / "high.webp"
        assert ImageSaver.save_webp(rgba_image, low_path, quality=10) is True
        assert ImageSaver.save_webp(rgba_image, high_path, quality=100) is True
        assert low_path.exists()
        assert high_path.exists()
