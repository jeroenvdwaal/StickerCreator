"""Tests for sticker resize utility (WhatsApp 512×512 compliance)."""

import numpy as np
from PIL import Image

from sticker_creator.utils.sticker_resize import resize_to_512


class TestResizeTo512:
    """Tests for resize_to_512()."""

    def test_square_input(self):
        """A square RGBA image is resized to exactly 512×512."""
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        rgba[50:150, 50:150, :3] = [255, 0, 0]
        rgba[50:150, 50:150, 3] = 255

        result = resize_to_512(rgba)
        assert result.shape == (512, 512, 4)
        assert result.dtype == np.uint8
        # Content should be centred; check non-zero pixels exist in centre
        assert result[200:300, 200:300, 3].sum() > 0

    def test_wide_input(self):
        """A wide image (width > height) is letterboxed with transparent padding."""
        rgba = np.zeros((100, 300, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255  # fully opaque

        result = resize_to_512(rgba)
        assert result.shape == (512, 512, 4)
        # The image fills the full width; centre row should be opaque
        assert result[256, :, 3].sum() > 0
        # Top and bottom edges should be transparent (letterbox padding)
        assert result[0, :, 3].sum() == 0
        assert result[511, :, 3].sum() == 0

    def test_tall_input(self):
        """A tall image (height > width) is pillarboxed with transparent padding."""
        rgba = np.zeros((300, 100, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255  # fully opaque

        result = resize_to_512(rgba)
        assert result.shape == (512, 512, 4)
        # The image fills the full height; centre column should be opaque
        assert result[:, 256, 3].sum() > 0
        # Left and right edges should be transparent (pillarbox padding)
        assert result[:, 0, 3].sum() == 0
        assert result[:, 511, 3].sum() == 0

    def test_alpha_preserved(self):
        """Alpha channel values are preserved through resize."""
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[20:80, 20:80, :3] = [64, 128, 192]
        rgba[20:80, 20:80, 3] = 128  # semi-transparent

        result = resize_to_512(rgba)
        assert result.shape == (512, 512, 4)
        # Semi-transparent region should still have alpha values
        centre_alpha = result[200:300, 200:300, 3]
        assert centre_alpha.max() > 0
        # Edges should be fully transparent
        assert result[0, 0, 3] == 0

    def test_small_input_upscaled(self):
        """A tiny image is upscaled correctly to fill the 512 canvas."""
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[:, :, :3] = [255, 255, 255]
        rgba[:, :, 3] = 255

        result = resize_to_512(rgba)
        assert result.shape == (512, 512, 4)
        # Should have opaque content in the centre
        assert result[200:300, 200:300, 3].sum() > 0

    def test_exact_512_input(self):
        """An already-512×512 image is returned unchanged (copy)."""
        rgba = np.random.randint(0, 256, (512, 512, 4), dtype=np.uint8)
        rgba[100:200, 100:200, 3] = 255

        result = resize_to_512(rgba)
        assert result.shape == (512, 512, 4)
        np.testing.assert_array_equal(result, rgba)

    def test_output_valid_png_roundtrip(self):
        """Resized output can be saved and re-loaded as a valid image."""
        rgba = np.zeros((150, 200, 4), dtype=np.uint8)
        rgba[30:120, 40:160, :3] = [255, 0, 0]
        rgba[30:120, 40:160, 3] = 255

        result = resize_to_512(rgba)
        # Save and reload to verify integrity
        import io
        from PIL import Image as PILImage

        buf = io.BytesIO()
        pil_img = PILImage.fromarray(result, "RGBA")
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        reloaded = PILImage.open(buf)
        assert reloaded.size == (512, 512)
        assert reloaded.mode == "RGBA"
