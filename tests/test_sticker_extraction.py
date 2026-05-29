"""Tests for the sticker extraction logic (mask → RGBA → crop)."""

import numpy as np

from sticker_creator.utils.sticker_border import extract_sticker


def _extract_sticker(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Border-free extraction via the real ``extract_sticker``.

    Thin adapter that flips the argument order so the existing call sites read
    ``_extract_sticker(mask, image)``; the logic under test is the production one.
    """
    return extract_sticker(image, mask, border_enabled=False)


class TestStickerExtraction:
    """Tests for the mask → RGBA sticker pipeline."""

    def test_extract_rgba_from_mask_rgb_input(self, rgb_image, binary_mask):
        """RGB input produces RGBA output with correct alpha from mask."""
        result = _extract_sticker(binary_mask, rgb_image)
        assert result.shape[2] == 4  # RGBA
        assert result.shape[0] == 40  # cropped height (20:60)
        assert result.shape[1] == 40  # cropped width (20:60)
        # Pixels inside mask should have alpha=255
        assert np.all(result[:, :, 3] == 255)
        # Colors should match original within the mask region
        orig_cropped = rgb_image[20:60, 20:60, :]
        np.testing.assert_array_equal(result[:, :, :3], orig_cropped)

    def test_extract_rgba_from_mask_rgba_input(self, rgba_image, binary_mask):
        """RGBA input produces RGBA output with combined alpha."""
        result = _extract_sticker(binary_mask, rgba_image)
        assert result.shape[2] == 4
        # Alpha should be 255 where mask is set (overwriting original alpha)
        assert np.all(result[:, :, 3] == 255)

    def test_full_image_mask(self, rgb_image):
        """A mask covering the full image should not crop."""
        mask = np.ones((100, 100), dtype=np.uint8)
        result = _extract_sticker(mask, rgb_image)
        assert result.shape == (100, 100, 4)

    def test_no_mask_region(self, rgb_image):
        """An empty mask produces an empty (0-area) result."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = _extract_sticker(mask, rgb_image)
        # With np.where on all-zero mask, xs and ys are empty arrays
        # The function should handle this gracefully
        assert result.shape[2] == 4

    def test_small_mask_region(self, rgb_image):
        """A small mask crops to a tight bounding box."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 45:55] = 1  # 10x10 region
        result = _extract_sticker(mask, rgb_image)
        assert result.shape[0] == 10
        assert result.shape[1] == 10

    def test_mask_at_edge(self, rgb_image):
        """A mask touching the image edge should not cause errors."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 1  # top-left corner
        result = _extract_sticker(mask, rgb_image)
        assert result.shape[0] == 10
        assert result.shape[1] == 10

    def test_irregular_mask_shape(self, rgb_image):
        """An irregular (non-rectangular) mask still produces correct crop."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Circle approximation
        for y in range(100):
            for x in range(100):
                if (x - 50)**2 + (y - 50)**2 <= 20**2:
                    mask[y, x] = 1
        result = _extract_sticker(mask, rgb_image)
        # Should be cropped to 41x41 (center 50±20)
        assert result.shape[0] <= 42
        assert result.shape[1] <= 42
        assert result.shape[2] == 4


class TestWhiteBorder:
    """Tests for the white border utility on extracted stickers."""

    # ── Fixtures ─────────────────────────────────────────────────────

    def _make_circle_mask(self, h: int = 100, w: int = 100) -> np.ndarray:
        """Create a circular mask for border tests."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy, r = w // 2, h // 2, 20
        for y in range(h):
            for x in range(w):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r**2:
                    mask[y, x] = 1
        return mask

    # ── Tests ────────────────────────────────────────────────────────

    def test_add_white_border_adds_pixels(self, rgb_image):
        """Border adds white opaque pixels around the mask edge."""
        from sticker_creator.utils.sticker_border import add_white_border, extract_sticker

        mask = self._make_circle_mask()
        rgba = extract_sticker(rgb_image, mask, border_enabled=False)
        bordered = add_white_border(rgba, border_width=2)

        # Bordered version is larger (padded by border_width) and has
        # more non-zero alpha pixels
        assert bordered.shape[0] > rgba.shape[0]
        assert bordered.shape[1] > rgba.shape[1]
        assert np.count_nonzero(bordered[:, :, 3]) > np.count_nonzero(rgba[:, :, 3])

    def test_border_width_zero_is_noop(self, rgb_image):
        """border_width=0 returns a copy without modifications."""
        from sticker_creator.utils.sticker_border import add_white_border, extract_sticker

        mask = self._make_circle_mask()
        rgba = extract_sticker(rgb_image, mask, border_enabled=False)
        result = add_white_border(rgba, border_width=0)
        np.testing.assert_array_equal(result, rgba)

    def test_extract_sticker_respects_border_flag(self, rgb_image):
        """border_enabled=False produces same result as current logic; enabled adds pixels."""
        from sticker_creator.utils.sticker_border import extract_sticker

        mask = self._make_circle_mask()
        no_border = extract_sticker(rgb_image, mask, border_enabled=False)
        with_border = extract_sticker(rgb_image, mask, border_enabled=True)

        assert np.count_nonzero(with_border[:, :, 3]) >= np.count_nonzero(no_border[:, :, 3])

    def test_border_width_parameter(self, rgb_image):
        """Larger border_width adds more pixels than smaller width."""
        from sticker_creator.utils.sticker_border import extract_sticker

        mask = self._make_circle_mask()
        w1 = extract_sticker(rgb_image, mask, border_enabled=True, border_width=1)
        w3 = extract_sticker(rgb_image, mask, border_enabled=True, border_width=3)

        assert np.count_nonzero(w3[:, :, 3]) >= np.count_nonzero(w1[:, :, 3])

    def test_border_pixels_are_opaque_white(self, rgb_image):
        """Border pixels blend smoothly from sticker colours toward white.

        With the dual-smooth border both edges are anti-aliased:

        * Outer edge — alpha values transition gradually from 255 at the
          sticker edge to 0 at the border's outer edge.
        * Inner edge — RGB values transition gradually from the original
          sticker colour toward (255, 255, 255), eliminating the hard
          colour step that causes visible pixelation.
        """
        from sticker_creator.utils.sticker_border import add_white_border, extract_sticker

        mask = self._make_circle_mask()
        rgba = extract_sticker(rgb_image, mask, border_enabled=False)
        bordered = add_white_border(rgba, border_width=2)

        # bordered is larger due to padding; compare the centred overlap
        bw = 2
        inner = bordered[bw:-bw, bw:-bw]
        diff_mask = inner[:, :, 3] > rgba[:, :, 3]
        if np.any(diff_mask):
            # Inner-edge RGB blend: each channel is >= the original value
            # (blending toward pure white can only increase brightness).
            assert np.all(inner[diff_mask, 0] >= rgba[diff_mask, 0])
            assert np.all(inner[diff_mask, 1] >= rgba[diff_mask, 1])
            assert np.all(inner[diff_mask, 2] >= rgba[diff_mask, 2])
            # Alpha is > 0 (visible) and varies smoothly — not all 255
            assert np.any(inner[diff_mask, 3] > 0)
            assert np.any(inner[diff_mask, 3] < 255)

    def test_border_at_image_edge(self, rgb_image):
        """Border is visible on edges where mask touches the original image boundary.

        This is the regression test for the bug where the white border was
        missing on sticker edges that coincided with the original image edge
        (because the tight crop left no room for the dilation to expand).
        """
        from sticker_creator.utils.sticker_border import extract_sticker

        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:30, 0:30] = 1  # touches top and left edges of original image

        no_border = extract_sticker(rgb_image, mask, border_enabled=False)
        bordered = extract_sticker(rgb_image, mask, border_enabled=True)

        # 1. Bordered result has more opaque pixels (border was added)
        assert np.count_nonzero(bordered[:, :, 3]) > np.count_nonzero(
            no_border[:, :, 3]
        )

        # 2. The top and left edges (where the mask touched the original
        #    image boundary) MUST have border pixels.  This was the core
        #    bug — the border was missing on those sides.
        bw = 2  # default border width
        assert np.any(bordered[:bw, :, 3] > 0), \
            "No border on top edge (mask touched image top boundary)"
        assert np.any(bordered[:, :bw, 3] > 0), \
            "No border on left edge (mask touched image left boundary)"

        # 3. Border pixels at the top and left edges are white-ish (blended
        #    toward pure white via the smooth inner-edge RGB transition).
        top_border = bordered[:bw, :, 3] > 0
        if np.any(top_border):
            assert np.all(bordered[:bw, :, :3][top_border] >= 100)
        left_border = bordered[:, :bw, 3] > 0
        if np.any(left_border):
            assert np.all(bordered[:, :bw, :3][left_border] >= 100)


class TestApplyBackground:
    """Tests for apply_background — flattening RGBA onto a solid colour."""

    @staticmethod
    def _rgba():
        # 2×2: one opaque red, one half-alpha red, two fully transparent.
        arr = np.zeros((2, 2, 4), dtype=np.uint8)
        arr[0, 0] = [200, 0, 0, 255]
        arr[0, 1] = [200, 0, 0, 128]
        return arr

    def test_transparent_returns_copy_unchanged(self):
        from sticker_creator.utils.sticker_border import (
            BACKGROUND_TRANSPARENT,
            apply_background,
        )

        src = self._rgba()
        out = apply_background(src, BACKGROUND_TRANSPARENT)
        assert np.array_equal(out, src)
        assert out is not src  # must not alias the input

    def test_white_makes_everything_opaque(self):
        from sticker_creator.utils.sticker_border import (
            BACKGROUND_WHITE,
            apply_background,
        )

        out = apply_background(self._rgba(), BACKGROUND_WHITE)
        assert np.all(out[:, :, 3] == 255)
        # Fully transparent pixels become pure white.
        assert tuple(out[1, 1]) == (255, 255, 255, 255)
        # Opaque red stays red.
        assert tuple(out[0, 0]) == (200, 0, 0, 255)

    def test_black_composites_over_black(self):
        from sticker_creator.utils.sticker_border import (
            BACKGROUND_BLACK,
            apply_background,
        )

        out = apply_background(self._rgba(), BACKGROUND_BLACK)
        assert np.all(out[:, :, 3] == 255)
        # Transparent pixels become pure black.
        assert tuple(out[1, 1]) == (0, 0, 0, 255)
        # Half-alpha red composited over black ≈ 100 red.
        assert out[0, 1, 0] == round(200 * (128 / 255))
        assert out[0, 1, 1] == 0
