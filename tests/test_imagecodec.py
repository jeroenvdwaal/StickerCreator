"""Tests for the image-representation codec.

QImage construction/format-conversion is exercised via the ``qapp`` fixture so
the round-trip runs against the real Qt code, not a stand-in.
"""

import numpy as np
import pytest

from sticker_creator.utils import imagecodec


@pytest.fixture
def rgba():
    """A 5×3 RGBA array with distinct, non-symmetric channel values."""
    arr = np.zeros((5, 3, 4), dtype=np.uint8)
    for y in range(5):
        for x in range(3):
            arr[y, x] = [y * 40, x * 80, (y + x) * 20, 255 - y * 10]
    return arr


class TestQImageRoundTrip:
    def test_pixels_preserved(self, qapp, rgba):
        out = imagecodec.from_qimage(imagecodec.to_qimage(rgba))
        assert np.array_equal(out, rgba)

    def test_qimage_owns_data(self, qapp, rgba):
        # Mutating the source after conversion must not change the QImage.
        src = rgba.copy()
        img = imagecodec.to_qimage(src)
        src[:] = 0
        out = imagecodec.from_qimage(img)
        assert np.array_equal(out, rgba)

    def test_non_contiguous_input(self, qapp, rgba):
        view = rgba[::-1]  # reversed-row view is non-contiguous
        out = imagecodec.from_qimage(imagecodec.to_qimage(view))
        assert np.array_equal(out, view)


class TestPilRoundTrip:
    def test_pixels_preserved(self, rgba):
        out = imagecodec.from_pil(imagecodec.to_pil(rgba))
        assert np.array_equal(out, rgba)

    def test_decode_png_round_trip(self, rgba):
        from sticker_creator.utils.file_io import ImageSaver

        png = ImageSaver.save_png_bytes(rgba)
        out = imagecodec.decode_png(png)
        assert np.array_equal(out, rgba)
