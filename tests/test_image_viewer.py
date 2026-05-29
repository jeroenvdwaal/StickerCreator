"""Tests for image viewer overlay rendering helpers.

These tests cover the helper functions used in image_viewer.py
that only require numpy and OpenCV — no QApplication needed.
"""

import numpy as np

from sticker_creator.widgets.mask_contours import mask_to_contours as _mask_to_contours
from sticker_creator.widgets.view_transform import ViewTransform


def _contour_to_points(contour: np.ndarray) -> list[tuple[float, float]]:
    """Convert an OpenCV contour (N, 1, 2) to a list of (x, y) tuples."""
    return [(float(pt[0][0]), float(pt[0][1])) for pt in contour]


class TestMaskToContours:
    """Tests for the shared mask_to_contours helper."""

    def test_empty_mask(self):
        """An all-zero mask produces no contours."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        contours = _mask_to_contours(mask)
        assert len(contours) == 0

    def test_full_mask(self):
        """An all-ones mask produces one contour."""
        mask = np.ones((100, 100), dtype=np.uint8)
        contours = _mask_to_contours(mask)
        assert len(contours) >= 1

    def test_rectangle_mask(self):
        """A rectangular region produces one contour with 4+ points."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 1
        contours = _mask_to_contours(mask)
        assert len(contours) >= 1
        # The contour should have at least 4 points for a rectangle
        assert contours[0].shape[0] >= 4

    def test_two_disjoint_regions(self):
        """Two separate regions produce two contours."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # top-left square
        mask[60:80, 60:80] = 1  # bottom-right square
        contours = _mask_to_contours(mask)
        assert len(contours) >= 2

    def test_single_pixel(self):
        """A single-pixel region still produces a contour."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1
        contours = _mask_to_contours(mask)
        assert len(contours) >= 1


class TestContourToPoints:
    """Tests for converting OpenCV contours to point lists."""

    def test_simple_square(self):
        """A square contour converts to a 4-point list."""
        contour = np.array([
            [[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]],
        ], dtype=np.float32)
        points = _contour_to_points(contour)
        assert len(points) == 4

    def test_triangle(self):
        """A triangle contour converts to a 3-point list."""
        contour = np.array([
            [[5, 5]], [[15, 5]], [[10, 15]],
        ], dtype=np.float32)
        points = _contour_to_points(contour)
        assert len(points) == 3

    def test_coordinates_preserved(self):
        """Point coordinates match the contour values."""
        contour = np.array([
            [[10, 20]], [[30, 40]],
        ], dtype=np.float32)
        points = _contour_to_points(contour)
        assert points[0] == (10.0, 20.0)
        assert points[1] == (30.0, 40.0)


class TestViewTransform:
    """Tests for the pure scene→pixel coordinate algebra extracted from the
    viewer (sticker_creator.widgets.view_transform.ViewTransform)."""

    def test_from_image_reads_dimensions(self):
        """from_image takes width/height from a (H, W, C) numpy image."""
        image = np.zeros((30, 80, 3), dtype=np.uint8)  # H=30, W=80
        t = ViewTransform.from_image(image)
        assert t.width == 80
        assert t.height == 30

    def test_scene_to_pixel_truncates(self):
        """Fractional scene coords truncate toward an integer pixel."""
        t = ViewTransform(width=100, height=100)
        assert t.scene_to_pixel(50.7, 30.2) == (50, 30)

    def test_scene_to_pixel_in_bounds_edges(self):
        """Last valid pixel is width-1 / height-1."""
        t = ViewTransform(width=100, height=100)
        assert t.scene_to_pixel(0, 0) == (0, 0)
        assert t.scene_to_pixel(99.9, 99.9) == (99, 99)

    def test_scene_to_pixel_negative_returns_none(self):
        """Negative scene coords fall outside the image."""
        t = ViewTransform(width=100, height=100)
        assert t.scene_to_pixel(-5, -10) is None

    def test_scene_to_pixel_overflow_returns_none(self):
        """Coords at or beyond the dimensions fall outside the image."""
        t = ViewTransform(width=100, height=100)
        assert t.scene_to_pixel(100, 50) is None
        assert t.scene_to_pixel(50, 100) is None
        assert t.scene_to_pixel(500, 500) is None

    def test_in_bounds(self):
        """Bounds check for a 100x100 image."""
        t = ViewTransform(width=100, height=100)
        assert t.in_bounds(0, 0) is True
        assert t.in_bounds(99, 99) is True
        assert t.in_bounds(50, 50) is True
        assert t.in_bounds(-1, 0) is False
        assert t.in_bounds(0, -1) is False
        assert t.in_bounds(100, 50) is False
        assert t.in_bounds(50, 100) is False
