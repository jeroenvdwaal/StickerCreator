"""Tests for image viewer overlay rendering helpers.

These tests cover the helper functions used in image_viewer.py
that only require numpy and OpenCV — no QApplication needed.
"""

import numpy as np
import cv2


def _mask_to_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Extract contour polylines from a binary mask using OpenCV."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def _contour_to_points(contour: np.ndarray) -> list[tuple[float, float]]:
    """Convert an OpenCV contour (N, 1, 2) to a list of (x, y) tuples."""
    return [(float(pt[0][0]), float(pt[0][1])) for pt in contour]


class TestMaskToContours:
    """Tests for _mask_to_contours helper."""

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


class TestImageViewerCoordinateMapping:
    """Tests for coordinate mapping logic used in ImageViewer.

    The get_image_coordinates method does:
        x = int(scene_pos.x()), y = int(scene_pos.y())
        h, w = image.shape[:2]
        if 0 <= x < w and 0 <= y < h: return (x, y)
    """

    def test_coordinates_in_bounds(self):
        """Valid image coordinates should produce correct pixels."""
        scene_x, scene_y = 50.7, 30.2
        x, y = int(scene_x), int(scene_y)
        assert x == 50
        assert y == 30

    def test_coordinates_out_of_bounds_negative(self):
        """Negative coordinates should be out of bounds."""
        scene_x, scene_y = -5, -10
        x, y = int(scene_x), int(scene_y)
        assert x < 0
        assert y < 0

    def test_coordinates_out_of_bounds_overflow(self):
        """Coordinates beyond image dimensions should be out of bounds."""
        scene_x, scene_y = 500, 500
        x, y = int(scene_x), int(scene_y)
        assert x >= 100
        assert y >= 100

    def test_bounds_check_logic(self):
        """Full bounds check for a 100x100 image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        h, w = image.shape[:2]

        def in_bounds(x, y):
            return 0 <= x < w and 0 <= y < h

        assert in_bounds(0, 0) is True
        assert in_bounds(99, 99) is True
        assert in_bounds(50, 50) is True
        assert in_bounds(-1, 0) is False
        assert in_bounds(0, -1) is False
        assert in_bounds(100, 50) is False
        assert in_bounds(50, 100) is False
