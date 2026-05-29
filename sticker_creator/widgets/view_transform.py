"""Pure coordinate algebra for the image viewer — no Qt.

The viewer renders the source image 1:1 into the graphics scene (pixmap at the
scene origin, no scaling), so a scene coordinate *is* an image-pixel coordinate
in floating point. Zoom and pan live in the ``QGraphicsView`` transform and are
already folded out by ``QGraphicsView.mapToScene`` before a point reaches here.

This module owns the remaining algebra: truncating a scene point to an integer
pixel and bounds-checking it against the image. Kept Qt-free so it is unit
testable without a ``QApplication``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViewTransform:
    """Maps scene coordinates to image-pixel coordinates for an image of a
    given size. Scene units equal image-pixel units (1:1 rendering)."""

    width: int
    height: int

    @classmethod
    def from_image(cls, image: np.ndarray) -> "ViewTransform":
        """Build a transform sized to a ``(H, W, ...)`` numpy image."""
        h, w = image.shape[:2]
        return cls(width=int(w), height=int(h))

    def in_bounds(self, x: int, y: int) -> bool:
        """True if pixel ``(x, y)`` falls inside the image."""
        return 0 <= x < self.width and 0 <= y < self.height

    def scene_to_pixel(
        self, scene_x: float, scene_y: float
    ) -> tuple[int, int] | None:
        """Truncate a scene point to integer pixel coords, or None if outside.

        Truncates toward zero (matching ``int()``); the bounds check rejects
        negative results, so the truncation direction never escapes the image.
        """
        x = int(scene_x)
        y = int(scene_y)
        if self.in_bounds(x, y):
            return (x, y)
        return None
