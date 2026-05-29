"""Pure contour extraction for the mask overlay — no Qt.

The image viewer draws the SAM 2 mask boundary by tracing the mask into contour
polylines (OpenCV), then turning each into a ``QPolygonF`` for painting. Only
that last step needs Qt; the tracing is pure numpy + OpenCV. It lived inline in
``image_viewer.py`` and was *copied* into the viewer's test so the algorithm
could be exercised without a ``QApplication`` — two copies that could drift.

This module is the single home for the tracing. The viewer imports it for
rendering; the test imports the same function it ships, so the thing under test
is the thing that runs.
"""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Trace a binary mask into external contour polylines.

    Returns a list of OpenCV contours, each an ``(N, 1, 2)`` int array. The
    result is a ``list`` (not the tuple OpenCV 4.13+ returns) so callers may
    mutate it (e.g. ``.clear()``).
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return list(contours)
