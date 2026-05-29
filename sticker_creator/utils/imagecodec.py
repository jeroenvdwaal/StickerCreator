"""Conversions between the app's canonical image form and Qt/PIL.

The canonical form throughout the app is a contiguous ``H×W×4`` uint8 RGBA
numpy array. The subtle parts of moving that in and out of ``QImage`` — picking
``Format_RGBA8888``, supplying the right stride, and owning the buffer so it
survives the source array being freed — were previously written twice (in the
balloon renderer and the clipboard). They live here once so the correctness of
the round-trip is verified in one place.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from PySide6.QtGui import QImage


def to_qimage(rgba: np.ndarray) -> QImage:
    """RGBA numpy array → owning ``QImage`` (``Format_RGBA8888``).

    The returned image owns its pixels (``.copy()``), so the caller may
    discard *rgba* immediately.
    """
    arr = np.ascontiguousarray(rgba)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return img.copy()


def from_qimage(img: QImage) -> np.ndarray:
    """``QImage`` (any format) → owning ``H×W×4`` uint8 RGBA numpy array."""
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = img.height(), img.width()
    ptr = img.constBits()
    return np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4)).copy()


def to_pil(rgba: np.ndarray) -> Image.Image:
    """RGBA numpy array → PIL ``Image`` in ``RGBA`` mode."""
    return Image.fromarray(np.ascontiguousarray(rgba), "RGBA")


def from_pil(img: Image.Image) -> np.ndarray:
    """PIL ``Image`` (any mode) → owning ``H×W×4`` uint8 RGBA numpy array."""
    return np.asarray(img.convert("RGBA"), dtype=np.uint8)


def decode_png(data: bytes) -> np.ndarray:
    """Decode PNG/any-PIL-readable bytes → ``H×W×4`` uint8 RGBA numpy array."""
    return from_pil(Image.open(io.BytesIO(data)))
