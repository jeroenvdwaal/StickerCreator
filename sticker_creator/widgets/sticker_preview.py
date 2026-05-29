"""Render-only sticker preview — checker background + RGBA pixmap."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QSizePolicy,
)

import numpy as np

_CHECKER_TILE = 8


def _checker_pixmap(w: int, h: int) -> QPixmap:
    pix = QPixmap(w, h)
    pix.fill(QColor(255, 255, 255))
    painter = QPainter(pix)
    shade = QColor(200, 200, 200)
    for y in range(0, h, _CHECKER_TILE):
        for x in range(0, w, _CHECKER_TILE):
            if (x // _CHECKER_TILE + y // _CHECKER_TILE) % 2:
                painter.fillRect(x, y, _CHECKER_TILE, _CHECKER_TILE, shade)
    painter.end()
    return pix


class StickerPreview(QGraphicsView):
    """Read-only sticker renderer — checkerboard background + RGBA sticker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0, 0)))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self._sticker: QPixmap | None = None
        self._sticker_array: np.ndarray | None = None

    def set_sticker(self, sticker_array: np.ndarray) -> None:
        if sticker_array is None:
            return
        sticker_array = np.ascontiguousarray(sticker_array)
        self._sticker_array = sticker_array
        height, width = sticker_array.shape[:2]
        fmt = (
            QImage.Format.Format_RGBA8888
            if sticker_array.shape[2] == 4
            else QImage.Format.Format_RGB888
        )
        qimg = QImage(
            sticker_array.data,
            width,
            height,
            sticker_array.strides[0],
            fmt,
        )
        self._sticker = QPixmap.fromImage(qimg.copy())

        self._scene.clear()
        checker = self._scene.addPixmap(_checker_pixmap(width, height))
        checker.setZValue(-1)
        self._scene.addPixmap(self._sticker)
        self._scene.setSceneRect(self._sticker.rect())
        self._fit()

    def clear(self) -> None:
        self._scene.clear()
        self._sticker = None
        self._sticker_array = None

    def get_array(self) -> np.ndarray | None:
        return self._sticker_array

    def has_sticker(self) -> bool:
        return self._sticker_array is not None

    def _fit(self) -> None:
        if self._sticker is not None:
            self.fitInView(self._sticker.rect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self._fit()
