"""Icon factory for in-canvas annotation toolbar buttons.

Theme icons are preferred everywhere else via QIcon.fromTheme; this module
exists only for the small set of glyphs that don't have a clean Breeze
equivalent for the image-viewer's mode buttons.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QIcon,
    QPixmap,
    QPainter,
    QColor,
    QPen,
)


_ICON_SIZE = 24


def _pixmap() -> QPixmap:
    pm = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pm.fill(Qt.GlobalColor.transparent)
    return pm


def _paint(painter_fn) -> QIcon:
    pm = _pixmap()
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter_fn(painter)
    painter.end()
    return QIcon(pm)


def _pen(color: str, width: int = 2) -> QPen:
    return QPen(QColor(color), width)


def _paint_undo(p: QPainter) -> None:
    p.setPen(_pen("#2c3e50", 2))
    p.drawLine(7, 12, 18, 12)
    p.drawLine(7, 12, 10, 8)
    p.drawLine(7, 12, 10, 16)


def _paint_hand(p: QPainter) -> None:
    p.setPen(_pen("#2c3e50", 2))
    p.drawEllipse(7, 9, 10, 10)
    for dx in (-4, 0, 4):
        p.drawEllipse(10 + dx, 3, 4, 8)


def _paint_plus(p: QPainter) -> None:
    p.setPen(_pen("#27ae60", 2))
    p.drawEllipse(3, 3, 18, 18)
    p.setPen(_pen("#27ae60", 3))
    p.drawLine(8, 12, 16, 12)
    p.drawLine(12, 8, 12, 16)


def _paint_minus(p: QPainter) -> None:
    p.setPen(_pen("#e74c3c", 2))
    p.drawEllipse(3, 3, 18, 18)
    p.setPen(_pen("#e74c3c", 3))
    p.drawLine(8, 12, 16, 12)


def _paint_trash(p: QPainter) -> None:
    p.setPen(_pen("#c0392b"))
    p.drawRect(5, 9, 14, 12)
    p.drawLine(3, 7, 21, 7)
    p.drawRect(9, 4, 6, 3)
    p.drawLine(9, 12, 9, 18)
    p.drawLine(12, 12, 12, 18)
    p.drawLine(15, 12, 15, 18)


def icon_undo() -> QIcon:
    return _paint(_paint_undo)


def icon_pan() -> QIcon:
    return _paint(_paint_hand)


def icon_add_positive() -> QIcon:
    return _paint(_paint_plus)


def icon_add_negative() -> QIcon:
    return _paint(_paint_minus)


def icon_clear() -> QIcon:
    return _paint(_paint_trash)
