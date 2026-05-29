"""Drag-and-drop overlay widget that appears when files are dragged over the window.

Shows a semi-transparent "Drop Image Here" prompt with supported formats.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

# Image file extensions accepted via drag-and-drop
ACCEPTED_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

_OVERLAY_COLOR = QColor(61, 174, 233, 40)  # KDE blue, very transparent
_BORDER_COLOR = QColor(61, 174, 233, 200)  # KDE blue, more opaque
_TEXT_COLOR = QColor(61, 174, 233, 220)


class DropOverlay(QWidget):
    """Full-window overlay shown during a drag-and-drop operation.

    Usage::

        overlay = DropOverlay(parent_widget)
        overlay.hide()  # hidden by default
        # Call overlay.show() during dragEnterEvent
        # Call overlay.hide() during dragLeaveEvent / dropEvent
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)
        self.setStyleSheet("background: transparent;")
        self.hide()

        # Track whether we're currently showing
        self._active = False

    def paintEvent(self, event) -> None:
        """Paint the semi-transparent overlay with drop zone hint."""
        if not self._active:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Fill entire area with semi-transparent overlay
        painter.fillRect(self.rect(), _OVERLAY_COLOR)

        # Calculate a centered box (60% of widget size)
        rect = self.rect()
        box_w = int(rect.width() * 0.5)
        box_h = int(rect.height() * 0.35)
        box_x = (rect.width() - box_w) // 2
        box_y = (rect.height() - box_h) // 2
        drop_rect = QRect(box_x, box_y, box_w, box_h)

        # Draw dashed border
        pen = QPen(_BORDER_COLOR, 3)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setDashPattern([8, 6])
        painter.setPen(pen)
        painter.setBrush(QColor(61, 174, 233, 15))  # even lighter fill
        painter.drawRoundedRect(drop_rect, 12, 12)

        # Draw icon text
        painter.setPen(_TEXT_COLOR)
        icon_font = QFont()
        icon_font.setPixelSize(36)
        painter.setFont(icon_font)
        painter.drawText(
            drop_rect.adjusted(0, -drop_rect.height() // 6, 0, 0),
            Qt.AlignmentFlag.AlignCenter,
            "📁",
        )

        # Draw "Drop Image Here" text
        text_font = QFont()
        text_font.setPixelSize(18)
        text_font.setBold(True)
        painter.setFont(text_font)
        painter.drawText(
            drop_rect.adjusted(0, drop_rect.height() // 8, 0, 0),
            Qt.AlignmentFlag.AlignCenter,
            "Drop Image Here",
        )

        # Draw supported formats
        formats_font = QFont()
        formats_font.setPixelSize(11)
        painter.setFont(formats_font)
        formats_text = "PNG · JPG · WebP · BMP · TIFF"
        painter.drawText(
            drop_rect.adjusted(0, drop_rect.height() // 4, 0, 0),
            Qt.AlignmentFlag.AlignCenter,
            formats_text,
        )

        painter.end()

    def showEvent(self, event) -> None:
        """Ensure we cover the full parent when shown."""
        self._active = True
        if self.parentWidget():
            self.resize(self.parentWidget().size())
            self.raise_()
        super().showEvent(event)
        self.update()

    def hideEvent(self, event) -> None:
        """Reset active flag when hidden."""
        self._active = False
        super().hideEvent(event)
        self.update()

    def activate(self) -> None:
        """Show the drop zone overlay."""
        self.show()
        self.raise_()

    def deactivate(self) -> None:
        """Hide the drop zone overlay."""
        self.hide()
