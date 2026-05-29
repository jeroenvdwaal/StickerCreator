"""Passive notification overlay — KDE HIG-aligned.

Mirrors `Kirigami.showPassiveNotification`: subtle bottom-anchored bubble using
the system palette, themed icons, no saturated brand colors. For low-priority
confirmations of actions whose result is not visible on screen (file saved,
clipboard copy).
"""

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QPoint, QEasingCurve
from PySide6.QtGui import QColor, QPainter, QBrush, QPen, QIcon, QPalette
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStyle


_TOAST_MIN_WIDTH = 240
_TOAST_MAX_WIDTH = 480
_TOAST_HEIGHT = 40
_TOAST_MARGIN = 16
_TOAST_SPACING = 6
_DISPLAY_MS = 4000
_ANIM_DURATION_MS = 220
_CORNER_RADIUS = 6
_ICON_PX = 16

_TYPE_ICONS = {
    "success": ("dialog-positive", "emblem-success", "dialog-ok"),
    "error":   ("dialog-error",),
    "info":    ("dialog-information",),
    "warning": ("dialog-warning",),
}


def _themed_icon(names: tuple[str, ...]) -> QIcon:
    for n in names:
        ic = QIcon.fromTheme(n)
        if not ic.isNull():
            return ic
    return QIcon()


class _ToastWidget(QWidget):
    """A single passive-notification bubble — palette-themed, self-dismissing."""

    def __init__(
        self,
        message: str,
        toast_type: str = "info",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._toast_type = toast_type

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        icon_names = _TYPE_ICONS.get(toast_type, _TYPE_ICONS["info"])
        icon = _themed_icon(icon_names)
        icon_label = QLabel()
        icon_label.setFixedSize(_ICON_PX, _ICON_PX)
        if not icon.isNull():
            icon_label.setPixmap(icon.pixmap(_ICON_PX, _ICON_PX))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        self._msg_label = QLabel(message)
        self._msg_label.setWordWrap(False)
        self._msg_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        layout.addWidget(self._msg_label, stretch=1)

        # Resolve palette-derived colors at construction (parent palette wins).
        pal = (parent.palette() if parent is not None else self.palette())
        self._bg_color = QColor(pal.color(QPalette.ColorRole.Window))
        self._bg_color.setAlpha(245)
        self._fg_color = pal.color(QPalette.ColorRole.WindowText)
        self._border_color = pal.color(QPalette.ColorRole.Mid)
        self._msg_label.setStyleSheet(f"color: {self._fg_color.name()};")

        # Compute width from message and clamp.
        fm = self._msg_label.fontMetrics()
        text_w = fm.horizontalAdvance(message)
        width = min(_TOAST_MAX_WIDTH, max(_TOAST_MIN_WIDTH, text_w + _ICON_PX + 56))
        self.setFixedSize(width, _TOAST_HEIGHT)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

        self._anim = QPropertyAnimation(self, b"pos")
        self._anim.setDuration(_ANIM_DURATION_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def start(self, target_pos: QPoint) -> None:
        """Slide up into place from below."""
        start_pos = QPoint(target_pos.x(), target_pos.y() + _TOAST_HEIGHT + _TOAST_MARGIN)
        self.move(start_pos)
        self.show()
        self._anim.setStartValue(start_pos)
        self._anim.setEndValue(target_pos)
        self._anim.start()
        self._timer.start(_DISPLAY_MS)

    def _on_timeout(self) -> None:
        """Slide down out of view, then delete."""
        end_pos = QPoint(self.x(), self.y() + _TOAST_HEIGHT + _TOAST_MARGIN)
        self._anim.setStartValue(self.pos())
        self._anim.setEndValue(end_pos)
        self._anim.finished.connect(self.deleteLater)
        self._anim.start()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        """Subtle palette-themed bubble with 1px contrasting outline."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setBrush(QBrush(self._bg_color))
        pen = QPen(self._border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, _CORNER_RADIUS, _CORNER_RADIUS)


class ToastOverlay(QWidget):
    """Overlay managing stacked passive notifications, anchored bottom-center."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, _TOAST_MARGIN)
        self._layout.setSpacing(_TOAST_SPACING)
        self._layout.setAlignment(
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter
        )

        self.raise_()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().resizeEvent(event)

    def _show(self, message: str, toast_type: str) -> None:
        toast = _ToastWidget(message, toast_type, self)
        self._layout.addWidget(toast)

        parent_rect = self.rect()
        target_x = (parent_rect.width() - toast.width()) // 2
        target_y = parent_rect.height() - _TOAST_MARGIN - toast.height()

        existing_count = self._layout.count()
        offset_y = (existing_count - 1) * (_TOAST_HEIGHT + _TOAST_SPACING)
        target_pos = QPoint(target_x, target_y - offset_y)

        toast.start(target_pos)

    def show_success(self, message: str) -> None:
        self._show(message, "success")

    def show_error(self, message: str) -> None:
        self._show(message, "error")

    def show_info(self, message: str) -> None:
        self._show(message, "info")

    def show_warning(self, message: str) -> None:
        self._show(message, "warning")
