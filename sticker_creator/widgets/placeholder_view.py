"""Empty-canvas placeholder shown when no image is loaded.

Follows the KDE HIG `PlaceholderMessage` pattern: themed icon, heading,
explanatory body, and primary actions. This is also the setup surface — the
segmentation model is chosen here (a preference applied before working on a
sticker), so the model selector and any model-related alert live here rather
than in the per-sticker inspector.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _themed(name: str, fallback: QIcon | None = None) -> QIcon:
    icon = QIcon.fromTheme(name)
    if icon.isNull() and fallback is not None:
        return fallback
    return icon


class PlaceholderView(QWidget):
    """Centered placeholder with icon + heading + body + Open/Paste actions.

    Also hosts the segmentation-model selector and the model-missing alert,
    since model choice is a setup step that happens before a sticker exists.
    """

    open_image_requested = Signal()
    paste_image_requested = Signal()
    model_selected = Signal(str)        # emitted on user choice (existing model)
    manage_models_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._suppress_model_signal = False
        self._setup_ui()
        self.setAcceptDrops(False)  # parent handles dnd

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(48, 48, 48, 48)
        outer.addStretch()

        icon_label = QLabel()
        icon = _themed("image-x-generic", _themed("insert-image"))
        if not icon.isNull():
            icon_label.setPixmap(icon.pixmap(64, 64))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(icon_label)

        heading = QLabel("Drop an image to start")
        heading.setObjectName("PlaceholderHeading")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(heading)

        body = QLabel(
            "Drag a picture here, open one from disk, "
            "or paste from the clipboard. Supported: PNG, JPG, WebP, BMP, TIFF."
        )
        body.setObjectName("PlaceholderBody")
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.setWordWrap(True)
        body.setMaximumWidth(440)
        outer.addLayout(self._centered(body))

        outer.addSpacing(16)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        open_btn = QPushButton("Open Image…")
        open_btn.setIcon(_themed("document-open"))
        open_btn.setObjectName("PrimaryButton")
        open_btn.setDefault(True)
        open_btn.setAutoDefault(True)
        open_btn.clicked.connect(self.open_image_requested.emit)
        btn_row.addWidget(open_btn)

        paste_btn = QPushButton("Paste from Clipboard")
        paste_btn.setIcon(_themed("edit-paste"))
        paste_btn.clicked.connect(self.paste_image_requested.emit)
        btn_row.addWidget(paste_btn)

        btn_row.addStretch()
        outer.addLayout(btn_row)

        # Inline alert (model missing / load errors).
        outer.addSpacing(16)
        self._alert = QFrame()
        self._alert.setObjectName("InlineAlert")
        self._alert.setProperty("severity", "info")
        self._alert.setMaximumWidth(440)
        alert_layout = QHBoxLayout(self._alert)
        alert_layout.setContentsMargins(10, 8, 8, 8)
        alert_layout.setSpacing(8)
        self._alert_label = QLabel()
        self._alert_label.setWordWrap(True)
        alert_layout.addWidget(self._alert_label, stretch=1)
        self._alert_btn = QPushButton("Manage Models…")
        self._alert_btn.clicked.connect(self.manage_models_requested.emit)
        alert_layout.addWidget(self._alert_btn)
        self._alert.setVisible(False)
        outer.addLayout(self._centered(self._alert))

        outer.addStretch()

        # Footer: de-emphasized model selector — a setup preference, not a
        # per-sticker control.
        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addStretch()

        model_label = QLabel("Segmentation model:")
        model_label.setObjectName("PlaceholderBody")
        footer.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        self.model_combo.currentIndexChanged.connect(self._on_model_index)
        footer.addWidget(self.model_combo)

        manage_link = QPushButton("Manage…")
        manage_link.setFlat(True)
        manage_link.setCursor(Qt.CursorShape.PointingHandCursor)
        manage_link.clicked.connect(self.manage_models_requested.emit)
        footer.addWidget(manage_link)

        footer.addStretch()
        outer.addLayout(footer)

        self._model_status = QLabel("")
        self._model_status.setObjectName("PlaceholderBody")
        self._model_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._model_status.setVisible(False)
        outer.addWidget(self._model_status)

    @staticmethod
    def _centered(widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(widget)
        row.addStretch()
        return row

    # ── Public API ─────────────────────────────────────────────────────────

    def set_models(self, models: list[dict], active: str | None) -> None:
        """Populate the combo with downloaded models."""
        self._suppress_model_signal = True
        self.model_combo.clear()
        downloaded = [m for m in models if m.get("is_downloaded")]
        if not downloaded:
            self.model_combo.addItem("(no models installed)", None)
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.setEnabled(True)
            for m in downloaded:
                self.model_combo.addItem(m["name"], m["name"])
            if active:
                idx = self.model_combo.findData(active)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
        self._suppress_model_signal = False

    def set_model_status(self, text: str) -> None:
        """Show inline status under the model combo. Empty hides it."""
        if text:
            self._model_status.setText(text)
            self._model_status.setVisible(True)
        else:
            self._model_status.clear()
            self._model_status.setVisible(False)

    def show_alert(self, message: str, severity: str = "info") -> None:
        self._alert_label.setText(message)
        self._alert.setProperty("severity", severity)
        self._alert.style().unpolish(self._alert)
        self._alert.style().polish(self._alert)
        self._alert.setVisible(True)

    def dismiss_alert(self) -> None:
        self._alert.setVisible(False)

    # ── Signal handlers ────────────────────────────────────────────────────

    def _on_model_index(self, index: int) -> None:
        if self._suppress_model_signal:
            return
        data = self.model_combo.itemData(index)
        if isinstance(data, str):
            self.model_selected.emit(data)
