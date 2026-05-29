"""Right-side inspector panel — sticker preview with inline controls.

Controls (border, background, export) live directly in this panel rather than
in a floating overlay. Model selection stays on the placeholder screen.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from sticker_creator.widgets.sticker_preview import StickerPreview
from sticker_creator.utils.balloon_renderer import (
    STYLE_AUTO, STYLE_SPEECH, STYLE_THOUGHT, STYLE_SHOUT, STYLE_WHISPER,
)


_SWATCH_PX = 16

# Background values must match sticker_border.BACKGROUND_*.
_BG_OPTIONS = (
    ("transparent", "Transparent"),
    ("white", "White"),
    ("black", "Black"),
)


def _swatch_icon(value: str) -> QIcon:
    px = _SWATCH_PX
    pix = QPixmap(px, px)
    if value == "transparent":
        pix.fill(QColor(255, 255, 255))
        p = QPainter(pix)
        tile = px // 2
        shade = QColor(200, 200, 200)
        p.fillRect(0, 0, tile, tile, shade)
        p.fillRect(tile, tile, tile, tile, shade)
        p.end()
    else:
        pix.fill(QColor(255, 255, 255) if value == "white" else QColor(0, 0, 0))
    return QIcon(pix)


def _hsep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setObjectName("InspectorSeparator")
    return line


class InspectorPanel(QWidget):
    """Right inspector — sticker preview with inline controls."""

    save_requested = Signal()
    copy_requested = Signal()
    border_toggled = Signal(bool)
    border_width_changed = Signal(int)
    background_changed = Signal(str)
    balloon_text_changed = Signal(str)
    balloon_style_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._setup_ui()

    # ── Layout ─────────────────────────────────────────────────────────────

    def _section_header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("InspectorSection")
        return lbl

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(0)

        # ── Preview ──────────────────────────────────────────────────────────
        self.preview = StickerPreview()
        self.preview.setObjectName("StickerPreviewWidget")
        root.addWidget(self.preview, stretch=1)

        self._dimensions_label = QLabel("No sticker yet")
        self._dimensions_label.setObjectName("InspectorHint")
        self._dimensions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._dimensions_label)

        root.addSpacing(10)
        root.addWidget(_hsep())
        root.addSpacing(10)

        # ── Border ───────────────────────────────────────────────────────────
        root.addWidget(self._section_header("Border"))
        root.addSpacing(6)

        border_row = QHBoxLayout()
        border_row.setSpacing(8)
        border_row.setContentsMargins(0, 0, 0, 0)

        self._border_check = QCheckBox("White border")
        self._border_check.setChecked(True)
        self._border_check.toggled.connect(self._on_border_toggled)
        border_row.addWidget(self._border_check)

        border_row.addStretch()

        self._width_slider = QSlider(Qt.Orientation.Horizontal)
        self._width_slider.setRange(0, 20)
        self._width_slider.setValue(7)
        self._width_slider.setFixedWidth(80)
        self._width_slider.setToolTip("Border width in pixels")
        self._width_slider.valueChanged.connect(self._on_width_changed)
        border_row.addWidget(self._width_slider)

        self._width_label = QLabel("7 px")
        self._width_label.setObjectName("InspectorHint")
        self._width_label.setFixedWidth(32)
        self._width_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        border_row.addWidget(self._width_label)

        root.addLayout(border_row)

        root.addSpacing(10)
        root.addWidget(_hsep())
        root.addSpacing(10)

        # ── Speech balloon ────────────────────────────────────────────────────
        root.addWidget(self._section_header("Speech Balloon"))
        root.addSpacing(6)

        self._balloon_input = QLineEdit()
        self._balloon_input.setPlaceholderText("Type balloon text…")
        self._balloon_input.setClearButtonEnabled(True)
        self._balloon_input.setObjectName("BalloonInput")
        root.addWidget(self._balloon_input)

        root.addSpacing(5)

        style_row = QHBoxLayout()
        style_row.setSpacing(4)
        style_row.setContentsMargins(0, 0, 0, 0)

        self._style_group = QButtonGroup(self)
        self._style_group.setExclusive(True)
        _style_opts = [
            (STYLE_AUTO,    "Auto"),
            (STYLE_SPEECH,  "Speech"),
            (STYLE_THOUGHT, "Thought"),
            (STYLE_SHOUT,   "Shout"),
            (STYLE_WHISPER, "Whisper"),
        ]
        for value, label in _style_opts:
            btn = QToolButton()
            btn.setObjectName("StyleButton")
            btn.setCheckable(True)
            btn.setChecked(value == STYLE_AUTO)
            btn.setText(label)
            btn.setToolTip(value.capitalize())
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("style_value", value)
            self._style_group.addButton(btn)
            style_row.addWidget(btn)
        style_row.addStretch()
        self._style_group.buttonClicked.connect(self._on_style_clicked)
        root.addLayout(style_row)

        # Debounce: only rebuild after the user pauses typing (or commits via
        # Enter / focus-out). 900 ms is long enough that mid-word edits don't
        # trigger a render, short enough to feel responsive at a pause.
        self._balloon_debounce = QTimer(self)
        self._balloon_debounce.setSingleShot(True)
        self._balloon_debounce.setInterval(900)
        self._balloon_debounce.timeout.connect(self._emit_balloon_text)
        self._balloon_input.textChanged.connect(lambda _: self._balloon_debounce.start())
        self._balloon_input.editingFinished.connect(self._commit_balloon_text)

        root.addSpacing(10)
        root.addWidget(_hsep())
        root.addSpacing(10)

        # ── Preview background ────────────────────────────────────────────────
        root.addWidget(self._section_header("Preview background"))
        root.addSpacing(6)

        bg_row = QHBoxLayout()
        bg_row.setSpacing(4)
        bg_row.setContentsMargins(0, 0, 0, 0)

        self._bg_group = QButtonGroup(self)
        self._bg_group.setExclusive(True)
        for value, label in _BG_OPTIONS:
            btn = QToolButton()
            btn.setObjectName("BgSwatchButton")
            btn.setCheckable(True)
            btn.setChecked(value == "transparent")
            btn.setIcon(_swatch_icon(value))
            btn.setText(label)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setToolTip(f"{label} preview background")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("bg_value", value)
            self._bg_group.addButton(btn)
            bg_row.addWidget(btn)
        bg_row.addStretch()
        self._bg_group.buttonClicked.connect(self._on_bg_clicked)
        root.addLayout(bg_row)

        root.addSpacing(10)
        root.addWidget(_hsep())
        root.addSpacing(10)

        # ── Export actions ────────────────────────────────────────────────────
        self._save_btn = QPushButton(
            QIcon.fromTheme("document-save"), "Save Sticker…"
        )
        self._save_btn.setObjectName("PrimaryButton")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save_requested)
        root.addWidget(self._save_btn)

        root.addSpacing(4)

        self._copy_btn = QPushButton(
            QIcon.fromTheme("edit-copy"), "Copy to Clipboard"
        )
        self._copy_btn.setObjectName("SecondaryButton")
        self._copy_btn.setEnabled(False)
        self._copy_btn.clicked.connect(self.copy_requested)
        root.addWidget(self._copy_btn)

    # ── Slots ───────────────────────────────────────────────────────────────

    def _on_border_toggled(self, enabled: bool) -> None:
        self._width_slider.setEnabled(enabled)
        self._width_label.setEnabled(enabled)
        self.border_toggled.emit(enabled)

    def _on_width_changed(self, value: int) -> None:
        self._width_label.setText(f"{value} px")
        self.border_width_changed.emit(value)

    def _on_bg_clicked(self, button) -> None:
        self.background_changed.emit(button.property("bg_value"))

    def _emit_balloon_text(self) -> None:
        self.balloon_text_changed.emit(self._balloon_input.text())

    def _commit_balloon_text(self) -> None:
        """Flush pending debounce immediately (Enter pressed or focus lost)."""
        if self._balloon_debounce.isActive():
            self._balloon_debounce.stop()
            self._emit_balloon_text()

    def _on_style_clicked(self, button) -> None:
        self.balloon_style_changed.emit(button.property("style_value"))

    # ── Public API ──────────────────────────────────────────────────────────

    def set_sticker(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            self.preview.clear()
            self._dimensions_label.setText("No sticker yet")
            self._save_btn.setEnabled(False)
            self._copy_btn.setEnabled(False)
            return
        self.preview.set_sticker(rgba)
        h, w = rgba.shape[:2]
        self._dimensions_label.setText(f"{w} × {h} px")
        self._save_btn.setEnabled(True)
        self._copy_btn.setEnabled(True)

    def set_border_width(self, value: int) -> None:
        """Update slider without emitting border_width_changed (avoids rebuild loop)."""
        self._width_slider.blockSignals(True)
        self._width_slider.setValue(value)
        self._width_label.setText(f"{value} px")
        self._width_slider.blockSignals(False)

    def dismiss_alert(self) -> None:
        pass  # Alerts live on the placeholder screen, not the inspector
