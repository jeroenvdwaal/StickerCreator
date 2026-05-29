"""First-run welcome dialog — text intro + 5-frame slideshow tour."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


_RES_DIR = Path(__file__).resolve().parent.parent / "resources" / "welcome"
_BEFORE = _RES_DIR / "before.jpg"
_AFTER = _RES_DIR / "after.png"
_SLIDE_PATHS = sorted(_RES_DIR.glob("slide_*.png"))

_SLIDE_TITLES = [
    "",                                              # intro slide — its own heading
    "1. Open an image",
    "2. Click the face — SAM 2 cuts it out",
    "3. Click the hair to include the whole head",
    "4. Type the speech-balloon text",
    "5. Save as PNG, WebP, or WhatsApp sticker",
]

_SLIDE_W = 720
_SLIDE_H = 497   # preserves the 900×621 source aspect ratio
_THUMB_PX = 180
_INTRO_THUMB_PX = 170

_INTRO_HTML = """
<h1 style="margin:0 0 14px 0; font-size:30px;">Welcome to Sticker Creator</h1>
<p style="margin:0 0 22px 0; font-size:16px; line-height:140%;">
Turn any photo into a transparent sticker — ready for WhatsApp,
Telegram, Signal, or any chat app.</p>

<p style="margin:0 0 8px 0; font-size:17px;"><b>How it works</b></p>
<ul style="margin:0 0 0 22px; padding:0; font-size:15px; line-height:165%;">
  <li>Open an image (file, clipboard, or drag-and-drop).</li>
  <li>Click the subject — SAM 2 cuts it out.</li>
  <li>Add a speech balloon and a white border.</li>
  <li>Save as PNG, WebP, or 512×512 WhatsApp sticker.</li>
</ul>
"""

_INTRO_FOOTER = (
    "Use the ‹ and › arrows below — or wait — to watch the quick tour."
)


def _thumb(path: Path, caption: str, size: int = _THUMB_PX) -> QWidget:
    box = QFrame()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)

    img = QLabel()
    img.setFixedSize(size, size)
    img.setAlignment(Qt.AlignmentFlag.AlignCenter)
    img.setStyleSheet(
        "background-color: palette(alternate-base);"
        "border: 1px solid palette(mid);"
        "border-radius: 8px;"
    )
    pix = QPixmap(str(path))
    if not pix.isNull():
        img.setPixmap(pix.scaled(
            size - 4, size - 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
    lay.addWidget(img, 0, Qt.AlignmentFlag.AlignCenter)

    cap = QLabel(caption)
    cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
    cap.setStyleSheet("color: palette(mid); font-weight: 600; font-size: 13px;")
    lay.addWidget(cap)
    return box


def _build_intro_slide() -> QWidget:
    page = QWidget()
    page.setFixedSize(_SLIDE_W, _SLIDE_H)

    outer = QVBoxLayout(page)
    outer.setContentsMargins(40, 28, 40, 24)
    outer.setSpacing(0)

    body = QHBoxLayout()
    body.setSpacing(36)

    text = QLabel(_INTRO_HTML)
    text.setWordWrap(True)
    text.setTextFormat(Qt.TextFormat.RichText)
    text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    body.addWidget(text, 1)

    compare = QVBoxLayout()
    compare.setSpacing(12)
    compare.addStretch()
    compare.addWidget(
        _thumb(_BEFORE, "Original", _INTRO_THUMB_PX),
        0, Qt.AlignmentFlag.AlignCenter,
    )
    arrow = QLabel("↓")
    arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
    arrow.setStyleSheet("font-size: 32px; color: palette(mid);")
    compare.addWidget(arrow)
    compare.addWidget(
        _thumb(_AFTER, "Sticker", _INTRO_THUMB_PX),
        0, Qt.AlignmentFlag.AlignCenter,
    )
    attribution = QLabel("Original from thispersondoesnotexist.com")
    attribution.setAlignment(Qt.AlignmentFlag.AlignCenter)
    attribution.setStyleSheet("color: palette(mid); font-size: 11px;")
    compare.addWidget(attribution, 0, Qt.AlignmentFlag.AlignCenter)
    compare.addStretch()
    body.addLayout(compare)
    outer.addLayout(body, 1)

    footer = QLabel(_INTRO_FOOTER)
    footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
    footer.setStyleSheet("color: palette(mid); font-size: 13px;")
    outer.addWidget(footer, 0, Qt.AlignmentFlag.AlignBottom)
    return page


def _build_image_slide(path: Path) -> QWidget:
    page = QWidget()
    page.setFixedSize(_SLIDE_W, _SLIDE_H)
    lay = QVBoxLayout(page)
    lay.setContentsMargins(0, 0, 0, 0)

    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    pix = QPixmap(str(path))
    if not pix.isNull():
        label.setPixmap(pix.scaled(
            _SLIDE_W, _SLIDE_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
    else:
        label.setText(f"(missing {path.name})")
    lay.addWidget(label)
    return page


class _DotIndicator(QWidget):
    """Page dots — filled circle marks the active slide."""

    _DOT_SPACING = 14
    _DOT_RADIUS = 5

    def __init__(self, count: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._count = count
        self._current = 0
        self.setFixedHeight(self._DOT_RADIUS * 2 + 8)
        width = count * self._DOT_SPACING + self._DOT_RADIUS * 2
        self.setFixedWidth(width)

    def set_current(self, idx: int) -> None:
        if idx == self._current:
            return
        self._current = idx
        self.update()

    def paintEvent(self, _event):  # type: ignore[override]
        from PySide6.QtGui import QColor, QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        active = self.palette().highlight().color()
        inactive = QColor(self.palette().mid().color())
        inactive.setAlpha(140)
        y = self.height() // 2
        total = (self._count - 1) * self._DOT_SPACING
        x0 = (self.width() - total) // 2
        for i in range(self._count):
            cx = x0 + i * self._DOT_SPACING
            if i == self._current:
                p.setBrush(active)
                p.setPen(active)
            else:
                p.setBrush(inactive)
                p.setPen(inactive)
            p.drawEllipse(cx - self._DOT_RADIUS, y - self._DOT_RADIUS,
                          self._DOT_RADIUS * 2, self._DOT_RADIUS * 2)
        p.end()


class WelcomeDialog(QDialog):
    """Modal first-run intro. Returns whether to suppress future shows."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome — Sticker Creator")
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 14)
        root.setSpacing(12)

        # ── Step title (rendered above the slide, not inside the PNG) ────────
        self._title = QLabel()
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(
            "QLabel { font-size: 20px; font-weight: 700;"
            "color: palette(text); padding: 6px 0; }"
        )
        self._title.setMinimumHeight(36)
        root.addWidget(self._title)

        # ── Slide carousel: ◀  [stack]  ▶ ─────────────────────────────────────
        carousel = QHBoxLayout()
        carousel.setSpacing(6)

        self._prev_btn = self._arrow_button("‹", "Previous slide")
        self._prev_btn.clicked.connect(self._go_prev)
        carousel.addWidget(self._prev_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        self._stack = QStackedWidget()
        self._stack.setFixedSize(_SLIDE_W, _SLIDE_H)
        self._stack.setStyleSheet(
            "QStackedWidget { border: 1px solid palette(mid); border-radius: 6px;"
            "background-color: palette(alternate-base); }"
        )
        self._stack.addWidget(_build_intro_slide())
        for path in _SLIDE_PATHS:
            self._stack.addWidget(_build_image_slide(path))
        carousel.addWidget(self._stack, 1)

        self._next_btn = self._arrow_button("›", "Next slide")
        self._next_btn.clicked.connect(self._go_next)
        carousel.addWidget(self._next_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(carousel)

        # ── Page dots ────────────────────────────────────────────────────────
        self._dots = _DotIndicator(self._stack.count())
        dots_row = QHBoxLayout()
        dots_row.addStretch()
        dots_row.addWidget(self._dots)
        dots_row.addStretch()
        root.addLayout(dots_row)

        # ── Footer ───────────────────────────────────────────────────────────
        self._dont_show = QCheckBox("Don't show this again")
        root.addWidget(self._dont_show)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Get Started")
        ok_btn.setDefault(True)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._stack.currentChanged.connect(self._on_slide_changed)
        self._on_slide_changed(0)

    # ── Slide navigation ─────────────────────────────────────────────────────

    def _arrow_button(self, glyph: str, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(glyph)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setAutoRaise(True)
        btn.setFixedSize(48, 80)
        btn.setStyleSheet(
            "QToolButton { font-size: 36px; font-weight: 700;"
            "color: palette(text); border: none; padding: 0; }"
            "QToolButton:hover { color: palette(highlight); }"
            "QToolButton:disabled { color: palette(mid); }"
        )
        return btn

    def _go_prev(self) -> None:
        count = self._stack.count()
        self._stack.setCurrentIndex((self._stack.currentIndex() - 1) % count)

    def _go_next(self) -> None:
        count = self._stack.count()
        self._stack.setCurrentIndex((self._stack.currentIndex() + 1) % count)

    def _on_slide_changed(self, idx: int) -> None:
        self._dots.set_current(idx)
        title = _SLIDE_TITLES[idx] if idx < len(_SLIDE_TITLES) else ""
        self._title.setText(title)
        # Reserve the slot so the slide area doesn't jump when the title is empty.
        self._title.setVisible(True)

    # ── Public ───────────────────────────────────────────────────────────────

    def dont_show_again(self) -> bool:
        return self._dont_show.isChecked()

    def keyPressEvent(self, event):  # type: ignore[override]
        if event.key() == Qt.Key.Key_Left:
            self._go_prev()
        elif event.key() == Qt.Key.Key_Right:
            self._go_next()
        else:
            super().keyPressEvent(event)

