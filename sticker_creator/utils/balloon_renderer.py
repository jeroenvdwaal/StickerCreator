"""Comic-strip speech balloon renderer.

The balloon is composited onto the sticker RGBA array *before* the white
sticker border is applied, so the balloon becomes part of the sticker
silhouette and the single continuous white border wraps both the subject
and the balloon.  The renderer therefore draws only the balloon **fill**
and **text** (no outline of its own) — the outline is supplied later by
``sticker_border.add_white_border``.

Placement uses the alpha channel: the balloon sits in the top-left or
top-right empty space (whichever side has more room), above the subject's
head, and the canvas is expanded on that side / upward when the text needs
more room.  The tail drops diagonally to the subject's top silhouette edge
near the mouth and stops at the edge, so it never crosses over the face.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
)

_FONT_PATH = (
    Path(__file__).parent.parent / "resources" / "fonts" / "Bangers-Regular.ttf"
)
_FONT_ID: int = -2   # -2 = not yet attempted

STYLE_AUTO = "auto"
STYLE_SPEECH = "speech"
STYLE_THOUGHT = "thought"
STYLE_SHOUT = "shout"
STYLE_WHISPER = "whisper"

SIDE_LEFT = "left"
SIDE_RIGHT = "right"

_TEXT_COLOR = QColor(20, 20, 20)
_FILL_WHITE = QColor(255, 255, 255)
_FILL_YELLOW = QColor(255, 218, 0)


# ── Font ──────────────────────────────────────────────────────────────────────

def _font_family() -> str:
    global _FONT_ID
    if _FONT_ID == -2:
        if _FONT_PATH.exists():
            _FONT_ID = QFontDatabase.addApplicationFont(str(_FONT_PATH))
        else:
            _FONT_ID = -1
    if _FONT_ID >= 0:
        families = QFontDatabase.applicationFontFamilies(_FONT_ID)
        if families:
            return families[0]
    return "Comic Sans MS"


def _make_font(size: int, bold: bool = False, italic: bool = False) -> QFont:
    f = QFont(_font_family(), size)
    f.setBold(bold)
    f.setItalic(italic)
    f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 108)
    return f


def _style_font(resolved: str, base_size: int) -> QFont:
    """Per-style font: shout is bold, whisper is smaller + italic (quiet)."""
    if resolved == STYLE_SHOUT:
        return _make_font(base_size, bold=True)
    if resolved == STYLE_WHISPER:
        return _make_font(max(15, int(base_size * 0.72)), italic=True)
    return _make_font(base_size)


# ── Style detection ───────────────────────────────────────────────────────────

def detect_style(text: str) -> str:
    """Infer balloon style from text content."""
    t = text.strip()
    if re.search(r"[zZ]{2,}|\.{3,}|💤|💭|🌙|🫧", t):
        return STYLE_THOUGHT
    if (t.startswith("(") and t.endswith(")")) or (
        t.startswith("*") and t.endswith("*")
    ):
        return STYLE_WHISPER
    if t.count("!") >= 2 or (
        len(t) > 2 and t == t.upper() and any(c.isalpha() for c in t)
    ):
        return STYLE_SHOUT
    return STYLE_SPEECH


# ── Public entry point ────────────────────────────────────────────────────────

def render_balloon(
    sticker: np.ndarray,
    text: str,
    style: str = STYLE_AUTO,
) -> np.ndarray:
    """Return a new RGBA array with the balloon merged into *sticker*.

    The balloon is drawn as an opaque fill (plus text) and added to the
    sticker's alpha so a later white-border pass wraps subject and balloon
    together.  No border and no tail are drawn here — the balloon simply
    sits beside the subject, just touching the silhouette.

    Parameters
    ----------
    sticker:
        H×W×4 RGBA uint8 sticker array (no border yet).
    text:
        Balloon text (may contain newlines).
    style:
        ``"auto"`` (default) infers from *text*; otherwise one of
        ``"speech"``, ``"thought"``, ``"shout"``, ``"whisper"``.
    """
    if not text.strip():
        return sticker

    resolved = detect_style(text) if style == STYLE_AUTO else style
    h, w = sticker.shape[:2]

    # ── Font + text measurement ───────────────────────────────────────────
    base_size = max(22, w // 13)
    font = _style_font(resolved, base_size)
    fm = QFontMetrics(font)

    max_text_w = int(w * 0.58)
    pad_x, pad_y = 22, 16

    lines = _wrap(text.strip(), fm, max_text_w - 2 * pad_x)
    text_w = max(fm.horizontalAdvance(ln) for ln in lines)
    text_h = fm.height() * len(lines)

    balloon_w = text_w + 2 * pad_x
    balloon_h = text_h + 2 * pad_y

    # ── Side selection: more empty margin wins ────────────────────────────
    _, _, subj_left, subj_right = _subject_bounds(sticker)
    left_space = subj_left
    right_space = (w - 1) - subj_right
    side = SIDE_RIGHT if right_space >= left_space else SIDE_LEFT

    # Balloon sits BESIDE the subject, pinned to the top of the image and
    # snug against the subject so it just touches the silhouette (a small
    # overlap fuses them under one border).  It never covers the face; when
    # the subject fills the frame the canvas grows on that side to make room.
    margin = 8
    b_top = margin

    # Some shapes bulge past their rect: the shout starburst (~22 %) and the
    # thought cloud's lobes (~0.42·r).  Reserve that as overflow so the body
    # only just touches the subject instead of poking into it.
    if resolved == STYLE_SHOUT:
        overflow = int(max(balloon_w, balloon_h) * 0.22)
    elif resolved == STYLE_THOUGHT:
        overflow = int(min(balloon_w, balloon_h) * 0.5 * 0.42) + 2
    else:
        overflow = 0

    # Subject edge within the balloon's vertical band, so the body touches at
    # the balloon's own height rather than at some distant widest row.  A
    # small overlap makes the balloon nudge into the subject a little.
    overlap = max(8, w // 40)
    band_edge = _side_edge_in_band(
        sticker, b_top, b_top + balloon_h, side, subj_left, subj_right
    )
    if side == SIDE_RIGHT:
        b_left = band_edge - overlap + overflow
    else:
        b_left = band_edge + overlap - balloon_w - overflow

    # ── Canvas expansion (grow the chosen side / vertically as needed) ────
    expand_left = max(0, -(b_left - margin) + overflow)
    expand_right = max(0, (b_left + balloon_w + margin + overflow) - w)
    expand_top = overflow
    expand_bottom = max(0, (b_top + balloon_h + margin + overflow) - h)

    new_w = w + expand_left + expand_right
    new_h = h + expand_top + expand_bottom
    sx = expand_left   # sticker origin in new canvas
    sy = expand_top

    # Adjusted coordinates in the new canvas
    bx = b_left + sx
    by = b_top + sy
    balloon_rect = QRectF(bx, by, balloon_w, balloon_h)

    # ── Render: subject first, balloon fill + text on top ─────────────────
    canvas = QImage(new_w, new_h, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(Qt.GlobalColor.transparent)

    p = QPainter(canvas)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    p.drawImage(sx, sy, _to_qimage(sticker))

    draw = {
        STYLE_SPEECH: _draw_speech,
        STYLE_THOUGHT: _draw_thought,
        STYLE_SHOUT: _draw_shout,
        STYLE_WHISPER: _draw_whisper,
    }[resolved]
    draw(p, balloon_rect, font, fm, lines, pad_x, pad_y)

    p.end()
    return _from_qimage(canvas)


# ── Balloon draw functions (fill + text only; border added later) ───────────────

def _fill(painter: QPainter, path: QPainterPath, color: QColor) -> None:
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawPath(path)


def _draw_speech(
    painter: QPainter,
    rect: QRectF,
    font: QFont,
    fm: QFontMetrics,
    lines: list[str],
    pad_x: int,
    pad_y: int,
) -> None:
    r = min(rect.height() * 0.38, 18.0)
    body = QPainterPath()
    body.addRoundedRect(rect, r, r)
    _fill(painter, body, _FILL_WHITE)
    _text(painter, font, fm, lines, rect, pad_x, pad_y, _TEXT_COLOR)


def _draw_whisper(
    painter: QPainter,
    rect: QRectF,
    font: QFont,
    fm: QFontMetrics,
    lines: list[str],
    pad_x: int,
    pad_y: int,
) -> None:
    # Quiet voice: a small, very-rounded pill.  The smaller italic font (set
    # in _style_font) carries the "whisper" read, since a dashed outline
    # can't survive being merged into the border.
    r = min(rect.height() * 0.5, 26.0)
    body = QPainterPath()
    body.addRoundedRect(rect, r, r)
    _fill(painter, body, _FILL_WHITE)
    _text(painter, font, fm, lines, rect, pad_x, pad_y, QColor(90, 90, 90))


def _draw_thought(
    painter: QPainter,
    rect: QRectF,
    font: QFont,
    fm: QFontMetrics,
    lines: list[str],
    pad_x: int,
    pad_y: int,
) -> None:
    # Dream/thought: a lumpy cloud body.
    _fill(painter, _cloud(rect), _FILL_WHITE)
    _text(painter, font, fm, lines, rect, pad_x, pad_y, _TEXT_COLOR)


def _draw_shout(
    painter: QPainter,
    rect: QRectF,
    font: QFont,
    fm: QFontMetrics,
    lines: list[str],
    pad_x: int,
    pad_y: int,
) -> None:
    cx = rect.center().x()
    cy = rect.center().y()
    rx_out = rect.width() / 2 * 1.22
    ry_out = rect.height() / 2 * 1.22
    rx_in = rect.width() / 2 * 0.88
    ry_in = rect.height() / 2 * 0.88

    burst = _starburst(cx, cy, rx_out, ry_out, rx_in, ry_in, spikes=16)
    _fill(painter, burst, _FILL_YELLOW)
    _text(painter, font, fm, lines, rect, pad_x, pad_y, _TEXT_COLOR)


# ── Shape helpers ─────────────────────────────────────────────────────────────

def _cloud(rect: QRectF) -> QPainterPath:
    """Lumpy cloud silhouette (for dream/thought bubbles)."""
    p = QPainterPath()
    p.addEllipse(rect)
    cx, cy = rect.center().x(), rect.center().y()
    rx, ry = rect.width() / 2, rect.height() / 2
    bump = min(rx, ry) * 0.42
    n = 11
    for i in range(n):
        a = 2 * math.pi * i / n
        px = cx + rx * 0.96 * math.cos(a)
        py = cy + ry * 0.96 * math.sin(a)
        lobe = QPainterPath()
        lobe.addEllipse(QPointF(px, py), bump, bump)
        p = p.united(lobe)
    return p


def _starburst(
    cx: float, cy: float,
    rx_out: float, ry_out: float,
    rx_in: float, ry_in: float,
    spikes: int = 16,
) -> QPainterPath:
    """Elliptical starburst (for shout bubbles)."""
    p = QPainterPath()
    total = spikes * 2
    for i in range(total):
        angle = math.pi * 2 * i / total - math.pi / 2
        if i % 2 == 0:
            x = cx + rx_out * math.cos(angle)
            y = cy + ry_out * math.sin(angle)
        else:
            x = cx + rx_in * math.cos(angle)
            y = cy + ry_in * math.sin(angle)
        if i == 0:
            p.moveTo(x, y)
        else:
            p.lineTo(x, y)
    p.closeSubpath()
    return p


# ── Text helper ───────────────────────────────────────────────────────────────

def _text(
    painter: QPainter,
    font: QFont,
    fm: QFontMetrics,
    lines: list[str],
    rect: QRectF,
    pad_x: int,
    pad_y: int,
    color: QColor,
) -> None:
    painter.setFont(font)
    painter.setPen(color)
    total_h = fm.height() * len(lines)
    y = rect.top() + (rect.height() - total_h) / 2 + fm.ascent()
    for line in lines:
        lw = fm.horizontalAdvance(line)
        x = rect.left() + (rect.width() - lw) / 2
        painter.drawText(QPointF(x, y), line)
        y += fm.height()


def _wrap(text: str, fm: QFontMetrics, max_width: int) -> list[str]:
    """Word-wrap *text* to fit within *max_width* pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if fm.horizontalAdvance(candidate) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


# ── Subject geometry ───────────────────────────────────────────────────────────

def _subject_bounds(sticker: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(top, bottom, left, right)`` of the opaque subject (alpha > 10).

    Falls back to the full frame for an empty alpha channel.
    """
    h, w = sticker.shape[:2]
    ys, xs = np.where(sticker[:, :, 3] > 10)
    if ys.size == 0:
        return 0, h - 1, 0, w - 1
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _side_edge_in_band(
    sticker: np.ndarray, y0: int, y1: int, side: str,
    default_left: int, default_right: int,
) -> int:
    """Outermost opaque column on *side* across rows ``[y0, y1)``.

    Used to seat the balloon against the subject at the balloon's own
    height.  Falls back to the subject's overall bound when the band holds
    no subject pixels.
    """
    h = sticker.shape[0]
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    cols = np.where((sticker[y0:y1, :, 3] > 10).any(axis=0))[0]
    if cols.size == 0:
        return default_right if side == SIDE_RIGHT else default_left
    return int(cols.max()) if side == SIDE_RIGHT else int(cols.min())


# ── QImage ↔ numpy ────────────────────────────────────────────────────────────

from sticker_creator.utils.imagecodec import (  # noqa: E402
    to_qimage as _to_qimage,
    from_qimage as _from_qimage,
)
