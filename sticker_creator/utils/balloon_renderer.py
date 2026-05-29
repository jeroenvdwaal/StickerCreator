"""Comic-strip speech balloon renderer (pure PIL — no Qt).

The balloon is composited onto the sticker RGBA array *before* the white
sticker border is applied, so the balloon becomes part of the sticker
silhouette and the single continuous white border wraps both the subject
and the balloon.  The renderer therefore draws only the balloon **fill**
and **text** (no outline of its own) — the outline is supplied later by
``sticker_border.add_white_border``.

Placement uses the alpha channel: the balloon sits in the top-left or
top-right empty space (whichever side has more room), above the subject's
head, and the canvas is expanded on that side / upward when the text needs
more room.

This module uses only Pillow + numpy so the whole load→mask→sticker pipeline
runs headless, without a Qt event loop.  PIL does not antialias filled
shapes, so the balloon graphics are drawn on a supersampled layer and
downscaled with LANCZOS to get smooth edges; the layer is then alpha-composited
over the subject (subject underneath, balloon on top), matching the old
QPainter draw order.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sticker_creator.utils.imagecodec import from_pil as _from_pil, to_pil as _to_pil

_FONT_PATH = (
    Path(__file__).parent.parent / "resources" / "fonts" / "Bangers-Regular.ttf"
)

STYLE_AUTO = "auto"
STYLE_SPEECH = "speech"
STYLE_THOUGHT = "thought"
STYLE_SHOUT = "shout"
STYLE_WHISPER = "whisper"

SIDE_LEFT = "left"
SIDE_RIGHT = "right"

_TEXT_COLOR = (20, 20, 20, 255)
_WHISPER_COLOR = (90, 90, 90, 255)
_FILL_WHITE = (255, 255, 255, 255)
_FILL_YELLOW = (255, 218, 0, 255)

# Supersampling factor for the balloon graphics layer (PIL has no shape AA).
_SS = 4


# ── Font ──────────────────────────────────────────────────────────────────────

def _make_font(size: int) -> ImageFont.FreeTypeFont:
    """Bangers at *size* px, falling back to PIL's default if the TTF is absent."""
    if _FONT_PATH.exists():
        return ImageFont.truetype(str(_FONT_PATH), size)
    return ImageFont.load_default(size)


def _style_size(resolved: str, base_size: int) -> int:
    """Per-style point size: whisper is smaller (quiet); others use the base."""
    if resolved == STYLE_WHISPER:
        return max(15, int(base_size * 0.72))
    return base_size


def _line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


# Shared zero-size canvas for text measurement (no event loop needed).
_MEASURE = ImageDraw.Draw(Image.new("RGBA", (1, 1)))


def _advance(font: ImageFont.FreeTypeFont, text: str) -> float:
    return _MEASURE.textlength(text, font=font)


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
    font = _make_font(_style_size(resolved, base_size))
    lh = _line_height(font)

    max_text_w = int(w * 0.58)
    pad_x, pad_y = 22, 16

    lines = _wrap(text.strip(), font, max_text_w - 2 * pad_x)
    text_w = max(_advance(font, ln) for ln in lines)
    text_h = lh * len(lines)

    balloon_w = int(text_w + 2 * pad_x)
    balloon_h = int(text_h + 2 * pad_y)

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
    balloon_rect = (float(bx), float(by), float(balloon_w), float(balloon_h))

    # ── Render: subject underneath, balloon fill + text on top ─────────────
    canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    canvas.alpha_composite(_to_pil(sticker), dest=(sx, sy))

    # Balloon graphics on a supersampled layer for smooth (antialiased) edges.
    layer = Image.new("RGBA", (new_w * _SS, new_h * _SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ss_font = _make_font(_style_size(resolved, base_size) * _SS)

    drawer = {
        STYLE_SPEECH: _draw_speech,
        STYLE_THOUGHT: _draw_thought,
        STYLE_SHOUT: _draw_shout,
        STYLE_WHISPER: _draw_whisper,
    }[resolved]
    drawer(draw, balloon_rect, ss_font, lines, pad_x, pad_y)

    layer = layer.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas.alpha_composite(layer)

    return _from_pil(canvas)


# ── Balloon draw functions (fill + text only; border added later) ───────────────
#
# Every shape is drawn in supersampled coordinates: the geometry computed in the
# new-canvas pixel space is multiplied by ``_SS`` here, then the whole layer is
# downscaled once in ``render_balloon``.

def _scaled_rect(rect: tuple[float, float, float, float]):
    x, y, w, h = rect
    return x * _SS, y * _SS, w * _SS, h * _SS


def _draw_speech(draw, rect, font, lines, pad_x, pad_y) -> None:
    x, y, w, h = _scaled_rect(rect)
    r = min(h * 0.38, 18.0 * _SS)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=_FILL_WHITE)
    _text(draw, font, lines, rect, pad_x, pad_y, _TEXT_COLOR)


def _draw_whisper(draw, rect, font, lines, pad_x, pad_y) -> None:
    # Quiet voice: a small, very-rounded pill.  The smaller font (set in
    # _style_size) carries the "whisper" read, since a dashed outline can't
    # survive being merged into the border.
    x, y, w, h = _scaled_rect(rect)
    r = min(h * 0.5, 26.0 * _SS)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=_FILL_WHITE)
    _text(draw, font, lines, rect, pad_x, pad_y, _WHISPER_COLOR)


def _draw_thought(draw, rect, font, lines, pad_x, pad_y) -> None:
    # Dream/thought: a lumpy cloud body — main ellipse plus a ring of lobes,
    # all the same fill so the overlapping draws union into one silhouette.
    x, y, w, h = _scaled_rect(rect)
    draw.ellipse([x, y, x + w, y + h], fill=_FILL_WHITE)
    cx, cy = x + w / 2, y + h / 2
    rx, ry = w / 2, h / 2
    bump = min(rx, ry) * 0.42
    n = 11
    for i in range(n):
        a = 2 * math.pi * i / n
        px = cx + rx * 0.96 * math.cos(a)
        py = cy + ry * 0.96 * math.sin(a)
        draw.ellipse(
            [px - bump, py - bump, px + bump, py + bump], fill=_FILL_WHITE
        )
    _text(draw, font, lines, rect, pad_x, pad_y, _TEXT_COLOR)


def _draw_shout(draw, rect, font, lines, pad_x, pad_y) -> None:
    x, y, w, h = _scaled_rect(rect)
    cx, cy = x + w / 2, y + h / 2
    rx_out, ry_out = w / 2 * 1.22, h / 2 * 1.22
    rx_in, ry_in = w / 2 * 0.88, h / 2 * 0.88
    pts = _starburst(cx, cy, rx_out, ry_out, rx_in, ry_in, spikes=16)
    draw.polygon(pts, fill=_FILL_YELLOW)
    # Shout reads as loud: fake-bold the glyphs by stroking them in the same
    # ink (Bangers ships Regular only, so there is no real bold face).
    stroke = max(1, getattr(font, "size", _SS) // 18)
    _text(draw, font, lines, rect, pad_x, pad_y, _TEXT_COLOR, stroke=stroke)


# ── Shape helpers ─────────────────────────────────────────────────────────────

def _starburst(
    cx: float, cy: float,
    rx_out: float, ry_out: float,
    rx_in: float, ry_in: float,
    spikes: int = 16,
) -> list[tuple[float, float]]:
    """Elliptical starburst polygon points (for shout bubbles)."""
    pts: list[tuple[float, float]] = []
    total = spikes * 2
    for i in range(total):
        angle = math.pi * 2 * i / total - math.pi / 2
        if i % 2 == 0:
            x = cx + rx_out * math.cos(angle)
            y = cy + ry_out * math.sin(angle)
        else:
            x = cx + rx_in * math.cos(angle)
            y = cy + ry_in * math.sin(angle)
        pts.append((x, y))
    return pts


# ── Text helper ───────────────────────────────────────────────────────────────

def _text(draw, font, lines, rect, pad_x, pad_y, color, stroke: int = 0) -> None:
    """Draw centred, vertically-centred lines inside *rect* (supersampled)."""
    x, y, w, h = _scaled_rect(rect)
    lh = _line_height(font)
    total_h = lh * len(lines)
    ty = y + (h - total_h) / 2
    for line in lines:
        lw = draw.textlength(line, font=font)
        tx = x + (w - lw) / 2
        draw.text(
            (tx, ty), line, font=font, fill=color, anchor="la",
            stroke_width=stroke, stroke_fill=color,
        )
        ty += lh


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap *text* to fit within *max_width* pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if _advance(font, candidate) <= max_width:
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
