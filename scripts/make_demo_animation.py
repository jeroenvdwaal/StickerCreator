#!/usr/bin/env python3
"""Render an animated demo of the Sticker Creator workflow.

Drives the real MainWindow headless: load image → click face → click
hair → type balloon text → final preview. Saves each step as a PNG and
assembles them into an animated GIF, overlaying captions, an exaggerated
cursor, and a click ripple where the user "clicks".
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEventLoop, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QImage

from sticker_creator.app import StickerCreatorApp  # noqa: E402

SRC = ROOT / "docs" / "images" / "thispersondoesnotexist.jpg"
OUT_DIR = ROOT / "docs" / "demo_frames"
META_FILE = OUT_DIR / "cursors.json"
GIF_OUT = ROOT / "docs" / "images" / "demo.gif"
SLIDES_DIR = ROOT / "sticker_creator" / "resources" / "welcome"

BALLOON = "I don't even exist! Can you believe it?"

FRAME_W = 900       # downscale window grab for GIF
GIF_DURATIONS = [3500, 3500, 3500, 4000, 4500]   # ms per frame (caption read time)

CAPTIONS = {
    1: "1. Open an image",
    2: "2. Click the face — SAM 2 cuts it out",
    3: "3. Click the hair to include the whole head",
    4: "4. Type balloon text — preview updates live",
    5: "5. Save as PNG, WebP, or WhatsApp sticker",
}


def wait_for(signal, timeout_ms: int = 60_000) -> None:
    loop = QEventLoop()
    conn = signal.connect(loop.quit)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()
    try:
        signal.disconnect(conn)
    except (RuntimeError, TypeError):
        pass


def pump(ms: int = 200) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


# ── PIL overlays ──────────────────────────────────────────────────────────────

def _add_caption(img, text: str, font):
    from PIL import Image, ImageDraw
    pad_x, pad_y = 20, 12
    draw_tmp = ImageDraw.Draw(img)
    bbox = draw_tmp.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    bar_h = text_h + 2 * pad_y

    new = Image.new("RGB", (img.width, img.height + bar_h), (24, 24, 26))
    new.paste(img, (0, 0))
    d = ImageDraw.Draw(new)
    tx = (img.width - text_w) // 2 - bbox[0]
    ty = img.height + pad_y - bbox[1]
    d.text((tx, ty), text, fill=(255, 255, 255), font=font)
    return new


def _draw_click_ripple(img, x: int, y: int) -> None:
    """Concentric red rings + glow indicating a mouse click."""
    from PIL import Image, ImageDraw
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    # Outer glow
    d.ellipse((x - 38, y - 38, x + 38, y + 38), outline=(255, 60, 50, 130), width=4)
    d.ellipse((x - 26, y - 26, x + 26, y + 26), outline=(255, 90, 70, 200), width=3)
    # Solid dot
    d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(255, 60, 50, 230))
    img.alpha_composite(overlay)


def _draw_highlight(img, x: int, y: int, w: int, h: int) -> None:
    """Rounded yellow rectangle outline calling out a widget."""
    from PIL import Image, ImageDraw
    pad = 6
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    rect = (x - pad, y - pad, x + w + pad, y + h + pad)
    # Outer glow + crisp inner stroke
    d.rounded_rectangle(rect, radius=10, outline=(255, 215, 0, 130), width=8)
    d.rounded_rectangle(rect, radius=10, outline=(255, 180, 0, 255), width=3)
    img.alpha_composite(overlay)


def _draw_cursor(img, x: int, y: int) -> None:
    """Big white arrow with black outline at (x, y) — the hot-spot is the tip."""
    from PIL import ImageDraw
    # Arrow polygon, hot-spot at first vertex.
    poly = [
        (x,         y),
        (x,         y + 28),
        (x + 7,     y + 21),
        (x + 12,    y + 32),
        (x + 16,    y + 30),
        (x + 11,    y + 19),
        (x + 20,    y + 19),
    ]
    d = ImageDraw.Draw(img)
    # Black outline (drawn slightly fatter)
    d.polygon(poly, fill=(0, 0, 0), outline=(0, 0, 0))
    # White fill on top
    shrunk = [(px + 1, py + 1) for (px, py) in poly]
    d.polygon(shrunk, fill=(255, 255, 255), outline=(0, 0, 0))


# ── Coordinate helpers ───────────────────────────────────────────────────────

def _viewer_image_to_window(viewer, mw, image_x: float, image_y: float) -> QPoint:
    """Map an image-pixel coord through the QGraphicsView into window space."""
    vp_pt = viewer.view.mapFromScene(QPointF(image_x, image_y))
    return viewer.view.viewport().mapTo(mw, vp_pt)


def _widget_center_to_window(widget, mw) -> QPoint:
    return widget.mapTo(mw, widget.rect().center())


# ── Snap ─────────────────────────────────────────────────────────────────────

def snap(window, idx: int, label: str) -> Path:
    pump(100)
    pix = window.grab()
    img: QImage = pix.toImage()
    if img.width() > FRAME_W:
        img = img.scaledToWidth(FRAME_W, Qt.TransformationMode.SmoothTransformation)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{idx:02d}_{label}.png"
    img.save(str(out))
    print(f"  frame {idx}: {out.name}  ({img.width()}x{img.height()})")
    return out


def main() -> int:
    if not SRC.exists():
        print(f"missing source image: {SRC}")
        return 1

    if "--reuse-frames" in sys.argv:
        cached = sorted(OUT_DIR.glob("[0-9][0-9]_*.png"))
        if not cached:
            print(f"no cached frames in {OUT_DIR}")
            return 1
        meta = json.loads(META_FILE.read_text()) if META_FILE.exists() else {}
        _rebuild_gif(cached, meta)
        return 0

    app = StickerCreatorApp([])
    app.window = None
    from sticker_creator.main_window import MainWindow
    mw = MainWindow()
    mw._welcome_shown = True
    mw.resize(1200, 760)
    mw.show()
    pump(300)

    print("waiting for model to load…")
    if mw.segmenter.active_model_name is None:
        wait_for(mw.segmenter.model_loaded, timeout_ms=120_000)
    pump(300)

    frames: list[Path] = []
    # Each entry: {"cursor": [x, y] | None, "click": true | false} in WINDOW coords.
    meta: dict[str, dict] = {}

    win_w = mw.width()
    scale = FRAME_W / win_w

    def record(idx: int, cursor: QPoint | None, click: bool,
               highlight_widget=None) -> None:
        entry: dict = {"cursor": None, "click": False, "highlight": None}
        if cursor is not None:
            entry["cursor"] = [int(cursor.x() * scale), int(cursor.y() * scale)]
            entry["click"] = click
        if highlight_widget is not None:
            tl = highlight_widget.mapTo(mw, QPoint(0, 0))
            entry["highlight"] = [
                int(tl.x() * scale),
                int(tl.y() * scale),
                int(highlight_widget.width() * scale),
                int(highlight_widget.height() * scale),
            ]
        meta[str(idx)] = entry

    # 1. Open image
    print("loading image…")
    mw._load_image(str(SRC))
    pump(400)
    frames.append(snap(mw, 1, "opened"))
    record(1, None, False)  # no cursor on the intro frame

    h, w = mw._current_image.shape[:2]
    face_pt = (w // 2, int(h * 0.55))
    hair_pt = (w // 2, int(h * 0.08))

    # 2. Face click
    print("click face…")
    mw.image_viewer.add_point(face_pt[0], face_pt[1], 1)
    wait_for(mw.segmenter.processing_finished, timeout_ms=120_000)
    pump(300)
    frames.append(snap(mw, 2, "face_click"))
    record(2, _viewer_image_to_window(mw.image_viewer, mw, *face_pt), True)

    # 3. Hair click
    print("click hair…")
    mw.image_viewer.add_point(hair_pt[0], hair_pt[1], 1)
    wait_for(mw.segmenter.processing_finished, timeout_ms=120_000)
    pump(300)
    frames.append(snap(mw, 3, "hair_click"))
    record(3, _viewer_image_to_window(mw.image_viewer, mw, *hair_pt), True)

    # 4. Type balloon text
    print("type balloon text…")
    mw.inspector._balloon_input.setText(BALLOON)
    mw.inspector._balloon_input.setFocus()
    mw.inspector._commit_balloon_text()
    pump(500)
    frames.append(snap(mw, 4, "balloon"))
    record(
        4,
        _widget_center_to_window(mw.inspector._balloon_input, mw),
        False,
        highlight_widget=mw.inspector._balloon_input,
    )

    # 5. Final / save
    pump(600)
    frames.append(snap(mw, 5, "final"))
    save_btn = getattr(mw.inspector, "_save_btn", None)
    if save_btn is not None:
        record(5, _widget_center_to_window(save_btn, mw), True)
    else:
        record(5, None, False)

    META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"saved {META_FILE.name}")

    _rebuild_gif(frames, meta)
    mw.close()
    return 0


def _composite_frame(path: Path, info: dict):
    """Apply cursor / click / highlight overlays. Returns RGBA PIL Image."""
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    hl = info.get("highlight")
    if hl is not None:
        _draw_highlight(img, *hl)
    cur = info.get("cursor")
    if cur is not None:
        if info.get("click"):
            _draw_click_ripple(img, cur[0], cur[1])
        _draw_cursor(img, cur[0], cur[1])
    return img


def _rebuild_gif(frame_paths: list[Path], meta: dict) -> None:
    print("assembling GIF…")
    from PIL import ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/google-noto/NotoSans-Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default()

    raw_slides = []   # RGBA frames with overlays but no caption (for slideshow)
    gif_frames = []   # RGB frames with caption bar baked in (for the GIF)
    for i, p in enumerate(frame_paths, start=1):
        info = meta.get(str(i), {})
        composite = _composite_frame(p, info)
        raw_slides.append(composite)
        rgb = composite.convert("RGB")
        caption = CAPTIONS.get(i, "")
        if caption:
            rgb = _add_caption(rgb, caption, font)
        gif_frames.append(rgb)

    durations = (GIF_DURATIONS + [GIF_DURATIONS[-1]] * len(gif_frames))[:len(gif_frames)]
    GIF_OUT.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        GIF_OUT,
        save_all=True,
        append_images=gif_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"saved {GIF_OUT}")

    # Slide PNGs for the welcome dialog: no caption bar — the dialog renders the
    # step title separately above the image.
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(raw_slides, start=1):
        out = SLIDES_DIR / f"slide_{i:02d}.png"
        frame.convert("RGB").save(out, optimize=True)
        print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
