#!/usr/bin/env python3
"""Generate a sample sticker for the help docs using the in-tree pipeline.

Drives the same pipeline the GUI uses (Segmenter → render_balloon →
add_white_border → resize_to_512), bypassing only the Qt UI shell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtGui import QGuiApplication  # noqa: E402
_app = QGuiApplication.instance() or QGuiApplication(sys.argv)

from sticker_creator.segmentation.segmenter import SegmenterWorker  # noqa: E402
from sticker_creator.utils.balloon_renderer import render_balloon, STYLE_AUTO  # noqa: E402
from sticker_creator.utils.file_io import ImageLoader, ImageSaver  # noqa: E402
from sticker_creator.utils.sticker_border import add_white_border, extract_sticker  # noqa: E402
from sticker_creator.utils.sticker_resize import ensure_min_size, resize_to_512  # noqa: E402


SRC = Path("/tmp/random_face.jpg")
OUT = ROOT / "docs" / "images" / "help_sticker_example.png"
MODEL = "sam2.1_hiera_tiny"
BALLOON_TEXT = "I don't even exist!"


def main() -> int:
    image = ImageLoader().load(SRC)
    if image is None:
        print(f"failed to load {SRC}")
        return 1
    h, w = image.shape[:2]
    print(f"loaded image {w}x{h}")

    worker = SegmenterWorker(model_dir=ROOT / "models")
    state: dict = {}
    worker.model_loaded.connect(lambda name: state.update(loaded=name))
    worker.model_error.connect(lambda err: state.update(error=err))
    worker.mask_ready.connect(lambda m: state.update(mask=m))

    print(f"loading model {MODEL}…")
    worker.load_model(MODEL)
    if "error" in state:
        print("model error:", state["error"])
        return 2
    print("model loaded")

    # Single positive point at face center; thispersondoesnotexist faces are centered.
    points = [{"x": w // 2, "y": int(h * 0.45), "label": 1}]
    print("segmenting…")
    worker.segment(image, points)
    if "mask" not in state:
        print("segmentation error:", state.get("error"))
        return 3
    mask = state["mask"]
    print(f"mask coverage: {(mask > 0).mean() * 100:.1f}%")

    print("extract…", flush=True)
    raw = extract_sticker(image, mask, border_enabled=False)
    print(f"  raw {raw.shape}", flush=True)
    print("render_balloon…", flush=True)
    with_balloon = render_balloon(raw, BALLOON_TEXT, STYLE_AUTO)
    print(f"  with_balloon {with_balloon.shape}", flush=True)
    print("add_white_border…", flush=True)
    bordered = add_white_border(with_balloon, border_width=7)
    print(f"  bordered {bordered.shape}", flush=True)
    print("resize…", flush=True)
    final = resize_to_512(ensure_min_size(bordered, 128))
    print(f"  final {final.shape}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not ImageSaver.save_png(final, OUT):
        print("save failed")
        return 4
    print(f"saved {OUT}  ({final.shape[1]}x{final.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
