"""The sticker pipeline: image + mask → finished sticker.

This is the single place that knows *how a sticker is built* — the order of
the steps and the rules between them:

1. ``build_raw_sticker`` — compose the cut-out from image + mask and upscale a
   tiny subject so the preview is not blocky. This is the border-free silhouette.
2. ``auto_border_width`` — the default border thickness derived from the raw
   sticker's size.
3. ``compose_sticker`` — merge the speech balloon into the silhouette *first*
   (so a single white border later wraps subject and balloon together), then add
   the border. The result is the saved/copied sticker, with no preview background.
4. ``resize_for_whatsapp`` — fit the finished sticker onto a 512×512 canvas for
   export/clipboard.

Preview background flattening stays in :mod:`sticker_creator.utils.sticker_border`
(``apply_background``) because it is a display concern, not part of the saved
sticker. Every step here is pure numpy/PIL — no Qt, no widgets — so the whole
pipeline (balloon included) is exercisable headless, without a running event
loop, and the GUI, the clipboard, and the help-image script all drive the same
code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sticker_creator.utils.balloon_renderer import STYLE_AUTO, render_balloon
from sticker_creator.utils.sticker_border import (
    BACKGROUND_TRANSPARENT,
    add_white_border,
    extract_sticker,
)
from sticker_creator.utils.sticker_resize import ensure_min_size, resize_to_512

# Border thickness as a fraction of the raw sticker's shorter side.
_BORDER_WIDTH_RATIO = 0.014
_MIN_RAW_SIDE = 128


@dataclass
class StickerOptions:
    """Everything the user controls about how the sticker is composed.

    ``background`` is a *preview* concern only — :func:`compose_sticker` never
    flattens onto it; callers apply it for display via
    :func:`sticker_creator.utils.sticker_border.apply_background`.
    """

    border_enabled: bool = True
    border_width: int = 7
    balloon_text: str = ""
    balloon_style: str = STYLE_AUTO
    background: str = BACKGROUND_TRANSPARENT


def build_raw_sticker(
    image: np.ndarray, mask: np.ndarray, min_side: int = _MIN_RAW_SIDE
) -> np.ndarray:
    """Border-free RGBA silhouette: extract from *mask*, upscale if tiny."""
    raw = extract_sticker(image, mask, border_enabled=False)
    return ensure_min_size(raw, min_side=min_side)


def auto_border_width(raw_sticker: np.ndarray) -> int:
    """Default border thickness (px) for a raw sticker of this size."""
    h, w = raw_sticker.shape[:2]
    return max(1, round(min(w, h) * _BORDER_WIDTH_RATIO))


def compose_sticker(raw_sticker: np.ndarray, options: StickerOptions) -> np.ndarray:
    """Balloon-then-border the raw silhouette into the finished sticker.

    The returned array is what gets saved/copied — it carries no preview
    background. Order is fixed: balloon merges into the silhouette before the
    border, so one continuous white border wraps subject and balloon together.
    """
    rgba = raw_sticker
    if options.balloon_text.strip():
        rgba = render_balloon(rgba, options.balloon_text, options.balloon_style)
    else:
        rgba = rgba.copy()

    if options.border_enabled:
        rgba = add_white_border(rgba, border_width=options.border_width)

    return rgba


def resize_for_whatsapp(sticker: np.ndarray) -> np.ndarray:
    """Fit a finished sticker onto a 512×512 transparent canvas for export."""
    return resize_to_512(sticker)
