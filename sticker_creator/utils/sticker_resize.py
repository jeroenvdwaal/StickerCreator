"""Utility to resize stickers to 512×512 for WhatsApp compatibility.

WhatsApp requires stickers to be WebP images. The standard size is 512×512
pixels. This module provides a high-quality resize that preserves the aspect
ratio and centres the sticker on a transparent 512×512 canvas.
"""

import numpy as np
from PIL import Image


def ensure_min_size(rgba: np.ndarray, min_side: int = 128) -> np.ndarray:
    """Upscale a sticker so its longer side is at least ``min_side`` pixels.

    Aspect ratio is preserved and high-quality Lanczos resampling is used.
    If the longer side already meets ``min_side`` the image is returned
    unchanged (a small sticker scaled into the preview otherwise looks blocky).

    Parameters
    ----------
    rgba : np.ndarray
        H×W×4 uint8 RGBA image.
    min_side : int
        Minimum length of the longer side, in pixels.

    Returns
    -------
    np.ndarray
        The original array if already large enough, else an upscaled copy.
    """
    h, w = rgba.shape[:2]
    longest = max(h, w)
    if longest == 0 or longest >= min_side:
        return rgba

    scale = min_side / longest
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    pil_img = Image.fromarray(rgba, "RGBA")
    pil_resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(pil_resized, dtype=np.uint8)


def resize_to_512(rgba: np.ndarray) -> np.ndarray:
    """Resize an RGBA sticker to fit within a 512×512 canvas.

    The sticker is resized (preserving aspect ratio) so that the longer
    dimension fits within 512 pixels, then centred on a 512×512 transparent
    canvas.  High-quality Lanczos resampling is used.

    Parameters
    ----------
    rgba : np.ndarray
        H×W×4 uint8 RGBA image.

    Returns
    -------
    np.ndarray
        512×512×4 uint8 RGBA image.
    """
    h, w = rgba.shape[:2]

    # If already exactly 512×512, return a copy
    if h == 512 and w == 512:
        return rgba.copy()

    # Determine scale factor so the longer side fits in 512
    scale = 512.0 / max(h, w)
    new_w = round(w * scale)
    new_h = round(h * scale)

    # Resize using Pillow with Lanczos (high quality)
    pil_img = Image.fromarray(rgba, "RGBA")
    pil_resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Centre on 512×512 canvas
    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    x_offset = (512 - new_w) // 2
    y_offset = (512 - new_h) // 2
    canvas.paste(pil_resized, (x_offset, y_offset), pil_resized)

    return np.asarray(canvas, dtype=np.uint8)
