"""Utility for adding a white border to extracted stickers.

The border is created by dilating the alpha mask using an isotropic
square-kernel maximum filter, then applying a tiny fixed-radius Gaussian
blur for 1-pixel anti-aliasing on **both** the inner edge (sticker →
white border) and the outer edge (border → transparency).

Unlike the previous implementation, the blur radius does **not** scale
with the border width, so thick borders stay crisp instead of turning
muddy.
"""

import numpy as np
from PIL import Image, ImageFilter

_DEFAULT_BORDER_WIDTH = 7
_AA_RADIUS = 0.8  # Fixed anti-aliasing blur — does NOT scale with border width

BACKGROUND_TRANSPARENT = "transparent"
BACKGROUND_WHITE = "white"
BACKGROUND_BLACK = "black"


def apply_background(rgba: np.ndarray, color: str) -> np.ndarray:
    """Flatten *rgba* onto a solid background color.

    Parameters
    ----------
    rgba : np.ndarray
        H×W×4 RGBA uint8 array.
    color : str
        ``"transparent"``, ``"white"``, or ``"black"``.

    Returns
    -------
    np.ndarray
        H×W×4 RGBA array.  For non-transparent colors every pixel has
        alpha=255 and the RGB channels are composited over the background.
    """
    if color == BACKGROUND_TRANSPARENT:
        return rgba.copy()
    bg_rgb = np.array([255, 255, 255], dtype=np.float32) if color == BACKGROUND_WHITE else np.zeros(3, dtype=np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32) * alpha + bg_rgb * (1.0 - alpha)
    out = np.empty_like(rgba)
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[:, :, 3] = 255
    return out


def add_white_border(rgba: np.ndarray, border_width: int = _DEFAULT_BORDER_WIDTH) -> np.ndarray:
    """Add a crisp white border around the opaque region of an RGBA image.

    The input is first padded by *border_width* pixels on all sides so that
    the border can be created even on edges that originally touched the
    crop boundary.

    The border is anti-aliased on **both** edges:

    * **Outer edge** (border → transparency) — smooth alpha falloff from
      a fixed-radius ``GaussianBlur`` applied to the dilated mask.
    * **Inner edge** (sticker → border) — smooth RGB transition achieved
      by blending each pixel's original colour toward white using a
      fixed-radius ``GaussianBlur`` of the *original* (non-dilated) mask
      as the blend weight.

    The key improvements over the previous implementation:

    1. Dilation uses PIL ``MaxFilter`` (isotropic square kernel) instead
       of iterative 4-connected expansion — eliminates staircasing on
       diagonal edges entirely, even for large border widths.
    2. The Gaussian blur radius is **fixed** at ``_AA_RADIUS`` ≈ 0.8 px,
       so increasing the border width does *not* make the edges blurrier.
    3. The inner edge is anti-aliased by blending original colours toward
       white over a 1-pixel transition zone, removing the hard pixel step
       that previously appeared at the sticker/border boundary.

    Parameters
    ----------
    rgba : np.ndarray
        H×W×4 uint8 RGBA image.
    border_width : int
        Visual width of the border in pixels (default 7).

    Returns
    -------
    np.ndarray
        A new array with a smooth white border added (larger than *rgba*
        by ``2 * border_width`` on each spatial axis).
    """
    if border_width <= 0:
        return rgba.copy()

    bw = border_width

    # ── Pad with transparent pixels ──────────────────────────────────
    padded = np.pad(
        rgba,
        ((bw, bw), (bw, bw), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # ── Binary alpha mask ────────────────────────────────────────────
    alpha_bin = padded[:, :, 3] > 0           # bool
    alpha_255 = alpha_bin.astype(np.uint8) * 255  # 0 / 255
    pil_alpha = Image.fromarray(alpha_255, mode="L")

    # ── Dilate using a MAX filter (isotropic square kernel) ─────────
    # Size 2*bw+1 ensures exactly *bw* pixels of dilation in all
    # directions — no staircasing from iterative 4-connected expansion.
    dilated = np.array(
        pil_alpha.filter(ImageFilter.MaxFilter(2 * bw + 1)),
        dtype=np.uint8,
    )

    # ── Anti-alias the OUTER edge (dilated mask → transparency) ─────
    # Fixed small radius — only enough for a 1-pixel smooth fade.
    dilated_aa = np.array(
        Image.fromarray(dilated).filter(
            ImageFilter.GaussianBlur(radius=_AA_RADIUS),
        ),
        dtype=np.uint8,
    )

    # ── Anti-alias the INNER edge (sticker colour → white border) ───
    original_aa = np.array(
        pil_alpha.filter(ImageFilter.GaussianBlur(radius=_AA_RADIUS)),
        dtype=np.uint8,
    )

    # ── Alpha channel ────────────────────────────────────────────────
    # Inside the original mask: fully opaque.
    # Border region: use dilated_aa for a smooth outer fade.
    new_alpha = np.where(alpha_bin, np.uint8(255), dilated_aa)

    # ── RGB channels ─────────────────────────────────────────────────
    # Blend original colour toward white at the inner edge using
    # original_aa as the blend weight.  original_aa ≈ 255 inside the
    # mask and ≈ 0 in the border region, with a smooth transition at
    # the boundary — so the colour ramp spans exactly ~1 pixel.
    blend = original_aa.astype(np.float32) / 255.0   # [0, 1]  smooth
    orig_float = padded[:, :, :3].astype(np.float32)
    white = np.float32(255.0)

    blended_rgb = orig_float * blend[:, :, np.newaxis] \
                  + white * (1.0 - blend[:, :, np.newaxis])

    # ── Compose result ───────────────────────────────────────────────
    result = padded.copy()
    # Round before casting to avoid truncation errors from floating-point
    # imprecision (e.g. 254.99998 → 254 when truncated, but should be 255).
    result[:, :, :3] = np.clip(np.round(blended_rgb), 0, 255).astype(np.uint8)
    result[:, :, 3] = new_alpha

    return result


def extract_sticker(
    image: np.ndarray,
    mask: np.ndarray,
    border_enabled: bool = True,
    border_width: int = _DEFAULT_BORDER_WIDTH,
) -> np.ndarray:
    """Extract a sticker from an image and mask, optionally adding a white border.

    Compose RGBA from image + mask, crop to the mask bounding box, then
    optionally add a white border.

    Parameters
    ----------
    image : np.ndarray
        H×W×3 (RGB) or H×W×4 (RGBA) uint8 image.
    mask : np.ndarray
        H×W uint8 or bool binary mask.
    border_enabled : bool
        Whether to add a white border (default True).
    border_width : int
        Width of the border in pixels (default 7).

    Returns
    -------
    np.ndarray
        H×W×4 RGBA sticker array.
    """
    h, w = image.shape[:2]

    # Compose RGBA
    if image.shape[2] == 3:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = (mask > 0).astype(np.uint8) * 255
    else:
        rgba = image.copy()
        rgba[:, :, 3] = (mask > 0).astype(np.uint8) * 255

    # Crop to mask bounding box
    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        # When border is enabled, expand the crop so the border has room
        # to appear even on edges that touch the original crop boundary.
        if border_enabled and border_width > 0:
            x1 = max(0, x1 - border_width)
            x2 = min(w, x2 + border_width)
            y1 = max(0, y1 - border_width)
            y2 = min(h, y2 + border_width)

        rgba = rgba[y1:y2, x1:x2].copy()

    # Add border if enabled
    if border_enabled:
        rgba = add_white_border(rgba, border_width=border_width)

    return rgba
