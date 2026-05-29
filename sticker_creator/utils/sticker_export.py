"""Sticker export — the one place that knows how each target is written.

A finished sticker can be exported to several targets, and each target carries
its own *policy*: which transform runs first, which encoder, at what quality.
That policy used to be a chain of ``if`` branches inside ``MainWindow``'s save
slot, so "what does a WhatsApp export actually do" was only answerable by
reading a UI handler. Here it is one small table:

* :data:`ExportFormat.PNG` — lossless, no resize.
* :data:`ExportFormat.WEBP` — WebP at near-lossless quality, no resize.
* :data:`ExportFormat.WHATSAPP_WEBP` — fit onto a 512×512 canvas first, then
  WebP at the smaller quality WhatsApp/Telegram expect.

The window's job shrinks to mapping its file-dialog choice to an
:class:`ExportFormat`; :func:`export_sticker` owns everything after that. The
function is pure I/O over numpy + :class:`ImageSaver`, so the policy is
testable without a dialog: assert that each format lands the right bytes.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np

from sticker_creator.utils.file_io import ImageSaver
from sticker_creator.utils.sticker_pipeline import resize_for_whatsapp

# WhatsApp/Telegram stickers are small WebP; the desktop formats stay crisp.
_WHATSAPP_QUALITY = 80
_WEBP_QUALITY = 100


class ExportFormat(Enum):
    """A sticker export target. The value is its conventional file suffix."""

    PNG = "png"
    WEBP = "webp"
    WHATSAPP_WEBP = "whatsapp_webp"


def export_sticker(
    sticker: np.ndarray, file_path: str | Path, fmt: ExportFormat
) -> bool:
    """Write *sticker* to *file_path* under the policy for *fmt*.

    Returns ``True`` on success (mirrors :class:`ImageSaver`'s contract).
    """
    if fmt is ExportFormat.WHATSAPP_WEBP:
        return ImageSaver.save_webp(
            resize_for_whatsapp(sticker), file_path, quality=_WHATSAPP_QUALITY
        )
    if fmt is ExportFormat.WEBP:
        return ImageSaver.save_webp(sticker, file_path, quality=_WEBP_QUALITY)
    return ImageSaver.save_png(sticker, file_path)
