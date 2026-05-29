"""Clipboard operations for image paste and copy using KDE/system clipboard."""

import numpy as np
from PySide6.QtCore import QByteArray, QCoreApplication, QMimeData
from PySide6.QtGui import QClipboard

from sticker_creator.utils import imagecodec


class ClipboardManager:
    """Handles clipboard operations with KDE integration via PySide6."""

    def __init__(self):
        self._clipboard = QCoreApplication.instance().clipboard()

    def paste_image(self) -> np.ndarray | None:
        """Paste an image from the system clipboard.

        First attempts to read via ``pixmap()`` (works with data set via
        :meth:`copy_image` / :meth:`copy_as_whatsapp_sticker`), then falls
        back to reading ``image/png`` MIME data directly.

        Returns:
            RGB numpy array (H, W, 3) or None if clipboard has no image.
        """
        # ── Attempt 1: pixmap() path ──────────────────────────────────
        pixmap = self._clipboard.pixmap(QClipboard.Mode.Clipboard)
        if pixmap is None or pixmap.isNull():
            pixmap = self._clipboard.pixmap(QClipboard.Mode.Selection)

        if pixmap is not None and not pixmap.isNull():
            qimage = pixmap.toImage()
            if not qimage.isNull():
                return imagecodec.from_qimage(qimage)[:, :, :3].copy()

        # ── Attempt 2: MIME data path (used by our own copy_image) ────
        mime = self._clipboard.mimeData(QClipboard.Mode.Clipboard)
        if mime is not None and mime.hasFormat("image/png"):
            png_bytes = mime.data("image/png")
            if png_bytes and len(png_bytes) > 0:
                try:
                    return imagecodec.decode_png(png_bytes.data())[:, :, :3].copy()
                except Exception:
                    pass

        return None

    def copy_image(self, image_rgba: np.ndarray) -> bool:
        """Copy an RGBA numpy array to the system clipboard.

        Uses ``setImageData()`` — the standard Qt mechanism — so the image
        is available to **every** application (browsers, image editors, ZapZap,
        etc.) via the system clipboard's native image formats.  Alpha
        transparency is preserved because the underlying ``QImage`` uses
        ``Format_RGBA8888``, and Qt converts to ``image/png`` (with alpha)
        for apps that request it.

        Args:
            image_rgba: (H, W, 4) numpy array in RGBA format.

        Returns:
            True on success.
        """
        qimage = imagecodec.to_qimage(image_rgba)  # owns its pixels

        mime = QMimeData()
        mime.setImageData(qimage)
        self._clipboard.setMimeData(mime, QClipboard.Mode.Clipboard)
        return True

    def copy_as_whatsapp_sticker(self, image_rgba: np.ndarray) -> bool:
        """Copy an RGBA array as a WhatsApp-compatible WebP sticker.

        The sticker is resized to 512×512 (centred with transparent padding),
        then placed on the clipboard via ``setImageData()`` so it is readable
        by **any** application — including ZapZap / WhatsApp Web / Electron.
        As a bonus the ``image/webp`` MIME type is also advertised for apps
        that natively support WebP.

        Args:
            image_rgba: (H, W, 4) numpy array in RGBA format.

        Returns:
            True on success.
        """
        from sticker_creator.utils.sticker_pipeline import resize_for_whatsapp
        from sticker_creator.utils.file_io import ImageSaver

        # 1. Resize to 512×512 with transparent padding
        resized = resize_for_whatsapp(image_rgba)

        # 2. Build MIME data with native image (for all apps) + WebP (bonus)
        qimage = imagecodec.to_qimage(resized)  # owns its pixels

        mime = QMimeData()
        mime.setImageData(qimage)  # ← standard path — works everywhere

        # Also advertise image/webp for apps that support it natively
        webp_bytes = ImageSaver.save_webp_bytes(resized, quality=80)
        if webp_bytes is not None:
            mime.setData("image/webp", QByteArray(webp_bytes))

        self._clipboard.setMimeData(mime, QClipboard.Mode.Clipboard)
        return True
