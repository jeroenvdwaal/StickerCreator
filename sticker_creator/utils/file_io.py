"""Image file I/O utilities for opening and saving images."""

from pathlib import Path

import numpy as np
import cv2
from PIL import Image


class ImageLoader:
    """Loads images from file paths or clipboard data."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}

    @staticmethod
    def load(file_path: str | Path) -> np.ndarray | None:
        """Load an image from file, returning an RGB numpy array.

        Returns None if the file cannot be loaded.
        """
        path = Path(file_path)
        if not path.exists():
            return None

        try:
            # Use OpenCV for fast loading
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                # Fall back to PIL
                pil_img = Image.open(path).convert("RGB")
                img = np.array(pil_img)
                return img

            # OpenCV loads as BGR, convert to RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except Exception:
            return None

    @staticmethod
    def load_from_bytes(data: bytes) -> np.ndarray | None:
        """Load an image from raw bytes (e.g., clipboard)."""
        try:
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None
        except Exception:
            return None


def _webp_supported() -> bool:
    """Check whether Pillow was built with WebP support."""
    return "WEBP" in Image.SAVE


class ImageSaver:
    """Saves sticker images to file."""

    @staticmethod
    def save_png(image_rgba: np.ndarray, file_path: str | Path) -> bool:
        """Save an RGBA numpy array as a PNG file with transparency.

        Args:
            image_rgba: (H, W, 4) numpy array in RGBA format.
            file_path: Destination path.

        Returns:
            True on success, False on failure.
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure RGBA
            if image_rgba.shape[2] != 4:
                raise ValueError("Image must have 4 channels (RGBA)")

            # Convert to PIL and save
            pil_img = Image.fromarray(image_rgba, "RGBA")
            pil_img.save(str(path), format="PNG")
            return True

        except Exception:
            return False

    @staticmethod
    def save_png_bytes(image_rgba: np.ndarray) -> bytes | None:
        """Encode an RGBA array as PNG bytes (for clipboard)."""
        try:
            pil_img = Image.fromarray(image_rgba, "RGBA")
            import io
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    @staticmethod
    def save_webp(image_rgba: np.ndarray, file_path: str | Path, quality: int = 90) -> bool:
        """Save an RGBA numpy array as a WebP file with transparency.

        Args:
            image_rgba: (H, W, 4) numpy array in RGBA format.
            file_path: Destination path.
            quality: WebP quality 0-100 (default 90). Lower = smaller file.

        Returns:
            True on success, False on failure.
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if image_rgba.shape[2] != 4:
                raise ValueError("Image must have 4 channels (RGBA)")

            pil_img = Image.fromarray(image_rgba, "RGBA")
            pil_img.save(str(path), format="WEBP", quality=quality)
            return True

        except Exception:
            return False

    @staticmethod
    def save_webp_bytes(image_rgba: np.ndarray, quality: int = 90) -> bytes | None:
        """Encode an RGBA array as WebP bytes (for clipboard).

        Args:
            image_rgba: (H, W, 4) numpy array in RGBA format.
            quality: WebP quality 0-100 (default 90).

        Returns:
            WebP bytes or None on failure.
        """
        try:
            pil_img = Image.fromarray(image_rgba, "RGBA")
            import io
            buf = io.BytesIO()
            pil_img.save(buf, format="WEBP", quality=quality)
            return buf.getvalue()
        except Exception:
            return None
