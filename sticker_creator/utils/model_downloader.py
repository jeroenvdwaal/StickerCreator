"""SAM 2 model checkpoint downloader with background-thread progress signals.

Provides a QObject-based worker that downloads model checkpoints from Meta's
official SAM 2 release server and emits progress/finished/error signals so the
UI can stay responsive during the download.
"""

from pathlib import Path

import requests
from PySide6.QtCore import QObject, Signal, Slot

from sticker_creator.utils.paths import user_model_dir

# Directory where checkpoints are stored (user-writable; XDG_DATA_HOME)
MODEL_DIR = user_model_dir()

# Default headers to avoid 403 responses from Meta's CDN
_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# SAM 2 model checkpoint URLs (from Meta's official releases)
_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"

MODEL_URLS: dict[str, str] = {
    "sam2.1_hiera_tiny":      _BASE_URL + "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small":     _BASE_URL + "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus": _BASE_URL + "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large":     _BASE_URL + "sam2.1_hiera_large.pt",
}

MODEL_SIZES: dict[str, str] = {
    "sam2.1_hiera_tiny":      "~148 MB",
    "sam2.1_hiera_small":     "~175 MB",
    "sam2.1_hiera_base_plus": "~308 MB",
    "sam2.1_hiera_large":     "~856 MB",
}


class ModelDownloader(QObject):
    """Downloads a SAM 2 checkpoint in a background thread.

    Signals:
        progress: Emitted with an integer percentage (0–100).
        finished: Emitted with the Path to the downloaded checkpoint file.
        error:    Emitted with an error message string.
    """

    progress = Signal(int)
    finished = Signal(object)  # Path
    error = Signal(str)

    def __init__(self, model_name: str = "sam2.1_hiera_tiny", parent: QObject | None = None):
        super().__init__(parent)
        self.model_name = model_name
        self._abort = False

    def abort(self) -> None:
        """Signal the download to stop at the next chunk."""
        self._abort = True

    @Slot()
    def run(self) -> None:
        """Execute the download. Call this from a QThread."""
        self._abort = False

        if self.model_name not in MODEL_URLS:
            self.error.emit(
                f"Unknown model '{self.model_name}'. "
                f"Available: {', '.join(MODEL_URLS.keys())}"
            )
            return

        url = MODEL_URLS[self.model_name]
        output_path = MODEL_DIR / f"{self.model_name}.pt"

        # Already exists with a plausible size — skip download
        _MIN_VALID_BYTES = 10 * 1024 * 1024  # 10 MB minimum for any real checkpoint
        if output_path.exists() and output_path.stat().st_size >= _MIN_VALID_BYTES:
            self.progress.emit(100)
            self.finished.emit(output_path)
            return

        # Remove a corrupted/incomplete file if present
        if output_path.exists():
            output_path.unlink()

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(
                url, stream=True, timeout=30, headers=_DEFAULT_HEADERS
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if self._abort:
                        f.close()
                        output_path.unlink(missing_ok=True)
                        return

                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = int(downloaded / total_size * 100)
                            self.progress.emit(pct)

            self.progress.emit(100)
            self.finished.emit(output_path)

        except requests.RequestException as e:
            output_path.unlink(missing_ok=True)
            self.error.emit(f"Download failed: {e}")
        except OSError as e:
            output_path.unlink(missing_ok=True)
            self.error.emit(f"File error: {e}")
