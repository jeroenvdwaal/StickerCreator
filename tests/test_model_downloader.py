"""Tests for the SAM 2 model downloader.

Covers the ``ModelDownloader`` QObject worker with mocked HTTP responses,
focusing on signal emission, progress reporting, abort, and error handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from PySide6.QtCore import QObject, QThread, Signal

from sticker_creator.utils.model_downloader import (
    MODEL_DIR,
    MODEL_URLS,
    ModelDownloader,
    _DEFAULT_HEADERS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def downloader() -> ModelDownloader:
    """Create a ModelDownloader instance."""
    return ModelDownloader("sam2.1_hiera_tiny")


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify module-level constants are well-defined."""

    def test_model_urls_contains_all_variants(self):
        """MODEL_URLS has all 4 SAM 2 model variants."""
        assert "sam2.1_hiera_tiny" in MODEL_URLS
        assert "sam2.1_hiera_small" in MODEL_URLS
        assert "sam2.1_hiera_base_plus" in MODEL_URLS
        assert "sam2.1_hiera_large" in MODEL_URLS

    def test_model_urls_are_https(self):
        """All model URLs use HTTPS."""
        for name, url in MODEL_URLS.items():
            assert url.startswith("https://"), f"{name} URL is not HTTPS"

    def test_model_dir_is_absolute(self):
        """MODEL_DIR is an absolute path."""
        assert isinstance(MODEL_DIR, Path)
        assert MODEL_DIR.is_absolute()
        assert MODEL_DIR.name == "models"

    def test_default_headers_contains_user_agent(self):
        """_DEFAULT_HEADERS includes a browser-like User-Agent."""
        assert "User-Agent" in _DEFAULT_HEADERS
        ua = _DEFAULT_HEADERS["User-Agent"]
        assert "Mozilla" in ua
        assert "Chrome" in ua


# ---------------------------------------------------------------------------
# ModelDownloader — construction
# ---------------------------------------------------------------------------


class TestModelDownloaderConstruction:
    """Verify the downloader initialises correctly."""

    def test_default_model_name(self):
        """Default model name is sam2.1_hiera_tiny."""
        dl = ModelDownloader()
        assert dl.model_name == "sam2.1_hiera_tiny"

    def test_custom_model_name(self):
        """Custom model name is stored."""
        dl = ModelDownloader("sam2.1_hiera_large")
        assert dl.model_name == "sam2.1_hiera_large"

    def test_abort_flag_initially_false(self, downloader: ModelDownloader):
        """Abort flag starts as False."""
        assert not downloader._abort

    def test_abort_sets_flag(self, downloader: ModelDownloader):
        """Calling abort() sets _abort to True."""
        downloader.abort()
        assert downloader._abort

    def test_signal_existence(self, downloader: ModelDownloader):
        """Signals are proper Signal instances."""
        assert isinstance(ModelDownloader.progress, Signal)
        assert isinstance(ModelDownloader.finished, Signal)
        assert isinstance(ModelDownloader.error, Signal)


# ---------------------------------------------------------------------------
# ModelDownloader — run() with mocked HTTP
# ---------------------------------------------------------------------------


class TestModelDownloaderRun:
    """Test the run() method with mocked ``requests.get``."""

    @patch("sticker_creator.utils.model_downloader.requests.get")
    def test_sends_user_agent_header(self, mock_get, qtbot, tmp_path):
        """The User-Agent header is sent with the request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"x" * 1000]
        mock_get.return_value = mock_response

        with patch("sticker_creator.utils.model_downloader.MODEL_DIR", tmp_path):
            dl = ModelDownloader("sam2.1_hiera_tiny")
            with qtbot.waitSignal(dl.finished, timeout=2000):
                dl.run()

        mock_get.assert_called_once()
        _call_kwargs = mock_get.call_args.kwargs
        assert "headers" in _call_kwargs
        assert _call_kwargs["headers"]["User-Agent"] == _DEFAULT_HEADERS["User-Agent"]

    @patch("sticker_creator.utils.model_downloader.requests.get")
    def test_emits_progress_and_finished(self, mock_get, qtbot, tmp_path):
        """Progress (50, 100) then finished are emitted for a successful download."""
        chunk = b"x" * 500
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [chunk, chunk]
        mock_get.return_value = mock_response

        with patch("sticker_creator.utils.model_downloader.MODEL_DIR", tmp_path):
            dl = ModelDownloader("sam2.1_hiera_tiny")
            progress_values: list[int] = []
            dl.progress.connect(lambda v: progress_values.append(v))

            with qtbot.waitSignal(dl.finished, timeout=2000):
                dl.run()

        assert 50 in progress_values
        assert 100 in progress_values

    @patch("sticker_creator.utils.model_downloader.requests.get")
    def test_emits_error_on_http_403(self, mock_get, qtbot, tmp_path):
        """HTTP 403 response emits the error signal."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = requests.RequestException(
            "403 Client Error"
        )
        mock_get.return_value = mock_response

        with patch("sticker_creator.utils.model_downloader.MODEL_DIR", tmp_path):
            dl = ModelDownloader("sam2.1_hiera_tiny")
            with qtbot.waitSignal(dl.error, timeout=2000) as blocker:
                dl.run()

        assert "403" in str(blocker.args[0])

    @patch("sticker_creator.utils.model_downloader.requests.get")
    def test_emits_error_on_network_failure(self, mock_get, qtbot, tmp_path):
        """Connection error emits the error signal."""
        mock_get.side_effect = requests.RequestException("Connection refused")

        with patch("sticker_creator.utils.model_downloader.MODEL_DIR", tmp_path):
            dl = ModelDownloader("sam2.1_hiera_tiny")
            with qtbot.waitSignal(dl.error, timeout=2000) as blocker:
                dl.run()

        assert "Connection refused" in str(blocker.args[0])

    @patch("sticker_creator.utils.model_downloader.requests.get")
    def test_abort_during_download(self, mock_get, qtbot, tmp_path):
        """Aborting mid-download stops without emitting finished or error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "10000000"}  # 10 MB

        # Generator that yields one chunk, sets abort, then yields another
        # so the loop sees _abort=True on the second iteration.
        def make_chunks(chunk_size: int = 1024 * 1024):
            yield b"x" * 1024 * 1024
            dl._abort = True
            yield b"y" * 1024 * 1024

        mock_response.iter_content.side_effect = make_chunks
        mock_get.return_value = mock_response

        with patch("sticker_creator.utils.model_downloader.MODEL_DIR", tmp_path):
            dl = ModelDownloader("sam2.1_hiera_tiny")
            finished_called: list[bool] = []
            error_called: list[bool] = []
            dl.finished.connect(lambda p: finished_called.append(True))
            dl.error.connect(lambda m: error_called.append(True))

            dl.run()

        assert not finished_called
        assert not error_called

    @patch("sticker_creator.utils.model_downloader.requests.get")
    def test_unknown_model_emits_error(self, mock_get, qtbot):
        """Unknown model name emits error without making HTTP request."""
        dl = ModelDownloader("nonexistent_model")

        with qtbot.waitSignal(dl.error, timeout=2000) as blocker:
            dl.run()

        assert "Unknown model" in str(blocker.args[0])
        mock_get.assert_not_called()

    @patch("sticker_creator.utils.model_downloader.requests.get")
    def test_already_exists_skips_download(self, mock_get, qtbot, tmp_path):
        """If the checkpoint already exists, emit finished without downloading."""
        with patch("sticker_creator.utils.model_downloader.MODEL_DIR", tmp_path):
            existing = tmp_path / "sam2.1_hiera_tiny.pt"
            # Must exceed _MIN_VALID_BYTES (10 MB) to pass the size sanity check
            existing.write_bytes(b"x" * (10 * 1024 * 1024))
            dl = ModelDownloader("sam2.1_hiera_tiny")

            with qtbot.waitSignal(dl.finished, timeout=2000):
                dl.run()

            mock_get.assert_not_called()

    @patch("sticker_creator.utils.model_downloader.requests.get")
    def test_file_write_error_emits_error(self, mock_get, qtbot, tmp_path):
        """OSError during file write emits error signal."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"x" * 1000]
        mock_get.return_value = mock_response

        with patch("sticker_creator.utils.model_downloader.MODEL_DIR", tmp_path):
            dl = ModelDownloader("sam2.1_hiera_tiny")
            with patch("builtins.open") as mock_open:
                mock_open.side_effect = OSError("Permission denied")

                with qtbot.waitSignal(dl.error, timeout=2000) as blocker:
                    dl.run()

        assert "Permission denied" in str(blocker.args[0])


# ---------------------------------------------------------------------------
# Integration: ModelDownloader in a background thread
# ---------------------------------------------------------------------------


class TestModelDownloaderThreading:
    """Verify the downloader works correctly when moved to a QThread."""

    @patch("sticker_creator.utils.model_downloader.requests.get")
    def test_download_in_background_thread(self, mock_get, qtbot, tmp_path):
        """Downloader emits finished when run from a background thread."""
        chunk = b"x" * 500
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [chunk, chunk]
        mock_get.return_value = mock_response

        with patch("sticker_creator.utils.model_downloader.MODEL_DIR", tmp_path):
            dl = ModelDownloader("sam2.1_hiera_tiny")
            thread = QThread()
            dl.moveToThread(thread)

            results: list[str] = []
            dl.finished.connect(lambda p: results.append("finished"))

            thread.started.connect(dl.run)
            thread.finished.connect(thread.deleteLater)

            with qtbot.waitSignal(dl.finished, timeout=5000):
                thread.start()

            thread.quit()
            thread.wait(2000)
            assert "finished" in results
