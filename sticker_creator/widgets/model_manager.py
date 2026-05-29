"""Model Manager dialog — overview, download, delete, and select active SAM 2 models.

Presents the SAM 2 variants as a list of cards (KDE HIG style: themed icons,
plain-language descriptions, accent for the active item, never color alone).
Each card downloads in its own ``QThread`` with inline progress, can be deleted
(red trash), and the active model is chosen with a visible "Use" action.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sticker_creator.segmentation.segmenter import Segmenter
from sticker_creator.utils.model_downloader import ModelDownloader


# Maximum simultaneous downloads allowed
_MAX_CONCURRENT_DOWNLOADS = 3

# Friendly name + one-line, jargon-free description per checkpoint.
_MODEL_META: dict[str, tuple[str, str]] = {
    "sam2.1_hiera_tiny": (
        "Tiny",
        "Fastest. A great default — works well on any hardware.",
    ),
    "sam2.1_hiera_small": (
        "Small",
        "A little slower than Tiny, with slightly cleaner edges.",
    ),
    "sam2.1_hiera_base_plus": (
        "Base+",
        "Balanced speed and edge accuracy.",
    ),
    "sam2.1_hiera_large": (
        "Large",
        "Best edge accuracy. Needs more memory and time.",
    ),
}


def _themed(*names: str) -> QIcon:
    """Return the first available themed icon from ``names``."""
    for n in names:
        icon = QIcon.fromTheme(n)
        if not icon.isNull():
            return icon
    return QIcon()


class _DownloadHandle(QObject):
    """Tracks an active download for one model row.

    Owns the ``ModelDownloader``, its ``QThread``, and forwards signals
    to the dialog so the UI can update the corresponding row.
    """

    progress = Signal(str, int)       # model_name, percentage
    finished = Signal(str, object)    # model_name, Path
    error = Signal(str, str)          # model_name, error_message

    def __init__(self, model_name: str, parent: QObject | None = None):
        super().__init__(parent)
        self.model_name = model_name
        self._downloader = ModelDownloader(model_name)
        self._thread = QThread(self)
        self._downloader.moveToThread(self._thread)

        # Forward signals
        self._downloader.progress.connect(self._on_progress)
        self._downloader.finished.connect(self._on_finished)
        self._downloader.error.connect(self._on_error)

        self._thread.started.connect(self._downloader.run)
        self._thread.finished.connect(self._thread.deleteLater)

    def start(self) -> None:
        """Begin the download in the background thread."""
        self._thread.start()

    def abort(self) -> None:
        """Cancel the download and clean up the thread."""
        self._downloader.abort()
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)

    @Slot(int)
    def _on_progress(self, pct: int) -> None:
        self.progress.emit(self.model_name, pct)

    @Slot(object)
    def _on_finished(self, path: object) -> None:
        self.finished.emit(self.model_name, path)
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.error.emit(self.model_name, msg)
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()


class ModelManager(QDialog):
    """Dialog for managing SAM 2 model checkpoints.

    Uses the provided ``Segmenter`` to query available models, trigger
    downloads, delete checkpoints, and switch the active model.

    Signals:
        models_changed: Emitted when the set of available models changes
            (download completed, model deleted, or active model switched).
    """

    models_changed = Signal()

    def __init__(self, segmenter: Segmenter, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Manage Models")
        self.setMinimumWidth(640)
        self.setMinimumHeight(480)

        self._segmenter = segmenter
        self._downloads: dict[str, _DownloadHandle] = {}
        self._active_download_count = 0

        self._build_ui()
        self.refresh()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header — themed icon + heading + explanation.
        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        icon_lbl = QLabel()
        head_icon = _themed("applications-science", "preferences-system", "download")
        if not head_icon.isNull():
            icon_lbl.setPixmap(head_icon.pixmap(32, 32))
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        header_row.addWidget(icon_lbl)

        head_text = QVBoxLayout()
        head_text.setSpacing(2)
        title = QLabel("Segmentation models")
        title.setObjectName("DialogHeading")
        head_text.addWidget(title)
        subtitle = QLabel(
            "Choose which SAM 2 model extracts stickers. Larger models trace "
            "edges more precisely but are slower to run."
        )
        subtitle.setObjectName("DialogSubtitle")
        subtitle.setWordWrap(True)
        head_text.addWidget(subtitle)
        header_row.addLayout(head_text, stretch=1)
        layout.addLayout(header_row)

        # Dismissible inline error bar (hidden until a download fails).
        self._error_bar = QFrame()
        self._error_bar.setObjectName("InlineAlert")
        self._error_bar.setProperty("severity", "error")
        err_layout = QHBoxLayout(self._error_bar)
        err_layout.setContentsMargins(10, 8, 8, 8)
        err_layout.setSpacing(8)
        err_icon = QLabel()
        ei = _themed("dialog-error", "data-error")
        if not ei.isNull():
            err_icon.setPixmap(ei.pixmap(16, 16))
        err_layout.addWidget(err_icon)
        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        err_layout.addWidget(self._error_label, stretch=1)
        err_dismiss = QPushButton("Dismiss")
        err_dismiss.setFlat(True)
        err_dismiss.clicked.connect(self._hide_error)
        err_layout.addWidget(err_dismiss)
        self._error_bar.setVisible(False)
        layout.addWidget(self._error_bar)

        # Scrollable model cards.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setSpacing(8)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.addStretch()
        scroll.setWidget(self._rows_container)
        layout.addWidget(scroll, stretch=1)

        # Standard dialog button box.
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    # ── Public API ─────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild the model cards from the segmenter's current state."""
        models = self._segmenter.available_models()
        active_name = self._segmenter.active_model_name

        self._clear_rows()
        for model_info in models:
            self._add_card(model_info, active_name)
        self._rows_layout.addStretch()

    # ── Card management ────────────────────────────────────────────────────

    def _clear_rows(self) -> None:
        """Remove all dynamically created card widgets and the trailing stretch."""
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                # Reparent out immediately so the old card stops painting now;
                # deleteLater alone leaves it visible (and overlapping the new
                # cards) until the event loop runs.
                widget.setParent(None)
                widget.deleteLater()

    def _add_card(self, model_info: dict, active_name: str | None) -> None:
        """Add one model card to the layout."""
        name = model_info["name"]
        size = model_info["size"]
        is_downloaded = model_info["is_downloaded"]
        is_active = model_info["is_active"]
        is_downloading = name in self._downloads
        title, desc = _MODEL_META.get(name, (name, ""))

        state = (
            "downloading" if is_downloading
            else "active" if is_active
            else "ready" if is_downloaded
            else "available"
        )

        card = QFrame()
        card.setObjectName("ModelCard")
        card.setProperty("state", state)
        card.setProperty("model_name", name)
        cl = QHBoxLayout(card)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.setSpacing(14)

        # Leading package icon — grayed when not yet on disk, solid once
        # downloaded. The active model is conveyed separately (pill + border).
        icon_lbl = QLabel()
        icon_lbl.setFixedSize(32, 32)
        pkg = _themed("package-x-generic", "application-x-archive", "package")
        if not pkg.isNull():
            mode = QIcon.Mode.Normal if is_downloaded else QIcon.Mode.Disabled
            icon_lbl.setPixmap(pkg.pixmap(32, 32, mode))
        icon_lbl.setToolTip("Downloaded" if is_downloaded else "Not downloaded")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        cl.addWidget(icon_lbl)

        # Text block: title (+ Active pill), description, meta, progress.
        text = QVBoxLayout()
        text.setSpacing(3)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("CardTitle")
        text.addWidget(title_lbl)

        if desc:
            desc_lbl = QLabel(desc)
            desc_lbl.setObjectName("CardSubtitle")
            desc_lbl.setWordWrap(True)
            text.addWidget(desc_lbl)

        # Explicit status word + size, in plain language. The active model is
        # shown by the highlighted card border; "In use" here is the matching
        # text cue so the state isn't conveyed by color alone.
        status_word = (
            "Downloading…" if is_downloading
            else "In use" if is_active
            else "Installed" if is_downloaded
            else "Not installed"
        )
        meta_lbl = QLabel(f"{status_word}  ·  {size}")
        meta_lbl.setObjectName("CardMeta")
        text.addWidget(meta_lbl)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setFormat("Downloading… %p%")
        progress_bar.setVisible(is_downloading)
        text.addWidget(progress_bar)

        cl.addLayout(text, stretch=1)

        # Action area.
        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        if is_downloading:
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setIcon(_themed("process-stop", "dialog-cancel"))
            cancel_btn.clicked.connect(lambda _=False, n=name: self._on_cancel_download(n))
            actions.addWidget(cancel_btn)
        elif is_downloaded and not is_active:
            use_btn = QPushButton("Use")
            use_btn.setObjectName("PrimaryButton")
            use_btn.setIcon(_themed("dialog-ok-apply", "checkmark", "dialog-ok"))
            use_btn.clicked.connect(lambda _=False, n=name: self._activate(n))
            actions.addWidget(use_btn)
            delete_btn = QPushButton("Delete")
            delete_btn.setIcon(_themed("edit-delete"))
            delete_btn.clicked.connect(lambda _=False, n=name: self._on_delete(n))
            actions.addWidget(delete_btn)
        elif is_downloaded and is_active:
            # The in-use model can't be deleted; the pill conveys its state and
            # no action is offered. Switch to another model first to remove it.
            pass
        else:
            download_btn = QPushButton("Download")
            download_btn.setObjectName("PrimaryButton")
            download_btn.setIcon(_themed("download", "folder-download"))
            download_btn.clicked.connect(lambda _=False, n=name: self._on_download(n))
            actions.addWidget(download_btn)

        cl.addLayout(actions)

        # Store references for inline progress updates.
        card.setProperty("progress_bar", progress_bar)

        self._rows_layout.addWidget(card)

    # ── Inline error bar ─────────────────────────────────────────────────────

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_bar.setVisible(True)

    def _hide_error(self) -> None:
        self._error_bar.setVisible(False)

    # ── Slot handlers ─────────────────────────────────────────────────────

    def _activate(self, name: str) -> None:
        """Make ``name`` the active model."""
        if name == self._segmenter.active_model_name:
            return
        self._segmenter.set_active_model(name)
        self.models_changed.emit()
        self.refresh()

    @Slot(str)
    def _on_download(self, model_name: str) -> None:
        """Start downloading a model in a background thread."""
        if model_name in self._downloads:
            return  # already downloading

        # Enforce concurrency limit.
        if self._active_download_count >= _MAX_CONCURRENT_DOWNLOADS:
            self._show_error(
                f"A maximum of {_MAX_CONCURRENT_DOWNLOADS} downloads can run at "
                "once. Wait for one to finish before starting another."
            )
            return

        self._hide_error()
        handle = _DownloadHandle(model_name, self)
        handle.progress.connect(self._on_download_progress)
        handle.finished.connect(self._on_download_finished)
        handle.error.connect(self._on_download_error)

        self._downloads[model_name] = handle
        self._active_download_count += 1
        self.refresh()
        handle.start()

    @Slot(str)
    def _on_cancel_download(self, model_name: str) -> None:
        """Cancel an ongoing download."""
        handle = self._downloads.pop(model_name, None)
        if handle is not None:
            handle.abort()
            self._active_download_count -= 1
            self.refresh()

    @Slot(str)
    def _on_delete(self, model_name: str) -> None:
        """Delete a downloaded checkpoint file."""
        self._segmenter.delete_model(model_name)
        self.models_changed.emit()
        self.refresh()

    @Slot(str, int)
    def _on_download_progress(self, model_name: str, pct: int) -> None:
        """Update the progress bar for a downloading model without a full rebuild."""
        if pct >= 100:
            return
        for i in range(self._rows_layout.count()):
            item = self._rows_layout.itemAt(i)
            widget = item.widget() if item else None
            if widget is not None and widget.property("model_name") == model_name:
                pb: QProgressBar | None = widget.property("progress_bar")
                if pb:
                    pb.setValue(pct)
                break

    @Slot(str, object)
    def _on_download_finished(self, model_name: str, path: object) -> None:
        """Handle a completed download."""
        self._downloads.pop(model_name, None)
        self._active_download_count -= 1
        self.models_changed.emit()
        self.refresh()

    @Slot(str, str)
    def _on_download_error(self, model_name: str, error_msg: str) -> None:
        """Handle a download error."""
        self._downloads.pop(model_name, None)
        self._active_download_count -= 1
        title = _MODEL_META.get(model_name, (model_name, ""))[0]
        self._show_error(f"Could not download {title}: {error_msg}")
        self.models_changed.emit()
        self.refresh()
