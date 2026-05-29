"""Main window — KDE HIG-aligned, hamburger-menu single-window layout.

Empty state: PlaceholderView. Loaded state: ImageViewer central, InspectorPanel
docked right. Live mask drives a live sticker; Save/Copy always operate on the
current sticker. No explicit "confirm" step, no step indicator.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QSize, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStackedWidget,
    QStyle,
    QStatusBar,
    QToolBar,
    QToolButton,
    QWidget,
)

from sticker_creator.segmentation.segmenter import Segmenter
from sticker_creator.session import StickerSession
from sticker_creator.utils.clipboard import ClipboardManager
from sticker_creator.utils.file_io import ImageLoader, ImageSaver
from sticker_creator.utils.sticker_pipeline import resize_for_whatsapp
from sticker_creator.utils.icons import icon_undo, icon_clear
from sticker_creator.utils.settings import Keys, get_settings
from sticker_creator.widgets.drop_overlay import ACCEPTED_EXTENSIONS, DropOverlay
from sticker_creator.widgets.image_viewer import ImageViewer
from sticker_creator.widgets.inspector_panel import InspectorPanel
from sticker_creator.widgets.model_manager import ModelManager
from sticker_creator.widgets.placeholder_view import PlaceholderView
from sticker_creator.widgets.help_dialog import HelpDialog
from sticker_creator.widgets.shortcuts_dialog import ShortcutsDialog
from sticker_creator.widgets.toast_notification import ToastOverlay
from sticker_creator.widgets.welcome_dialog import WelcomeDialog


def _sanitize_for_filename(text: str, max_len: int = 40) -> str:
    """Reduce arbitrary text to a safe filename fragment.

    Keeps alphanumerics, spaces, hyphens, underscores; collapses runs to
    single underscore; strips leading/trailing separators.
    """
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    cleaned = re.sub(r"[\s_-]+", "_", cleaned).strip("_")
    return cleaned[:max_len]


def _themed(name: str, fallback_sp: QStyle.StandardPixmap | None = None,
            style: QStyle | None = None) -> QIcon:
    icon = QIcon.fromTheme(name)
    if not icon.isNull():
        return icon
    if fallback_sp is not None and style is not None:
        return style.standardIcon(fallback_sp)
    return icon


class MainWindow(QMainWindow):
    """Single-window sticker editor."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sticker Creator")
        self.setMinimumSize(900, 600)
        self.resize(1200, 760)

        # Workflow state lives in the session; the window owns only UI-local state.
        self.session = StickerSession(self)
        self._seg_start_time: float | None = None
        self._welcome_shown: bool = False

        # Services
        self.segmenter = Segmenter()
        self.image_loader = ImageLoader()
        self.image_saver = ImageSaver()
        self.clipboard_manager = ClipboardManager()

        # Overlays (must exist before connections — model load may emit early)
        self._toast = ToastOverlay(self)
        self._drop_overlay = DropOverlay(self)

        self._build_central()
        self._build_inspector_dock()
        self._build_toolbar()
        self._build_statusbar()
        self._wire_signals()
        self._restore_window_state()
        self._refresh_model_list()

        self.setAcceptDrops(True)
        self.placeholder.set_model_status("Loading model…")
        self.segmenter.load_model()

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_central(self) -> None:
        self.image_viewer = ImageViewer()
        self.placeholder = PlaceholderView()

        self._stack = QStackedWidget()
        self._stack.addWidget(self.placeholder)    # 0
        self._stack.addWidget(self.image_viewer)   # 1
        self.setCentralWidget(self._stack)

    def _build_inspector_dock(self) -> None:
        self.inspector = InspectorPanel()
        self._dock = QDockWidget("Sticker", self)
        self._dock.setObjectName("InspectorDock")
        self._dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self._dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._dock.setWidget(self.inspector)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)
        self.resizeDocks([self._dock], [340], Qt.Orientation.Horizontal)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main Toolbar")
        tb.setObjectName("MainToolbar")
        tb.setMovable(False)
        tb.setFloatable(False)
        size = self.style().pixelMetric(QStyle.PixelMetric.PM_ToolBarIconSize)
        tb.setIconSize(QSize(size, size))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        self._toolbar = tb

        style = self.style()

        self._open_action = QAction(
            _themed("document-open", QStyle.StandardPixmap.SP_DialogOpenButton, style),
            "Open…", self,
        )
        self._open_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_action.setToolTip("Open an image (Ctrl+O)")
        self._open_action.triggered.connect(self._on_open_image)
        tb.addAction(self._open_action)

        self._paste_action = QAction(
            _themed("edit-paste"), "Paste", self,
        )
        self._paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        self._paste_action.setToolTip("Paste image from clipboard (Ctrl+V)")
        self._paste_action.triggered.connect(self._on_paste_image)
        tb.addAction(self._paste_action)

        tb.addSeparator()

        self._undo_action = QAction(
            QIcon.fromTheme("edit-undo", icon_undo()), "Undo Point", self
        )
        self._undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_action.setToolTip("Undo last point (Ctrl+Z)")
        self._undo_action.triggered.connect(self.session.undo_prompt)
        self._undo_action.setEnabled(False)
        self.addAction(self._undo_action)

        self._clear_action = QAction(
            QIcon.fromTheme("edit-clear", icon_clear()), "Clear All", self
        )
        self._clear_action.setShortcut(QKeySequence("Ctrl+R"))
        self._clear_action.setToolTip("Clear all points (Ctrl+R)")
        self._clear_action.triggered.connect(self.session.clear_prompts)
        self._clear_action.setEnabled(False)
        self.addAction(self._clear_action)

        self._save_action = QAction(
            _themed("document-save", QStyle.StandardPixmap.SP_DialogSaveButton, style),
            "Save Sticker…", self,
        )
        self._save_action.setShortcut(QKeySequence.StandardKey.Save)
        self._save_action.setToolTip("Save sticker (Ctrl+S)")
        self._save_action.triggered.connect(self._on_save_sticker)
        self._save_action.setEnabled(False)
        tb.addAction(self._save_action)

        self._copy_action = QAction(
            _themed("edit-copy"), "Copy Sticker", self,
        )
        self._copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        self._copy_action.setToolTip("Copy sticker to clipboard (Ctrl+C)")
        self._copy_action.triggered.connect(self._on_copy_sticker)
        self._copy_action.setEnabled(False)
        tb.addAction(self._copy_action)

        from PySide6.QtWidgets import QSizePolicy
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self._help_action = QAction(
            _themed("help-contents", QStyle.StandardPixmap.SP_DialogHelpButton, style),
            "Help", self,
        )
        self._help_action.setShortcut(QKeySequence("F1"))
        self._help_action.setToolTip("Open help (F1)")
        self._help_action.triggered.connect(self._on_help)
        tb.addAction(self._help_action)

        self._hamburger_btn = QToolButton()
        self._hamburger_btn.setIcon(_themed(
            "application-menu",
            QStyle.StandardPixmap.SP_TitleBarMenuButton, style,
        ))
        self._hamburger_btn.setText("Menu")
        self._hamburger_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._hamburger_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._hamburger_btn.setToolTip("Application menu")
        self._hamburger_btn.setMenu(self._build_app_menu())
        tb.addWidget(self._hamburger_btn)

    def _build_app_menu(self) -> QMenu:
        menu = QMenu(self)

        menu.addAction(self._open_action)
        menu.addAction(self._paste_action)
        menu.addSeparator()
        menu.addAction(self._save_action)
        menu.addAction(self._copy_action)
        menu.addSeparator()

        manage_models = QAction(_themed("preferences-system"), "Manage Models…", self)
        manage_models.setShortcut(QKeySequence("Ctrl+M"))
        manage_models.triggered.connect(self._on_manage_models)
        menu.addAction(manage_models)

        shortcuts = QAction(_themed("input-keyboard"), "Keyboard Shortcuts…", self)
        shortcuts.setShortcut(QKeySequence("Ctrl+/"))
        shortcuts.triggered.connect(self._on_shortcuts)
        menu.addAction(shortcuts)

        # The toolbar already owns the F1-bound Help action; the menu entry
        # just re-exposes it so users browsing the menu can find it too.
        menu.addAction(self._help_action)

        welcome = QAction(_themed("user-home"), "Welcome…", self)
        welcome.triggered.connect(self._show_welcome)
        menu.addAction(welcome)

        about = QAction(_themed("help-about"), "About Sticker Creator", self)
        about.triggered.connect(self._on_about)
        menu.addAction(about)

        menu.addSeparator()

        quit_act = QAction(_themed("application-exit"), "Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        menu.addAction(quit_act)

        return menu

    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self._status = bar
        self._model_status_label = QLabel()
        self._model_status_label.setObjectName("StatusModelLabel")
        bar.addPermanentWidget(self._model_status_label)
        self._update_model_status_label()

    # ── Signal wiring ──────────────────────────────────────────────────────

    def _wire_signals(self) -> None:
        # Session — workflow state drives the widgets through these signals.
        self.session.image_changed.connect(self._on_session_image_changed)
        self.session.prompt_points_changed.connect(self._on_session_points_changed)
        self.session.segment_requested.connect(self._on_segment_requested)
        self.session.mask_changed.connect(self.image_viewer.set_mask)
        self.session.border_width_changed.connect(self.inspector.set_border_width)
        self.session.sticker_changed.connect(self._on_session_sticker_changed)

        # Segmenter
        self.segmenter.model_loaded.connect(self._on_model_loaded)
        self.segmenter.model_error.connect(self._on_model_error)
        self.segmenter.mask_ready.connect(self._on_mask_ready)
        self.segmenter.processing_started.connect(self._on_seg_started)
        self.segmenter.processing_finished.connect(self._on_seg_finished)

        # Image viewer — raw view intents drive the session's workflow.
        self.image_viewer.point_clicked.connect(self.session.add_prompt)
        self.image_viewer.undo_requested.connect(self.session.undo_prompt)
        self.image_viewer.clear_requested.connect(self.session.clear_prompts)

        # Placeholder (also hosts model setup)
        self.placeholder.open_image_requested.connect(self._on_open_image)
        self.placeholder.paste_image_requested.connect(self._on_paste_image)
        self.placeholder.model_selected.connect(self._on_model_selected)
        self.placeholder.manage_models_requested.connect(self._on_manage_models)

        # Inspector
        self.inspector.save_requested.connect(self._on_save_sticker)
        self.inspector.copy_requested.connect(self._on_copy_sticker)
        self.inspector.border_toggled.connect(self._on_border_toggled)
        self.inspector.border_width_changed.connect(self._on_border_width_changed)
        self.inspector.background_changed.connect(self._on_background_changed)
        self.inspector.balloon_text_changed.connect(self._on_balloon_text_changed)
        self.inspector.balloon_style_changed.connect(self._on_balloon_style_changed)

    # ── Image loading ──────────────────────────────────────────────────────

    @Slot()
    def _on_open_image(self) -> None:
        start_dir = get_settings().value(Keys.LAST_OPEN_DIR, str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif);;All Files (*)",
        )
        if path:
            self._load_image(path)

    @Slot()
    def _on_paste_image(self) -> None:
        image = self.clipboard_manager.paste_image()
        if image is None:
            self._status.showMessage("No image found in clipboard", 3000)
            return
        self.session.set_image(image, path=None)

    def _load_image(self, file_path: str) -> None:
        image = self.image_loader.load(file_path)
        if image is None:
            QMessageBox.warning(self, "Open Image", f"Failed to load image:\n{file_path}")
            return
        get_settings().setValue(Keys.LAST_OPEN_DIR, str(Path(file_path).parent))
        self.session.set_image(image, path=file_path)

    # ── Session → widgets ────────────────────────────────────────────────────

    @Slot(object)
    def _on_session_image_changed(self, image) -> None:
        if image is None:
            return
        self._stack.setCurrentIndex(1)
        self.image_viewer.set_image(image)  # toolstrip enable + mode switch happens inside
        self.inspector.dismiss_alert()

    @Slot(object)
    def _on_session_points_changed(self, points: list[dict]) -> None:
        self.image_viewer.set_points(points)
        self._refresh_action_state()

    @Slot(object)
    def _on_session_sticker_changed(self, preview) -> None:
        self.inspector.set_sticker(preview)
        self._refresh_action_state()

    # ── Segmentation events ────────────────────────────────────────────────

    @Slot(object, object)
    def _on_segment_requested(self, image, points: list[dict]) -> None:
        """Dispatch a session segmentation request to the background service."""
        if self.segmenter.active_model_name is None:
            self._status.showMessage("A segmentation model is required.", 4000)
            return
        self.segmenter.segment(image=image, points=points)

    @Slot(object)
    def _on_mask_ready(self, mask: np.ndarray) -> None:
        if self._seg_start_time is not None:
            elapsed = time.time() - self._seg_start_time
            self._status.showMessage(f"Segmented in {elapsed:.2f}s", 3000)
            self._seg_start_time = None

        self.session.set_mask(mask)

    @Slot()
    def _on_seg_started(self) -> None:
        self._seg_start_time = time.time()
        self.image_viewer.set_busy(True)

    @Slot()
    def _on_seg_finished(self) -> None:
        self.image_viewer.set_busy(False)

    # ── Model events ───────────────────────────────────────────────────────

    def _update_model_status_label(self) -> None:
        name = self.segmenter.active_model_name
        if name:
            self._model_status_label.setText(f"  Model: {name}  ")
        else:
            self._model_status_label.setText("  No model loaded  ")

    @Slot(str)
    def _on_model_loaded(self, model_name: str) -> None:
        self.placeholder.set_model_status("")
        self.placeholder.dismiss_alert()
        self._refresh_model_list()
        self._update_model_status_label()

    @Slot(str)
    def _on_model_error(self, _err: str) -> None:
        self.placeholder.set_model_status("")
        self.placeholder.show_alert(
            "No segmentation model installed. Install one to start segmenting.",
            severity="info",
        )
        self._refresh_model_list()
        self._update_model_status_label()

    @Slot()
    def _on_manage_models(self) -> None:
        dialog = ModelManager(self.segmenter, self)
        dialog.models_changed.connect(self._refresh_model_list)
        dialog.exec()

    @Slot(str)
    def _on_model_selected(self, name: str) -> None:
        if name and name != self.segmenter.active_model_name:
            self.placeholder.set_model_status(f"Loading {name}…")
            self._model_status_label.setText(f"  Loading {name}…  ")
            self.segmenter.set_active_model(name)

    def _refresh_model_list(self) -> None:
        models = self.segmenter.available_models()
        self.placeholder.set_models(models, self.segmenter.active_model_name)

    # ── Border / Background ────────────────────────────────────────────────

    @Slot(bool)
    def _on_border_toggled(self, enabled: bool) -> None:
        self.session.set_border_enabled(enabled)

    @Slot(int)
    def _on_border_width_changed(self, width: int) -> None:
        self.session.set_border_width(width)

    @Slot(str)
    def _on_background_changed(self, color: str) -> None:
        self.session.set_background(color)

    @Slot(str)
    def _on_balloon_text_changed(self, text: str) -> None:
        self.session.set_balloon_text(text)

    @Slot(str)
    def _on_balloon_style_changed(self, style: str) -> None:
        self.session.set_balloon_style(style)

    # ── Save / Copy ────────────────────────────────────────────────────────

    @Slot()
    def _on_save_sticker(self) -> None:
        sticker = self.session.current_sticker
        if sticker is None:
            return
        image_path = self.session.image_path
        base = Path(image_path).stem if image_path else "sticker"
        balloon_slug = _sanitize_for_filename(self.session.balloon_text)
        if balloon_slug:
            default_name = f"{base}_{balloon_slug}_sticker.png"
        else:
            default_name = f"{base}_sticker.png"

        settings = get_settings()
        start_dir = settings.value(Keys.LAST_SAVE_DIR, str(Path.home()))
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Sticker",
            str(Path(start_dir) / default_name),
            (
                "PNG Image (*.png);;"
                "WebP Image (*.webp);;"
                "WhatsApp Sticker — 512×512 WebP (*.webp);;"
                "All Files (*)"
            ),
        )
        if not file_path:
            return
        path = Path(file_path)
        is_whatsapp = "WhatsApp Sticker" in (selected_filter or "")
        if is_whatsapp:
            self.image_saver.save_webp(resize_for_whatsapp(sticker), path, quality=80)
        elif path.suffix.lower() == ".webp":
            self.image_saver.save_webp(sticker, path, quality=100)
        else:
            self.image_saver.save_png(sticker, path)
        settings.setValue(Keys.LAST_SAVE_DIR, str(path.parent))
        self._toast.show_success(f"Saved: {path.name}")

    @Slot()
    def _on_copy_sticker(self) -> None:
        sticker = self.session.current_sticker
        if sticker is None:
            return
        self.clipboard_manager.copy_image(sticker)
        self._toast.show_success("Sticker copied to clipboard")

    # ── Misc actions ───────────────────────────────────────────────────────

    @Slot()
    def _on_help(self) -> None:
        HelpDialog(self).show()

    @Slot()
    def _on_shortcuts(self) -> None:
        ShortcutsDialog(self).show()

    @Slot()
    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About Sticker Creator",
            "<h3>Sticker Creator</h3>"
            "<p>Extract transparent stickers from images using Meta's SAM 2 "
            "segmentation model.</p>"
            "<p>Built with PySide6 for KDE.</p>",
        )

    # ── Action enablement ──────────────────────────────────────────────────

    def _refresh_action_state(self) -> None:
        has_image = self.session.current_image is not None
        has_points = has_image and self.session.has_prompt_points
        has_sticker = self.session.has_sticker

        self._undo_action.setEnabled(has_points)
        self._clear_action.setEnabled(has_points)
        self._save_action.setEnabled(has_sticker)
        self._copy_action.setEnabled(has_sticker)

    # ── Drag and drop ──────────────────────────────────────────────────────

    def dragEnterEvent(self, event):  # type: ignore[override]
        mime = event.mimeData()
        if not mime.hasUrls():
            return
        for url in mime.urls():
            path = url.toLocalFile()
            if any(path.lower().endswith(ext) for ext in ACCEPTED_EXTENSIONS):
                event.acceptProposedAction()
                self._drop_overlay.activate()
                return

    def dragLeaveEvent(self, event):  # type: ignore[override]
        self._drop_overlay.deactivate()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):  # type: ignore[override]
        self._drop_overlay.deactivate()
        mime = event.mimeData()
        if not mime.hasUrls():
            return
        for url in mime.urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in ACCEPTED_EXTENSIONS:
                event.acceptProposedAction()
                self._load_image(path)
                return

    # ── Window state ───────────────────────────────────────────────────────

    def _restore_window_state(self) -> None:
        s = get_settings()
        geom = s.value(Keys.GEOMETRY)
        state = s.value(Keys.WINDOW_STATE)
        if geom:
            self.restoreGeometry(geom)
        if state:
            self.restoreState(state)

    def showEvent(self, event):  # type: ignore[override]
        super().showEvent(event)
        if self._welcome_shown:
            return
        self._welcome_shown = True
        s = get_settings()
        if s.value(Keys.SHOW_WELCOME, True, type=bool):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._show_welcome)

    def _show_welcome(self) -> None:
        dlg = WelcomeDialog(self)
        dlg.exec()
        if dlg.dont_show_again():
            s = get_settings()
            s.setValue(Keys.SHOW_WELCOME, False)
            s.sync()

    def closeEvent(self, event):  # type: ignore[override]
        s = get_settings()
        s.setValue(Keys.GEOMETRY, self.saveGeometry())
        s.setValue(Keys.WINDOW_STATE, self.saveState())
        s.sync()
        self.segmenter.shutdown()
        self._toast.deleteLater()
        self._drop_overlay.deleteLater()
        super().closeEvent(event)
