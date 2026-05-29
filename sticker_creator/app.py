"""Application setup and theme configuration for Sticker Creator."""

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QStyleFactory,
    QMessageBox,
)

from sticker_creator.main_window import MainWindow


class StickerCreatorApp:
    """Application wrapper that sets up QApplication with KDE/Breeze theming."""

    APP_ID = "org.kde.stickercreator"
    APP_NAME = "Sticker Creator"
    ORG_NAME = "KDE"
    ORG_DOMAIN = "kde.org"

    def __init__(self, argv: list[str]):
        """Initialize the QApplication with KDE-friendly settings."""
        QApplication.setApplicationName(self.APP_NAME)
        QApplication.setOrganizationName(self.ORG_NAME)
        QApplication.setOrganizationDomain(self.ORG_DOMAIN)
        QApplication.setApplicationDisplayName(self.APP_NAME)
        QApplication.setDesktopFileName(f"{self.APP_ID}.desktop")
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

        self.qapp = QApplication(argv)
        self._apply_window_icon()
        self._configure_theme()
        self._apply_stylesheet()

        self.window: MainWindow | None = None

    def _configure_theme(self) -> None:
        """Configure the application theme, preferring KDE Breeze."""
        # Try to use system (KDE Breeze) style
        available_styles = QStyleFactory.keys()
        preferred_styles = ["breeze", "fusion"]

        for style in preferred_styles:
            if style.lower() in [s.lower() for s in available_styles]:
                self.qapp.setStyle(style)
                break

        # Enhance dark palette detection
        palette = self.qapp.palette()
        is_dark = palette.color(QPalette.Window).lightness() < 128

        if is_dark:
            self._apply_dark_palette(palette)

    def _apply_dark_palette(self, palette: QPalette) -> None:
        """Refine palette for dark mode if needed."""
        palette.setColor(QPalette.Highlight, QColor("#3daee9"))  # KDE blue
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        self.qapp.setPalette(palette)

    def _apply_stylesheet(self) -> None:
        """Load the bundled stylesheet if it is available."""
        style = self._load_stylesheet()
        if style:
            self.qapp.setStyleSheet(style)

    def _apply_window_icon(self) -> None:
        """Set the application/window icon.

        Prefer the bundled SVG so Qt rasterizes crisply at every size; fall
        back to the system theme icon ``sticker-creator`` (matching the
        ``.desktop`` file) so an installed theme override wins.
        """
        svg_path = Path(__file__).resolve().parent / "resources" / "icons" / "app_icon.svg"
        icon = QIcon(str(svg_path)) if svg_path.exists() else QIcon()
        theme_icon = QIcon.fromTheme("sticker-creator")
        if not theme_icon.isNull():
            icon = theme_icon
        if not icon.isNull():
            self.qapp.setWindowIcon(icon)

    def _load_stylesheet(self) -> str | None:
        """Load optional QSS stylesheet from resources."""
        style_path = Path(__file__).resolve().parent / "resources" / "style.qss"
        if style_path.exists():
            return style_path.read_text()
        return None

    def run(self) -> int:
        """Run the application, returning the exit code."""
        self.window = MainWindow()
        self.window.show()
        return self.qapp.exec()
