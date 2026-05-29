"""In-app help dialog backed by the Qt Help framework.

Renders the bundled ``sticker_creator.qhc`` collection via ``QHelpEngine``
with a contents tree, keyword index, and full-text search on the left,
and a custom ``QTextBrowser`` on the right that resolves ``qthelp://``
resource URLs through the engine.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QByteArray, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QTextDocument
from PySide6.QtHelp import QHelpEngine
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtCore import QStandardPaths
except ImportError:  # pragma: no cover
    QStandardPaths = None


_HELP_DIR = Path(__file__).resolve().parent.parent / "resources" / "help"
_QHC_NAME = "sticker_creator.qhc"
_QCH_NAME = "sticker_creator.qch"


def _writable_collection_path() -> Path:
    """Stage the read-only bundled help into a writable user-data location.

    ``QHelpEngine`` writes user state into the ``.qhc`` collection file, so
    it must live somewhere writable. We mirror the bundle into
    ``AppDataLocation/help/`` on first use and run the engine from there.
    """
    if QStandardPaths is None:
        return _HELP_DIR / _QHC_NAME
    base = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )) / "help"
    base.mkdir(parents=True, exist_ok=True)
    for name in (_QHC_NAME, _QCH_NAME):
        src = _HELP_DIR / name
        dst = base / name
        if src.exists() and (
            not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime
        ):
            shutil.copy2(src, dst)
    return base / _QHC_NAME


def _palette_stylesheet(widget: QWidget) -> str:
    """Build a small CSS sheet driven by the current Qt palette.

    QTextBrowser can't read OS dark-mode hints, so we plug in concrete
    colors here. Re-evaluated whenever a new document is loaded so the
    help follows the user's theme.
    """
    pal = widget.palette()
    text = pal.text().color().name()
    link = pal.link().color().name()
    accent = pal.highlight().color().name()
    code_bg = pal.alternateBase().color().name()
    mid = pal.mid().color().name()
    return (
        f"body {{ color: {text}; }}\n"
        f"h1, h2, h3 {{ color: {accent}; }}\n"
        f"h1 {{ border-bottom: 1px solid {mid}; }}\n"
        f"code, pre {{ background-color: {code_bg}; color: {text}; }}\n"
        f"th {{ background-color: {code_bg}; color: {text}; }}\n"
        f"th, td {{ border: 1px solid {mid}; }}\n"
        f"a {{ color: {link}; }}\n"
        f"img {{ border: 1px solid {mid}; }}\n"
    )


class _HelpBrowser(QTextBrowser):
    """QTextBrowser that resolves ``qthelp://`` resources through QHelpEngine."""

    def __init__(self, engine: QHelpEngine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self._on_anchor)
        self._stylesheet = _palette_stylesheet(self)

    def loadResource(self, type_: int, name: QUrl):  # type: ignore[override]
        if name.scheme() == "qthelp":
            data: QByteArray = self._engine.fileData(name)
            return data
        return super().loadResource(type_, name)

    def setSource(self, name: QUrl, type_=None):  # type: ignore[override]
        if name.scheme() == "qthelp":
            data = self._engine.fileData(name)
            text = bytes(data).decode("utf-8", errors="replace")
            doc = QTextDocument(self)
            doc.setBaseUrl(name)
            doc.setDefaultStyleSheet(self._stylesheet)
            doc.setHtml(text)
            self.setDocument(doc)
            return
        super().setSource(name, type_) if type_ is not None else super().setSource(name)

    def changeEvent(self, event):  # type: ignore[override]
        # Refresh the palette-derived stylesheet when the application theme
        # changes (e.g., user toggled dark mode).
        from PySide6.QtCore import QEvent
        if event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.ThemeChange,
        ):
            self._stylesheet = _palette_stylesheet(self)
            doc = self.document()
            if doc is not None:
                doc.setDefaultStyleSheet(self._stylesheet)
                # Force a re-layout so the new colors take effect.
                html = doc.toHtml()
                doc.setHtml(html)
        super().changeEvent(event)

    def _on_anchor(self, url: QUrl) -> None:
        if url.scheme() in {"http", "https", "mailto"}:
            QDesktopServices.openUrl(url)
            return
        if url.isRelative():
            url = self.source().resolved(url)
        self.setSource(url)


class HelpDialog(QDialog):
    """Modal-less help window with contents / index / search panes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Help — Sticker Creator")
        self.setMinimumSize(900, 640)
        self.resize(1080, 760)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        qhc_path = _writable_collection_path()
        self._engine = QHelpEngine(str(qhc_path), self)
        self._engine.setupData()
        self._engine.searchEngine().reindexDocumentation()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self._browser = _HelpBrowser(self._engine, self)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._browser)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 820])
        root.addWidget(splitter, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_btn: QPushButton = buttons.button(QDialogButtonBox.StandardButton.Close)
        close_btn.setDefault(True)
        buttons.rejected.connect(self.reject)
        wrapper = QVBoxLayout()
        wrapper.setContentsMargins(12, 8, 12, 12)
        wrapper.addWidget(buttons)
        root.addLayout(wrapper)

        # Show the home page once the engine is ready.
        home_url = QUrl("qthelp://org.kde.stickercreator/sticker-creator/index.html")
        self._browser.setSource(home_url)

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        # Contents tree.
        contents = self._engine.contentWidget()
        contents.linkActivated.connect(self._browser.setSource)
        # Expand the root node so all sections are visible immediately.
        self._engine.contentModel().contentsCreated.connect(contents.expandAll)
        tabs.addTab(contents, "Contents")

        # Keyword index.
        index = self._engine.indexWidget()
        index.linkActivated.connect(lambda url, _kw="": self._browser.setSource(url))
        index_holder = QWidget()
        index_layout = QVBoxLayout(index_holder)
        index_layout.setContentsMargins(4, 4, 4, 4)
        index_layout.setSpacing(4)
        filter_edit = QLineEdit()
        filter_edit.setPlaceholderText("Filter keywords…")
        filter_edit.textChanged.connect(self._filter_index)
        index_layout.addWidget(filter_edit)
        index_layout.addWidget(index)
        self._index_widget = index
        tabs.addTab(index_holder, "Index")

        # Full-text search.
        search_engine = self._engine.searchEngine()
        search_holder = QWidget()
        search_layout = QVBoxLayout(search_holder)
        search_layout.setContentsMargins(4, 4, 4, 4)
        search_layout.setSpacing(4)
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search…")
        self._search_edit.returnPressed.connect(self._run_search)
        search_layout.addWidget(self._search_edit)
        self._search_results = QListWidget()
        self._search_results.itemActivated.connect(self._on_search_hit)
        search_layout.addWidget(self._search_results, 1)
        search_engine.searchingFinished.connect(self._on_search_done)
        tabs.addTab(search_holder, "Search")

        return tabs

    # ── Search wiring ────────────────────────────────────────────────────────

    def _filter_index(self, text: str) -> None:
        """Live-filter the keyword index by prefix.

        ``QHelpIndexModel.filter`` isn't exposed cleanly in PySide6, so we
        drive the visible index widget via its native ``filterIndices`` slot.
        """
        try:
            self._index_widget.filterIndices(text)
        except AttributeError:
            pass

    def _run_search(self) -> None:
        query = self._search_edit.text().strip()
        if not query:
            self._search_results.clear()
            return
        self._engine.searchEngine().search(query)

    def _on_search_done(self) -> None:
        self._search_results.clear()
        results = self._engine.searchEngine().searchResults(0, 50)
        for hit in results:
            item = QListWidgetItem(hit.title() or hit.url().toString())
            item.setData(Qt.ItemDataRole.UserRole, hit.url())
            item.setToolTip(hit.url().toString())
            self._search_results.addItem(item)

    def _on_search_hit(self, item: QListWidgetItem) -> None:
        url: QUrl = item.data(Qt.ItemDataRole.UserRole)
        if url is not None:
            self._browser.setSource(url)
