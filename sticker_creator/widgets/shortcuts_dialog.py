"""Keyboard Shortcuts reference dialog."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QWidget,
)


_SHORTCUTS: list[tuple[str, str, str]] = [
    # (shortcut, action, category)
    ("Ctrl+O",       "Open Image",             "File"),
    ("Ctrl+V",       "Paste from Clipboard",   "File"),
    ("Ctrl+S",       "Save Sticker",           "File"),
    ("Ctrl+Shift+S", "Save Sticker As…",       "File"),
    ("Ctrl+Q",       "Quit",                   "File"),
    ("Ctrl+C",       "Copy Sticker",           "Edit"),
    ("Ctrl+R",       "Clear Points",           "Edit"),
    ("Ctrl+Z",       "Undo Last Point",        "Edit"),
    ("Return",       "Confirm Segmentation",   "Segment"),
    ("Ctrl+M",       "Manage Models",          "Models"),
    ("",             "",                       ""),  # separator
    # Mouse / annotation
    ("Left-click",   "Place point (in Add mode)",  "Mouse"),
    ("Shift+click",  "Stay in Add mode after click", "Mouse"),
    ("Mouse wheel",  "Zoom in / out",               "Mouse"),
    ("Drag",         "Pan the image",               "Mouse"),
]


class ShortcutsDialog(QDialog):
    """Modeless dialog listing all keyboard and mouse shortcuts."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(440, 400)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Keyboard & Mouse Shortcuts")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # ── Table ──────────────────────────────────────────────────────
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setHighlightSections(False)

        # Populate rows
        table.setRowCount(len(_SHORTCUTS))
        for row, (key, action, category) in enumerate(_SHORTCUTS):
            if not key and not action:
                # Separator
                table.setRowHeight(row, 8)
                item_key = QTableWidgetItem("")
                item_key.setFlags(Qt.ItemFlag.NoItemFlags)
                item_action = QTableWidgetItem("")
                item_action.setFlags(Qt.ItemFlag.NoItemFlags)
                table.setItem(row, 0, item_key)
                table.setItem(row, 1, item_action)
            else:
                key_item = QTableWidgetItem(key)
                key_font = QFont("monospace")
                key_font.setPointSize(11)
                key_font.setBold(True)
                key_item.setFont(key_font)
                key_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                table.setItem(row, 0, key_item)

                action_item = QTableWidgetItem(action)
                action_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                )
                table.setItem(row, 1, action_item)

        table.resizeRowsToContents()
        layout.addWidget(table, stretch=1)

        # Close button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
