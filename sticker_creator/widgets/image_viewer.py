"""Zoomable/pannable image viewer with point markers and boundary overlay.

This widget displays the source image, handles mouse interaction for placing
segmentation points, and renders the SAM 2 mask boundary overlay with
semi-transparent coloring.
"""

import numpy as np

from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QLineF, QEvent, QSize, QTimer
from PySide6.QtGui import (
    QIcon,
    QPixmap,
    QImage,
    QPainter,
    QPen,
    QBrush,
    QColor,
    QPolygonF,
    QCursor,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
    QToolButton,
)

from sticker_creator.utils.icons import (
    icon_pan,
    icon_add_positive,
    icon_add_negative,
    icon_undo,
    icon_clear,
)
from sticker_creator.widgets.mask_contours import mask_to_contours
from sticker_creator.widgets.view_transform import ViewTransform



_OVERLAY_ICON_SIZE = QSize(18, 18)


# ─── Helper: compute contour poly from mask ───────────────────────────────


def _contour_to_polygon(contour: np.ndarray) -> QPolygonF:
    """Convert an OpenCV contour (N,1,2) to a QPolygonF."""
    pts = [QPointF(float(p[0][0]), float(p[0][1])) for p in contour]
    return QPolygonF(pts)


def _create_checkerboard_pattern(
    width: int, height: int, tile: int = 8
) -> QPixmap:
    """Create a small checkerboard pixmap for the transparency background."""
    pix = QPixmap(width, height)
    pix.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pix)
    light = QColor(200, 200, 200)
    dark = QColor(170, 170, 170)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            color = light if ((x // tile) + (y // tile)) % 2 == 0 else dark
            painter.fillRect(x, y, tile, tile, color)
    painter.end()
    return pix


# ─── Overlay group (mask tint + point markers) ───────────────────────────


class _OverlayGroup(QGraphicsItem):
    """Group item that paints point markers and mask overlay on the scene.

    This is attached to the pixmap item so transforms stay in image-pixel space.
    """

    def __init__(self, parent_item: QGraphicsItem | None = None):
        super().__init__(parent_item)
        self._points: list[dict] = []
        self._mask: np.ndarray | None = None
        self._image_shape: tuple[int, ...] | None = None
        self._contours: list[np.ndarray] = []

    def set_points(self, points: list[dict]):
        self._points = list(points)
        self.update()

    def set_mask(self, mask: np.ndarray | None):
        self._mask = mask
        if mask is not None:
            self._contours = mask_to_contours(mask)
        else:
            self._contours = []
        self.update()

    def clear_all(self):
        self._points.clear()
        self._mask = None
        self._contours.clear()
        self.update()

    def boundingRect(self) -> QRectF:
        # Cover the entire parent pixmap area
        parent = self.parentItem()
        if parent and isinstance(parent, QGraphicsPixmapItem):
            return parent.boundingRect()
        return QRectF()

    def paint(self, painter: QPainter, option, widget=None):
        # ── 1. Mask fill (semi-transparent overlay) ──────────────────────
        if self._mask is not None:
            rect = self.boundingRect()
            w = int(rect.width())
            h = int(rect.height())

            # Create a coloured overlay image sized to the mask
            orig_h, orig_w = self._mask.shape[:2]

            # Resize mask overlay to match scene dimensions if needed
            overlay = np.zeros((h, w, 4), dtype=np.uint8)

            # Blue-green tint for selected region
            selected = self._mask > 0
            overlay[selected, 0] = 46   # B
            overlay[selected, 1] = 180  # G
            overlay[selected, 2] = 247  # R  (BGR for OpenCV is actually RGB here)
            overlay[selected, 3] = 60   # Alpha — semi-transparent

            # Convert to QImage and draw
            qoverlay = QImage(
                overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888
            )
            painter.drawImage(QPointF(0, 0), qoverlay)

            # ── 2. Boundary contour line ─────────────────────────────
            pen = QPen(QColor("#3daee9"), 2.5)  # KDE blue
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            for contour in self._contours:
                polygon = _contour_to_polygon(contour)
                painter.drawPolygon(polygon)

        # ── 3. Point markers ──────────────────────────────────────────────
        for pt in self._points:
            x, y, label = pt["x"], pt["y"], pt["label"]

            # Outer circle (larger, semi-transparent)
            if label == 1:  # Positive — green
                outer = QColor(76, 175, 80, 80)
                inner = QColor(76, 175, 80)
                cross = QColor(255, 255, 255)
            else:  # Negative — red
                outer = QColor(244, 67, 54, 80)
                inner = QColor(244, 67, 54)
                cross = QColor(255, 255, 255)

            # Outer glow
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(outer))
            r = 14
            painter.drawEllipse(QPointF(x, y), r, r)

            # Inner dot
            painter.setBrush(QBrush(inner))
            painter.drawEllipse(QPointF(x, y), 6, 6)

            # Cross-hair
            pen_cross = QPen(cross, 1.5)
            pen_cross.setCosmetic(True)
            painter.setPen(pen_cross)
            painter.drawLine(QLineF(x - 4, y, x + 4, y))
            painter.drawLine(QLineF(x, y - 4, x, y + 4))


# ─── Image Viewer ─────────────────────────────────────────────────────────


class ImageViewer(QWidget):
    """Interactive image viewer with zoom, pan, point annotation, and overlay rendering."""

    # Signals — raw view intents. The session owns the canonical point list and
    # pushes it back via set_points(); the viewer holds points only to render.
    point_clicked = Signal(int, int, int)  # x, y, label
    undo_requested = Signal()
    clear_requested = Signal()

    # Point roles
    POSITIVE = 1
    NEGATIVE = 0

    # Annotation modes
    MODE_PAN = 0
    MODE_ADD_POSITIVE = 1
    MODE_ADD_NEGATIVE = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._transform: ViewTransform | None = None
        self._pixmap: QPixmap | None = None
        self._mask: np.ndarray | None = None
        self._points: list[dict] = []

        # Mode state
        self._annotation_mode: int = self.MODE_PAN

        # Hover preview
        self._hover_pos: tuple[int, int] | None = None  # image coords
        self._hover_item: QGraphicsEllipseItem | None = None

        self._setup_ui()
        self._setup_connections()

        # Keep scene overlay item reference
        self._overlay: _OverlayGroup | None = None
        self._pixmap_item: QGraphicsPixmapItem | None = None

        # Install event filter on the view's viewport for mouse tracking
        self.view.viewport().installEventFilter(self)
        self.view.viewport().setMouseTracking(True)

    # ── UI setup ───────────────────────────────────────────────────────────

    def _setup_ui(self):
        """Build the widget layout — tool strip at top, view fills the rest."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Embedded tool strip ──────────────────────────────────────
        self._toolstrip = QFrame()
        self._toolstrip.setObjectName("ViewerToolstrip")
        strip_layout = QHBoxLayout(self._toolstrip)
        strip_layout.setContentsMargins(6, 3, 6, 3)
        strip_layout.setSpacing(2)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)

        def _strip_btn(icon, text, tooltip, *, checkable=False):
            btn = QToolButton()
            btn.setObjectName("StripButton")
            btn.setIcon(icon)
            btn.setText(text)
            btn.setToolTip(tooltip)
            btn.setCheckable(checkable)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            return btn

        self._pan_btn = _strip_btn(
            icon_pan(), "Pan", "Pan and zoom — drag to move (P)", checkable=True
        )
        self._pan_btn.setChecked(True)
        self._mode_group.addButton(self._pan_btn, self.MODE_PAN)
        strip_layout.addWidget(self._pan_btn)

        self._pos_btn = _strip_btn(
            icon_add_positive(), "Foreground",
            "Mark foreground — click on the subject (F)", checkable=True
        )
        self._mode_group.addButton(self._pos_btn, self.MODE_ADD_POSITIVE)
        strip_layout.addWidget(self._pos_btn)

        self._neg_btn = _strip_btn(
            icon_add_negative(), "Background",
            "Mark background — click to exclude (B / Shift+click / right-click)", checkable=True
        )
        self._mode_group.addButton(self._neg_btn, self.MODE_ADD_NEGATIVE)
        strip_layout.addWidget(self._neg_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setObjectName("StripSeparator")
        strip_layout.addWidget(sep)

        self._undo_btn = _strip_btn(
            QIcon.fromTheme("edit-undo", icon_undo()), "Undo", "Undo last point (Ctrl+Z)"
        )
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self.undo_requested)
        strip_layout.addWidget(self._undo_btn)

        self._clear_btn = _strip_btn(
            QIcon.fromTheme("edit-clear", icon_clear()), "Clear",
            "Remove all points (Ctrl+R)"
        )
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self.clear_requested)
        strip_layout.addWidget(self._clear_btn)

        self._point_count_label = QLabel("")
        self._point_count_label.setObjectName("PointCountLabel")
        self._point_count_label.setVisible(False)
        strip_layout.addWidget(self._point_count_label)

        strip_layout.addStretch()
        self._mode_group.idClicked.connect(self._on_mode_selected)

        # Keyboard shortcuts for mode switching
        from PySide6.QtGui import QKeySequence, QShortcut
        for key, mode in (("P", self.MODE_PAN), ("F", self.MODE_ADD_POSITIVE), ("B", self.MODE_ADD_NEGATIVE)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda m=mode: self.set_mode(m))

        undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        undo_sc.activated.connect(self.undo_requested)

        layout.addWidget(self._toolstrip)

        # Disable tool strip until an image is loaded
        self._toolstrip.setEnabled(False)

        # ── Graphics view ────────────────────────────────────────────
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setAcceptDrops(False)
        self.view.viewport().setAcceptDrops(False)
        self.view.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.view.setMinimumSize(400, 300)
        self.view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.view, stretch=1)

        self._build_overlays()

    def _make_overlay_btn(
        self, icon: QIcon, tooltip: str, *, checkable: bool = False
    ) -> QToolButton:
        btn = QToolButton(self.view.viewport())
        btn.setIcon(icon)
        btn.setIconSize(_OVERLAY_ICON_SIZE)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        btn.setAutoRaise(False)
        btn.setCheckable(checkable)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setObjectName("ViewerOverlayButton")
        return btn

    def _build_overlays(self) -> None:
        """Create floating overlay widgets parented to the viewport."""
        vp = self.view.viewport()

        # Hint label (auto-fades after first image load)
        self._hint_label = QLabel(vp)
        self._hint_label.setObjectName("ViewerHint")
        self._hint_label.setText(
            "Click to mark foreground · Shift+click or right-click for background"
        )
        self._hint_label.setVisible(False)
        self._hint_fade_timer = QTimer(self)
        self._hint_fade_timer.setSingleShot(True)
        self._hint_fade_timer.timeout.connect(
            lambda: self._hint_label.setVisible(False)
        )

        # Bottom-right zoom overlay
        self._zoom_overlay = QFrame(vp)
        self._zoom_overlay.setObjectName("ViewerOverlay")
        zoom_layout = QHBoxLayout(self._zoom_overlay)
        zoom_layout.setContentsMargins(6, 4, 6, 4)
        zoom_layout.setSpacing(2)

        self._zoom_out_btn = self._make_overlay_btn(
            QIcon.fromTheme("zoom-out"), "Zoom out (Ctrl+−)"
        )
        if self._zoom_out_btn.icon().isNull():
            self._zoom_out_btn.setText("−")
            self._zoom_out_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        zoom_layout.addWidget(self._zoom_out_btn)

        self.zoom_label = QLabel("100%", self._zoom_overlay)
        self.zoom_label.setObjectName("ZoomLabel")
        self.zoom_label.setMinimumWidth(46)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_layout.addWidget(self.zoom_label)

        self._zoom_in_btn = self._make_overlay_btn(
            QIcon.fromTheme("zoom-in"), "Zoom in (Ctrl++)"
        )
        if self._zoom_in_btn.icon().isNull():
            self._zoom_in_btn.setText("+")
            self._zoom_in_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        zoom_layout.addWidget(self._zoom_in_btn)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setObjectName("OverlaySeparator")
        zoom_layout.addWidget(sep2)

        self._fit_btn = self._make_overlay_btn(
            QIcon.fromTheme("zoom-fit-best"), "Fit image to view"
        )
        if self._fit_btn.icon().isNull():
            self._fit_btn.setText("Fit")
            self._fit_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        zoom_layout.addWidget(self._fit_btn)

        # Busy overlay — translucent blocker shown during segmentation
        self._busy_overlay = QFrame(vp)
        self._busy_overlay.setObjectName("BusyOverlay")
        busy_layout = QVBoxLayout(self._busy_overlay)
        busy_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        busy_layout.setSpacing(10)
        busy_label = QLabel("Segmenting…")
        busy_label.setObjectName("BusyLabel")
        busy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        busy_bar = QProgressBar()
        busy_bar.setRange(0, 0)
        busy_bar.setFixedWidth(220)
        busy_bar.setTextVisible(False)
        busy_layout.addWidget(busy_label)
        busy_layout.addWidget(busy_bar, 0, Qt.AlignmentFlag.AlignCenter)
        self._busy_overlay.hide()

        self._zoom_overlay.adjustSize()
        self._hint_label.adjustSize()
        self._position_overlays()

    def _position_overlays(self) -> None:
        vp = self.view.viewport()
        margin = 8
        self._hint_label.adjustSize()
        self._hint_label.move(margin, margin)
        self._zoom_overlay.adjustSize()
        self._zoom_overlay.move(
            max(margin, vp.width() - self._zoom_overlay.width() - margin),
            max(margin, vp.height() - self._zoom_overlay.height() - margin),
        )
        self._busy_overlay.setGeometry(0, 0, vp.width(), vp.height())

    def _setup_connections(self):
        self._zoom_in_btn.clicked.connect(lambda: self._zoom_step(1.15))
        self._zoom_out_btn.clicked.connect(lambda: self._zoom_step(1 / 1.15))
        self._fit_btn.clicked.connect(self.fit_to_view)

    # ── Mode management ────────────────────────────────────────────────────

    @property
    def annotation_mode(self) -> int:
        """Return the current annotation mode (MODE_PAN, MODE_ADD_POSITIVE, or MODE_ADD_NEGATIVE)."""
        return self._annotation_mode

    def set_mode(self, mode: int) -> None:
        """Set the annotation mode and update drag mode, cursor, and toolbar."""
        if mode not in (self.MODE_PAN, self.MODE_ADD_POSITIVE, self.MODE_ADD_NEGATIVE):
            return
        self._annotation_mode = mode
        if mode == self.MODE_PAN:
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        btn = self._mode_group.button(mode)
        if btn and not btn.isChecked():
            btn.setChecked(True)
        self._update_cursor()
        self._clear_hover_preview()

    def _on_mode_selected(self, mode_id: int) -> None:
        self._annotation_mode = mode_id
        if mode_id == self.MODE_PAN:
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._update_cursor()
        self._clear_hover_preview()

    def _update_undo_clear_buttons(self) -> None:
        n = len(self._points)
        has_points = n > 0
        self._undo_btn.setEnabled(has_points)
        self._clear_btn.setEnabled(has_points)
        if has_points:
            self._point_count_label.setText(f"{n} point{'s' if n != 1 else ''}")
            self._point_count_label.setVisible(True)
        else:
            self._point_count_label.setVisible(False)

    def _update_cursor(self) -> None:
        """Update the cursor based on current mode."""
        if self._annotation_mode == self.MODE_PAN:
            self.view.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        else:
            self.view.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    # ── Point operations ───────────────────────────────────────────────────

    def set_points(self, points: list[dict]) -> None:
        """Render the given prompt points (image-pixel coords).

        Render-only: the session owns the canonical list and calls this after
        every change. The viewer never mutates points itself — clicks go out as
        :data:`point_clicked` and come back here.
        """
        self._points = [dict(p) for p in points]
        if self._overlay:
            self._overlay.set_points(self._points)
        self._update_undo_clear_buttons()

    # ── Hover preview ──────────────────────────────────────────────────────

    def _update_hover_preview(self, scene_pos: QPointF) -> None:
        """Show a ghost circle at the cursor position in Add modes."""
        self._clear_hover_preview()

        if self._annotation_mode == self.MODE_PAN or self._image is None:
            return

        coords = self.get_image_coordinates(scene_pos)
        if coords is None:
            return

        x, y = coords
        self._hover_pos = (x, y)

        # Determine colour based on active mode
        if self._annotation_mode == self.MODE_ADD_POSITIVE:
            color = QColor(76, 175, 80, 100)  # green, semi-transparent
        else:
            color = QColor(244, 67, 54, 100)  # red, semi-transparent

        # Create a temporary ellipse at the hover position
        item = QGraphicsEllipseItem(x - 8, y - 8, 16, 16)
        item.setPen(QPen(color, 1.5))
        item.setBrush(QBrush(color))
        item.setZValue(100)  # above the overlay
        self.scene.addItem(item)
        self._hover_item = item

    def _clear_hover_preview(self) -> None:
        """Remove the hover preview item from the scene."""
        if self._hover_item is not None:
            self.scene.removeItem(self._hover_item)
            self._hover_item = None
        self._hover_pos = None

    # ── Event filter for mouse tracking on viewport ────────────────────────

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        """Filter viewport events.

        Handles:
        - MouseMove → hover preview
        - MouseButtonPress → place point (modifier-aware: Shift+left or right = negative;
          middle = let view pan)
        - Wheel → zoom
        - Leave → clear hover preview
        - Resize → reposition floating overlays
        """
        if obj is self.view.viewport():
            etype = event.type()

            if etype == QEvent.Type.Resize:
                self._position_overlays()

            elif etype == QEvent.Type.MouseMove:
                scene_pos = self.view.mapToScene(event.position().toPoint())
                self._update_hover_preview(scene_pos)

            elif etype == QEvent.Type.MouseButtonPress:
                if self._image is None:
                    return super().eventFilter(obj, event)

                btn = event.button()
                mods = event.modifiers()

                # Negative-point shortcut: right-click OR Shift+left-click
                shift_left = (
                    btn == Qt.MouseButton.LeftButton
                    and bool(mods & Qt.KeyboardModifier.ShiftModifier)
                )
                if btn == Qt.MouseButton.RightButton or shift_left:
                    scene_pos = self.view.mapToScene(event.position().toPoint())
                    coords = self.get_image_coordinates(scene_pos)
                    if coords is not None:
                        x, y = coords
                        self.point_clicked.emit(x, y, self.NEGATIVE)
                        self._clear_hover_preview()
                        event.accept()
                        return True

                # Mode-driven left click
                if (
                    btn == Qt.MouseButton.LeftButton
                    and self._annotation_mode != self.MODE_PAN
                ):
                    scene_pos = self.view.mapToScene(event.position().toPoint())
                    coords = self.get_image_coordinates(scene_pos)
                    if coords is not None:
                        x, y = coords
                        label = (
                            self.POSITIVE
                            if self._annotation_mode == self.MODE_ADD_POSITIVE
                            else self.NEGATIVE
                        )
                        self.point_clicked.emit(x, y, label)
                        self._clear_hover_preview()
                        event.accept()
                        return True

            elif etype == QEvent.Type.Wheel:
                if event.angleDelta().y() > 0:
                    self._zoom_step(1.15)
                else:
                    self._zoom_step(1 / 1.15)
                event.accept()
                return True

            elif etype == QEvent.Type.Leave:
                self._clear_hover_preview()

        return super().eventFilter(obj, event)

    # ── Public image/mask API ──────────────────────────────────────────────

    def set_image(self, image: np.ndarray) -> None:
        """Set the image to display. Must be RGB (H,W,3) or RGBA (H,W,4)."""
        self._image = image
        self._transform = ViewTransform.from_image(image)
        self._mask = None
        self._points = []
        self._clear_hover_preview()
        self._render_image()
        self._toolstrip.setEnabled(True)
        self.set_mode(self.MODE_ADD_POSITIVE)
        self._update_undo_clear_buttons()

        # Show input hint, fade after a few seconds
        self._hint_label.setVisible(True)
        self._position_overlays()
        self._hint_fade_timer.start(6000)

    def set_busy(self, busy: bool) -> None:
        """Show or hide the segmentation busy overlay."""
        self._busy_overlay.setVisible(busy)
        if busy:
            self._busy_overlay.raise_()
            self._position_overlays()

    def set_mask(self, mask: np.ndarray) -> None:
        """Set the segmentation mask and update the overlay."""
        self._mask = mask
        if self._overlay:
            self._overlay.set_mask(mask)

    def fit_to_view(self) -> None:
        """Zoom to fit the image in the view."""
        if self._pixmap_item:
            self.view.fitInView(
                self._pixmap_item.sceneBoundingRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            self._update_zoom_label()

    def get_image_coordinates(
        self, scene_pos: QPointF
    ) -> tuple[int, int] | None:
        """Convert scene position to image pixel coordinates."""
        if self._transform is None:
            return None
        return self._transform.scene_to_pixel(scene_pos.x(), scene_pos.y())

    # ── Internal rendering ────────────────────────────────────────────────

    def _render_image(self) -> None:
        """Convert numpy image to pixmap and display with overlay."""
        if self._image is None:
            return

        self.scene.clear()
        self._hover_item = None  # cleared by scene.clear()
        self._pixmap_item = None
        self._overlay = None

        height, width = self._image.shape[:2]

        # Convert numpy to QImage
        if self._image.shape[2] == 4:
            qimg = QImage(
                self._image.data,
                width,
                height,
                self._image.strides[0],
                QImage.Format.Format_RGBA8888,
            )
        else:
            qimg = QImage(
                self._image.data,
                width,
                height,
                self._image.strides[0],
                QImage.Format.Format_RGB888,
            )

        self._pixmap = QPixmap.fromImage(qimg)
        self._pixmap_item = self.scene.addPixmap(self._pixmap)
        self.scene.setSceneRect(QRectF(self._pixmap.rect()))

        # Create overlay group attached to the pixmap item
        self._overlay = _OverlayGroup(self._pixmap_item)
        self._overlay.set_points(self._points)
        if self._mask is not None:
            self._overlay.set_mask(self._mask)

        self.fit_to_view()

    def _zoom_step(self, factor: float) -> None:
        self.view.scale(factor, factor)
        self._update_zoom_label()

    def _update_zoom_label(self) -> None:
        zoom_pct = int(self.view.transform().m11() * 100)
        self.zoom_label.setText(f"{zoom_pct}%")

    # ── Mouse events (fallback — actual handling is in eventFilter) ─────────

    def wheelEvent(self, event):
        """Fallback: let the QGraphicsView handle wheel events natively."""
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Fallback: let the QGraphicsView handle mouse press natively."""
        super().mousePressEvent(event)
