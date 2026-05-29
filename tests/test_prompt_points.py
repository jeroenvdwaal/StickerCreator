"""Tests for the ImageViewer prompt-points seam.

The viewer no longer owns prompt points — the session does. The viewer is a
dumb view: it emits raw click/undo/clear intents and renders whatever point
list the session pushes back via ``set_points``. These lock that contract in.
"""

import pytest

pytestmark = pytest.mark.requires_qt

from sticker_creator.widgets.image_viewer import ImageViewer


@pytest.fixture
def viewer(qtbot, rgb_image):
    w = ImageViewer()
    qtbot.addWidget(w)
    w.set_image(rgb_image)  # enables the toolstrip; clears any points
    return w


class TestRendersPushedPoints:
    def test_buttons_disabled_when_empty(self, viewer):
        viewer.set_points([])
        assert viewer._undo_btn.isEnabled() is False
        assert viewer._clear_btn.isEnabled() is False
        assert viewer._point_count_label.isVisible() is False

    def test_set_points_enables_undo_clear_and_count(self, viewer):
        viewer.set_points([
            {"x": 10, "y": 20, "label": ImageViewer.POSITIVE},
            {"x": 30, "y": 40, "label": ImageViewer.NEGATIVE},
        ])
        assert viewer._undo_btn.isEnabled() is True
        assert viewer._clear_btn.isEnabled() is True
        assert "2" in viewer._point_count_label.text()

    def test_set_points_copies_input(self, viewer):
        points = [{"x": 1, "y": 2, "label": ImageViewer.POSITIVE}]
        viewer.set_points(points)
        points[0]["x"] = 999
        points.append({"x": 0, "y": 0, "label": 0})
        # Mutating the caller's list must not affect what the viewer rendered.
        assert viewer._points == [{"x": 1, "y": 2, "label": ImageViewer.POSITIVE}]


class TestEmitsIntents:
    def test_undo_button_emits_undo_requested(self, viewer, qtbot):
        viewer.set_points([{"x": 1, "y": 2, "label": ImageViewer.POSITIVE}])
        with qtbot.waitSignal(viewer.undo_requested, timeout=500):
            viewer._undo_btn.click()

    def test_clear_button_emits_clear_requested(self, viewer, qtbot):
        viewer.set_points([{"x": 1, "y": 2, "label": ImageViewer.POSITIVE}])
        with qtbot.waitSignal(viewer.clear_requested, timeout=500):
            viewer._clear_btn.click()
