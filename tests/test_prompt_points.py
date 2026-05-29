"""Tests for the ImageViewer prompt-points seam.

These lock in the public interface MainWindow uses to read segmentation
prompts, so it no longer reaches into the viewer's private ``_points`` list.
"""

import pytest

pytestmark = pytest.mark.requires_qt

from sticker_creator.widgets.image_viewer import ImageViewer


@pytest.fixture
def viewer(qtbot):
    w = ImageViewer()
    qtbot.addWidget(w)
    return w


class TestPromptPoints:
    def test_empty_by_default(self, viewer):
        assert viewer.prompt_points() == []
        assert viewer.has_prompt_points() is False

    def test_reports_added_points(self, viewer):
        viewer.add_point(10, 20, ImageViewer.POSITIVE)
        viewer.add_point(30, 40, ImageViewer.NEGATIVE)
        pts = viewer.prompt_points()
        assert viewer.has_prompt_points() is True
        assert pts == [
            {"x": 10, "y": 20, "label": ImageViewer.POSITIVE},
            {"x": 30, "y": 40, "label": ImageViewer.NEGATIVE},
        ]

    def test_returns_a_copy(self, viewer):
        viewer.add_point(1, 2, ImageViewer.POSITIVE)
        pts = viewer.prompt_points()
        pts[0]["x"] = 999
        pts.append({"x": 0, "y": 0, "label": 0})
        # Mutating the returned list/dicts must not affect the viewer.
        assert viewer.prompt_points() == [
            {"x": 1, "y": 2, "label": ImageViewer.POSITIVE}
        ]
