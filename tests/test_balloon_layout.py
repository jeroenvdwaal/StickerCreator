"""Tests for balloon layout planning.

``plan_balloon`` returns pure geometry (a ``BalloonLayout``) with no drawing,
so placement decisions — side selection, canvas growth, wrapping, seating the
body against the subject — are assertable on plain data without a draw surface
or a Qt event loop.
"""

import numpy as np
import pytest

from sticker_creator.utils.balloon_renderer import (
    SIDE_LEFT,
    SIDE_RIGHT,
    STYLE_SHOUT,
    STYLE_THOUGHT,
    STYLE_WHISPER,
    BalloonLayout,
    detect_style,
    plan_balloon,
    render_balloon,
)


def _sticker(w=300, h=300, subj=(60, 240, 50, 250)) -> np.ndarray:
    """RGBA canvas with one opaque rectangle subject at (left,right,top,bottom)."""
    left, right, top, bottom = subj
    a = np.zeros((h, w, 4), dtype=np.uint8)
    a[top:bottom, left:right] = (200, 30, 30, 255)
    return a


class TestPlanBalloon:
    def test_blank_text_returns_none(self):
        assert plan_balloon(_sticker(), "   ") is None

    def test_returns_layout_for_text(self):
        layout = plan_balloon(_sticker(), "Hello!")
        assert isinstance(layout, BalloonLayout)

    def test_style_resolved_from_text(self):
        # detect_style is the source of truth; layout must carry the resolved style.
        layout = plan_balloon(_sticker(), "psst (quiet)")
        assert layout.style == detect_style("psst (quiet)")

    def test_explicit_style_overrides_detection(self):
        layout = plan_balloon(_sticker(), "Hello", style=STYLE_THOUGHT)
        assert layout.style == STYLE_THOUGHT

    def test_side_picks_emptier_margin(self):
        # Subject hugs the left edge → more room on the right → balloon goes right.
        right_room = _sticker(subj=(10, 120, 50, 250))
        assert plan_balloon(right_room, "Hi").side == SIDE_RIGHT
        # Subject hugs the right edge → balloon goes left.
        left_room = _sticker(subj=(180, 290, 50, 250))
        assert plan_balloon(left_room, "Hi").side == SIDE_LEFT

    def test_balloon_seated_beside_subject(self):
        # Subject on the left; balloon on the right must start at/after its edge.
        s = _sticker(subj=(10, 120, 50, 250))
        layout = plan_balloon(s, "Hi")
        bx = layout.balloon_rect[0] - layout.sticker_origin[0]
        # Body left edge sits near the subject's right edge (within the overlap).
        assert bx >= 120 - 20

    def test_canvas_grows_when_subject_fills_frame(self):
        # Subject spans the full width → no side room → canvas must expand.
        full = _sticker(subj=(0, 300, 0, 300))
        layout = plan_balloon(full, "Hello there friend")
        new_w, _ = layout.canvas_size
        assert new_w > 300

    def test_long_text_wraps_to_multiple_lines(self):
        layout = plan_balloon(_sticker(), "one two three four five six seven eight")
        assert len(layout.lines) >= 2

    def test_shout_reserves_overflow_above(self):
        # Starburst bulges past its rect → top expands so the canvas leaves room.
        layout = plan_balloon(_sticker(), "WOW", style=STYLE_SHOUT)
        assert layout.sticker_origin[1] > 0  # sy > 0 means top grew

    @pytest.mark.parametrize("style", [STYLE_WHISPER, STYLE_THOUGHT, STYLE_SHOUT])
    def test_font_size_positive(self, style):
        layout = plan_balloon(_sticker(), "text", style=style)
        assert layout.font_size > 0


class TestRenderMatchesPlan:
    def test_render_uses_plan_canvas_size(self):
        s = _sticker()
        layout = plan_balloon(s, "Hello there")
        out = render_balloon(s, "Hello there")
        new_w, new_h = layout.canvas_size
        assert out.shape[:2] == (new_h, new_w)

    def test_blank_text_returns_input_unchanged(self):
        s = _sticker()
        out = render_balloon(s, "  ")
        assert out is s
