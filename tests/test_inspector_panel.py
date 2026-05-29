"""Tests for InspectorPanel — control state → StickerOptions mapping.

The inspector packs all five composition controls into a single
``StickerOptions`` bundle and emits it via ``options_changed``. These pin that
mapping (and that user interaction actually emits) so the consolidation refactor
can't silently send the session the wrong bundle.
"""

import pytest

pytestmark = pytest.mark.requires_qt

from sticker_creator.utils.balloon_renderer import STYLE_AUTO, STYLE_SHOUT
from sticker_creator.utils.sticker_border import BACKGROUND_TRANSPARENT
from sticker_creator.utils.sticker_pipeline import StickerOptions
from sticker_creator.widgets.inspector_panel import InspectorPanel


@pytest.fixture
def inspector(qapp):
    return InspectorPanel()


def _check_button(group, prop, value):
    for btn in group.buttons():
        if btn.property(prop) == value:
            btn.setChecked(True)
            return
    raise AssertionError(f"no button with {prop}={value!r}")


def test_defaults_match_stickeroptions_defaults(inspector):
    opts = inspector._current_options()
    assert opts == StickerOptions(
        border_enabled=True,
        border_width=7,
        balloon_text="",
        balloon_style=STYLE_AUTO,
        background=BACKGROUND_TRANSPARENT,
    )


def test_current_options_snapshots_every_control(inspector):
    inspector._border_check.setChecked(False)
    inspector._width_slider.setValue(15)
    inspector._balloon_input.setText("boom")
    _check_button(inspector._style_group, "style_value", STYLE_SHOUT)
    _check_button(inspector._bg_group, "bg_value", "black")

    opts = inspector._current_options()
    assert opts.border_enabled is False
    assert opts.border_width == 15
    assert opts.balloon_text == "boom"
    assert opts.balloon_style == STYLE_SHOUT
    assert opts.background == "black"


def test_width_change_emits_options(inspector, qtbot):
    with qtbot.waitSignal(inspector.options_changed, timeout=1000) as blocker:
        inspector._width_slider.setValue(12)
    assert blocker.args[0].border_width == 12


def test_border_toggle_emits_and_disables_width(inspector, qtbot):
    with qtbot.waitSignal(inspector.options_changed, timeout=1000) as blocker:
        inspector._border_check.setChecked(False)
    assert blocker.args[0].border_enabled is False
    assert inspector._width_slider.isEnabled() is False


def test_set_border_width_does_not_emit(inspector, qtbot):
    """Programmatic width sync (auto-derived) must not loop back as a change."""
    with qtbot.assertNotEmitted(inspector.options_changed):
        inspector.set_border_width(9)
    assert inspector._width_slider.value() == 9
