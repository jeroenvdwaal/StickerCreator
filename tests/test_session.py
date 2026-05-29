"""Tests for StickerSession — the workflow state machine.

The mask → raw → border path is Qt-free; these run without a QMainWindow. The
``qapp`` fixture is present because QObject signals are used, and the balloon
compose step needs a QGuiApplication.
"""

import numpy as np
import pytest

from sticker_creator.session import StickerSession
from sticker_creator.utils.sticker_border import BACKGROUND_TRANSPARENT


@pytest.fixture
def image_and_mask():
    image = np.full((300, 300, 3), 64, dtype=np.uint8)
    image[50:250, 60:240] = [200, 30, 30]
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[50:250, 60:240] = 1
    return image, mask


@pytest.fixture
def session(qapp):
    return StickerSession()


def _collect(signal):
    """Connect a recorder to *signal*; return the list it appends emissions to."""
    seen = []
    signal.connect(lambda v=None: seen.append(v))
    return seen


class TestImageLifecycle:
    def test_set_image_emits_and_clears(self, session, image_and_mask):
        image, _ = image_and_mask
        imgs = _collect(session.image_changed)
        stickers = _collect(session.sticker_changed)
        session.set_image(image, path="/tmp/a.png")
        assert session.current_image is image
        assert session.image_path == "/tmp/a.png"
        assert imgs[-1] is image
        assert stickers[-1] is None        # derived state cleared
        assert session.current_sticker is None
        assert session.has_sticker is False

    def test_new_image_invalidates_prior_sticker(self, session, image_and_mask):
        image, mask = image_and_mask
        session.set_image(image)
        session.set_mask(mask)
        assert session.has_sticker is True
        session.set_image(np.full((10, 10, 3), 1, dtype=np.uint8))
        assert session.current_sticker is None


class TestMask:
    def test_set_mask_derives_sticker_and_border(self, session, image_and_mask):
        image, mask = image_and_mask
        widths = _collect(session.border_width_changed)
        masks = _collect(session.mask_changed)
        stickers = _collect(session.sticker_changed)
        session.set_image(image)
        session.set_mask(mask)
        assert session.has_sticker is True
        assert masks[-1] is mask
        assert widths and widths[-1] >= 1          # auto border width emitted
        assert stickers[-1] is not None            # preview emitted
        assert stickers[-1].shape[2] == 4

    def test_set_mask_without_image_is_ignored(self, session, image_and_mask):
        _, mask = image_and_mask
        session.set_mask(mask)
        assert session.has_sticker is False

    def test_clear_derived_resets(self, session, image_and_mask):
        image, mask = image_and_mask
        session.set_image(image)
        session.set_mask(mask)
        masks = _collect(session.mask_changed)
        stickers = _collect(session.sticker_changed)
        session.clear_derived()
        assert session.current_sticker is None
        assert masks[-1] is None
        assert stickers[-1] is None
        # Image survives a derived-state clear.
        assert session.current_image is image


class TestOptions:
    def test_border_width_recomposes_only_when_enabled(self, session, image_and_mask):
        image, mask = image_and_mask
        session.set_image(image)
        session.set_mask(mask)
        session.set_border_enabled(False)
        stickers = _collect(session.sticker_changed)
        session.set_border_width(20)
        assert stickers == []                       # disabled → no recompose
        session.set_border_enabled(True)
        assert stickers[-1] is not None

    def test_background_does_not_change_saved_sticker(self, session, image_and_mask):
        image, mask = image_and_mask
        session.set_image(image)
        session.set_mask(mask)
        saved_before = session.current_sticker.copy()
        session.set_background("white")
        # Preview background is display-only; the saved sticker is unchanged.
        assert np.array_equal(session.current_sticker, saved_before)

    def test_options_before_mask_do_not_emit_sticker(self, session):
        stickers = _collect(session.sticker_changed)
        session.set_balloon_text("hi")
        session.set_background("black")
        assert all(s is None for s in stickers)
