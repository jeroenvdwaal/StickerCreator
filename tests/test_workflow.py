"""Integration tests for StickerWorkflow — the session ↔ segmenter loop.

These drive the whole click → segment → mask → sticker path with a real
``Segmenter`` facade (background thread) backed by a ``FakeEngine``, but no
``QMainWindow``. They are the regression net for the wiring that used to live
as relay slots on the window: if the workflow stops dispatching, or stops
feeding masks back into the session, the sticker never appears and these fail.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_qt

from sticker_creator.segmentation.segmenter import Segmenter
from sticker_creator.session import StickerSession
from sticker_creator.workflow import StickerWorkflow

from tests.test_segmenter_engine import FakeEngine, fake_settings  # noqa: F401


@pytest.fixture
def image_and_mask():
    image = np.full((300, 300, 3), 64, dtype=np.uint8)
    image[50:250, 60:240] = [200, 30, 30]
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[50:250, 60:240] = 1
    return image, mask


def test_click_runs_segmenter_and_produces_sticker(
    qtbot, model_dir, fake_settings, image_and_mask  # noqa: F811
):
    image, mask = image_and_mask
    session = StickerSession()
    segmenter = Segmenter(engine=FakeEngine(model_dir, mask=mask))
    workflow = StickerWorkflow(session, segmenter)
    assert workflow is not None  # keep a reference: a parentless QObject is GC'd
    try:
        with qtbot.waitSignal(segmenter.model_loaded, timeout=2000):
            segmenter.load_model("sam2.1_hiera_tiny")

        session.set_image(image)
        # A click should travel: session.segment_requested → workflow →
        # segmenter.segment (thread) → mask_ready → session.set_mask → sticker.
        with qtbot.waitSignal(session.sticker_changed, timeout=2000):
            session.add_prompt(150, 150, session.POSITIVE)

        assert session.has_sticker is True
    finally:
        segmenter.shutdown()


def test_click_without_model_signals_model_required(
    qtbot, model_dir, fake_settings, image_and_mask  # noqa: F811
):
    image, _ = image_and_mask
    session = StickerSession()
    segmenter = Segmenter(engine=FakeEngine(model_dir))  # never loaded
    workflow = StickerWorkflow(session, segmenter)
    try:
        prompted = []
        workflow.model_required.connect(lambda: prompted.append(True))
        segment_calls = []
        segmenter._segment_requested.connect(lambda *a: segment_calls.append(a))

        session.set_image(image)
        session.add_prompt(150, 150, session.POSITIVE)

        assert prompted == [True]      # the no-model prompt fired
        assert segment_calls == []     # nothing dispatched to the segmenter
        assert session.has_sticker is False
    finally:
        segmenter.shutdown()
