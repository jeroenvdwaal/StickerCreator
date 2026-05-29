"""Tests for the segmentation engine seam.

These exercise the Qt worker and the public :class:`Segmenter` facade against a
``FakeEngine`` — the whole load → segment workflow with no torch import and no
real checkpoint. The production :class:`SamSegmenter` is checked only for
protocol conformance here; its inference lives in ``sam_segmenter.py``.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_qt

from sticker_creator.segmentation.engine import SegmentationEngine
from sticker_creator.segmentation.model_registry import (
    MIN_CHECKPOINT_BYTES,
    ModelRegistry,
)
from sticker_creator.segmentation.sam_segmenter import SamSegmenter
from sticker_creator.segmentation.segmenter import Segmenter, SegmenterWorker


class FakeEngine:
    """In-memory :class:`SegmentationEngine` that returns a canned mask."""

    def __init__(self, model_dir, mask=None):
        self.registry = ModelRegistry(model_dir)
        self._mask = mask
        self.current_model_name = None
        self._is_loaded = False
        self.load_calls: list[str | None] = []
        self.segment_calls: list[tuple] = []

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self, model_name=None) -> str:
        self.load_calls.append(model_name)
        name = model_name or "fake-model"
        self.current_model_name = name
        self._is_loaded = True
        return name

    def segment(self, image, points):
        self.segment_calls.append((image, list(points)))
        if len(points) == 0:
            return None
        return self._mask


@pytest.fixture
def canned_mask() -> np.ndarray:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    return mask


@pytest.fixture
def fake_settings(monkeypatch):
    """Back the active-model accessors with a dict (no real QSettings)."""
    store: dict[str, str] = {}
    import sticker_creator.segmentation.segmenter as seg_mod

    monkeypatch.setattr(
        seg_mod.app_settings, "get_active_model",
        lambda: store.get("active_model") or None,
    )
    monkeypatch.setattr(
        seg_mod.app_settings, "set_active_model",
        lambda name: store.__setitem__("active_model", name),
    )
    monkeypatch.setattr(
        seg_mod.app_settings, "clear_active_model",
        lambda: store.pop("active_model", None),
    )
    return store


def _write_valid_checkpoint(path):
    with open(path, "wb") as f:
        f.truncate(MIN_CHECKPOINT_BYTES + 1)
    return path


class TestProtocolConformance:
    def test_sam_segmenter_is_an_engine(self, model_dir):
        assert isinstance(SamSegmenter(model_dir), SegmentationEngine)

    def test_fake_is_an_engine(self, model_dir):
        assert isinstance(FakeEngine(model_dir), SegmentationEngine)


class TestSegmenterWorker:
    """Drive the worker's slots directly (synchronous, no thread)."""

    def test_load_emits_model_loaded_and_persists(
        self, qtbot, model_dir, fake_settings
    ):
        engine = FakeEngine(model_dir)
        worker = SegmenterWorker(engine=engine)

        with qtbot.waitSignal(worker.model_loaded, timeout=1000) as blocker:
            worker.load_model("sam2.1_hiera_tiny")

        assert blocker.args == ["sam2.1_hiera_tiny"]
        assert engine.load_calls == ["sam2.1_hiera_tiny"]
        assert fake_settings["active_model"] == "sam2.1_hiera_tiny"

    def test_load_none_resolves_persisted_model(
        self, qtbot, model_dir, fake_settings
    ):
        fake_settings["active_model"] = "sam2.1_hiera_small"
        engine = FakeEngine(model_dir)
        worker = SegmenterWorker(engine=engine)

        with qtbot.waitSignal(worker.model_loaded, timeout=1000) as blocker:
            worker.load_model(None)

        assert blocker.args == ["sam2.1_hiera_small"]

    def test_load_error_emits_model_error(self, qtbot, model_dir, fake_settings):
        engine = FakeEngine(model_dir)

        def boom(model_name=None):
            raise RuntimeError("no checkpoint")

        engine.load_model = boom
        worker = SegmenterWorker(engine=engine)

        with qtbot.waitSignal(worker.model_error, timeout=1000) as blocker:
            worker.load_model("x")
        assert blocker.args == ["no checkpoint"]

    def test_segment_emits_mask_and_lifecycle(
        self, qtbot, model_dir, fake_settings, rgb_image, canned_mask
    ):
        engine = FakeEngine(model_dir, mask=canned_mask)
        worker = SegmenterWorker(engine=engine)
        worker.load_model("sam2.1_hiera_tiny")

        received: list[np.ndarray] = []
        worker.mask_ready.connect(received.append)

        points = [{"x": 30, "y": 30, "label": 1}]
        with qtbot.waitSignals(
            [worker.processing_started, worker.mask_ready,
             worker.processing_finished],
            timeout=1000, order="strict",
        ):
            worker.segment(rgb_image, points)

        np.testing.assert_array_equal(received[0], canned_mask)
        assert engine.segment_calls[0][1] == points

    def test_segment_without_model_errors(
        self, qtbot, model_dir, fake_settings, rgb_image
    ):
        worker = SegmenterWorker(engine=FakeEngine(model_dir))
        with qtbot.waitSignal(worker.model_error, timeout=1000) as blocker:
            worker.segment(rgb_image, [{"x": 1, "y": 1, "label": 1}])
        assert blocker.args == ["Model not loaded yet"]

    def test_segment_empty_points_is_noop(
        self, qtbot, model_dir, fake_settings, rgb_image
    ):
        engine = FakeEngine(model_dir)
        engine.load_model("m")
        worker = SegmenterWorker(engine=engine)
        # No signal should fire; just assert the call returns without raising
        # and the engine was never asked to segment.
        worker.segment(rgb_image, [])
        assert engine.segment_calls == []


class TestSegmenterFacade:
    """End-to-end through the public facade, on a real background thread."""

    def test_load_then_segment_over_thread(
        self, qtbot, model_dir, fake_settings, rgb_image, canned_mask
    ):
        engine = FakeEngine(model_dir, mask=canned_mask)
        seg = Segmenter(engine=engine)
        try:
            with qtbot.waitSignal(seg.model_loaded, timeout=2000) as loaded:
                seg.load_model("sam2.1_hiera_tiny")
            assert loaded.args == ["sam2.1_hiera_tiny"]

            with qtbot.waitSignal(seg.mask_ready, timeout=2000) as ready:
                seg.segment(rgb_image, [{"x": 30, "y": 30, "label": 1}])
            np.testing.assert_array_equal(ready.args[0], canned_mask)
        finally:
            seg.shutdown()

    def test_available_models_reflects_registry(
        self, model_dir, fake_settings
    ):
        _write_valid_checkpoint(model_dir / "sam2.1_hiera_tiny.pt")
        fake_settings["active_model"] = "sam2.1_hiera_tiny"
        seg = Segmenter(engine=FakeEngine(model_dir))
        try:
            by_name = {m["name"]: m for m in seg.available_models()}
            assert by_name["sam2.1_hiera_tiny"]["is_downloaded"]
            assert by_name["sam2.1_hiera_tiny"]["is_active"]
            assert not by_name["sam2.1_hiera_large"]["is_downloaded"]
        finally:
            seg.shutdown()

    def test_delete_model_unlinks_and_clears_active(
        self, model_dir, fake_settings
    ):
        _write_valid_checkpoint(model_dir / "sam2.1_hiera_tiny.pt")
        fake_settings["active_model"] = "sam2.1_hiera_tiny"
        seg = Segmenter(engine=FakeEngine(model_dir))
        try:
            assert seg.delete_model("sam2.1_hiera_tiny") is True
            assert not (model_dir / "sam2.1_hiera_tiny.pt").exists()
            assert "active_model" not in fake_settings
            assert seg.delete_model("sam2.1_hiera_large") is False
        finally:
            seg.shutdown()
