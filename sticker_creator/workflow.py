"""The sticker workflow — routing between the session and the segmenter.

:class:`StickerSession` owns the document state and emits
:data:`~sticker_creator.session.StickerSession.segment_requested` when a click
should re-run segmentation; the :class:`~sticker_creator.segmentation.segmenter.Segmenter`
runs SAM 2 off-thread and emits ``mask_ready`` when done. Neither knows about
the other. This module is the one place that joins them:

* a segmentation request is dispatched to the segmenter only when a model is
  loaded — otherwise :data:`model_required` fires so the UI can prompt;
* a mask coming back from the segmenter is fed into the session, which
  re-derives the sticker.

That routing used to live as two relay slots on ``MainWindow``, which made the
loop only reachable through a ``QMainWindow``. Here it is a small ``QObject``
seam: a test constructs a session + a ``Segmenter(engine=FakeEngine)`` and
drives a click end-to-end with no window and no event loop of its own.

The window keeps the genuinely UI-local concerns it always owned — the busy
spinner, the "segmented in Xs" status, model management — and listens to
:data:`model_required` for the no-model prompt.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from sticker_creator.segmentation.segmenter import Segmenter
from sticker_creator.session import StickerSession


class StickerWorkflow(QObject):
    """Joins :class:`StickerSession` to :class:`Segmenter`.

    Signals:
        model_required(): a segmentation was requested while no model is
            loaded; the UI should prompt the user to install/select one.
    """

    model_required = Signal()

    def __init__(
        self,
        session: StickerSession,
        segmenter: Segmenter,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._segmenter = segmenter

        # session click → segment ; segmenter result → session re-derive.
        session.segment_requested.connect(self._dispatch_segment)
        segmenter.mask_ready.connect(session.set_mask)

    @Slot(object, object)
    def _dispatch_segment(self, image: np.ndarray, points: list[dict]) -> None:
        """Run the segmenter for this request, or signal that a model is missing."""
        if self._segmenter.active_model_name is None:
            self.model_required.emit()
            return
        self._segmenter.segment(image=image, points=points)
