"""The segmentation engine seam.

:class:`sticker_creator.segmentation.segmenter.SegmenterWorker` does not depend
on SAM 2 directly — it depends on this *interface*: load a model by name, turn
an image plus point prompts into a mask, and expose a :class:`ModelRegistry` so
the facade can list/delete checkpoints.  Inference in / mask out, pure numpy.

Two adapters implement it, which is what makes it a real seam rather than a
hypothetical one:

* :class:`sticker_creator.segmentation.sam_segmenter.SamSegmenter` — production.
  Holds the torch model, the compatibility patches, and the embedding cache.
* a fake (see ``tests/``) that returns a canned mask, so the whole
  load → segment → sticker workflow can be exercised without a real checkpoint
  or a torch import.

The protocol is :func:`runtime_checkable`, so an adapter can be asserted to
conform with ``isinstance(obj, SegmentationEngine)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from sticker_creator.segmentation.model_registry import ModelRegistry


@runtime_checkable
class SegmentationEngine(Protocol):
    """Headless segmentation contract the Qt worker drives on a thread."""

    @property
    def registry(self) -> ModelRegistry:
        """Catalogue of checkpoints on disk (for listing / deletion)."""
        ...

    @property
    def is_loaded(self) -> bool:
        """True once a model is loaded and ready to :meth:`segment`."""
        ...

    @property
    def current_model_name(self) -> str | None:
        """Name of the loaded model, or ``None`` if none loaded."""
        ...

    def load_model(self, model_name: str | None = None) -> str:
        """Load *model_name* (or a default) and return its resolved name."""
        ...

    def segment(
        self, image: np.ndarray, points: list[dict]
    ) -> np.ndarray | None:
        """Point-prompted segmentation → ``H×W`` uint8 mask (or ``None``)."""
        ...
