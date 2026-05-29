"""Single source of truth for SAM 2 model checkpoints.

The registry owns every fact about a checkpoint that is *not* inference:
the known variants, their download URLs and human-readable sizes, the Hydra
config each maps to, where the ``.pt`` file lives on disk, and what counts as
a valid (non-corrupted) checkpoint.

It depends on ``pathlib`` only — no torch, no PySide6 — so the segmenter, the
downloader, the model-manager UI, and the test-suite can all share the same
validity logic instead of each re-deriving it. Active-model *persistence*
(which model is selected) is a separate concern and lives with the segmenter,
which has QSettings; the registry only knows what is on disk.
"""

from __future__ import annotations

from pathlib import Path

# All known SAM 2 model variants, smallest first.
KNOWN_MODELS: list[str] = [
    "sam2.1_hiera_tiny",
    "sam2.1_hiera_small",
    "sam2.1_hiera_base_plus",
    "sam2.1_hiera_large",
]

# 10 MB — any real SAM 2 checkpoint is far larger; smaller files are
# placeholders or truncated downloads and must be treated as absent.
MIN_CHECKPOINT_BYTES = 10 * 1024 * 1024

# Meta's official SAM 2.1 release server.
_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"

MODEL_URLS: dict[str, str] = {
    "sam2.1_hiera_tiny":      _BASE_URL + "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small":     _BASE_URL + "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus": _BASE_URL + "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large":     _BASE_URL + "sam2.1_hiera_large.pt",
}

MODEL_SIZES: dict[str, str] = {
    "sam2.1_hiera_tiny":      "~148 MB",
    "sam2.1_hiera_small":     "~175 MB",
    "sam2.1_hiera_base_plus": "~308 MB",
    "sam2.1_hiera_large":     "~856 MB",
}

# Map from our checkpoint stem names to the Hydra config paths build_sam2 expects.
CONFIG_MAP: dict[str, str] = {
    "sam2.1_hiera_tiny":      "configs/sam2.1/sam2.1_hiera_t",
    "sam2.1_hiera_small":     "configs/sam2.1/sam2.1_hiera_s",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+",
    "sam2.1_hiera_large":     "configs/sam2.1/sam2.1_hiera_l",
    # Legacy SAM 2.0 names
    "sam2_hiera_tiny":        "sam2_hiera_t",
    "sam2_hiera_small":       "sam2_hiera_s",
    "sam2_hiera_base_plus":   "sam2_hiera_b+",
    "sam2_hiera_large":       "sam2_hiera_l",
}


def config_name(model_name: str) -> str:
    """Hydra config path for a checkpoint stem (falls back to the name itself)."""
    return CONFIG_MAP.get(model_name, model_name)


class ModelRegistry:
    """Filesystem-backed catalogue of checkpoints under a model directory."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

    # ── Paths ────────────────────────────────────────────────────────────────

    def checkpoint_path(self, model_name: str) -> Path:
        """Expected ``.pt`` path for *model_name* (no existence check)."""
        return self.model_dir / f"{model_name}.pt"

    def _legacy_path(self, model_name: str) -> Path:
        """Path under the legacy ``sam2_`` naming for *model_name*."""
        return self.model_dir / f"{model_name.replace('sam2.1_', 'sam2_')}.pt"

    # ── Validity ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_valid_file(path: Path) -> bool:
        return path.exists() and path.stat().st_size >= MIN_CHECKPOINT_BYTES

    def is_downloaded(self, model_name: str) -> bool:
        """True when a valid checkpoint for *model_name* exists on disk."""
        return self._is_valid_file(self.checkpoint_path(model_name))

    def find_loadable(self, model_name: str) -> Path | None:
        """Return a valid checkpoint path for *model_name*, or ``None``.

        Tries the canonical ``sam2.1_*`` name first, then the legacy
        ``sam2_*`` name. Only files that pass the :data:`MIN_CHECKPOINT_BYTES`
        size guard are returned, so corrupted/truncated files are skipped.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for candidate in (self.checkpoint_path(model_name), self._legacy_path(model_name)):
            if self._is_valid_file(candidate):
                return candidate
        return None

    def list_downloaded(self) -> list[str]:
        """Stems of every ``.pt`` file present in the model directory."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        for fpath in sorted(self.model_dir.glob("*.pt")):
            if fpath.stem not in names:
                names.append(fpath.stem)
        return names
