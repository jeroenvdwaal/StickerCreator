# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PySide6 desktop app for KDE that extracts stickers from images using Meta's SAM 2 segmentation model. Output is a transparent, optionally white-bordered, 512x512 sticker suitable for WhatsApp/Telegram.

## Common commands

```bash
# Run app (dev)
python3 main.py

# Install entry point (then run `sticker-creator`)
pip install -e .

# Download default SAM 2 checkpoint into models/
python3 models/download_model.py
python3 models/download_model.py --model sam2.1_hiera_small

# Tests (pytest, fixtures in tests/conftest.py)
pytest
pytest tests/test_segmenter.py
pytest tests/test_file_io.py::TestImageLoader::test_load_png
```

No lint/format config checked in. Headless test boxes need `QT_QPA_PLATFORM=offscreen`.

## Architecture

Entry: `main.py` → `sticker_creator.app.StickerCreatorApp` (sets KDE Breeze style + `resources/style.qss`) → `MainWindow`.

`MainWindow` (`sticker_creator/main_window.py`) orchestrates everything. Holds the pipeline state: `_current_image`, `_current_mask`, `_raw_sticker` (no border), `_current_sticker` (with optional border). Border toggle/width re-derives `_current_sticker` from `_raw_sticker` without re-running segmentation.

Pipeline:
1. Load image — `utils/file_io.ImageLoader` or clipboard via `utils/clipboard.ClipboardManager`, drop via `widgets/drop_overlay.DropOverlay`. Empty state shown by `widgets/placeholder_view.PlaceholderView`.
2. User clicks points on `widgets/image_viewer.ImageViewer` (positive/negative prompts).
3. `segmentation/segmenter.Segmenter` runs SAM 2 in a `QThread` via `_load_requested` / `_segment_requested` signals. Image embeddings cached per-image; predictor cached per-model.
4. `utils/sticker_border.extract_sticker` + `add_white_border`, then `utils/sticker_resize.resize_to_512`.
5. `widgets/sticker_preview.StickerPreview` shows result; save/copy via `ImageSaver` / `ClipboardManager`.

Workflow UX driven by `widgets/inspector_panel.InspectorPanel` (right-side controls), `widgets/toast_notification.ToastOverlay`, `widgets/model_manager.ModelManager` (model switch dialog), `widgets/shortcuts_dialog`.

## SAM 2 integration — IMPORTANT

`segmenter.py` carries non-obvious patches required by the current Python 3.14 / torch 2.11 / sam2 1.1 environment. Do not strip them without understanding why:

- `_patch_sam2_transforms()` — injects pure-torch drop-in for `sam2.utils.transforms` because `torchvision::nms` is broken under torch 2.11. Must run before `SAM2ImagePredictor` is imported.
- `_patch_sam2_checkpoint_loading()` — forces `weights_only=False` for `build_sam2._load_checkpoint` (older pickle protocol).
- `_CONFIG_MAP` — maps checkpoint stem (`sam2.1_hiera_tiny`) to Hydra config path (`configs/sam2.1/sam2.1_hiera_t`). Passing stem directly raises `MissingConfigException`.
- `_MIN_CHECKPOINT_BYTES` (~10 MB) — guards against corrupted/placeholder `.pt` files; downloader re-downloads if too small.
- `predict()` expects numpy `(N,2)` coords and `(N,)` labels in original image pixel space — not pre-scaled tensors.

Settings persisted via `QSettings("KDE", "Sticker Creator")`. Active model name lives there.

## Models

Checkpoints in `models/*.pt`. Known: `sam2.1_hiera_{tiny,small,base_plus,large}`. Tiny (~80 MB) is the default. `models/download_model.py` downloads from `dl.fbaipublicfiles.com`. UI prompts download via `utils/model_downloader` + `widgets/model_manager` if missing/corrupt.

## Packaging

- `packaging/flatpak/io.github.jeroenvdwaal.StickerCreator.yml` — Flatpak manifest
- `packaging/flatpak/build.sh` (`make flatpak-install`, `make flatpak-bundle`)
- `sticker-creator.desktop` — desktop file, `APP_ID = io.github.jeroenvdwaal.StickerCreator`
