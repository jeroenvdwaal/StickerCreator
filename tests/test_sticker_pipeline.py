"""Tests for the sticker pipeline.

Every path — including the balloon — is pure numpy+PIL and runs without a Qt
event loop. That headless testability is the point of the StickerPipeline module.
"""

import numpy as np
import pytest

from sticker_creator.utils.sticker_border import BACKGROUND_TRANSPARENT
from sticker_creator.utils.sticker_pipeline import (
    StickerOptions,
    auto_border_width,
    build_raw_sticker,
    compose_sticker,
    resize_for_whatsapp,
)


@pytest.fixture
def image_and_mask():
    """A 300×300 RGB image with a centred opaque rectangle mask.

    The subject (180×200) is already larger than the 128 px min side, so
    ``build_raw_sticker`` crops without upscaling.
    """
    image = np.full((300, 300, 3), 64, dtype=np.uint8)
    image[50:250, 60:240] = [200, 30, 30]
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[50:250, 60:240] = 1
    return image, mask


class TestBuildRawSticker:
    def test_crops_to_mask_bbox(self, image_and_mask):
        image, mask = image_and_mask
        raw = build_raw_sticker(image, mask)
        # Subject is 180 wide × 200 tall; border-free crop matches the bbox.
        assert raw.shape == (200, 180, 4)
        assert raw[:, :, 3].min() == 255  # fully opaque subject

    def test_upscales_tiny_subject(self):
        image = np.full((40, 40, 3), 10, dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:20, 10:20] = 1  # 10×10 subject — under the 128 min side
        raw = build_raw_sticker(image, mask)
        assert max(raw.shape[:2]) >= 128


class TestAutoBorderWidth:
    def test_scales_with_size_and_floors_at_one(self):
        small = np.zeros((50, 50, 4), dtype=np.uint8)
        big = np.zeros((1000, 1000, 4), dtype=np.uint8)
        assert auto_border_width(small) == 1          # max(1, round(50*0.014))
        assert auto_border_width(big) == 14            # round(1000*0.014)


class TestComposeSticker:
    def test_no_border_returns_same_size(self, image_and_mask):
        image, mask = image_and_mask
        raw = build_raw_sticker(image, mask)
        out = compose_sticker(raw, StickerOptions(border_enabled=False))
        assert out.shape == raw.shape

    def test_border_grows_canvas(self, image_and_mask):
        image, mask = image_and_mask
        raw = build_raw_sticker(image, mask)
        out = compose_sticker(
            raw, StickerOptions(border_enabled=True, border_width=7)
        )
        # add_white_border pads by border_width on every side.
        assert out.shape[0] == raw.shape[0] + 14
        assert out.shape[1] == raw.shape[1] + 14

    def test_does_not_mutate_input(self, image_and_mask):
        image, mask = image_and_mask
        raw = build_raw_sticker(image, mask)
        before = raw.copy()
        compose_sticker(raw, StickerOptions(border_enabled=True, border_width=5))
        assert np.array_equal(raw, before)

    def test_background_is_not_baked_in(self, image_and_mask):
        image, mask = image_and_mask
        raw = build_raw_sticker(image, mask)
        # background is a preview-only concern; the composed sticker keeps
        # its transparent regions regardless of the option.
        opts = StickerOptions(border_enabled=False, background="white")
        out = compose_sticker(raw, opts)
        assert out.shape[2] == 4

    def test_balloon_merges_and_borders_together(self, image_and_mask):
        image, mask = image_and_mask
        raw = build_raw_sticker(image, mask)
        opts = StickerOptions(
            border_enabled=True, border_width=6, balloon_text="Hello!"
        )
        out = compose_sticker(raw, opts)
        # Balloon expands the canvas beyond the bare subject+border.
        assert out.shape[0] >= raw.shape[0]
        assert out.shape[1] >= raw.shape[1]
        assert out[:, :, 3].max() == 255


class TestResizeForWhatsapp:
    def test_fits_512_canvas(self, image_and_mask):
        image, mask = image_and_mask
        raw = build_raw_sticker(image, mask)
        out = resize_for_whatsapp(raw)
        assert out.shape == (512, 512, 4)
