from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from src.tracking.fixed_grid import FixedGridConfig, FixedGridProcessor, grid_0915_config


def test_only_configured_pixels_change_and_input_is_not_mutated():
    image = np.random.default_rng(15).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    before = image.copy()
    process = FixedGridProcessor(FixedGridConfig(48, 32, (9, 33), (7, 21)))
    output = process(image)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(output[~process.mask()], image[~process.mask()])
    assert not np.shares_memory(output, image)


@pytest.mark.parametrize("radius", [0, 1, 2])
@pytest.mark.parametrize("channels", [1, 3])
def test_linear_plane_recovers_overlay_and_intersections_exactly(radius, channels):
    y, x = np.mgrid[:32, :48]
    truth = (20 + x + 2 * y).astype(np.uint8)
    if channels == 3:
        truth = np.stack([truth, truth + 10, truth + 20], axis=-1)
    process = FixedGridProcessor(FixedGridConfig(48, 32, (9, 33), (7, 21), radius))
    observed = truth.copy()
    observed[process.mask()] = 85
    result = process(observed)
    np.testing.assert_array_equal(result, truth)


def test_known_smooth_texture_has_small_reconstruction_error():
    y, x = np.mgrid[:80, :100]
    truth = np.rint(120 + 25 * np.sin(x / 9) + 15 * np.cos(y / 11)).astype(np.uint8)
    process = FixedGridProcessor(FixedGridConfig(100, 80, (20, 70), (25, 60)))
    observed = truth.copy()
    observed[process.mask()] = 85
    output = process(observed)
    error = np.abs(output.astype(float) - truth.astype(float))
    assert error.max() <= 1
    assert error[process.mask()].mean() < 0.4
    # This is a known-ground-truth check, not a claim that smoother videos have
    # more accurate displacements; texture overwritten by a real grid is lost.


def test_crop_after_processing_uses_original_coordinates():
    process = FixedGridProcessor(grid_0915_config())
    original = np.full((768, 1024), 140, np.uint8)
    original[process.mask()] = 85
    crop = process(original)[100:700, 50:950]
    np.testing.assert_array_equal(crop, np.full_like(crop, 140))
    assert process.mask()[100:700, 50:950][27, 120]
    with pytest.raises(ValueError, match="before crop/resize"):
        process(original[100:])


def test_intersections_use_four_unmasked_corners_not_overlay_values():
    image = np.zeros((5, 5), np.uint8)
    image[1, 1], image[1, 3], image[3, 1], image[3, 3] = 20, 40, 80, 100
    process = FixedGridProcessor(FixedGridConfig(5, 5, (2,), (2,)))
    image[process.mask()] = 255
    output = process(image)
    assert output[2, 2] == 60


def test_merged_neighboring_bands_never_use_another_overlay_pixel():
    truth = np.tile(np.arange(20, dtype=np.uint8) * 5, (15, 1))
    process = FixedGridProcessor(FixedGridConfig(20, 15, (7, 8, 9), (5, 6), 1))
    observed = truth.copy()
    observed[process.mask()] = 255
    np.testing.assert_array_equal(process(observed), truth)


def test_no_implicit_grid_when_coordinates_are_empty():
    image = np.random.default_rng(4).integers(0, 256, (10, 12), dtype=np.uint8)
    process = FixedGridProcessor(FixedGridConfig(12, 10))
    assert not process.mask().any()
    np.testing.assert_array_equal(process(image), image)


def test_config_is_deeply_immutable_and_signature_is_canonical():
    lines = [8, 3]
    config = FixedGridConfig(20, 15, lines, (5,))
    lines.append(12)
    assert config.vertical_lines == (3, 8)
    with pytest.raises(FrozenInstanceError):
        config.width = 99
    process = FixedGridProcessor(config)
    with pytest.raises(AttributeError):
        process.config = FixedGridConfig(20, 15)
    metadata = json.loads(json.dumps(process.metadata()))
    assert metadata["vertical_lines"] == [3, 8]
    metadata["vertical_lines"].append(12)
    assert process.metadata()["vertical_lines"] == [3, 8]
    assert process.signature() == FixedGridProcessor(FixedGridConfig(20, 15, (3, 8), (5,))).signature()
    assert process.signature() != FixedGridProcessor(FixedGridConfig(20, 15, (3, 9), (5,))).signature()
    assert process.signature() != FixedGridProcessor(FixedGridConfig(20, 15, (3, 8), (5,), 1)).signature()


@pytest.mark.parametrize("args", [
    (20, 15, (0,), ()), (20, 15, (19,), ()), (20, 15, (), (14,)),
    (20, 15, (1,), (), 1), (20, 15, (2.5,), ()), (20, 15, (), (), -1),
])
def test_invalid_or_unbounded_bands_fail_explicitly(args):
    with pytest.raises(ValueError):
        FixedGridConfig(*args)


def test_0915_preset_replaces_exactly_confirmed_one_pixel_lines():
    process = FixedGridProcessor(grid_0915_config())
    mask = process.mask()
    assert mask.sum() == 5 * 1024 + 5 * 768 - 25
    assert mask[127, 300] and not mask[126, 300] and not mask[128, 300]
    assert mask[200, 341] and not mask[200, 340] and not mask[200, 342]


@pytest.mark.parametrize("shape,dtype", [((15, 20, 4), np.uint8), ((15, 20), np.float32)])
def test_unsupported_frame_layout_or_dtype_rejected(shape, dtype):
    with pytest.raises(ValueError, match="uint8 grayscale or BGR"):
        FixedGridProcessor(FixedGridConfig(20, 15))(np.zeros(shape, dtype=dtype))
