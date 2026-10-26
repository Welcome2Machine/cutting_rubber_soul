"""Regression coverage for deployed segmentation and its image/cache contract."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.segmentation import (
    ConvGRUCell, TemporalSegmenter, build_model, conservative_postprocess,
    semantic_overlay, segmentation_frame_features, frame_features,
    redact_overlay_region, remove_fixed_grid, refine_material_boundary,
    SourceFrameCache, SegmentationFeatureCache,
)


def test_unlabelled_top_is_preserved_without_title_or_false_boundary():
    frame = np.full((80, 120, 3), 137, np.uint8)
    mask = np.ones((80, 120), np.uint8)
    mask[:40] = 255
    overlay = semantic_overlay(frame, mask, 1, 0, annotate=False)
    np.testing.assert_array_equal(overlay[:40], frame[:40])


def test_postprocess_fills_a_small_background_bubble_but_never_a_blade_hole() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[4:60, 4:60] = 1
    cv2.circle(mask, (20, 20), 4, 0, -1)  # lubricant-like small background bubble
    mask[32:42, 30:40] = 2  # visible blade must remain excluded from material
    cleaned = conservative_postprocess(mask, max_material_hole_area=100, min_blade_component_area=20)
    assert cleaned[20, 20] == 1
    assert cleaned[36, 35] == 2


def test_postprocess_drops_only_tiny_disconnected_blade_flecks() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, :30] = 1
    mask[10:45, 34:40] = 2
    mask[3, 60] = 2
    cleaned = conservative_postprocess(mask, min_blade_component_area=20)
    assert cleaned[20, 36] == 2
    assert cleaned[3, 60] == 0


def test_postprocess_drops_small_material_island_but_keeps_border_material() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[:, :42] = 1
    mask[20:28, 58:66] = 1  # isolated bright-patch false positive
    cleaned = conservative_postprocess(mask, min_isolated_material_area=100)
    assert cleaned[10, 10] == 1
    assert cleaned[24, 62] == 0


def test_postprocess_preserves_a_narrow_retraction_gap() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[4:76, 4:76] = 1
    mask[18:58, 36:40] = 0  # narrow real separation, area below hole threshold
    cleaned = conservative_postprocess(mask, max_material_hole_area=500)
    assert cleaned[30, 37] == 0


def test_legacy_path_is_bitwise_identical():
    frame = np.random.default_rng(1).integers(0, 256, (768, 1024, 3), dtype=np.uint8)
    config = {'source_crop_top_px': 40, 'grid_suppression': {'enabled': True}}
    old = frame_features(remove_fixed_grid(redact_overlay_region(
        cv2.resize(frame[40:], (512, 384), interpolation=cv2.INTER_AREA), config), config))
    np.testing.assert_array_equal(segmentation_frame_features(frame, (384, 512), config), old)


def test_grid_removed_in_source_coordinates_before_nonuniform_resize():
    frame = np.full((768, 1024, 3), 160, np.uint8)
    frame[[127, 255, 383, 511, 639]] = 85
    frame[:, [170, 341, 511, 682, 853]] = 85
    original = frame.copy()
    config = {'source_crop_top_px': 40, 'grid_coordinate_space': 'source_0915',
              'grid_suppression': {'enabled': True}}
    result = segmentation_frame_features(frame, (384, 512), config)
    expected = frame_features(np.full((384, 512, 3), 160, np.uint8))
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(frame, original)


def test_disabled_grid_does_not_modify_pixels():
    frame = np.random.default_rng(2).integers(0, 256, (768, 1024, 3), dtype=np.uint8)
    config = {'source_crop_top_px': 40, 'grid_coordinate_space': 'source_0915',
              'grid_suppression': {'enabled': False}}
    expected = frame_features(cv2.resize(frame[40:], (512, 384), interpolation=cv2.INTER_AREA))
    np.testing.assert_array_equal(segmentation_frame_features(frame, (384, 512), config), expected)


def test_source_mode_rejects_cropped_input_and_unknown_coordinate_space():
    with pytest.raises(ValueError):
        segmentation_frame_features(np.zeros((728, 1024, 3), np.uint8), (384, 512),
            {'grid_coordinate_space': 'source_0915', 'grid_suppression': {'enabled': True}})
    with pytest.raises(ValueError):
        segmentation_frame_features(np.zeros((768, 1024, 3), np.uint8), (384, 512),
                                     {'grid_coordinate_space': 'guess'})


def scene():
    rng = np.random.default_rng(12)
    im = np.zeros((100, 150, 3), np.uint8)
    im[:] = 6
    im[:, :80] = rng.integers(65, 95, (100, 80, 1), dtype=np.uint8)
    labels = np.zeros((100, 150), np.uint8)
    labels[:, :83] = 1  # known three-pixel outward bias
    labels[:5] = 255
    labels[20:40, 75:85] = 2
    labels[60:65, 30:35] = 0  # real enclosed gap must remain untouched
    return im, labels


def test_image_edge_correction_preserves_semantics_and_interior_gap():
    im, labels = scene()
    original = labels.copy()
    before_image = im.copy()
    result, info = refine_material_boundary(im, labels)
    assert info['accepted']
    assert np.all(result[50:95, :80] == labels[50:95, :80])
    assert np.all(result[50:95, 80:83] == 0)
    protected = cv2.dilate(((labels == 2) | (labels == 255)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(bool)
    np.testing.assert_array_equal(result[protected], labels[protected])
    np.testing.assert_array_equal(labels, original)
    np.testing.assert_array_equal(im, before_image)
    assert np.all(result[60:65, 30:35] == 0)


def test_new_unseeded_hole_is_removed_without_filling_real_gap(monkeypatch):
    im, labels = scene()
    def faulty_cut(image, trimap, *args):
        trimap[70, 80] = cv2.GC_PR_BGD  # singleton hole inside original material
    monkeypatch.setattr(cv2, 'grabCut', faulty_cut)
    result, info = refine_material_boundary(im, labels)
    assert info['accepted']
    assert info['projected_speckle_pixels'] == 1
    np.testing.assert_array_equal(result, labels)


def test_topology_change_after_projection_rejects_candidate(monkeypatch):
    import src.segmentation as module
    im, labels = scene()
    answers = iter([(1, 0), (2, 0)])
    monkeypatch.setattr(module, '_topology', lambda binary: next(answers))
    result, info = refine_material_boundary(im, labels)
    assert not info['accepted']
    assert info['reason'] == 'rejected_component_or_hole_change'
    np.testing.assert_array_equal(result, labels)


def test_empty_material_is_identity():
    im = np.zeros((50, 50, 3), np.uint8)
    labels = np.zeros((50, 50), np.uint8)
    result, info = refine_material_boundary(im, labels)
    assert not info['accepted']
    np.testing.assert_array_equal(result, labels)


@pytest.mark.parametrize('case', ['float_frame', 'wrong_class', 'wrong_shape', 'negative_radius'])
def test_invalid_input_rejected(case):
    im, labels = scene()
    params = {}
    if case == 'float_frame':
        im = im.astype(np.float32)
    elif case == 'wrong_class':
        labels[50, 50] = 4
    elif case == 'wrong_shape':
        labels = labels[:-1]
    else:
        params['band_radius'] = -1
    with pytest.raises(ValueError):
        refine_material_boundary(im, labels, **params)


def test_feature_cache_matches_uncached_overlapping_windows(tmp_path):
    path=tmp_path/'input.avi'
    out=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),12,(96,80))
    rng=np.random.default_rng(4)
    for _ in range(45):out.write(rng.integers(0,256,(80,96,3),dtype=np.uint8))
    out.release()
    a=SourceFrameCache(path,16);b=SourceFrameCache(path,16)
    cached=SegmentationFeatureCache(a,(32,40),{'source_crop_top_px':4})
    plain=SegmentationFeatureCache(b,(32,40),{'source_crop_top_px':4},enabled=False)
    try:
        for centre in range(45):
            a.prefetch_through(centre+4);b.prefetch_through(centre+4)
            for offset in (-4,-2,0,2,4):
                np.testing.assert_array_equal(cached.read(centre+offset),plain.read(centre+offset))
            assert len(cached.cache)<=16
    finally:a.close();b.close()


def test_convgru_cell_preserves_spatial_shape() -> None:
    cell = ConvGRUCell(8, 12)
    value = torch.randn(2, 8, 16, 20)
    hidden = cell(value)
    assert hidden.shape == (2, 12, 16, 20)
    assert cell(value, hidden).shape == hidden.shape


def test_deployed_checkpoint_contract():
    checkpoint = Path(__file__).resolve().parents[1] / 'models/segmentation/clean_roi_20260909/checkpoint_best.pt'
    if not checkpoint.is_file():
        pytest.skip('Local deployment checkpoint is not distributed in Git')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_model(payload['config']['model']).eval()
    model.load_state_dict(payload['state_dict'], strict=True)
    sequence = torch.zeros(1, model.sequence_length, model.input_channels, 16, 24)
    with torch.inference_mode():
        logits = model(sequence)
    assert logits.shape == (1, model.num_classes, 16, 24)
    assert torch.isfinite(logits).all()
