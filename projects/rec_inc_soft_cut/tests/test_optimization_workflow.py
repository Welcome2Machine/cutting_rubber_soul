"""The research runner must preserve causal cutoffs, IDs and production evidence."""
from pathlib import Path

import numpy as np
import pytest

from src.workflows import optimize


def test_prefix_uses_first_crossing_and_keeps_preceding_depth_fluctuations():
    frames = np.arange(10, 16)
    tips = np.c_[np.ones(6), [0., 4., 3.5, 9.9, 10.1, 9.8]]
    f, t, d = optimize.depth_prefix(frames, tips, 100., 1.)
    np.testing.assert_array_equal(f, [10, 11, 12, 13])
    np.testing.assert_allclose(d, [0., .4, .35, .99])
    changed = tips.copy()
    changed[5] = [20, -900]
    np.testing.assert_array_equal(optimize.depth_prefix(frames, changed, 100., 1.)[0], f)


@pytest.mark.parametrize('scale,cutoff', [(0, 1), (1, 0), (np.nan, 1), (1, np.inf)])
def test_unknown_scale_and_cutoff_cannot_make_a_depth_window(scale, cutoff):
    with pytest.raises(ValueError):
        optimize.depth_prefix([0, 1], [[0, 0], [0, 1]], scale, cutoff)


def test_future_missing_geometry_cannot_invalidate_an_already_complete_prefix():
    frames = np.arange(6)
    tips = np.c_[np.zeros(6), [0., 4., 9.9, 10.1, np.nan, np.nan]]
    f, _, d = optimize.depth_prefix(frames, tips, 100., 1.)
    np.testing.assert_array_equal(f, [0, 1, 2])
    np.testing.assert_allclose(d, [0., .4, .99])
    tips[1, 1] = np.nan
    with pytest.raises(ValueError, match='through the first cutoff crossing'):
        optimize.depth_prefix(frames, tips, 100., 1.)


def test_zero_surface_reference_is_valid_but_negative_depth_is_not():
    optimize.inspect_config({'cases': [], 'surface_pilot_depths_mm': [0., .5]})
    with pytest.raises(ValueError, match='finite and positive'):
        optimize.inspect_config({'cases': [], 'surface_pilot_depths_mm': [-.1]})


def test_contour_diagnostics_use_segments_and_report_missing_contour():
    mapped = dict(mapped_xy_px=np.array([[2., 5.], [18., 5.]]),
                  segment_valid=np.array([True]), s_px=np.array([0., 16.]))
    mask = np.zeros((20, 20), np.uint8)
    result = optimize.surface_diagnostics(mapped, np.array([10., 6.]), mask)
    assert result['nearest_valid_surface_tip_gap_px'] == 1.
    assert result['tip_to_semantic_edge_px'] is None
    assert result['valid_surface_arc_fraction'] == 1.
    mapped['segment_valid'][:] = False
    assert optimize.surface_diagnostics(mapped, np.array([10., 6.]), mask)['nearest_valid_surface_tip_gap_px'] is None


def test_feature_ablation_refuses_changed_baseline_and_compares_common_ids(tmp_path):
    runs = {'legacy': {'synthetic': [{'scenario': 'x'}]}, 'material-masked': {}}
    common = dict(truth=np.array([[0., 0.], [1., 0.]]), reference_points=np.zeros((2, 2)),
                  baseline_positions=np.zeros((2, 2)), baseline_supported=np.array([True, True]))
    for mode in runs:
        (tmp_path/mode).mkdir()
        np.savez(tmp_path/mode/'synthetic_x.npz', **common, positions=np.array([[.1, 0], [10, 0]]),
                 supported=np.array([True, mode == 'legacy']))
    result = optimize.feature_comparison(tmp_path, runs)[0]
    assert result['common_supported_count'] == 1
    assert result['material-masked']['epe_common_px'] == pytest.approx(.1)
    assert result['material-masked']['epe_all_px'] == pytest.approx(4.55)
    common['baseline_positions'][0, 0] = 2
    np.savez(tmp_path/'material-masked/synthetic_x.npz', **common, positions=np.zeros((2, 2)),
             supported=np.array([True, False]))
    with pytest.raises(ValueError, match='identical truth and baseline'):
        optimize.feature_comparison(tmp_path, runs)


def test_saved_frame_ids_must_be_looked_up_not_used_as_array_indices():
    np.testing.assert_array_equal(optimize.aligned_rows([2, 5, 11, 18], [5, 18]), [1, 3])
    with pytest.raises(ValueError, match='absent'):
        optimize.aligned_rows([2, 5, 11], [3])
    with pytest.raises(ValueError, match='ordered'):
        optimize.aligned_rows([2, 2, 5], [2])


def test_spatial_subsampling_does_not_duplicate_or_change_material_ids():
    p = np.random.default_rng(143).normal(size=(300, 2))
    subset = optimize.spread_subset(p, 30)
    np.testing.assert_array_equal(subset, optimize.spread_subset(p, 30))
    assert len(set(subset)) == 30
    ids = np.arange(300)*7+40
    np.testing.assert_array_equal(ids[subset], (np.arange(300)*7+40)[subset])


def test_check_only_has_no_output_or_algorithm_side_effects(tmp_path, monkeypatch):
    config = tmp_path/'config.json'
    config.write_text('{"cases": [], "candidate": {}}')
    monkeypatch.setattr(optimize, 'synthetic_benchmark', lambda *a: pytest.fail('computed'))
    out = tmp_path/'new'
    assert optimize.main(['--config', str(config), '--out', str(out), '--check-only']) == 0
    assert not out.exists()


def test_existing_results_and_source_directory_are_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(optimize, 'ROOT', tmp_path)
    config = tmp_path/'config.json'
    config.write_text('{"cases": [], "candidate": {}}')
    existing = tmp_path/'results/existing'
    existing.mkdir(parents=True)
    evidence = existing/'keep.txt'
    evidence.write_text('immutable')
    with pytest.raises(FileExistsError):
        optimize.main(['--config', str(config), '--out', str(existing), '--stage', 'synthetic'])
    with pytest.raises(ValueError, match='only below'):
        optimize.main(['--config', str(config), '--out', str(tmp_path/'data'), '--stage', 'synthetic'])
    assert evidence.read_text() == 'immutable'


@pytest.mark.parametrize('key', ['pilot_depths_mm', 'surface_pilot_depths_mm'])
def test_surface_and_mesh_pilots_reject_unknown_depth_requests(key):
    with pytest.raises(ValueError, match='finite and positive'):
        optimize.inspect_config({'cases': [], key: [float('nan')]})


def test_all_runs_separate_surface_ablation_without_colliding_with_mesh(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(optimize, 'ROOT', tmp_path)
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'cases': [{'name': 'case'}]}))
    monkeypatch.setattr(optimize, 'inspect_config', lambda _: [])
    monkeypatch.setattr(optimize, 'synthetic_benchmark', lambda *args: [])
    def pilot(case, config, out):
        (out/case['name']).mkdir()
        return []
    def surface(case, config, out):
        (out/case['name']).mkdir()
        return [{'case': case['name']}]
    monkeypatch.setattr(optimize, 'pilot_case', pilot)
    monkeypatch.setattr(optimize, 'surface_benchmark', surface)
    monkeypatch.setattr(optimize, 'report', lambda *args: None)
    out = tmp_path/'results/new'
    assert optimize.main(['--config', str(config), '--out', str(out), '--stage', 'all']) == 0
    summary = json.loads((out/'summary.json').read_text())
    assert summary['surface'][0]['relative_directory'] == 'surface_curve_ablation/case'
    assert (out/'case').is_dir() and (out/'surface_curve_ablation/case').is_dir()
