"""Independent geometry truth and fail-closed configuration checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from softcut_solution.blade_profiles import fit_outline, largest_component, scale_bar
from softcut_solution.blade_config import build_config, workbench_config_module

ROOT = Path(__file__).resolve().parents[1]


def test_exact_wedge_recovers_known_dimensions_and_two_face_angles():
    x = np.arange(10., 241.)
    top = 40 - np.minimum(x-10, 100)*.1
    bottom = 40 + np.minimum(x-10, 100)*.1
    q = fit_outline(x, top, bottom, .01)
    assert q['thickness_mm'] == pytest.approx(.2, abs=1e-10)
    assert q['included_angle_deg'] == pytest.approx(2*np.degrees(np.arctan(.1)), abs=1e-8)
    assert q['virtual_sharp_apex_px'] == pytest.approx(10, abs=1e-8)
    assert q['bevel_lengths_mm'] == pytest.approx([1, 1], abs=1e-8)


def test_asymmetric_wedge_not_forced_to_be_symmetric():
    x = np.arange(10., 241.)
    top = 40 - np.minimum(x-10, 100)*.05
    bottom = 40 + np.minimum(x-10, 100)*.2
    q = fit_outline(x, top, bottom, .01)
    assert q['thickness_mm'] == pytest.approx(.25, abs=1e-10)
    assert q['face_angles_deg'] == pytest.approx(np.degrees(np.arctan([.05,.2])), abs=1e-8)


def test_scale_uses_full_pixel_cells_and_rejects_missing_bar():
    rgb = np.zeros((100, 200, 3))
    rgb[90:93, 100:150, 0] = 1
    s = scale_bar(rgb, 500)
    assert s['span_pixels'] == 50
    assert s['um_per_pixel'] == 10
    assert s['endpoint_sensitivity_um_per_pixel'] == pytest.approx([500/52,500/48])
    with pytest.raises(ValueError):
        scale_bar(np.zeros_like(rgb), 500)


def test_noise_component_does_not_join_blade():
    a = np.zeros((20, 20), dtype=bool)
    a[4:10, 3:15] = True
    a[19, 19] = True
    assert largest_component(a).sum() == 72


@pytest.fixture
def inputs():
    catalog = json.loads((ROOT/'configs/blade_profiles.measured.json').read_text(encoding='utf-8'))
    return catalog, workbench_config_module().catalog()['defaults']


def test_export_requires_view_approximation_and_independent_radius(inputs):
    catalog, base = inputs
    kwargs = dict(tip_radius_mm=.02, radius_source='test assumption, not a measurement')
    with pytest.raises(ValueError, match='normal to'):
        build_config(catalog, 'a_low_profile', base, **kwargs)
    with pytest.raises(ValueError, match='symmetric approximation'):
        build_config(catalog, 'a_low_profile', base, confirm_view_plane=True, **kwargs)
    with pytest.raises(ValueError, match='do not resolve'):
        build_config(catalog, 'a_low_profile', base, confirm_view_plane=True,
                     accept_symmetric_approximation=True, tip_radius_mm=None, radius_source='unknown')


def test_export_validates_against_real_workbench_without_loading_fe(inputs):
    catalog, base = inputs
    original = deepcopy(base)
    before = set(sys.modules)
    c = build_config(catalog, 'a_low_profile', base, tip_radius_mm=.02,
        radius_source='test-only independent assumption', confirm_view_plane=True,
        accept_symmetric_approximation=True)
    assert c['blade']['thickness_mm'] == pytest.approx(.24577, abs=1e-5)
    assert c['blade']['included_angle_deg'] == pytest.approx(13.31685, abs=1e-5)
    assert c['material'] == base['material']
    assert c['geometry']['angle_deg'] == base['geometry']['angle_deg']
    assert base == original
    assert not any(m.startswith(('torch','ipctk','prototype.')) for m in set(sys.modules)-before)
    command = workbench_config_module().solver_command(c, ROOT/'runs/test-not-executed')
    assert command[command.index('--blade-angle')+1] == str(c['blade']['included_angle_deg'])
    with pytest.raises(ValueError, match='圆鼻半径'):
        build_config(catalog, 'a_low_profile', base, tip_radius_mm=1,
            radius_source='invalid test', confirm_view_plane=True, accept_symmetric_approximation=True)


def test_all_profiles_preserve_unresolved_microscopic_geometry(inputs):
    catalog, _ = inputs
    assert len(catalog['profiles']) == 5
    for p in catalog['profiles']:
        assert p['tip_radius_mm'] is None
        assert not p['roughness_resolved']
        assert not p['qualification']['physical_prediction_qualified']
        assert p['identity']['use_state'] == 'unknown'
        assert p['sensitivity_envelope']['flank_angle_asymmetry_deg'][0] >= 0
