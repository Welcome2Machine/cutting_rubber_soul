"""Versioned configuration and Gauss-event reporting; no FE loading."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.run_cohesive_cut import read_config, gauss_centers, events
from prototype.cohesive_mixed import MixedModeLaw


ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_configurations_and_legacy_remain_distinct():
    for name in ['fracture_demo.json', 'fracture_mixed.json', 'fracture_network.json',
                 'fracture_network_regularized.json']:
        config = read_config(ROOT/'configs'/name)
        assert config['target_mm'] <= 2
    assert read_config(ROOT/'configs/fracture_network.json')['fracture_topology'] == 'edge_network'


@pytest.mark.parametrize('changes', [
    {'no_experimental_force_fitting': False}, {'angle_deg': 90},
    {'angle_deg': float('nan')}, {'angle_deg': True},
    {'cohesive_kind': 'automatic'}, {'fracture_topology': 'unknown'},
    {'network_predictor': 'skip_ccd'}, {'network_predictor': None},
    {'cohesive': {'normal_strength_mpa': .2}},
])
def test_incomplete_or_ambiguous_fracture_configuration_rejected(tmp_path, changes):
    config = json.loads((ROOT/'configs/fracture_network.json').read_text())
    config.update(changes)
    path = tmp_path/'input.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        read_config(path)


def test_mixed_angle_does_not_silently_use_legacy_radial_law(tmp_path):
    config = json.loads((ROOT/'configs/fracture_demo.json').read_text())
    config['angle_deg'] = 57
    path = tmp_path/'legacy.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Legacy'):
        read_config(path)


def test_gauss_locations_and_partial_damage_event_keep_identity(tmp_path):
    law = MixedModeLaw()
    x = np.array([[0.,0.], [.006,0.], [0.,1.], [.006,1.]])
    segments = np.array([[0,1,2,3]])
    centers = gauss_centers(x, segments)
    np.testing.assert_allclose(centers[0,:,0], .003)
    np.testing.assert_allclose(centers[0,:,1], (.5*(1-1/np.sqrt(3)), .5*(1+1/np.sqrt(3))))
    (tmp_path/'states').mkdir()
    history = np.array([[law.final_opening_mm, 0.]])
    np.savez(tmp_path/'states/0000.npz', cohesive_maximum=history,
             interface_segments=segments, reference_xy=x, positions=x)
    steps = [dict(depth_mm=.01, reaction_fz_n=0., fracture=dict(
        max_damage=1., separated_bonds=0, surface_connected_crack_length_mm=0))]
    result = events(steps, tmp_path, law)
    assert result['damage_onset']['interface_segment_and_gauss_ids'] == [[0,0]]
    assert result['first_fully_separated_segment']['status'] == 'not_observed_in_accepted_prefix'
    assert 'bond_ids' not in result['damage_onset']


@pytest.mark.parametrize('changes', [
    {'closure_regularization': None}, {'preparation_clearance_mm': 0.},
    {'minimum_increment_mm': float('nan')}, {'minimum_increment_mm': .1},
    {'schema': 'softcut-cohesive/2'},
    {'fracture_topology': 'prescribed_seam'}, {'ptc_model_backtracks': True},
    {'ptc_model_backtracks': 31}, {'ptc_model_backtracks': -1},
    {'ptc_residual_guard': 1}, {'ptc_residual_guard': 'true'},
    {'ptc_residual_guard': True, 'ptc_model_backtracks': 0},
    {'ptc_shift_policy': 'automatic'},
    {'ptc_shift_policy': 'backtrack_aware', 'ptc_model_backtracks': 0},
])
def test_regularization_requires_explicit_versioned_controls(tmp_path, changes):
    config = json.loads((ROOT/'configs/fracture_network_regularized.json').read_text())
    config.update(changes)
    path = tmp_path/'input.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        read_config(path)


def test_compression_energy_is_not_counted_twice_in_increment_work():
    from tools.run_cohesive_cut import energy_ledger
    def state(depth, total, compression):
        return dict(depth_mm=depth, reaction_fz_n=2., friction_work_n_mm=0.,
            material=dict(equilibrium_energy_n_mm=0., overstress_energy_n_mm=0.,
                          cumulative_viscous_dissipation_n_mm=0.),
            fracture=dict(cohesive_stored_energy_n_mm=total, compression_stored_energy_n_mm=compression,
                          fracture_dissipation_n_mm=0., ipc_barrier_stored_energy_n_mm=0.,
                          max_body_step_motion_mm=1.))
    row = energy_ledger([state(0., 0., 0.), state(1., 2., .7)])[0]
    assert row['unaccounted_energy_n_mm'] == 0.
    assert row['compression_stored_change_n_mm'] == .7
    assert row['damageable_cohesive_stored_change_n_mm'] == 1.3


@pytest.mark.parametrize('length', [None, 0., -1., True, 1e-4])
def test_bounded_pressure_length_is_derived_from_gap_not_free_parameter(tmp_path, length):
    config = json.loads((ROOT/'configs/fracture_network_regularized.json').read_text())
    config['cohesive']['compression_regularization_mm'] = length
    path = tmp_path/'input.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='length must equal'):
        read_config(path)


def test_linear_closure_cannot_silently_enable_bounded_pressure(tmp_path):
    config = json.loads((ROOT/'configs/fracture_network_regularized.json').read_text())
    config['closure_regularization'] = 'reversible_compression_penalty'
    path = tmp_path/'input.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Linear compression'):
        read_config(path)


@pytest.mark.parametrize('width', [0., -1., True, float('nan'), 'small'])
def test_tip_refinement_width_is_explicit_positive_geometry(tmp_path, width):
    config = json.loads((ROOT/'configs/fracture_network_regularized.json').read_text())
    config['mesh']['fine_halfwidth_mm'] = width
    path = tmp_path/'input.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='fine_halfwidth'):
        read_config(path)
