import json
import numpy as np
import pytest

from tools.fracture_onset import neighborhood_operator, first_crossing, analyze_case
from workbench.results import saved_state_fields
from test_workbench_results import MATERIAL, X, TRI, manufactured_case


def test_full_plane_strain_stress_keeps_out_of_plane_and_energy_split():
    x = X @ np.diag([2., .5])
    fields, energy, _ = saved_state_fields(X, x, TRI, np.eye(3)[None], MATERIAL, include_mechanics=True)
    np.testing.assert_allclose(np.diagonal(fields['cauchy_MPa'][0]), 1.38*(np.array([4., .25, 1.])-1.75))
    assert fields['cauchy_MPa'][0, 2, 2] != 0
    np.testing.assert_allclose(fields['equilibrium_density_MPa']+fields['nonequilibrium_density_MPa'], energy)
    np.testing.assert_allclose(fields['principal_stretch'], [[.5, 1., 2.]])


def test_stress_direction_rotates_and_metric_is_never_reset():
    theta = .51
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = np.diag([4., .25, 1.])[None]
    a, _, _ = saved_state_fields(X, X, TRI, A, MATERIAL, include_mechanics=True)
    b, _, _ = saved_state_fields(X, X @ R.T + 3, TRI, A, MATERIAL, include_mechanics=True)
    np.testing.assert_allclose(a['principal_stress_MPa'], b['principal_stress_MPa'], atol=1e-12)
    assert abs(np.dot(b['inplane_normal'][0], R @ a['inplane_normal'][0])) == pytest.approx(1)


def test_reference_average_preserves_constant_and_weights_volume():
    op = neighborhood_operator([[0., 0.], [.1, 0.]], [1., 3.], .2)
    np.testing.assert_allclose(op @ np.ones(2), 1)
    assert (op @ np.array([0., 4.]))[0] == pytest.approx(3/1.75)
    with pytest.raises(ValueError, match='pairs'):
        neighborhood_operator([[0., 0.], [.1, 0.]], [1., 3.], .2, max_pairs=1)


def history(values):
    return [dict(index=i, depth_mm=.1*i, fz_N=float(i), q_inplane_MPa=v,
                 location_reference_mm=[0,1], local_normal=[1,0]) for i,v in enumerate(values)]


def test_first_event_not_largest_later_peak_or_interpolated_equilibrium():
    event = first_crossing(history([.1, .8, .2, 1.5]), .5)
    assert event['state_indices'] == [0,1]
    assert event['depth_interval_mm'] == [0.,.1]
    assert not event['calibrated']
    assert first_crossing(history([1]), .5)['status'] == 'already_above_at_first_qualified_state'
    assert first_crossing(history([.1]), .5, prefix_complete=False)['status'] == 'censored_by_qualification'
    assert first_crossing(history([.1]), None)['status'] == 'missing_threshold'


def test_saved_analysis_preserves_source_and_stops_at_audit_hole(tmp_path):
    source = manufactured_case(tmp_path/'source')
    auditpath = source/'audit.json'
    audit = json.loads(auditpath.read_text())
    audit['rows'][1]['passed'] = False
    auditpath.write_text(json.dumps(audit))
    before = (source/'result.json').read_bytes()
    result = analyze_case(source, tmp_path/'analysis', length_mm=.2, threshold=1.,
                          tip_radius_mm=10., clamp_exclusion_mm=0)
    assert len(result['rows']) == 1
    assert result['event']['status'] == 'censored_by_qualification'
    assert (source/'result.json').read_bytes() == before
    assert result['rows'][0]['q_inplane_MPa'] == pytest.approx(0.)
    assert (tmp_path/'analysis/index.html').is_file()


def test_modified_saved_state_is_rejected(tmp_path):
    source = manufactured_case(tmp_path/'source')
    with (source/'states/0000.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        analyze_case(source, tmp_path/'analysis', tip_radius_mm=10.)


def test_coupled_damage_history_cannot_be_labelled_intact(tmp_path):
    source = manufactured_case(tmp_path/'source')
    path = source/'result.json'
    result = json.loads(path.read_text())
    result['schema'] = 'softcut-cohesive-result/1'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='coupled fracture'):
        analyze_case(source, tmp_path/'analysis')
