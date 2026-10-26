import numpy as np
import pytest

from src.analysis.loading_blade_reference import (prepare_loading_blade_reference,
    guard_loading_reference, restrict_loading_case, group_cutoff, guarded_prefix)
from src.analysis.contact_motion_evidence import supported_local_slopes


def test_irregular_time_interpolation_preserves_observed_positions():
    t = np.array([0., .02, .06, .1, .15, .21, .25])
    p = np.column_stack((5 + 2*t, 8 + 4*t))
    supplied = np.ones(len(t), bool)
    supplied[2:4] = False
    original = p.copy()
    p[2:4] = 999
    result = prepare_loading_blade_reference(p, supplied, t)
    np.testing.assert_array_equal(result['tips_xy'][supplied], original[supplied])
    np.testing.assert_allclose(result['tips_xy'], original)
    np.testing.assert_array_equal(result['state'], [2, 2, 1, 1, 2, 2, 2])
    assert result['metadata']['interpolated_frames'] == 2
    np.testing.assert_array_equal(p[2:4], np.full((2, 2), 999))


def test_unbounded_and_long_gaps_are_never_extrapolated_or_filled():
    t = np.arange(50)*.02
    p = np.column_stack((2*t, 3*t))
    supported = np.ones(len(t), bool)
    supported[:2] = False
    supported[-3:] = False
    supported[10:30] = False
    result = prepare_loading_blade_reference(p, supported, t, max_gap_s=.2)
    assert np.all(result['state'][~supported] == 0)
    assert np.isnan(result['tips_xy'][~supported]).all()
    assert np.isnan(result['velocity_xy_px_s'][~supported]).all()
    assert result['metadata']['extrapolated_coordinates'] == 0


def test_loading_endpoint_uses_no_future_withdrawal_and_marks_one_sided():
    t = np.arange(80)*.0125
    p = np.column_stack((100 + .4*t, 150 + 20*t))
    result = prepare_loading_blade_reference(p, np.ones(len(t), bool), t)
    np.testing.assert_allclose(result['velocity_xy_px_s'], np.tile([.4, 20.], (len(t), 1)), atol=1e-9)
    assert result['velocity_state'][0] == result['velocity_state'][-1] == 1
    assert result['velocity_single_sided'][0] and result['velocity_single_sided'][-1]
    assert np.all(result['velocity_state'][8:-8] == 2)


def test_interpolation_marks_velocity_conditional():
    t = np.arange(100)*.0125
    p = np.column_stack((np.ones(len(t))*100, 150 + 20*t))
    good = np.ones(len(t), bool)
    good[30:61] = False
    result = prepare_loading_blade_reference(p, good, t)
    assert np.all(result['state'][30:61] == 1)
    assert np.all(result['velocity_state'][30:61] == 1)
    np.testing.assert_allclose(result['velocity_xy_px_s'][30:61, 1], 20., atol=1e-9)
    assert result['metadata']['longest_missing_frames'] == 31
    assert result['metadata']['longest_bounded_endpoint_span_s'] == pytest.approx(.4)


def test_robust_velocity_reduces_single_bad_registration_influence():
    t = np.arange(100)*.0125
    p = np.column_stack((100 + .4*t, 150 + 20*t))
    p[51] += [8, -12]
    result = prepare_loading_blade_reference(p, np.ones(len(t), bool), t)
    assert abs(result['velocity_xy_px_s'][50, 1] - 20.) < 1.
    np.testing.assert_array_equal(result['tips_xy'], p)
    assert result['velocity_fit_rms_px'][50] > 1.


def test_rejects_non_monotonic_time_and_bad_shapes():
    with pytest.raises(ValueError, match='strictly increasing'):
        prepare_loading_blade_reference(np.zeros((4, 2)), np.ones(4), [0, .1, .1, .2])
    with pytest.raises(ValueError, match='T,2'):
        prepare_loading_blade_reference(np.zeros((4, 3)), np.ones(4), np.arange(4))


def test_empty_or_all_unknown_series_has_no_invented_reference():
    for n in (0, 10):
        result = prepare_loading_blade_reference(np.zeros((n, 2)), np.zeros(n, bool), np.arange(n)*.02)
        assert not result['state'].any()
        assert not result['velocity_state'].any()
        assert np.isnan(result['tips_xy']).all()


def test_blade_guard_rejects_cluster_without_bending_good_points():
    t = np.arange(200)*.0125
    expected = np.column_stack((np.full(200, 450.), 200+150*t))
    xy = expected.copy()
    xy[70:76, 1] = 117.
    ref = guard_loading_reference(xy, np.ones(200, bool), t)
    assert ref['rejected'][70:76].all()
    assert ref['rejected'].sum() == 6
    assert (ref['state'][70:76] == 1).all()
    np.testing.assert_allclose(ref['tips_xy'], expected)
    assert not ref['accepted'][70:76].any()

def test_long_blade_gap_remains_unknown():
    t = np.arange(200)*.0125
    xy = np.column_stack((np.full(200, 450.), 200+150*t))
    good = np.ones(200, bool)
    good[40:100] = False
    ref = guard_loading_reference(xy, good, t)
    assert np.isnan(ref['tips_xy'][40:100]).all()

def make_case():
    t = np.arange(120)*.0125
    tips = np.column_stack((20+0*t, 10+100*t))
    points = np.stack((np.column_stack((2*t, 3*t**2)),
                       np.column_stack((5+t, 7+2*t))), axis=1)
    case = dict(frames=np.arange(10, 120), times=t, um_per_px=10., positions=points,
                original_support=np.ones((120,2), bool), support=np.ones((120,2), bool),
                edges=np.ones((120,2), bool), gap=np.zeros((110,2)),
                material=np.ones((110,2), bool), legacy=np.zeros((110,4)),
                original_depth=np.zeros(110))
    return case, tips, np.ones(120, bool)

def test_later_motion_cannot_change_prefix_coordinates_or_velocity():
    case, tips, good = make_case()
    prefix = restrict_loading_case(case, tips, good, .75)
    assert prefix['frames'][-1] == 70
    assert prefix['cutoff']['actual_depth_mm'] == pytest.approx(.75)
    altered, changed_tips, changed_good = make_case()
    altered['positions'][72:] += 1000
    changed_tips[72:, 1] = -1000  # A later return below cutoff cannot re-enter.
    p2 = restrict_loading_case(altered, changed_tips, changed_good, .75)
    for p in (prefix, p2):
        p['derivative'] = supported_local_slopes(p['positions'], p['times'], p['support'],
                                                motion_step_valid=p['edges'])
    np.testing.assert_array_equal(prefix['frames'], p2['frames'])
    np.testing.assert_allclose(prefix['reference']['tips_xy'], p2['reference']['tips_xy'])
    np.testing.assert_allclose(prefix['derivative']['slope'], p2['derivative']['slope'], equal_nan=True)
    assert not prefix['derivative']['valid'][-1].any()  # Endpoint prediction remains explicit.
    assert len(case['positions']) == 120

def test_endpoint_never_uses_an_observation_after_cutoff_to_fill_a_gap():
    case, tips, good = make_case()
    good[68:71] = False
    prefix = restrict_loading_case(case, tips, good, .75)
    assert prefix['frames'][-1] == 67
    assert prefix['reference']['state'][-1] == 2
    assert np.all(prefix['reference']['tips_xy'][:,1] <= tips[67,1])

def test_rejects_unobserved_datum_and_invalid_depth():
    case, tips, good = make_case()
    good[10] = False
    with pytest.raises(ValueError, match='first-contact'):
        restrict_loading_case(case, tips, good, .75)
    with pytest.raises(ValueError, match='positive'):
        restrict_loading_case(case, tips, good, -1)

@pytest.mark.parametrize('name, expected', [
    ('0930l30a1', 1300), ('1008n30a2', 1300), ('1014n33a3', 1300),
    ('1007L60A1', 2200), ('1014l57a2', 2200), ('1014n57a1', 2200),
    ('1005L60C2up', 550), ('1014n57c3up', 550),
    ('1005L60C1down', 850), ('1014n57c2down', 850)])
def test_declared_group_mapping(name, expected):
    assert group_cutoff(name)['depth_um'] == expected

def test_unrecognized_case_is_never_silently_assigned_a_cutoff():
    with pytest.raises(ValueError):
        group_cutoff('new_unknown_case')

def test_all_post_cutoff_positions_and_tips_are_excluded_from_guard_and_analysis():
    case, tips, good = make_case()
    prefix = guarded_prefix(case, tips, good, .75)
    changed, altered, supported = make_case()
    changed['positions'][71:] = 1.e6
    altered[71:] = [10000., 10000.]
    second = guarded_prefix(changed, altered, supported, .75)
    assert prefix['frames'][-1] == 70
    assert prefix['cutoff']['maximum_analysis_source_frame'] == 70
    assert len(prefix['positions']) == 71
    for key in ('tips_xy', 'velocity_xy_px_s', 'accepted', 'analysis_qualified'):
        np.testing.assert_allclose(prefix['reference'][key], second['reference'][key], equal_nan=True)

def test_prefix_guard_rejects_wrong_shaft_cluster_without_using_later_loading():
    case, tips, good = make_case()
    tips[40:43, 1] -= 50.
    prefix = guarded_prefix(case, tips, good, .75)
    guard = prefix['reference']
    local = np.isin(prefix['frames'], [40, 41, 42])
    assert guard['rejected'][local].all()
    assert not guard['analysis_qualified'][local].any()
    assert prefix['cutoff']['actual_depth_mm'] <= .75
