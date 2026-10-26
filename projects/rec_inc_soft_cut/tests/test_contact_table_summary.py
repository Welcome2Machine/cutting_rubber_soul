import numpy as np
import pytest
from src.analysis.contact_table_summary import summarize_curve, extrapolate_cutoff, interpolate_internal_gaps


def make_data(y, updates=True):
    y = np.asarray(y, float)
    return dict(depth_mm=np.linspace(0, .3, len(y)), offset_um=y,
                transport_absolute_used=np.zeros(len(y), bool),
                transport_velocity_used=np.broadcast_to(updates, (len(y),)).copy(),
                um_per_px=np.array(6.), sigma_um=np.full(len(y), 20.))


def test_last_available_is_not_endpoint():
    r = summarize_curve([0, .1, .2], [0, 20, np.nan], [1, 2, 0], 200)
    assert r['table_endpoint_um'] is None
    assert r['last_estimate_um'] == 20 and r['last_estimate_depth_um'] == 100
    assert r['last_estimate_kind'] == 'conditional_interpolation'


def test_only_initialized_zero_cannot_prove_zero_slip():
    r = extrapolate_cutoff(make_data(np.zeros(31), updates=False), 550)
    assert r['tail_estimate_um'] is None
    assert r['tail_estimate_kind'] == 'initialization_only'


def test_constant_observed_curve_can_hold_but_prediction_only_tail_cannot():
    y = 100 + .2*np.sin(np.arange(31))
    r = extrapolate_cutoff(make_data(y), 550)
    assert r['tail_estimate_kind'] == 'near_zero_curve_hold'
    assert r['tail_estimate_um'] == y[-1]
    data = make_data(np.full(31, 100.), updates=False)
    data['transport_absolute_used'][0] = True
    r = extrapolate_cutoff(data, 550)
    assert r['tail_estimate_kind'] == 'insufficient_trend_hold_assumption'
    assert r['tail_window_min_um'] is None


def test_linear_trend_projects_to_requested_depth_without_changing_input():
    data = make_data(np.linspace(0, 90, 31))
    before = data['offset_um'].copy()
    r = extrapolate_cutoff(data, 550)
    assert r['tail_estimate_um'] == pytest.approx(165.)
    assert r['tail_estimate_kind'] == 'local_linear_extrapolation'
    np.testing.assert_array_equal(data['offset_um'], before)


def test_no_crossing_gap_to_obtain_a_trend_and_no_post_cutoff_samples():
    data = make_data(np.linspace(0, 90, 31))
    data['offset_um'][25:30] = np.nan
    assert extrapolate_cutoff(data, 550)['tail_estimate_kind'] == 'insufficient_trend_hold_assumption'
    data = make_data(np.linspace(0, 90, 31))
    data['offset_um'][21:] = 1.e6
    r = extrapolate_cutoff(data, 200)
    assert r['tail_estimate_um'] == pytest.approx(60.)
    assert r['tail_estimate_kind'] == 'available_at_cutoff'


def test_depth_weights_and_preserved_originals_without_endpoint_extrapolation():
    x = np.array([0, .1, .2, .4, .5, .8, 1.])
    y = np.array([np.nan, 100, np.nan, np.nan, 400, np.nan, np.nan])
    saved = y.copy()
    r = interpolate_internal_gaps(x, y)
    np.testing.assert_allclose(r['offset_completed_um'], [np.nan, 100, 175, 325, 400, np.nan, np.nan], equal_nan=True)
    np.testing.assert_array_equal(r['estimate_kind'], [0, 1, 2, 2, 1, 0, 0])
    np.testing.assert_array_equal(y, saved)
    assert r['gaps'][0]['span_mm'] == pytest.approx(.4)

def test_negative_trends_are_not_forced_monotone_or_clipped():
    r = interpolate_internal_gaps([0, 1, 2], [400, np.nan, -200])
    assert r['offset_completed_um'][1] == 100

def test_must_not_cross_cutoff_or_missing_depth():
    for x, eligible in (([0, 1, 2], [True, True, False]),
                        ([0, 1, 2], [True, False, True]),
                        ([0, np.nan, 2], [True, True, True]),
                        ([0, 3, 2], [True, True, True])):
        r = interpolate_internal_gaps(x, [10, np.nan, 30], eligible=eligible)
        assert np.isnan(r['offset_completed_um'][1])

def test_small_depth_reversals_do_not_sort_or_change_sample_identity():
    r = interpolate_internal_gaps([0, .6, .59, 1], [0, np.nan, np.nan, 100])
    np.testing.assert_allclose(r['offset_completed_um'], [0, 60, 59, 100])
    assert r['gaps'][0]['depth_backsteps'] == 1

@pytest.mark.parametrize('y', [[np.nan]*3, [1, np.nan, np.nan], [1, 2, 3]])
def test_no_two_sided_gap_keeps_every_value(y):
    r = interpolate_internal_gaps([0, 1, 2], y)
    np.testing.assert_array_equal(r['offset_completed_um'], y)
    assert not r['gaps']
