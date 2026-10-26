"""Behavioural contracts for both retained temporal evidence recipes."""
import numpy as np
import pytest
from src.analysis.contact_transport_filter import fuse_bounded, fuse_windowed


def bounded_inputs(velocity, dt=.025, sigma=.1):
    v = np.asarray(velocity, float)
    t = np.arange(len(v))*dt
    data = dict(times=t, initial=dict(contact_s_px=100.))
    obs = dict(side_flux_px_s=np.repeat(v[:, None], 2, axis=1),
        side_flux_sigma_px_s=np.full((len(v), 2), sigma), side_flux_valid=np.ones((len(v), 2), bool),
        absolute_s_px=np.full(len(v), np.nan), absolute_sigma_px=np.ones(len(v)),
        absolute_valid=np.zeros(len(v), bool))
    return data, obs

@pytest.mark.parametrize('sign', [-1, 1])
def test_signed_constant_motion_is_retained(sign):
    d, o = bounded_inputs(np.full(81, sign*10.))
    r = fuse_bounded(d, o, 5.)
    assert np.all(r['offset_available'])
    assert not r['holding'].any()
    assert r['offset'][-1] == pytest.approx(sign*20., abs=.15)

def test_zero_oscillation_holds_and_does_not_use_future_evidence():
    d, o = bounded_inputs(np.r_[np.full(40, 10.), .2*np.sin(np.arange(80))])
    r = fuse_bounded(d, o, 5.)
    held = np.flatnonzero(r['holding'])
    assert len(held) > 40
    assert np.ptp(r['offset'][held]) == 0.
    assert np.all(r['velocity'][held] == 0.)
    assert held[0] >= 40
    p = fuse_bounded(dict(d, times=d['times'][:50]), {k: v[:50] for k, v in o.items()}, 5.)
    np.testing.assert_allclose(p['offset'], r['offset'][:50])

def test_missing_motion_is_not_zero_and_forecast_is_bounded():
    d, o = bounded_inputs(np.full(100, -20.))
    o['side_flux_valid'][30:] = False
    r = fuse_bounded(d, o, 5.)
    assert not r['holding'].any()
    assert np.isnan(r['offset'][41:]).all()
    assert np.ptp(r['filtered_state'][41:, 0]) < 1e-10
    assert r['filtered_state'][-1, 0] > -25.

def test_supported_hold_persists_explicitly_across_missing_observations():
    d, o = bounded_inputs(np.r_[np.full(30, 10.), np.zeros(90)])
    o['side_flux_valid'][80:] = False
    r = fuse_bounded(d, o, 5.)
    assert (r['state'][80:] == 4).all()
    assert np.ptp(r['offset'][80:]) == 0.
    assert r['prediction_only'][80:].all()

@pytest.mark.parametrize('sign', [-1, 1])
def test_hold_releases_for_resolved_new_motion(sign):
    d, o = bounded_inputs(np.r_[np.full(30, 10.), np.zeros(50), np.full(50, sign*15.)])
    r = fuse_bounded(d, o, 5.)
    assert r['holding'][70]
    assert r['hold_released'][80]
    assert not r['holding'][80:].any()
    assert sign*(r['offset'][-1]-r['offset'][80]) > 15.

def test_large_opposite_or_high_uncertainty_velocities_cannot_establish_zero():
    d, o = bounded_inputs(np.full(80, 10.))
    o['side_flux_px_s'][:, 1] *= -1
    r = fuse_bounded(d, o, 5.)
    assert r['side_incompatible'].all()
    assert not r['holding'].any()
    d, o = bounded_inputs(np.zeros(80), sigma=100.)
    assert not fuse_bounded(d, o, 5.)['holding'].any()

def test_velocity_after_long_gap_does_not_restore_missing_displacement():
    d, o = bounded_inputs(np.full(110, 10.))
    o['side_flux_valid'][30:65] = False
    r = fuse_bounded(d, o, 5.)
    assert np.isnan(r['offset'][65:]).all()
    o['absolute_valid'][90:] = True
    o['absolute_s_px'][90:] = 100.+10*d['times'][90:]
    r = fuse_bounded(d, o, 5.)
    assert r['offset_available'][90:].all()

def test_unilateral_unresolved_sign_is_not_integrated():
    d, o = bounded_inputs(np.full(80, -10.), sigma=20.)
    o['side_flux_valid'][:, 0] = False
    r = fuse_bounded(d, o, 5.)
    assert r['weak_single_side_rejected'].all()
    assert not r['velocity_used'].any()

def windowed_inputs(velocity, sigma=.1, dt=.025):
    v = np.asarray(velocity, float)
    data = dict(times=np.arange(len(v))*dt, initial=dict(contact_s_px=100.))
    obs = dict(side_flux_px_s=np.repeat(v[:, None], 2, axis=1),
        side_flux_sigma_px_s=np.full((len(v), 2), sigma),
        side_flux_valid=np.ones((len(v), 2), bool),
        absolute_s_px=np.full(len(v), np.nan), absolute_sigma_px=np.ones(len(v)),
        absolute_valid=np.zeros(len(v), bool))
    return data, obs

@pytest.mark.parametrize('sign', [-1., 1.])
def test_weak_unilateral_motion_is_retained_without_sign_selection(sign):
    d, o = windowed_inputs(np.full(160, sign), sigma=10.)
    o['side_flux_valid'][:, 0] = False
    r = fuse_windowed(d, o, 5.)
    assert r['weak_single_side_retained'].all()
    assert r['velocity_used'].all()
    assert not r['holding'].any()
    assert r['offset'][-1] == pytest.approx(sign*d['times'][-1], abs=.02)

def test_slow_resolved_motion_is_not_frozen_just_because_under_pixel_window():
    d, o = windowed_inputs(np.full(160, .5), sigma=.1)
    r = fuse_windowed(d, o, 5.)
    assert not r['holding'].any()
    assert r['offset'][-1] > 1.8

def test_zero_centred_high_noise_is_not_evidence_of_sticking():
    d, o = windowed_inputs(np.zeros(120), sigma=20.)
    r = fuse_windowed(d, o, 5.)
    assert r['velocity_used'].all()
    assert not r['holding'].any()

def test_true_zero_after_motion_holds_without_fracture_input():
    d, o = windowed_inputs(np.r_[np.full(30, 10.), np.zeros(90)])
    r = fuse_windowed(d, o, 5.)
    held = np.flatnonzero(r['holding'])
    assert len(held) > 50
    assert np.ptp(r['offset'][held]) == 0.
    assert r['fracture_cutoff_required'] is False
    assert r['fracture_detected'] is False

@pytest.mark.parametrize('sign', [-1., 1.])
def test_gradual_small_restart_releases_hold(sign):
    d, o = windowed_inputs(np.r_[np.full(30, 10.), np.zeros(50), np.full(80, sign)])
    r = fuse_windowed(d, o, 5.)
    assert r['holding'][70]
    assert r['hold_released'][80:120].any()
    assert not r['holding'][-1]
    assert sign*(r['offset'][-1]-r['offset'][80]) > 1.

def test_gap_does_not_become_zero_and_local_trend_can_resume_without_absolute_datum():
    d, o = windowed_inputs(np.full(160, 10.))
    o['side_flux_valid'][30:80] = False
    r = fuse_windowed(d, o, 5.)
    assert not r['holding'].any()
    assert np.isnan(r['offset'][80:]).all()
    assert r['local_velocity_available'][80:].all()
    assert r['local_segment_id'][80] != r['local_segment_id'][29]
    assert r['local_segment_offset_px'][80] == 0.
    assert r['local_segment_offset_px'][-1] == pytest.approx(19.75)
    assert np.ptp(r['search_reference_s'][41:80]) < 1.e-8

def test_opposite_sides_cannot_cancel_into_zero():
    d, o = windowed_inputs(np.full(120, 10.))
    o['side_flux_px_s'][:, 1] *= -1
    r = fuse_windowed(d, o, 5.)
    assert r['side_incompatible'].all()
    assert not r['holding'].any()

def test_a_missing_side_at_window_end_does_not_make_a_zero_witness():
    d, o = windowed_inputs(np.zeros(120))
    o['side_flux_valid'][15:30, 1] = False
    r = fuse_windowed(d, o, 5.)
    assert not r['zero_evidence'][15:30].any()

def test_no_measurements_only_has_imposed_zero_and_short_prediction():
    d, o = windowed_inputs(np.zeros(120))
    o['side_flux_valid'][:] = False
    r = fuse_windowed(d, o, 5.)
    assert r['state'][0] == 5
    assert not (r['state'] == 2).any()
    assert not r['local_velocity_available'].any()
    assert not r['holding'].any()
    assert not r['offset_available'][-1]

def test_future_observations_do_not_change_past_fusion():
    d, o = windowed_inputs(np.r_[np.full(30, 10.), np.zeros(50), np.full(40, -15.)])
    full = fuse_windowed(d, o, 5.)
    for length in (25, 60, 91):
        prefix = fuse_windowed(dict(d, times=d['times'][:length]),
                      {key: value[:length] for key, value in o.items()}, 5.)
        for key in ('offset', 'state', 'zero_evidence', 'side_window_delta_px'):
            np.testing.assert_allclose(prefix[key], full[key][:length], equal_nan=True)
