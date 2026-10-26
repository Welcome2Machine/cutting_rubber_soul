import numpy as np
import pytest

from src.analysis.contact_transport_reference import smooth_contact_transport


def run(times, velocity, absolute=None, av=None, vv=None, initial_sigma=2., **kwargs):
    flux = np.repeat(np.broadcast_to(np.asarray(velocity, float), (len(times),))[:, None], 2, axis=1)
    valid = np.ones_like(flux, bool) if vv is None else vv
    if absolute is not None:
        absolute = np.broadcast_to(np.asarray(absolute, float), (len(times),))
        av = np.isfinite(absolute) if av is None else av
    return smooth_contact_transport(times, 0., initial_sigma, flux, np.full_like(flux, .1), valid,
        absolute_s=absolute, absolute_sigma=None if absolute is None else np.full(len(times), .2),
        absolute_valid=av, acceleration_sigma=kwargs.pop('acceleration_sigma', .3), **kwargs)


def test_constant_absolute_bias_is_not_spurious_slip():
    t = np.arange(31.)*.1
    result = run(t, 0., absolute=-6.)
    assert np.allclose(result['offset'], 0., atol=1.e-11)
    assert np.allclose(result['bias'], -6., atol=1.e-11)
    assert np.allclose(result['velocity'], 0., atol=1.e-11)
    assert result['bias_initialized_from_first_absolute']
    assert result['absolute_used'][0]
    assert result['offset_sigma'][0] == 0.
    assert result['smoothed_state'].shape == (len(t), 3)


@pytest.mark.parametrize('speed', [-7., 4., 10000.])
def test_real_signed_slip_is_retained_in_presence_of_constant_absolute_bias(speed):
    t = np.r_[0., .13, .4, .65, 1., 1.8, 2.]
    result = run(t, speed, absolute=speed*t-6.)
    assert np.allclose(result['offset'], speed*t, atol=1.e-4)
    assert np.allclose(result['bias'], -6., atol=1.e-4)
    assert np.allclose(result['velocity'], speed, atol=1.e-4)
    assert not result['temporal_monotonicity_imposed']


def test_first_late_absolute_does_not_reset_real_slip_to_zero():
    t = np.arange(11.)
    absolute = np.full(len(t), np.nan)
    absolute[5:] = 2.*t[5:]
    result = run(t, 2., absolute=absolute)
    assert not result['bias_initialized_from_first_absolute']
    assert not result['absolute_used'][0]
    assert result['initial_bias'] == 0.
    assert result['initial_bias_sigma'] == 2.
    assert np.allclose(result['offset'], 2.*t, atol=1.e-4)
    assert abs(result['bias'][-1]) < 1.e-4


def test_late_absolute_recovers_slip_when_flux_missing_in_hidden_interval():
    t = np.arange(11.)
    vv = np.zeros((len(t), 2), bool)
    vv[0] = True
    absolute = np.full(len(t), np.nan)
    absolute[-1] = 20.
    result = run(t, 1., absolute=absolute, vv=vv, initial_sigma=.1)
    assert result['offset'][-1] > 19.
    assert result['offset'][0] == 0.
    assert not result['bias_initialized_from_first_absolute']
    assert result['prediction_only'][1:-1].all()


def test_missing_both_data_types_predicts_and_forward_uncertainty_grows():
    t = np.arange(8.)
    vv = np.zeros((len(t), 2), bool)
    vv[0] = True
    absolute = np.full(len(t), np.nan)
    absolute[0] = -6.
    result = run(t, 3., absolute=absolute, vv=vv)
    assert np.allclose(result['offset'], 3.*t, atol=1.e-5)
    assert result['prediction_only'][1:].all()
    assert np.all(np.diff(result['filtered_covariance'][1:, 0, 0]) > 0.)
    assert np.allclose(result['bias'], -6.)
    assert np.allclose(result['smoothed_state'], result['filtered_state'])


def test_bilateral_disagreement_is_not_treated_as_independent_precision():
    t = np.arange(5.)
    flux = np.tile([-5., 9.], (len(t), 1))
    result = smooth_contact_transport(t, 0., 1., flux, np.ones_like(flux),
        np.ones_like(flux, bool), acceleration_sigma=.5)
    assert np.all(result['fused_sigma'] == 7.)
    assert np.all(result['fused_flux'] == 2.)
    assert result['side_incompatible'].all()


def test_same_frame_absolute_and_flux_use_covariance_upper_bound():
    t = np.arange(7.)
    result = run(t, 0., absolute=-6.)
    assert np.all(result['shared_observation_noise_multiplier'] == 2.)
    assert np.allclose(result['effective_observation_variance'][:, 0], 2.*.2**2)
    assert np.allclose(result['effective_observation_variance'][:, 1], 2.*.1**2)
    assert result['initial_bias_sigma'] == pytest.approx(np.sqrt(2.)*.2)
    # The shared nuisance state induces correlations with displacement; it is
    # not repeatedly added as independent uncertainty to each absolute datum.
    assert (result['smoothed_covariance'][1:, 0, 2] < 0.).all()
    assert np.allclose(result['bias_sigma'], result['bias_sigma'][0])


def test_first_absolute_is_consumed_once_not_twice():
    t = np.array([0.])
    vv = np.zeros((1, 2), bool)
    result = run(t, 0., absolute=-6., vv=vv)
    assert result['bias'][0] == -6.
    assert result['bias_sigma'][0] == pytest.approx(.2)
    assert result['offset'][0] == result['offset_sigma'][0] == 0.


def test_zero_external_baseline_sigma_without_first_absolute_is_well_conditioned():
    t = np.arange(5.)
    absolute = np.arange(5.)*2.
    av = np.ones(5, bool)
    av[0] = False
    result = run(t, 2., absolute=absolute, av=av, initial_sigma=0.)
    assert np.allclose(result['offset'], 2.*t, atol=1.e-5)
    assert np.allclose(result['bias'], 0.)
    assert np.isfinite(result['smoothed_covariance']).all()


def test_shared_bias_covariance_is_positive_semidefinite_and_datum_exact():
    t = np.arange(21.)*.1
    speed = 3.*np.cos(t*np.pi)
    absolute = 3.*np.sin(t*np.pi)/np.pi-6.
    result = run(t, speed, absolute=absolute, acceleration_sigma=5.)
    assert np.min(np.linalg.eigvalsh(result['smoothed_covariance'])) > -1.e-10
    assert result['offset'][0] == result['offset_sigma'][0] == 0.
    assert result['velocity'].min() < -2.9
    assert result['velocity'].max() > 2.9
