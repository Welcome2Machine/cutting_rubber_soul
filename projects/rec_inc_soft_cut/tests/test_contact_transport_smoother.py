import numpy as np
import pytest

from src.analysis.contact_transport_smoother import smooth_contact_transport


def run(times, speed, *, valid=None, sigma=.1, **kwargs):
    speed = np.broadcast_to(np.asarray(speed, float), (len(times),))
    flux = np.column_stack((speed, speed))
    valid = np.ones_like(flux, bool) if valid is None else valid
    return smooth_contact_transport(times, 20., .7, flux, np.full_like(flux, sigma), valid,
        acceleration_sigma=kwargs.pop('acceleration_sigma', .3), **kwargs)


def test_stationary_material_address_and_exact_baseline():
    result = run(np.arange(20.) * .1, 0.)
    assert np.allclose(result['offset'], 0.)
    assert result['offset_sigma'][0] == 0.
    assert result['offset_low'][0] == result['offset_high'][0] == 0.
    assert result['reference_low'][0] == 20. - 2. * .7
    assert result['baseline_sigma'] == .7
    assert result['uncertainty_kind'].endswith('not_statistical_confidence_interval')


@pytest.mark.parametrize('velocity', [-7., 4., 10000.])
def test_known_signed_transport_is_not_clipped_or_made_monotone(velocity):
    times = np.r_[0., .13, .4, .65, 1., 1.8, 2.]
    result = run(times, velocity)
    assert np.allclose(result['offset'], velocity * times, atol=1.e-4)
    assert np.allclose(result['velocity'], velocity, atol=1.e-4)
    assert not result['temporal_monotonicity_imposed']


def test_agreeing_sides_do_not_get_independent_sqrt_two_precision():
    result = run(np.arange(5.), 3., sigma=2.)
    assert np.all(result['fused_sigma'] == 2.)
    assert not result['side_incompatible'].any()


def test_bilateral_disagreement_inflates_uncertainty_and_is_preserved():
    times = np.arange(6.)
    flux = np.tile([-5., 9.], (len(times), 1))
    result = smooth_contact_transport(times, 0., 0., flux, np.ones_like(flux),
        np.ones_like(flux, bool), acceleration_sigma=.5)
    assert np.all(result['fused_flux'] == 2.)
    assert np.all(result['fused_sigma'] == 7.)
    assert result['side_incompatible'].all()
    assert np.all(result['fused_flux_range'] == [-7., 11.])


def test_missing_data_only_predicts_and_forward_uncertainty_grows():
    times = np.arange(8.)
    valid = np.zeros((len(times), 2), bool)
    valid[0] = True
    result = run(times, 3., valid=valid)
    assert result['prediction_only'][1:].all()
    assert np.isnan(result['fused_flux'][1:]).all()
    assert np.allclose(result['offset'], 3. * times, atol=1.e-5)
    assert np.all(np.diff(result['filtered_covariance'][1:, 0, 0]) > 0.)
    assert np.all(np.diff(result['filtered_covariance'][1:, 1, 1]) > 0.)
    assert np.allclose(result['smoothed_state'], result['filtered_state'])


def test_returning_absolute_address_reanchors_and_smooths_hidden_interval():
    times = np.arange(11.)
    valid = np.zeros((len(times), 2), bool)
    valid[0] = True
    absent = run(times, 1., valid=valid)
    addresses = np.full(len(times), np.nan)
    addresses[-1] = 40.
    av = np.isfinite(addresses)
    result = run(times, 1., valid=valid, absolute_s=addresses,
        absolute_sigma=np.full(len(times), .1), absolute_valid=av)
    assert abs(result['reference_s'][-1] - 40.) < .05
    assert result['offset_sigma'][-1] < absent['offset_sigma'][-1]
    assert result['offset_sigma'][5] < absent['offset_sigma'][5]
    assert result['absolute_used'][-1]
    assert result['prediction_only'][1:-1].all()


def test_positive_and_negative_flow_remain_possible_throughout_record():
    times = np.arange(21.) * .1
    speed = 5. * np.cos(times * np.pi)
    result = run(times, speed, sigma=.01, acceleration_sigma=5.)
    assert result['velocity'].min() < -4.9
    assert result['velocity'].max() > 4.9
    assert np.diff(result['offset']).min() < -.3
    assert np.diff(result['offset']).max() > .3


def test_persistent_large_velocity_change_is_not_an_innovation_outlier():
    times = np.arange(20.)
    speed = np.r_[np.zeros(7), np.full(13, 50.)]
    result = run(times, speed, sigma=.2, acceleration_sigma=5.)
    assert not result['isolated_velocity_inconsistency'].any()
    assert np.all(result['innovation_noise_multiplier'][:, 1] == 1.)
    assert result['velocity'][-1] > 49.


def test_isolated_bad_observation_is_softly_downweighted_not_replaced():
    times = np.arange(15.)
    speed = np.ones(len(times))
    speed[7] = 200.
    valid = np.ones((len(times), 2), bool)
    valid[:, 1] = False
    result = run(times, speed, valid=valid, sigma=.1, acceleration_sigma=.1)
    assert result['fused_flux'][7] == 200.
    assert result['isolated_velocity_inconsistency'][7]
    assert result['innovation_noise_multiplier'][7, 1] > 1.
    assert result['velocity'][7] < 5.


def test_bilateral_brief_slip_pulse_is_not_rejected_as_an_outlier():
    times = np.arange(15.)
    speed = np.zeros(len(times))
    speed[7] = 200.
    result = run(times, speed, sigma=.1, acceleration_sigma=1.)
    assert not result['isolated_velocity_inconsistency'][7]
    assert result['innovation_noise_multiplier'][7, 1] == 1.
    assert result['velocity'][7] > 195.


def test_robust_downweighting_can_be_disabled_for_sensitivity_analysis():
    times = np.arange(7.)
    speed = np.ones(len(times))
    speed[3] = 50.
    valid = np.ones((len(times), 2), bool)
    valid[:, 1] = False
    result = run(times, speed, valid=valid, robust_innovation_z=np.inf)
    assert np.all(result['innovation_noise_multiplier'] == 1.)


def test_nonfinite_invalid_inputs_cannot_turn_missing_flux_into_zero():
    times = np.arange(5.)
    flux = np.full((5, 2), np.nan)
    flux[0, 0] = -2.
    sigma = np.full_like(flux, .1)
    result = smooth_contact_transport(times, 0., 0., flux, sigma,
        np.ones_like(flux, bool), acceleration_sigma=0., process_noise_floor=.01)
    assert result['side_count'].tolist() == [1, 0, 0, 0, 0]
    assert np.allclose(result['offset'], -2. * times, atol=1.e-5)
    assert result['effective_acceleration_sigma'] == .01


def test_first_absolute_observation_does_not_move_the_fixed_datum():
    times = np.arange(3.)
    result = run(times, 0., absolute_s=np.array([50., np.nan, np.nan]),
                 absolute_sigma=np.ones(3), absolute_valid=np.array([True, False, False]))
    assert result['initial_absolute_residual'] == 30.
    assert not result['absolute_used'][0]
    assert result['offset'][0] == 0.


def test_invalid_times_and_side_shapes_are_rejected():
    with pytest.raises(ValueError, match='strictly increasing'):
        run(np.array([0., 0., 1.]), 0.)
    with pytest.raises(ValueError, match='side arrays'):
        smooth_contact_transport(np.arange(3.), 0., 0., np.zeros((3, 1)),
            np.ones((3, 1)), np.ones((3, 1), bool), acceleration_sigma=.1)
