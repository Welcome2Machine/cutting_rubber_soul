"""Baseline-aware material contact transport with a shared absolute-address bias.

State = [offset from first contact, address velocity, constant address bias].
The absolute observation is ``absolute_s - initial_s = offset + bias``. A
first-frame absolute mismatch therefore calibrates a shared bias; it is never
silently integrated as physical slip. Only offset[0] is fixed exactly to zero.
These envelopes quantify conditional model sensitivity, not statistical CIs.
"""
from __future__ import annotations

import numpy as np

from .contact_transport_smoother import (
    _isolated_inconsistency, _psd, smooth_contact_transport as _legacy_fusion,
)


def smooth_contact_transport(times, initial_s, initial_sigma, side_flux, side_sigma,
                             side_valid, absolute_s=None, absolute_sigma=None,
                             absolute_valid=None, *, acceleration_sigma,
                             process_noise_floor=1.e-6, initial_velocity_sigma=None,
                             incompatibility_z=3., robust_innovation_z=6.):
    """Fuse signed bilateral transport and absolute addresses without datum drift.

    Input units and arguments match ``contact_transport_smoother``. The white
    acceleration scale has units address/time**1.5, and contributes no process
    noise to the constant bias. If frame zero has an absolute observation, its
    residual initializes bias and its measurement uncertainty remains in the
    shared state covariance. This frame is not consumed a second time. Without
    that observation, bias has mean zero and prior sigma ``initial_sigma``;
    a later first absolute observation is an ordinary update, never a new datum.

    Flux and absolute estimates from the same frame may use the same tracked
    points. Whenever both are used, their measurement variances are multiplied
    by two. For any unknown correlation rho in [-1, 1], 2*diag(sigma**2) bounds
    the corresponding two-observation covariance in the PSD sense. This does
    not claim to model arbitrary temporal correlation or drifting model bias.

    Return keys match the previous smoother, with state/covariance dimension
    three and extra ``bias``/``bias_sigma`` diagnostics. ``offset_sigma``
    marginalizes over the shared bias, while the reference-address envelope
    also includes the separately reported external baseline uncertainty.
    """
    # Reuse the frozen implementation's validation and conservative bilateral
    # fusion. Its two-state trajectory is discarded in full, never blended in.
    result = _legacy_fusion(times, initial_s, initial_sigma, side_flux, side_sigma,
        side_valid, absolute_s, absolute_sigma, absolute_valid,
        acceleration_sigma=acceleration_sigma, process_noise_floor=process_noise_floor,
        initial_velocity_sigma=initial_velocity_sigma, incompatibility_z=incompatibility_z,
        robust_innovation_z=robust_innovation_z)
    times = result['times']
    count = len(times)
    noise = result['effective_acceleration_sigma']
    observed_flux, observed_sigma = result['fused_flux'], result['fused_sigma']
    velocity_valid = result['velocity_used']
    isolated_flux = result['isolated_velocity_inconsistency']
    address_valid = result['absolute_used'].copy()
    if absolute_s is None:
        absolute_offset = np.full(count, np.nan)
        address_sigma = np.full(count, np.nan)
    else:
        absolute_offset = np.asarray(absolute_s, float)-float(initial_s)
        address_sigma = np.asarray(absolute_sigma, float)
        address_valid[0] = bool(np.asarray(absolute_valid, bool)[0]
            and np.isfinite(absolute_offset[0]) and np.isfinite(address_sigma[0])
            and address_sigma[0] > 0.)
    paired = address_valid & velocity_valid
    shared_multiplier = np.where(paired, 2., 1.)
    first_absolute = bool(address_valid[0])
    bias_initial = float(absolute_offset[0]) if first_absolute else 0.
    bias_variance = (float(address_sigma[0])**2*shared_multiplier[0]
                     if first_absolute else float(initial_sigma)**2)
    isolated_address = _isolated_inconsistency(times, absolute_offset, address_sigma,
        address_valid, noise, robust_innovation_z, False)
    shape = (count, 3)
    predicted, filtered = np.zeros(shape), np.zeros(shape)
    predicted_cov = np.zeros((count, 3, 3))
    filtered_cov = np.zeros_like(predicted_cov)
    transitions = np.repeat(np.eye(3)[None], count, axis=0)
    innovation = np.full((count, 2), np.nan)
    innovation_z = np.full((count, 2), np.nan)
    robust_multiplier = np.ones((count, 2))
    observation_variance = np.full((count, 2), np.nan)
    x = np.array([0., 0., bias_initial])
    covariance = np.diag([0., result['initial_velocity_sigma']**2, bias_variance])
    address_H, velocity_H = np.array([1., 0., 1.]), np.array([0., 1., 0.])
    for i in range(count):
        if i:
            dt = float(times[i]-times[i-1])
            A = np.array([[1., dt, 0.], [0., 1., 0.], [0., 0., 1.]])
            Q = np.zeros((3, 3))
            Q[:2, :2] = noise**2*np.array([[dt**3/3., dt**2/2.], [dt**2/2., dt]])
            transitions[i] = A
            x, covariance = A @ x, _psd(A @ covariance @ A.T+Q)
        predicted[i], predicted_cov[i] = x, covariance
        observations = (
            (0, absolute_offset[i], address_sigma[i], address_valid[i], isolated_address[i], address_H),
            (1, observed_flux[i], observed_sigma[i], velocity_valid[i], isolated_flux[i], velocity_H),
        )
        for component, value, error, available, isolated, H in observations:
            if not available:
                continue
            if i == 0 and component == 0:
                # Already incorporated exactly once into the bias prior.
                observation_variance[i, component] = bias_variance
                continue
            residual = float(value-H @ x)
            R = float(error**2*shared_multiplier[i])
            variance = float(H @ covariance @ H+R)
            standardized = residual/np.sqrt(variance)
            inflation = max(1., (abs(standardized)/robust_innovation_z)**2) if isolated else 1.
            R *= inflation
            gain = covariance @ H/(H @ covariance @ H+R)
            x = x+gain*residual
            I_KH = np.eye(3)-np.outer(gain, H)
            covariance = _psd(I_KH @ covariance @ I_KH.T+R*np.outer(gain, gain))
            innovation[i, component], innovation_z[i, component] = residual, standardized
            robust_multiplier[i, component] = inflation
            observation_variance[i, component] = R
        filtered[i], filtered_cov[i] = x, covariance
    smoothed, smoothed_cov = filtered.copy(), filtered_cov.copy()
    for i in range(count-2, -1, -1):
        cross = filtered_cov[i] @ transitions[i+1].T
        try:
            gain = np.linalg.solve(predicted_cov[i+1], cross.T).T
        except np.linalg.LinAlgError:
            gain = cross @ np.linalg.pinv(predicted_cov[i+1])
        smoothed[i] = filtered[i]+gain @ (smoothed[i+1]-predicted[i+1])
        smoothed_cov[i] = _psd(filtered_cov[i]+gain @ (smoothed_cov[i+1]-predicted_cov[i+1]) @ gain.T)
    # The first-contact offset is a physical datum, not the first usable later
    # observation. Its exact zero persists through backward conditioning.
    for states, covariances in ((smoothed, smoothed_cov), (filtered, filtered_cov)):
        states[0, 0] = 0.
        covariances[0, 0, :] = covariances[0, :, 0] = 0.
    offset, velocity, bias = smoothed.T
    offset_sigma, velocity_sigma, bias_sigma = np.sqrt(
        np.maximum(np.diagonal(smoothed_cov, axis1=1, axis2=2), 0.)).T
    result.update(offset=offset, offset_sigma=offset_sigma,
        offset_low=offset-2.*offset_sigma, offset_high=offset+2.*offset_sigma,
        reference_s=float(initial_s)+offset,
        reference_low=float(initial_s)+offset-2.*(offset_sigma+float(initial_sigma)),
        reference_high=float(initial_s)+offset+2.*(offset_sigma+float(initial_sigma)),
        velocity=velocity, velocity_sigma=velocity_sigma,
        velocity_low=velocity-2.*velocity_sigma, velocity_high=velocity+2.*velocity_sigma,
        bias=bias, bias_sigma=bias_sigma, bias_low=bias-2.*bias_sigma, bias_high=bias+2.*bias_sigma,
        filtered_state=filtered, filtered_covariance=filtered_cov,
        predicted_state=predicted, predicted_covariance=predicted_cov,
        smoothed_state=smoothed, smoothed_covariance=smoothed_cov,
        absolute_used=address_valid, prediction_only=~(address_valid | velocity_valid),
        innovation=innovation, innovation_z=innovation_z,
        innovation_noise_multiplier=robust_multiplier,
        shared_observation_noise_multiplier=shared_multiplier,
        effective_observation_variance=observation_variance,
        isolated_absolute_inconsistency=isolated_address,
        bias_initialized_from_first_absolute=first_absolute,
        initial_bias=float(bias_initial), initial_bias_sigma=float(np.sqrt(bias_variance)),
        constant_absolute_bias=True, bias_process_noise=0.,
        absolute_bias_corrected_reference=np.asarray(absolute_offset)+float(initial_s)-bias,
        baseline_uncertainty_note=('offset bands marginalize the shared absolute-observation bias; '
            'reference bands additionally include the external first-contact baseline envelope'),
        state_columns=('offset_from_first_contact', 'reference_address_velocity', 'shared_absolute_address_bias'),
        observation_correlation_note=('same-frame absolute and flux variances are doubled when both are used; '
            'arbitrary temporal correlation and time-varying bias are not identified'))
    return result
