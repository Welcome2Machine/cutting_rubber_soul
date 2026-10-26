"""Conditional material-address transport with conservative bilateral fusion.

The latent state is [s - initial_s, ds/dt]. No compression, monotonicity,
sticking, or saturation assumption enters this model. Reported +/-2-sigma
ranges are model sensitivity envelopes, not validated confidence intervals.
"""
from __future__ import annotations

import numpy as np


def _psd(matrix):
    matrix = (matrix + matrix.T) / 2.
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 0.)) @ vectors.T


def _isolated_inconsistency(times, values, sigma, valid, noise, threshold, velocity):
    """Flag isolated contradictions, not persistent large changes in velocity.

    This is only supporting evidence for soft innovation downweighting. In
    particular, the first/last observation and a new sustained velocity are
    never classified as outliers solely because the prior predicts less slip.
    """
    flagged = np.zeros(len(times), bool)
    indices = np.flatnonzero(valid)
    for before, i, after in zip(indices[:-2], indices[1:-1], indices[2:]):
        span = times[after] - times[before]
        fraction = (times[i] - times[before]) / span
        predicted = (1. - fraction) * values[before] + fraction * values[after]
        process_variance = noise ** 2 * (span if velocity else span ** 3 / 3.)
        variance = (sigma[i] ** 2 + (1. - fraction) ** 2 * sigma[before] ** 2
                    + fraction ** 2 * sigma[after] ** 2 + process_variance)
        if velocity:
            # A step into a sustained new speed has disagreeing neighbours;
            # it must not be mistaken for an isolated observation failure.
            neighbour_variance = sigma[before] ** 2 + sigma[after] ** 2 + process_variance
            if abs(values[after] - values[before]) > threshold * np.sqrt(neighbour_variance):
                continue
        flagged[i] = abs(values[i] - predicted) > threshold * np.sqrt(variance)
    return flagged


def smooth_contact_transport(times, initial_s, initial_sigma, side_flux, side_sigma,
                             side_valid, absolute_s=None, absolute_sigma=None,
                             absolute_valid=None, *, acceleration_sigma,
                             process_noise_floor=1.e-6, initial_velocity_sigma=None,
                             incompatibility_z=3., robust_innovation_z=6.):
    """Fuse material-coordinate velocities and reliable absolute addresses.

    Args:
        times: Strictly increasing physical times, shape (T,).
        initial_s, initial_sigma: First-contact address and its separately
            reported sensitivity. The conditional offset is exactly zero at
            the first time. Baseline error is common, never repeatedly added
            as independent measurement noise at subsequent times.
        side_flux, side_sigma, side_valid: Arrays (T, 2). Flux is ds/dt,
            expressed with the same reference-surface orientation on both sides.
            Invalid/missing observations are ignored, never replaced by zero.
        absolute_s, absolute_sigma, absolute_valid: Optional arrays (T,).
            Values use the same reference coordinate as initial_s. These are
            conditional on the supplied baseline. The first-time absolute
            value is diagnostic only because the baseline is fixed.
        acceleration_sigma: Required square root of continuous white-
            acceleration spectral intensity. For s in px and time in seconds,
            units are px / second**1.5, not acceleration per image frame.
        process_noise_floor: Positive floor in the same units. The interval
            transition covariance is sigma**2 * [[dt**3/3, dt**2/2],
                                                [dt**2/2, dt]].
        initial_velocity_sigma: Optional uncertainty of initial velocity.
            Omission uses a diffuse, data-scaled prior, not a low-slip prior.
        robust_innovation_z: Positive threshold for soft variance inflation
            when an innovation ALSO contradicts adjacent observations. No
            velocity value is clipped; downweighted data remain in the result.
            Two mutually compatible sides confirm even a brief large velocity,
            so such a pulse is not temporally downweighted. Set to infinity
            to disable innovation downweighting for sensitivity analysis.

    Returns:
        Dictionary with forward and RTS states/covariances, offset and velocity
        sensitivity ranges, conservative fused observations, and quality flags.
        `offset_sigma` is CONDITIONAL on fixed initial_s; initial_sigma is
        returned separately. `reference_low/high` adds the baseline envelope
        by interval addition, without assuming independent baseline errors.
    """
    times = np.asarray(times, float)
    flux, sigma = np.asarray(side_flux, float), np.asarray(side_sigma, float)
    valid = np.asarray(side_valid, bool).copy()
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('times must be nonempty, finite, and strictly increasing')
    count = len(times)
    if flux.shape != (count, 2) or sigma.shape != flux.shape or valid.shape != flux.shape:
        raise ValueError('side arrays must have shape (len(times), 2)')
    if (not np.isfinite([initial_s, initial_sigma, acceleration_sigma, process_noise_floor,
                         incompatibility_z]).all() or np.isnan(robust_innovation_z)
            or initial_sigma < 0 or acceleration_sigma < 0 or process_noise_floor <= 0
            or incompatibility_z <= 0 or robust_innovation_z <= 0):
        raise ValueError('finite baseline and nonnegative noise scales are required; floor/thresholds must be positive')
    noise = max(float(acceleration_sigma), float(process_noise_floor))
    valid &= np.isfinite(flux) & np.isfinite(sigma) & (sigma > 0)
    side_count = valid.sum(axis=1)
    observed_flux = np.full(count, np.nan)
    observed_sigma = np.full(count, np.nan)
    disagreement = np.full(count, np.nan)
    disagreement_z = np.full(count, np.nan)
    incompatible = np.zeros(count, bool)
    flux_range = np.full((count, 2), np.nan)
    for i in np.flatnonzero(side_count):
        values, errors = flux[i, valid[i]], sigma[i, valid[i]]
        spread = float(np.ptp(values))
        observed_flux[i] = float(np.median(values))
        # Correlated same-direction errors must not get a sqrt(2) improvement.
        observed_sigma[i] = max(float(errors.max()), spread / 2.)
        flux_range[i] = [np.min(values - 2. * errors), np.max(values + 2. * errors)]
        if len(values) == 2:
            disagreement[i] = spread
            disagreement_z[i] = spread / np.linalg.norm(errors)
            incompatible[i] = disagreement_z[i] > incompatibility_z
    velocity_valid = side_count > 0
    if absolute_s is None:
        if absolute_sigma is not None or absolute_valid is not None:
            raise ValueError('absolute_s is required when absolute diagnostics are supplied')
        absolute = np.full(count, np.nan)
        address_sigma = np.full(count, np.nan)
        address_valid = np.zeros(count, bool)
    else:
        if absolute_sigma is None or absolute_valid is None:
            raise ValueError('absolute_s, absolute_sigma, and absolute_valid must be supplied together')
        absolute = np.asarray(absolute_s, float)
        address_sigma = np.asarray(absolute_sigma, float)
        address_valid = np.asarray(absolute_valid, bool).copy()
        if absolute.shape != (count,) or address_sigma.shape != (count,) or address_valid.shape != (count,):
            raise ValueError('absolute arrays must have shape (len(times),)')
        address_valid &= np.isfinite(absolute) & np.isfinite(address_sigma) & (address_sigma > 0)
    initial_absolute_residual = float(absolute[0] - initial_s) if address_valid[0] else np.nan
    address_valid[0] = False
    absolute_offset = absolute - initial_s
    if initial_velocity_sigma is None:
        data_scale = max(1., float(np.max(abs(observed_flux[velocity_valid]), initial=0.)),
                         float(np.max(observed_sigma[velocity_valid], initial=0.)))
        initial_velocity_sigma = 1000. * data_scale
    if not np.isfinite(initial_velocity_sigma) or initial_velocity_sigma <= 0:
        raise ValueError('initial_velocity_sigma must be positive and finite')
    isolated_flux = _isolated_inconsistency(times, observed_flux, observed_sigma,
        velocity_valid, noise, robust_innovation_z, True)
    # A bilateral, mutually compatible pulse may be genuine stick-slip. It
    # must not be rejected merely because it is short or departs from a prior.
    isolated_flux &= (side_count < 2) | incompatible
    isolated_address = _isolated_inconsistency(times, absolute_offset, address_sigma,
        address_valid, noise, robust_innovation_z, False)
    predicted = np.zeros((count, 2))
    predicted_cov = np.zeros((count, 2, 2))
    filtered = np.zeros((count, 2))
    filtered_cov = np.zeros((count, 2, 2))
    innovation = np.full((count, 2), np.nan)  # columns: address, velocity
    innovation_z = np.full((count, 2), np.nan)
    noise_multiplier = np.ones((count, 2))
    x = np.zeros(2)
    covariance = np.diag([0., float(initial_velocity_sigma) ** 2])
    transitions = np.repeat(np.eye(2)[None], count, axis=0)
    for i in range(count):
        if i:
            dt = float(times[i] - times[i - 1])
            A = np.array([[1., dt], [0., 1.]])
            Q = noise ** 2 * np.array([[dt ** 3 / 3., dt ** 2 / 2.], [dt ** 2 / 2., dt]])
            transitions[i] = A
            x, covariance = A @ x, _psd(A @ covariance @ A.T + Q)
        predicted[i], predicted_cov[i] = x, covariance
        observations = ((0, absolute_offset[i], address_sigma[i], address_valid[i], isolated_address[i]),
                        (1, observed_flux[i], observed_sigma[i], velocity_valid[i], isolated_flux[i]))
        for component, value, error, available, isolated in observations:
            if not available:
                continue
            residual = float(value - x[component])
            R = float(error ** 2)
            total = float(covariance[component, component] + R)
            standardized = residual / np.sqrt(total)
            inflation = max(1., (abs(standardized) / robust_innovation_z) ** 2) if isolated else 1.
            R *= inflation
            K = covariance[:, component] / (covariance[component, component] + R)
            x = x + K * residual
            # Joseph update avoids subtracting near-equal covariance matrices.
            I_KH = np.eye(2)
            I_KH[:, component] -= K
            covariance = _psd(I_KH @ covariance @ I_KH.T + R * np.outer(K, K))
            innovation[i, component], innovation_z[i, component] = residual, standardized
            noise_multiplier[i, component] = inflation
        filtered[i], filtered_cov[i] = x, covariance
    smoothed, smoothed_cov = filtered.copy(), filtered_cov.copy()
    for i in range(count - 2, -1, -1):
        cross = filtered_cov[i] @ transitions[i + 1].T
        try:
            gain = np.linalg.solve(predicted_cov[i + 1], cross.T).T
        except np.linalg.LinAlgError:
            gain = cross @ np.linalg.pinv(predicted_cov[i + 1])
        smoothed[i] = filtered[i] + gain @ (smoothed[i + 1] - predicted[i + 1])
        smoothed_cov[i] = _psd(filtered_cov[i] + gain @ (smoothed_cov[i + 1] - predicted_cov[i + 1]) @ gain.T)
    # Conditioning on the datum means its relative displacement has zero
    # variance, even when future measurements revise the initial velocity.
    smoothed[0, 0] = filtered[0, 0] = 0.
    smoothed_cov[0, 0, :] = smoothed_cov[0, :, 0] = 0.
    filtered_cov[0, 0, :] = filtered_cov[0, :, 0] = 0.
    offset, velocity = smoothed.T
    offset_sigma = np.sqrt(smoothed_cov[:, 0, 0])
    velocity_sigma = np.sqrt(smoothed_cov[:, 1, 1])
    return dict(times=times.copy(), offset=offset, offset_sigma=offset_sigma,
        offset_low=offset - 2. * offset_sigma, offset_high=offset + 2. * offset_sigma,
        reference_s=initial_s + offset,
        reference_low=initial_s + offset - 2. * (offset_sigma + initial_sigma),
        reference_high=initial_s + offset + 2. * (offset_sigma + initial_sigma),
        velocity=velocity, velocity_sigma=velocity_sigma,
        velocity_low=velocity - 2. * velocity_sigma, velocity_high=velocity + 2. * velocity_sigma,
        filtered_state=filtered, filtered_covariance=filtered_cov,
        predicted_state=predicted, predicted_covariance=predicted_cov,
        smoothed_state=smoothed, smoothed_covariance=smoothed_cov,
        fused_flux=observed_flux, fused_sigma=observed_sigma, fused_flux_range=flux_range,
        side_used=valid, side_count=side_count, side_disagreement=disagreement,
        side_disagreement_z=disagreement_z, side_incompatible=incompatible,
        absolute_used=address_valid, velocity_used=velocity_valid,
        prediction_only=~(address_valid | velocity_valid),
        innovation=innovation, innovation_z=innovation_z,
        innovation_noise_multiplier=noise_multiplier,
        isolated_velocity_inconsistency=isolated_flux,
        isolated_absolute_inconsistency=isolated_address,
        initial_s=float(initial_s), baseline_sigma=float(initial_sigma),
        initial_absolute_residual=initial_absolute_residual,
        initial_velocity_sigma=float(initial_velocity_sigma),
        acceleration_sigma=float(acceleration_sigma), effective_acceleration_sigma=noise,
        baseline_fixed=True, temporal_monotonicity_imposed=False,
        uncertainty_kind='conditional_model_sensitivity_not_statistical_confidence_interval',
        baseline_uncertainty_note='offset bands condition on fixed initial_s; reference bands add the separate baseline envelope',
        state_columns=('offset_from_initial_s', 'reference_address_velocity'))
