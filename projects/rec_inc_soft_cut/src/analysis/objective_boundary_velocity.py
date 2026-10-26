"""Curved-boundary velocity fit with an explicit shared rigid-rotation mode.

Local tangential/normal polynomial components alone cannot represent rigid
rotation on a curved boundary. One shared Cartesian rotation mode completes
the model without treating angular motion as reference-surface transport.
Redundant coefficients are allowed only when the requested contact velocity
is identifiable. Uncertainty remains a model sensitivity scale.
"""
from __future__ import annotations
import numpy as np


def _design(X, relative_xy, tangent, normal):
    n, p = X.shape
    Z = np.zeros((n, 2, 2*p+1))
    Z[:, 0, :p] = X
    Z[:, 1, p:2*p] = X
    rotation = np.column_stack((-relative_xy[:, 1], relative_xy[:, 0]))/60.
    Z[:, 0, -1] = np.sum(rotation*tangent, axis=1)
    Z[:, 1, -1] = np.sum(rotation*normal, axis=1)
    return Z


def fit_objective_velocity(X, current_relative_xy, tangent, normal, relative_velocity,
                           weights, C, contact_relative_xy, contact_tangent,
                           contact_normal, *, floor, maximum_condition):
    Z = _design(X, current_relative_xy, tangent, normal)
    E = _design(C[None], np.asarray(contact_relative_xy)[None],
        np.asarray(contact_tangent)[None], np.asarray(contact_normal)[None])[0]
    values = np.column_stack((np.sum(relative_velocity*tangent, axis=1),
                              np.sum(relative_velocity*normal, axis=1)))
    base = np.asarray(weights, float); w = base.copy()
    p = X.shape[1]
    if np.linalg.matrix_rank(X) < p:
        return None

    def solve(w):
        w = w/max(float(w.mean()), 1.e-12)
        A = (Z*np.sqrt(w)[:, None, None]).reshape(-1, Z.shape[-1])
        y = (values*np.sqrt(w)[:, None]).ravel()
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        tol = np.finfo(float).eps*max(A.shape)*s[0]*10.
        rank = int(np.sum(s > tol))
        if rank == 0:
            return None
        condition = float((s[0]/s[rank-1])**2)
        if condition > maximum_condition:
            return None
        # Nonunique coefficient decompositions are harmless only if contact
        # evaluation annihilates their exact nullspace. Never truncate a weak
        # but identified direction just to manufacture a stable contact value.
        null = Vt[rank:]
        ambiguity = float(np.linalg.norm(E@null.T, ord=2)) if len(null) else 0.
        if ambiguity > 1.e-8*max(float(np.linalg.norm(E, ord=2)), 1.):
            return None
        inverse = (Vt[:rank].T/(s[:rank]**2))@Vt[:rank]
        beta = Vt[:rank].T@((U[:, :rank].T@y)/s[:rank])
        return beta, inverse, rank, condition, ambiguity, w

    for _ in range(3):
        solution = solve(w)
        if solution is None:
            return None
        beta = solution[0]
        residual = np.linalg.norm(Z@beta-values, axis=1)
        scale = max(floor, 1.4826*float(np.median(abs(residual-np.median(residual)))))
        w = base*np.minimum(1., 2.5*scale/np.maximum(residual, 1.e-12))
    solution = solve(w)
    if solution is None:
        return None
    beta, inverse, rank, condition, ambiguity, w = solution
    residual = Z@beta-values
    rms = float(np.sqrt(np.average(np.sum(residual**2, axis=1), weights=w)))
    covariance_factor = E@inverse@E.T
    return dict(beta=np.column_stack((beta[:p], beta[p:2*p])), joint_beta=beta,
        rms=rms, weights=w, evaluation=E@beta,
        leverage=float(np.linalg.eigvalsh(covariance_factor).max()),
        angular_rate=float(beta[-1]/60.), rank=rank, columns=Z.shape[-1],
        condition=condition, contact_estimability_error=ambiguity)


def predict_objective_velocity(fit, X, current_relative_xy, tangent, normal):
    return _design(X, current_relative_xy, tangent, normal)@fit['joint_beta']


def objective_motion_holdout(X, current_relative_xy, tangent, normal, relative_velocity,
                             weights, depth, arc_distance, C, contact_relative_xy,
                             contact_tangent, contact_normal, cfg):
    errors = []
    values = np.column_stack((np.sum(relative_velocity*tangent, axis=1),
                              np.sum(relative_velocity*normal, axis=1)))
    for test in (depth <= np.quantile(depth, .3), arc_distance <= np.quantile(arc_distance, .25)):
        train = ~test
        if test.sum() < 3 or train.sum() < X.shape[1]+3:
            return np.inf
        fit = fit_objective_velocity(X[train], current_relative_xy[train], tangent[train],
            normal[train], relative_velocity[train], weights[train], C, contact_relative_xy,
            contact_tangent, contact_normal, floor=cfg.velocity_noise_floor_px_s,
            maximum_condition=cfg.maximum_condition)
        if fit is None:
            return np.inf
        prediction = predict_objective_velocity(fit, X[test], current_relative_xy[test], tangent[test], normal[test])
        errors.append(float(np.sqrt(np.average(np.sum((prediction-values[test])**2, axis=1), weights=weights[test]))))
    return max(errors)
