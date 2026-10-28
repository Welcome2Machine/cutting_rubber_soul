"""Optional fresh-residual Newton/Levenberg globalization, with host CCD/J.

This changes directions, never physical forces, time, energies or histories.
Residual minimization may stagnate at a nonzero stationary point; that is a
failed load, not equilibrium. A root need not be dynamically stable.
"""
import json
import time
import warnings

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve


def directions(jacobian, residual):
    """Newton first, then column-scaled normal-equation LM descent directions.

    Normal equations square conditioning. Explicit descent/linear residual
    checks and the unchanged force gate prevent a bad solve being accepted.
    Regularization is dimensionless in column-normalized coordinates.
    """
    a = jacobian.tocsc()
    r = np.asarray(residual, float)
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        try:
            p = spsolve(a, -r)
            if np.isfinite(p).all() and np.linalg.norm(a@p+r) <= .05*np.linalg.norm(r):
                yield 'newton', p
        except (MatrixRankWarning, RuntimeError):
            pass
    scale = 1./np.maximum(np.sqrt(np.asarray(a.power(2).sum(axis=0)).ravel()), 1e-12)
    b = a@diags(scale)
    normal = (b.T@b).tocsc()
    rhs = -b.T@r
    for damping in (1e-8, 1e-6, 1e-4, 1e-2, 1., 100.):
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            try:
                matrix = normal+diags(np.full(a.shape[1], damping))
                z = spsolve(matrix, rhs)
            except (MatrixRankWarning, RuntimeError):
                continue
        p = scale*z
        if (np.isfinite(p).all() and float(r@(a@p)) < 0.
                and np.linalg.norm(matrix@z-rhs) <= 1e-6*max(np.linalg.norm(rhs), 1e-15)):
            yield f'lm_{damping:g}', p


def solve(problem, x, depth, deadline):
    from .ipc_reference import BASELINE, LoadStepFailure
    tolerance = getattr(problem, 'force_tolerance_n', BASELINE['force_tolerance_n'])
    if not 0 < tolerance <= BASELINE['force_tolerance_n']:
        raise ValueError('Specialized force acceptance may only be tightened')
    counters = dict(method='newton_lm', iterations=0, trials=0, newton_updates=0,
                    lm_updates=0, tolerance_n=tolerance, physical_damping=False)
    problem.force_globalization_summary = counters
    primary = hasattr(problem, 'set_primary_increment')
    increment = problem.step_increment(x).copy() if primary else None
    for iteration in range(problem.max_iterations+1):
        if time.monotonic() >= deadline:
            raise TimeoutError('Fresh-residual Newton/LM wall budget')
        problem.trial = x.copy()
        if primary:
            problem.set_primary_increment(x, increment)
        lag = problem.lag(x)
        _, gradient, hessian, minj = problem.evaluate(x, lag, projected=False)
        if hasattr(problem, 'update_constraint_free'):
            problem.update_constraint_free(x, gradient)
        residual = gradient[problem.free]
        norm = float(np.linalg.norm(residual))
        counters.update(iterations=iteration, residual_n=norm)
        print('FORCE_NEWTON_LM '+json.dumps(dict(depth_mm=float(depth), **counters)), flush=True)
        if not np.isfinite(norm) or minj <= 0:
            raise LoadStepFailure('Invalid fresh Newton/LM state')
        if norm <= tolerance:
            return x, iteration
        if iteration == problem.max_iterations:
            break
        a = problem.coupled_friction_tangent(x, lag, hessian)[problem.free][:, problem.free]
        accepted = False
        for name, p in directions(a, residual):
            if time.monotonic() >= deadline:
                raise TimeoutError('Fresh-residual Newton/LM wall budget')
            slope = float(residual@(a@p))
            if not np.isfinite(slope) or slope >= 0.:
                continue
            direction = np.zeros(problem.total)
            direction[problem.free] = p
            direction = direction.reshape(x.shape)
            maximum = float(np.max(np.abs(p)))
            if maximum == 0.:
                continue
            # One complete host CCD query; every subsequent trial is its prefix.
            _, alpha, safe = problem._ptc_trial(x, direction, maximum)
            for cut in range(21):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Fresh-residual Newton/LM wall budget')
                if primary:
                    trial_increment = increment+alpha*direction
                    trial = problem.previous+trial_increment
                    if np.array_equal(trial_increment, increment):
                        break
                    problem.set_primary_increment(trial, trial_increment)
                else:
                    trial = x+alpha*direction
                    if np.array_equal(trial, x):
                        break
                _, gt, _, jt = problem.evaluate(trial, problem.lag(trial), tangent=False)
                nt = float(np.linalg.norm(gt[problem.free]))
                counters['trials'] += 1
                if (np.isfinite(nt) and jt > 0 and
                        (nt <= tolerance or nt**2 <= norm**2+2e-4*alpha*slope)):
                    counters['newton_updates' if name == 'newton' else 'lm_updates'] += 1
                    counters.update(last_direction=name, last_alpha=float(alpha),
                                    last_ccd_j_alpha=float(safe), last_backtracks=cut)
                    x = trial
                    if primary:
                        increment = trial_increment
                    accepted = True
                    break
                alpha *= .5
            if accepted:
                break
        if not accepted:
            if primary:
                problem.set_primary_increment(x, increment)
            raise LoadStepFailure(f'Fresh-residual Newton/LM stagnation at {depth:g} mm: {norm:g} N')
    raise LoadStepFailure(f'Fresh-residual Newton/LM iteration limit at {depth:g} mm')
