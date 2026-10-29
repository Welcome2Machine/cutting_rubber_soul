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


def solve_increment_ptc(problem, x, depth, deadline):
    """Fresh-force PTC in primary increments, for physical dynamic candidates.

    The model-error acceptance permits leaving a nonzero residual-merit minimum.
    The positive metric shift is only a linear solver device. Physical residual,
    time, mass, material and energy remain unchanged. Unlike the legacy PTC,
    sub-ulp position changes remain represented in the primary displacement.
    """
    from .ipc_reference import BASELINE, LoadStepFailure
    tolerance=problem.force_tolerance_n
    if not 0<tolerance<=BASELINE['force_tolerance_n']:
        raise ValueError('Force acceptance may only be tightened')
    increment=problem.step_increment(x).copy();shift=.1
    counters=dict(method='primary_increment_ptc',iterations=0,trials=0,physical_damping=False)
    problem.force_globalization_summary=counters
    for iteration in range(problem.max_iterations+1):
        if time.monotonic()>=deadline:raise TimeoutError('Primary-increment PTC wall budget')
        problem.set_primary_increment(x,increment);problem.trial=x.copy()
        lag=problem.lag(x)
        _,g,h,j,metric=problem.evaluate(x,lag,projected=False,with_metric=True)
        problem.update_constraint_free(x,g);r=g[problem.free];norm=float(np.linalg.norm(r))
        counters.update(iterations=iteration,residual_n=norm,metric_shift=shift)
        print('INCREMENT_PTC '+json.dumps(dict(depth_mm=float(depth),**counters)),flush=True)
        if not np.isfinite(norm) or j<=0:raise LoadStepFailure('Invalid increment PTC state')
        if norm<=tolerance:return x,iteration
        if iteration==problem.max_iterations:break
        a=problem.coupled_friction_tangent(x,lag,h)[problem.free][:,problem.free]
        m=metric[problem.free][:,problem.free]+diags(np.full(len(r),BASELINE['hessian_regularization_n_mm']))
        for retry in range(18):
            if time.monotonic()>=deadline:raise TimeoutError('Primary-increment PTC wall budget')
            with warnings.catch_warnings():
                warnings.simplefilter('error',MatrixRankWarning)
                try:direction_free=spsolve(a+shift*m,-r)
                except (MatrixRankWarning,RuntimeError):shift=max(.1,2*shift);continue
            maximum=float(np.max(np.abs(direction_free)))
            if not np.isfinite(direction_free).all() or maximum==0:
                shift=max(.1,2*shift);continue
            direction=np.zeros(problem.total);direction[problem.free]=direction_free
            direction=direction.reshape(x.shape)
            _,alpha,safe=problem._ptc_trial(x,direction,maximum)
            trial_increment=increment+alpha*direction
            trial=problem.previous+trial_increment
            # Preserve exact unilateral placement when arithmetic rounds past
            # the plane. Only coordinates rounded below the safe boundary move.
            below=trial[:problem.n,0]<problem.config.drop_s_mm
            trial[:problem.n,0][below]=problem.config.drop_s_mm
            trial_increment[:problem.n,0][below]=problem.config.drop_s_mm-problem.previous[:problem.n,0][below]
            problem.set_primary_increment(trial,trial_increment)
            _,gt,_,jt=problem.evaluate(trial,problem.lag(trial),tangent=False)
            rt=gt[problem.free];nt=float(np.linalg.norm(rt))
            error=float(np.linalg.norm(rt-r-alpha*(a@direction_free)))
            counters['trials']+=1
            if (jt>0 and np.isfinite(nt) and not np.array_equal(trial_increment,increment)
                    and (nt<=tolerance or error<=.25*max(norm,nt))):
                x,increment=trial,trial_increment
                counters.update(last_alpha=float(alpha),last_ccd_j_alpha=float(safe),last_model_error_n=error)
                shift=max(1e-10,min(1e6,shift*nt/max(norm,1e-15)/1.2))
                break
            shift=max(.1,2*shift)
        else:
            problem.set_primary_increment(x,increment)
            raise LoadStepFailure(f'Primary-increment PTC model mismatch at {depth:g} mm: {norm:g} N')
    raise LoadStepFailure(f'Primary-increment PTC iteration limit at {depth:g} mm')


def solve_friction_continuation(problem, x, depth, deadline):
    """Solver-only smoothing continuation, always ending with original friction.

    A physical dt and all committed histories remain fixed. Intermediate
    smoothing values are neither material parameters nor accepted energies.
    Direct LM/PTC is tried first for slow steps; fast stick/slip transitions
    use a smooth-to-original force homotopy. All iterations retain full CCD/J.
    """
    from .ipc_reference import LoadStepFailure
    friction=problem.friction;maximum=problem.max_iterations
    seed=x.copy();seed_increment=problem.step_increment(x).copy()
    phases=[];total=0
    problem.friction_continuation_summary=dict(original_friction_restored=False,phases=phases)
    softening=float(np.max(2*problem.seam_area*problem.strength/(2*problem.toughness/problem.strength)))
    switch=10*np.sqrt(problem.mass.min()/softening)
    try:
        if problem.material_dt>switch:
            for method,cap,wall in ((solve,25,1.),(solve_increment_ptc,60,2.)):
                problem.max_iterations=min(maximum,cap)
                problem.set_primary_increment(seed,seed_increment)
                try:
                    value,n=method(problem,seed.copy(),depth,min(deadline,time.monotonic()+wall))
                    phases.append(dict(factor=1.,method=method.__name__,iterations=n,accepted=True))
                    return value,n
                except (LoadStepFailure,TimeoutError) as exc:
                    phases.append(dict(factor=1.,method=method.__name__,accepted=False,error=str(exc)))
        value=seed.copy();problem.set_primary_increment(value,seed_increment)
        for factor in (1000.,100.,10.,1.):
            problem.friction=(friction if factor==1. else problem.ipc.FrictionPotential(
                problem.friction_velocity_epsilon*problem.material_dt*factor))
            stage=value.copy();increment=problem.step_increment(stage).copy()
            problem.max_iterations=min(maximum,35)
            try:
                value,n=solve(problem,stage.copy(),depth,min(deadline,time.monotonic()+2.))
                method='newton_lm'
            except (LoadStepFailure,TimeoutError):
                problem.set_primary_increment(stage,increment)
                problem.max_iterations=maximum
                value,n=solve_increment_ptc(problem,stage.copy(),depth,deadline)
                method='increment_ptc'
            total+=n+1
            phases.append(dict(factor=factor,method=method,iterations=n,accepted=True))
        return value,total
    finally:
        problem.friction=friction
        problem.max_iterations=maximum
        problem.friction_continuation_summary['original_friction_restored']=True
