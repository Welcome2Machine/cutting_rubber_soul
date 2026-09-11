"""Sparse pseudo-arclength predictor/corrector for a fixed numerical path.

The parameter is auxiliary, never physical time. See method notes R21/R24.
Smooth simple folds motivate the bordering; nonsmooth paths have no general
convergence guarantee. Every accepted point satisfies its augmented equations.
"""
import time
import numpy as np
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import splu


class PathFailure(RuntimeError):
    def __init__(self, message, point, *, budget=False,tangent=None,step=None):
        super().__init__(message)
        self.point=np.array(point,copy=True)
        self.budget=budget
        self.tangent=None if tangent is None else np.array(tangent,copy=True)
        self.step=step


def bordered(jac, parameter_derivative, plane):
    return bmat([[csc_matrix(jac),csc_matrix(parameter_derivative[:,None])],
                 [csc_matrix(plane[None,:-1]),csc_matrix([[plane[-1]]])]],format='csc')


def follow_path(evaluate, start, *, target=1., step=.05, max_step=.2,
                min_step=1e-7, max_steps=100, max_corrector=12,
                tolerance=1e-3, seconds=180., accept=None, observer=None,event=None,initial_tangent=None):
    """Follow H(x,a)=0 with a normalized, consistently oriented tangent.

    evaluate(point, tangent) returns H and, when requested, H_x/H_a.
    Coordinates and residuals must already have fixed, dimensionless scales.
    accept(point) may return an independently validated physical result.
    A target crossing is corrected on a=target, rather than interpolated and
    reported as a root. Rejected corrections never update the path history.
    Optional event returns a switching equation/gradient and an outgoing
    one-sided Jacobian. At a sharp corner its branch sign, not an incoming
    tangent dot product, sets the outgoing direction.
    """
    z=np.array(start,dtype=float,copy=True)
    if (z.ndim!=1 or len(z)<2 or not np.all(np.isfinite(z)) or
        not np.isfinite(target) or target<=z[-1] or
        not 0<min_step<=step<=max_step or not np.isfinite(max_step) or
        max_steps<1 or max_corrector<1 or not np.isfinite(seconds) or seconds<=0 or
        not np.isfinite(tolerance) or tolerance<=0):
        raise ValueError('Invalid continuation start or budget')
    deadline=time.monotonic()+seconds;t=None
    def path_failure(message,*,budget=False):
        return PathFailure(message,z,budget=budget,tangent=t,step=step)
    def call(point,tangent):
        if time.monotonic()>deadline:
            raise path_failure('Auxiliary path wall budget',budget=True)
        return evaluate(point,tangent)
    def emit(**entry):
        if observer is not None:observer(entry)
    h,j,b=call(z,True)
    if np.linalg.norm(h)>tolerance:
        raise ValueError('Continuation start does not satisfy the auxiliary equations')
    plane=np.zeros(len(z));plane[-1]=1.
    orientation=plane if initial_tangent is None else np.asarray(initial_tangent,dtype=float)
    if orientation.shape!=z.shape or not np.all(np.isfinite(orientation)) or np.linalg.norm(orientation)==0:
        raise ValueError('Invalid initial path tangent')
    def tangent(jac,derivative,previous):
        direction=splu(bordered(jac,derivative,previous)).solve(np.r_[np.zeros(len(z)-1),1.])
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction)==0:
            raise RuntimeError('Invalid path tangent')
        direction/=np.linalg.norm(direction)
        if direction@previous<0:direction=-direction
        return direction
    try:t=tangent(j,b,orientation)
    except RuntimeError as exc:raise path_failure(str(exc)) from exc
    emit(event='start',parameter=float(z[-1]),tangent_parameter=float(t[-1]))
    for count in range(max_steps):
        if accept is not None:
            result=accept(z)
            if result is not None:return z,result
        predicted=z+step*t
        crossing=(z[-1]<target<=predicted[-1])
        normal=plane if crossing else t
        if crossing:predicted=z+(target-z[-1])/t[-1]*t
        switch=event(z,predicted) if event is not None else None
        if switch is not None:
            predicted=z+switch['fraction']*(predicted-z)
            crossing=False
        def constraint(point):
            return switch['equation'](point) if switch is not None else (normal@(point-predicted),normal)
        v=predicted.copy();converged=False;merit=None;iteration=-1
        failure='corrector iteration limit'
        try:
            for iteration in range(max_corrector):
                h,j,b=call(v,True)
                value,gradient=constraint(v)
                residual=np.r_[h,value]
                merit=float(np.linalg.norm(residual))
                if merit<=tolerance:
                    if switch is None and not crossing and z[-1]<target<=v[-1]:
                        # Curvature can cross the target even when the tangent
                        # predictor did not. Correct the target plane as well.
                        crossing=True;normal=plane
                        predicted=z+(target-z[-1])/(v[-1]-z[-1])*(v-z)
                        v=predicted.copy()
                        continue
                    converged=True;break
                delta=splu(bordered(j,b,gradient)).solve(-residual)
                if not np.all(np.isfinite(delta)):
                    failure='nonfinite correction';break
                for cut in range(12):
                    trial=v+(.5**cut)*delta
                    # Avoid accepting a distant intersection of the hyperplane.
                    if np.linalg.norm(trial-predicted)>2*step:continue
                    try:
                        hh=call(trial,False)[0]
                        new_merit=np.linalg.norm(np.r_[hh,constraint(trial)[0]])
                    except FloatingPointError:continue
                    if new_merit<merit*(1-1e-4*.5**cut):
                        v=trial;break
                else:
                    failure='corrector line search or trust radius';break
        except (RuntimeError,FloatingPointError) as exc:
            if isinstance(exc,PathFailure):raise
            emit(event='corrector_error',message=str(exc))
            failure=str(exc)
        if not converged:
            emit(event='reject',parameter=float(z[-1]),step=float(step),
                 reason=failure,corrector_iterations=iteration+1,auxiliary_residual=merit)
            step*=.5
            if step<min_step:raise path_failure('Auxiliary path step below minimum')
            continue
        # The corrector has already evaluated the tangent at this point.
        try:
            if switch is None:new_t=tangent(j,b,t)
            else:
                side_j=switch['jacobian'](v,j)
                gradient=switch['equation'](v)[1]*switch['direction']
                new_t=tangent(side_j,b,gradient)
                if switch.get('on_accept') is not None:switch['on_accept'](v)
                emit(event='switch',label=switch['label'],parameter=float(v[-1]),
                     outgoing_direction=switch['direction'],incoming_alignment=float(new_t@t))
        except RuntimeError as exc:raise path_failure(str(exc)) from exc
        angle=float(new_t@t)
        z=v;t=new_t
        emit(event='point',index=count+1,parameter=float(z[-1]),step=float(step),
             corrector_iterations=iteration+1,auxiliary_residual=merit,
             tangent_parameter=float(t[-1]),tangent_alignment=angle)
        if crossing:
            if accept is None:return z,None
            result=accept(z)
            if result is not None:return z,result
            raise path_failure('Auxiliary target reached but original acceptance failed')
        if iteration<=3 and angle>.9:step=min(max_step,1.4*step)
        elif iteration>=7 or angle<.5:step=max(min_step,.5*step)
    raise path_failure('Auxiliary path step budget',budget=True)
