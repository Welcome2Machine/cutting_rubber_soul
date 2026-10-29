"""Geometry-partitioned original Coulomb work with explicitly empirical uncertainty."""
import time
import numpy as np
from scipy.integrate import quad
from virtual_path_geometry import geometry_partitions


def interval_decision(released_energy, fracture_work, work, uncertainty, converged):
    values=np.asarray([released_energy,fracture_work,work,uncertainty],float)
    if not np.isfinite(values).all() or fracture_work<=0 or work<0 or uncertainty<0:
        raise ValueError('Invalid virtual work interval')
    low=released_energy-fracture_work-(work+uncertainty)
    high=released_energy-fracture_work-max(0.,work-uncertainty)
    status=('accepted' if low>=0 else 'rejected' if high<0 else 'unknown') if converged else 'unknown'
    return dict(status=status,margin_lower_n_mm=float(low),margin_upper_n_mm=float(high),
        uncertainty_is_rigorous_bound=False)


def integrate_power(power, cuts, *, deadline, absolute_tolerance=1e-7, relative_tolerance=1e-6):
    """Adaptive Gauss-Kronrod on every geometry panel; no unvisited panel omitted."""
    cuts=np.unique(np.asarray(cuts,float))
    if len(cuts)<2 or cuts[0]!=0 or cuts[-1]!=1 or not np.isfinite(cuts).all():
        raise ValueError('Complete finite partition of [0,1] required')
    if min(absolute_tolerance,relative_tolerance)<=0:raise ValueError('Positive quadrature controls required')
    calls=0;cache={}
    def checked(s):
        nonlocal calls
        if time.monotonic()>=deadline:raise TimeoutError('Virtual work integration budget')
        if s not in cache:
            value=float(power(s));calls+=1
            if not np.isfinite(value) or value< -1e-12:raise ValueError('Invalid virtual Coulomb power')
            cache[s]=max(0.,value)
        return cache[s]
    panels=[];total=error=0.;warnings=[]
    try:
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            value,err,info,*message=quad(checked,float(lo),float(hi),epsabs=absolute_tolerance/(len(cuts)-1),
                epsrel=relative_tolerance,limit=100,full_output=1)
            total+=value;error+=err;warnings.extend(message)
            panels.append(dict(a=float(lo),b=float(hi),work_n_mm=value,empirical_error_n_mm=err,evaluations=info['neval']))
    except TimeoutError as exc:
        return dict(method='geometry_partitioned_adaptive_gauss_kronrod',converged=False,error=str(exc),
            work_n_mm=None,empirical_uncertainty_n_mm=None,partial_work_n_mm=total,
            completed_panels=len(panels),total_panels=len(cuts)-1,evaluations=calls,panels=panels,
            uncertainty_is_rigorous_bound=False)
    uncertainty=2*error
    return dict(method='geometry_partitioned_adaptive_gauss_kronrod',converged=not warnings and
        uncertainty<=2*(absolute_tolerance+relative_tolerance*abs(total)),work_n_mm=float(total),
        empirical_uncertainty_n_mm=float(uncertainty),warnings=warnings,panels=panels,
        completed_panels=len(panels),total_panels=len(cuts)-1,evaluations=calls,
        absolute_tolerance_n_mm=absolute_tolerance,relative_tolerance=relative_tolerance,
        uncertainty_is_rigorous_bound=False)


def virtual_work(virtual, start, end, deadline):
    started=time.perf_counter()
    cuts,geometry=geometry_partitions(virtual,start,end)
    delta=(end-start)[virtual.surface]
    def power(s):
        lag=virtual.lag(start+s*(end-start))
        return 2*float(virtual.friction.gradient(lag,virtual.cmesh,delta)@delta.ravel())
    result=integrate_power(power,cuts,deadline=deadline)
    result.update(elapsed_s=time.perf_counter()-started,partition_points=cuts,geometry=geometry,
        unchanged_straight_path=True,material_and_damage_history_frozen=True)
    return result
