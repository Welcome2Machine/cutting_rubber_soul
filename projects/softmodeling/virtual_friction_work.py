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
    """Global empirical error control over all geometry-seeded GK21 panels.

    QUADPACK supplies one embedded rule per panel. Its local tolerance is not an
    acceptance test: the heap refines the largest error until the SUM satisfies
    the unchanged global target. This avoids demanding tiny absolute errors on
    negligible panels when the total work is large.
    """
    import heapq
    cuts=np.unique(np.asarray(cuts,float))
    if len(cuts)<2 or cuts[0]!=0 or cuts[-1]!=1 or not np.isfinite(cuts).all():
        raise ValueError('Complete finite partition of [0,1] required')
    if min(absolute_tolerance,relative_tolerance)<=0:raise ValueError('Positive quadrature controls required')
    calls=0;cache={};panels={};heap=[];counter=0;total=error=0.;complete_initial=False
    def checked(s):
        nonlocal calls
        if time.monotonic()>=deadline:raise TimeoutError('Virtual work integration budget')
        if s not in cache:
            value=float(power(s));calls+=1
            if not np.isfinite(value) or value< -1e-12:raise ValueError('Invalid virtual Coulomb power')
            cache[s]=max(0.,value)
        return cache[s]
    def panel(lo,hi):
        # A deliberately nonrestrictive LOCAL request obtains exactly one GK21
        # rule. Only the independent global summed-error test below can accept.
        value,err,info,*message=quad(checked,float(lo),float(hi),epsabs=float('inf'),
            epsrel=0.,limit=1,full_output=1)
        # limit=1 deliberately requests the embedded GK21 rule only. Its
        # 'maximum subdivisions' status belongs to the local QUADPACK driver;
        # subdivision and convergence are owned by our global controller.
        unexpected=any(not m.startswith('The maximum number of subdivisions (1)') for m in message)
        if unexpected or info['last']!=1 or not np.isfinite([value,err]).all() or err<0:
            raise ValueError('Invalid elementary Gauss-Kronrod panel')
        return dict(a=float(lo),b=float(hi),work_n_mm=value,empirical_error_n_mm=err,evaluations=info['neval'])
    def insert(q):
        nonlocal counter
        panels[counter]=q;heapq.heappush(heap,(-q['empirical_error_n_mm'],counter));counter+=1
    issue=None
    try:
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            q=panel(lo,hi);insert(q);total+=q['work_n_mm'];error+=q['empirical_error_n_mm']
        complete_initial=True
        while error>absolute_tolerance+relative_tolerance*abs(total):
            if len(panels)>=32768:raise TimeoutError('Global quadrature panel budget')
            _,index=heapq.heappop(heap);parent=panels[index];mid=(parent['a']+parent['b'])/2
            if mid in (parent['a'],parent['b']):raise TimeoutError('Quadrature parameter resolution exhausted')
            left=panel(parent['a'],mid);right=panel(mid,parent['b'])
            # Replace only after both children exist: timeout retains a complete
            # prior partition, but its unresolved value cannot decide fracture.
            del panels[index];insert(left);insert(right)
            total+=left['work_n_mm']+right['work_n_mm']-parent['work_n_mm']
            error=max(0.,error+left['empirical_error_n_mm']+right['empirical_error_n_mm']-parent['empirical_error_n_mm'])
    except TimeoutError as exc:
        issue=str(exc)
    ordered=sorted(panels.values(),key=lambda q:q['a'])
    total=float(sum(q['work_n_mm'] for q in ordered));error=float(sum(q['empirical_error_n_mm'] for q in ordered))
    converged=complete_initial and issue is None and error<=absolute_tolerance+relative_tolerance*abs(total)
    return dict(method='geometry_partitioned_adaptive_gauss_kronrod',error_controller='global_sum_heap',
        converged=converged,work_n_mm=total if complete_initial else None,
        empirical_uncertainty_n_mm=2*error if complete_initial else None,
        partial_work_n_mm=total,error=issue,panels=ordered,evaluations=calls,
        completed_panels=len(ordered),total_panels=len(ordered) if complete_initial else len(cuts)-1,
        initial_geometry_panels=len(cuts)-1,complete_path_partition=complete_initial,
        absolute_tolerance_n_mm=absolute_tolerance,relative_tolerance=relative_tolerance,
        uncertainty_is_rigorous_bound=False)


def virtual_work(virtual, start, end, deadline):
    started=time.perf_counter()
    delta=(end-start)[virtual.surface]
    def power(s):
        lag=virtual.lag(start+s*(end-start))
        return 2*float(virtual.friction.gradient(lag,virtual.cmesh,delta)@delta.ravel())
    cuts=[];geometry={}
    try:
        cuts,geometry=geometry_partitions(virtual,start,end)
        result=integrate_power(power,cuts,deadline=deadline)
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
        result=dict(method='geometry_partitioned_adaptive_gauss_kronrod',converged=False,
            work_n_mm=None,empirical_uncertainty_n_mm=None,error=f'{type(exc).__name__}: {exc}',
            uncertainty_is_rigorous_bound=False)
    result.update(elapsed_s=time.perf_counter()-started,partition_points=cuts,geometry=geometry,
        unchanged_straight_path=True,material_and_damage_history_frozen=True)
    return result
