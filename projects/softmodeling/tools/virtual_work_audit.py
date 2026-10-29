"""Independent composite Legendre audit (production uses adaptive Gauss-Kronrod)."""
import time
import numpy as np
from virtual_path_geometry import geometry_partitions


def audit_work(v, start, end, deadline):
    cuts,geometry=geometry_partitions(v,start,end)
    delta=end-start;sd=delta[v.surface];values=[];evaluations=0
    for order in (8,16,32):
        nodes,weights=np.polynomial.legendre.leggauss(order);total=0.
        for left,right in zip(cuts[:-1],cuts[1:]):
            for z,w in zip(nodes,weights):
                if time.monotonic()>=deadline:raise TimeoutError('Independent virtual work audit budget')
                s=(left+right)/2+(right-left)*z/2
                collisions=v.lag(start+s*delta)
                power=2*float(v.friction.gradient(collisions,v.cmesh,sd)@sd.ravel())
                if not np.isfinite(power) or power < -1e-12:raise ValueError('Invalid audit friction power')
                total+=w*(right-left)*power/2;evaluations+=1
        values.append(total)
    errors=np.abs(np.diff(values));uncertainty=2*float(max(errors))
    converged=bool(max(errors)<=1e-7+1e-6*abs(values[-1]))
    return dict(method='independent_composite_legendre_8_16_32',work_n_mm=float(values[-1]),
        empirical_uncertainty_n_mm=uncertainty,converged=converged,orders=[8,16,32],values_n_mm=values,
        evaluations=evaluations,partition_count=len(cuts)-1,uncertainty_is_rigorous_bound=False)
