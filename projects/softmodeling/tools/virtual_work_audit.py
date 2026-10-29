"""Independent composite Legendre audit (production uses adaptive Gauss-Kronrod)."""
import time
import numpy as np
from virtual_path_geometry import geometry_partitions


def audit_work(v, start, end, deadline):
    cuts,geometry=geometry_partitions(v,start,end)
    delta=end-start;sd=delta[v.surface];values=[];evaluations=0
    # Independent arithmetic: form the affine contact coordinates in extended
    # precision and translate before conversion to the IPC binary64 input.
    # This is the same straight path, not an alternative virtual process.
    a=start.astype(np.longdouble);d=end.astype(np.longdouble)-a
    if geometry['features']:
        near=min(geometry['features'],key=lambda q:q['gap_mm'])
        vertex=v.surface[near['vertices'][-1]]
        origin=a[vertex]+np.longdouble(near['s'])*d[vertex]
    else:
        origin=a[v.surface[0]].copy()
    relative=a-origin
    for order in (8,16,32):
        nodes,weights=np.polynomial.legendre.leggauss(order);total=0.
        for left,right in zip(cuts[:-1],cuts[1:]):
            for z,w in zip(nodes,weights):
                if time.monotonic()>=deadline:raise TimeoutError('Independent virtual work audit budget')
                s=(left+right)/2+(right-left)*z/2
                collisions=v.lag(np.asarray(relative+np.longdouble(s)*d,float))
                power=2*float(v.friction.gradient(collisions,v.cmesh,sd)@sd.ravel())
                if not np.isfinite(power) or power < -1e-12:raise ValueError('Invalid audit friction power')
                total+=w*(right-left)*power/2;evaluations+=1
        values.append(total)
    errors=np.abs(np.diff(values));uncertainty=2*float(max(errors))
    converged=bool(max(errors)<=1e-7+1e-6*abs(values[-1]))
    return dict(method='independent_composite_legendre_8_16_32',work_n_mm=float(values[-1]),
        empirical_uncertainty_n_mm=uncertainty,converged=converged,orders=[8,16,32],values_n_mm=values,
        evaluations=evaluations,partition_count=len(cuts)-1,uncertainty_is_rigorous_bound=False,
        coordinate_evaluation=dict(method='extended_precision_relative_affine_then_binary64',
            origin_mm=np.asarray(origin,float).tolist(),mantissa_bits=int(np.finfo(np.longdouble).nmant+1),
            unchanged_mathematical_path=True,shifted_coordinates_used_only_for_contact_power=True))
