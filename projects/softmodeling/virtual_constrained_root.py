"""Diagnostic Newton direction constrained by one virtual-chord feature.

The linearized gap is a search constraint only. Original force, local/full CCD,
positive J and boundaries decide acceptance. No constraint reaction is accepted
as a physical force. Other contact pairs remain in complete native CCD.
"""
import time
import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.sparse.linalg import splu
from prototype.ipc_reference import LoadStepFailure


def signed_chord_gap(anchor, endpoint, edge, vertex):
    """Signed interior EV minimum and endpoint derivative (envelope theorem)."""
    i,j=edge;k=vertex;delta=endpoint-anchor
    e0=anchor[j]-anchor[i];de=delta[j]-delta[i]
    w0=anchor[k]-anchor[i];dw=delta[k]-delta[i]
    e=[P([e0[q],de[q]]) for q in (0,1)]
    w=[P([w0[q],dw[q]]) for q in (0,1)]
    cross=e[0]*w[1]-e[1]*w[0];rr=e[0]*e[0]+e[1]*e[1]
    stationary=2*cross.deriv()*rr-cross*rr.deriv()
    roots=stationary.roots() if np.max(np.abs(stationary.coef),initial=0)>0 else []
    samples=[0.,1.]+[float(z.real) for z in roots if abs(z.imag)<1e-9 and 0<z.real<1]
    options=[]
    for s in samples:
        ee=e0+s*de;ww=w0+s*dw;length=float(np.linalg.norm(ee))
        if not length:continue
        beta=float(ee@ww/length**2)
        if not 0<=beta<=1:continue
        normal=np.array([-ee[1],ee[0]])/length
        options.append((float(normal@ww),s,beta,normal))
    if not options:raise ValueError('Selected EV has no interior minimum')
    gap,s,beta,normal=min(options,key=lambda q:q[0])
    gradient=np.zeros_like(endpoint)
    gradient[k]+=s*normal
    gradient[i]-=s*(1-beta)*normal
    gradient[j]-=s*beta*normal
    return gap,gradient,dict(s=s,beta=beta,gap_mm=gap,edge=list(edge),vertex=vertex)


def closest_chord_constraint(v, endpoint):
    from virtual_path_geometry import geometry_partitions
    _,geometry=geometry_partitions(v,v.previous,endpoint)
    if not geometry['features']:
        raise ValueError('No close virtual-chord feature to constrain')
    feature=min(geometry['features'],key=lambda q:q['gap_mm'])
    s=feature['s'];points=v.previous+s*(endpoint-v.previous)
    ids=v.surface[np.asarray(feature['vertices'],int)];gradient=np.zeros_like(endpoint)
    if feature['kind']=='EV':
        i,j,k=ids;e=points[j]-points[i];w=points[k]-points[i];length=np.linalg.norm(e)
        if not length:raise ValueError('Degenerate chord edge')
        beta=float(e@w/length**2);normal=np.array([-e[1],e[0]])/length
        if normal@w<0:normal=-normal
        gap=float(normal@w)
        gradient[k]+=s*normal;gradient[i]-=s*(1-beta)*normal;gradient[j]-=s*beta*normal
    else:
        i,j=ids;w=points[i]-points[j];gap=float(np.linalg.norm(w))
        if not gap:raise ValueError('Unresolved coincident chord vertices')
        normal=w/gap;gradient[i]+=s*normal;gradient[j]-=s*normal
    return gap,gradient,dict(feature,vertices_global=ids.tolist(),recomputed_gap_mm=gap)


def constrained_direction(jacobian,residual,normal,target):
    """Least linear force residual with n.p >= target; no squared matrix formed."""
    factor=splu(jacobian.tocsc());newton=factor.solve(-residual)
    direction=newton.copy();active=float(normal@newton)<target
    if active:
        z=factor.solve(normal,trans='T');w=factor.solve(z);den=float(normal@w)
        if den<=0 or not np.isfinite(den):raise LoadStepFailure('Invalid constrained solve compliance')
        direction+=w*((target-float(normal@newton))/den)
    return direction,newton,active


def solve_constrained(v,seed,depth,deadline,*,gap_floor=None):
    """Original-equation recovery using the closest full-chord feature.

    The search target is ten CCD resolution units by default, not a physical
    clearance condition. Every accepted root retains the original force gate.
    """
    if np.any(v.virtual_solver_load):raise ValueError('Original zero-load equation required')
    tolerance=getattr(v,'ffm_ccd_tolerance',None)
    if gap_floor is None:
        if tolerance is None:raise ValueError('Explicit CCD resolution required for constrained recovery')
        gap_floor=10*tolerance
    if not 0<gap_floor<v.dhat:raise ValueError('Invalid numerical search clearance')
    x=seed.copy();trace=[]
    for iteration in range(20):
        if time.monotonic()>=deadline:raise TimeoutError('Constrained root budget')
        v.set_primary_increment(x,x-v.previous);v.trial=x.copy()
        lag=v.lag(x);_,g,h,j=v.evaluate(x,lag,projected=False);v.update_constraint_free(x,g)
        residual=g[v.free];norm=float(np.sqrt(2)*np.linalg.norm(residual))
        gap,n,feature=closest_chord_constraint(v,x)
        row=dict(iteration=iteration,original_residual_n=norm,min_j=float(j),feature=feature,
            search_clearance_mm=gap_floor,clearance_is_not_physical_parameter=True)
        trace.append(row);v.constrained_root_trace=trace
        if not np.isfinite(norm) or j<=0:raise LoadStepFailure('Invalid constrained Newton state')
        if norm<=1e-4 and v.accepted_chord_is_safe(x):return x,trace
        jac=v.coupled_friction_tangent(x,lag,h)[v.free][:,v.free].tocsc();nf=n.ravel()[v.free]
        target=max(-.5*gap,gap_floor-gap)
        direction,newton,active=constrained_direction(jac,residual,nf,target)
        row.update(constraint_active=active,target_linear_gap_change_mm=target,
            newton_linear_gap_change_mm=float(nf@newton),constrained_linear_gap_change_mm=float(nf@direction),
            original_linear_solve_residual_n=float(np.linalg.norm(jac@newton+residual)),
            predicted_full_residual_n=float(np.sqrt(2)*np.linalg.norm(residual+jac@direction)))
        full=np.zeros(v.total);full[v.free]=direction;full=full.reshape(x.shape)
        _,alpha,_=v._ptc_trial(x,full,float(np.max(np.abs(direction))))
        row['initial_ccd_alpha']=float(alpha);accepted=False
        for cut in range(20):
            if time.monotonic()>=deadline:raise TimeoutError('Constrained root line-search budget')
            candidate=x+alpha*full
            v.set_primary_increment(candidate,candidate-v.previous)
            _,gt,_,jt=v.evaluate(candidate,v.lag(candidate),tangent=False)
            nt=float(np.sqrt(2)*np.linalg.norm(gt[v.free]))
            if np.isfinite(nt) and jt>0 and nt<norm and v.accepted_chord_is_safe(candidate):
                x=candidate;accepted=True;row.update(alpha=float(alpha),new_residual_n=nt);break
            alpha*=.5
        if not accepted:raise LoadStepFailure('Constrained original-force residual stagnated')
    raise LoadStepFailure('Constrained root iteration limit')
