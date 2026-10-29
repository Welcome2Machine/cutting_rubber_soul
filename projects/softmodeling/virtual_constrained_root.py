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


def solve_constrained(v,seed,depth,deadline,*,edge=(49,50),vertex=48,gap_floor=2e-10):
    """Constrained linear least-residual step without forming normal equations."""
    x=seed.copy();trace=[]
    for iteration in range(20):
        if time.monotonic()>=deadline:raise TimeoutError('Constrained root budget')
        v.set_primary_increment(x,x-v.previous);v.trial=x.copy()
        lag=v.lag(x);_,g,h,j=v.evaluate(x,lag,projected=False);v.update_constraint_free(x,g)
        residual=g[v.free];norm=float(np.sqrt(2)*np.linalg.norm(residual))
        gap,n,feature=signed_chord_gap(v.previous,x,edge,vertex)
        row=dict(iteration=iteration,original_residual_n=norm,min_j=float(j),feature=feature)
        trace.append(row);v.constrained_root_trace=trace
        if norm<=1e-4:
            if j>0 and v.accepted_chord_is_safe(x):return x,trace
            raise LoadStepFailure('Original root failed full chord/positive J')
        jac=v.coupled_friction_tangent(x,lag,h)[v.free][:,v.free].tocsc()
        factor=splu(jac);newton=factor.solve(-residual);nf=n.ravel()[v.free]
        target=max(-.5*gap,gap_floor-gap)
        direction=newton.copy();active=float(nf@newton)<target
        if active:
            z=factor.solve(nf,trans='T');w=factor.solve(z);den=float(nf@w)
            if den<=0 or not np.isfinite(den):raise LoadStepFailure('Invalid constrained solve compliance')
            direction+=w*((target-float(nf@newton))/den)
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
