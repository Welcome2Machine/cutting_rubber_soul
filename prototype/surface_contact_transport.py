"""Objective positive-weight chart contact with event-aware trial history.

Experimental local interface, NOT a native improved-max substitution. Uses a
transported tangential velocity one-form rather than the gradient of a finite
arc length. Its slave traction obeys the specified regularized Coulomb bound
also at curved finite gaps; moment balance includes master shape motion.
Pure trials never commit history. Projection/path ambiguity fails closed.
Point/curve paths use a conservative relative-speed distance bound. This does
not certify curve/curve crossings, global quadrature or a self-contact model.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.optimize import brentq
from .curve_contact import physical_barrier,friction_mollifier


class ContactPathError(ValueError):pass


def immutable(value):
    a=np.array(value,dtype=float,copy=True);a.setflags(write=False);return a


@dataclass(frozen=True)
class ContactHistory:
    positions: np.ndarray
    active: bool
    parameter: float | None
    generation: int=0
    chart_hash: str=''

    def __post_init__(self):object.__setattr__(self,'positions',immutable(self.positions))


def chart_hash(chart):return hashlib.sha256(chart.parameters.tobytes()).hexdigest()


def initial_history(chart,positions,*,dhat,side=1):
    x=np.asarray(positions,float);q=chart.project(x[0],x[1:],side=side)
    if q['gap']<=0:raise ContactPathError('Nonpositive initial smooth-surface gap')
    active=q['gap']<dhat
    return ContactHistory(x,active,q['parameter'] if active else None,chart_hash=chart_hash(chart))


def corotated_path(old,current,fraction):
    """2D objective endpoint interpolant AND exact endpoint derivatives.

    Master-centroid translation and least-squares rigid rotation are separated
    from deformation. A superposed endpoint rotation does not fake activation.
    This is an explicitly chosen within-step path, not measured intermediate
    motion. A full loading solution must independently check temporal accuracy.
    """
    old,x=np.asarray(old,float),np.asarray(current,float);z=float(fraction)
    if old.shape!=x.shape or old.ndim!=2 or old.shape[1]!=2 or not 0<=z<=1:
        raise ValueError('Matching 2D positions and fraction in [0,1] required')
    c0,c1=old[1:].mean(axis=0),x[1:].mean(axis=0)
    a0,a1=old[1:]-c0,x[1:]-c1
    A=float(np.sum(a0*a1));B=float(np.sum(a0[:,0]*a1[:,1]-a0[:,1]*a1[:,0]))
    scale=float(np.linalg.norm(a0)*np.linalg.norm(a1))
    if scale<=1e-20 or A*A+B*B<1e-20*scale**2:raise ContactPathError('Ambiguous corotational frame')
    theta=float(np.arctan2(B,A))
    if abs(theta)>np.pi-1e-7:raise ContactPathError('Half-turn path needs temporal resolution')
    def rotation(t):return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    R=np.array([[0.,-1.],[1.,0.]]);Q=rotation(z*theta);Q0=rotation(-theta)
    Z=(x-c1)@Q0.T;W=(1-z)*(old-c0)+z*Z
    y=c0+z*(c1-c0)+W@Q.T
    velocity=c1-c0+(Z-(old-c0)+theta*(W@R.T))@Q.T
    d=x.size;dc=np.zeros((2,d));dc[:,2:]=np.tile(np.eye(2),(1,len(x)-1))/(len(x)-1)
    dA=np.zeros_like(x);dB=np.zeros_like(x);dA[1:]=a0;dB[1:]=a0@R.T
    dt=((A*dB-B*dA)/(A*A+B*B)).ravel()
    dy=np.empty((len(x),2,d));Qall=Q@Q0
    for i in range(len(x)):
        direct=np.zeros((2,d));direct[:,2*i:2*i+2]=np.eye(2)
        dy[i]=z*dc+z*Qall@(direct-dc)+z*np.outer(Q@R@(W[i]-Z[i]),dt)
    return y,velocity,dy.reshape(d,d)


def relative_speed_bound(chart,old,x):
    """Bound relative point/curve speed in the corotating linear path.

    A cubic spline's Bezier control hull bounds its value, including between
    nodes. Rigid translation/rotation cancels from point/curve distance.
    """
    c0,c1=old[1:].mean(axis=0),x[1:].mean(axis=0)
    a0,a1=old[1:]-c0,x[1:]-c1
    angle=np.arctan2(np.sum(a0[:,0]*a1[:,1]-a0[:,1]*a1[:,0]),np.sum(a0*a1))
    Q=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    delta=(x-c1)@Q.T-(old-c0);curve=chart.geometry(delta[1:]);bound=0.
    for i,h in enumerate(np.diff(chart.parameters)):
        a,b,c,d=curve.c[:,i,:]
        controls=np.array([d,d+c*h/3,d+2*c*h/3+b*h*h/3,d+c*h+b*h*h+a*h**3])
        bound=max(bound,float(np.linalg.norm(delta[0]-controls,axis=1).max()))
    return bound*(1+1e-12)+1e-14


def contact_path(chart,history,x,*,dhat,side=1,interval_tolerance=1e-8,max_evaluations=4096):
    """Conservative point/curve clearance and isolated event brackets.

    All intervals are classified using distance Lipschitz bounds, not absence
    of sampled hits. An unresolved same-sign/tangent/multiple event fails.
    Temporal event accuracy is explicitly bounded in the interpolation fraction.
    This certificate does NOT bound entire FE edges against each other.
    """
    old=history.positions
    cache={}
    def at(t):
        if t in cache:return cache[t]
        if len(cache)>=max_evaluations:raise ContactPathError('Point/curve path certificate budget exhausted')
        y,vel,dy=corotated_path(old,x,t)
        q=chart.project(y[0],y[1:],side=side)
        if q['gap']<=0:raise ContactPathError('Within-step smooth-surface penetration')
        cache[t]=(q,vel,dy);return cache[t]
    bound=relative_speed_bound(chart,old,x);stack=[(0.,1.)];uncertain=[];clearance=[]
    while stack:
        lo,hi=stack.pop();ga=at(lo)[0]['gap'];gb=at(hi)[0]['gap']
        lower=min(ga,gb)-bound*(hi-lo)/2;upper=max(ga,gb)+bound*(hi-lo)/2
        if lower>dhat or (lower>0 and upper<dhat):clearance.append(lower);continue
        if hi-lo<=interval_tolerance:
            if lower<=0:raise ContactPathError('Uncertified point/curve clearance')
            uncertain.append((lo,hi));clearance.append(lower);continue
        mid=(lo+hi)/2;stack.extend(((mid,hi),(lo,mid)))
    merged=[]
    for lo,hi in sorted(uncertain):
        if merged and lo<=merged[-1][1]+1e-14:merged[-1][1]=hi
        else:merged.append([lo,hi])
    events=[]
    for lo,hi in merged:
        fa,fb=at(lo)[0]['gap']-dhat,at(hi)[0]['gap']-dhat
        if lo==0 and abs(fa)<1e-13:root=0.
        elif hi==1 and abs(fb)<1e-13:root=1.
        elif fa*fb>=0:raise ContactPathError('Unresolved same-sign/tangent event')
        else:root=brentq(lambda t:at(t)[0]['gap']-dhat,lo,hi,xtol=1e-14,rtol=1e-14)
        events.append(dict(fraction=root,bracket=[lo,hi],kind='activate' if fb<fa else 'detach'))
    if len(events)>1:raise ContactPathError('Multiple activation/detachment events need resolved steps')
    proof=dict(evaluations=len(cache),relative_speed_bound_mm=bound,
        min_clearance_lower_bound_mm=min(clearance),events=events,interval_tolerance=interval_tolerance,
        scope='Point/curve only, under corotated endpoint interpolation; not whole-edge/self-contact CCD')
    if history.active or at(1.)[0]['gap']>=dhat:return None,proof
    if len(events)!=1 or events[0]['kind']!='activate':raise ContactPathError('Missing certified activation bracket')
    root=events[0]['fraction']
    if root==0:
        q=at(0.)[0]
        return dict(fraction=0.,parameter=q['parameter'],parameter_gradient=np.zeros(x.size),
                    fraction_gradient=np.zeros(x.size)),proof
    q,vel,dy=at(root);dg=q['gap_gradient'].ravel();du=q['parameter_gradient'].ravel()
    slope=float(dg@vel.ravel())
    if slope>=-1e-10:raise ContactPathError('Tangential or reversed activation needs resolved step')
    dr=-(dg@dy)/slope
    anchor_gradient=du@dy+float(du@vel.ravel())*dr
    return dict(fraction=float(root),parameter=q['parameter'],parameter_gradient=anchor_gradient,
                fraction_gradient=dr),proof


def evaluate_contact(chart,current,history,*,weight,dhat,stiffness,mu,epsilon,side=1):
    """Exact fresh local Jacobian, with OLD accepted history held fixed.

    h=D/|c'|>0, b=h*grad(u), w=h*(u-u_anchor).
    R_f=mu*N*f1(w)*b; b_slave=t exactly. This non-potential friction
    residual is conjugate to the transported tangent velocity h*dot(u).
    w is its backward-Euler increment, not a finite path arc length.
    Activation-anchor dependence is differentiated, not frozen in Newton.
    """
    x=np.asarray(current,float)
    if not all(np.isfinite(a) for a in (weight,dhat,stiffness,mu,epsilon)) or min(weight,dhat,stiffness,epsilon)<=0 or mu<0:
        raise ValueError('Positive weight/scales and nonnegative friction required')
    if x.shape!=history.positions.shape or history.chart_hash!=chart_hash(chart):
        raise ValueError('History geometry/material chart mismatch')
    q=chart.differential(x[0],x[1:],side=side);gap=q['gap'];d=x.size
    if gap<=0:raise ContactPathError('Nonpositive smooth-surface gap')
    prior=chart.project(history.positions[0],history.positions[1:],side=side)
    if history.active:
        if prior['gap']>=dhat or history.parameter is None or abs(prior['parameter']-history.parameter)>1e-10:
            raise ValueError('Accepted history does not match prior contact')
    event,path_proof=contact_path(chart,history,x,dhat=dhat,side=side)
    active=gap<dhat
    if not active:
        zeros=np.zeros_like(x)
        result=dict(gradient=zeros,normal_gradient=zeros.copy(),friction_gradient=zeros.copy(),
            jacobian=np.zeros((d,d)),slip_mm=0.,friction_work_nmm=0.,normal_capacity_n=0.,
            normal_energy_nmm=0.,status='detached' if history.active else 'inactive')
    else:
        if history.active:
            anchor=history.parameter;da=np.zeros(d)
        else:
            if event is None:raise ContactPathError('Active trial missing activation event')
            anchor=event['parameter'];da=event['parameter_gradient']
        h=q['transport_metric'];dh=q['metric_gradient'];du=q['parameter_gradient'].ravel()
        beta=q['transport_covector'];dbeta=q['transport_jacobian'];dg=q['gap_gradient'].ravel()
        slip=h*(q['parameter']-anchor);ds=dh*(q['parameter']-anchor)+h*(du-da)
        b,bp,bpp=(float(a[0]) for a in physical_barrier(np.array([gap]),dhat,stiffness))
        _,f1,f2=(float(a[0]) for a in friction_mollifier(np.array([slip]),epsilon))
        N=-bp;ng=weight*bp*dg;fg=weight*mu*N*f1*beta
        J=weight*(bpp*np.outer(dg,dg)+bp*q['gap_hessian']+mu*(
            -bpp*f1*np.outer(beta,dg)+N*f2*np.outer(beta,ds)+N*f1*dbeta))
        result=dict(gradient=(ng+fg).reshape(x.shape),normal_gradient=ng.reshape(x.shape),
            friction_gradient=fg.reshape(x.shape),jacobian=J,slip_mm=float(slip),
            friction_work_nmm=float(weight*mu*N*f1*slip),normal_capacity_n=weight*N,
            normal_energy_nmm=weight*b,status='persistent' if history.active else 'activated',
            slave_tangent=q['tangent'],slave_tangential_metric=float(beta[:2]@q['tangent']))
    result.update(gap_mm=float(gap),parameter=q['parameter'],activation=event,path_proof=path_proof,active=active,
        trial_positions=immutable(x),parent_generation=history.generation,chart_hash=history.chart_hash,
        parent_positions_sha256=hashlib.sha256(history.positions.tobytes()).hexdigest(),
        scope='Local positive chart and point/curve path only; no whole-edge/self-contact/global FE qualification')
    return result


def commit_contact(history,trial,*,equilibrium_accepted):
    """Caller may commit only AFTER its separate global physical acceptance."""
    if not equilibrium_accepted:raise ValueError('Global equilibrium must be accepted before committing contact')
    if (trial['parent_generation']!=history.generation or trial['chart_hash']!=history.chart_hash
            or trial['parent_positions_sha256']!=hashlib.sha256(history.positions.tobytes()).hexdigest()):
        raise ValueError('Stale/foreign contact trial')
    return ContactHistory(trial['trial_positions'],trial['active'],trial['parameter'] if trial['active'] else None,
                          history.generation+1,history.chart_hash)
