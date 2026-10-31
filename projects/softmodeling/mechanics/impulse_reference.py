"""Event-resolved planar Coulomb reference, N/mm/s units.

One true point mass, affine translating rigid plane, constant external force on
one requested interval. Events split the physical trajectory; Newton iterates
are never paths. e=0 impact and Coulomb impulse are explicit model assumptions.
This kernel is NOT a finite-strain FE integrator or a replacement IPC audit.
"""
import numpy as np


def advance(position, velocity, time_s, dt, *, mass, force, wall_velocity=(0.,0.),
            wall_offset=0., mu=.35, max_events=32):
    x=np.array(position,dtype=float,copy=True);v=np.array(velocity,dtype=float,copy=True)
    f=np.asarray(force,float);w=np.asarray(wall_velocity,float)
    if x.shape!=(2,) or v.shape!=(2,) or f.shape!=(2,) or w.shape!=(2,):raise ValueError('Planar vectors required')
    if not np.all(np.isfinite([*x,*v,*f,*w,mass,dt,time_s,wall_offset,mu])) or mass<=0 or dt<=0 or mu<0:raise ValueError('Invalid reference input')
    t=float(time_s);end=t+dt;oldx=x.copy();oldv=v.copy();events=[];segments=[]
    impulse=np.zeros(2);ext_impulse=np.zeros(2);work_ext=work_wall=dfric=dimpact=0.;minimum_gap=float('inf')
    def gap():return float(x[1]-wall_offset-w[1]*t)
    for iteration in range(max_events):
        remaining=end-t
        g=gap();minimum_gap=min(minimum_gap,g)
        if g < -1e-12:raise ValueError('Initial/intermediate penetration; no position projection permitted')
        relative=v-w
        if g<=1e-12 and relative[1]<-1e-12:
            before=v.copy();pn=-mass*relative[1];pt=float(np.clip(-mass*relative[0],-mu*pn,mu*pn))
            P=np.array([pt,pn]);v+=P/mass;average=.5*(before+v)-w
            friction=-pt*average[0];normal=-pn*average[1]
            if min(friction,normal)<-1e-12:raise RuntimeError('Impact law supplies unaccounted energy')
            impulse+=P;work_wall+=float(P@w);dfric+=friction;dimpact+=normal
            events.append(dict(kind='impact',time_s=t,position=x.tolist(),velocity_before=before.tolist(),velocity_after=v.tolist(),impulse_n_s=P.tolist(),friction_n_mm=friction,normal_impact_loss_n_mm=normal))
            continue
        # Output is right-continuous: consume an impact at end before returning.
        # Its impulse and work above are part of this interval, including at h=0.
        if remaining<=0:break
        contact=g<=1e-12 and abs(relative[1])<=1e-12 and f[1]<=0
        contact_force=np.zeros(2);event=None;h=remaining
        if contact:
            contact_force[1]=-f[1];limit=mu*contact_force[1]
            if abs(relative[0])<=1e-12 and abs(f[0])<=limit:
                contact_force[0]=-f[0]
            else:
                sign=np.sign(relative[0]) if abs(relative[0])>1e-12 else np.sign(f[0])
                contact_force[0]=-limit*sign
                at=(f[0]+contact_force[0])/mass
                stop=-relative[0]/at if at*relative[0]<0 else float('inf')
                if stop>0 and stop<=h:h=stop;event='stick'
        else:
            a=f[1]/mass;b=relative[1]
            roots=[]
            if abs(a)<1e-30:
                if b<0:roots=[-g/b]
            else:
                discriminant=b*b-2*a*g
                if discriminant>=0:
                    root=np.sqrt(discriminant)
                    # Stable quadratic roots for g+b*t+a*t²/2=0.
                    q=-b-np.copysign(root,b)
                    if q:roots=[q/a,2*g/q]
            # A dimensionful absolute cutoff can discard a real return to the
            # wall (tiny outward velocity under a stiff inward load). Resolve
            # every positive event; the event budget bounds degenerate cases.
            approaching=[z for z in roots if z>0 and b+a*z<0]
            if approaching and min(approaching)<=h:h=min(approaching);event='contact'
        a=(f+contact_force)/mass;beforex=x.copy();beforev=v.copy();beforet=t
        dx=h*v+.5*h*h*a;x+=dx;v+=h*a;t+=h
        P=h*contact_force;impulse+=P;ext_impulse+=h*f
        work_ext+=float(f@dx);work_wall+=float(P@w)
        diss=-float(contact_force@(dx-h*w));dfric+=diss
        if diss < -1e-12:raise RuntimeError('Slip segment supplied friction energy')
        rel0=beforex[1]-wall_offset-w[1]*beforet
        candidates=[rel0,gap()]
        if a[1]>0:
            tau=-(beforev[1]-w[1])/a[1]
            if 0<tau<h:candidates.append(rel0+(beforev[1]-w[1])*tau+.5*a[1]*tau*tau)
        minimum_gap=min(minimum_gap,*candidates)
        segments.append(dict(start_s=beforet,end_s=t,position_before=beforex.tolist(),position_after=x.tolist(),velocity_before=beforev.tolist(),velocity_after=v.tolist(),contact=bool(contact),contact_force_n=contact_force.tolist(),minimum_gap_mm=float(min(candidates))))
        if event:events.append(dict(kind=event,time_s=t,position=x.tolist(),velocity_after=v.tolist()))
    else:raise RuntimeError('Reference event budget exhausted')
    if minimum_gap < -1e-12:raise ValueError('Continuous quadratic trajectory penetrates plane')
    dk=.5*mass*float(v@v-oldv@oldv)
    defect=work_ext+work_wall-dk-dfric-dimpact
    momentum=mass*(v-oldv)-impulse-ext_impulse
    return dict(position=x,velocity=v,time_s=end,events=events,segments=segments,contact_impulse_n_s=impulse,
        external_impulse_n_s=ext_impulse,momentum_residual_n_s=momentum,external_work_n_mm=work_ext,
        wall_work_n_mm=work_wall,friction_dissipation_n_mm=dfric,normal_impact_loss_n_mm=dimpact,
        kinetic_delta_n_mm=dk,numerical_work_n_mm=defect,minimum_gap_mm=minimum_gap,
        displacement=x-oldx,assumption='constant force and affine planar wall; event resolved e=0 Coulomb; no FE transfer')


def midpoint_stop_counterexample(v0,h,mass,friction):
    """Exact solution of the midpoint velocity inclusion, including its bad root."""
    average=np.sign(v0)*max(abs(v0)-h*friction/(2*mass),0.)
    v1=2*average-v0;impulse=mass*(v1-v0)
    return dict(velocity=v1,displacement=h*average,impulse=impulse,
                friction_work=-impulse*average,kinetic_delta=.5*mass*(v1*v1-v0*v0))
