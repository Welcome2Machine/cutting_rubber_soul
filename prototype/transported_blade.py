"""Experimental P1 reference-surface integral with rigid-blade path slip.

Reuses ONLY the global assembly/whole-edge guard of CurveBladeProblem. Replaces
its blade friction kinematics, force and fresh tangent as one interface. Native
material self-contact stays complete and unchanged (including its known finite-
gap objectivity limitation). No native blade EV/VV is added to this integral.
Rigid blade translation only; positions in the blade frame encode the exact
previous accepted history. There is no independent elastic stick state to reset.
"""
from functools import lru_cache
import hashlib
import json
import numpy as np
from scipy.optimize import brentq
from .curve_contact import CurveBladeProblem,physical_barrier,friction_mollifier
from .initiation import rounded_blade_distance,_blade_boundary_vertices
from .ipc_reference import BASELINE,LoadStepFailure


@lru_cache(maxsize=2)
def gauss(order):return np.polynomial.legendre.leggauss(order)


class RigidBladePath:
    """Exact geometry, analytic partition events, checked path quadrature.

    Outside this convex blade, closest-feature switches occur on perpendicular
    endpoint lines of the straight faces (also the circle/flank seams). Each
    distance piece is affine or radial. Add radial closest-approach fractions
    to certify its minimum on a straight relative-motion path. No sampling-only
    inference of absence of collision/contact is used.
    """
    def __init__(self,config,options,depth):
        self.config,self.options,self.depth=config,options,depth
        polygon=_blade_boundary_vertices(config,options,depth)
        self.starts=polygon[[0,1,-3,-2,-1]];self.ends=polygon[[1,2,-2,-1,0]]
        self.edges=self.ends-self.starts
        angle=-np.pi+np.deg2rad(config.blade_tip_angle_deg/2)
        center=polygon[2]-options.tip_radius_mm*np.array([np.cos(angle),np.sin(angle)])
        self.centers=np.vstack((polygon,center));self.polygon=polygon

    def project(self,points):
        penetration,n,dn=rounded_blade_distance(np.atleast_2d(points),self.config,self.options,self.depth,with_tangent=True)
        t=np.column_stack((-n[:,1],n[:,0]));dt=np.stack((-dn[:,1,:],dn[:,0,:]),axis=1)
        return -penetration,n,dn,t,dt

    def breaks(self,old,current):
        v=current-old;v2=float(v@v);values=[0.,1.]
        if v2<1e-30:return np.array(values)
        denom=self.edges@v
        for endpoints in (self.starts,self.ends):
            numerator=np.einsum('ei,ei->e',endpoints-old,self.edges)
            for a,b in zip(numerator,denom):
                if abs(b)>1e-20 and 0<a/b<1:values.append(float(a/b))
        for t in (self.centers-old)@v/v2:
            if 0<t<1:values.append(float(t))
        return np.unique(values)

    def path(self,old,current,dhat):
        old,current=np.asarray(old),np.asarray(current);v=current-old
        knots=self.breaks(old,current);g,n,_,_,_=self.project(old+knots[:,None]*v)
        if not np.isfinite(g).all() or np.min(g)<=0:raise LoadStepFailure('Analytic blade path crosses solid')
        active=g<dhat
        # Distance to a convex solid is convex; its dhat sublevel interval is
        # connected. Outside endpoints with interior contact cannot be skipped.
        if not active[-1]:
            if not active[0] and np.any(active):raise LoadStepFailure('Unresolved blade contact excursion within load chord')
            return dict(start=1.,start_gradient=np.zeros(2),knots=knots,minimum_gap_mm=float(g.min()),
                        status='detached' if active[0] else 'inactive')
        if active[0]:start=0.;gradient=np.zeros(2);status='persistent'
        else:
            j=np.flatnonzero(active)[0];lo,hi=knots[j-1],knots[j]
            start=brentq(lambda a:self.project(old+a*v)[0][0]-dhat,lo,hi,xtol=1e-14,rtol=1e-14)
            normal=self.project(old+start*v)[1][0];slope=float(normal@v)
            if slope>=-1e-12:raise LoadStepFailure('Blade activation is not a transverse entry')
            gradient=-start*normal/slope;status='activated'
        return dict(start=float(start),start_gradient=gradient,knots=knots,minimum_gap_mm=float(g.min()),status=status)

    def slip(self,old,current,dhat):
        path=self.path(old,current,dhat);a=path['start'];v=current-old
        if a==1:return 0.,np.zeros(2),path
        knots=np.r_[a,path['knots'][(path['knots']>a)&(path['knots']<1)],1.]
        def integrate(order):
            xi,gw=gauss(order);lo,hi=knots[:-1],knots[1:]
            lam=((lo[:,None]+hi[:,None])/2+(hi-lo)[:,None]*xi/2).ravel()
            weight=((hi-lo)[:,None]*gw/2).ravel()
            _,_,_,t,dt=self.project(old+lam[:,None]*v)
            rate=t@v
            derivative=t+lam[:,None]*np.einsum('qij,i->qj',dt,v)
            return float(weight@rate),weight@derivative
        s,ds=integrate(16);s8,d8=integrate(8)
        if abs(s-s8)>1e-11 or np.linalg.norm(ds-d8)>1e-7:
            raise LoadStepFailure('Blade path integration accuracy gate failed')
        if a>0:
            ta=self.project(old+a*v)[3][0];ds-=float(ta@v)*path['start_gradient']
        path['slip_quadrature_error_mm']=abs(s-s8);path['derivative_quadrature_error']=float(np.linalg.norm(ds-d8))
        return s,ds,path


def transported_contact(geometry,current,previous,weights,dhat,stiffness,mu,epsilon):
    """Full fresh force Jacobian at fixed accepted old positions.

    s=integral_a^1 t(y0+l*(y1-y0)).(y1-y0) dl; geometric seams partition
    integration but contribute no boundary jump since t is continuous outside
    the convex blade. The activation endpoint contributes -t(y_a).v*da/dy1.
    Force remains tangent with magnitude <= mu*N, not gradient of s.
    """
    current,previous=np.asarray(current),np.asarray(previous)
    gap,n,dn,t,dt=geometry.project(current);b,bp,bpp=physical_barrier(gap,dhat,stiffness)
    slip=np.zeros(len(gap));ds=np.zeros_like(current);events=[]
    for i in np.flatnonzero(gap<dhat):
        slip[i],ds[i],path=geometry.slip(previous[i],current[i],dhat)
        events.append(dict(quadrature_local_id=int(i),**path))
    f0,f1,f2=friction_mollifier(slip,epsilon);N=-bp
    normal=bp[:,None]*n;friction=(mu*N*f1)[:,None]*t
    hn=bpp[:,None,None]*n[:,:,None]*n[:,None,:]+bp[:,None,None]*dn
    jf=mu*(-bpp[:,None,None]*f1[:,None,None]*t[:,:,None]*n[:,None,:]
        +(N*f2)[:,None,None]*t[:,:,None]*ds[:,None,:]+(N*f1)[:,None,None]*dt)
    eigen,V=np.linalg.eigh(hn);metric=(V*np.maximum(eigen,0)[:,None,:])@V.transpose(0,2,1)
    metric+=(mu*N*f2)[:,None,None]*t[:,:,None]*t[:,None,:]
    return dict(energy=float(weights@(b+mu*N*f0)),gradient=weights[:,None]*(normal+friction),
        normal_gradient=weights[:,None]*normal,friction_gradient=weights[:,None]*friction,
        jacobian=weights[:,None,None]*(hn+jf),metric=weights[:,None,None]*metric,
        friction_work=float(weights@(mu*N*f1*slip)),slip_mm=slip,slip_gradient=ds,events=events,
        gap_mm=gap,normal_capacity_n=weights*N)


class TransportedBladeProblem(CurveBladeProblem):
    """Explicit candidate; original global PTC and physical force gate unchanged."""
    def __init__(self,**kwargs):
        self.path_geometry=None
        super().__init__(**kwargs)
        self.path_geometry=RigidBladePath(self.config,self.options,self.start)
        self.contact_control=dict(policy='rigid-blade-path-slip-v1',
            integration='fixed reference P1 boundary Gauss; one-sided, not native IPC equivalent',
            friction_history='exact previous accepted material quadrature positions in translating blade frame',
            self_contact='complete native improved-max; finite-gap rotation limitation retained',
            epsilon_u_mm=BASELINE['friction_smoothing_mm'],contact_spacing_mm=self.contact_spacing,
            contact_order=self.contact_order,guard_facets=self.guard_facets,
            quadrature_sha256=hashlib.sha256(self.Q.data.tobytes()+self.Q.indices.tobytes()+self.Q.indptr.tobytes()+self.weights.tobytes()).hexdigest())

    def _blade_data(self,x):
        if self.path_geometry is None:self.path_geometry=RigidBladePath(self.config,self.options,self.start)
        geometry=self.path_geometry;shift=x[self.n]-self.rest[self.n];oldshift=self.previous[self.n]-self.rest[self.n]
        points=self.Q@x-shift;oldpoints=self.Q@self.previous-oldshift
        minimum=geometry.polygon.min(axis=0).copy();minimum[1]-=self.options.tip_radius_mm
        maximum=geometry.polygon.max(axis=0)
        selected=np.flatnonzero(np.all((points>=minimum-self.dhat)&(points<=maximum+self.dhat),axis=1))
        ids=(2*selected[:,None]+np.arange(2)).ravel();D=self.D[ids,:]
        data=transported_contact(geometry,points[selected],oldpoints[selected],self.weights[selected],self.dhat,
            self.contact_stiffness,self.config.friction_coefficient,BASELINE['friction_smoothing_mm'])
        data['quadrature_ids']=selected
        return data,D,float(data['gap_mm'].min()) if len(selected) else np.inf

    def acceptance(self,x,depth,iterations,outer_iterations):
        accepted=super().acceptance(x,depth,iterations,outer_iterations)
        # Check the explicitly chosen old-to-new history path, in addition to
        # Newton trial CCD. Inactive-inactive contact excursions are unresolved.
        safe=self.safe_step(self.previous,x-self.previous)
        if safe<1.-1e-10:raise LoadStepFailure('Whole-load chord geometry certificate failed')
        shift=x[self.n]-self.rest[self.n];oldshift=self.previous[self.n]-self.rest[self.n]
        points=self.Q@x-shift;old=self.Q@self.previous-oldshift
        low=self.path_geometry.polygon.min(axis=0).copy();low[1]-=self.options.tip_radius_mm
        high=self.path_geometry.polygon.max(axis=0)
        selected=np.flatnonzero(np.all(np.maximum(points,old)>=low-self.dhat,axis=1)&
                                np.all(np.minimum(points,old)<=high+self.dhat,axis=1))
        counts={};min_gap=np.inf
        for i in selected:
            path=self.path_geometry.path(old[i],points[i],self.dhat)
            counts[path['status']]=counts.get(path['status'],0)+1;min_gap=min(min_gap,path['minimum_gap_mm'])
        data,_,_=self._blade_data(x)
        accepted.update(contact_discretization='rigid_blade_path_slip_reference_gauss',
            blade_path_states=counts,blade_path_minimum_gap_mm=None if not np.isfinite(min_gap) else min_gap,
            blade_max_path_quadrature_error_mm=max((e.get('slip_quadrature_error_mm',0) for e in data['events']),default=0.),
            blade_max_path_derivative_error=max((e.get('derivative_quadrature_error',0) for e in data['events']),default=0.),
            blade_friction_work_definition='endpoint capacity times signed integrated tangent slip; not exact integrated friction energy')
        return accepted

    def checkpoint_data(self):
        return dict(super().checkpoint_data(),blade_contact_policy_json=np.array(json.dumps(self.contact_control,sort_keys=True)))

    def audit_state(self,state,record):
        if json.loads(str(state['blade_contact_policy_json']))!=self.contact_control:
            raise ValueError('Blade contact history/quadrature policy changed')
        return super().audit_state(state,record)

    def _audit_extra_acceptance(self,fresh,record):
        if fresh['blade_path_states']!=record['blade_path_states']:raise ValueError('Blade activation history mismatch')
        return dict(blade_path_history_verified=True)
