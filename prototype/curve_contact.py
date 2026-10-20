"""Explicit analytic rigid-blade candidate; default IPC remains unchanged.

Reference-boundary Gauss quadrature of the physical IPC log barrier and the
same regularized Coulomb law. This ONE-SIDED integral is not algebraically
equivalent to IPC's symmetric improved-max contact discretization. Material
self-contact remains native IPC. A circumscribed blade polygon supplies a
conservative whole-edge CCD guard, including contacts between Gauss points.
Only translation of the rigid blade and P1 body elements are supported.
"""
import numpy as np
from scipy.sparse import coo_matrix, kron, eye

from .ipc_reference import BASELINE,LoadStepFailure,determinant_step
from .ipc_viscoelastic import ViscoelasticProblem
from .initiation import rounded_blade_distance,_blade_boundary_vertices


def physical_barrier(gap,dhat,stiffness):
    """B(d^2), dB/dd, d2B/dd2; same physical scaling as ipctk 1.6."""
    d=np.asarray(gap,dtype=float)
    if not np.isfinite(d).all() or np.any(d<=0):
        raise FloatingPointError('Nonpositive/nonfinite analytic blade gap')
    b=np.zeros_like(d);first=b.copy();second=b.copy();mask=d<dhat
    s=d[mask]**2;D=dhat*dhat;log=np.log(s/D);factor=stiffness/dhat**3
    b[mask]=-factor*(s-D)**2*log
    bs=factor*(D-s)*(2*log-D/s+1)
    bss=factor*((D/s+2)*D/s-2*log-3)
    first[mask]=2*d[mask]*bs;second[mask]=2*bs+4*s*bss
    return b,first,second


def friction_mollifier(s,eps):
    """Potential f0(|s|), its odd first derivative and even second derivative."""
    a=np.abs(s);small=a<eps
    energy=np.where(small,-a**3/(3*eps**2)+a*a/eps+eps/3,a)
    first=np.where(small,s*(2/eps-a/eps**2),np.sign(s))
    second=np.where(small,2/eps-2*a/eps**2,0.)
    return energy,first,second


def point_contact(gap,normal,normal_jacobian,relative,weights,dhat,stiffness,mu,eps):
    """Work-conjugate force, FULL fresh Jacobian, and auxiliary PSD metric.

    The force is the gradient with frozen friction geometry; its fresh Jacobian
    includes geometry and normal-capacity derivatives. Fresh friction energy
    itself is NOT a global potential for this residual.
    """
    b,bp,bpp=physical_barrier(gap,dhat,stiffness)
    n=normal;dn=normal_jacobian;t=np.column_stack((-n[:,1],n[:,0]))
    dt=np.stack((-dn[:,1,:],dn[:,0,:]),axis=1)
    s=np.einsum('qi,qi->q',t,relative);f0,f1,f2=friction_mollifier(s,eps)
    N=-bp;ds=t+np.einsum('qi,qij->qj',relative,dt)
    normal_g=bp[:,None]*n;friction_g=(mu*N*f1)[:,None]*t
    hn=bpp[:,None,None]*n[:,:,None]*n[:,None,:]+bp[:,None,None]*dn
    hf=(mu*N*f2)[:,None,None]*t[:,:,None]*t[:,None,:]
    jf=mu*(-bpp[:,None,None]*f1[:,None,None]*t[:,:,None]*n[:,None,:]
           +(N*f2)[:,None,None]*t[:,:,None]*ds[:,None,:]
           +(N*f1)[:,None,None]*dt)
    eigen,v=np.linalg.eigh(hn)
    metric=(v*np.maximum(eigen,0)[:,None,:])@v.transpose(0,2,1)+hf
    return dict(energy=float(weights@(b+mu*N*f0)),
        gradient=weights[:,None]*(normal_g+friction_g),normal_gradient=weights[:,None]*normal_g,
        friction_gradient=weights[:,None]*friction_g,
        jacobian=weights[:,None,None]*(hn+jf),metric=weights[:,None,None]*metric,
        friction_work=float(weights@(mu*N*f1*s)))


def blocks(values):
    ids=2*np.arange(len(values))[:,None]+np.arange(2)
    return coo_matrix((values.ravel(),(np.broadcast_to(ids[:,:,None],values.shape).ravel(),
        np.broadcast_to(ids[:,None,:],values.shape).ravel())),shape=(2*len(values),2*len(values))).tocsc()


class CurveBladeProblem(ViscoelasticProblem):
    """Experimental curve contact with native body self-contact and CCD."""
    def __init__(self,*,contact_spacing=.0025,contact_order=2,guard_facets=256,**kwargs):
        if kwargs.get('case','oblique')!='oblique' or kwargs.get('element','p1')!='p1':
            raise ValueError('Curve blade diagnostic supports oblique P1 only')
        if not np.isfinite(contact_spacing) or contact_spacing<=0 or contact_order not in (2,3,4):
            raise ValueError('Positive contact spacing and order 2, 3 or 4 required')
        if not isinstance(guard_facets,int) or guard_facets<32:
            raise ValueError('Conservative envelope requires at least 32 facets')
        super().__init__(**kwargs)
        self.contact_stiffness=float(kwargs.get('stiffness',BASELINE['barrier_stiffness_mpa']))*self.config.width_mm
        self.contact_spacing=float(contact_spacing);self.contact_order=contact_order
        self.guard_facets=guard_facets
        count=int(np.count_nonzero(self.surface<self.n));self.body_surface=self.surface[:count]
        be=np.asarray(self.cmesh.edges);be=be[np.all(be<count,axis=1)]
        self.body_edges=self.body_surface[be]
        self.self_mesh=self.ipc.CollisionMesh(self.rest[self.body_surface],be)
        self.self_dofs=(2*self.body_surface[:,None]+np.arange(2)).ravel()
        # Fixed quadrature: no geometry-dependent subdivision or weight derivatives.
        xi,gw=np.polynomial.legendre.leggauss(contact_order)
        lengths=np.linalg.norm(np.diff(self.rest[self.body_edges],axis=1)[:,0],axis=1)
        repeats=np.maximum(1,np.ceil(lengths/contact_spacing).astype(int))
        edge_ids=np.repeat(np.arange(len(be)),repeats*contact_order)
        fraction=np.concatenate([((np.arange(n)[:,None]+(xi+1)/2)/n).ravel() for n in repeats])
        weights=np.concatenate([np.tile(gw*L/(2*n),n) for L,n in zip(lengths,repeats)])
        shape=np.column_stack((1-fraction,fraction));nodes=self.body_edges[edge_ids]
        nq=len(weights);rows=np.repeat(np.arange(nq),2)
        self.Q=coo_matrix((shape.ravel(),(rows,nodes.ravel())),shape=(nq,len(self.rest))).tocsr()
        # Relative rigid translation is stored at the first blade node. Other blade
        # coordinates are prescribed and independently checked at acceptance.
        anchor=coo_matrix((-np.ones(nq),(np.arange(nq),np.full(nq,self.n))),shape=self.Q.shape)
        self.D=kron(self.Q+anchor,eye(2),format='csr');self.weights=weights
        self.guard_rest=self._guard_vertices(guard_facets)
        ge=np.column_stack((np.arange(len(self.guard_rest)),np.roll(np.arange(len(self.guard_rest)),-1)))+count
        gr=np.vstack([self.rest[self.body_surface],self.guard_rest])
        self.guard_mesh=self.ipc.CollisionMesh(gr,np.vstack([be,ge]))
        self.guard_mesh.can_collide=self.ipc.make_static_obstacle_filter(count)
        half=(np.pi-np.deg2rad(self.config.blade_tip_angle_deg))/(2*guard_facets)
        self.envelope_error_mm=self.options.tip_radius_mm*(1/np.cos(half)-1)
        if self.ipc.has_intersections(self.guard_mesh,gr):raise ValueError('Initial conservative guard intersects')
        data,D,_=self._blade_data(self.rest)
        if np.linalg.norm(D.T@data['gradient'].ravel())>BASELINE['force_tolerance_n']:
            raise ValueError('Analytic reference contact prestress')

    def _guard_vertices(self,facets):
        polygon=_blade_boundary_vertices(self.config,self.options,self.start)
        a=np.deg2rad(self.config.blade_tip_angle_deg/2);theta=np.linspace(-np.pi+a,-a,facets+1)
        center=polygon[2]-self.options.tip_radius_mm*np.array([np.cos(theta[0]),np.sin(theta[0])])
        mid=(theta[:-1]+theta[1:])/2;radius=self.options.tip_radius_mm/np.cos((theta[1]-theta[0])/2)
        intersections=center+radius*np.column_stack((np.cos(mid),np.sin(mid)))
        return np.vstack([polygon[:2],intersections,polygon[-2:]])

    def _guard_positions(self,x):
        return np.vstack([x[self.body_surface],self.guard_rest+x[self.n]-self.rest[self.n]])

    def _self_collisions(self,x):
        c=self.ipc.NormalCollisions();c.use_area_weighting=True
        c.collision_set_type=self.ipc.NormalCollisions.IMPROVED_MAX_APPROX
        c.build(self.self_mesh,x[self.body_surface],self.dhat)
        return c

    def lag(self,x):
        # Only native self-contact needs a toolkit lag object. The blade uses
        # fresh analytic geometry and a consistently differentiated force below.
        if self.config.friction_coefficient==0:return None
        f=self.ipc.TangentialCollisions()
        f.build(self.self_mesh,x[self.body_surface],self._self_collisions(x),self.barrier,self.config.friction_coefficient)
        return f

    def _blade_data(self,x):
        shift=x[self.n]-self.rest[self.n];points=self.Q@x-shift
        polygon=_blade_boundary_vertices(self.config,self.options,self.start)
        # Outside this expanded bounding box the compact barrier is exactly zero.
        minimum=polygon.min(axis=0);minimum[1]-=self.options.tip_radius_mm
        maximum=polygon.max(axis=0)
        selected=np.flatnonzero(np.all((points>=minimum-self.dhat)&(points<=maximum+self.dhat),axis=1))
        penetration,n,dn=rounded_blade_distance(points[selected],self.config,self.options,self.start,with_tangent=True)
        gap=-penetration
        ids=(2*selected[:,None]+np.arange(2)).ravel();D=self.D[ids,:]
        relative=(D@(x-self.previous).ravel()).reshape(-1,2)
        data=point_contact(gap,n,dn,relative,self.weights[selected],self.dhat,
            self.contact_stiffness,self.config.friction_coefficient,BASELINE['friction_smoothing_mm'])
        return data,D,float(gap.min()) if len(gap) else np.inf

    def evaluate(self,x,lag=None,tangent=True,projected=True,*,with_metric=False,with_pressure_blocks=False):
        if with_pressure_blocks:raise ValueError('Curve candidate has no mixed pressure blocks')
        r,k,dofs,energy,minj=self.material_bulk((x[:self.n]-self.rest[:self.n]).ravel(),with_tangent=tangent)
        g=np.zeros(self.total);np.add.at(g,dofs.ravel(),r.ravel())
        c=self._self_collisions(x);sx=x[self.body_surface];delta=(x-self.previous)[self.body_surface]
        f=self.lag(x) if self.config.friction_coefficient>0 else None
        energy+=self.barrier(c,self.self_mesh,sx)
        sg=self.barrier.gradient(c,self.self_mesh,sx)
        if f is not None:
            energy+=self.friction(f,self.self_mesh,delta);sg+=self.friction.gradient(f,self.self_mesh,delta)
        g[self.self_dofs]+=sg
        data,D,_=self._blade_data(x);energy+=data['energy'];g+=D.T@data['gradient'].ravel()
        A=M=None
        if tangent:
            rr=np.broadcast_to(dofs[:,:,None],k.shape).ravel();cc=np.broadcast_to(dofs[:,None,:],k.shape).ravel()
            def assemble(local,projection,blade,full):
                matrix=coo_matrix((local.ravel(),(rr,cc)),shape=(self.total,self.total)).tocsc()
                ch=self.barrier.hessian(c,self.self_mesh,sx,projection)
                if f is not None:
                    ch+=self.friction.hessian(f,self.self_mesh,delta,projection)
                    if full:
                        ch-=self.friction.force_jacobian(f,self.self_mesh,self.rest[self.body_surface],
                            (x-self.rest)[self.body_surface],delta,self.barrier,
                            self.ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS)
                ch=ch.tocoo()
                matrix+=coo_matrix((ch.data,(self.self_dofs[ch.row],self.self_dofs[ch.col])),shape=matrix.shape).tocsc()
                return matrix+D.T@blocks(blade)@D
            if not projected:A=assemble(k,self.ipc.PSDProjectionMethod.NONE,data['jacobian'],True)
            if projected or with_metric:
                e,v=np.linalg.eigh(k);kp=(v*np.maximum(e,0)[:,None,:])@v.transpose(0,2,1)
                M=assemble(kp,self.ipc.PSDProjectionMethod.CLAMP,data['metric'],False)
            if projected:A=M
        result=(float(energy),g,A,minj)
        return result+(M,) if with_metric else result

    def coupled_friction_tangent(self,x,lag,frozen_hessian):
        # evaluate(projected=False) already returns the complete fresh Jacobian.
        return frozen_hessian

    def safe_step(self,x,direction):
        F=np.eye(2)+np.einsum('eai,eaJ->eiJ',(x[:self.n]-self.rest[:self.n])[self.mesh.triangles],self.grad)
        D=np.einsum('eai,eaJ->eiJ',direction[:self.n][self.mesh.triangles],self.grad)
        return min(determinant_step(F,D),self.ipc.compute_collision_free_stepsize(
            self.guard_mesh,self._guard_positions(x),self._guard_positions(x+direction)))

    def acceptance(self,x,depth,iterations,outer_iterations):
        energy,g,_,minj=self.evaluate(x,tangent=False);norm=float(np.linalg.norm(g[self.free]))
        if not np.isfinite(norm) or norm>BASELINE['force_tolerance_n']:raise LoadStepFailure(f'Unbalanced curve contact: {norm:g} N')
        prescribed=self.rest[self.n:]-[0.,depth-self.start]
        if not np.allclose(x[self.n:],prescribed,rtol=0,atol=self.blade_boundary_roundoff_tolerance(depth)):
            raise RuntimeError('Curve candidate rigid translation boundary mismatch')
        gx=self._guard_positions(x)
        if minj<=0 or self.ipc.has_intersections(self.guard_mesh,gx):raise RuntimeError('Invalid curve guard geometry')
        c=self.ipc.NormalCollisions();c.build(self.guard_mesh,gx,self.dhat)
        gap=float(np.sqrt(c.compute_minimum_distance(self.guard_mesh,gx)))
        if gap<=0:raise RuntimeError('Nonpositive conservative whole-edge clearance')
        data,D,qgap=self._blade_data(x);work=data['friction_work'];f=self.lag(x)
        if f is not None:
            delta=(x-self.previous)[self.body_surface]
            work+=float(self.friction.gradient(f,self.self_mesh,delta)@delta.ravel())
        if not np.isfinite(work) or work< -1e-12:raise RuntimeError('Invalid curve friction work')
        normal=D.T@data['normal_gradient'].ravel()
        fz=float(-g.reshape(-1,2)[self.n:,1].sum());nfz=float(-normal.reshape(-1,2)[self.n:,1].sum())
        accepted=dict(depth_mm=depth,reaction_fz_n=fz,residual_n=norm,min_j=minj,iterations=iterations,
            friction_outer_iterations=outer_iterations,friction_work_n_mm=work,energy_n_mm=energy,
            reaction_normal_fz_n=nfz,reaction_friction_fz_n=fz-nfz,
            minimum_distance_mm=gap if np.isfinite(gap) else None,
            analytic_quadrature_min_gap_mm=qgap if np.isfinite(qgap) else None,
            conservative_envelope_error_mm=self.envelope_error_mm,
            contact_discretization='analytic_blade_reference_gauss',contact_spacing_mm=self.contact_spacing,
            contact_order=self.contact_order,guard_facets=self.guard_facets)
        return self.material_acceptance(x,depth,accepted)
