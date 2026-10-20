"""Isolated integral AL rigid-blade contact; no production-default changes.

Tractions and quadrature weights are per REFERENCE boundary area. The rigid
blade translates; rubber kinematics and the finite Maxwell law remain finite
strain. This is not the retired weak nodal multiplier/IPM implementation and
not an algebraically equivalent IPC barrier. Body self-contact retains IPC.
"""
import time
import heapq

import numpy as np
from scipy.sparse import coo_matrix, kron, eye, diags, bmat
from scipy.sparse.linalg import spsolve

from .inclined_loading import InclinedPathProblem
from .initiation import rounded_blade_distance
from .ipc_reference import BASELINE, LoadStepFailure, determinant_step
from .compact_p2_assembly import compact_matrices
from .curve_contact import blocks
from .separated_self_contact import SeparatedContactDomainError


def contact_constraints_satisfied(checks,multipliers):
    """Use the ORIGINAL final contact tolerances also for Newton termination."""
    values=np.array([checks['integrated_projection_defect_n'],
                     checks['maximum_cone_violation_mpa'],
                     multipliers[:,0].min(initial=0)])
    return bool(np.isfinite(values).all() and values[0]<=5e-5 and values[1]<=1e-8 and values[2]>=-1e-8)


def coulomb_projection(gap,slip,multipliers,penalty,mu):
    """AL root projection and selected generalized derivatives (gap, slip)."""
    ln,lt=multipliers.T;rho=penalty
    pressure=np.maximum(0.,ln-rho*gap);trial=lt+rho*slip
    shear=np.clip(trial,-mu*pressure,mu*pressure)
    active=pressure>0;stick=active & (abs(trial)<mu*pressure);sign=np.sign(trial)
    dk=np.zeros((len(ln),2,2));dl=np.zeros_like(dk)
    dk[:,0,0]=-rho*active;dk[:,1,0]=-rho*mu*sign*active*(~stick);dk[:,1,1]=rho*stick
    dl[:,0,0]=active;dl[:,1,0]=mu*sign*active*(~stick);dl[:,1,1]=stick
    return pressure,shear,dk,dl,stick


def augmented_traction(gap, normal, normal_jacobian, relative, multipliers,
                       penalty, mu, *, augmented=True):
    """Return tractions, resisting force and the full nonsymmetric Jacobian.

    gap>0 outside, normal points out of the rigid tool. lambda_t has the sign
    of material slip; external traction on rubber is p*n-tau*t. Multipliers
    are algebraic iterates, never committed material/friction history.
    """
    n=normal; dn=normal_jacobian
    t=np.column_stack((-n[:,1],n[:,0])); dt=np.stack((-dn[:,1,:],dn[:,0,:]),axis=1)
    slip=np.einsum('qi,qi->q',relative,t)
    ds=t+np.einsum('qi,qij->qj',relative,dt)
    pressure,shear,dk,_,stick=coulomb_projection(gap,slip,multipliers,penalty,mu)
    dp=dk[:,0,0,None]*n;dshear=dk[:,1,0,None]*n+dk[:,1,1,None]*ds
    if not augmented:
        pressure,shear=multipliers.T;dp=np.zeros_like(n);dshear=np.zeros_like(n)
    gradient=-pressure[:,None]*n+shear[:,None]*t
    jac=(-n[:,:,None]*dp[:,None,:]-pressure[:,None,None]*dn
         +t[:,:,None]*dshear[:,None,:]+shear[:,None,None]*dt)
    return dict(pressure=pressure,shear=shear,slip=slip,normal=n,tangent=t,
                gradient=gradient,jacobian=jac,stick=stick)


def convex_edge_gap_bounds(distance, a, b, *, accuracy=1e-7, maximum=80):
    """Certified lower/upper bounds using convex signed-distance supports.

    A convex body's signed distance is convex. Endpoint tangent planes bound
    it from below, including an interior penetration missed by both vertices.
    Returns a valid lower bound even if the iteration cap is reached; acceptance
    uses the LOWER bound, never the best sampled (upper-bound) gap.
    """
    if accuracy<=0 or maximum<1:raise ValueError('Positive bound accuracy/iteration limit required')
    v=np.asarray(b)-a
    points=np.array([a,b]);g,n,_=distance(points);s=n@v
    if not np.isfinite(np.r_[g,s]).all():raise LoadStepFailure('Nonfinite edge distance/support')
    padding=64*np.finfo(float).eps*max(1.,np.max(abs(points)),np.max(abs(g)),np.max(abs(s)))
    intervals=[(0.,1.,g[0],g[1],s[0],s[1])]
    upper=float(min(g)); evaluations=2
    for _ in range(maximum):
        lower=[]; split=[]
        for l,r,gl,gr,sl,sr in intervals:
            z=(gr-sr*r-gl+sl*l)/(sl-sr) if abs(sl-sr)>1e-20 else (l+r)/2
            z=float(np.clip(z,l,r))
            # The minimum of two affine supports can be at EITHER endpoint,
            # especially when both slopes have the same sign. Intersection
            # alone is not a lower bound on the interval minimum.
            candidates=np.array([l,z,r])
            values=np.maximum(gl+sl*(candidates-l),gr+sr*(candidates-r))
            j=int(np.argmin(values))
            lower.append(float(values[j])-padding)
            split.append(float(candidates[j]))
        i=int(np.argmin(lower));bound=float(lower[i])
        if bound>upper+padding:raise LoadStepFailure('Inconsistent convex edge distance bounds')
        if upper-bound<=accuracy:return bound,upper,evaluations
        l,r,gl,gr,sl,sr=intervals.pop(i);z=split[i]
        if not l+1e-14<z<r-1e-14:z=(l+r)/2
        gz,nz,_=distance(np.array([a+z*v]));gz=float(gz[0]);sz=float(nz[0]@v)
        if not np.isfinite([gz,sz]).all():raise LoadStepFailure('Nonfinite edge distance/support')
        upper=min(upper,gz);evaluations+=1
        intervals.extend(((l,z,gl,gz,sl,sz),(z,r,gz,gr,sz,sr)))
    return float(min(lower)),upper,evaluations


def lipschitz_edge_gap_bounds(distance,a,b,*,threshold,accuracy=1e-7,maximum=20000):
    """Independent whole-edge audit without convexity or distance derivatives.

    Endpoint cones of a 1-Lipschitz signed distance enclose its minimum.
    Stop when the penetration gate is decidable, not at a claimed precise
    penetration extremum. Iteration exhaustion fails closed.
    """
    v=np.asarray(b)-a;length=float(np.linalg.norm(v));g=distance(np.array([a,b]))[0]
    padding=64*np.finfo(float).eps*max(1.,np.max(abs(a)),np.max(abs(b)),length)
    def bound(l,r,gl,gr):
        if not np.isfinite([gl,gr]).all():raise LoadStepFailure('Nonfinite Lipschitz distance')
        span=(r-l)*length
        if abs(gl-gr)>span+padding:raise LoadStepFailure('Distance violates Lipschitz inequality')
        return min(gl,gr,(gl+gr-span)/2)-padding
    heap=[(bound(0.,1.,*g),0.,1.,float(g[0]),float(g[1]))]
    upper=float(min(g));evaluations=2
    for _ in range(maximum):
        low,l,r,gl,gr=heap[0]
        if upper-low<=accuracy or low>=threshold or upper<threshold:
            return float(low),upper,evaluations
        heapq.heappop(heap);t=(l+r)/2
        gm=float(distance(np.array([a+t*v]))[0][0]);evaluations+=1
        upper=min(upper,gm)
        heapq.heappush(heap,(bound(l,t,gl,gm),l,t,gl,gm))
        heapq.heappush(heap,(bound(t,r,gm,gr),t,r,gm,gr))
    raise LoadStepFailure('Independent whole-edge bound budget exhausted')


class AugmentedBladeProblem(InclinedPathProblem):
    """Weak edge-P0 contact on the SAME local-P2 body; monolithic AL roots.

    Multipliers live on original boundary edges, NOT at every Gauss point.
    Their constraints are reference-area averages. The independent pointwise
    gap bound is therefore essential and is not implied by weak equilibrium.
    """
    def __init__(self, *, contact_spacing=.00125, penalty=250.,
                 penetration_tolerance=.0002, **kwargs):
        super().__init__(**kwargs)
        if self.frame_rotation!=0 or self.nodal_volume is not None:
            raise ValueError('First AL candidate requires untranslated lab axes and no pressure blocks')
        if not all(np.isfinite(z) and z>0 for z in (contact_spacing,penalty,penetration_tolerance)):
            raise ValueError('AL spacing/penalty/penetration tolerance must be positive')
        self.contact_spacing=float(contact_spacing);self.penalty=float(penalty)
        self.penetration_tolerance=float(penetration_tolerance)
        self.contact_formulation='weak-P0-AL-rigid-blade-v2'
        count=int(np.count_nonzero(self.surface<self.n))
        self.body_surface=self.surface[:count]
        be=np.asarray(self.cmesh.edges);be=be[np.all(be<count,axis=1)]
        self.body_edges=self.body_surface[be]
        self.self_mesh=self.ipc.CollisionMesh(self.rest[self.body_surface],be)
        self.self_dofs=(2*self.body_surface[:,None]+np.arange(2)).ravel()
        self.qedges=np.empty(0,dtype=int);self.qfraction=np.empty(0);self.qweights=np.empty(0)
        self.multipliers=np.empty((0,2));self.last_geometry_audit={}
        self._set_quadrature(self.qedges,self.qfraction,self.qweights)
        self.predictor='affine'  # no duplicate long secant/affine trials

    def distance(self,points,x):
        shift=x[self.n]-self.blade_at_zero[0]
        penetration,n,dn=rounded_blade_distance(points-shift,self.config,self.options,0.,with_tangent=True)
        return -penetration,n,dn

    def _set_quadrature(self,edges,fractions,weights):
        self.qedges=np.asarray(edges,dtype=int);self.qfraction=np.asarray(fractions,dtype=float)
        self.qweights=np.asarray(weights,dtype=float)
        if (not np.isfinite(self.qfraction).all() or not np.isfinite(self.qweights).all()
                or len(edges)!=len(fractions) or len(edges)!=len(weights)
                or np.any(self.qedges<0) or np.any(self.qedges>=len(self.body_edges))):
            raise ValueError('Invalid AL quadrature checkpoint')
        xi,w=np.polynomial.legendre.leggauss(3)
        for e in np.unique(self.qedges):
            ids=np.flatnonzero(self.qedges==e);k=len(ids)//3
            if not k or len(ids)%3:raise ValueError('AL requires composite order-three quadrature')
            ref=np.linalg.norm(np.diff(self.rest[self.body_edges[e]],axis=0)[0])*self.config.width_mm
            np.testing.assert_allclose(self.qfraction[ids],((np.arange(k)[:,None]+(xi+1)/2)/k).ravel(),rtol=0,atol=1e-14)
            np.testing.assert_allclose(self.qweights[ids],np.tile(w*ref/(2*k),k),rtol=1e-13,atol=1e-14)
        nq=len(edges);nodes=self.body_edges[self.qedges]
        self.Q=coo_matrix((np.column_stack((1-self.qfraction,self.qfraction)).ravel(),
            (np.repeat(np.arange(nq),2),nodes.ravel())),shape=(nq,len(self.rest))).tocsr()
        anchor=coo_matrix((-np.ones(nq),(np.arange(nq),np.full(nq,self.n))),shape=self.Q.shape)
        self.D=kron(self.Q+anchor,eye(2),format='csr')
        self.contact_edges,inverse=np.unique(self.qedges,return_inverse=True)
        self.edgeQ=coo_matrix((np.ones(nq),(np.arange(nq),inverse)),shape=(nq,len(self.contact_edges))).tocsr()
        self.edge_area=np.asarray(self.edgeQ.T@self.qweights).ravel()
        self.average=diags(1/self.edge_area)@self.edgeQ.T@diags(self.qweights)

    def prepare_quadrature(self,x):
        """Freeze current-length subdivisions for this load, including a halo.

        Independent whole-edge auditing detects contact outside the halo and
        unacceptable current span. Such an event rejects the load; it is never
        hidden by evaluating only the available integration points.
        """
        edge=x[self.body_edges];length=np.linalg.norm(edge[:,1]-edge[:,0],axis=1)
        gap=self.distance(edge.mean(axis=1),x)[0]
        selected=np.flatnonzero(gap-length/2 < .05)
        xi,w=np.polynomial.legendre.leggauss(3)
        ref_length=np.linalg.norm(np.diff(self.rest[self.body_edges],axis=1)[:,0],axis=1)
        repeat=np.maximum(1,np.ceil(length[selected]/self.contact_spacing).astype(int))
        edges=np.repeat(selected,repeat*3)
        fractions=np.concatenate([((np.arange(k)[:,None]+(xi+1)/2)/k).ravel() for k in repeat]) if len(selected) else np.empty(0)
        weights=np.concatenate([np.tile(w*ref_length[e]*self.config.width_mm/(2*k),k)
                                for e,k in zip(selected,repeat)]) if len(selected) else np.empty(0)
        self._set_quadrature(edges,fractions,weights)
        self.multipliers=np.zeros((len(self.contact_edges),2))

    def contact_data(self,x,*,augmented=True):
        points=self.Q@x;gap,n,dn=self.distance(points,x)
        relative=(self.D@(x-self.previous).ravel()).reshape(-1,2)
        t=np.column_stack((-n[:,1],n[:,0]));dt=np.stack((-dn[:,1,:],dn[:,0,:]),axis=1)
        slip=np.einsum('qi,qi->q',relative,t);ds=t+np.einsum('qi,qij->qj',relative,dt)
        gapbar=self.average@gap;sbar=self.average@slip
        ln,lt=self.multipliers.T
        pressure,shear,dk,dl,stick=coulomb_projection(gapbar,sbar,self.multipliers,
                                                     self.penalty,self.config.friction_coefficient)
        pq=self.edgeQ@ln;tq=self.edgeQ@lt
        gradient=-pq[:,None]*n+tq[:,None]*t
        geom=-pq[:,None,None]*dn+tq[:,None,None]*dt
        B=self.D.T@blocks(self.qweights[:,None,None]*np.stack((-n,t),axis=2))@kron(self.edgeQ,eye(2),format='csr')
        kinematic=kron(self.average,eye(2),format='csr')@blocks(np.stack((n,ds),axis=1))@self.D
        constraint=self.multipliers-np.column_stack((pressure,shear))
        return dict(gap=gap,normal=n,tangent=t,pressure=pq,shear=tq,slip=slip,
            force=self.D.T@(self.qweights[:,None]*gradient).ravel(),
            jacobian=self.qweights[:,None,None]*geom,force_lambda=B,
            constraint=constraint,constraint_u=-blocks(dk)@kinematic,
            constraint_lambda=eye(2*len(ln),format='csc')-blocks(dl),
            gap_average=gapbar,slip_average=sbar,stick=stick,kinematic=kinematic)

    def _self_terms(self,x,tangent):
        c=self.ipc.NormalCollisions();c.use_area_weighting=True
        c.collision_set_type=self.ipc.NormalCollisions.IMPROVED_MAX_APPROX
        sx=x[self.body_surface];c.build(self.self_mesh,sx,self.dhat)
        f=self.ipc.TangentialCollisions()
        f.build(self.self_mesh,sx,c,self.barrier,self.config.friction_coefficient)
        delta=(x-self.previous)[self.body_surface]
        energy=self.barrier(c,self.self_mesh,sx)+self.friction(f,self.self_mesh,delta)
        fg=self.friction.gradient(f,self.self_mesh,delta)
        g=self.barrier.gradient(c,self.self_mesh,sx)+fg
        h=None
        if tangent:
            h=self.barrier.hessian(c,self.self_mesh,sx,self.ipc.PSDProjectionMethod.NONE)
            h+=self.friction.hessian(f,self.self_mesh,delta,self.ipc.PSDProjectionMethod.NONE)
            h-=self.friction.force_jacobian(f,self.self_mesh,self.rest[self.body_surface],
                (x-self.rest)[self.body_surface],delta,self.barrier,
                self.ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS)
            h=h.tocoo();h=coo_matrix((h.data,(self.self_dofs[h.row],self.self_dofs[h.col])),shape=(self.total,self.total)).tocsc()
        return float(energy),g,h,float(fg@delta.ravel())

    def evaluate(self,x,lag=None,tangent=True,projected=False,*,with_metric=False,
                 with_pressure_blocks=False,augmented=True):
        if with_pressure_blocks or with_metric or projected:
            raise ValueError('AL candidate uses its full nonsymmetric tangent, not an IPC PSD metric')
        r,k,dofs,energy,minj=self.material_bulk((x[:self.n]-self.rest[:self.n]).ravel(),with_tangent=tangent)
        g=np.zeros(self.total);np.add.at(g,dofs.ravel(),r.ravel())
        se,sg,sh,_=self._self_terms(x,tangent);g[self.self_dofs]+=sg
        q=self.contact_data(x,augmented=augmented);g+=q['force'];h=None
        if tangent:
            h=compact_matrices(k,dofs,self.total,self.linear_elements,self.active_elements)[0]
            h+=sh+self.D.T@blocks(q['jacobian'])@self.D
        # No claim that the Coulomb residual is a global energy gradient.
        return float(energy+se),g,h,minj

    def safe_step(self,x,direction):
        if self.enrichment is not None:
            fraction=self.enrichment.safe_step(x[:self.n]-self.rest[:self.n],direction[:self.n])
        else:
            F=np.eye(2)+np.einsum('eai,eaJ->eiJ',(x[:self.n]-self.rest[:self.n])[self.mesh.triangles],self.grad)
            D=np.einsum('eai,eaJ->eiJ',direction[:self.n][self.mesh.triangles],self.grad)
            fraction=determinant_step(F,D)
        collision_fraction=self.ipc.compute_collision_free_stepsize(self.self_mesh,
                    x[self.body_surface],(x+direction)[self.body_surface])
        self.last_safe_bounds=dict(J=float(fraction),self_ccd=float(collision_fraction))
        return min(fraction,collision_fraction)

    def geometry_audit(self,x,*,independent=False):
        edge=x[self.body_edges];length=np.linalg.norm(edge[:,1]-edge[:,0],axis=1)
        midgap=self.distance(edge.mean(axis=1),x)[0]
        possible=np.flatnonzero(midgap-length/2<=self.penetration_tolerance)
        present=set(self.qedges.tolist());minimum=np.inf;sampled=np.inf;count=0
        for e in possible:
            if independent:
                low,up,n=lipschitz_edge_gap_bounds(lambda z:self.distance(z,x),*edge[e],
                    threshold=-self.penetration_tolerance)
            else:low,up,n=convex_edge_gap_bounds(lambda z:self.distance(z,x),*edge[e])
            if low>up+1e-12:raise LoadStepFailure('Invalid whole-edge gap interval')
            minimum=min(minimum,low);sampled=min(sampled,up);count+=n
            if low<=0 and e not in present:
                raise LoadStepFailure('AL active contact outside quadrature halo')
            if e in present:
                repeats=np.count_nonzero(self.qedges==e)/3
                if length[e]/repeats>1.5*self.contact_spacing:
                    raise LoadStepFailure('AL current integration cells stretched beyond qualified span')
        if minimum < -self.penetration_tolerance:
            raise LoadStepFailure(f'AL whole-edge penetration {max(0.,-minimum):g} mm exceeds tolerance')
        if self.ipc.has_intersections(self.self_mesh,x[self.body_surface]):
            raise LoadStepFailure('AL body boundary self-intersection')
        return dict(minimum_signed_gap_lower_mm=float(minimum) if np.isfinite(minimum) else None,
            sampled_minimum_gap_upper_mm=float(sampled) if np.isfinite(sampled) else None,
            penetration_bound_mm=max(0.,float(-minimum)),gap_bound_accuracy_mm=1e-7,
            method='independent-Lipschitz-gate' if independent else 'convex-support-endpoints-v2',
            gate_stopped=independent,
            audited_edges=len(possible),distance_evaluations=count,
            penetration_tolerance_mm=self.penetration_tolerance,
            strict_blade_CCD=False,body_self_CCD=True)

    def constraint_audit(self,x):
        q=self.contact_data(x);change=q['constraint']
        # L1 integrated traction fixed-point residual cannot cancel across nodes.
        defect=float(self.edge_area@np.linalg.norm(change,axis=1))
        cone=np.maximum(abs(self.multipliers[:,1])-self.config.friction_coefficient*self.multipliers[:,0],0.)
        return q,dict(integrated_projection_defect_n=defect,
            maximum_cone_violation_mpa=float(cone.max(initial=0)),
            quadrature_penetration_mm=max(0.,float(-q['gap'].min(initial=np.inf))))

    def _equilibrate(self,x,deadline):
        for iteration in range(45):
            if time.monotonic()>deadline:raise TimeoutError('AL displacement solve wall budget')
            tick=time.perf_counter();_,g,h,j=self.evaluate(x)
            contact,checks=self.constraint_audit(x);weights=np.repeat(self.edge_area,2)
            r=np.r_[g[self.free],weights*contact['constraint'].ravel()]
            norm=float(np.linalg.norm(r));physical=float(np.linalg.norm(g[self.free]));self.trial=x.copy()
            print(f'AL_NEWTON iteration={iteration} residual_N={physical:.9g} contact_defect_N={checks["integrated_projection_defect_n"]:.9g} minJ={j:.9g} assembly_s={time.perf_counter()-tick:.6f}',flush=True)
            if physical<=BASELINE['force_tolerance_n']*.25 and contact_constraints_satisfied(checks,self.multipliers):return x,iteration
            A=bmat([[h[self.free][:,self.free],contact['force_lambda'][self.free]],
                    [diags(weights)@contact['constraint_u'][:,self.free],
                     diags(weights)@contact['constraint_lambda']]],format='csc')
            scale=np.r_[np.maximum(abs(h.diagonal()[self.free]),1.),np.zeros(len(weights))]
            accepted=False
            for shift in (0.,1e-5,1e-3,.1,1.):
                if time.monotonic()>deadline:raise TimeoutError('AL linear solve wall budget')
                matrix=A+diags(shift*scale+np.r_[np.full(len(self.free),1e-9),np.zeros(len(weights))]);du=self.linear_direction(matrix,-r)
                if not np.isfinite(du).all():continue
                lin=float(np.linalg.norm(matrix@du+r))
                if lin>max(1e-8*norm,1e-10):raise RuntimeError('AL inaccurate linear direction')
                slope=float(r@(A@du))
                if slope>=0:continue
                direction=np.zeros(self.total);direction[self.free]=du[:len(self.free)];direction=direction.reshape(-1,2)
                delta_lambda=du[len(self.free):].reshape(-1,2);before_lambda=self.multipliers.copy()
                alpha=min(1.,self.safe_step(x,direction),.01/max(abs(direction).max(),1e-30))
                for back in range(14):
                    if time.monotonic()>deadline:raise TimeoutError('AL line search wall budget')
                    trial=x+alpha*direction
                    self.multipliers=before_lambda+alpha*delta_lambda
                    try:
                        _,gt,_,_=self.evaluate(trial,tangent=False)
                    except SeparatedContactDomainError:
                        self.multipliers=before_lambda.copy()
                        print(f'AL_DOMAIN_REJECT alpha={alpha:.9g} reason=nonadjacent_proximity',flush=True)
                        alpha*=.5
                        continue
                    ct=self.contact_data(trial)['constraint']
                    nt=float(np.linalg.norm(np.r_[gt[self.free],weights*ct.ravel()]))
                    if nt*nt <= norm*norm+2e-4*alpha*slope:
                        x=trial;accepted=True
                        print(f'AL_UPDATE alpha={alpha:.9g} shift={shift:g} backtracks={back} joint_residual_N={nt:.9g}',flush=True)
                        break
                    self.multipliers=before_lambda
                    alpha*=.5
                if accepted:break
            if not accepted:raise LoadStepFailure('AL nonsymmetric Newton residual line search failed')
        raise LoadStepFailure('AL displacement iteration limit')

    def linear_direction(self,matrix,rhs):
        """Default CPU direct solve; the caller always checks the true residual."""
        return spsolve(matrix,rhs)

    def step(self,x,old,depth,deadline):
        # Transaction: only material acceptance commits physical history.
        self.previous=x.copy();self.previous_material_metric=self.material_metric.copy()
        self.material_dt=(depth-old)/self.loading_rate
        if self.material_dt<=0:raise ValueError('Positive physical time increment required')
        value=self.affine_loading_candidate(x,old,depth)
        if self.safe_step(x,value-x)<1.-1e-10:raise LoadStepFailure('AL initial material/self-contact trajectory invalid')
        self.prepare_quadrature(value);self.trial=value.copy()
        value,used=self._equilibrate(value,deadline)
        record=self.acceptance(value,depth,used,1)
        material=self.material_response(value)
        self.material_metric=material['metric'].copy()
        self.material_dissipation+=record['material']['viscous_dissipation_increment_n_mm']
        record['material']['cumulative_viscous_dissipation_n_mm']=self.material_dissipation
        self._remember_predictor_acceptance(value,depth,old)
        return value,record

    def acceptance(self,x,depth,iterations,outer_iterations):
        # Exact multiplier equilibrium, NOT the penalty-augmented surrogate.
        energy,g,_,minj=self.evaluate(x,tangent=False,augmented=False)
        norm=float(np.linalg.norm(g[self.free]));_,constraint=self.constraint_audit(x)
        if not np.isfinite(norm) or norm>BASELINE['force_tolerance_n'] or minj<=0:
            raise LoadStepFailure(f'AL physical equilibrium failed: {norm:g} N')
        if not contact_constraints_satisfied(constraint,self.multipliers):
            raise LoadStepFailure('AL contact complementarity/cone failed')
        if not np.allclose(x[self.n:],self.prescribed_blade_positions(depth),rtol=0,atol=self.blade_boundary_roundoff_tolerance(depth)):
            raise RuntimeError('AL prescribed blade mismatch')
        if not np.array_equal(x[self.mesh.bottom_nodes],self.rest[self.mesh.bottom_nodes]):
            raise RuntimeError('AL fixed support mismatch')
        geo=self.geometry_audit(x);q=self.contact_data(x,augmented=False)
        work=float(self.qweights@(q['shear']*q['slip']))+self._self_terms(x,False)[3]
        if not np.isfinite(work) or work < -1e-10:raise LoadStepFailure('AL negative friction work')
        force=-g.reshape(-1,2)[self.n:].sum(axis=0)
        normal=self.D.T@(-self.qweights[:,None]*q['pressure'][:,None]*q['normal']).ravel()
        nfz=float(-normal.reshape(-1,2)[self.n:,1].sum())
        record=dict(depth_mm=float(depth),reaction_x_n=float(force[0]),reaction_fz_n=float(force[1]),
            residual_n=norm,min_j=float(minj),iterations=iterations,friction_outer_iterations=outer_iterations,
            energy_n_mm=energy,friction_work_n_mm=work,reaction_normal_fz_n=nfz,
            reaction_friction_fz_n=float(force[1]-nfz),minimum_distance_mm=geo['minimum_signed_gap_lower_mm'],
            al_constraints=constraint,al_geometry=geo,contact_discretization=self.contact_formulation,
            quadrature_points=len(self.qweights),max_pressure_mpa=float(q['pressure'].max(initial=0)))
        reaction=force;N=float(reaction@self.en);T=float(-reaction@self.es)
        direction=-self.en if depth<=0 else self.loading_direction;ratio=0. if depth<=0 else self.tangent_ratio
        record['motion']=dict(normal_depth_mm=float(depth),tangent_travel_mm=max(0.,depth)*self.tangent_ratio,
            laboratory_translation_mm=(self.prescribed_blade_positions(depth)[0]-self.blade_at_zero[0]).tolist(),
            laboratory_reaction_xy_n=force.tolist(),normal_reaction_n=N,tangent_resistance_n=T,
            generalized_resistance_n=N+ratio*T,actuator_power_nmm_s=float(-reaction@direction*self.loading_rate))
        return self.material_acceptance(x,depth,record)

    def checkpoint_data(self):
        return dict(super().checkpoint_data(),al_edges=self.qedges,al_fractions=self.qfraction,
                    al_weights=self.qweights,al_multipliers=self.multipliers)

    def audit_state(self,state,record):
        saved=(self.qedges.copy(),self.qfraction.copy(),self.qweights.copy(),self.multipliers.copy())
        try:
            self._set_quadrature(state['al_edges'],state['al_fractions'],state['al_weights'])
            self.multipliers=state['al_multipliers'].copy()
            if self.multipliers.shape!=(len(self.contact_edges),2) or not np.isfinite(self.multipliers).all():
                raise ValueError('Invalid AL multiplier checkpoint')
            row=super().audit_state(state,record)
            previous=self.previous
            try:
                self.previous=state['previous_positions']
                row['contact_constraints']=self.constraint_audit(state['positions'])[1]
                row['geometry']=self.geometry_audit(state['positions'])
                row['geometry_lipschitz']=self.geometry_audit(state['positions'],independent=True)
            finally:self.previous=previous
            return row
        finally:
            self._set_quadrature(*saved[:3]);self.multipliers=saved[3]
