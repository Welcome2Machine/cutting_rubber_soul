"""Weak P1 boundary contact against the analytic moving rigid cutter.

Reference-area quadrature, continuous P1 multipliers and weighted nodal
Signorini/Coulomb conditions. This is an integral discretization, not a
pointwise constraint at each Gauss point. The independent whole-edge gap
audit remains necessary. Tractions are per reference area, in N/mm^2.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags, kron, eye
from .contact import coulomb_return


class SurfaceContact:
    def __init__(self, problem):
        self.p = problem
        boundary=problem.edges
        if hasattr(problem.mesh,'elements'):
            boundary=np.column_stack((boundary,problem.mesh.boundary_midpoints))
        self.displacement_nodes=np.unique(boundary)
        edges = np.searchsorted(self.displacement_nodes, boundary)
        abscissa, gauss_weights = np.polynomial.legendre.leggauss(problem.options.contact_quadrature_order)
        if problem.options.contact_discretization=='quadratic_integral':
            abscissa=np.r_[(abscissa-1)/2,(abscissa+1)/2]
            gauss_weights=np.tile(gauss_weights/2,2)
        nq=len(abscissa)
        shape = np.column_stack(((1-abscissa)/2, (1+abscissa)/2))
        if edges.shape[1]==3:
            t=(1+abscissa)/2
            shape=np.column_stack(((1-t)*(1-2*t),t*(2*t-1),4*t*(1-t)))
        lengths = np.linalg.norm(np.diff(problem.mesh.reference_xy[problem.edges],axis=1)[:,0],axis=1)
        self.w = (lengths[:,None]*problem.config.width_mm*gauss_weights/2).ravel()
        rows = np.broadcast_to(np.arange(len(self.w)).reshape(-1,nq,1), (len(edges),nq,edges.shape[1]))
        cols = np.broadcast_to(edges[:,None,:], rows.shape)
        self.Qu = coo_matrix((np.broadcast_to(shape,rows.shape).ravel(),
                            (rows.ravel(),cols.ravel())),shape=(len(self.w),len(self.displacement_nodes))).tocsr()
        self.Q=self.Qu
        if problem.options.contact_discretization=='quadratic_integral':
            t=(abscissa+1)/2
            multiplier_shape=np.column_stack((np.maximum(1-2*t,0),np.maximum(2*t-1,0),1-abs(2*t-1)))
            self.Q=coo_matrix((np.broadcast_to(multiplier_shape,rows.shape).ravel(),
                (rows.ravel(),cols.ravel())),shape=self.Qu.shape).tocsr()
        if problem.options.contact_discretization=='quadratic_nodes':
            self.Qu=eye(len(self.displacement_nodes),format='csr');self.Q=self.Qu
            self.w=problem.weights.copy()
        if problem.options.contact_discretization=='macro':
            from .macro_element import boundary_coordinates
            points=self.Qu@problem.mesh.reference_xy[self.displacement_nodes]
            parent_id,fraction=boundary_coordinates(problem.mesh,points)
            parent_nodes=problem.mesh.macro_boundary_edges[parent_id]
            cols=np.searchsorted(problem.contact_nodes,parent_nodes)
            self.Q=coo_matrix((np.column_stack((1-fraction,fraction)).ravel(),
                (np.repeat(np.arange(len(points)),2),cols.ravel())),shape=(len(points),len(problem.contact_nodes))).tocsr()
        self.Q2 = kron(self.Q,eye(2),format='csr')
        self.average = diags(1/problem.weights)@self.Q.T@diags(self.w)
        self.average2 = kron(self.average,eye(2),format='csr')
        ids = (2*self.displacement_nodes[:,None]+np.arange(2)).ravel()
        self.E = coo_matrix((np.ones(len(ids)),(np.arange(len(ids)),ids)),
                           shape=(len(ids),problem.mesh.dof_count)).tocsr()
        self.A = kron(self.Qu,eye(2),format='csr')@self.E

    def freeze(self, u, depth, lam, *, affine_normal=True):
        """Affine, work-conjugate contact geometry and fixed Tresca capacity.

        This is an auxiliary nonlinear subproblem, never a physical state.
        At its outer fixed point geometry and Coulomb roots are recovered.
        """
        from .initiation import rounded_blade_distance
        x = self.Qu@(self.p.mesh.reference_xy+u.reshape(-1,2))[self.displacement_nodes]
        gap,normal = rounded_blade_distance(x,self.p.config,self.p.options,depth)
        return dict(anchor=x.copy(),gap=gap.copy(),normal=normal.copy(),
                    capacity=np.maximum(lam[:,0],0.).copy(),affine_normal=affine_normal)

    def evaluate(self, u, previous, depth, old_depth, lam, *, with_tangent, formulation, roundoff_guard, frozen=None):
        # Import here to avoid a geometry/contact module cycle.
        from .initiation import rounded_blade_distance
        p,c = self.p,self.p.config
        x = self.Qu@(p.mesh.reference_xy+u.reshape(-1,2))[self.displacement_nodes]
        old = self.Qu@(p.mesh.reference_xy+previous.reshape(-1,2))[self.displacement_nodes]
        g,n,dn = rounded_blade_distance(x,c,p.options,depth,with_tangent=True)
        if frozen is not None and frozen['affine_normal']:
            n=frozen['normal']
            g=frozen['gap']-np.einsum('qi,qi->q',x-frozen['anchor'],n)
            dn=np.zeros_like(dn)
        t = np.column_stack((-n[:,1],n[:,0]))
        dt = np.stack((-dn[:,1,:],dn[:,0,:]),axis=1)
        if frozen is not None:
            t=np.column_stack((-frozen['normal'][:,1],frozen['normal'][:,0]))
            dt=np.zeros_like(dt)
        relative = x-old-np.array([0.,old_depth-depth])
        slip = np.einsum('qi,qi->q',relative,t)
        gap,slip_bar = self.average@g,self.average@slip
        projection_gap = gap.copy()
        projection_slip = slip_bar.copy()
        compliance=0.
        if frozen is not None and 'traction_reference' in frozen:
            compliance=frozen['compliance']
            projection_gap-=compliance*(lam[:,0]-frozen['traction_reference'][:,0])
            projection_slip-=compliance*(lam[:,1]-frozen['traction_reference'][:,1])
        if roundoff_guard:
            guard = 32*np.finfo(float).eps*max(1.,c.length_mm,c.height_mm,abs(depth),np.abs(x).max())
            projection_gap[np.abs(projection_gap)<=guard] = 0.
        kn,kt = c.contact_surface_penalty_n_mm3,p.options.tangential_stiffness_mpa_per_mm
        pressure = np.maximum(0,lam[:,0]+kn*projection_gap)
        active = pressure>0
        independent = (formulation or p.options.coulomb_formulation)=='jean_moreau'
        capacity = np.maximum(0,lam[:,0]) if independent else pressure
        if frozen is not None:
            capacity=frozen['capacity']
        shear,ds,dp,_,stick = coulomb_return(projection_slip,lam[:,1],capacity,c.friction_coefficient,kt,c.adhesive_shear_strength_n_mm2)
        if frozen is not None:
            dp=np.zeros_like(dp)  # the outer normal traction is held fixed
        traction = self.Q@lam
        fq = self.w[:,None]*(traction[:,0,None]*n-traction[:,1,None]*t)
        force = self.Qu.T@fq
        q = dict(force=force,pressure=pressure,shear=shear,gap=gap,slip=slip_bar,
                 global_force=self.E.T@force.ravel(),
                 normal=self.average@n,stick=stick,
                 dissipation=p.weights*lam[:,1]*slip_bar,
                 quadrature_gap=g,quadrature_traction=traction,quadrature_normal=n)
        if not with_tangent:
            return q
        # Local 2x2 block matrices, interleaved vector components.
        def blocks(values):
            size = len(values)
            ids = 2*np.arange(size)[:,None]+np.arange(2)
            return coo_matrix((values.ravel(),
                (np.broadcast_to(ids[:,:,None],values.shape).ravel(),
                 np.broadcast_to(ids[:,None,:],values.shape).ravel())),shape=(2*size,2*size)).tocsr()
        # Map surface displacement degrees of freedom to the full body.
        A = self.A
        geom = self.w[:,None,None]*(traction[:,0,None,None]*dn-traction[:,1,None,None]*dt)
        q['global_K'] = -(A.T@blocks(geom)@A)
        q['global_B'] = -(A.T@blocks(self.w[:,None,None]*np.stack((n,-t),axis=2))@self.Q2)
        dslip = t+np.einsum('qi,qij->qj',relative,dt)
        derivatives = np.stack((-n,dslip),axis=1)
        averaged_derivative = self.average2@blocks(derivatives)@A
        q['global_kinematic_u']=averaged_derivative
        returns = np.zeros((len(lam),2,2))
        returns[:,0,0] = kn*active
        returns[:,1,0] = 0. if independent else dp*kn*active
        returns[:,1,1] = ds
        q['global_C'] = blocks(returns)@averaged_derivative
        multiplier_derivative = np.zeros_like(returns)
        multiplier_derivative[:,0,0] = active.astype(float)*(1-kn*compliance)-1
        multiplier_derivative[:,1,0] = dp*((lam[:,0]>0) if independent else active)
        multiplier_derivative[:,1,1] = stick.astype(float)-1-ds*compliance
        q['global_D'] = blocks(multiplier_derivative)
        return q
