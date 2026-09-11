"""Finite-strain P2 displacement with full integration or mixed P1 pressure.

The pressure mass is consistent, not lumped. Pressure is eliminated only in
the nonlinear residual; its increment remains in the sparse Newton system.
Refinement transfers the previous P2 displacement polynomial exactly.
"""
from dataclasses import replace
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .model_config import FEMMesh
from .initiation import IntactProblem, intact_mesh
from .material import response


L=np.array([[1/3]*3,[.059715871789770,.470142064105115,.470142064105115],
    [.470142064105115,.059715871789770,.470142064105115],
    [.470142064105115,.470142064105115,.059715871789770],
    [.797426985353087,.101286507323456,.101286507323456],
    [.101286507323456,.797426985353087,.101286507323456],
    [.101286507323456,.101286507323456,.797426985353087]])
W=np.array([.225]+[.132394152788506]*3+[.125939180544827]*3)


def shape(l):
    return np.concatenate((l*(2*l-1),4*l[..., [0,1,2]]*l[..., [1,2,0]]),axis=-1)


def quadratic_minimum(values):
    """Exact min of a scalar quadratic over a triangle from its six nodal values."""
    a,b,c,ab,bc,ca=values.T
    rr=2*(a+b-2*ab);ss=2*(a+c-2*ca)
    r=b-a-rr;s=c-a-ss;rs=4*(bc-a-.5*(r+s)-.25*(rr+ss))
    low=np.minimum(np.minimum(a,b),c)
    for constant,linear,quad in ((a,r,rr),(a,s,ss),(c,r-s+rs-2*ss,rr+ss-rs)):
        t=np.divide(-linear,2*quad,out=np.zeros_like(quad),where=abs(quad)>1e-30)
        inside=(quad>0)&(t>0)&(t<1)
        low=np.minimum(low,np.where(inside,constant+linear*t+quad*t*t,np.inf))
    det=4*rr*ss-rs*rs
    x=np.divide(rs*s-2*ss*r,det,out=np.zeros_like(det),where=abs(det)>1e-30)
    y=np.divide(rs*r-2*rr*s,det,out=np.zeros_like(det),where=abs(det)>1e-30)
    inside=(rr>0)&(det>0)&(x>0)&(y>0)&(x+y<1)
    low=np.minimum(low,np.where(inside,a+r*x+s*y+rr*x*x+ss*y*y+rs*x*y,np.inf))
    return low


def quadratic_mesh(base,boundary):
    from .macro_element import refine_macro_mesh
    mesh,_=refine_macro_mesh(base,boundary)
    children=mesh.triangles.reshape(-1,4,3)
    mesh.elements=np.column_stack((base.triangles,children[:,0,1],children[:,1,2],children[:,0,2]))
    mesh.plot_triangles=mesh.triangles.copy()
    mesh.triangles=base.triangles.copy()
    mesh.pressure_node_count=base.node_count
    lookup={tuple(sorted((int(a),int(b)))):int(m) for e in mesh.elements
        for a,b,m in ((e[0],e[1],e[3]),(e[1],e[2],e[4]),(e[2],e[0],e[5]))}
    mesh.boundary_midpoints=np.array([lookup[tuple(sorted(edge))] for edge in boundary],int)
    for name in ('macro_triangles','macro_node_count','macro_parents'):
        delattr(mesh,name)
    return mesh


class QuadraticProblem(IntactProblem):
    def __init__(self,config,options,strength_surface=None):
        config.validate();options.validate(config)
        self.config,self.options,self.strength_surface=config,options,strength_surface
        coarse=replace(options,volumetric_discretization='element',contact_discretization='integral',
            surface_spacing_mm=2*options.surface_spacing_mm,refinement_mm=2*options.refinement_mm)
        base,edges=intact_mesh(replace(config,mesh_size_mm=2*config.mesh_size_mm),coarse,
            fine_halfwidth_mm=max(2*options.refinement_mm,8*options.tip_radius_mm))
        self.mesh=quadratic_mesh(base,edges)
        self._rebuild_boundary(edges)

    def step(self,previous,old_depth,depth,dt,old_state):
        from .initiation import ContactSolveError
        velocity=getattr(self,'_accepted_velocity',None)
        result=None
        failures=[]
        spatial=getattr(self,'_spatial_predictor',None)
        if spatial is not None:
            reference,seed,warm,target=spatial
            self._spatial_predictor=None
            if target==depth and np.array_equal(reference,previous):
                try:
                    result=self._mixed_step(previous,old_depth,depth,dt,old_state,initial=seed,initial_multipliers=warm)
                    result[2]['contact_solver']+=' (refined equilibrium predictor)'
                except ContactSolveError as exc:failures.append(exc)
                except FloatingPointError:pass
        if (result is None and self.options.contact_discretization in ('quadratic_nodes','quadratic_integral') and velocity is not None
                and abs(getattr(self,'_velocity_depth',np.inf)-old_depth)<1e-12
                and depth>old_depth and getattr(self,'_velocity_loading',False)):
            seed=previous+(depth-old_depth)*velocity
            seed[self.fixed]=previous[self.fixed]
            try:
                result=self._mixed_step(previous,old_depth,depth,dt,old_state,initial=seed)
                result[2]['contact_solver']+=' (accepted-path secant predictor)'
            except ContactSolveError as exc:failures.append(exc)
            except FloatingPointError:pass
        if result is None:result=super().step(previous,old_depth,depth,dt,old_state,predictor_failures=failures)
        q=self.bulk_response(result[0])
        error=np.max(abs(self.config.bulk_modulus_n_mm2*(q['J']-1)-q['pressure']),axis=1)
        result[2]['max_volume_projection_error_mpa']=float(error.max())
        self._accepted_velocity=(result[0]-previous)/(depth-old_depth) if depth!=old_depth else np.zeros_like(previous)
        self._velocity_depth=depth;self._velocity_loading=depth>old_depth
        return result

    def _rebuild_boundary(self,edges):
        # The base cache supplies boundary weights, fixed DOFs and averaging.
        super()._rebuild_boundary(edges)
        if self.options.contact_discretization in ('quadratic_nodes','quadratic_integral'):
            from .surface_contact import SurfaceContact
            boundary=np.column_stack((edges,self.mesh.boundary_midpoints))
            lengths=np.linalg.norm(np.diff(self.mesh.reference_xy[edges],axis=1)[:,0],axis=1)
            rule=[1/6,1/6,2/3] if self.options.contact_discretization=='quadratic_nodes' else [.25,.25,.5]
            weights=lengths[:,None]*self.config.width_mm*np.array(rule)
            area=np.zeros(self.mesh.node_count);np.add.at(area,boundary.ravel(),weights.ravel())
            self.contact_nodes=np.unique(boundary);self.weights=area[self.contact_nodes]
            self.cdofs=2*self.contact_nodes[:,None]+np.arange(2)
            self.surface_contact=SurfaceContact(self)
        mesh=self.mesh;area,grad,_=mesh.bulk_data
        self.grad=np.concatenate(((4*L[None,:,:,None]-1)*grad[:,None,:,:],
            4*(L[None,:, [1,2,0],None]*grad[:,None,[0,1,2],:]
               +L[None,:, [0,1,2],None]*grad[:,None,[1,2,0],:])),axis=2)
        self.volumes=area*self.config.width_mm
        vertices=np.vstack((np.eye(3),[[.5,.5,0],[0,.5,.5],[.5,0,.5]]))
        self.vertex_grad=np.concatenate(((4*vertices[None,:,:,None]-1)*grad[:,None,:,:],
            4*(vertices[None,:, [1,2,0],None]*grad[:,None,[0,1,2],:]
               +vertices[None,:, [0,1,2],None]*grad[:,None,[1,2,0],:])),axis=2)
        self.dofs=(2*mesh.elements[:,:,None]+np.arange(2)).reshape(-1,12)
        sizes=(len(self.dofs),12,12)
        self.rows=np.broadcast_to(self.dofs[:,:,None],sizes).ravel()
        self.cols=np.broadcast_to(self.dofs[:,None,:],sizes).ravel()
        if self.options.volumetric_discretization=='taylor_hood':
            tri=mesh.triangles;mass=self.volumes[:,None,None]*(np.ones((3,3))+np.eye(3))/(12*self.config.bulk_modulus_n_mm2)
            self.mass=coo_matrix((mass.ravel(),(np.broadcast_to(tri[:,:,None],mass.shape).ravel(),
                np.broadcast_to(tri[:,None,:],mass.shape).ravel())),shape=(mesh.pressure_node_count,)*2).tocsc()
            self.mass_factor=splu(self.mass)

    def jacobian_bounds(self,u):
        nodal_F=np.eye(2)+np.einsum('eai,eqaJ->eqiJ',u.reshape(-1,2)[self.mesh.elements],self.vertex_grad)
        values=np.linalg.det(nodal_F)
        return quadratic_minimum(values),-quadratic_minimum(-values)

    def bulk_response(self,u,tangent=False):
        lower,upper=self.jacobian_bounds(u)
        minimum=float(lower.min())
        if minimum<=0:raise FloatingPointError('P2 element has nonpositive Jacobian inside its reference triangle')
        F=np.eye(2)+np.einsum('eai,eqaJ->eqiJ',u.reshape(-1,2)[self.mesh.elements],self.grad)
        q=response(F.reshape(-1,2,2),self.config,tangent=tangent)
        J=q['J'].reshape(-1,7);K=self.config.bulk_modulus_n_mm2
        if self.options.volumetric_discretization=='quadratic_full':
            q['P']=q['P'].reshape(-1,7,2,2)
            if tangent:q['tangent']=q['tangent'].reshape(-1,7,2,2,2,2)
            q.update(F=F,J=J,pressure=K*(J-1),min_j=minimum,max_j=float(upper.max()),
                bulk_energy=float(np.einsum('e,q,eq->',self.volumes,W,q['energy'].reshape(-1,7))))
            return q
        b=np.zeros(self.mesh.pressure_node_count)
        local=np.einsum('e,q,qa,eq->ea',self.volumes,W,L,J-1)
        np.add.at(b,self.mesh.triangles.ravel(),local.ravel())
        pressure=self.mass_factor.solve(b);pq=pressure[self.mesh.triangles]@L.T
        cofactor=np.empty_like(F)
        cofactor[:,:,0,0]=F[:,:,1,1];cofactor[:,:,1,1]=F[:,:,0,0]
        cofactor[:,:,0,1]=-F[:,:,1,0];cofactor[:,:,1,0]=-F[:,:,0,1]
        difference=pq-K*(J-1)
        q['P']=q['P'].reshape(-1,7,2,2)+difference[:,:,None,None]*cofactor
        q['cauchy']+=difference.ravel()[:,None,None]*np.eye(3)
        energy=float(np.einsum('e,q,eq->',self.volumes,W,q['energy'].reshape(-1,7)-K/2*(J-1)**2)+.5*b@pressure)
        if tangent:
            h=np.zeros((2,2,2,2));h[0,0,1,1]=h[1,1,0,0]=1;h[0,1,1,0]=h[1,0,0,1]=-1
            q['tangent']=q['tangent'].reshape(-1,7,2,2,2,2)+difference[:,:,None,None,None,None]*h-K*cofactor[:,:,:,:,None,None]*cofactor[:,:,None,None,:,:]
        q.update(F=F,J=J,pressure=pq,cofactor=cofactor,bulk_energy=energy,min_j=minimum,max_j=float(upper.max()))
        return q

    def assemble(self,u,previous,depth,old_depth,dt,multipliers,old_shear=None,*,mixed=False,formulation=None,with_tangent=True,roundoff_guard=True,pressure_blocks=False,frozen_contact=None):
        if dt<=0:raise ValueError('Physical time increment must be positive')
        q=self.bulk_response(u,with_tangent);m=self.mesh;c=self.config
        br=np.einsum('e,q,eqiJ,eqaJ->eai',self.volumes,W,q['P'],self.grad).reshape(-1,12)
        r=np.zeros(m.dof_count);np.add.at(r,self.dofs.ravel(),br.ravel())
        if old_shear is None:old_shear=np.zeros(len(multipliers))
        contact=self.surface_contact.evaluate(u,previous,depth,old_depth,np.column_stack((multipliers,old_shear)),
            with_tangent=with_tangent,formulation=formulation,roundoff_guard=roundoff_guard,frozen=frozen_contact)
        r-=contact['global_force'];matrix=None
        if with_tangent:
            bk=np.einsum('e,q,eqaJ,eqiJkL,eqbL->eaibk',self.volumes,W,self.grad,q['tangent'],self.grad,optimize=True).reshape(-1,12,12)
            matrix=coo_matrix((bk.ravel(),(self.rows,self.cols)),shape=(m.dof_count,)*2).tocsr()+contact['global_K']
            if self.options.volumetric_discretization=='taylor_hood':
                Hlocal=np.einsum('e,q,qb,eqiJ,eqaJ->ebai',self.volumes,W,L,q['cofactor'],self.grad).reshape(-1,3,12)
                H=coo_matrix((Hlocal.ravel(),(np.broadcast_to(m.triangles[:,:,None],Hlocal.shape).ravel(),
                    np.broadcast_to(self.dofs[:,None,:],Hlocal.shape).ravel())),shape=(m.pressure_node_count,m.dof_count)).tocsr()
                contact.update(pressure_coupling=H,pressure_mass_matrix=self.mass)
            # Matrix is the local block; H.T M^-1 H is represented by the
            # pressure saddle block. Never materialize a dense Schur matrix.
        force=contact['force']
        d=dict(reaction_fz_n=float(-force[:,1].sum()),reaction_fx_n=float(-force[:,0].sum()),
            min_j=q['min_j'],max_j=q['max_j'],bulk_energy_n_mm=q['bulk_energy'],
            max_pressure_mpa=float(contact['pressure'].max(initial=0)),max_pressure_node=int(self.contact_nodes[contact['pressure'].argmax()]),
            max_incremental_displacement_mm=float(np.linalg.norm((u-previous).reshape(-1,2),axis=1).max(initial=0)),
            max_penetration_mm=float(contact['gap'].max(initial=0)),stick_nodes=int(contact['stick'].sum()),
            slip_nodes=int(((contact['pressure']>0)&~contact['stick']).sum()),friction_dissipation_n_mm=float(contact['dissipation'].sum()),
            global_force_balance_error_n=float(np.linalg.norm(r[self.fixed].reshape(-1,2).sum(axis=0)+force.sum(axis=0))))
        return r,matrix,contact,d

    def refine_contact(self,previous,old_state,edge_indices,*,trial=None):
        return self._refine(previous,old_state,edge_indices,trial=trial)

    def refine_bulk(self,previous,old_state,elements,*,trial=None):
        tri=self.mesh.triangles[np.asarray(elements,dtype=int)]
        edges=np.stack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]),axis=1)
        lengths=np.linalg.norm(np.diff(self.mesh.reference_xy[edges],axis=2)[:,:,0],axis=2)
        longest=edges[np.arange(len(tri)),lengths.argmax(axis=1)]
        return self._refine(previous,old_state,[],internal_edges=np.unique(np.sort(longest,axis=1),axis=0),trial=trial)

    def _refine(self,previous,old_state,edge_indices,*,internal_edges=None,trial=None):
        oldmesh=self.mesh;n=oldmesh.pressure_node_count
        coarse=FEMMesh(oldmesh.reference_xy[:n],oldmesh.local_sn[:n],oldmesh.triangles.copy(),
            np.empty((0,2),int),np.empty(0),np.flatnonzero(np.isclose(oldmesh.local_sn[:n,1],0)),np.empty(0,int),np.zeros(n,int))
        coarse.macro_triangles=coarse.triangles.copy();coarse.macro_parents=np.arange(len(coarse.triangles))
        coarse.macro_node_count=n;coarse.macro_boundary_edges=self.edges.copy()
        temp=IntactProblem.__new__(IntactProblem);temp.config=self.config;temp.strength_surface=None
        temp.options=replace(self.options,contact_discretization='integral',volumetric_discretization='element')
        temp.mesh=coarse;temp._rebuild_boundary(self.edges.copy())
        coarse_state=old_state[np.searchsorted(self.contact_nodes,temp.contact_nodes)]
        _,warm=temp.refine_contact(np.zeros(2*n),coarse_state,edge_indices,internal_edges=internal_edges)
        newmesh=quadratic_mesh(temp.mesh,temp.edges)
        parents=temp.mesh.macro_parents
        oldtri=oldmesh.reference_xy[oldmesh.triangles[parents]]
        dm=np.stack((oldtri[:,1]-oldtri[:,0],oldtri[:,2]-oldtri[:,0]),axis=2)
        rs=np.einsum('eij,eaj->eai',np.linalg.inv(dm),newmesh.reference_xy[newmesh.elements]-oldtri[:,None,0])
        bary=np.concatenate((1-rs.sum(axis=2,keepdims=True),rs),axis=2)
        values=np.einsum('eab,ebi->eai',shape(bary),previous.reshape(-1,2)[oldmesh.elements[parents]])
        new=np.zeros((newmesh.node_count,2));new[newmesh.elements.ravel()]=values.reshape(-1,2)
        if np.max(abs(new[newmesh.elements]-values))>1e-10:raise ValueError('Inconsistent P2 history transfer')
        if hasattr(self,'_accepted_velocity'):
            sampled=np.einsum('eab,ebi->eai',shape(bary),self._accepted_velocity.reshape(-1,2)[oldmesh.elements[parents]])
            velocity=np.zeros_like(new);velocity[newmesh.elements.ravel()]=sampled.reshape(-1,2)
            self._accepted_velocity=velocity.ravel()
        self.mesh=newmesh;self._rebuild_boundary(temp.edges)
        if self.options.contact_discretization in ('quadratic_nodes','quadratic_integral'):
            # Contact multipliers are a Newton warm start, not constitutive
            # internal variables. Transfer the old nonnegative piecewise-linear
            # traction on half edges; this also preserves the Coulomb cone.
            from scipy.spatial import cKDTree
            old_boundary=np.stack((np.column_stack((oldmesh.macro_boundary_edges[:,0],oldmesh.boundary_midpoints)),
                np.column_stack((oldmesh.boundary_midpoints,oldmesh.macro_boundary_edges[:,1]))),axis=1).reshape(-1,2)
            points=oldmesh.reference_xy[old_boundary];_,near=cKDTree(points.mean(axis=1)).query(newmesh.reference_xy[self.contact_nodes],k=min(8,len(points)))
            near=np.asarray(near).reshape(len(self.contact_nodes),-1);a=points[near,0];v=points[near,1]-a
            fraction=np.clip(np.sum((newmesh.reference_xy[self.contact_nodes,None]-a)*v,axis=2)/np.sum(v*v,axis=2),0,1)
            error=np.sum((newmesh.reference_xy[self.contact_nodes,None]-a-fraction[:,:,None]*v)**2,axis=2)
            choice=error.argmin(axis=1);ids=np.arange(len(choice));pick=near[ids,choice];t=fraction[ids,choice]
            if error[ids,choice].max()>1e-16:raise ValueError('Cannot transfer quadratic boundary warm start')
            old_nodes=np.unique(old_boundary);values=old_state[np.searchsorted(old_nodes,old_boundary[pick])]
            warm=(1-t[:,None])*values[:,0]+t[:,None]*values[:,1]
            if trial is not None:
                trial_u,trial_lam,target=trial
                sampled=np.einsum('eab,ebi->eai',shape(bary),trial_u.reshape(-1,2)[oldmesh.elements[parents]])
                predictor=np.zeros_like(new);predictor[newmesh.elements.ravel()]=sampled.reshape(-1,2)
                values=trial_lam[np.searchsorted(old_nodes,old_boundary[pick])]
                predictor_lam=(1-t[:,None])*values[:,0]+t[:,None]*values[:,1]
                self._spatial_predictor=(new.ravel().copy(),predictor.ravel(),predictor_lam,target)
        return new.ravel(),warm

    def contact_predictor(self,previous,depth):
        from .initiation import rounded_blade_distance
        seed=previous.copy();nodes=self.surface_contact.displacement_nodes
        x=(self.mesh.reference_xy+previous.reshape(-1,2))[nodes]
        gap,normal=rounded_blade_distance(x,self.config,self.options,depth)
        seed.reshape(-1,2)[nodes]+=np.maximum(gap,0)[:,None]*normal
        seed[self.fixed]=previous[self.fixed]
        return seed

    def feasible_contact_predictor(self,previous,old_depth,depth,u,lam):
        """Restart from the near-equilibrium trial with active nodes on the tool.

        This is only an initial guess for weak integral contact; the quadrature
        gaps, original force balance and friction must still be re-solved.
        """
        from .initiation import rounded_blade_distance
        nodes=self.contact_nodes
        x=(self.mesh.reference_xy+u.reshape(-1,2))[nodes]
        gap,normal=rounded_blade_distance(x,self.config,self.options,depth)
        correction=np.where(lam[:,0]>0,gap,np.maximum(gap,0))[:,None]*normal
        for cut in range(12):
            seed=u.copy();seed.reshape(-1,2)[nodes]+=(.5**cut)*correction
            seed[self.fixed]=previous[self.fixed]
            try:self.bulk_response(seed)
            except FloatingPointError:continue
            warm=lam.copy();warm[:,0]=np.maximum(warm[:,0],0.)
            bound=self.config.friction_coefficient*warm[:,0]
            warm[:,1]=np.clip(warm[:,1],-bound,bound)
            q=self.surface_contact.evaluate(seed,previous,depth,old_depth,warm,with_tangent=False,formulation=None,roundoff_guard=False)
            warm[:,1]=np.clip(q['shear'],-bound,bound)
            print(f'RETRY depth={depth:.8g}; active-surface predictor fraction={.5**cut:g}',flush=True)
            return seed,warm
        raise FloatingPointError('Active-surface predictor has no valid deformation')

    def edge_penetration(self,u,depth,*,with_edges=False):
        from .initiation import rounded_blade_distance
        from .model_config import local_to_lab,_rotation_vectors
        xy=self.mesh.reference_xy+u.reshape(-1,2)
        a,b=xy[self.edges].transpose(1,0,2);mid=xy[self.mesh.boundary_midpoints]
        A=2*(a+b-2*mid);B=4*mid-3*a-b
        _,en=_rotation_vectors(self.config.inclination_deg)
        center=local_to_lab(np.array([[self.config.drop_s_mm,self.config.height_mm]]),self.config)[0]+self.options.tip_radius_mm*en-np.array([0.,depth])
        t=np.broadcast_to(np.linspace(0,1,17),(len(a),17)).copy()
        points=A[:,None]*t[:,:,None]**2+B[:,None]*t[:,:,None]+a[:,None]
        gaps=rounded_blade_distance(points.reshape(-1,2),self.config,self.options,depth)[0].reshape(len(a),-1).max(axis=1)
        # Exact stationary points of distance to the circle on curved edges.
        near=np.flatnonzero(np.linalg.norm(a-center,axis=1)<np.linalg.norm(b-a,axis=1)*3+2*self.options.tip_radius_mm)
        for i in near:
            v=a[i]-center
            roots=np.roots([4*A[i]@A[i],6*A[i]@B[i],2*(B[i]@B[i]+2*A[i]@v),2*B[i]@v])
            valid=roots.real[(abs(roots.imag)<1e-9)&(roots.real>=0)&(roots.real<=1)]
            if len(valid):gaps[i]=max(gaps[i],rounded_blade_distance(A[i]*valid[:,None]**2+B[i]*valid[:,None]+a[i],self.config,self.options,depth)[0].max())
        # Certify the threshold on the entire polynomial edge. Signed distance
        # is 1-Lipschitz, so endpoint gaps plus half an arc-length upper bound
        # bound each unsampled interval. Only ambiguous near-contact intervals
        # are bisected; no dense sampling of remote specimen edges is needed.
        tol=self.config.contact_penetration_tolerance_mm
        active=np.flatnonzero(gaps<=tol)
        ids=np.repeat(active,16);left=np.tile(np.linspace(0,1,17)[:-1],len(active));right=left+1/16
        for refinement in range(22):
            if not len(ids):break
            ends=np.stack((left,right),axis=1)
            x=A[ids,None]*ends[:,:,None]**2+B[ids,None]*ends[:,:,None]+a[ids,None]
            values=rounded_blade_distance(x.reshape(-1,2),self.config,self.options,depth)[0].reshape(-1,2)
            np.maximum.at(gaps,ids,values.max(axis=1))
            speed=np.linalg.norm(2*A[ids,None]*ends[:,:,None]+B[ids,None],axis=2).max(axis=1)
            upper=values.max(axis=1)+speed*(right-left)/2
            ambiguous=(upper>tol)&(gaps[ids]<=tol)
            if refinement==21:
                np.maximum.at(gaps,ids[ambiguous],upper[ambiguous]);break
            ids0=ids[ambiguous];l=left[ambiguous];r=right[ambiguous];mid=(l+r)/2
            ids=np.repeat(ids0,2);left=np.column_stack((l,mid)).ravel();right=np.column_stack((mid,r)).ravel()
        maximum=float(gaps.max(initial=0))
        return (maximum,np.flatnonzero(gaps>self.config.contact_penetration_tolerance_mm)) if with_edges else maximum

    def fields(self,u):
        # Export every integration point, not only the element-centroid value.
        from .fracture_criteria import stress_measures
        q=self.bulk_response(u);F2=q['F'].reshape(-1,2,2)
        F=np.tile(np.eye(3),(len(F2),1,1));F[:,:2,:2]=F2
        eigen,directions=np.linalg.eigh(F@F.transpose(0,2,1));stretch=np.sqrt(eigen)
        principal=np.einsum('eai,eab,ebi->ei',directions,q['cauchy'],directions)
        J=q['J'].ravel();nominal=np.maximum(0,J[:,None]*principal/stretch)
        index=nominal.argmax(axis=1);maximum=nominal[np.arange(len(J)),index]
        first,second=stress_measures(F,q['cauchy'])
        reference=np.einsum('qa,eai->eqi',L,self.mesh.reference_xy[self.mesh.triangles])
        current=reference+np.einsum('qa,eai->eqi',shape(L),u.reshape(-1,2)[self.mesh.elements])
        def average(values):return np.repeat(self.average@(values.reshape(-1,7)@W),7)
        fields=dict(cauchy_principal_mpa=principal,principal_stretches=stretch,volumetric_pressure_mpa=q['pressure'].ravel(),
            volume_projection_error_mpa=(self.config.bulk_modulus_n_mm2*(q['J']-1)-q['pressure']).ravel(),
            deformation_gradient_xzy=F,cauchy_stress_xzy_mpa=q['cauchy'],first_piola_stress_xzy_mpa=first,second_piola_stress_xzy_mpa=second,
            cauchy_mean_tensile_stress_mpa=principal.mean(axis=1),nominal_tensile_mpa=maximum,averaged_nominal_tensile_mpa=average(maximum),
            tensile_direction_xzy=directions[np.arange(len(J)),:,index],reference_centroid_xz_mm=reference.reshape(-1,2),
            current_centroid_xz_mm=current.reshape(-1,2),reference_centroid_sn_mm=np.einsum('qa,eai->eqi',L,self.mesh.local_sn[self.mesh.triangles]).reshape(-1,2),J=J,
            field_element_ids=np.repeat(np.arange(len(self.mesh.triangles)),7),field_sampling='seven integration points per P2 triangle; legacy centroid keys denote sample coordinates')
        if self.strength_surface is not None:
            values=self.strength_surface.index(second);fields.update(multiaxial_strength_index=values,averaged_multiaxial_strength_index=average(values))
        return fields
