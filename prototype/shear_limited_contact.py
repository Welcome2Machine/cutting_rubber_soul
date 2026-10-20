"""Opt-in nodal-pressure Coulomb shear limit for the native 2D interface.

Normal IPC and body self-contact are unchanged. Current rubber dual areas
convert the assembled blade-contact nodal force to an averaged pressure.
The smooth cap is a model choice, not a measured coating property or the
pointwise Coulomb--Orowan law. Its full pressure/area/interpolation derivative
is added to the native nonsymmetric friction Jacobian. See docs/history/topics/interface_candidate.md.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags


def limited_mu(pressure, mu, strength):
    """Return effective mu and -d(log mu_eff)/d(log p), continuous at p=0."""
    p=np.asarray(pressure,dtype=float)
    if (not np.isfinite(p).all() or np.any(p<0) or not np.isfinite(mu) or mu<=0
            or not np.isfinite(strength) or strength<=0):
        raise ValueError('Nonnegative finite pressure and positive finite interface constants required')
    ratio=mu*p/strength
    scale=np.sqrt(np.hypot(1.,ratio*ratio))
    value=mu/scale
    sensitivity=1.-1./scale**4
    return value,sensitivity


class ShearLimitedContact:
    """Mixin installed explicitly; leaves the default IPC class unchanged."""
    def enable_shear_limit(self,strength):
        if not np.isfinite(strength) or strength<=0: raise ValueError('Positive shear limit in MPa required')
        if self.contact_formulation!='improved-max':
            raise ValueError('Shear limit is qualified only for native improved-max contact')
        self.shear_limit_mpa=float(strength)
        self.interface_body_count=int(np.count_nonzero(self.surface<self.n))
        self.interface_mesh=self.ipc.CollisionMesh(self.rest[self.surface],self.cmesh.edges)
        labels=(self.surface>=self.n).astype(np.int32)
        self.interface_mesh.can_collide=self.ipc.make_vertex_patches_filter(labels)
        self.interface_edges=np.asarray(self.cmesh.edges)[np.all(np.asarray(self.cmesh.edges)<self.interface_body_count,axis=1)]

    def interface_field(self,x,*,derivative=False):
        ns=len(self.surface); nb=self.interface_body_count; sx=x[self.surface]
        c=self.ipc.NormalCollisions();c.use_area_weighting=True
        c.collision_set_type=self.ipc.NormalCollisions.IMPROVED_MAX_APPROX
        c.build(self.interface_mesh,sx,self.dhat)
        g=self.barrier.gradient(c,self.interface_mesh,sx).reshape(ns,2)
        normal=np.linalg.norm(g[:nb],axis=1)
        edge=self.interface_edges; vector=sx[edge[:,1]]-sx[edge[:,0]]
        length=np.linalg.norm(vector,axis=1)
        if np.any(length<=0):raise FloatingPointError('Collapsed current interface edge')
        area=np.zeros(nb); half=.5*self.config.width_mm
        np.add.at(area,edge[:,0],half*length);np.add.at(area,edge[:,1],half*length)
        if np.any(area<=0):raise FloatingPointError('Nonpositive rubber boundary dual area')
        pressure=normal/area
        mu,eta=limited_mu(pressure,self.config.friction_coefficient,self.shear_limit_mpa)
        field=dict(mu=mu,pressure_mpa=pressure,normal_n=normal,current_area_mm2=area,
                   normal_gradient=g,collisions=c)
        if derivative:
            hn=self.barrier.hessian(c,self.interface_mesh,sx,self.ipc.PSDProjectionMethod.NONE).tocsr()
            unit=np.divide(g[:nb],normal[:,None],out=np.zeros((nb,2)),where=normal[:,None]>0)
            projection=coo_matrix((unit.ravel(),(np.repeat(np.arange(nb),2),np.arange(2*nb))),shape=(nb,2*ns)).tocsr()
            rows=[];cols=[];data=[]
            for endpoint in (0,1):
                for direction,sign in ((0,-1.),(1,1.)):
                    rows.extend(np.repeat(edge[:,endpoint],2))
                    cols.extend((2*edge[:,direction,None]+np.arange(2)).ravel())
                    data.extend((sign*half*vector/length[:,None]).ravel())
            da=coo_matrix((data,(rows,cols)),shape=(nb,2*ns)).tocsr()
            a=np.divide(-mu*eta,normal,out=np.zeros_like(mu),where=normal>0)
            field['mu_jacobian']=diags(a)@projection@hn+diags(mu*eta/area)@da
        return field

    def interface_coefficient(self,c,x,field,*,derivative=False):
        ids=np.asarray(c.vertex_ids(self.cmesh.edges,self.cmesh.faces),dtype=int)
        ids=ids[ids>=0]; body=ids<self.interface_body_count
        if body.all() or not body.any():return None
        if np.count_nonzero(body)==1:
            node=int(ids[body][0]); value=field['mu'][node]
            row=field['mu_jacobian'].getrow(node) if derivative else None
        elif len(ids)==3 and not body[0] and body[1:].all():
            # Native point-edge order is [point, edge0, edge1].
            beta=float(c.closest_point[0]); a,b=map(int,ids[1:])
            value=(1-beta)*field['mu'][a]+beta*field['mu'][b]
            row=None
            if derivative:
                row=(1-beta)*field['mu_jacobian'].getrow(a)+beta*field['mu_jacobian'].getrow(b)
                db=np.asarray(c.compute_closest_point_jacobian(x[self.surface[ids]].ravel())).ravel()
                dofs=(2*ids[:,None]+np.arange(2)).ravel()
                row+=coo_matrix(((field['mu'][b]-field['mu'][a])*db,(np.zeros(len(dofs)),dofs)),
                                shape=(1,2*len(self.surface))).tocsr()
        else:
            raise NotImplementedError('Shear limit supports native 2D blade VV/EV stencils only')
        return float(value),row,ids

    def lag(self,x):
        lag=super().lag(x)
        if lag is None:return lag
        field=self.interface_field(x)
        for i in range(len(lag)):
            c=lag[i]; entry=self.interface_coefficient(c,x,field)
            if entry is None:continue
            mu=entry[0]
            c.mu_s=c.mu_k=c.mu_s_effective_lagged=c.mu_k_effective_lagged=mu
        return lag

    def coupled_friction_tangent(self,x,lag,frozen_hessian):
        value=super().coupled_friction_tangent(x,lag,frozen_hessian)
        if lag is None:return value
        field=self.interface_field(x,derivative=True)
        rows=[];cols=[];data=[]
        for i in range(len(lag)):
            c=lag[i];entry=self.interface_coefficient(c,x,field,derivative=True)
            if entry is None:continue
            mu,dm,ids=entry
            if not np.isclose(c.mu_s_effective_lagged,mu,rtol=1e-12,atol=0):
                raise RuntimeError('Fresh interface coefficient differs from lagged residual')
            grad=self.friction.gradient(c,(x-self.previous)[self.surface[ids]].ravel())/mu
            dm.eliminate_zeros()
            rd=(2*self.surface[ids,None]+np.arange(2)).ravel();cd=self.surface_dofs[dm.indices]
            rows.extend(np.repeat(rd,len(cd)));cols.extend(np.tile(cd,len(rd)))
            data.extend((grad[:,None]*dm.data[None,:]).ravel())
        return value+coo_matrix((data,(rows,cols)),shape=value.shape).tocsc()

    def acceptance(self,x,depth,iterations,outer_iterations):
        result=super().acceptance(x,depth,iterations,outer_iterations)
        field=self.interface_field(x);active=field['normal_n']>1e-8
        lag=self.lag(x);delta=(x-self.previous)[self.surface]
        work=float(self.friction.gradient(lag,self.cmesh,delta)@delta.ravel())
        if not np.isfinite(work) or work < -1e-10:
            raise RuntimeError('Shear-limited interface has negative/nonfinite discrete friction work')
        result['interface']=dict(law='nodal-current-area-smooth-shear-limit',strength_mpa=self.shear_limit_mpa,
            smooth_min_power=4,active_material_nodes=int(active.sum()),
            pressure_max_mpa=float(field['pressure_mpa'][active].max()) if active.any() else None,
            effective_mu_min=float(field['mu'][active].min()) if active.any() else None,
            friction_increment_work_n_mm=work,
            scope='Nodal averaged pressure, not a pointwise stress or experimentally identified interface')
        return result


def export_interface_history(p,out,report):
    """Accepted material-point motion and averaged blade traction, never cracks.

    Fixed reference stations cover the whole top surface; every additional
    active blade-contact material node is included. Contact tangential motion
    is a per-step nodal estimate, not accumulated slip or a unique tip identity.
    """
    import csv
    from .model_config import _rotation_vectors
    es,en=_rotation_vectors(p.config.inclination_deg)
    nb=p.interface_body_count; ids=p.surface[:nb]
    sn=p.mesh.local_sn[ids]; top=np.isclose(sn[:,1],p.config.height_mm,atol=1e-9,rtol=0)
    top_ids=np.flatnonzero(top)
    targets=np.unique(np.r_[np.arange(0.,p.config.length_mm+.001,.5),
                           p.config.drop_s_mm+np.arange(-2.,2.001,.05)])
    stations=np.unique([top_ids[np.argmin(abs(sn[top_ids,0]-s))] for s in targets
                        if sn[top_ids,0].min()<=s<=sn[top_ids,0].max()])
    fields=['depth_mm','node_id','reference_offset_mm','current_x_mm','current_y_mm',
        'displacement_s_mm','displacement_n_mm','relative_increment_s_mm','relative_increment_n_mm',
        'blade_contact_supported','nodal_pressure_mpa','effective_mu','current_dual_area_mm2',
        'relative_tangent_increment_mm','normal_gradient_x_n','normal_gradient_y_n']
    with (out/'interface_motion.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,record in enumerate(report['steps']):
            with np.load(out/'states'/f'{i:04d}.npz',allow_pickle=False) as state:
                x=state['positions'];previous=state['previous_positions']
                f=p.interface_field(x);active=f['normal_n']>1e-8
                selected=np.union1d(stations,np.flatnonzero(active))
                blade_increment=np.mean((x-previous)[p.n:],axis=0)
                for j in selected:
                    node=int(ids[j]);u=x[node]-p.rest[node];rel=x[node]-previous[node]-blade_increment
                    gn=f['normal_gradient'][j];normal=f['normal_n'][j]
                    tangent=np.array([-gn[1],gn[0]])/normal if active[j] else None
                    writer.writerow(dict(depth_mm=record['depth_mm'],node_id=node,
                        reference_offset_mm=sn[j,0]-p.config.drop_s_mm,current_x_mm=x[node,0],current_y_mm=x[node,1],
                        displacement_s_mm=u@es,displacement_n_mm=u@en,
                        relative_increment_s_mm=rel@es,relative_increment_n_mm=rel@en,
                        blade_contact_supported=int(active[j]),nodal_pressure_mpa=f['pressure_mpa'][j],
                        effective_mu=f['mu'][j],current_dual_area_mm2=f['current_area_mm2'][j],
                        relative_tangent_increment_mm=float(rel@tangent) if tangent is not None else None,
                        normal_gradient_x_n=gn[0],normal_gradient_y_n=gn[1]))
