"""Oriented 2D GCP body contact with incremental regularized Coulomb friction.

Normal potential/derivatives use IPC Toolkit's GCP. Self friction is an explicit
project model, NOT a claim to reproduce a missing upstream friction binding.
It uses the compressive radial derivative of each GCP pair, current closest
material weights/tangent, and the last ACCEPTED positions. All derivatives of
those refreshed quantities enter its nonsymmetric tangent. No mutable pair
history, hard k-ring filtering, or separated-only rejection is used.

Fixed dhat is allowed only if the oriented reference has no GCP interactions
within 2*dhat: then the published reference-adaptive min(dc/2,dhat) is uniform.
General reference shapes which fail that certificate are rejected explicitly.
"""
import hashlib
import json
import numpy as np
from scipy.sparse import csc_matrix,coo_matrix
from .ipc_reference import BASELINE,LoadStepFailure

POLICY='oriented-gcp2-radial-coulomb-v1'


def orient_boundary(reference,triangles,edges):
    """Solid lies to the left, decided in reference space (including for P2)."""
    incidence={}
    for tri in triangles:
        for i in range(3):
            a,b,c=map(int,(tri[i],tri[(i+1)%3],tri[(i+2)%3]))
            key=tuple(sorted((a,b)))
            if key in incidence:incidence[key]=None
            else:incidence[key]=c
    result=np.asarray(edges,dtype=int).copy()
    for i,(a,b) in enumerate(result):
        c=incidence.get(tuple(sorted((int(a),int(b)))))
        if c is None:raise ValueError('GCP requires actual exterior boundary edges')
        t=reference[b]-reference[a];v=reference[c]-reference[a]
        cross=t[0]*v[1]-t[1]*v[0]
        if cross==0:raise ValueError('Degenerate reference boundary cell')
        if cross<0:result[i]=[b,a]
    return result


def slip_mollifier(s,eps):
    """Zero-at-zero C2 potential; its derivative is bounded by one."""
    a=abs(s)
    if a>=eps:return a-eps/3,np.sign(s),0.
    return s*s/eps-a**3/(3*eps*eps),s*(2/eps-a/(eps*eps)),2/eps-2*a/(eps*eps)


def pair_kinematics(x,edge_vertex):
    """Closest weights and exact current-geometry differentials, local DOFs."""
    nvar=x.size;w=np.zeros(len(x));dw=np.zeros((len(x),nvar))
    if edge_vertex:
        a,b,p=x[0],x[1],x[2];e=b-a;v=p-a;l2=float(e@e)
        if l2<=0:raise FloatingPointError('Collapsed GCP edge')
        beta=float(v@e/l2)
        # Native GCP may retain a zero-weight EV exactly at an endpoint tie;
        # its mollifier hands the interaction to VV continuously there.
        if not -1e-12<=beta<=1+1e-12:raise ValueError('GCP EV projection outside its represented edge')
        beta=float(np.clip(beta,0.,1.))
        db=np.zeros(nvar);db[4:6]=e/l2;db[2:4]=(v-2*beta*e)/l2;db[:2]=-db[2:4]-db[4:6]
        w[:3]=[-(1-beta),-beta,1.];dw[0]=db;dw[1]=-db;split=2
    else:
        w[0]=-1.;w[3]=1.;split=3
    delta=w@x;distance=float(np.linalg.norm(delta))
    if distance<=0:raise FloatingPointError('GCP surfaces touch/intersect')
    J=np.kron(w,np.eye(2))+x.T@dw
    normal=delta/distance;jn=(np.eye(2)-np.outer(normal,normal))@J/distance
    rot=np.array([[0.,-1.],[1.,0.]])
    return w,dw,normal,jn,rot@normal,rot@jn,split,distance


class GeometricSelfContact:
    def __init__(self,ipc,reference,edges,*,dhat,stiffness_mpa,width_mm,mu,eps,
                 alpha_t=.5,alpha_n=.1):
        self.ipc=ipc;self.reference=np.asarray(reference).copy();self.edges=np.asarray(edges,dtype=np.int32).copy()
        n=len(reference)
        if reference.shape!=(n,2) or not np.isfinite(reference).all():raise ValueError('Finite 2D boundary required')
        if not np.all(np.bincount(self.edges[:,0],minlength=n)==1) or not np.all(np.bincount(self.edges[:,1],minlength=n)==1):
            raise ValueError('Closed oriented manifold: one incoming/outgoing edge per vertex required')
        if not all(np.isfinite(v) and v>0 for v in (dhat,stiffness_mpa,width_mm,eps)) or not np.isfinite(mu) or mu<0:
            raise ValueError('Invalid GCP dimensions or friction coefficient')
        self.outgoing=np.empty(n,dtype=int);self.incoming=np.empty(n,dtype=int)
        self.outgoing[self.edges[:,0]]=self.edges[:,1];self.incoming[self.edges[:,1]]=self.edges[:,0]
        self.mesh=ipc.CollisionMesh(include_vertex=[True]*n,orient_vertex=[True]*n,
                                    full_rest_positions=reference,edges=self.edges)
        self.mesh.init_adjacencies()
        if ipc.has_intersections(self.mesh,reference):raise ValueError('Intersecting GCP reference')
        self.params=ipc.SmoothContactParameters(dhat,alpha_t,0.,alpha_n,0.,1)
        self.potential=ipc.SmoothPotential(self.params)
        expanded=ipc.SmoothContactParameters(2*dhat,alpha_t,0.,alpha_n,0.,1)
        cert=ipc.SmoothCollisions();cert.build(self.mesh,reference,expanded,False)
        if len(cert):raise ValueError('Nonuniform reference-adaptive GCP support required; unsupported reference')
        # Native 2D GCP uses two length measures and b(distance/dhat), hence
        # raw potential is mm^2. thickness(mm)*stiffness(N/mm^2) -> N/mm.
        self.scale=width_mm*stiffness_mpa;self.mu=float(mu);self.eps=float(eps)
        self.spec=dict(policy=POLICY,dhat_mm=dhat,stiffness_mpa=stiffness_mpa,width_mm=width_mm,
            self_mu=mu,slip_regularization_mm=eps,alpha_t=alpha_t,alpha_n=alpha_n,r=1,
            reference_support='certified no active GCP pair at 2*dhat; uniform adaptive branch',
            oriented_edges_sha256=hashlib.sha256(self.edges.tobytes()).hexdigest(),
            toolkit_version=ipc.__version__)
        self.last={}

    def stencil(self,c):
        a,b=int(c[0]),int(c[1]);nd=c.n_dofs()
        def point(v):return [v,int(self.outgoing[v]),int(self.incoming[v])]
        if nd==10:ids=list(map(int,self.edges[a]))+point(b);ev=True
        elif nd==12:ids=point(a)+point(b);ev=False
        else:raise RuntimeError(f'Unsupported GCP binding/stencil size {nd}; closed 2D only')
        return np.array(ids,dtype=int),ev

    def terms(self,x,previous,*,tangent,with_metric=False):
        if with_metric and not tangent:raise ValueError('GCP metric requires tangent')
        if x.shape!=self.reference.shape or previous.shape!=x.shape:raise ValueError('GCP field shape mismatch')
        c=self.ipc.SmoothCollisions();c.build(self.mesh,x,self.params,False)
        energy=self.scale*float(self.potential(c,self.mesh,x));g=self.scale*self.potential.gradient(c,self.mesh,x)
        H=self.scale*self.potential.hessian(c,self.mesh,x,self.ipc.PSDProjectionMethod.NONE) if tangent else None
        M=self.scale*self.potential.hessian(c,self.mesh,x,self.ipc.PSDProjectionMethod.CLAMP) if with_metric else None
        metric_rows=[];metric_cols=[];metric_values=[]
        work=0.;friction_energy=0.;rows=[];cols=[];values=[];max_cone=0.;normal_sum=0.;minimum=None
        for k in range(len(c)):
            pair=c[k];ids,ev=self.stencil(pair);local=x[ids];old=previous[ids]
            w,dw,n,jn,t,jt,split,dist=pair_kinematics(local,ev)
            minimum=dist if minimum is None else min(minimum,dist)
            # No string parsing: the native closed-2D stencil layout is checked
            # by native/local energy and scatter equivalence in the tests.
            if self.mu==0:continue
            normal_g=self.scale*self.potential.gradient(pair,local.ravel())
            resultant=normal_g.reshape(-1,2)[split:].sum(axis=0)
            N=max(0.,float(-n@resultant));normal_sum+=N
            if N==0:continue
            delta=local-old;relative=w@delta;s=float(t@relative)
            phi,dphi,ddphi=slip_mollifier(s,self.eps)
            B=(w[:,None]*t).ravel();force=self.mu*N*dphi*B
            dofs=(2*ids[:,None]+np.arange(2)).ravel();np.add.at(g,dofs,force)
            work+=self.mu*N*dphi*s;friction_energy+=self.mu*N*phi
            max_cone=max(max_cone,abs(self.mu*N*dphi)-self.mu*N)
            if tangent:
                hlocal=self.scale*self.potential.hessian(pair,local.ravel(),self.ipc.PSDProjectionMethod.NONE)
                dr=hlocal.reshape(len(local),2,-1)[split:].sum(axis=0)
                dN=-n@dr-resultant@jn
                ds=t@(np.kron(w,np.eye(2))+delta.T@dw)+relative@jt
                dB=(dw[:,None,:]*t[None,:,None]+w[:,None,None]*jt[None,:,:]).reshape(len(dofs),-1)
                jac=self.mu*(np.outer(B,dphi*dN+N*ddphi*ds)+N*dphi*dB)
                rows.extend(np.repeat(dofs,len(dofs)));cols.extend(np.tile(dofs,len(dofs)));values.extend(jac.ravel())
                if with_metric:
                    # Auxiliary PSD metric only: frozen pair force/geometry.
                    # The physical H above retains ALL refreshed derivatives.
                    metric_rows.extend(np.repeat(dofs,len(dofs)))
                    metric_cols.extend(np.tile(dofs,len(dofs)))
                    metric_values.extend((self.mu*N*ddphi*np.outer(B,B)).ravel())
        if tangent and rows:H+=coo_matrix((values,(rows,cols)),shape=(x.size,x.size)).tocsc()
        if with_metric and metric_rows:
            M+=coo_matrix((metric_values,(metric_rows,metric_cols)),shape=(x.size,x.size)).tocsc()
        self.last=dict(active_pairs=len(c),candidate_count=c.n_candidates(),normal_potential_n_mm=energy,
            friction_surrogate_n_mm=friction_energy,friction_work_n_mm=work,
            compressive_pair_force_sum_n=normal_sum,cone_violation_n=max_cone,
            minimum_active_gap_mm=minimum)
        if (not np.isfinite(g).all() or not np.isfinite(energy+work)
                or (tangent and not np.isfinite(H.data).all()) or work < -1e-10 or max_cone>1e-10):
            raise LoadStepFailure('GCP invalid force/dissipation/cone')
        result=(energy+friction_energy,g,H,work)
        return result+(M,) if with_metric else result


class GeometricSelfContactMixin:
    def _self_terms(self,x,tangent):
        e,g,h,w=self.geometric_self.terms(x[self.body_surface],self.previous[self.body_surface],tangent=tangent)
        if tangent:
            h=h.tocoo();h=coo_matrix((h.data,(self.self_dofs[h.row],self.self_dofs[h.col])),shape=(self.total,self.total)).tocsc()
        return e,g,h,w

    def acceptance(self,x,depth,iterations,outer_iterations):
        result=super().acceptance(x,depth,iterations,outer_iterations)
        result['geometric_self_contact']=dict(self.geometric_self.last)
        return result

    def checkpoint_data(self):
        return dict(super().checkpoint_data(),body_self_contact_policy=np.array(POLICY),
            geometric_self_spec=np.array(json.dumps(self.geometric_self.spec,sort_keys=True)))

    def audit_state(self,state,record):
        if str(state.get('body_self_contact_policy',''))!=POLICY:raise ValueError('GCP checkpoint policy mismatch')
        if str(state.get('geometric_self_spec',''))!=json.dumps(self.geometric_self.spec,sort_keys=True):raise ValueError('GCP checkpoint parameters mismatch')
        return super().audit_state(state,record)

    def _audit_extra_acceptance(self,fresh,record):
        result=super()._audit_extra_acceptance(fresh,record)
        for key,value in fresh['geometric_self_contact'].items():
            saved=record['geometric_self_contact'][key]
            if value is None or saved is None:
                if value!=saved:raise ValueError('GCP audit missing-value mismatch: '+key)
            else:np.testing.assert_allclose(value,saved,rtol=1e-9,atol=1e-10)
        return dict(result,geometric_self_contact=fresh['geometric_self_contact'])


def enable_geometric_self_contact(p,*,stiffness_mpa=BASELINE['barrier_stiffness_mpa']):
    if isinstance(p,GeometricSelfContactMixin):raise ValueError('GCP already enabled')
    if 'separated-only' in p.contact_formulation:raise ValueError('Do not layer GCP over the separated-domain diagnostic')
    edges=orient_boundary(p.rest,p.mesh.triangles,p.body_edges)
    local=np.searchsorted(p.body_surface,edges)
    gcp=GeometricSelfContact(p.ipc,p.rest[p.body_surface],local,dhat=p.dhat,
        stiffness_mpa=stiffness_mpa,width_mm=p.config.width_mm,mu=p.config.friction_coefficient,
        eps=BASELINE['friction_smoothing_mm'])
    p.geometric_self=gcp
    p.__class__=type('Geometric'+type(p).__name__,(GeometricSelfContactMixin,type(p)),{})
    # CCD and intersection audit use the SAME geometry with consistent orientation.
    p.self_mesh=gcp.mesh;p.contact_formulation+=';'+POLICY
    return p
