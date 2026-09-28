"""Finite-strain, zero-gap DEBONDED interface component benchmark.

Two independent rubber meshes and a rounded rigid wedge. Augmented-Lagrangian
unilateral contact permits a closed crack; no artificial initial gap, deletion,
or damage colouring substitutes for the independent displacement fields.
The seam is prescribed ONLY in this component test. No nucleation prediction.
"""
from dataclasses import dataclass
from collections import Counter
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits
from .material import response
from .model_config import FEMConfig
from .ipc_reference import determinant_step, BASELINE, BASELINE_PATH
from .crack_contact_geometry import signed_gap_derivatives, nearest_segment
from run_support import write_json, save_npz, source_hashes


DEFAULT_PENALTY = 20000.
FORCE_CONVENTION = (
    'Global x points right and y points up. clamp_external_force_n and '
    'knife_external_force_n are forces applied by the displacement actuators '
    '(energy gradients), work-conjugate to their prescribed displacements. '
    'The rubber force on the knife has the opposite sign.'
)


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _finite_positive(value, name, *, allow_zero=False):
    if (not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
        bound = 'nonnegative' if allow_zero else 'positive'
        raise ValueError(f'{name} must be finite and {bound}')


@dataclass(frozen=True)
class Loading:
    compression: float = 0.
    sliding: float = 0.
    tip_y: float = 1.15

    def __post_init__(self):
        if not np.all(np.isfinite([self.compression, self.sliding, self.tip_y])):
            raise ValueError('Loading values must be finite')


def wedge_gap(point, tip_y, radius=.02, half_angle_deg=11.5, side=1):
    """Exact signed distance to the infinite rounded wedge, outward gradient.

    The cap and straight faces are C1 joined. side resolves the interior medial
    axis during infeasible AL trials; accepted points lie outside the wedge.
    """
    q=np.asarray(point,float)-[0.,tip_y+radius]
    angle=np.deg2rad(half_angle_deg); s,c=np.sin(angle),np.cos(angle)
    if q[1]*c+abs(q[0])*s <= 0.:
        length=float(np.linalg.norm(q))
        if length < 1e-14: return -radius,np.array([float(side),0.])
        return length-radius,q/length
    sign=np.sign(q[0]) if abs(q[0])>1e-14 else side
    return abs(q[0])*c-q[1]*s-radius,np.array([sign*c,-s])


def wedge_edge_gap(a,b,tip_y,radius,side):
    """Minimum gap on a complete edge, not merely its end nodes.

    Candidate minima: endpoints, closest point to cap centre, medial axis,
    and both cap/face junctions. Envelope derivative gives nodal force weights.
    """
    a=np.asarray(a);b=np.asarray(b);v=b-a; centre=np.array([0.,tip_y+radius])
    vv=float(v@v)
    if vv<=1e-24: raise ValueError('Degenerate boundary edge')
    candidates=[0.,1.,float(np.clip((centre-a)@v/vv,0.,1.))]
    if abs(v[0])>1e-14:candidates.append(float(np.clip(-a[0]/v[0],0.,1.)))
    angle=np.deg2rad(11.5)
    for sign in (-1.,1.):
        normal=np.array([sign*np.sin(angle),np.cos(angle)])
        den=float(normal@v)
        if abs(den)>1e-14:
            candidates.append(float(np.clip(normal@(centre-a)/den,0.,1.)))
    values=[wedge_gap(a+t*v,tip_y,radius,side=side) for t in candidates]
    index=int(np.argmin([q[0] for q in values]));g,n=values[index]
    t=candidates[index];point=a+t*v
    if 1e-12<t<1.-1e-12 and abs(point[0])<1e-12 and point[1]>=centre[1] and abs(v[0])>1e-14:
        # The edge's minimum lies on the interior medial axis: its parameter
        # moves with the endpoints. This is the envelope subgradient satisfying
        # n.edge=0, not either arbitrary side's unit normal. Such a state is
        # infeasible and never passes the final knife penetration test.
        n=np.array([np.sin(angle)*v[1]/v[0],-np.sin(angle)])
    return float(g),n,t


def split_mesh(nx=5,ny=16):
    """Two unit blocks; seam node coordinates coincide but IDs do not."""
    _integer(nx, 'nx');_integer(ny, 'ny')
    points=[];triangles=[];seams=[];boundaries=[];clamps=[];sides=[]
    for side in (-1,1):
        offset=len(points)
        # Concentrate elements near the independent interface, retain outer bulk.
        xs=np.linspace(0.,1.,nx+1)**1.6
        xs=-xs[::-1] if side<0 else xs
        ys=np.linspace(0.,1.,ny+1)
        points.extend([[a,b] for b in ys for a in xs])
        node=lambda i,j:offset+j*(nx+1)+i
        for j in range(ny):
            for i in range(nx):
                a,b,c,d=node(i,j),node(i+1,j),node(i,j+1),node(i+1,j+1)
                triangles.extend([[a,b,d],[a,d,c]])
        seam=np.array([node(nx if side<0 else 0,j) for j in range(ny+1)])
        seams.append(seam)
        clamps.append(np.array([node(0 if side<0 else nx,j) for j in range(ny+1)]))
        boundary=np.array([node(i,0) for i in range(nx+1)]+[node(nx,j) for j in range(1,ny+1)]
                          +[node(i,ny) for i in range(nx-1,-1,-1)]+[node(0,j) for j in range(ny-1,0,-1)])
        boundaries.append(np.column_stack((boundary,np.roll(boundary,-1))))
        sides.extend([side]*((nx+1)*(ny+1)))
    return np.array(points),np.array(triangles),seams,boundaries,clamps,np.array(sides)


class ClosedCrackProblem:
    def __init__(self,nx=5,ny=16,*,penalty=DEFAULT_PENALTY,knife_mu=.35):
        _finite_positive(penalty, 'penalty');_finite_positive(knife_mu, 'knife_mu', allow_zero=True)
        self.config=FEMConfig();self.radius=.02;self.penalty=penalty;self.knife_mu=knife_mu
        self.reference,self.tri,self.seams,self.boundaries,self.clamps,self.sides=split_mesh(nx,ny)
        self.n=len(self.reference);self.ndof=2*self.n
        T=self.reference[self.tri]
        matrix=np.stack((T[:,1]-T[:,0],T[:,2]-T[:,0]),axis=-1)
        self.volumes=np.linalg.det(matrix)/2*self.config.width_mm
        self.grad=np.array([[-1.,-1.],[1.,0.],[0.,1.]])[None]@np.linalg.inv(matrix)
        self.dofs=(2*self.tri[:,:,None]+np.arange(2)).reshape(-1,6)
        self.fixed=(2*np.concatenate(self.clamps)[:,None]+np.arange(2)).ravel()
        self.free=np.setdiff1d(np.arange(self.ndof),self.fixed)
        self.knife_edges=np.vstack(self.boundaries)
        self.contact_count=sum(map(len,self.seams))+len(self.knife_edges)
        self.multipliers=np.zeros(self.contact_count)
        self.previous=self.reference.copy();self.previous_load=Loading()
        self.log_history=[]

    def deformation(self,x):
        return np.eye(2)+np.einsum('eai,eaJ->eiJ',(x-self.reference)[self.tri],self.grad)

    def bulk(self,x,tangent=True):
        q=response(self.deformation(x),self.config,tangent=tangent)
        re=np.einsum('e,eiJ,eaJ->eai',self.volumes,q['P'],self.grad).reshape(-1,6)
        r=np.bincount(self.dofs.ravel(),weights=re.ravel(),minlength=self.ndof)
        H=None
        if tangent:
            C=q['tangent'].reshape(-1,2,2,2,2)
            ke=np.einsum('e,eaJ,eiJkL,ebL->eaibk',self.volumes,self.grad,C,self.grad,optimize=True).reshape(-1,6,6)
            # Search metric only. Physical force and energy use the original law.
            w,V=np.linalg.eigh((ke+ke.transpose(0,2,1))/2)
            ke=(V*np.maximum(w,0.)[:,None,:])@V.transpose(0,2,1)
            rows=np.broadcast_to(self.dofs[:,:,None],ke.shape).ravel()
            cols=np.broadcast_to(self.dofs[:,None,:],ke.shape).ravel()
            H=coo_matrix((ke.ravel(),(rows,cols)),shape=(self.ndof,self.ndof)).tocsc()
        return float(self.volumes@q['energy']),r,H,q

    def contacts(self,x,load):
        gaps=[];rows=[];cols=[];values=[];groups=[];friction=[]
        def add(g,nodes,gradient,group,slip=None):
            row=len(gaps);gaps.append(g);groups.append(group)
            dofs=(2*np.asarray(nodes)[:,None]+np.arange(2)).ravel()
            rows.extend([row]*len(dofs));cols.extend(dofs);values.extend(np.asarray(gradient).ravel())
            if slip is not None:friction.append((row,*slip))
        # Symmetric two-pass collocation; NOT claimed to be mortar/unbiased.
        for body,seam in enumerate(self.seams):
            target=self.seams[1-body]
            edges=np.column_stack((target[:-1],target[1:]))
            if body==1:edges=edges[:,::-1]  # outward normal of the left block
            for node in seam:
                index,t=nearest_segment(x[node],x,edges);a,b=edges[index]
                v=x[b]-x[a];raw=float((x[node]-x[a])@v/(v@v))
                if raw < -1e-9 or raw > 1.+1e-9:
                    add(1.,[node,a,b],np.zeros(6),'seam');continue
                g,G,_=signed_gap_derivatives(x[node],x[a],x[b])
                add(g,[node,a,b],G,'seam')
        for a,b in self.knife_edges:
            side=int(self.sides[a]);g,n,t=wedge_edge_gap(x[a],x[b],load.tip_y,self.radius,side)
            weights=np.array([1.-t,t]);tangent=np.array([-n[1],n[0]])/np.linalg.norm(n)
            B=(weights[:,None]*tangent).ravel()
            offset=(load.tip_y-self.previous_load.tip_y)*tangent[1]
            add(g,[a,b],weights[:,None]*n,'knife_left' if side<0 else 'knife_right',
                (np.array([a,b]),B,float(offset)))
        G=coo_matrix((values,(rows,cols)),shape=(len(gaps),self.ndof)).tocsr()
        return np.array(gaps),G,np.array(groups),friction

    def friction_terms(self,x,lag,tangent=True):
        residual=np.zeros(self.ndof);energy=0.;dissipation=0.;knife_gradient=np.zeros(2)
        H=coo_matrix((self.ndof,self.ndof)).tocsc()
        epsilon=BASELINE['friction_smoothing_mm']
        for force,nodes,B,offset in lag:
            dofs=(2*nodes[:,None]+np.arange(2)).ravel()
            slip=float(B@(x-self.previous)[nodes].ravel()-offset)
            root=np.sqrt(slip*slip+epsilon*epsilon)
            value=force*slip/root
            residual[dofs]+=value*B
            energy+=force*(root-epsilon);dissipation+=value*slip
            knife_gradient-=value*B.reshape(-1,2).sum(axis=0)
            if tangent:
                local=force*epsilon**2/root**3*np.outer(B,B)
                H+=coo_matrix((local.ravel(),(np.repeat(dofs,len(dofs)),np.tile(dofs,len(dofs)))),shape=H.shape).tocsc()
        return energy,residual,H,float(dissipation),knife_gradient

    def friction_lag(self,x,load,lam,friction):
        return [(self.knife_mu*lam[row],nodes,B,offset) for row,nodes,B,offset in friction
                if lam[row]>0 and self.knife_mu>0]

    def objective(self,x,load,lam,lag,tangent=True):
        E,r,H,q=self.bulk(x,tangent)
        gaps,G,groups,friction=self.contacts(x,load)
        active=np.maximum(0.,lam-self.penalty*gaps)
        E+=float((active@active-lam@lam)/(2*self.penalty))
        r-=G.T@active
        Ef,rf,Hf,_,_=self.friction_terms(x,lag,tangent)
        E+=Ef;r+=rf
        if tangent:
            H+=G.T@diags(self.penalty*(active>0))@G+Hf
        return E,r,H,q

    def prescribe(self,x,load):
        for side,nodes in zip((-1,1),self.clamps):
            x[nodes]=self.reference[nodes]+[-side*load.compression,side*load.sliding/2]
        return x

    def geometry_audit(self,x,load):
        gaps,G,groups,friction=self.contacts(x,load)
        A=x[self.boundaries[0]];B=x[self.boundaries[1]]
        cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
        av=A[:,1]-A[:,0];bv=B[:,1]-B[:,0]
        o1=cross(av[:,None],B[None,:,0]-A[:,None,0]);o2=cross(av[:,None],B[None,:,1]-A[:,None,0])
        o3=cross(bv[None,:],A[:,None,0]-B[None,:,0]);o4=cross(bv[None,:],A[:,None,1]-B[None,:,0])
        # Use the same explicit spatial tolerance as the AL gap test, rather
        # than rejecting arbitrarily small roundoff/finite-tolerance overlap.
        d1=o1/np.linalg.norm(av,axis=1)[:,None];d2=o2/np.linalg.norm(av,axis=1)[:,None]
        d3=o3/np.linalg.norm(bv,axis=1)[None,:];d4=o4/np.linalg.norm(bv,axis=1)[None,:]
        opposite=lambda a,b:((a>1e-7)&(b < -1e-7))|((a < -1e-7)&(b>1e-7))
        proper=int(np.count_nonzero(opposite(d1,d2)&opposite(d3,d4)))
        return gaps,G,groups,friction,proper

    def solve(self,old_x,load,deadline,*,max_outer=80,max_inner=80):
        if np.shape(old_x)!=self.reference.shape or not np.all(np.isfinite(old_x)):
            raise ValueError('Previous positions must have the mesh shape and be finite')
        _finite_positive(deadline, 'deadline');_integer(max_outer, 'max_outer');_integer(max_inner, 'max_inner')
        self.previous=old_x.copy();x=self.prescribe(old_x.copy(),load)
        lam=self.multipliers.copy();self.log_history=[];start=time.perf_counter()
        self._trial_state(x,lam,load,-1,-1,'initial')
        for outer in range(max_outer):
            self._trial_state(x,lam,load,outer,-1,'outer_start')
            if time.monotonic()>=deadline:raise TimeoutError('Closed-crack wall budget')
            gaps,G,groups,friction=self.contacts(x,load)
            lag=self.friction_lag(x,load,lam,friction)
            for inner in range(max_inner):
                self._trial_state(x,lam,load,outer,inner,'inner')
                if time.monotonic()>=deadline:raise TimeoutError('Closed-crack wall budget')
                E,r,H,q=self.objective(x,load,lam,lag)
                norm=float(np.linalg.norm(r[self.free]))
                if norm<=2e-5:break
                du=spsolve(H[self.free][:,self.free]+diags(np.full(len(self.free),1e-7)),-r[self.free])
                direction=np.zeros(self.ndof);direction[self.free]=du;direction=direction.reshape(-1,2)
                slope=float(r@direction.ravel())
                if not np.isfinite(slope) or slope>=0:raise RuntimeError('Non-descent AL metric')
                D=np.einsum('eai,eaJ->eiJ',direction[self.tri],self.grad)
                alpha=min(1.,determinant_step(self.deformation(x),D),.04/max(np.max(abs(du)),1e-30))
                for cut in range(35):
                    candidate=x+alpha*direction
                    self._trial_state(candidate,lam,load,outer,inner,'line_search',cut)
                    Et=self.objective(candidate,load,lam,lag,False)[0]
                    if Et<=E+1e-4*alpha*slope+1e-13:
                        x=candidate;break
                    alpha*=.5
                else:raise RuntimeError('AL energy search failed')
            else:raise RuntimeError('AL inner iteration budget')
            self._trial_state(x,lam,load,outer,inner,'geometry_audit')
            gaps,G,groups,friction,proper=self.geometry_audit(x,load)
            fresh_lam=np.maximum(0.,lam-self.penalty*gaps)
            self._trial_state(x,fresh_lam,load,outer,inner,'acceptance_audit')
            fresh_lag=self.friction_lag(x,load,fresh_lam,friction)
            energy,bulk_force,_,q=self.bulk(x,False)
            _,fr,_,diss,knife_fr=self.friction_terms(x,fresh_lag,False)
            residual=bulk_force-G.T@fresh_lam+fr
            norm=float(np.linalg.norm(residual[self.free]));penetration=float(max(0.,-gaps.min()))
            complement=float(np.max(abs(fresh_lam*gaps)))
            record=dict(outer=outer,inner=inner,force_residual_n=norm,penetration_mm=penetration,
                        complementarity_n_mm=complement,min_J=float(q['J'].min()),proper_crossings=proper)
            self.log_history.append(record);print('AL '+json.dumps(record),flush=True)
            lam=fresh_lam
            if norm<=1e-4 and penetration<=1e-7 and complement<=1e-6 and proper==0:
                contact_reaction={}
                knife_normal=np.zeros(2)
                for label in ('seam','knife_left','knife_right'):
                    mask=groups==label
                    force=np.asarray(-G[mask].T@lam[mask]).reshape(-1,2).sum(axis=0)
                    contact_reaction[label]=dict(active=int(np.count_nonzero(lam[mask]>1e-8)),
                        multiplier_sum_n=float(lam[mask].sum()),body_residual_sum_n=force.tolist())
                    if label!='seam':knife_normal-=force
                singular=np.linalg.svd(self.deformation(x),compute_uv=False)
                seam_jump=x[self.seams[1]]-x[self.seams[0]]
                delta=(x-self.previous).ravel();du_knife=np.array([0.,load.tip_y-self.previous_load.tip_y])
                knife_gradient=knife_normal+knife_fr
                external=float(residual[self.fixed]@delta[self.fixed]+knife_gradient@du_knife)
                normal_work=float((-G.T@lam)@delta+knife_normal@du_knife)
                endpoint_defect=float(external+residual[self.free]@delta[self.free]-bulk_force@delta-normal_work-diss)
                result=dict(**record,elastic_energy_n_mm=energy,friction_work_n_mm=diss,
                    external_endpoint_work_n_mm=external,normal_contact_work_n_mm=normal_work,
                    endpoint_virtual_work_defect_n_mm=endpoint_defect,
                    clamp_external_force_n=residual[self.fixed].tolist(),knife_external_force_n=knife_gradient.tolist(),
                    contacts=contact_reaction,max_J=float(q['J'].max()),
                    max_F_condition=float(np.max(singular[:,0]/singular[:,1])),
                    paired_normal_jump_range_mm=[float(seam_jump[:,0].min()),float(seam_jump[:,0].max())],
                    paired_tangential_jump_range_mm=[float(seam_jump[:,1].min()),float(seam_jump[:,1].max())],
                    max_contact_regularization_energy_n_mm=float(lam@lam/(2*self.penalty)),
                    elapsed_s=time.perf_counter()-start)
                if abs(endpoint_defect)>1e-8 or diss < -1e-12:raise RuntimeError('Endpoint work/negative dissipation audit')
                self.multipliers=lam.copy()
                self.previous_load=load
                return x,result
        raise RuntimeError('AL outer iteration budget')

    def _trial_state(self,x,lam,load,outer,inner,phase,line_search=-1):
        """Keep trial evidence separate from committed physical history."""
        self.trial=x.copy();self.trial_multipliers=lam.copy();self.trial_load=load
        self.trial_iteration=dict(outer=outer,inner=inner,phase=phase,line_search=line_search)


def schedule():
    rows=[]
    for c in (.005,.01,.02,.04):rows.append(('compress',Loading(c)))
    for s in (.02,.05,.08):rows.append(('slide_closed',Loading(.04,s)))
    for s in (.04,0.):rows.append(('unslide',Loading(.04,s)))
    for c in (.02,0.,-.02,-.04):rows.append(('open',Loading(c)))
    for c in (-.02,0.,.01):rows.append(('reclose',Loading(c)))
    for tip in np.arange(1.12,.789,-.01):rows.append(('knife',Loading(.01,0.,float(tip))))
    return rows


def validate_run_args(args):
    for name in ('nx','ny','threads'):_integer(getattr(args,name),name)
    _integer(args.max_steps,'max_steps',minimum=0)
    for name in ('penalty','seconds'):_finite_positive(getattr(args,name),name)


def trapezoidal_work(previous_force,current_force,increment):
    """External work using forces at the two accepted physical endpoints."""
    return float(.5*(np.asarray(previous_force)+np.asarray(current_force))@np.asarray(increment))


def _schedule_progress(report,full_rows,requested_rows):
    full=Counter(stage for stage,_ in full_rows)
    requested=Counter(stage for stage,_ in requested_rows)
    accepted=Counter(row['stage'] for row in report['steps'])
    report.update(full_schedule_step_count=len(full_rows),requested_step_count=len(requested_rows),
        accepted_step_count=len(report['steps']),full_schedule_completed=len(report['steps'])==len(full_rows),
        completed_stages=[stage for stage,count in full.items() if accepted[stage]==count],
        requested_stages_completed=[stage for stage,count in requested.items() if accepted[stage]==count],
        stage_counts={stage:dict(full=count,requested=requested[stage],accepted=accepted[stage])
                      for stage,count in full.items()})


def _visualize_result(out,has_accepted_states):
    """Postprocessing has its own status and never changes solver completion."""
    status=dict(visualization_status='running' if has_accepted_states else 'skipped',
        reason=None if has_accepted_states else 'No accepted states',result_file='result.json')
    status_path=out/'visualization_status.json';write_json(status_path,status)
    if not has_accepted_states:return
    try:
        from tools import visualize_closed_crack
        script=Path(visualize_closed_crack.__file__).read_bytes()
        target=out/'source'/'tools'/'visualize_closed_crack.py';target.parent.mkdir(exist_ok=True)
        target.write_bytes(script);status['script_sha256']=hashlib.sha256(script).hexdigest()
        visualize_closed_crack.render(out)
        status.update(visualization_status='completed',manifest='visualization/manifest.json')
    except Exception as exc:
        status.update(visualization_status='failed',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    write_json(status_path,status)


def run(args):
    validate_run_args(args)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'states').mkdir()
    fingerprints=source_hashes(snapshot=out/'source')
    baseline_bytes=BASELINE_PATH.read_bytes()
    fingerprints['prototype/ipc_baseline.json']=hashlib.sha256(baseline_bytes).hexdigest()
    (out/'source'/'prototype'/'ipc_baseline.json').write_bytes(baseline_bytes)
    full_rows=schedule();requested_rows=full_rows[:args.max_steps] if args.max_steps else full_rows
    report=dict(status='running',completed=False,component_test=True,nucleation_prediction=False,
        prescribed_debonded_seam=True,experimental_validation=False,parameters=vars(args).copy(),
        source_sha256=fingerprints,baseline_config_sha256=fingerprints['prototype/ipc_baseline.json'],steps=[],
        visualization_status_file='visualization_status.json',
        requested_schedule='full' if len(requested_rows)==len(full_rows) else 'prefix',
        force_convention=FORCE_CONVENTION,
        energy_audit=dict(external_work='Trapezoidal integration of consecutive accepted actuator forces',
            endpoint_virtual_work='Assembly identity only; not an incremental energy balance',
            interpretation='Energy residual is diagnostic; load-step convergence has not been established'),
        assumptions=['Pre-existing vertical seam for contact component validation only',
                     'Crack-face friction zero; knife friction 0.35 with stated smooth slip law',
                     'Two-pass nodal seam contact, not a mortar patch-test guarantee',
                     'Two 1 x 1 x 20 mm blocks; total component geometry 2 x 1 x 20 mm',
                     'Small two-block geometry does not replace 60 degree 3 mm experiment'])
    report['parameters']['output']=str(out);_schedule_progress(report,full_rows,requested_rows)
    write_json(out/'result.json',report)
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log),threadpool_limits(limits=args.threads):
        p=ClosedCrackProblem(args.nx,args.ny,penalty=args.penalty);x=p.reference.copy()
        material=dict(shear_modulus_n_mm2=p.config.shear_modulus_n_mm2,poisson_ratio=p.config.poisson_ratio,
            bulk_modulus_n_mm2=p.config.bulk_modulus_n_mm2,width_mm=p.config.width_mm,
            law='isochoric neo-Hookean plus quadratic-J volumetric energy',kinematics='plane_strain',F33=1.)
        geometry=dict(length_mm=2.,height_mm=1.,width_mm=p.config.width_mm,
            per_body_dimensions_mm=[1.,1.,p.config.width_mm],body_count=2,
            x_extent_mm=[-1.,1.],y_extent_mm=[0.,1.],reference_volume_mm3=float(p.volumes.sum()),
            prescribed_seam_x_mm=0.,initial_seam_opening_mm=0.)
        contact=dict(crack_face_mu=0.,knife_mu=p.knife_mu,knife_tip_radius_mm=p.radius,
            knife_full_angle_deg=23.,penalty_n_mm=p.penalty,friction_smoothing_mm=BASELINE['friction_smoothing_mm'])
        report.update(material=material,geometry=geometry,contact=contact,body_nodes=p.n,elements=len(p.tri))
        write_json(out/'config.json',dict(material=material,geometry=geometry,contact=contact,
            parameters=report['parameters'],assumptions=report['assumptions'],force_convention=FORCE_CONVENTION,
            full_schedule=[dict(stage=stage,loading=vars(load)) for stage,load in full_rows]))
        previous_energy=p.bulk(x,False)[0]
        previous_clamp=np.zeros(len(p.fixed));previous_knife=np.zeros(2)
        save_npz(out/'initial_state.npz',positions=x,reference_xy=p.reference,triangles=p.tri,
            accepted_multipliers=p.multipliers,clamp_external_force_n=previous_clamp,
            knife_external_force_n=previous_knife,**vars(p.previous_load))
        cumulative_work=0.;cumulative_dissipation=0.;initial_energy=previous_energy
        start=time.monotonic()
        try:
            for index,(stage,load) in enumerate(requested_rows):
                old=x.copy();old_multipliers=p.multipliers.copy();old_load=p.previous_load
                try:x,balance=p.solve(x,load,start+args.seconds)
                except Exception:
                    iteration=getattr(p,'trial_iteration',dict(outer=-1,inner=-1,phase='initial',line_search=-1))
                    trial_load=getattr(p,'trial_load',load)
                    save_npz(out/'rejected.npz',positions=getattr(p,'trial',old),previous_positions=old,
                        reference_xy=p.reference,triangles=p.tri,
                        trial_multipliers=getattr(p,'trial_multipliers',old_multipliers),accepted_multipliers=old_multipliers,
                        trial_loading_json=json.dumps(vars(trial_load)),accepted_loading_json=json.dumps(vars(old_load)),
                        trial_iteration_json=json.dumps(iteration),**vars(trial_load))
                    report['failed_trial_iteration']=iteration
                    write_json(out/'failure.json',dict(stage=stage,index=index,loading=vars(load),
                        accepted_loading=vars(old_load),trial_loading=vars(trial_load),trial_iteration=iteration,
                        last_accepted_index=index-1,outer_history=p.log_history))
                    p.multipliers=old_multipliers;p.previous_load=old_load;p.previous=old.copy();raise
                balance.update(stage=stage,loading=vars(load),index=index)
                delta=(x-old).ravel()[p.fixed];knife_delta=np.array([0.,load.tip_y-old_load.tip_y])
                current_clamp=np.asarray(balance['clamp_external_force_n']);current_knife=np.asarray(balance['knife_external_force_n'])
                clamp_work=trapezoidal_work(previous_clamp,current_clamp,delta)
                knife_work=trapezoidal_work(previous_knife,current_knife,knife_delta)
                external_work=clamp_work+knife_work;change=balance['elastic_energy_n_mm']-previous_energy
                surplus=external_work-change-balance['friction_work_n_mm']
                tolerance=1e-5+.05*max(abs(change),abs(external_work))
                cumulative_work+=external_work;cumulative_dissipation+=balance['friction_work_n_mm']
                balance.update(clamp_trapezoidal_work_n_mm=clamp_work,knife_trapezoidal_work_n_mm=knife_work,
                    external_trapezoidal_work_n_mm=external_work,elastic_energy_change_n_mm=change,
                    trapezoidal_energy_surplus_n_mm=surplus,energy_audit_tolerance_n_mm=tolerance,
                    energy_balance_resolved=abs(surplus)<=tolerance,cumulative_external_work_n_mm=cumulative_work,
                    cumulative_friction_work_n_mm=cumulative_dissipation,
                    cumulative_energy_surplus_n_mm=cumulative_work-(balance['elastic_energy_n_mm']-initial_energy)-cumulative_dissipation)
                q=p.bulk(x,False)[3]
                save_npz(out/'states'/f'{index:04d}.npz',positions=x,previous_positions=old,reference_xy=p.reference,
                    triangles=p.tri,cauchy=q['cauchy'],J=q['J'],multipliers=p.multipliers,seams=np.array(p.seams),
                    clamp_external_force_n=current_clamp,knife_external_force_n=current_knife,**vars(load))
                report['steps'].append(balance);previous_energy=balance['elastic_energy_n_mm']
                previous_clamp=current_clamp.copy();previous_knife=current_knife.copy()
                _schedule_progress(report,full_rows,requested_rows)
                print('ACCEPT '+json.dumps(balance),flush=True);write_json(out/'result.json',report)
            report.update(status='completed' if report['full_schedule_completed'] else 'prefix_completed',completed=True)
        except Exception as exc:
            report.update(status='failed',error=f'{type(exc).__name__}: {exc}',attempted_stage=stage,attempted_load=vars(load))
            traceback.print_exc()
        report['elapsed_s']=time.monotonic()-start;write_json(out/'result.json',report)
        _visualize_result(out,bool(report['steps']))
    print(json.dumps({k:report.get(k) for k in ('status','completed','full_schedule_completed','accepted_step_count','elapsed_s','error')}))
    return report['completed']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nx',type=int,default=5);parser.add_argument('--ny',type=int,default=16)
    parser.add_argument('--penalty',type=float,default=DEFAULT_PENALTY)
    parser.add_argument('--seconds',type=float,default=180.)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--max-steps',type=int,default=0,help='Bounded prefix; 0 executes the full component schedule')
    args=parser.parse_args()
    try:validate_run_args(args)
    except ValueError as exc:parser.error(str(exc))
    raise SystemExit(0 if run(args) else 1)


if __name__=='__main__':main()
