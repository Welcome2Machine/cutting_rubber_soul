"""Audit a saved failed contact state without advancing the physical history."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_support import source_hashes
from scipy.sparse import bmat,coo_matrix,diags
from scipy.sparse.linalg import splu,onenormest,LinearOperator
from .contact_study import load_failed_problem,mesh_metrics,write_json


def audit(directory):
    p,previous,state,old=load_failed_problem(directory)
    bounded=not (directory/'status.json').exists()
    status=json.loads((directory/('result.json' if bounded else 'status.json')).read_text(encoding='utf-8'))
    depth=status['failed_depth_mm'] if bounded else status['failed_attempt']['depth_mm']
    dt=(depth-old)/p.config.speed_mm_s if bounded else status['failed_attempt']['dt_s']
    with np.load(directory/('failed_trial.npz' if bounded else 'failed_attempt.npz')) as z:
        u,lam=z['displacement' if bounded else 'trial_displacement'],z['contact_multipliers' if bounded else 'trial_contact_multipliers']
    r,k,q,d=p.assemble(u,previous,depth,old,dt,lam[:,0],lam[:,1],mixed=True)
    selected=np.flatnonzero((q['pressure']>0)|np.any(lam!=0,axis=1))
    mdofs=2*np.arange(len(selected))[:,None]+np.arange(2)
    shape=(len(selected),2,2)
    ru=np.broadcast_to(p.cdofs[selected,:,None],shape).ravel()
    cm=np.broadcast_to(mdofs[:,None,:],shape).ravel()
    rm=np.broadcast_to(mdofs[:,:,None],shape).ravel()
    cu=np.broadcast_to(p.cdofs[selected,None,:],shape).ravel()
    if p.surface_contact is not None:
        chosen=(2*selected[:,None]+np.arange(2)).ravel()
        B=q['global_B'][p.free][:,chosen]
        C=q['global_C'][chosen][:,p.free]
        D=q['global_D'][chosen][:,chosen]
    else:
        B=coo_matrix((-q['dforce_dlambda'][selected].ravel(),(ru,cm)),shape=(len(u),2*len(selected))).tocsr()[p.free]
        C=coo_matrix((q['projection_u'][selected].ravel(),(rm,cu)),shape=(2*len(selected),len(u))).tocsr()[:,p.free]
        D=coo_matrix((q['projection_lambda'][selected].ravel(),(rm,cm)),shape=(2*len(selected),2*len(selected))).tocsr()
    W=diags(np.repeat(p.weights[selected],2))
    jac=bmat([[k[p.free][:,p.free],B],[W@C,W@D]],format='csc')
    def evaluate(displacement,multiplier):
        rr,_,contact,_=p.assemble(displacement,previous,depth,old,dt,multiplier[:,0],multiplier[:,1],mixed=True)
        defect=np.column_stack((contact['pressure'],contact['shear']))-multiplier
        return np.r_[rr[p.free],(p.weights[selected,None]*defect[selected]).ravel()],contact
    rhs,_=evaluate(u,lam);lu=splu(jac);delta=lu.solve(-rhs)
    result=dict(source=str(directory),source_status=status,mesh=mesh_metrics(p),
        trial_residual_n=float(np.linalg.norm(r[p.free])),selected_contact_nodes=len(selected),
        linear_relative_residual=float(np.linalg.norm(jac@delta+rhs)/np.linalg.norm(rhs)),
        linear_residual_definition='norm2(J*delta+R)/norm2(R)',
        condition_estimate_note='1-norm estimate depends on the displacement/traction units; not proof of causation')
    inverse=LinearOperator(jac.shape,matvec=lu.solve,rmatvec=lambda x:lu.solve(x,trans='T'))
    # scipy's estimator samples signs from NumPy's legacy RNG. Preserve the
    # caller's state while making repeated audits of this state reproducible.
    random_state=np.random.get_state()
    try:
        np.random.seed(17)
        result['jacobian_condition_1_estimate']=float(onenormest(jac)*onenormest(inverse))
    finally:
        np.random.set_state(random_state)
    result['condition_estimator_seed']=17
    row_scale=1/np.maximum(np.abs(jac).max(axis=1).toarray().ravel(),1e-30)
    row_scaled=diags(row_scale)@jac
    col_scale=1/np.maximum(np.abs(row_scaled).max(axis=0).toarray().ravel(),1e-30)
    scaled=(row_scaled@diags(col_scale)).tocsc();scaled_lu=splu(scaled)
    scaled_delta=col_scale*scaled_lu.solve(-row_scale*rhs)
    result['equilibrated_linear_solve']=dict(relative_direction_difference=float(np.linalg.norm(scaled_delta-delta)/np.linalg.norm(delta)),
        original_relative_residual=float(np.linalg.norm(jac@scaled_delta+rhs)/np.linalg.norm(rhs)))
    directions=[]
    rng=np.random.default_rng(17)
    for kind in ('bulk_and_contact','contact_local','newton'):
        if kind=='newton':v=delta/max(np.linalg.norm(delta),1e-30)
        else:
            du=rng.normal(size=len(u));du[p.fixed]=0
            if kind=='contact_local':
                keep=np.unique(p.cdofs[selected]);du[np.setdiff1d(np.arange(len(u)),keep)]=0
            du/=max(np.linalg.norm(du),1e-30)
            dl=rng.normal(size=(len(selected),2));dl*=250/max(np.linalg.norm(dl),1e-30)
            v=np.r_[du[p.free],dl.ravel()]
        du=np.zeros_like(u);du[p.free]=v[:len(p.free)]
        dl=np.zeros_like(lam);dl[selected]=v[len(p.free):].reshape(-1,2)
        expected=jac@v
        for h in ((1e-3,1e-4,1e-5,1e-7) if kind=='newton' else (1e-7,1e-8,1e-9)):
            a,qa=evaluate(u+h*du,lam+h*dl);b,qb=evaluate(u-h*du,lam-h*dl)
            fd=(a-b)/(2*h)
            directions.append(dict(kind=kind,epsilon=h,
                relative_error=float(np.linalg.norm(expected-fd)/max(np.linalg.norm(fd),1e-30)),
                absolute_error=float(np.linalg.norm(expected-fd)),analytic_action_norm=float(np.linalg.norm(expected)),
                normal_branch_changes=int(np.sum((qa['pressure']>0)!=(qb['pressure']>0))),
                friction_branch_changes=int(np.sum(qa['stick']!=qb['stick']))))
    result['directional_checks']=directions
    # Audit the actual Newton step in the SAME all-contact, tolerance-scaled
    # merit used by surface-contact globalization. A good linear solve alone
    # does not imply its full nonlinear step stays on the linearized branches.
    def metrics(displacement,multiplier):
        rr,_,contact,diagnostics=p.assemble(displacement,previous,depth,old,dt,
            multiplier[:,0],multiplier[:,1],mixed=True,with_tangent=False)
        defect=np.column_stack((contact['pressure'],contact['shear']))-multiplier
        return dict(residual_n=float(np.linalg.norm(rr[p.free])),
            projection_mpa=float(np.max(np.abs(defect))),
            normalized_merit=float(np.hypot(np.linalg.norm(rr[p.free])/p.config.newton_tolerance_n,
                np.linalg.norm(defect)/p.config.contact_multiplier_tolerance_n_mm2)),
            min_J=float(diagnostics['min_j']),
            normal_branch_changes=int(np.sum((contact['pressure']>0)!=(q['pressure']>0))),
            friction_branch_changes=int(np.sum(contact['stick']!=q['stick'])))
    baseline=metrics(u,lam);backtracking=[]
    for cut in range(p.config.max_line_search_iterations):
        alpha=.5**cut;trial=u.copy();trial[p.free]+=alpha*delta[:len(p.free)]
        multiplier=lam.copy();multiplier[selected]+=alpha*delta[len(p.free):].reshape(-1,2)
        try:
            values=metrics(trial,multiplier)
            values['armijo_pass']=bool(values['normalized_merit']<baseline['normalized_merit']*(1-1e-4*alpha))
        except FloatingPointError as exc:
            values=dict(admissible_domain=False,error=str(exc),armijo_pass=False)
        backtracking.append(dict(alpha=alpha,**values))
    result['newton_backtracking']=dict(baseline=baseline,trials=backtracking,
        note='Diagnostic only; no physical history advanced and no converged state certified')
    physical=p.assemble(previous,previous,old,old,.01,state[:,0],state[:,1])
    fields=p.fields(previous);loads=p.weights*state[:,0]
    result['accepted_state']=dict(residual_n=float(np.linalg.norm(physical[0][p.free])),
        residual_note='Zero-increment replay, not the original accepted loading increment',
        min_J=float(fields['J'].min()),max_J=float(fields['J'].max()),
        local_nominal_strength_ratio=float(fields['nominal_tensile_mpa'].max()/p.config.tensile_strength_n_mm2),
        max_reference_pressure_mpa=float(state[:,0].max()),
        max_node_normal_load_fraction=float(loads.max()/loads.sum()),
        max_boundary_penetration_mm=p.edge_penetration(previous,old))
    xy=p.mesh.reference_xy[p.mesh.triangles];v=np.roll(xy,-1,axis=1)-xy;l=np.linalg.norm(v,axis=2)
    angles=np.rad2deg(np.arccos(np.clip(np.sum(v*(-np.roll(v,1,axis=1)),axis=2)/(l*np.roll(l,1,axis=1)),-1,1)))
    bad=np.argsort(angles.min(axis=1))[:5]
    result['worst_triangles']=[dict(element=int(i),min_angle_deg=float(angles[i].min()),
        local_sn_mm=p.mesh.local_sn[p.mesh.triangles[i]].tolist()) for i in bad]
    result['audit_source_sha256']=source_hashes()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.source);write_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('linear_relative_residual','jacobian_condition_1_estimate','equilibrated_linear_solve')},indent=2))


if __name__=='__main__':main()
