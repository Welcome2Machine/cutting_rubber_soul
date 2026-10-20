"""Bounded real-state qualification of the optional shear-limited interface."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits
from run_support import compute_process_policy,write_json
from tools.check_material_shape import COMMON
from tools.run_material_candidate import make_problem


def run(out,strength):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    root=Path(__file__).resolve().parents[1]
    report=dict(status='running',strength_mpa=strength,rows=[],input_sha256={},
        scope='Original histories, fixed-state derivative/limit checks; NOT candidate equilibrium forces')
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    report['source_sha256']={str(p.relative_to(root)):sha(p) for pattern in ('prototype/*.py','mechanics/*.py') for p in root.glob(pattern)}
    try:
        with threadpool_limits(2),compute_process_policy():
            for angle,case in ((33,'01_angle33'),(57,'02_angle57')):
                folder=root/'outputs/native_angles_2mm_20261016'/case
                original=read(folder/'result.json');old_audit=read(folder/'audit.json')
                for name,digest in original['source_sha256'].items():
                    if name.startswith(('prototype/','mechanics/')) and sha(root/name)!=digest:
                        raise ValueError('Original physical source changed: '+name)
                base=make_problem(angle=angle,**COMMON)
                p=make_problem(angle=angle,**COMMON,shear_limit_mpa=strength,actual_affine_policy=True)
                for index in (25,50):
                    path=folder/'states'/f'{index:04d}.npz';digest=sha(path)
                    if digest!=old_audit['rows'][index]['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                    report['input_sha256'][str(path)]=digest
                    with np.load(path,allow_pickle=False) as z:state={k:z[k] for k in z.files}
                    control=base.audit_state(state,original['steps'][index])
                    x=state['positions'];p.previous=state['previous_positions'].copy()
                    p.previous_material_metric=state['previous_material_metric'].copy();p.material_dt=float(state['material_dt_s'])
                    lag=p.lag(x);delta=(x-p.previous)[p.surface]
                    local=p.friction.hessian(lag,p.cmesh,delta,p.ipc.PSDProjectionMethod.NONE).tocoo()
                    H=coo_matrix((local.data,(p.surface_dofs[local.row],p.surface_dofs[local.col])),shape=(p.total,p.total)).tocsc()
                    J=p.coupled_friction_tangent(x,lag,H)
                    def gradient(y):
                        g=np.zeros(p.total)
                        g[p.surface_dofs]=p.friction.gradient(p.lag(y),p.cmesh,(y-p.previous)[p.surface])
                        return g
                    errors=[];rng=np.random.default_rng(181033+angle+index)
                    for direction in range(2):
                        v=np.zeros_like(x);v[p.surface]=rng.normal(size=(len(p.surface),2))*.01
                        predicted=J@v.ravel()
                        for h in (1e-6,3e-7):
                            actual=(gradient(x+h*v)-gradient(x-h*v))/(2*h)
                            error=float(np.linalg.norm(actual-predicted)/max(1e-12,np.linalg.norm(actual)))
                            errors.append(error)
                            if error>3e-4:raise RuntimeError(f'Fresh friction tangent error {error}')
                    field=p.interface_field(x);active=field['normal_n']>1e-8
                    fresh=p.evaluate(x,lag,tangent=False)[1]
                    work=float(gradient(x)@(x-p.previous).ravel())
                    if work < -1e-10:raise ValueError('Negative candidate friction work')
                    # Huge cap gives original law with identical geometry and histories.
                    p.shear_limit_mpa=1e12
                    base.previous=p.previous.copy();native=base.lag(x)
                    error=float(np.linalg.norm(p.friction.gradient(p.lag(x),p.cmesh,delta)-
                                                base.friction.gradient(native,base.cmesh,delta)))
                    p.shear_limit_mpa=strength
                    if error>1e-10:raise RuntimeError('Infinite-cap native limit failed')
                    row=dict(angle=angle,depth_mm=float(state['depth_mm']),control_audit=control,
                        tangent_relative_errors=errors,infinite_cap_gradient_error_n=error,
                        candidate_fixed_field_residual_n=float(np.linalg.norm(fresh[p.free])),
                        candidate_friction_work_n_mm=work,
                        active_material_nodes=int(active.sum()),pressure_quantiles_mpa=np.quantile(field['pressure_mpa'][active],[0,.25,.5,.75,1]).tolist(),
                        mu_quantiles=np.quantile(field['mu'][active],[0,.25,.5,.75,1]).tolist())
                    report['rows'].append(row);write_json(out/'qualification.json',report)
                    print('QUALIFIED',angle,row['depth_mm'],max(errors),flush=True)
            report.update(status='passed',new_fe_path=False)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_s']=time.monotonic()-start
        report['sources_unchanged']=all(sha(root/k)==v for k,v in report['source_sha256'].items())
        report['inputs_unchanged']=all(sha(Path(k))==v for k,v in report['input_sha256'].items())
        write_json(out/'qualification.json',report)
        (out/'qualification_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps({k:report.get(k) for k in ('status','elapsed_s','error')}),flush=True)


def review_queue(out,manifest,state):
    """Automatic common-range FE/experiment decision; never launch on a fit."""
    import csv
    from tools.run_material_candidate import load_force_csv
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
    ed,ef,_=load_force_csv(manifest['experiment'])
    with Path(manifest['image_estimates']).open(encoding='utf-8-sig',newline='') as stream:
        image=list(csv.DictReader(stream))
    anchors=(.25,.5);rows=[];curves={};surface=[]
    for job in state['jobs']:
        folder=Path(job['output'])
        if not (folder/'result.json').exists():continue
        candidate=read(folder/'result.json');base=read(Path(job['baseline'])/'result.json')
        if not candidate['steps']:continue
        angle=job['angle'];cr=candidate['steps'];br=base['steps']
        if candidate['signature']['config']!=base['signature']['config'] or candidate['signature']['material']!=base['signature']['material']:
            raise ValueError('Interface comparison changed common bulk/geometry/friction parameters')
        expected=dict(name='nodal-current-area-smooth-shear-limit',strength_mpa=manifest['shear_limit_mpa'],
                      smooth_min_power=4,area='current rubber boundary dual area')
        if candidate['signature'].get('interface_law')!=expected:raise ValueError('Interface law differs from manifest')
        cd=np.array([r['depth_mm'] for r in cr]);cf=np.array([r['reaction_fz_n'] for r in cr])
        bd=np.array([r['depth_mm'] for r in br]);bf=np.array([r['reaction_fz_n'] for r in br])
        curves[angle]=(cd,cf,bd,bf)
        for d in anchors:
            if d>cd[-1]+1e-10:continue
            match=next((r for r in cr if abs(r['depth_mm']-d)<1e-10),None)
            if match is None:raise ValueError('Required nominal checkpoint missing')
            if angle==57:
                exp=float(np.interp(d,ed,ef));lo=hi=None
            else:
                point=next(r for r in image if abs(float(r['depth_mm'])-d)<1e-10)
                exp=float(point['force_n']);lo=float(point['lower_n']);hi=float(point['upper_n'])
            rows.append(dict(angle=angle,depth_mm=d,baseline_fz_n=float(np.interp(d,bd,bf)),
                candidate_fz_n=match['reaction_fz_n'],experiment_n=exp,visual_lower_n=lo,visual_upper_n=hi,
                min_j=match['min_j'],tip_status=match['tip_contact']['status'],interface=match['interface']))
        # Compare identical reference material nodes, not spatially nearest points.
        p=make_problem(angle=angle,**COMMON)
        sn=p.mesh.local_sn;select=(np.abs(sn[:,1]-p.config.height_mm)<1e-9)&(abs(sn[:,0]-p.config.drop_s_mm)<=1.)
        for d in anchors:
            ci=next((i for i,r in enumerate(cr) if abs(r['depth_mm']-d)<1e-10),None)
            bi=next((i for i,r in enumerate(br) if abs(r['depth_mm']-d)<1e-10),None)
            if ci is None or bi is None:continue
            with np.load(folder/'states'/f'{ci:04d}.npz',allow_pickle=False) as c, np.load(Path(job['baseline'])/'states'/f'{bi:04d}.npz',allow_pickle=False) as b:
                np.testing.assert_array_equal(c['reference_xy'],b['reference_xy'])
                du=c['positions'][:p.n][select]-b['positions'][:p.n][select]
                surface.append(dict(angle=angle,depth_mm=d,reference_band_halfwidth_mm=1.,nodes=int(select.sum()),
                    displacement_change_rms_mm=float(np.sqrt(np.mean(np.sum(du*du,axis=1)))),
                    displacement_change_max_mm=float(np.linalg.norm(du,axis=1).max()),
                    interpretation='Same material-node displacement change, NOT validation or cumulative slip'))
    complete=all(j.get('mechanics_completed') is True for j in state['jobs']) and len(rows)==4
    answer=dict(status='completed_comparison' if complete else 'partial_or_unavailable',rows=rows,
                surface_change=surface,force_screen_passed=False,experimental_validation=False,
                next_action='No automatic depth extension; joint force and motion review required')
    if complete:
        experiment={a:[next(r for r in rows if r['angle']==a and abs(r['depth_mm']-d)<1e-10)['experiment_n'] for d in anchors] for a in (33,57)}
        def metric(key):
            v={a:np.array([next(r for r in rows if r['angle']==a and abs(r['depth_mm']-d)<1e-10)[key] for d in anchors]) for a in (33,57)}
            return dict(rmse33_n=float(np.sqrt(np.mean((v[33]-experiment[33])**2))),
                rmse57_n=float(np.sqrt(np.mean((v[57]-experiment[57])**2))),
                ratio_error_rms=float(np.sqrt(np.mean((v[33]/v[57]-np.array(experiment[33])/experiment[57])**2))))
        bm=metric('baseline_fz_n');cm=metric('candidate_fz_n')
        gates=dict(force33=cm['rmse33_n']<=.85*bm['rmse33_n'],force57=cm['rmse57_n']<=bm['rmse57_n']+.1,
                   angle_ratio=cm['ratio_error_rms']<=.85*bm['ratio_error_rms'])
        answer.update(baseline_metrics=bm,candidate_metrics=cm,gates=gates,force_screen_passed=all(gates.values()),
            gate_definition='Same predeclared two-anchor 15% / +0.1 N / 15% screen; central screenshot readings are estimates, not ground truth')
    write_json(out/'interface_review.json',answer)
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--strength',type=float,default=1.)
    a=parser.parse_args();run(a.output,a.strength)
