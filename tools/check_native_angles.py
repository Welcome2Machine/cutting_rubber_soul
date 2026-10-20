"""Read-only exact-history audit of the completed native angle study.

Hash every checkpoint; independently reassemble selected physical states.
Work partitions and compressed areas are diagnostics, not causal sensitivities.
"""
import argparse
import contextlib
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from prototype.model_config import _rotation_vectors
from run_support import compute_process_policy, write_json, save_npz
from tools.run_material_candidate import make_problem, signature
from tools.check_angle_modes import prepare, deformation, tensor_parts


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(study):
    manifest=read(study/'manifest.json'); status=read(study/'queue_status.json')
    if status['status']!='finished' or not status['all_cases_completed']:
        raise ValueError('This audit requires a completed native study')
    root=Path(__file__).resolve().parents[1]
    for name,digest in manifest['source_sha256'].items():
        if sha(Path(manifest['source'])/name)!=digest:
            raise ValueError('Frozen source changed: '+name)
        if name.startswith(('prototype/','mechanics/')) or name=='run_support.py':
            if sha(root/name)!=digest:raise ValueError('Working physical source differs: '+name)
    for name,digest in manifest['input_sha256'].items():
        if sha(name)!=digest:raise ValueError('Frozen input changed: '+name)
    cases=[];checkpoints=0
    for job in status['reference_cases']+status['jobs']:
        path=Path(job['output']); report=read(path/'result.json')
        audit=read(Path(job.get('current_audit',path/'audit.json')))
        if not report['completed'] or not audit['passed'] or len(report['steps'])!=audit['states']:
            raise ValueError('Incomplete or unaudited case')
        for name,digest in report['source_sha256'].items():
            if sha(path/'source'/name)!=digest:raise ValueError('Case source changed: '+name)
        for i,row in enumerate(audit['rows']):
            if sha(path/'states'/f'{i:04d}.npz')!=row['checkpoint_sha256']:
                raise ValueError('Audited checkpoint changed')
            if abs(row['depth_mm']-report['steps'][i]['depth_mm'])>1e-12:
                raise ValueError('Audit state order changed')
            checkpoints+=1
        cases.append((job,path,report))
    return manifest,cases,dict(frozen_sources=len(manifest['source_sha256']),
        frozen_inputs=len(manifest['input_sha256']),checkpoints=checkpoints,
        manifest_sha256=sha(study/'manifest.json'),passed=True)


def field(p,state,record):
    fresh=p.audit_state(state,record);prepare(p,state)
    x=state['positions'];F=deformation(p,x);q=p.material_response(x)
    stresses,energies=tensor_parts(F,q,p.material)
    area,grad,_=p.mesh.bulk_data;vol=area*p.config.width_mm
    velocity=(x-p.previous)/(p.material_dt*p.loading_rate)
    dF=np.einsum('eai,eaJ->eiJ',velocity[:p.n][p.mesh.triangles],grad)
    work={name:float(np.sum(vol*np.einsum('eiJ,eiJ->e',P,dF))) for name,P in stresses.items()}
    c=p.collisions(x);lag=p.lag(x);sx=x[p.surface]
    ng=p.barrier.gradient(c,p.cmesh,sx)
    fg=p.friction.gradient(lag,p.cmesh,(x-p.previous)[p.surface])
    work['all_normal_contact']=float(ng@velocity[p.surface].ravel())
    work['all_friction']=float(fg@velocity[p.surface].ravel())
    _,g,_,_=p.evaluate(x,lag,tangent=False)
    bound=float(np.linalg.norm(g[p.free])*np.linalg.norm(velocity.ravel()[p.free]))
    closure=sum(work.values())-record['reaction_fz_n']
    if abs(closure)>bound+2e-7:raise RuntimeError('Independent work closure failed')
    b=F@F.transpose(0,2,1);eig,Q=np.linalg.eigh(b)
    h=(Q*(.5*np.log(eig))[:,None,:])@Q.transpose(0,2,1)
    es,en=_rotation_vectors(p.config.inclination_deg);R=np.column_stack([es,en]);hs=R.T@h@R
    tip=x[p.n:][np.argmin(x[p.n:,1])]
    centroids=x[:p.n][p.mesh.triangles].mean(axis=1)
    near=np.linalg.norm(centroids-tip,axis=1)<1.;weights=area[near]/area[near].sum()
    pressure=p.material.moduli()[1]*(1.-q['J'])
    sigma=q['P']@F.transpose(0,2,1)/q['J'][:,None,None]
    row=dict(angle_deg=p.config.inclination_deg,depth_mm=record['depth_mm'],fresh_audit=fresh,
        work_n=work,work_closure_n=closure,work_residual_bound_n=bound,
        energy_n_mm={name:float(vol@value) for name,value in energies.items()},
        min_J=float(q['J'].min()),max_stretch=float(np.sqrt(eig.max())),
        compressed_reference_area_mm2={str(t):float(area[q['J']<t].sum()) for t in (.95,.8,.5,.2,.1)},
        near_1mm=dict(reference_area_mm2=float(area[near].sum()),
            normal_log_strain_rms=float(np.sqrt(weights@hs[near,1,1]**2)),
            shear_log_strain_rms=float(np.sqrt(weights@(2*hs[near,0,1])**2)),
            volume_pressure_rms_mpa=float(np.sqrt(weights@pressure[near]**2))),
        tip_contact=record['tip_contact'],scope='Exact-history conditional field; work fractions are NOT parameter sensitivity')
    fields=dict(reference_xy=p.rest[:p.n],positions=x[:p.n],triangles=p.mesh.triangles,
        blade=x[p.n:],J=q['J'],volume_pressure_mpa=pressure,cauchy=sigma,
        normal_log_strain=hs[:,1,1],shear_log_strain=2*hs[:,0,1])
    return row,fields


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',type=Path,default=Path('outputs/native_angles_2mm_20261016'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    manifest,cases,verified=verify(args.study)
    report=dict(status='running',verification=verified,fields=[],cases=[],new_loading_solves=0)
    write_json(out/'report.json',report)
    started=time.monotonic()
    with (out/'run.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),\
            threadpool_limits(2),compute_process_policy():
        for job,path,result in cases:
            p=make_problem(angle=job['angle'],**manifest['material'],mu=manifest['mu'])
            if json.loads(json.dumps(signature(p)))!=result['signature']:
                raise ValueError('Current problem signature differs from completed case')
            rows=result['steps']
            report['cases'].append(dict(angle=job['angle'],states=len(rows),
                tip_status_counts=dict(Counter(r['tip_contact']['status'] for r in rows)),
                material_dt_range_s=[min(r['material']['dt_s'] for r in rows),max(r['material']['dt_s'] for r in rows)],
                post_zero_increment_range_mm=[min(r['load_step']['increment_mm'] for r in rows[1:]),
                                             max(r['load_step']['increment_mm'] for r in rows[1:])]))
            for depth in (.25,.5,1.,2.):
                i=next(i for i,r in enumerate(rows) if abs(r['depth_mm']-depth)<1e-10)
                with np.load(path/'states'/f'{i:04d}.npz',allow_pickle=False) as state:
                    row,fields=field(p,state,rows[i])
                report['fields'].append(row)
                save_npz(out/f'angle{job["angle"]:g}_depth{depth:g}.npz',**fields)
                write_json(out/'report.json',report)
                print(f'FIELD angle={job["angle"]} depth={depth} fresh_pass=True',flush=True)
    report.update(status='completed',elapsed_s=time.monotonic()-started)
    write_json(out/'report.json',report)
    print(json.dumps(dict(status=report['status'],fresh_states=len(report['fields']),verification=verified)))


if __name__=='__main__':main()
