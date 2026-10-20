"""Five independent matched-normal-history paths, with bounded numerical controls.

Serial frozen queue after the existing discretization queue. Primary paths and
controls stop at NORMAL depth .1 mm. No automatic .25/1/2/3 mm extension.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from prototype.inclined_loading import InclinedPathProblem
from run_support import source_hashes,write_json
from tools import check_native_mesh as runner
from tools import run_material_queue as queue
from tools.check_cutting_discretization import signature as layout_signature
from tools.run_cutting_discretization import verify_case,wait_for_predecessor

ROOT=Path(__file__).resolve().parents[1]
CASES=(('A0',0,'normal'),('B33',33,'normal'),('C33',33,'oblique'),
       ('B57',57,'normal'),('C57',57,'oblique'))
CONTROLS=('base','halfstep','order4','radius1')


def make_problem(angle,mode,*,radius=.5,order=3,frame_rotation=0.,coarse=False,compact_assembly=False):
    p=InclinedPathProblem(specimen_angle=angle,path_mode=mode,normal_speed=.1,
        frame_rotation=frame_rotation,maxwell_shear=1.24,relaxation_s=1.7,
        bulk_modulus=23.34333333333331,case='oblique',mu=.35,shear_modulus=.14,
        ogden_alpha=2.,dhat=.0025,local_spacing=.25 if coarse else .05,
        surface_spacing=.025 if coarse else .005,nose_facets=64,tip_radius=.02,
        fine_halfwidth=1. if coarse else 3.,max_iterations=500,predictor='secant',
        hessian_mode='regularized',element='p1',kinematics='plane_strain',
        enrichment_radius=radius,quadrature_order=order)
    p.stagnation_window=40
    if compact_assembly:
        from prototype.compact_p2_assembly import CompactP2AssemblyMixin
        p.__class__=type('CompactInclinedPathProblem',(CompactP2AssemblyMixin,InclinedPathProblem),{})
    return p


def signature(p):
    return dict(layout_signature(p),loading_path=p.motion_signature())


def geometry_audit(p,base,**unused):
    assert p.config.to_dict()==base.config.to_dict() and p.material==base.material
    assert p.loading_rate==.1 and base.loading_rate==.2
    np.testing.assert_array_equal(p.mesh.reference_xy[:base.n],base.mesh.reference_xy)
    np.testing.assert_array_equal(p.mesh.triangles,base.mesh.triangles)
    np.testing.assert_array_equal(p.mesh.bottom_nodes,base.mesh.bottom_nodes)
    np.testing.assert_array_equal(p.cmesh.edges,base.cmesh.edges)
    np.testing.assert_allclose(p.prescribed_blade_positions(0),base.prescribed_blade_positions(0),rtol=0,atol=2e-14)
    np.testing.assert_array_equal(p.previous,p.rest)
    np.testing.assert_array_equal(p.material_metric,np.tile(np.eye(3),(len(p.material_metric),1,1)))
    return dict(passed=True,common_material=True,original_trace=True,original_fixed_bottom=True,
        zero_depth_blade_identical=True,fresh_histories=True,motion=p.motion_signature(),
        scope='Motion and normal speed intentionally differ from legacy vertical runs')


def snapshot(out):
    frozen=source_hashes(packages=('prototype','mechanics'),snapshot=out/'source')
    for name in ('tools/run_angle_path_study.py','tools/check_native_mesh.py',
        'tools/run_material_candidate.py','tools/check_cutting_discretization.py',
        'tools/run_cutting_discretization.py','tools/run_material_queue.py','prototype/ipc_baseline.json'):
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest);frozen[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    return frozen


def export_case(out,r,*,problem=None):
    """Original top material identities, full forces and work; never filled slip."""
    settings=r['settings'];p=(problem if problem is not None else
        make_problem(settings['angle'],settings['mode'],
            radius=settings['radius'],order=settings['order']))
    local=p.mesh.local_sn[:p.base_nodes]
    nodes=np.flatnonzero(np.isclose(local[:,1],p.config.height_mm,atol=1e-10))
    work=0.;previous=None;force_rows=[]
    with (out/'surface_material_motion.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['normal_depth_mm','reference_node_id','reference_s_from_contact_mm',
            'tangent_displacement_mm','outward_normal_displacement_mm','x_mm','y_mm'])
        for i,row in enumerate(r['steps']):
            m=row['motion'];q=row['depth_mm']
            if previous is not None:
                work+=.5*(m['generalized_resistance_n']+previous['generalized_resistance_n'])*(q-previous['normal_depth_mm'])
            previous=m
            t=row['tip_contact']
            force_rows.append(dict(normal_depth_mm=q,normal_speed_mm_s=.1,
                tangent_travel_mm=m['tangent_travel_mm'],time_from_zero_s=m['elapsed_from_zero_s'],
                laboratory_x_mm=m['laboratory_translation_mm'][0],
                laboratory_downward_mm=-m['laboratory_translation_mm'][1],
                laboratory_rx_n=m['laboratory_reaction_xy_n'][0],
                laboratory_fz_n=m['laboratory_reaction_xy_n'][1],
                normal_reaction_n=m['normal_reaction_n'],tangent_resistance_n=m['tangent_resistance_n'],
                actuator_power_nmm_s=m['actuator_power_nmm_s'],
                input_work_from_zero_trapezoid_nmm=work,
                material_dissipation_from_start_nmm=row['material']['cumulative_viscous_dissipation_n_mm'],
                friction_incremental_work_nmm=row['friction_work_n_mm'],
                min_j=row['min_j'],tip_status=t['status'],
                reference_offset_mm=t.get('reference_offset_mm') if t['status']=='unique' else None))
            with np.load(out/'states'/f'{i:04d}.npz') as z:
                x=z['positions'][nodes];u=x-p.rest[nodes]
            w.writerows((q,int(n),local[n,0]-p.config.drop_s_mm,float(du@p.es),float(du@p.en),*xy)
                        for n,du,xy in zip(nodes,u,x))
    with (out/'force_motion.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(force_rows[0]));w.writeheader();w.writerows(force_rows)
    r['observation_scope']=dict(depth_mm='nominal reference-normal depth, NOT lab downward depth',
        work='trapezoidal actuator work from q=0; approximate, not exact dissipation balance',
        material_dissipation='includes shared prezero preparation; backward Euler endpoint integration',
        contact='pole/capsule geometric support, not true contact or accumulated slip')
    write_json(out/'result.json',r)


def case(args):
    args.level=0
    r=runner.run(args,problem_factory=lambda angle,level,radius:make_problem(
        angle,args.mode,radius=radius,order=args.order,compact_assembly=getattr(args,'compact_assembly',False)),geometry_checker=geometry_audit,
        source_snapshot=snapshot,problem_signature=signature,
        diagnostic_metadata=dict(name='matched-normal-path-v1',angle=args.angle,mode=args.mode,
            radius=args.radius,order=args.order,normal_speed=.1,prezero='shared-normal'))
    if r['steps']:
        try:export_case(args.output,r)
        except Exception as exc:
            r.update(status='export_failed',completed=False,export_error=repr(exc))
            write_json(args.output/'result.json',r);raise
    return r


def case_job(out,name,angle,mode,tag):
    j=dict(name=f'{name}_{tag}',case=name,angle=angle,mode=mode,tag=tag,mu=.35,
        target_mm=.1,radius=.5,order=3,increment=.01,solve_seconds=2400.,status='pending')
    if tag=='pilot':j.update(target_mm=.005,increment=.005,solve_seconds=360.)
    elif tag=='base':j['resume']=str(out/f'{name}_pilot')
    elif tag=='halfstep':j.update(increment=.005,solve_seconds=3600.)
    elif tag=='order4':j['order']=4
    elif tag=='radius1':j['radius']=1.
    else:raise ValueError(tag)
    j['output']=str(out/j['name'])
    return j


def command(m,j):
    if j['target_mm']>.1 or (j['case'],j['angle'],j['mode']) not in CASES:
        raise ValueError('Only five explicit .1-normal-mm paths authorized')
    cmd=[m['python'],'-X','utf8','-u','-m','tools.run_angle_path_study','--phase','case',
        '--output',j['output'],'--angle',str(j['angle']),'--mode',j['mode'],
        '--depth',str(j['target_mm']),'--seconds',str(j['solve_seconds'])]
    for key in ('radius','order','increment'):cmd+=['--'+key,str(j[key])]
    if j.get('resume'):cmd+=['--resume',j['resume']]
    return cmd


def verify_job(j):
    out=Path(j['output']);r=queue.read(out/'result.json')
    for key in ('angle','mode','radius','order','increment'):
        if r['settings'][key]!=j[key]:raise ValueError('Requested path mismatch: '+key)
    path=r['signature']['loading_path']
    if path['normal_speed_mm_s']!=.1 or path['path_mode']!=j['mode']:
        raise ValueError('Wrong loading history')
    if r.get('export_error') or not (out/'force_motion.csv').is_file():
        raise ValueError('Missing force/motion observations')
    evidence=verify_case(out);write_json(out/'evidence_hashes.json',evidence)
    return evidence


def compare_control(base,control,end=.1):
    a,b=Path(base),Path(control);ra,rb=queue.read(a/'result.json'),queue.read(b/'result.json')
    for key in ('config','material','loading_rate','loading_path','contact_formulation','dhat'):
        if ra['signature'][key]!=rb['signature'][key]:raise ValueError('Physical history mismatch: '+key)
    r=ra['steps'];s=rb['steps'];rows=[]
    with np.load(a/'states/0000.npz') as z:
        n=int(z['base_body_nodes']);reference=z['reference_xy'][:n]
    theta=np.deg2rad(ra['settings']['angle']);normal=np.array([np.sin(theta),np.cos(theta)])
    top=np.flatnonzero(np.isclose(reference@normal,ra['signature']['config']['height_mm'],atol=1e-10))
    for i,x in enumerate(r):
        q=x['depth_mm']
        if q>end+1e-12:continue
        match=next(((k,y) for k,y in enumerate(s) if abs(y['depth_mm']-q)<1e-10),None)
        if match is None:continue
        k,y=match
        with np.load(a/'states'/f'{i:04d}.npz') as z1,np.load(b/'states'/f'{k:04d}.npz') as z2:
            np.testing.assert_array_equal(z1['reference_xy'][top],z2['reference_xy'][top])
            motion=np.linalg.norm(z1['positions'][top]-z2['positions'][top],axis=1)
        components={key:abs(x['motion'][key]-y['motion'][key])/max(.1,abs(x['motion'][key]))
                    for key in ('normal_reaction_n','tangent_resistance_n','generalized_resistance_n')}
        components['laboratory_fz_n']=abs(x['reaction_fz_n']-y['reaction_fz_n'])/max(.1,abs(x['reaction_fz_n']))
        rows.append(dict(normal_depth_mm=q,relative_force_errors=components,
            full_top_max_motion_mm=float(motion.max()),first_min_j=x['min_j'],second_min_j=y['min_j']))
    covered=all(any(abs(x['normal_depth_mm']-q)<1e-10 for x in rows) for q in np.arange(0,end+1e-12,.01))
    force=max((max(x['relative_force_errors'].values()) for x in rows if x['normal_depth_mm']>0),default=float('inf'))
    surface=max((x['full_top_max_motion_mm'] for x in rows),default=float('inf'))
    return dict(passed=bool(covered and force<=.02 and surface<=.002),rows=rows,
        all_nominal_depths=covered,max_force_fraction=force,max_surface_mm=surface,
        scope='Post-zero time, quadrature or local enrichment control; not global field/contact convergence')


def report(out,m,state):
    jobs={(j['case'],j['tag']):j for j in state['jobs']};controls=[];pairs=[]
    for name,angle,mode in CASES:
        base=jobs[name,'base']
        for tag in CONTROLS[1:]:
            other=jobs[name,tag]
            if all(j.get('mechanics_completed') and j.get('evidence_verified') for j in (base,other)):
                try:c=compare_control(base['output'],other['output'])
                except Exception as exc:c=dict(passed=False,error=repr(exc))
                controls.append(dict(case=name,control=tag,**c))
    for angle in (33,57):
        b,c=jobs[f'B{angle}','base'],jobs[f'C{angle}','base']
        if not all(j.get('evidence_verified') for j in (b,c)):continue
        rb,rc=queue.read(Path(b['output'])/'result.json'),queue.read(Path(c['output'])/'result.json')
        rows=[]
        for row in rb['steps']:
            q=row['depth_mm'];other=next((s for s in rc['steps'] if abs(s['depth_mm']-q)<1e-10),None)
            if other is None:continue
            nb,nc=row['motion']['normal_reaction_n'],other['motion']['normal_reaction_n']
            rows.append(dict(normal_depth_mm=q,normal_only_n=nb,combined_normal_n=nc,
                delta_normal_n=nc-nb,normal_reduction_fraction=1-nc/nb if abs(nb)>.1 and q>0 else None,
                normal_only_tangent_n=row['motion']['tangent_resistance_n'],
                combined_tangent_n=other['motion']['tangent_resistance_n'],
                combined_fz_n=other['reaction_fz_n'],normal_only_fz_n=row['reaction_fz_n']))
        pair_controls=[x for x in controls if x['case'] in (f'B{angle}',f'C{angle}')]
        pairs.append(dict(angle=angle,rows=rows,complete=all(j.get('mechanics_completed') for j in (b,c)),
            numerical_controls_passed=len(pair_controls)==6 and all(x['passed'] for x in pair_controls)))
    result=dict(status=state['status'],updated_utc=queue.now(),controls=controls,paired_contrasts=pairs,
        joint_gate_passed=len(controls)==15 and all(x['passed'] for x in controls),
        experimental_validation=False,depth_coordinate='nominal normal depth',
        scope='Different speed/prezero paths from original experiment; diagnostic contrasts, not a fitted experimental curve',
        no_automatic_extension=True)
    write_json(out/'study_report.json',result)
    lines=['# Matched normal-history study','',f'Status: {state["status"]}',
        'Depth below is NORMAL depth; all histories are new. Original production queues unchanged.','',
        '| Case | Status | Normal depth mm | Fz N | Fresh audit / hashes |','|---|---|---:|---:|---|']
    for j in state['jobs']:
        lines.append(f'| {j["name"]} | {j["status"]} | {j.get("last_depth_mm")} | {j.get("endpoint_fz_n")} | {j.get("evidence_verified",False)} |')
    lines+=['',f'All 15 numerical controls pass: {result["joint_gate_passed"]}',
        'No automatic .25/1/2/3 mm extension. Force and full original top motion must both pass.',
        'B/C differences are controlled contrasts, not unique additive material/friction causes.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def prepare(out,after):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Already prepared')
    frozen=snapshot(out)
    inputs={}
    # Actual .005 pilots must be completed with these exact sources before
    # launching. Controls start fresh; base paths resume only their own pilot.
    for name,angle,mode in CASES:
        j=case_job(out,name,angle,mode,'pilot')
        if not queue.summarize_case(j['output'],j,0)['mechanics_completed']:
            raise ValueError('Missing successful actual pilot: '+name)
        if queue.read(Path(j['output'])/'result.json')['source_sha256']!=frozen:
            raise ValueError('Pilot sources no longer match')
        inputs.update(verify_job(j))
    after=after.resolve()
    for name in ('manifest.json','launch.json'):
        path=after/name;inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    jobs=[case_job(out,name,angle,mode,tag) for tag in CONTROLS for name,angle,mode in CASES]
    m=dict(python=sys.executable,source=str(out/'source'),source_sha256=frozen,input_sha256=inputs,
        runtime_path=str(ROOT/'.runtime/ipc-python'),jobs=jobs,solve_seconds_per_case=2400.,audit_allowance_s=600.,
        predecessor=dict(output=str(after),wait_seconds=25200.,poll_seconds=30.),
        created_utc=queue.now(),plan='docs/history/topics/angle_force_mechanism.md',target_normal_mm=.1,
        no_automatic_extension=True)
    queue.verify_inputs(m);write_json(out/'manifest.json',m)
    state=dict(status='prepared',jobs=copy.deepcopy(jobs));write_json(out/'queue_status.json',state)
    report(out,m,state)
    return m


def worker(out):
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),started_utc=queue.now()),f)
    m=queue.read(out/'manifest.json');state=dict(status='running',jobs=copy.deepcopy(m['jobs']),started_utc=queue.now())
    if not wait_for_predecessor(out,m,state):report(out,m,state);return
    for j in state['jobs']:
        try:
            j.update(queue.run_case(out,m,j,state,command_factory=command))
            verify_job(j);j['evidence_verified']=True
        except Exception as exc:
            j.update(status='failed_or_unverified',mechanics_completed=False,evidence_verified=False,error=repr(exc))
            traceback.print_exc()
        write_json(out/'queue_status.json',state)
        try:report(out,m,state)
        except Exception as exc:
            state.setdefault('report_errors',[]).append(dict(case=j['name'],error=repr(exc)))
            traceback.print_exc()
    state.update(status='finished',finished_utc=queue.now(),all_cases_completed=all(j.get('mechanics_completed') for j in state['jobs']))
    try:
        queue.verify_inputs(m)
        for name,h in m['source_sha256'].items():
            if hashlib.sha256((Path(m['source'])/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
        for j in state['jobs']:
            if j.get('evidence_verified'):verify_job(j)
        state['terminal_hash_check_passed']=True
    except Exception as exc:state.update(terminal_hash_check_passed=False,terminal_error=repr(exc))
    write_json(out/'queue_status.json',state);report(out,m,state)


def launch(out):
    m=queue.read(out/'manifest.json');queue.verify_inputs(m)
    for name,h in m['source_sha256'].items():
        if hashlib.sha256((Path(m['source'])/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
    if (out/'launch.json').exists() or (out/'worker_started.json').exists():raise RuntimeError('Already launched')
    cmd=[m['python'],'-X','utf8','-u','-m','tools.run_angle_path_study','--phase','worker','--output',str(out)]
    env=os.environ.copy();env['PYTHONPATH']=m['runtime_path'];env['OPENBLAS_NUM_THREADS']='2';env['OMP_NUM_THREADS']='2'
    extra={}
    if os.name=='nt':
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=subprocess.SW_HIDE
        extra=dict(startupinfo=startup,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        p=subprocess.Popen(cmd,cwd=m['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            start_new_session=os.name!='nt',**extra)
    result=dict(worker_pid=p.pid,command=cmd,started_utc=queue.now());write_json(out/'launch.json',result)
    print(json.dumps(result))


def main():
    a=argparse.ArgumentParser(__doc__)
    a.add_argument('--phase',choices=('case','prepare','launch','worker','report'),required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--after',type=Path)
    a.add_argument('--angle',type=float,choices=(0.,33.,57.),default=0.)
    a.add_argument('--mode',choices=('normal','oblique'),default='normal')
    a.add_argument('--radius',type=float,default=.5);a.add_argument('--order',type=int,choices=(3,4),default=3)
    a.add_argument('--depth',type=float,default=.005);a.add_argument('--increment',type=float,default=.005)
    a.add_argument('--seconds',type=float,default=360.);a.add_argument('--threads',type=int,default=2)
    a.add_argument('--resume',type=Path)
    a.add_argument('--compact-assembly',action='store_true',help='Opt-in equivalent P1/P2 sparse assembly; case diagnostics only')
    args=a.parse_args();args.output=args.output.resolve()
    if args.compact_assembly and args.phase!='case':
        a.error('--compact-assembly is qualified only for explicit case diagnostics; no queue promotion')
    if not all(np.isfinite(v) and v>0 for v in (args.depth,args.increment,args.seconds,args.radius)) or args.depth>.1 or args.threads<1:
        a.error('Require positive finite bounded controls, normal depth <= .1 mm')
    if args.phase=='case':case(args)
    elif args.phase=='prepare':
        if args.after is None:a.error('--after existing zero-degree queue is required')
        prepare(args.output,args.after)
    elif args.phase=='launch':launch(args.output)
    elif args.phase=='worker':worker(args.output)
    else:report(args.output,queue.read(args.output/'manifest.json'),queue.read(args.output/'queue_status.json'))


if __name__=='__main__':main()
