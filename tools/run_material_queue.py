"""Frozen, detached intact Maxwell cases (default: 60 -> 30 -> 0).

Resume only the 60-degree history, or use --fresh for independent rest states. Every case
has a finite budget and advances after success, numerical failure or crash.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from run_support import source_hashes, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize_case(folder,job,returncode,forced_status=None):
    """An exited process and result flag alone do not establish completion."""
    folder=Path(folder)
    try:r=read(folder/'result.json')
    except (OSError,ValueError):r={}
    try:audit=read(folder/'audit.json')
    except (OSError,ValueError):audit={}
    rows=r.get('steps',[])
    tol=r.get('force_tolerance_n',math.inf)
    finite=lambda x:isinstance(x,(int,float)) and math.isfinite(x)
    records_ok=bool(rows) and finite(tol) and 0<tol<=1e-4
    for i,row in enumerate(rows):
        vals=[row.get(k) for k in ('depth_mm','reaction_fz_n','residual_n','min_j','minimum_distance_mm')]
        records_ok=records_ok and all(finite(v) for v in vals)
        if records_ok:
            records_ok=(0<=row['residual_n']<=tol and row['min_j']>0 and row['minimum_distance_mm']>0
                and (folder/'states'/f'{i:04d}.npz').is_file()
                and (i==0 or row['depth_mm']>rows[i-1]['depth_mm']))
    actual_angle=r.get('signature',{}).get('config',{}).get('inclination_deg')
    actual_mu=r.get('signature',{}).get('config',{}).get('friction_coefficient')
    requested_mu_ok='mu' not in job or actual_mu==job['mu']
    requested_interface_ok=('shear_limit_mpa' not in job or
        r.get('signature',{}).get('interface_law',{}).get('strength_mpa')==job['shear_limit_mpa'])
    requested_discretization_ok=True
    if job.get('diagnostic')=='cutting_discretization':
        settings=r.get('settings',{})
        requested_discretization_ok=all(settings.get(k)==job[k] for k in
            ('kind','kinematics','radius','order','increment'))
        requested_discretization_ok &= not r.get('surface_export_error',False)
    accepted_prefix_ok=bool(records_ok)
    increment=r.get('settings',{}).get('increment')
    records_ok=records_ok and finite(increment) and increment>0
    if records_ok:
        nominal=[min(i*increment,job['target_mm']) for i in range(math.ceil(job['target_mm']/increment-1e-12)+1)]
        depths=[x['depth_mm'] for x in rows]
        next_anchor=0
        for depth in depths:
            if next_anchor<len(nominal) and abs(depth-nominal[next_anchor])<=1e-9:next_anchor+=1
        records_ok=(next_anchor==len(nominal) and abs(depths[0])<=1e-9
            and all(0<b-a<=increment+1e-9 for a,b in zip(depths[:-1],depths[1:])))
    reached=bool(rows) and abs(rows[-1].get('depth_mm',-1)-job['target_mm'])<=1e-9
    audit_ok=(audit.get('passed') is True and audit.get('states')==len(rows)
              and len(audit.get('rows',[]))==len(rows)
              and all(x.get('passed') is True for x in audit.get('rows',[])))
    completed=(forced_status is None and returncode==0 and r.get('status')=='completed'
        and r.get('completed') is True and r.get('target_mm')==job['target_mm']
        and actual_angle==job['angle'] and requested_mu_ok and requested_interface_ok
        and requested_discretization_ok and records_ok and reached and audit_ok)
    status='completed' if completed else (forced_status or r.get('status','process_failed'))
    if not completed and status in ('completed','running','resuming','initializing'):status='incomplete_or_unverified'
    return dict(status='completed' if completed else status,mechanics_completed=bool(completed),
        angle=job['angle'],mu=actual_mu,requested_mu_matches=requested_mu_ok,
        requested_interface_matches=requested_interface_ok,
        requested_discretization_matches=bool(requested_discretization_ok),
        target_mm=job['target_mm'],returncode=returncode,
        accepted_states=len(rows),last_depth_mm=rows[-1]['depth_mm'] if rows else None,
        endpoint_fz_n=rows[-1]['reaction_fz_n'] if rows else None,
        min_j=min((x.get('min_j',math.inf) for x in rows),default=None),
        fresh_audit_passed=bool(audit_ok),accepted_record_checks_passed=accepted_prefix_ok,
        full_target_coverage_passed=bool(records_ok),
        error=r.get('error') or r.get('audit_error'),solver_elapsed_s=r.get('elapsed_s'),
        prior_elapsed_s=r.get('prior_elapsed_s',0.),
        interpretation='Numerical result only; material/time/space/contact migration not experimentally validated')


def make_command(manifest,job):
    if job.get('diagnostic')=='cutting_discretization':
        if not 0<job['target_mm']<=.25 or job['angle'] not in (0,33,57):
            raise ValueError('Bounded cutting discretization: 0/33/57 through .25 mm only')
        cmd=[manifest['python'],'-X','utf8','-u','-m','tools.check_cutting_discretization',
            '--output',job['output'],'--angle',str(job['angle']),
            '--depth',str(job['target_mm']),
            '--seconds',str(job.get('solve_seconds',manifest['solve_seconds_per_case'])),
            '--threads','2']
        for key in ('kind','kinematics','radius','order','increment'):
            cmd+=['--'+key,str(job[key])]
        if job.get('resume'):cmd+=['--resume',job['resume']]
        return cmd
    if job.get('diagnostic')=='native_mesh':
        if job['target_mm']>.5 or job.get('level')!=1 or job.get('radius_mm')!=1.:
            raise ValueError('Qualified native mesh diagnostic is level1/radius1 through .5 mm only')
        cmd=[manifest['python'],'-X','utf8','-u','-m','tools.check_native_mesh',
            '--phase','run','--output',job['output'],'--angle',str(job['angle']),
            '--level','1','--radius','1','--depth',str(job['target_mm']),
            '--seconds',str(job.get('solve_seconds',manifest['solve_seconds_per_case'])),
            '--threads','2']
        if job.get('resume'):cmd+=['--resume',job['resume']]
        return cmd
    material=manifest.get('material',{})
    cmd=[manifest['python'],'-X','utf8','-u','-m','tools.run_material_candidate',
        '--output',job['output'],'--angle',str(job['angle']),
        '--equilibrium-shear',str(material.get('equilibrium_shear',.47)),
        '--maxwell-shear',str(material.get('maxwell_shear',.94)),
        '--relaxation-s',str(material.get('relaxation_s',.4)),
        '--loading-rate',str(material.get('loading_rate',.2)),
        '--mu',str(manifest.get('mu',.35)),
        '--depth',str(job['target_mm']),'--increment','0.01',
        '--seconds',str(job.get('solve_seconds',manifest['solve_seconds_per_case'])),'--threads','2']
    if job.get('resume'):cmd+=['--resume',job['resume']]
    if material.get('bulk_modulus') is not None:
        cmd+=['--bulk-modulus',str(material['bulk_modulus'])]
    for key in ('equilibrium_alpha','maxwell_alpha'):
        if key in material:
            cmd+=['--'+key.replace('_','-'),str(material[key])]
    if manifest.get('actual_affine_policy'):
        cmd+=['--actual-affine-policy']
    if manifest.get('shear_limit_mpa') is not None:
        cmd+=['--shear-limit-mpa',str(manifest['shear_limit_mpa'])]
    # The existing reference and screenshot are 60-degree evidence only.
    if job['angle']==60:
        if manifest.get('reference'):cmd+=['--reference',manifest['reference']]
    if job['angle']==manifest.get('image_angle',60) and manifest.get('image_estimates'):
        cmd+=['--image-estimates',manifest['image_estimates'],'--image-angle',str(job['angle'])]
    if job['angle']==manifest.get('experiment_angle',60) and manifest.get('experiment'):
        cmd+=['--experiment',manifest['experiment'],'--experiment-angle',str(job['angle'])]
    return cmd


def verify_inputs(manifest):
    for name,digest in manifest.get('input_sha256',{}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen comparison input changed: '+name)


def audit_reference(folder,out,material,mu,target):
    """Reuse a COMPLETE reference only after fresh equations and both histories pass."""
    from tools.run_material_candidate import make_problem,signature,audit
    from threadpoolctl import threadpool_limits
    from run_support import compute_process_policy
    folder=folder.resolve();old=read(folder/'result.json');angle=old['signature']['config']['inclination_deg']
    job=dict(name=f'reused_angle{angle:g}',angle=angle,mu=mu,target_mm=target,output=str(folder))
    status=summarize_case(folder,job,0)
    if not status['mechanics_completed']:raise ValueError('Reference must already be audited and complete at the requested target')
    destination=out/f'reused_angle{angle:g}_audit.json'
    with threadpool_limits(2),compute_process_policy():
        p=make_problem(angle=angle,mu=mu,**material)
        if json.loads(json.dumps(signature(p)))!=old['signature']:
            raise ValueError('Reference material, geometry, mesh or interface differs')
        fresh=audit(p,folder,old,destination=destination)
    previous=read(folder/'audit.json')
    inputs={str(folder/'result.json'):hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),
        str(destination):hashlib.sha256(destination.read_bytes()).hexdigest()}
    for i,row in enumerate(fresh['rows']):
        if row['checkpoint_sha256']!=previous['rows'][i]['checkpoint_sha256']:
            raise ValueError('Completed reference checkpoint changed')
        inputs[str(folder/'states'/f'{i:04d}.npz')]=row['checkpoint_sha256']
    return dict(job,**{k:v for k,v in status.items() if k not in job},reused=True,
        current_audit=str(destination),provenance='Existing complete result; freshly audited, no new loading solve'),inputs


def run_case(out,manifest,job,state,*,command_factory=None):
    verify_inputs(manifest)
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((Path(manifest['source'])/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen queue source changed: '+name)
    if manifest.get('experiment') and hashlib.sha256(Path(manifest['experiment']).read_bytes()).hexdigest()!=manifest['experiment_sha256']:
        raise RuntimeError('Frozen experimental comparison source changed')
    if Path(job['output']).exists():raise RuntimeError('Refuse to overwrite existing case: '+job['output'])
    env=os.environ.copy();env['PYTHONPATH']=manifest['runtime_path']
    env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
    cmd=(make_command if command_factory is None else command_factory)(manifest,job)
    job.update(status='starting',command=cmd,started_utc=now())
    write_json(out/'queue_status.json',state)
    started=time.monotonic();forced=None
    with (out/(job['name']+'.stdout.log')).open('wb') as stdout,\
         (out/(job['name']+'.stderr.log')).open('wb') as stderr:
        kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        child=subprocess.Popen(cmd,cwd=manifest['source'],env=env,stdin=subprocess.DEVNULL,
            stdout=stdout,stderr=stderr,**kwargs)
        job.update(status='running',solver_pid=child.pid);write_json(out/'queue_status.json',state)
        try:code=child.wait(timeout=job.get('solve_seconds',manifest['solve_seconds_per_case'])+manifest['audit_allowance_s'])
        except subprocess.TimeoutExpired:
            forced='process_timeout';child.terminate()
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=30)
    result=summarize_case(job['output'],job,code,forced)
    result.update(wall_elapsed_s=time.monotonic()-started,finished_utc=now())
    return result


def compare_mesh_case(job):
    """Same-angle accepted curves only; a small h-change does not exclude locking."""
    import numpy as np
    candidate=read(Path(job['output'])/'result.json')
    reference=read(Path(job['baseline'])/'result.json')
    if candidate['signature']['config']!=reference['signature']['config']:
        raise ValueError('Mesh comparison requires identical physical configuration')
    for key in ('material','loading_rate','contact_formulation','dhat','physical_force_tolerance_n'):
        if candidate['signature'][key]!=reference['signature'][key]:
            raise ValueError('Mesh comparison changed physics: '+key)
    cr=candidate['steps'];br=reference['steps']
    if not cr:return dict(status='no_accepted_states',full_target_available=False)
    depths=np.array([r['depth_mm'] for r in br]);forces=np.array([r['reaction_fz_n'] for r in br])
    anchors=[];rows=[]
    for r in cr:
        d=r['depth_mm']
        if d<depths[0] or d>depths[-1]+1e-12:continue
        bf=float(np.interp(d,depths,forces));j=int(np.argmin(abs(depths-d)))
        match=br[j] if abs(depths[j]-d)<1e-10 else None
        tip_delta=None
        if match and r['tip_contact']['status']==match['tip_contact']['status']=='unique':
            tip_delta=r['tip_contact']['reference_offset_mm']-match['tip_contact']['reference_offset_mm']
        row=dict(depth_mm=d,refined_fz_n=r['reaction_fz_n'],baseline_fz_n=bf,
            delta_fz_n=r['reaction_fz_n']-bf,relative_force_change=r['reaction_fz_n']/bf-1 if abs(bf)>1e-12 else None,
            refined_min_J=r['min_j'],baseline_min_J=match['min_j'] if match else None,
            refined_tip_status=r['tip_contact']['status'],baseline_tip_status=match['tip_contact']['status'] if match else None,
            reference_offset_difference_mm=tip_delta)
        rows.append(row)
        if any(abs(d-a)<1e-10 for a in (.01,.1,.25,.5)):anchors.append(row)
    full=bool(candidate.get('completed') and abs(cr[-1]['depth_mm']-job['target_mm'])<1e-10)
    answer=dict(status='complete_comparison' if full else 'accepted_prefix_only',full_target_available=full,
        baseline_sha256=hashlib.sha256((Path(job['baseline'])/'result.json').read_bytes()).hexdigest(),
        common_end_mm=rows[-1]['depth_mm'] if rows else None,anchors=anchors,rows=rows,
        force_difference_rms_n=float(np.sqrt(np.mean([r['delta_fz_n']**2 for r in rows]))) if rows else None,
        experimental_validation=False,
        scope='One local P1 interior refinement, identical contact trace. Small change does NOT exclude locking, '
              'boundary/remote mesh errors or establish spatial convergence. Missing pole identities remain missing.')
    write_json(Path(job['output'])/'mesh_comparison.json',answer)
    return {k:v for k,v in answer.items() if k!='rows'}


def export_summary(out,state):
    """Keep only observed accepted curves; no interpolation of missing contact."""
    lines=['# Finite-Maxwell angle queue','',
        'Each case retains its accepted prefix even on failure. No fracture is enabled.',
        'Parameters remain hypotheses; force agreement does not validate contact migration.','',
        '| Angle | Status | Accepted depth mm | Fz N | Minimum J | Fresh audit |',
        '|---:|---|---:|---:|---:|---|']
    curves=[]
    with (out/'accepted_curves.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['angle_deg','depth_mm','fz_n','min_j','tip_status','reference_offset_mm'])
        for job in state.get('reference_cases',[])+state['jobs']:
            lines.append(f"| {job['angle']} | {job['status']} | {job.get('last_depth_mm')} | {job.get('endpoint_fz_n')} | {job.get('min_j')} | {job.get('fresh_audit_passed')} |")
            path=Path(job['output'])/'result.json'
            if not path.is_file():continue
            r=read(path);rows=r.get('steps',[])
            if rows:curves.append((job,rows))
            for s in rows:
                tip=s.get('tip_contact',{})
                w.writerow([job['angle'],s['depth_mm'],s['reaction_fz_n'],s['min_j'],tip.get('status'),
                    tip.get('reference_offset_mm') if tip.get('status')=='unique' else None])
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not curves:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for (job,rows),color,style in zip(curves,['#0072B2','#D55E00','#009E73'],['-','--',':']):
            d=[s['depth_mm'] for s in rows]
            label=f"{job['angle']} deg ({'reused complete' if job.get('reused') else job['status']})"
            axes[0].plot(d,[s['reaction_fz_n'] for s in rows],color=color,ls=style,label=label)
            migration=[s['tip_contact'].get('reference_offset_mm') if s['tip_contact']['status']=='unique' else float('nan') for s in rows]
            axes[1].plot(d,migration,color=color,ls=style,label=label)
        axes[0].set(ylabel='Fz (N)');axes[1].set(ylabel='Reference tip-contact offset (mm)')
        for ax in axes:
            ax.set(xlabel='Downward depth (mm)',xlim=(0,state['target_mm']));ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'Intact Maxwell candidate, mu={state.get("mu",.35):g} | accepted data only; missing contact left as gaps')
        fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def worker(out):
    manifest=read(out/'manifest.json')
    # Exclusive marker prevents a second worker from running this manifest.
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),started_utc=now()),f)
    state=dict(status='running',worker_pid=os.getpid(),started_utc=now(),
        target_mm=manifest['target_mm'],mu=manifest.get('mu',.35),jobs=manifest['jobs'],
        reference_cases=manifest.get('reference_cases',[]))
    write_json(out/'queue_status.json',state)
    for job in state['jobs']:
        print(f"START angle={job['angle']} output={job['output']}",flush=True)
        try:result=run_case(out,manifest,job,state)
        except Exception as exc:
            result=dict(status='worker_case_failed',mechanics_completed=False,error=f'{type(exc).__name__}: {exc}',finished_utc=now())
            (out/(job['name']+'.traceback.log')).write_text(traceback.format_exc(),encoding='utf-8')
        job.update(result);write_json(out/(job['name']+'.outcome.json'),job)
        if job.get('diagnostic')=='native_mesh' and (Path(job['output'])/'result.json').exists():
            try:
                job['mesh_comparison']=compare_mesh_case(job)
                from tools.run_material_candidate import compare
                result_report=read(Path(job['output'])/'result.json')
                experiment=Path(manifest['experiment']) if job['angle']==manifest.get('experiment_angle') else None
                image=Path(manifest['image_estimates']) if job['angle']==manifest.get('image_angle') else None
                compare(Path(job['output']),result_report,image_estimates=image,experiment=experiment,
                        image_angle=manifest.get('image_angle',33),experiment_angle=manifest.get('experiment_angle',57))
            except Exception as exc:
                job['comparison_error']=f'{type(exc).__name__}: {exc}'
        write_json(out/'queue_status.json',state)
        try:export_summary(out,state)
        except Exception as exc:
            state.setdefault('postprocessing_errors',[]).append(f'{job["name"]}: {type(exc).__name__}: {exc}')
        print(f"END angle={job['angle']} status={job['status']} accepted_mm={job.get('last_depth_mm')}",flush=True)
    state.update(status='finished',finished_utc=now(),
        all_cases_completed=all(j.get('mechanics_completed') is True for j in state['jobs']))
    if manifest.get('diagnostic')=='shear_limited_interface':
        try:
            from tools.check_shear_limit import review_queue
            state['interface_review']=review_queue(out,manifest,state)
        except Exception as exc:
            state['interface_review_error']=f'{type(exc).__name__}: {exc}'
    write_json(out/'queue_status.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    start=parser.add_mutually_exclusive_group()
    start.add_argument('--resume60',type=Path)
    start.add_argument('--fresh',action='store_true',help='All selected angles start from rest; no history transfer')
    parser.add_argument('--depth',type=float,default=2.)
    parser.add_argument('--seconds',type=float,default=21600.)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--image-estimates',type=Path)
    parser.add_argument('--experiment',type=Path)
    parser.add_argument('--image-angle',type=float,default=60.)
    parser.add_argument('--experiment-angle',type=float,default=60.)
    parser.add_argument('--completed-reference',type=Path,action='append',default=[])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--launch-prepared',action='store_true')
    parser.add_argument('--angles',type=int,choices=(60,57,33,30,0),nargs='+',default=[60,30,0])
    parser.add_argument('--equilibrium-shear',type=float,default=.47)
    parser.add_argument('--bulk-modulus',type=float)
    parser.add_argument('--maxwell-shear',type=float,default=.94)
    parser.add_argument('--relaxation-s',type=float,default=.4)
    parser.add_argument('--loading-rate',type=float,default=.2)
    parser.add_argument('--mu',type=float,default=.35)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.worker:worker(out);return
    if args.launch_prepared:
        if (out/'worker_started.json').exists() or (out/'launch.json').exists():
            parser.error('Prepared queue has already been launched')
        manifest=read(out/'manifest.json');verify_inputs(manifest)
        launch(out,manifest);return
    if args.resume60 is None and not args.fresh:parser.error('Choose --resume60 or --fresh')
    if not all(math.isfinite(v) and v>0 for v in (args.depth,args.seconds,args.mu)):
        parser.error('positive finite depth, seconds and mu required')
    if (args.resume60 and args.angles[0]!=60) or len(set(args.angles))!=len(args.angles):
        parser.error('Resume 60 first; each selected angle occurs only once')
    old=None
    if args.resume60:
        old=read(args.resume60/'result.json');audit=read(args.resume60/'audit.json')
        if (not old.get('completed') or old['signature']['config']['inclination_deg']!=60.
            or not audit.get('passed') or audit.get('states')!=len(old['steps'])
            or args.depth<=old['last_depth_mm']):
            parser.error('Resume source must be an audited completed 60-degree prefix below the target')
    material={k:getattr(args,k) for k in
              ('equilibrium_shear','bulk_modulus','maxwell_shear','relaxation_s','loading_rate')}
    for key,value in material.items():
        if key=='bulk_modulus' and value is None:continue
        if not math.isfinite(value) or value<=0:parser.error('Positive finite material parameters required')
    # Catch accidental default-parameter resumes before launching a costly child.
    if old is not None:
        sig=old['signature'];cfg=sig['config'];mat=sig['material']
        expected_nu=cfg['poisson_ratio'] if args.bulk_modulus is None else (
            (3*args.bulk_modulus-2*args.equilibrium_shear)/(2*(3*args.bulk_modulus+args.equilibrium_shear)))
        if (cfg['shear_modulus_n_mm2']!=args.equilibrium_shear
                or cfg['poisson_ratio']!=expected_nu or mat['mu_ov']!=[args.maxwell_shear]
                or mat['relaxation_s']!=args.relaxation_s or sig['loading_rate']!=args.loading_rate
                or cfg['friction_coefficient']!=args.mu
                or (args.bulk_modulus is None and cfg['poisson_ratio']!=.49)):
            parser.error('Requested material/friction differs from the exact resume history')
    if args.experiment:
        from tools.run_material_candidate import load_force_csv
        _,_,rate=load_force_csv(args.experiment)
        if not math.isclose(rate,args.loading_rate,rel_tol=0,abs_tol=1e-12):
            parser.error('Experimental loading speed differs')
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];frozen=out/'source'
    hashes=source_hashes(packages=('prototype','mechanics'),snapshot=frozen)
    for relative in ('tools/run_material_candidate.py','tools/run_material_queue.py','prototype/ipc_baseline.json'):
        destination=frozen/relative;destination.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(root/relative,destination);hashes[relative]=hashlib.sha256(destination.read_bytes()).hexdigest()
    # Preflight the actual operator-source identity before detaching. Full state
    # reassembly still happens inside the first solver; no accepted history reset.
    from tools.run_material_candidate import resume_compatibility
    current={k:v for k,v in hashes.items() if k!='tools/run_material_queue.py'}
    if old is not None:
        resume_compatibility(old,dict(signature=old['signature'],source_sha256=current))
    jobs=[dict(name=f'{i:02d}_angle{angle}',angle=angle,mu=args.mu,target_mm=args.depth,
        output=str(out/f'{i:02d}_angle{angle}'),status='pending',
        resume=str(args.resume60.resolve()) if angle==60 and args.resume60 else None)
        for i,angle in enumerate(args.angles,1)]
    if args.experiment:shutil.copy2(args.experiment,out/'experiment.csv')
    if args.image_estimates:shutil.copy2(args.image_estimates,out/'image_estimates.csv')
    manifest=dict(created_utc=now(),python=sys.executable,source=str(frozen),source_sha256=hashes,
        runtime_path=str(root/'.runtime/ipc-python'),target_mm=args.depth,mu=args.mu,
        solve_seconds_per_case=args.seconds,audit_allowance_s=1800.,jobs=jobs,
        reference=str(args.reference.resolve()) if args.reference else None,
        image_estimates=str(out/'image_estimates.csv') if args.image_estimates else None,
        image_angle=args.image_angle,experiment_angle=args.experiment_angle,
        experiment=str(out/'experiment.csv') if args.experiment else None,
        experiment_sha256=hashlib.sha256((out/'experiment.csv').read_bytes()).hexdigest() if args.experiment else None,
        material=material,input_sha256={},reference_cases=[],
        policy='Selected cases only; resume only when explicitly requested; others start from rest; advance after terminal cases')
    for key in ('experiment','image_estimates'):
        if manifest.get(key):
            path=Path(manifest[key]);manifest['input_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for folder in args.completed_reference:
        reference,inputs=audit_reference(folder,out,material,args.mu,args.depth)
        if reference['angle'] in args.angles or any(j['angle']==reference['angle'] for j in manifest['reference_cases']):
            raise ValueError('A reused angle must not also run or appear twice')
        manifest['reference_cases'].append(reference);manifest['input_sha256'].update(inputs)
    write_json(out/'manifest.json',manifest)
    if args.prepare_only:
        print('PREPARED '+str(out));return
    launch(out,manifest)


def launch(out,manifest):
    """Launch the immutable prepared queue once, hidden on Windows."""
    verify_inputs(manifest)
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((Path(manifest['source'])/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen queue source changed: '+name)
    if (out/'worker_started.json').exists() or (out/'launch.json').exists():
        raise RuntimeError('Queue already launched')
    frozen=Path(manifest['source'])
    command=[sys.executable,'-X','utf8','-u','-m','tools.run_material_queue','--output',str(out),'--worker']
    env=os.environ.copy();env['PYTHONPATH']=manifest['runtime_path']
    env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
    startup=None;flags=0
    if os.name=='nt':
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=subprocess.SW_HIDE
        flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        process=subprocess.Popen(command,cwd=frozen,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            startupinfo=startup,creationflags=flags,start_new_session=os.name!='nt')
    launch=dict(worker_pid=process.pid,command=command,started_utc=now(),manifest=str(out/'manifest.json'))
    write_json(out/'launch.json',launch);print(json.dumps(launch,indent=2))


if __name__=='__main__':main()
