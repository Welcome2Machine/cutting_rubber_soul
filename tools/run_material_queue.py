"""Bounded 0/33/57-degree Maxwell cases, each from its own rest state.

One common material and performance policy; advance after each outcome.
Actual accepted histories, whole-path audits and HTML remain in the chosen output.
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
from prototype.material_performance import performance_policy
from tools.render_solver_dashboard import write_snapshot


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
        and actual_angle==job['angle'] and requested_mu_ok and records_ok and reached and audit_ok)
    status='completed' if completed else (forced_status or r.get('status','process_failed'))
    if not completed and status in ('completed','running','resuming','initializing'):status='incomplete_or_unverified'
    return dict(status='completed' if completed else status,mechanics_completed=bool(completed),
        angle=job['angle'],mu=actual_mu,requested_mu_matches=requested_mu_ok,
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
    material=manifest.get('material',{})
    cmd=[manifest['python'],'-X','utf8','-u','-m','tools.run_material_candidate',
        '--output',job['output'],'--angle',str(job['angle']),
        '--equilibrium-shear',str(material.get('equilibrium_shear',.14)),
        '--maxwell-shear',str(material.get('maxwell_shear',1.24)),
        '--relaxation-s',str(material.get('relaxation_s',1.7)),
        '--loading-rate',str(material.get('loading_rate',.2)),
        '--mu',str(manifest.get('mu',.35)),
        '--depth',str(job['target_mm']),'--increment',str(manifest.get('increment',.01)),
        '--seconds',str(job.get('solve_seconds',manifest['solve_seconds_per_case'])),'--threads',str(manifest.get('threads',2))]
    if material.get('bulk_modulus') is not None:
        cmd+=['--bulk-modulus',str(material['bulk_modulus'])]
    for key in ('equilibrium_alpha','maxwell_alpha'):
        if key in material:
            cmd+=['--'+key.replace('_','-'),str(material[key])]
    policy=manifest.get('performance_policy', performance_policy())
    cmd += ['--material-batch-size',str(policy['material_batch_size']),
            '--linear-backend',policy['linear_backend'],'--linear-threads',str(policy['linear_threads'] or 4)]
    # Explicit per-input angles prevent cross-angle comparison.
    if job['angle']==manifest.get('reference_angle',60):
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


def run_case(out,manifest,job,state,*,command_factory=None):
    verify_inputs(manifest)
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((Path(manifest['source'])/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Queue source changed since preparation: '+name)
    if manifest.get('experiment') and hashlib.sha256(Path(manifest['experiment']).read_bytes()).hexdigest()!=manifest['experiment_sha256']:
        raise RuntimeError('Frozen experimental comparison source changed')
    if Path(job['output']).exists():raise RuntimeError('Refuse to overwrite existing case: '+job['output'])
    env=os.environ.copy();env['PYTHONPATH']=manifest['runtime_path']
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
    env['SOFTMODELING_PARDISO_RUNTIME']=manifest['pardiso_runtime']
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
        try:
            deadline=time.monotonic()+job.get('solve_seconds',manifest['solve_seconds_per_case'])+manifest['audit_allowance_s']
            while child.poll() is None:
                if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(cmd,deadline-started)
                try:write_snapshot(out,refresh=True)
                except Exception as exc:state.setdefault('postprocessing_errors',[]).append(str(exc))
                try:child.wait(timeout=min(15,max(.01,deadline-time.monotonic())))
                except subprocess.TimeoutExpired:pass
            code=child.returncode
        except subprocess.TimeoutExpired:
            forced='process_timeout';child.terminate()
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=30)
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)
    result=summarize_case(job['output'],job,code,forced)
    if result['mechanics_completed']:
        try:
            from tools.qualify_intact_baseline import verify_case
            files=verify_case(Path(job['output']))
            r=read(Path(job['output'])/'result.json')
            if r.get('performance_policy')!=manifest['performance_policy']:
                raise ValueError('Recorded performance policy differs')
            if any(manifest['source_sha256'].get(k)!=v for k,v in r['source_sha256'].items()):
                raise ValueError('Recorded case sources differ from the prepared queue')
            if manifest['performance_policy']['material_batch_size'] or manifest['performance_policy']['linear_backend']!='native':
                independent=r.get('independent_audit_policy',{})
                if not (independent.get('fresh_problem') is True and independent.get('material_batch_size')==0
                        and independent.get('states')==len(r['steps'])):
                    raise ValueError('Missing independent native whole-path audit')
            result['verified_files']=len(files)
        except Exception as exc:
            result.update(status='evidence_failed',mechanics_completed=False,error=str(exc))
    result.update(wall_elapsed_s=time.monotonic()-started,finished_utc=now())
    return result


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
        write_json(out/'queue_status.json',state)
        try:export_summary(out,state)
        except Exception as exc:
            state.setdefault('postprocessing_errors',[]).append(f'{job["name"]}: {type(exc).__name__}: {exc}')
        print(f"END angle={job['angle']} status={job['status']} accepted_mm={job.get('last_depth_mm')}",flush=True)
    state.update(status='finished',finished_utc=now(),
        all_cases_completed=all(j.get('mechanics_completed') is True for j in state['jobs']))
    write_json(out/'queue_status.json',state)
    write_snapshot(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fresh',action='store_true',help='Compatibility spelling: every case starts from rest')
    parser.add_argument('--depth',type=float,default=2.)
    parser.add_argument('--increment',type=float,default=.01)
    parser.add_argument('--seconds',type=float,default=21600.)
    parser.add_argument('--angles',type=int,choices=(60,57,33,30,0),nargs='+',default=[0,33,57])
    parser.add_argument('--equilibrium-shear',type=float,default=.14)
    parser.add_argument('--bulk-modulus',type=float,default=23.34333333333331)
    parser.add_argument('--maxwell-shear',type=float,default=1.24)
    parser.add_argument('--relaxation-s',type=float,default=1.7)
    parser.add_argument('--loading-rate',type=float,default=.2)
    parser.add_argument('--mu',type=float,default=.35)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--material-batch-size',type=int,default=0)
    parser.add_argument('--linear-backend',choices=('native','pardiso'),default='native')
    parser.add_argument('--linear-threads',type=int,default=4)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--launch-prepared',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();out=args.output.resolve()
    if args.worker:worker(out);return
    if args.launch_prepared:
        if (out/'worker_started.json').exists() or (out/'launch.json').exists():
            parser.error('Prepared queue has already been launched')
        manifest=read(out/'manifest.json');verify_inputs(manifest);launch(out,manifest);return
    if not all(math.isfinite(v) and v>0 for v in (args.depth,args.increment,args.seconds,args.mu)) or args.depth>2:
        parser.error('Positive finite controls and target depth <= 2 mm required')
    if len(set(args.angles))!=len(args.angles):parser.error('Each angle occurs only once')
    material={k:getattr(args,k) for k in ('equilibrium_shear','bulk_modulus','maxwell_shear','relaxation_s','loading_rate')}
    if not all(math.isfinite(v) and v>0 for v in material.values()):
        parser.error('Positive finite material parameters required')
    try:
        policy=performance_policy(material_batch_size=args.material_batch_size,
            linear_backend=args.linear_backend,linear_threads=args.linear_threads,ipc_threads=args.threads)
    except ValueError as exc:parser.error(str(exc))
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    hashes=source_hashes(packages=('prototype','mechanics'))
    for relative in ('tools/run_material_candidate.py','tools/run_material_queue.py',
                     'tools/render_solver_dashboard.py','tools/qualify_intact_baseline.py','prototype/ipc_baseline.json'):
        hashes[relative]=hashlib.sha256((root/relative).read_bytes()).hexdigest()
    jobs=[dict(name=f'{i:02d}_angle{angle}',angle=angle,mu=args.mu,target_mm=args.depth,
        output=str(out/f'{i:02d}_angle{angle}'),status='pending',resume=None)
        for i,angle in enumerate(args.angles,1)]
    manifest=dict(created_utc=now(),python=sys.executable,source=str(root),source_sha256=hashes,
        runtime_path=str(root)+os.pathsep+str(root/'.runtime/ipc-python'),
        pardiso_runtime=str(root/'.runtime/pardiso'),target_mm=args.depth,increment=args.increment,
        mu=args.mu,threads=args.threads,performance_policy=policy,
        solve_seconds_per_case=args.seconds,audit_allowance_s=1800.,jobs=jobs,
        material=material,input_sha256={},policy='Independent rest histories; one common material; bounded serial cases')
    write_json(out/'manifest.json',manifest)
    if args.prepare_only:print('PREPARED '+str(out));return
    launch(out,manifest)


def launch(out,manifest):
    """Launch the prepared queue once, hidden on Windows; no source copies."""
    verify_inputs(manifest)
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((Path(manifest['source'])/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Queue source changed since preparation: '+name)
    if (out/'worker_started.json').exists() or (out/'launch.json').exists():
        raise RuntimeError('Queue already launched')
    frozen=Path(manifest['source'])
    command=[sys.executable,'-X','utf8','-u','-m','tools.run_material_queue','--output',str(out),'--worker']
    env=os.environ.copy();env['PYTHONPATH']=manifest['runtime_path']
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
    env['SOFTMODELING_PARDISO_RUNTIME']=manifest['pardiso_runtime']
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
