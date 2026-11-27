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
from tools import material_queue_control as control


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
        '--mu',str(job.get('mu',manifest.get('mu',.35))),
        '--depth',str(job['target_mm']),'--increment',str(job.get('increment',manifest.get('increment',.01))),
        '--seconds',str(job.get('solve_seconds',manifest['solve_seconds_per_case'])),'--threads',str(manifest.get('threads',2))]
    if material.get('bulk_modulus') is not None:
        cmd+=['--bulk-modulus',str(material['bulk_modulus'])]
    for key in ('equilibrium_alpha','maxwell_alpha'):
        if key in material:
            cmd+=['--'+key.replace('_','-'),str(material[key])]
    policy=manifest.get('performance_policy', performance_policy())
    if manifest.get('physical_core_affinity'):cmd.append('--physical-core-affinity')
    cmd += ['--material-batch-size',str(policy['material_batch_size']),
            '--linear-backend',policy['linear_backend'],'--linear-threads',str(policy['linear_threads'] or 4)]
    if manifest.get('diagnostics'):cmd+=['--diagnostics']
    if manifest.get('blade_spacing') is not None:cmd+=['--blade-spacing',str(manifest['blade_spacing'])]
    # Legacy 'resume' remains ignored. Only the explicit, prequalified entry
    # below is allowed; the candidate runner repeats its full resume audit.
    for key,value in job.get('options',{}).items():
        flag='--'+key.replace('_','-')
        if key=='enforce_physical_chord':
            if value:cmd.append(flag)
        else:cmd.extend([flag,str(value)])
    if job.get('resume_source'):cmd+=['--resume',job['resume_source']]
    # Explicit per-input angles prevent cross-angle comparison.
    if job['angle']==manifest.get('reference_angle',60):
        if manifest.get('reference'):cmd+=['--reference',manifest['reference']]
    if job['angle']==manifest.get('image_angle',60) and manifest.get('image_estimates'):
        cmd+=['--image-estimates',manifest['image_estimates'],'--image-angle',str(job['angle'])]
    experiment=manifest.get('experiments',{}).get(f'{job["angle"]:g}')
    if experiment:
        cmd+=['--experiment',experiment,'--experiment-angle',str(job['angle'])]
    elif job['angle']==manifest.get('experiment_angle',60) and manifest.get('experiment'):
        cmd+=['--experiment',manifest['experiment'],'--experiment-angle',str(job['angle'])]
    return cmd


def bind_short_paths(manifest,proof):
    """Authenticate one physically compatible short path per production case."""
    from tools.qualify_intact_baseline import verify_case
    from prototype.contact_trace import audit_coverage
    paths=read(proof).get('case_results',{})
    if set(paths)!={job['case_id'] for job in manifest['jobs']}:
        raise ValueError('Short evidence must name every production case exactly once')
    for job in manifest['jobs']:
        folder=(proof.parent/paths[job['case_id']]).resolve()
        r=read(folder/'result.json')
        actual=r.get('resolved_model',{})
        if actual!=job['resolved_model'] or r.get('performance_policy')!=manifest['performance_policy']:
            raise ValueError('Short evidence resolved model, solver policy or backend differs: '+job['name'])
        if any(manifest['source_sha256'].get(k)!=v for k,v in r['source_sha256'].items()) or not r['source_sha256']:
            raise ValueError('Short evidence source mismatch')
        expected={k:v for k,v in manifest['source_sha256'].items() if k.startswith(('prototype/','mechanics/')) or k in ('run_support.py','tools/run_material_candidate.py','tools/material_queue_control.py','numerical_runtime.py')}
        if expected!=r['source_sha256']:raise ValueError('Incomplete numerical source identities')
        if any(r['settings'].get(k)!=v for k,v in dict(job['options'],increment=job['increment']).items()):
            raise ValueError('Short evidence loading controls differ')
        steps=r['steps'];prefix,_=audit_coverage(folder,r,None,None)
        if not steps or prefix!=len(steps) or steps[-1]['depth_mm']+1e-10<job['required_short_depth_mm']:
            raise ValueError('Insufficient independently audited short depth')
        if not all(s.get('sliding_amount',{}).get('status') and s.get('contact_accounting') and
                   max(map(abs,s['contact_accounting']['force_closure_fx_fz_n']))<1e-6 for s in steps):
            raise ValueError('Missing or inconsistent short-path observations')
        if not any(s['sliding_amount']['native_supported'] for s in steps):
            raise ValueError('No loaded nose evidence on this path')
        manifest['input_sha256'].update(verify_case(folder))
        job['short_binding']=dict(source=str(folder),states=len(steps),depth_mm=steps[-1]['depth_mm'])
    manifest['short_evidence_verified']=True


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
    env['SOFTMODELING_QUEUE_PAUSE']=str(out/'pause_requested.json')
    cmd=(make_command if command_factory is None else command_factory)(manifest,job)
    job.update(status='starting',command=cmd,started_utc=now())
    write_json(out/'queue_status.json',state)
    started=time.monotonic();forced=None
    if (out/'pause_requested.json').exists():return dict(status='paused',audit_pending=True)
    with (out/(job['name']+'.stdout.log')).open('xb') as stdout,\
         (out/(job['name']+'.stderr.log')).open('wb') as stderr:
        kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        child=subprocess.Popen(cmd,cwd=manifest['source'],env=env,stdin=subprocess.DEVNULL,
            stdout=stdout,stderr=stderr,**kwargs)
        job.update(status='running',solver_pid=child.pid,process_identity=control.identity(child.pid));write_json(out/'queue_status.json',state)
        try:import psutil
        except ImportError:psutil=None
        try:
            deadline=time.monotonic()+job.get('remaining_wall_s',job.get('solve_seconds',manifest['solve_seconds_per_case'])+manifest['audit_allowance_s'])
            while child.poll() is None:
                if 'case_id' in job and psutil is not None:
                    try:
                        usage=psutil.Process(child.pid).memory_info()
                        job['peak_solver_memory_bytes']=max(job.get('peak_solver_memory_bytes',0),getattr(usage,'peak_wset',usage.rss))
                    except psutil.Error:pass
                state['heartbeat_utc']=now();write_json(out/'queue_status.json',state)
                if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(cmd,deadline-started)
                try:
                    if 'case_id' not in job:write_snapshot(out,refresh=True)
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
    if (out/'pause_requested.json').is_file():
        rfile=Path(job['output'])/'result.json'
        r=read(rfile) if rfile.is_file() else {}
        return dict(status='paused',audit_pending=True,last_depth_mm=r.get('last_depth_mm'),
            active_wall_s=time.monotonic()-started,active_solve_s=r.get('active_solve_s',r.get('elapsed_s',time.monotonic()-started)))
    result=summarize_case(job['output'],job,code,forced)
    if result['mechanics_completed']:
        try:
            from tools.qualify_intact_baseline import verify_case
            files=verify_case(Path(job['output']))
            r=read(Path(job['output'])/'result.json')
            if 'case_id' in job:
                expected=job['resolved_model'];actual=r['resolved_model']
                for key in ('config','material','friction','initial_depth_mm','loading_rate_mm_s','damage_off_semantics'):
                    if expected[key]!=actual[key]:raise ValueError('Different resolved case '+key)
                for key in ('blade_reference_sha256','body_reference_sha256','triangles_sha256','landmarks'):
                    if expected['geometry'][key]!=actual['geometry'][key]:raise ValueError('Different actual geometry '+key)
                for key,value in dict(job['options'],increment=job['increment'],depth=job['target_mm']).items():
                    if r['settings'].get(key)!=value:raise ValueError('Different recorded setting '+key)
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
    if 'case_id' in job:
        folder=Path(job['output'])
        if (folder/'audit.json').is_file() and (folder/'result.json').is_file():
            audited=read(folder/'audit.json')
            prefix=audited.get('qualified_prefix_states',0)
            if prefix:
                from prototype.contact_trace import export_trace
                from workbench.results import export_case
                try:
                    exports=out/'exports'/job['case_id']
                    trace=export_trace(folder,exports/'contact_trace',qualified_prefix=True,
                        checkpoint=lambda:control.checkpoint(out/'pause_requested.json'))
                    export_case(folder,exports/'presentation',trace_path=exports/'contact_trace/contact_trace.json')
                    eligible=[s for s in trace['steps'] if s['qualified']]
                    valid=[s for s in eligible if s['tip_contact']['status']=='unique']
                    final=eligible[-1]
                    halfwidth=job['options']['fine_halfwidth']
                    intervals=[pair for step in eligible for pair in
                        step['blade_contact_footprint']['reference_offset_intervals_mm']]
                    extent=max((abs(v) for pair in intervals for v in pair),default=None)
                    result.update(contact_address_abs_max_mm=extent,
                        fine_halfwidth_mm=halfwidth,footprint_leaves_fine_band=extent>halfwidth if extent is not None else None,
                        min_j=min(step['deformation']['min_j'] for step in eligible),
                        relative_motion_qualification=trace['relative_motion_definition']['qualification'])
                    if result['mechanics_completed'] and len(eligible)!=len(read(folder/'result.json')['steps']):
                        raise ValueError('Final authenticated trace did not cover the full solver path')
                    result.update(qualified_depth_mm=final['depth_mm'],qualified_states=len(eligible),
                        sa_endpoint=final['sliding_amount'],
                        sa_unique_states=sum(s['sliding_amount']['status']=='native_nose_contact' for s in eligible),
                        sa_native_support_states=sum(bool(s['sliding_amount']['native_supported']) for s in eligible),
                        pole_valid_states=len(valid),pole_valid_fraction=len(valid)/len(eligible),
                        pole_last_mm=valid[-1]['tip_contact']['reference_offset_mm'] if valid else None,
                        pole_last_depth_mm=valid[-1]['depth_mm'] if valid else None,
                        pole_endpoint_status=final['tip_contact']['status'],
                        result_sha256=trace['source_result_sha256'],
                        audit_sha256=hashlib.sha256((folder/'audit.json').read_bytes()).hexdigest(),
                        presentation=str(exports/'presentation'), trace=str(exports/'contact_trace'),
                        fz_2mm_n=final['force']['reaction_fz_n'] if result['mechanics_completed'] and job['target_mm']==2. else None,
                        fx_2mm_n=final['force']['reaction_x_n'] if result['mechanics_completed'] and job['target_mm']==2. else None)
                    if trace.get('visualization',{}).get('status')!='completed':
                        raise ValueError(str(trace.get('visualization')))
                except control.QueuePaused:
                    return dict(status='paused',audit_pending=False,export_pending=True,
                        active_solve_s=r.get('elapsed_s',0.) if result['mechanics_completed'] else result.get('solver_elapsed_s',0.),
                        active_wall_s=time.monotonic()-started,last_depth_mm=result.get('last_depth_mm'))
                except Exception as exc:
                    result.update(status='export_failed',export_error=str(exc),full_delivery=False)
            else:result.update(qualified_states=0,qualified_depth_mm=None)
    result.update(wall_elapsed_s=time.monotonic()-started,finished_utc=now())
    return result


def export_summary(out,state):
    """Keep only observed accepted curves; no interpolation of missing contact."""
    if any('case_id' in j for j in state['jobs']):
        fields=['case_id','angle','mu','status','accepted_states','qualified_states','qualified_depth_mm',
                'sa_unique_states','sa_native_support_states','mechanics_completed','fresh_audit_passed','fz_2mm_n','fx_2mm_n','pole_last_mm',
                'pole_last_depth_mm','pole_valid_fraction','pole_endpoint_status','wall_elapsed_s',
                'error','export_error','output','result_sha256','audit_sha256','peak_solver_memory_bytes','min_j','fine_halfwidth_mm','contact_address_abs_max_mm','footprint_leaves_fine_band']
        with (out/'summary.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields+['blade_shape','tip_radius','accuracy'])
            writer.writeheader()
            for job in state['jobs']:
                row={k:job.get(k) for k in fields}
                writer.writerow(dict(row,blade_shape=job['options']['blade_shape'],
                    tip_radius=job['options']['tip_radius'],accuracy='short path screening only'))
        return
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
        json.dump(dict(pid=os.getpid(),started_utc=now(),identity=control.identity(os.getpid())),f)
    state=dict(status='running',worker_pid=os.getpid(),started_utc=now(),
        target_mm=manifest['target_mm'],mu=manifest.get('mu',.35),jobs=manifest['jobs'],
        reference_cases=manifest.get('reference_cases',[]))
    write_json(out/'queue_status.json',state)
    for job in state['jobs']:
        if (out/'pause_requested.json').is_file():
            state.update(status='paused',audit_pending=True);write_json(out/'queue_status.json',state);return
        if job.get('reused'):continue
        print(f"START angle={job['angle']} output={job['output']}",flush=True)
        if 'case_id' in job:
            try:
                verify_inputs(manifest)
                for name,digest in manifest['source_sha256'].items():
                    if hashlib.sha256((Path(manifest['source'])/name).read_bytes()).hexdigest()!=digest:
                        raise RuntimeError('Source changed: '+name)
                if shutil.disk_usage(out).free<manifest.get('minimum_free_bytes',0):
                    raise OSError('Insufficient free disk space for next case')
            except (ValueError,OSError,RuntimeError) as exc:
                state.update(status='system_error',error=str(exc),finished_utc=now())
                write_json(out/'queue_status.json',state);export_summary(out,state);return
        try:result=run_case(out,manifest,job,state)
        except Exception as exc:
            result=dict(status='worker_case_failed',mechanics_completed=False,error=f'{type(exc).__name__}: {exc}',finished_utc=now())
            (out/(job['name']+'.traceback.log')).write_text(traceback.format_exc(),encoding='utf-8')
        if result.get('status')=='paused' or (out/'pause_requested.json').exists():
            job.update(result)
            state.update(status='paused',audit_pending=True)
            write_json(out/'queue_status.json',state);export_summary(out,state);return
        if 'case_id' in job:
            problem=' '.join(str(result.get(k,'')) for k in ('error','export_error'))
            stderr=out/(job['name']+'.stderr.log')
            if stderr.exists():problem+=' '+stderr.read_bytes()[-4096:].decode('utf-8','replace')
            system_error=(result.get('status')=='worker_case_failed' or not (Path(job['output'])/'result.json').exists()
                or any(s in problem for s in ('No space left','not enough space','MemoryError','DLL load failed',
                                             'ModuleNotFoundError','PermissionError','WinError 112')))
            if system_error:
                job.update(result);state.update(status='system_error',error=problem,finished_utc=now())
                write_json(out/'queue_status.json',state);export_summary(out,state);return
        job.update(result);write_json(out/(job['name']+'.outcome.json'),job)
        write_json(out/'queue_status.json',state)
        try:export_summary(out,state)
        except Exception as exc:
            state.setdefault('postprocessing_errors',[]).append(f'{job["name"]}: {type(exc).__name__}: {exc}')
        print(f"END angle={job['angle']} status={job['status']} accepted_mm={job.get('last_depth_mm')}",flush=True)
    # Compare only actual, freshly audited curves after all case outcomes.
    command=[manifest.get('python',sys.executable),'-X','utf8','-m','tools.plot_experiment_angles',
             '--output',str(out/'experiment_comparison'),'--max-depth',str(manifest['target_mm'])]
    for job in state['jobs']:
        experiment=manifest.get('experiments',{}).get(f'{job["angle"]:g}')
        if experiment and job.get('fresh_audit_passed') and job.get('accepted_states',0)>1:
            command+=['--case',str(job['angle']),job['output'],experiment]
    if manifest.get('trials'):
        completed={j['case_id']:j for j in state['jobs'] if j.get('fresh_audit_passed') and j.get('accepted_states',0)>1}
        trials=[dict(t,folder=completed[t['case_id']]['output']) for t in manifest['trials'] if t['case_id'] in completed]
        if trials:
            path=out/'comparison_trials.json';write_json(path,trials);command+=['--trial-map',str(path)]
    if '--case' in command or '--trial-map' in command:
        try:
            with (out/'experiment_comparison.log').open('wb') as log:
                subprocess.run(command,cwd=manifest['source'],stdout=log,stderr=subprocess.STDOUT,
                    check=True,timeout=120,**({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}))
            state['experiment_comparison']='experiment_comparison/index.html'
        except Exception as exc:
            state.setdefault('postprocessing_errors',[]).append(f'joint comparison: {type(exc).__name__}: {exc}')
    state.update(status='finished',finished_utc=now(),
        all_cases_completed=all(j.get('mechanics_completed') is True and j.get('status')=='completed' for j in state['jobs']))
    write_json(out/'queue_status.json',state)
    if not any('case_id' in j for j in state['jobs']):write_snapshot(out)


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
    parser.add_argument('--physical-core-affinity',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--blade-spacing',type=float)
    parser.add_argument('--trial-map',type=Path,help='Explicit trial_id/case_id, experimental conditions, units and comparison windows')
    parser.add_argument('--experiment-case',nargs=2,action='append',metavar=('ANGLE','CSV'),
        help='Explicit original force CSV per angle; compare actual audited prefixes after the queue')
    parser.add_argument('--resume-case',nargs=2,action='append',metavar=('ANGLE','DIRECTORY'),
        help='Explicit own-source audited prefix; identical signature, performance and physical increment required')
    parser.add_argument('--material-batch-size',type=int,default=0)
    parser.add_argument('--linear-backend',choices=('native','pardiso'),default='native')
    parser.add_argument('--linear-threads',type=int,default=4)
    parser.add_argument('--short-evidence',type=Path,help='Completed short-path screening evidence to bind in the manifest')
    parser.add_argument('--case-config',type=Path,help='Explicit ordered case identities, options and budgets')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--launch-prepared',action='store_true')
    parser.add_argument('--foreground',action='store_true',
        help='Keep this process attached until the queue ends; required inside Slurm allocations')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    actions=parser.add_mutually_exclusive_group()
    for flag in ('status','pause','resume-queue'):
        actions.add_argument('--'+flag,action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.foreground and (args.prepare_only or args.status or args.pause or args.worker):
        parser.error('--foreground applies only to a new, prepared or resumed queue launch')
    if args.status or args.pause or args.resume_queue:
        if args.resume_queue:
            started=[]
            def start_resumed(folder,manifest):
                started.append(launch(folder,manifest,foreground=args.foreground))
            value=control.restart(out,start_resumed)
            print(json.dumps(value,indent=2))
            # Wait only after restart releases its control lock and publishes
            # active_attempt.json, so status and pause remain usable.
            if args.foreground:return wait_for_worker(Path(value['directory']),started[0])
        else:
            value=control.status(out) if args.status else control.pause(out)
            print(json.dumps(value,indent=2))
        return
    if args.worker:worker(out);return
    if args.launch_prepared:
        if (out/'worker_started.json').exists() or (out/'launch.json').exists():
            parser.error('Prepared queue has already been launched')
        manifest=read(out/'manifest.json');verify_inputs(manifest)
        process=launch(out,manifest,foreground=args.foreground)
        return wait_for_worker(out,process) if args.foreground else None
    if not all(math.isfinite(v) and v>0 for v in (args.depth,args.increment,args.seconds,args.mu)) or args.depth>2:
        parser.error('Positive finite controls and target depth <= 2 mm required')
    if len(set(args.angles))!=len(args.angles):parser.error('Each angle occurs only once')
    if args.blade_spacing is not None and (not math.isfinite(args.blade_spacing) or not 0<args.blade_spacing<=.05):
        parser.error('Blade spacing must be in (0, .05] mm')
    material={k:getattr(args,k) for k in ('equilibrium_shear','bulk_modulus','maxwell_shear','relaxation_s','loading_rate')}
    if not all(math.isfinite(v) and v>0 for v in material.values()):
        parser.error('Positive finite material parameters required')
    try:
        policy=performance_policy(material_batch_size=args.material_batch_size,
            linear_backend=args.linear_backend,linear_threads=args.linear_threads,ipc_threads=args.threads)
    except ValueError as exc:parser.error(str(exc))
    experiments={};inputs={}
    from prototype.material_case import load_force_csv
    for angle,path in args.experiment_case or []:
        try:angle=int(angle)
        except ValueError:parser.error('Experiment angle must be an integer case angle')
        key=str(angle)
        if angle not in args.angles or key in experiments:parser.error('Unique experiment angle must occur in the queue')
        path=Path(path).resolve()
        _,_,speed=load_force_csv(path)
        if abs(speed-args.loading_rate)>1e-12:parser.error('Experimental loading rate differs')
        experiments[key]=str(path);inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    resumes={}
    if args.resume_case and not args.case_config:
        from prototype.material_case import make_problem,signature,resume_compatibility
        from prototype.material_performance import require_same_performance
        from tools.qualify_intact_baseline import verify_case
        physical_sources=source_hashes(packages=('prototype','mechanics'))
        repo=Path(__file__).resolve().parents[1]
        for relative in ('tools/run_material_candidate.py','tools/material_queue_control.py','numerical_runtime.py','prototype/ipc_baseline.json'):
            physical_sources[relative]=hashlib.sha256((repo/relative).read_bytes()).hexdigest()
        for angle,path in args.resume_case:
            try:
                angle=int(angle);key=str(angle);path=Path(path).resolve()
                if angle not in args.angles or key in resumes:raise ValueError('Unique resume angle must occur in queue')
                old=read(path/'result.json');require_same_performance(old,policy)
                if old['settings']['increment']!=args.increment:raise ValueError('Resume physical increment differs')
                if not old['steps'] or old['steps'][-1]['depth_mm']>=args.depth-1e-12:
                    raise ValueError('Resume needs an accepted prefix shorter than the requested target')
                problem=make_problem(angle=angle,mu=args.mu,blade_spacing=args.blade_spacing,**material)
                expected=dict(signature=json.loads(json.dumps(signature(problem))),source_sha256=physical_sources,
                    solver_policy=dict(version=1,predictor='secant',affine_stagnation_window=0))
                resume_compatibility(old,expected)
                inputs.update(verify_case(path));resumes[key]=str(path)
            except (ValueError,RuntimeError,OSError,KeyError) as exc:parser.error(str(exc))
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    hashes=source_hashes(packages=('prototype','mechanics'))
    for relative in ('tools/run_material_candidate.py','tools/run_material_queue.py',
                     'tools/render_solver_dashboard.py','tools/qualify_intact_baseline.py',
                     'tools/plot_experiment_angles.py','tools/material_queue_control.py','numerical_runtime.py','prototype/ipc_baseline.json'):
        hashes[relative]=hashlib.sha256((root/relative).read_bytes()).hexdigest()
    jobs=[dict(name=f'{i:02d}_angle{angle}',angle=angle,mu=args.mu,target_mm=args.depth,
        output=str(out/f'{i:02d}_angle{angle}'),status='pending',resume=None)
        for i,angle in enumerate(args.angles,1)]
    if args.case_config:
        import re
        path=args.case_config.resolve()
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        settings=read(path);jobs=[]
        allowed={'blade_shape','tip_radius','blade_thickness','blade_angle','dhat','nose_facets',
                 'local_spacing','surface_spacing','fine_halfwidth','coarse_spacing','contact_layer_depth','reference_layout','size_gradient','drop_s',
                 'preparation_clearance_mm','friction_velocity_mm_s','enforce_physical_chord',
                 'minimum_increment_mm','blade_spacing','friction_globalization','predictor','max_iterations'}
        for case in settings['cases']:
            name=case['case_id']
            if not re.fullmatch(r'[A-Za-z0-9_]+',name) or any(j['name'].lower()==name.lower() for j in jobs):
                raise ValueError('Case ID must be path-safe and unique')
            opts=dict(settings['common'])
            opts.update(case.get('options',{}))
            profile=case.get('profile')
            if profile:
                catalog_path=root.parents[1]/'configs/blade_profiles.measured.json'
                catalog=read(catalog_path)
                matches=[p for p in catalog['profiles'] if p['identity']['id']==profile]
                if len(matches)!=1 or not matches[0]['qualification']['side_view_confirmed']:
                    raise ValueError('Unknown or unqualified macro profile: '+str(profile))
                values=matches[0]['apparent_profile']
                for option,field in [('blade_thickness','thickness_mm'),('blade_angle','included_angle_deg')]:
                    if option in opts and not math.isclose(opts[option],values[field],abs_tol=1e-12,rel_tol=0):
                        raise ValueError('Explicit macro input disagrees with measured catalog: '+option)
                    opts[option]=values[field]
                inputs[str(catalog_path)]=hashlib.sha256(catalog_path.read_bytes()).hexdigest()
            if set(opts)-allowed or opts.get('enforce_physical_chord') is not True:
                raise ValueError('Unknown case setting or missing full-chord acceptance')
            if not (isinstance(case['seconds'],(int,float)) and math.isfinite(case['seconds']) and case['seconds']>0
                    and 0<=case['angle']<=75 and math.isfinite(case['mu']) and case['mu']>=0):
                raise ValueError('Invalid case loading controls: '+name)
            required_depth=case.get('required_short_depth_mm',.6 if opts.get('tip_radius',.02)<=.002 else .1)
            if (isinstance(required_depth,bool) or not isinstance(required_depth,(int,float)) or
                    not math.isfinite(required_depth) or not 0<required_depth<=2.):
                raise ValueError('Invalid required evidence depth for '+name)
            if opts.get('blade_shape') not in ('double','single_up','single_down'):
                raise ValueError('Unknown blade shape: '+name)
            if opts.get('friction_globalization','model-error') not in ('model-error','residual-merit') or opts.get('predictor','secant') not in ('secant','affine'):
                raise ValueError('Unknown solver policy: '+name)
            if type(opts.get('max_iterations',500)) is not int or opts.get('max_iterations',500)<1:
                raise ValueError('Invalid iteration limit: '+name)
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                   or v<0 or (v==0 and k!='contact_layer_depth')
                   for k,v in opts.items() if k not in ('blade_shape','enforce_physical_chord','predictor','friction_globalization','reference_layout')):
                raise ValueError('Invalid numerical setting: '+name)
            if 'reference_layout' in opts:
                layout_path=(path.parent/opts['reference_layout']).resolve()
                inputs[str(layout_path)]=hashlib.sha256(layout_path.read_bytes()).hexdigest()
                opts['reference_layout']=str(layout_path)
            jobs.append(dict(case_id=name,name=name,profile=profile,angle=case['angle'],mu=case['mu'],options=opts,
                increment=case.get('increment',args.increment),solve_seconds=case['seconds'],
                required_short_depth_mm=required_depth,
                output=str(out/name),target_mm=args.depth,status='queued'))
        if not jobs:raise ValueError('Case list is empty')
        for relative in ('run_support.py','workbench/results.py','workbench/config.py','tools/material_queue_control.py'):
            hashes[relative]=hashlib.sha256((root/relative).read_bytes()).hexdigest()
    case_resumes={}
    if args.case_config:
        for key,path in args.resume_case or []:
            if key in case_resumes or key not in {j['case_id'] for j in jobs}:
                parser.error('Explicit queues require a unique case_id for resume')
            case_resumes[key]=Path(path).resolve()
    for job in jobs:
        if str(job['angle']) in resumes:job['resume_source']=resumes[str(job['angle'])]
    manifest=dict(created_utc=now(),python=sys.executable,source=str(root),source_sha256=hashes,
        runtime_path=str(root)+os.pathsep+str(Path(os.environ.get('SOFTMODELING_IPC_RUNTIME',root/'.runtime/ipc-python'))),
        minimum_free_bytes=8*1024**3,pardiso_runtime=str(Path(os.environ.get('SOFTMODELING_PARDISO_RUNTIME',root/'.runtime/pardiso'))),target_mm=args.depth,increment=args.increment,
        mu=args.mu,threads=args.threads,performance_policy=policy,diagnostics=args.diagnostics,
        physical_core_affinity=args.physical_core_affinity,
        blade_spacing=args.blade_spacing,
        solve_seconds_per_case=args.seconds,audit_allowance_s=1800.,jobs=jobs,
        material=material,input_sha256=inputs,experiments=experiments,
        policy='Independent rest histories unless explicit own-source prefix passed all resume guards; one common material; bounded serial cases')
    if args.trial_map:
        if not args.case_config or experiments:parser.error('--trial-map requires explicit cases and exclusive experiment mapping')
        path=args.trial_map.resolve()
        submission=read(path); ids=set(); by_id={j['case_id']:j for j in jobs}
        for trial in submission:
            identity=(trial['case_id'],trial['trial_id'])
            if identity in ids or identity[0] not in by_id:parser.error('Duplicate or unknown experimental condition')
            ids.add(identity);job=by_id[identity[0]]
            if (trial['angle']!=job['angle'] or trial['shape']!=job['options']['blade_shape'] or
                    (job.get('profile') and trial['profile']!=job['profile']) or trial['radius_mm']!=job['options']['tip_radius'] or trial['lubrication']!='no_added_lubricant'):
                parser.error('Trial and job conditions differ')
            source=(path.parent/trial['csv']).resolve();rate=load_force_csv(source)[2]
            if abs(rate-material['loading_rate'])>1e-12 or abs(rate-trial['speed_mm_s'])>1e-12:
                parser.error('Trial speed differs from FE or CSV')
            trial.update(csv=str(source),folder=job['output']);inputs[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['trials']=submission
    if args.case_config and experiments and len({j['angle'] for j in jobs})<len(jobs):
        parser.error('Angle-only experiment input is ambiguous here; use --trial-map')
    if args.short_evidence:
        proof=args.short_evidence.resolve()
        manifest['short_evidence']=str(proof)
        manifest['input_sha256'][str(proof)]=hashlib.sha256(proof.read_bytes()).hexdigest()
    if args.case_config:
        manifest.update(batch_id=out.name,case_config=str(args.case_config.resolve()),
            free_disk_bytes_at_prepare=shutil.disk_usage(out).free,logical_cpus=os.cpu_count(),
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            case_state_path=str(out/'queue_status.json'),summary_path=str(out/'summary.csv'))
        manifest['audit_allowance_s']=float(settings.get('audit_allowance_s',7200))
        manifest['minimum_free_bytes']=int(settings.get('minimum_free_bytes',8*1024**3))
        if not math.isfinite(manifest['audit_allowance_s']) or manifest['audit_allowance_s']<=0 or manifest['minimum_free_bytes']<1:
            raise ValueError('Invalid audit or disk budget')
        from prototype.material_case import make_problem, resolved_model
        for job in jobs:
            options={k:v for k,v in job['options'].items() if k!='minimum_increment_mm'}
            host=make_problem(angle=job['angle'],mu=job['mu'],**material,**options)
            job['resolved_model']=json.loads(json.dumps(resolved_model(host)))
            job['reference_contact_check']=host.reference_contact_check
            if job['name'] in case_resumes:
                from tools.qualify_intact_baseline import verify_case
                from prototype.material_case import signature,resume_compatibility
                from prototype.material_performance import require_same_performance
                directory=case_resumes[job['name']];old=read(directory/'result.json')
                require_same_performance(old,policy)
                if old.get('resolved_model')!=job['resolved_model'] or old['settings']['increment']!=job['increment']:
                    raise ValueError('Resume-case geometry, material, solver policy or increment differs')
                expected_sources={k:v for k,v in hashes.items() if k.startswith(('prototype/','mechanics/')) or k in ('run_support.py','tools/run_material_candidate.py','tools/material_queue_control.py','numerical_runtime.py')}
                resume_compatibility(old,dict(signature=json.loads(json.dumps(signature(host))),
                    solver_policy=old['solver_policy'],source_sha256=expected_sources))
                manifest['input_sha256'].update(verify_case(directory))
                job['resume_source']=str(directory)
            import numpy as np
            points=host.mesh.reference_xy[host.mesh.triangles]
            edges=np.roll(points,-1,axis=1)-points
            lengths=np.linalg.norm(edges,axis=2)
            cosines=np.sum(edges*np.roll(-edges,1,axis=1),axis=2)/(lengths*np.roll(lengths,1,axis=1))
            job['mesh_quality']=dict(minimum_angle_deg=float(np.degrees(np.arccos(np.clip(cosines,-1,1))).min()),
                h_over_R=options['surface_spacing']/options['tip_radius'],
                dhat_over_R=options['dhat']/options['tip_radius'],
                nose_sagitta_mm=float(np.max(options['tip_radius']-np.sqrt(np.maximum(0.,
                    options['tip_radius']**2-.25*np.linalg.norm(np.diff(host.rest[host.n+2:host.n+3+options['nose_facets']],axis=0),axis=1)**2)))))

            job['command']=make_command(manifest,job)
            del host
    if args.short_evidence and args.case_config:bind_short_paths(manifest,args.short_evidence.resolve())
    write_json(out/'manifest.json',manifest)
    if args.prepare_only:print('PREPARED '+str(out));return
    process=launch(out,manifest,foreground=args.foreground)
    return wait_for_worker(out,process) if args.foreground else None


def launch(out,manifest,*,foreground=False):
    """Launch once; foreground workers stay in the scheduler's process session."""
    if os.environ.get('SLURM_JOB_ID') and not foreground:
        raise RuntimeError('Slurm queue launches require --foreground; detached workers outlive their job step')
    if manifest.get('case_config') and manifest['target_mm']==2. and not manifest.get('short_evidence_verified'):
        raise ValueError('Production needs case-bound audited short evidence')
    verify_inputs(manifest)
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((Path(manifest['source'])/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Queue source changed since preparation: '+name)
    if (out/'worker_started.json').exists() or (out/'launch.json').exists():
        raise RuntimeError('Queue already launched')
    frozen=Path(manifest['source'])
    if os.path.abspath(sys.executable)!=os.path.abspath(manifest['python']):
        raise RuntimeError('Launch using the prepared Python interpreter')
    command=[sys.executable,'-X','utf8','-u','-m','tools.run_material_queue','--output',str(out),'--worker']
    env=os.environ.copy();env['PYTHONPATH']=manifest['runtime_path']
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
    env['SOFTMODELING_PARDISO_RUNTIME']=manifest['pardiso_runtime']
    startup=None;flags=0
    if os.name=='nt' and not foreground:
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=subprocess.SW_HIDE
        flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
    with (out/'launch_claim.json').open('x',encoding='utf8') as claim:
        json.dump(dict(identity=control.identity(os.getpid()),directory=str(out),time=now()),claim)
    with (out/'worker.stdout.log').open('xb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        process=subprocess.Popen(command,cwd=frozen,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            startupinfo=startup,creationflags=flags,start_new_session=os.name!='nt' and not foreground)
    launch=dict(worker_pid=process.pid,identity=control.identity(process.pid),command=command,
                foreground=foreground,started_utc=now(),manifest=str(out/'manifest.json'))
    write_json(out/'launch.json',launch);print(json.dumps(launch,indent=2))
    return process


def wait_for_worker(out,process):
    """Expose worker failures and incomplete mechanics to a batch scheduler."""
    code=process.wait()
    receipt=read(out/'launch.json')
    write_json(out/'worker_exit.json',dict(identity=receipt['identity'],directory=str(out),
               returncode=code,ended_utc=now()))
    if code:return code if code>0 else 128-code
    try:state=read(out/'queue_status.json')
    except (OSError,ValueError):return 1
    if state.get('status')=='paused':return 2
    return 0 if state.get('status')=='finished' and state.get('all_cases_completed') is True else 1


if __name__=='__main__':raise SystemExit(main())
