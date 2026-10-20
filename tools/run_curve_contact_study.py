"""Frozen analytic-blade path qualification, separate from production queues.

Two common-material angles; base, independent contact quadrature, independent
post-zero time-step refinement. Only the identical base pilots may resume.
No material fitting, fracture, source changes or relaxed physical acceptance.
"""
import argparse
import contextlib
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from run_support import source_hashes, write_json, save_npz, compute_process_policy
from prototype.curve_contact import CurveBladeProblem
from prototype.ipc_reference import BASELINE, adaptive_load_steps
from prototype.tip_contact import TipContactTracker
from tools.run_material_candidate import audit, signature, load_force_csv
from tools.run_material_queue import summarize_case

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def load(path):
    with np.load(path,allow_pickle=False) as state:
        return {k:state[k].copy() for k in state.files}


def make_problem(job):
    return CurveBladeProblem(maxwell_shear=1.24,relaxation_s=1.7,loading_rate=.2,
        bulk_modulus=23.34333333333331,case='oblique',mu=.35,shear_modulus=.14,
        ogden_alpha=2.,dhat=.0025,local_spacing=.05,surface_spacing=.005,nose_facets=64,
        inclination=job['angle'],tip_radius=.02,fine_halfwidth=3.,max_iterations=200,
        predictor='secant',hessian_mode='regularized',element='p1',
        contact_spacing=job['contact_spacing'],contact_order=job['contact_order'],guard_facets=256)


def curve_signature(p):
    value=signature(p)
    value['rigid_contact']=dict(method='analytic_blade_reference_gauss',
        spacing_mm=p.contact_spacing,order=p.contact_order,guard_facets=p.guard_facets,
        max_iterations=p.max_iterations,friction_smoothing_mm=BASELINE['friction_smoothing_mm'])
    return json.loads(json.dumps(value))


class CurveObserver:
    """Common dense nose for observation only; pole is exact, missing stays null."""
    def __init__(self,p,facets=1024):
        self.n=p.n;self.rest_anchor=p.rest[p.n].copy()
        a=np.deg2rad(p.config.blade_tip_angle_deg/2)
        theta=np.linspace(-np.pi+a,-a,facets+1)
        center=p.rest[p.n+2+p.nose_facets//2]+[0,p.options.tip_radius_mm]
        self.blade=np.vstack([p.rest[p.n:p.n+2],center+p.options.tip_radius_mm*
            np.column_stack((np.cos(theta),np.sin(theta))),p.rest[-2:]])
        self.tracker=TipContactTracker(np.vstack([p.rest[:p.n],self.blade]),p.mesh.triangles,
            p.n,p.config,p.dhat,facets)
        self.metadata=dict(self.tracker.metadata,method='analytic_circle_dense_observation_only',
            solve_geometry='exact circle and straight flanks; not this observation polygon',
            circle_sag_bound_mm=p.options.tip_radius_mm*(1-np.cos((np.pi-2*a)/(2*facets))),
            pole='exact circle lowest point; no nearest-projection filling',
            interpretation='geometric dhat support and reference migration; not zero-gap truth or accumulated slip')

    def observe(self,x,depth):
        value=np.vstack([x[:self.n],self.blade+x[self.n]-self.rest_anchor])
        return self.tracker.observe(value,depth)


def hausdorff(a,b):
    if not a or not b:return None
    def directed(source,target):
        probes=list(np.asarray(source).ravel())
        for left,right in zip(target[:-1],target[1:]):
            mid=(left[1]+right[0])/2
            if any(lo<=mid<=hi for lo,hi in source):probes.append(mid)
        return max(min(max(lo-x,0,x-hi) for lo,hi in target) for x in probes)
    return float(max(directed(a,b),directed(b,a)))


def verify_manifest(manifest):
    for relative,digest in manifest['source_sha256'].items():
        if sha(Path(manifest['source'])/relative)!=digest:raise RuntimeError('Frozen source changed: '+relative)
    for path,digest in manifest['input_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen study input changed: '+path)


def make_jobs(out,target=.5):
    jobs=[]
    for variant,spacing,order,increment,budget in (
        ('base',.0025,2,.01,3600.),('quadrature',.00125,3,.01,3600.),('halfstep',.0025,2,.005,5400.)):
        for angle in (30,60):
            name=f'{len(jobs)+1:02d}_{variant}_{angle}'
            jobs.append(dict(name=name,angle=angle,mu=.35,variant=variant,target_mm=target,
                contact_spacing=spacing,contact_order=order,increment=increment,seconds=budget,
                audit_allowance_s=600.,resume_pilot=variant=='base',status='pending',output=str(out/name)))
    return jobs


def make_angle_jobs(out,angles,target=2.):
    """Fresh independent angles; no inherited 30/60 history or refinement gate."""
    if (not angles or len(angles)!=len(set(angles)) or
            any(a not in (0,33,57) for a in angles) or not .01<=target<=2.):
        raise ValueError('Authorized angle queue: unique 0/33/57 degrees, target .01--2 mm')
    jobs=[]
    for i,angle in enumerate(angles):
        name=f'{i+1:02d}_angle{angle}'
        jobs.append(dict(name=name,angle=angle,mu=.35,variant='base',target_mm=target,
            contact_spacing=.0025,contact_order=2,increment=.01,seconds=21600.,
            audit_allowance_s=1800.,resume_pilot=False,status='pending',output=str(out/name)))
    return jobs


def refinement_allowed(job,jobs):
    if job['variant']=='base':return True
    parent=next(j for j in jobs if j['variant']=='base' and j['angle']==job['angle'])
    return parent.get('mechanics_completed') is True and parent.get('fresh_audit_passed') is True


def restore_pilot(p,manifest,job,out,report):
    """Reassemble the entire exact prefix before copying and restoring histories."""
    if job['variant']!='base' or job['increment']!=.01:
        raise ValueError('Changed integration or time grid must start fresh')
    pilot=manifest['pilots'][str(job['angle'])]
    if curve_signature(p)!=pilot['signature']:raise ValueError('Pilot signature mismatch')
    for rel,digest in manifest['physical_source_sha256'].items():
        if sha(Path(manifest['source'])/rel)!=digest:raise ValueError('Pilot physics changed')
    prior_x=p.rest.copy();prior_metric=p.material_metric.copy();prior_depth=p.start;dissipation=0.;fresh=[]
    for i,record in enumerate(pilot['steps']):
        item=pilot['states'][i];path=Path(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('Pilot checkpoint changed')
        state=load(path)
        np.testing.assert_array_equal(state['previous_positions'],prior_x)
        np.testing.assert_array_equal(state['previous_material_metric'],prior_metric)
        np.testing.assert_allclose(float(state['material_dt_s']),(record['depth_mm']-prior_depth)/p.loading_rate,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(float(state['depth_mm']),record['depth_mm'],atol=1e-12,rtol=0)
        fresh.append(p.audit_state(state,record))
        dissipation+=record['material']['viscous_dissipation_increment_n_mm']
        np.testing.assert_allclose(float(state['cumulative_viscous_dissipation_n_mm']),dissipation,atol=1e-10,rtol=1e-10)
        prior_x=state['positions'].copy();prior_metric=state['material_metric'].copy();prior_depth=record['depth_mm']
    write_json(out/'resume_audit.json',dict(passed=True,rows=fresh,scope='Identical analytic-contact prefix, both exact histories'))
    for i,item in enumerate(pilot['states']):
        shutil.copy2(item['path'],out/'states'/f'{i:04d}.npz')
        report['checkpoint_sha256'][f'{i:04d}.npz']=item['sha256']
    report['steps']=copy.deepcopy(pilot['steps']);report['prior_elapsed_s']=pilot['elapsed_s']
    p.material_metric=prior_metric;p.material_dissipation=dissipation;p.previous=prior_x.copy()
    histories=[(r['depth_mm'],load(pilot['states'][i]['path'])['positions'])
        for i,r in list(enumerate(pilot['steps']))[-2:]]
    p.seed_predictor_history(histories)
    print(f'RESUME_READY angle={job["angle"]} states={len(fresh)} depth_mm={prior_depth} both_histories=True',flush=True)
    return prior_x,prior_depth


def export_case(out,report):
    with (out/'fz_depth.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['depth_mm','fz_n'])
        w.writerows((r['depth_mm'],r['reaction_fz_n']) for r in report['steps'])
    with (out/'contact_depth.csv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['depth_mm','pole_status','reference_offset_mm','nose_intervals_mm','min_j'])
        for r in report['steps']:
            o=r.get('tip_contact',{})
            w.writerow([r['depth_mm'],o.get('status'),o.get('reference_offset_mm'),
                json.dumps(o.get('nose_offset_intervals_mm',[])),r['min_j']])


def solve_case(out,index):
    manifest=read(out/'manifest.json');verify_manifest(manifest);job=manifest['jobs'][index]
    folder=Path(job['output']);folder.mkdir(exist_ok=False);(folder/'states').mkdir()
    report=dict(status='initializing',completed=False,steps=[],checkpoint_sha256={},
        source_sha256=manifest['source_sha256'],target_mm=job['target_mm'],settings=job,
        force_tolerance_n=BASELINE['force_tolerance_n'],rejected_load_attempts=[],
        parameter_status='same material for every angle; exploratory, not independently identified')
    def save():write_json(folder/'result.json',report)
    save();p=None;x=None;started=time.monotonic()
    with (folder/'run.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),\
         threadpool_limits(limits=2),compute_process_policy() as policy:
        report['runtime_process_policy']=policy
        try:
            p=make_problem(job);p.stagnation_window=40;report['signature']=curve_signature(p)
            report['mesh']=dict(nodes=p.n,elements=len(p.mesh.triangles))
            report['ipc_binary_sha256']=sha(p.ipc.__file__)
            observer=CurveObserver(p);report['contact_observation']=observer.metadata
            x=p.rest.copy();old=p.start
            if job['resume_pilot']:x,old=restore_pilot(p,manifest,job,folder,report)
            for i,r in enumerate(report['steps']):
                r['tip_contact']=observer.observe(load(folder/'states'/f'{i:04d}.npz')['positions'],r['depth_mm'])
            report.update(status='running',last_depth_mm=float(old));save()
            deadline=time.monotonic()+job['seconds'];solve_started=time.monotonic()
            targets=np.r_[0.,np.arange(job['increment'],job['target_mm']-1e-12,job['increment']),job['target_mm']]
            def rejected(event):report['rejected_load_attempts'].append(event);save()
            for target in targets[targets>old+1e-12]:
                for x,record in adaptive_load_steps(p,x,old,float(target),deadline,.000625,rejected):
                    old=record['depth_mm'];name=f'{len(report["steps"]):04d}.npz'
                    save_npz(folder/'states'/name,reference_xy=p.rest,triangles=p.mesh.triangles,
                        positions=x,previous_positions=p.previous,depth_mm=old,body_nodes=p.n,**p.checkpoint_data())
                    report['checkpoint_sha256'][name]=sha(folder/'states'/name)
                    report['steps'].append(record);report['last_depth_mm']=old
                    record['tip_contact']=observer.observe(x,old)
                    report['solve_elapsed_s']=time.monotonic()-solve_started;save()
                    print(f'ACCEPT depth_mm={old:.6f} Fz_N={record["reaction_fz_n"]:.9g} residual_N={record["residual_n"]:.6g}',flush=True)
            report.update(status='completed',completed=True)
        except Exception as exc:
            report.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',completed=False,
                error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
            if p is not None and x is not None and p.material_dt is not None:
                save_npz(folder/'failed_trial.npz',positions=getattr(p,'trial',x),previous_positions=p.previous,**p.checkpoint_data())
        report['elapsed_s']=time.monotonic()-started;save()
        try:
            if p is not None and report['steps']:
                audited=audit(p,folder,report)
                for i,r in enumerate(audited['rows']):
                    if r['checkpoint_sha256']!=report['checkpoint_sha256'][f'{i:04d}.npz']:
                        raise RuntimeError('Accepted checkpoint changed after save')
        except Exception as exc:
            report.update(status='audit_failed',completed=False,audit_error=f'{type(exc).__name__}: {exc}');traceback.print_exc()
    report['total_elapsed_s']=time.monotonic()-started;save();export_case(folder,report)
    print(json.dumps({k:report.get(k) for k in ('status','completed','last_depth_mm','elapsed_s','error')}),flush=True)


def matched_records(rows,other):
    lookup={round(r['depth_mm'],9):(i,r) for i,r in enumerate(other)}
    return [(i,r,*lookup[round(r['depth_mm'],9)]) for i,r in enumerate(rows) if round(r['depth_mm'],9) in lookup]


def compare_paths(folder,report,reference_folder,reference,*,observers=None):
    """Fields only at identical accepted depths; never interpolate contact identity."""
    values=[]
    for i,r,j,other in matched_records(report['steps'],reference['steps']):
        a=load(folder/'states'/f'{i:04d}.npz');b=load(reference_folder/'states'/f'{j:04d}.npz')
        np.testing.assert_array_equal(a['reference_xy'][:int(a['body_nodes'])],b['reference_xy'][:int(b['body_nodes'])])
        np.testing.assert_array_equal(a['triangles'],b['triangles'])
        if observers:
            observer,surface=observers;oa=observer.observe(a['positions'],r['depth_mm']);ob=observer.observe(b['positions'],other['depth_mm'])
        else:raise ValueError('Common observation geometry required')
        difference=r['reaction_fz_n']-other['reaction_fz_n']
        pole=None if oa['reference_offset_mm'] is None or ob['reference_offset_mm'] is None else oa['reference_offset_mm']-ob['reference_offset_mm']
        nose=hausdorff(oa['nose_offset_intervals_mm'],ob['nose_offset_intervals_mm'])
        availability_match=(oa['status']==ob['status'] and bool(oa['nose_offset_intervals_mm'])==bool(ob['nose_offset_intervals_mm']))
        values.append(dict(depth_mm=r['depth_mm'],delta_fz_n=difference,
            force_screen_passed=abs(difference)<=max(.2,.05*abs(other['reaction_fz_n'])),
            pole_delta_mm=pole,pole_status=oa['status'],reference_pole_status=ob['status'],nose_hausdorff_mm=nose,
            pole_comparison_available=pole is not None,observation_availability_matches=availability_match,
            observable_screen_passed=availability_match and (pole is None or abs(pole)<=.02) and (nose is None or nose<=.02),
            max_surface_delta_mm=float(np.linalg.norm(a['positions'][surface]-b['positions'][surface],axis=1).max()),
            min_j=r['min_j'],reference_min_j=other['min_j']))
    return dict(matched_states=len(values),rows=values,
        comparison_scope='Matching accepted depths only; two-level engineering screen, not mesh/time convergence',
        all_force_screens=bool(values) and all(x['force_screen_passed'] for x in values),
        all_available_observable_screens=bool(values) and all(x['observable_screen_passed'] for x in values))


def experimental_comparison(rows,angle,manifest):
    if not rows:return None
    d=np.array([r['depth_mm'] for r in rows]);f=np.array([r['reaction_fz_n'] for r in rows])
    if angle==60:
        ed,ef,rate=load_force_csv(manifest['experiment60'])
        mask=(ed>=d[0])&(ed<=d[-1]);delta=np.interp(ed[mask],d,f)-ef[mask]
        anchors=[]
        for depth in (0.,.1,.25,.5,.75,1.,1.5,2.):
            if d[0]<=depth<=d[-1]:anchors.append(dict(depth_mm=depth,fe_n=float(np.interp(depth,d,f)),experiment_n=float(np.interp(depth,ed,ef))))
        return dict(source='confirmed 60-degree CSV, already used in exploratory material selection',
            range_mm=[float(d[0]),float(d[-1])],rmse_n=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
            anchors=anchors,baseline='raw .51 N preserved, no re-zeroing',independent_validation=False)
    estimates=read(manifest['experiment30']);anchors=[]
    for e in estimates:
        if d[0]<=e['depth_mm']<=d[-1]:
            anchors.append(dict(e,fe_n=float(np.interp(e['depth_mm'],d,f))))
    return dict(source='1008n33a1 user screenshot manual readings; no confirmed raw CSV',
        range_mm=[float(d[0]),float(d[-1])],anchors=anchors,
        uncertainty='visual reading ranges, not statistical confidence intervals',independent_validation=False)


def load_image_estimates(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream,strict=True)
        names=('depth_mm','experiment_image_estimate_n','image_reading_lower_n','image_reading_upper_n')
        if not set(names)<=set(reader.fieldnames or []):raise ValueError('Unexpected screenshot estimate schema')
        rows=[]
        for r in reader:
            if len(rows)>=100:raise ValueError('Unexpected size of manual anchor table')
            d,f,lo,hi=(float(r[k]) for k in names)
            if not np.isfinite([d,f,lo,hi]).all() or lo>f or f>hi or d<0 or (rows and d<=rows[-1]['depth_mm']):
                raise ValueError('Invalid or unordered visual estimates')
            rows.append(dict(depth_mm=d,estimate_n=f,lower_n=lo,upper_n=hi))
    if not rows:raise ValueError('Empty visual estimates')
    return rows


def summarize(out,state):
    manifest=read(out/'manifest.json');verify_manifest(manifest)
    if manifest.get('kind')=='angle_queue':
        return summarize_angles(out,state,manifest)
    data=[];lines=['# Analytic blade: independent path qualification','',
        'Numerical completion, discretization comparison and experimental agreement are separate.',
        'Half-step cases halve increments AFTER nominal depth zero; the stress-free preparation to zero is unchanged.','',
        '| Case | Status | Depth mm | Fz N | Path min J | Fresh audit |',
        '|---|---|---:|---:|---:|---|']
    for job in state['jobs']:
        lines.append(f'| {job["name"]} | {job["status"]} | {job.get("last_depth_mm")} | {job.get("endpoint_fz_n")} | {job.get("min_j")} | {job.get("fresh_audit_passed")} |')
        path=Path(job['output'])/'result.json'
        if not path.exists() or job.get('fresh_audit_passed') is not True:continue
        r=read(path);data.append((job,r))
    analysis=dict(jobs=[],experimental_validation=False,history_scope='independent candidate paths; common material across angles')
    with threadpool_limits(limits=2),compute_process_policy():
        for job,r in data:
            review_path=Path(job['output'])/'comparison.json'
            if review_path.exists():entry=read(review_path)
            else:
                p=make_problem(job);observer=CurveObserver(p)
                entry=dict(name=job['name'],experiment=experimental_comparison(r['steps'],job['angle'],manifest))
                log=(Path(job['output'])/'run.log').read_text(encoding='utf-8')
                assembly=[float(v) for v in re.findall(r'PERF_ASSEMBLY seconds=([0-9.]+)',log)]
                entry['cost']=dict(solve_elapsed_s=r.get('solve_elapsed_s'),elapsed_s=r.get('elapsed_s'),
                    prior_elapsed_s=r.get('prior_elapsed_s',0.),assembly_count=len(assembly),
                    mean_assembly_s=float(np.mean(assembly)) if assembly else None,
                    rejected_trials=log.count('FRICTION_PTC_REJECT'),accepted_states=len(r['steps']))
                reference=manifest['references'][str(job['angle'])];reference_folder=Path(reference['folder'])
                entry['legacy_same_mu']=compare_paths(Path(job['output']),r,reference_folder,read(reference_folder/'result.json'),observers=(observer,p.body_surface))
                if job['variant']!='base':
                    base=next(j for j in state['jobs'] if j['variant']=='base' and j['angle']==job['angle'])
                    entry['independent_refinement']=compare_paths(Path(job['output']),r,Path(base['output']),read(Path(base['output'])/'result.json'),observers=(observer,p.body_surface))
                write_json(review_path,entry)
            analysis['jobs'].append(entry)
    write_json(out/'analysis.json',analysis)
    (out/'summary.md').write_text('\n'.join(lines)+'\n\nOnly freshly audited accepted prefixes are plotted. Missing pole contacts remain gaps.\n',encoding='utf-8')
    if not data:return
    plot_summary(out,manifest,data)


def plot_summary(out,manifest,data,*,preview=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey='row',layout='constrained')
        colors={'base':'#0072B2','quadrature':'#D55E00','halfstep':'#009E73'}
        styles={'base':'-','quadrature':'--','halfstep':':'}
        for col,angle in enumerate((30,60)):
            upper,lower=axes[:,col]
            ref=read(Path(manifest['references'][str(angle)]['folder'])/'result.json')['steps']
            ref=[r for r in ref if r['depth_mm']<=manifest['target_mm']+1e-10]
            upper.plot([r['depth_mm'] for r in ref],[r['reaction_fz_n'] for r in ref],color='#777777',ls='-.',label='Original IPC reference')
            if angle==60:
                ed,ef,_=load_force_csv(manifest['experiment60']);mask=ed<=manifest['target_mm']
                upper.plot(ed[mask],ef[mask],color='black',lw=1,label='Experiment 1014 (calibration trial)')
            else:
                e=read(manifest['experiment30']);e=[v for v in e if v['depth_mm']<=manifest['target_mm']]
                upper.errorbar([v['depth_mm'] for v in e],[v['estimate_n'] for v in e],
                    yerr=np.array([[v['estimate_n']-v['lower_n'] for v in e],[v['upper_n']-v['estimate_n'] for v in e]]),
                    color='black',fmt='s',capsize=4,label='Experiment image (visual range)')
            for job,r in data:
                if job['angle']!=angle:continue
                rows=r['steps'];d=[v['depth_mm'] for v in rows];variant=job['variant']
                label=f'{variant} ({job["status"]})'
                upper.plot(d,[v['reaction_fz_n'] for v in rows],color=colors[variant],ls=styles[variant],label=label)
                migration=[v.get('tip_contact',{}).get('reference_offset_mm') for v in rows]
                lower.plot(d,[np.nan if v is None else v for v in migration],color=colors[variant],ls=styles[variant],label=variant)
            upper.set(title=f'{angle} degrees, common material, mu=0.35',ylabel='Fz (N)');upper.legend(fontsize=8)
            lower.set(xlabel='Downward depth (mm)',ylabel='Pole reference offset (mm)')
            lower.text(.02,.98,'Gaps: no unique lowest-pole contact',transform=lower.transAxes,va='top',fontsize=8)
            for ax in (upper,lower):ax.set_xlim(0,manifest['target_mm']);ax.grid(alpha=.2)
        fig.suptitle(('LAYOUT CHECK: previous .05 mm pilots | NOT new study results' if preview else
            'Analytic circle contact diagnostic | accepted data only | no fracture'))
        fig.savefig(out/'comparison.png',dpi=160,facecolor='white');plt.close(fig)
    write_json(out/'figure_provenance.json',dict(audience='internal diagnostic, no journal target',
        data='analysis.json and per-case result.json/fz_depth.csv/contact_depth.csv',
        transformations='no smoothing or force rescaling; FE lines connect accepted depths; missing pole values NaN',
        uncertainty='30-degree screenshot visual ranges, not confidence intervals',
        alt_text='Two angles with common material: force versus depth above; reference lowest-pole migration below. Missing contact is left as gaps. Original IPC and separate experimental evidence are shown.',
        preview=preview,manual_review='Pending visual inspection; automated figures are not visual QA'))


def summarize_curve_case(folder,job,returncode,forced_status=None):
    result=summarize_case(folder,job,returncode,forced_status)
    try:report=read(Path(folder)/'result.json')
    except (OSError,ValueError):report={}
    matches=(report.get('signature')==job['expected_signature'] and
             report.get('settings',{}).get('increment')==job['increment'])
    result['full_candidate_signature_matches']=matches
    if not matches:
        result.update(mechanics_completed=False,status='configuration_mismatch')
    return result


def run_job(out,manifest,job,index,state):
    verify_manifest(manifest)
    if Path(job['output']).exists():raise RuntimeError('Refusing to overwrite case output')
    command=[manifest['python'],'-X','utf8','-u','-m','tools.run_curve_contact_study',
        '--output',str(out),'--case-index',str(index)]
    env=os.environ.copy();env['PYTHONPATH']=manifest['runtime_path'];env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
    started=time.monotonic();forced=None
    with (out/(job['name']+'.stdout.log')).open('wb') as stdout,(out/(job['name']+'.stderr.log')).open('wb') as stderr:
        kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        child=subprocess.Popen(command,cwd=manifest['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**kwargs)
        job.update(status='running',solver_pid=child.pid,started_utc=now(),command=command);write_json(out/'queue_status.json',state)
        try:code=child.wait(timeout=job['seconds']+job['audit_allowance_s'])
        except subprocess.TimeoutExpired:
            forced='process_timeout';child.terminate()
            try:code=child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=20)
    result=summarize_curve_case(job['output'],job,code,forced)
    result.update(wall_elapsed_s=time.monotonic()-started,finished_utc=now())
    return result


def worker(out):
    manifest=read(out/'manifest.json')
    with (out/'worker_started.json').open('x',encoding='utf-8') as stream:json.dump(dict(pid=os.getpid(),time=now()),stream)
    state=dict(status='running',worker_pid=os.getpid(),started_utc=now(),jobs=copy.deepcopy(manifest['jobs']))
    write_json(out/'queue_status.json',state)
    for i,job in enumerate(state['jobs']):
        if not refinement_allowed(job,state['jobs']):
            job.update(status='skipped_base_not_completed',mechanics_completed=False);write_json(out/'queue_status.json',state);continue
        print('START '+job['name'],flush=True)
        try:job.update(run_job(out,manifest,job,i,state))
        except Exception as exc:
            job.update(status='worker_case_failed',mechanics_completed=False,error=f'{type(exc).__name__}: {exc}')
            (out/(job['name']+'.traceback.log')).write_text(traceback.format_exc(),encoding='utf-8')
        write_json(out/(job['name']+'.outcome.json'),job);write_json(out/'queue_status.json',state)
        try:summarize(out,state)
        except Exception as exc:
            state.setdefault('postprocessing_errors',[]).append(dict(case=job['name'],error=f'{type(exc).__name__}: {exc}'))
            (out/(job['name']+'.postprocess.log')).write_text(traceback.format_exc(),encoding='utf-8')
        print('END '+job['name']+' '+job['status'],flush=True)
    state.update(status='finished',finished_utc=now(),all_cases_completed=all(j.get('mechanics_completed') is True for j in state['jobs']))
    try:summarize(out,state)
    except Exception as exc:state.setdefault('postprocessing_errors',[]).append(dict(case='terminal',error=f'{type(exc).__name__}: {exc}'))
    write_json(out/'queue_status.json',state)


def prepare(out,pilot_root,target=.5):
    if not .06<=target<=.5:raise ValueError('This diagnostic is bounded to [.06, .5] mm')
    pilot=read(pilot_root/'pilot.json');qualified=read(pilot_root/'review.json')
    if pilot['status']!='completed' or qualified['status']!='completed' or not all(r['all_shallow_screens_passed'] for r in qualified['pilots']):
        raise ValueError('Independent analytic pilots did not qualify')
    if len(pilot['rows'])!=2 or {r['angle'] for r in pilot['rows']}!={30,60} or any(r['status']!='completed' or r['target_mm']!=.05 for r in pilot['rows']):
        raise ValueError('Expected the two completed .05 mm pilots')
    hashes=source_hashes()
    if hashes!=pilot['source_sha256']:raise ValueError('Current physics differ from qualified pilots')
    for name,digest in hashes.items():
        if sha(pilot_root/'source'/name)!=digest:raise ValueError('Qualified source changed')
    if sha(ROOT/'prototype/ipc_baseline.json')!=pilot['baseline_sha256']:raise ValueError('Baseline configuration changed')
    old=ROOT/'outputs/relaxation_partition_20261015'
    image=old/'experiment30_review/comparison_anchors.csv';estimates=load_image_estimates(image)
    experiment=ROOT/'docs/1014n57a1__Fz__curve_0.csv'
    if not np.isclose(load_force_csv(experiment)[2],.2,atol=1e-12,rtol=0):raise ValueError('Experimental speed differs')
    out.mkdir(parents=True,exist_ok=False);frozen=out/'source';source_hashes(snapshot=frozen)
    physical=dict(hashes);all_hashes=dict(hashes)
    for name in ('prototype/ipc_baseline.json','tools/run_curve_contact_study.py','tools/run_material_candidate.py','tools/run_material_queue.py'):
        destination=frozen/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination);all_hashes[name]=sha(destination)
    physical['prototype/ipc_baseline.json']=pilot['baseline_sha256']
    manifest=dict(python=sys.executable,source=str(frozen),source_sha256=all_hashes,
        physical_source_sha256=physical,runtime_path=str(ROOT/'.runtime/ipc-python'),input_sha256={},
        created_utc=now(),target_mm=target,jobs=make_jobs(out,target),pilots={},references={})
    with threadpool_limits(limits=2),compute_process_policy():
        for job in manifest['jobs']:job['expected_signature']=curve_signature(make_problem(job))
        for row in pilot['rows']:
            angle=row['angle'];job=next(j for j in manifest['jobs'] if j['angle']==angle and j['variant']=='base')
            p=make_problem(job);states=[]
            for i,record in enumerate(row['steps']):
                path=pilot_root/f'pilot{angle}'/f'{i:04d}.npz';digest=sha(path)
                if digest!=row['checkpoint_sha256'][path.name]:raise ValueError('Pilot state changed')
                p.audit_state(load(path),record);states.append(dict(path=str(path),sha256=digest));manifest['input_sha256'][str(path)]=digest
            manifest['pilots'][str(angle)]=dict(signature=curve_signature(p),steps=row['steps'],states=states,elapsed_s=row['elapsed_s'])
    for angle,folder in ((30,old/'angles30_0_2mm/01_angle30'),(60,old/'depth_2mm/01_angle60')):
        manifest['references'][str(angle)]=dict(folder=str(folder))
        manifest['input_sha256'][str(folder/'result.json')]=sha(folder/'result.json')
        for i,r in enumerate(read(folder/'result.json')['steps']):
            if r['depth_mm']<=target+1e-10:
                path=folder/'states'/f'{i:04d}.npz';manifest['input_sha256'][str(path)]=sha(path)
    destination=out/'experiment60.csv';shutil.copy2(experiment,destination)
    manifest['experiment60']=str(destination);manifest['input_sha256'][str(destination)]=sha(destination)
    write_json(out/'experiment30.json',estimates);manifest['experiment30']=str(out/'experiment30.json')
    manifest['experiment30_provenance']=dict(source=str(image),sha256=sha(image),meaning='user1008n33a1 screenshot coarse estimates; 3-degree difference ignored per instruction')
    manifest['input_sha256'][manifest['experiment30']]=sha(out/'experiment30.json')
    manifest['policy']=dict(material='one common material, mu=.35; no fitting, fracture or default promotion',
        resume='ONLY same-angle identical base .05 pilots; other variants fresh',
        refinement_gate='same-angle base target completed AND fresh audit passed',
        time_refinement='.01 -> .005 after nominal zero; same initial rest-to-zero preparation; adaptive substeps retained',
        comparison='max(.2N,5% reference Fz), .02 mm available reference contact; screening not convergence/experimental validation',
        failure='advance to other angle; skip refinements whose base did not complete; no old queues')
    write_json(out/'manifest.json',manifest);verify_manifest(manifest);return manifest


def launch(out,manifest):
    if (out/'launch.json').exists():raise FileExistsError('Study has already been launched')
    if manifest.get('preflight',{}).get('status')!='passed':raise RuntimeError('Actual continuation preflight required')
    command=[manifest['python'],'-X','utf8','-u','-m','tools.run_curve_contact_study','--output',str(out),'--worker']
    env=os.environ.copy();env['PYTHONPATH']=manifest['runtime_path'];env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
    startup=None;flags=0
    if os.name=='nt':
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=subprocess.SW_HIDE
        flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
    # Exclusive claim precedes opening logs: a duplicate cannot erase live logs.
    with (out/'launch.json').open('x',encoding='utf-8') as claim,\
         (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        process=subprocess.Popen(command,cwd=manifest['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            startupinfo=startup,creationflags=flags,start_new_session=os.name!='nt')
        record=dict(worker_pid=process.pid,command=command,started_utc=now());json.dump(record,claim,indent=2)
    return record


def preflight(out):
    """Two actual .05 -> .06 continuations; keep them as the new base prefixes."""
    manifest=read(out/'manifest.json');verify_manifest(manifest)
    if (out/'launch.json').exists():raise RuntimeError('Cannot modify a launched study')
    folder=out/'preflight';folder.mkdir(exist_ok=False)
    short=copy.deepcopy(manifest);short['target_mm']=.06
    short['jobs']=copy.deepcopy(manifest['jobs'][:2])
    for job in short['jobs']:
        job.update(target_mm=.06,seconds=180.,output=str(folder/job['name']))
    short['policy']['purpose']='Bounded end-to-end continuation/audit/plot check before detached study'
    write_json(folder/'manifest.json',short);worker(folder)
    status=read(folder/'queue_status.json')
    if not status['all_cases_completed'] or status.get('postprocessing_errors'):
        raise RuntimeError('Continuation/postprocessing preflight did not pass; no full study launch')
    for job in status['jobs']:
        report=read(Path(job['output'])/'result.json');states=[]
        for i,row in enumerate(report['steps']):
            path=Path(job['output'])/'states'/f'{i:04d}.npz';digest=sha(path)
            if digest!=report['checkpoint_sha256'][path.name]:raise RuntimeError('Preflight checkpoint changed')
            states.append(dict(path=str(path),sha256=digest));manifest['input_sha256'][str(path)]=digest
        manifest['pilots'][str(job['angle'])]=dict(signature=report['signature'],steps=report['steps'],states=states,
            elapsed_s=report['elapsed_s']+report.get('prior_elapsed_s',0.))
    manifest['policy']['resume']='Same-angle identical analytic base prefixes, freshly audited through .06 in preflight; other variants fresh'
    manifest['preflight']=dict(status='passed',folder=str(folder),depth_mm=.06)
    write_json(out/'manifest.json',manifest);verify_manifest(manifest)
    print('PREFLIGHT_PASSED own exact histories through .06 mm for both angles',flush=True)


def prepare_angles(out,angles,target=2.):
    """Reuse the qualified analytic physics; freeze one explicit 2 mm queue."""
    jobs=make_angle_jobs(out,angles,target)
    qualified=ROOT/'outputs/curve_contact_depth_20261015'
    prior=read(qualified/'manifest.json');verify_manifest(prior)
    status=read(qualified/'queue_status.json')
    if status['status']!='finished' or len(status['jobs'])!=6 or not all(
            j.get('mechanics_completed') and j.get('fresh_audit_passed') for j in status['jobs']):
        raise RuntimeError('Completed six-case analytic qualification required')
    # Reporting/queue code may change; each qualified physical source may not.
    physical=prior['physical_source_sha256']
    for rel,digest in physical.items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Qualified physical source changed: '+rel)
    out.mkdir(exist_ok=False,parents=True);frozen=out/'source'
    hashes=source_hashes(snapshot=frozen)
    for rel in ('prototype/ipc_baseline.json','tools/run_curve_contact_study.py',
                'tools/run_material_candidate.py','tools/run_material_queue.py'):
        dest=frozen/rel;dest.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(ROOT/rel,dest);hashes[rel]=sha(dest)
    manifest=dict(kind='angle_queue',python=sys.executable,source=str(frozen),
        source_sha256=hashes,physical_source_sha256=physical,input_sha256={},
        runtime_path=str(ROOT/'.runtime/ipc-python'),created_utc=now(),target_mm=target,
        jobs=jobs,pilots={},experiments={},qualification=dict(folder=str(qualified),
            manifest_sha256=sha(qualified/'manifest.json'),status_sha256=sha(qualified/'queue_status.json')),
        policy=dict(material='common Ginf=.14/Gv=1.24/tau=1.7/K=23.34333333333331; NOT stress-accelerated flow',
            geometry='fine P1; real .02 mm circle, 99755 23 deg/.245 mm; bonded bottom, plane strain',
            acceptance='original 1e-4 N, positive physical J and collision-free whole-edge CCD',
            resume='fresh from stress-free reference; only same-angle new preflight may seed continuation',
            failure='advance after success, failure or timeout; no old queue restart',
            evidence='Fz and reference-material migration; missing pole stays null, not accumulated slip',
            budget='21600 s solve plus1800 s setup/audit per full case',
            scope='user-authorized 0/33/57 to2 mm; no fracture, no per-angle material fit'))
    with threadpool_limits(limits=2),compute_process_policy() as policy:
        checks=[]
        for job in jobs:
            p=make_problem(job);job['expected_signature']=curve_signature(p)
            # Construction runs the original reference-force/geometry preflight.
            checks.append(dict(angle=job['angle'],signature=job['expected_signature'],
                nodes=p.n,elements=len(p.mesh.triangles),reference_depth_mm=p.start))
        manifest['reference_checks']=dict(status='passed',cases=checks,compute_policy=policy)
    if 57 in angles:
        src=ROOT/'docs/1014n57a1__Fz__curve_0.csv';dst=out/'experiment57.csv'
        if not np.isclose(load_force_csv(src)[2],.2,rtol=0,atol=1e-12):raise ValueError('Experiment loading rate mismatch')
        shutil.copy2(src,dst);manifest['input_sha256'][str(dst)]=sha(dst)
        manifest['experiments']['57']=dict(kind='csv',path=str(dst),source=str(src),
            interpretation='User1014n57a1, previously treated as nominal60; now compare requested57 explicitly; raw force offset retained')
    if 33 in angles:
        src=ROOT/'outputs/relaxation_partition_20261015/experiment30_review/comparison_anchors.csv'
        dst=out/'experiment33.json';write_json(dst,load_image_estimates(src))
        manifest['input_sha256'][str(dst)]=sha(dst)
        manifest['experiments']['33']=dict(kind='image',path=str(dst),source=str(src),source_sha256=sha(src),
            interpretation='1008n33a1 manual screenshot estimates, visual ranges only')
    # No invented raw 0-degree curve or borrowed 57-degree observations.
    if 0 in angles:manifest['experiments']['0']=dict(kind='unavailable',interpretation='No numeric0-degree curve selected')
    write_json(out/'manifest.json',manifest);verify_manifest(manifest)
    return manifest


def angle_experiment(rows,angle,manifest):
    item=manifest.get('experiments',{}).get(str(angle),{})
    if item.get('kind')=='csv':
        result=experimental_comparison(rows,60,dict(experiment60=item['path']))
    elif item.get('kind')=='image':
        result=experimental_comparison(rows,30,dict(experiment30=item['path']))
    else:return dict(available=False,source=item.get('interpretation','No matching experiment selected'))
    if result is not None:
        result.update(source=item['interpretation'],angle_deg=angle,available=True)
    return result


def summarize_angles(out,state,manifest):
    """All requested angles and full accepted range; no 30/60 hard-coded plots."""
    data=[];entries=[]
    lines=['# 0 / 33 / 57 degree intact queue','',
        'Common exploratory material; numerical completion is not experimental validation.',
        'Only terminally audited accepted prefixes are compared. Missing contact remains missing.','',
        '| Angle | Status | Last depth mm | Fz N | Path min J | Fresh audit |',
        '|---|---|---:|---:|---:|---|']
    for job in state['jobs']:
        lines.append(f'| {job["angle"]} | {job["status"]} | {job.get("last_depth_mm")} | {job.get("endpoint_fz_n")} | {job.get("min_j")} | {job.get("fresh_audit_passed")} |')
        path=Path(job['output'])/'result.json'
        if not path.exists() or job.get('fresh_audit_passed') is not True:continue
        report=read(path);data.append((job,report))
        entry=dict(angle_deg=job['angle'],status=job['status'],accepted_states=len(report['steps']),
            range_mm=[report['steps'][0]['depth_mm'],report['steps'][-1]['depth_mm']],
            experiment=angle_experiment(report['steps'],job['angle'],manifest),
            min_j=job.get('min_j'),solve_elapsed_s=report.get('solve_elapsed_s'),
            prior_elapsed_s=report.get('prior_elapsed_s',0.),experimental_validation=False)
        write_json(Path(job['output'])/'comparison.json',entry);entries.append(entry)
    write_json(out/'analysis.json',dict(jobs=entries,experimental_validation=False))
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not data:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':9}):
        fig,axes=plt.subplots(2,len(state['jobs']),figsize=(4.4*len(state['jobs']),7),
            sharex=True,sharey='row',squeeze=False,layout='constrained')
        for col,job in enumerate(state['jobs']):
            angle=job['angle'];upper,lower=axes[:,col]
            item=manifest.get('experiments',{}).get(str(angle),{})
            if item.get('kind')=='csv':
                ed,ef,_=load_force_csv(item['path']);mask=ed<=manifest['target_mm']
                upper.plot(ed[mask],ef[mask],color='black',lw=1,label='Experiment1014n57a1 (raw N)')
            elif item.get('kind')=='image':
                e=[v for v in read(item['path']) if v['depth_mm']<=manifest['target_mm']]
                if e:upper.errorbar([v['depth_mm'] for v in e],[v['estimate_n'] for v in e],
                    yerr=np.array([[v['estimate_n']-v['lower_n'] for v in e],[v['upper_n']-v['estimate_n'] for v in e]]),
                    color='black',fmt='s',ms=3,capsize=3,label='Experiment image (visual range)')
            else:upper.text(.03,.96,'No numeric experiment selected',transform=upper.transAxes,va='top')
            match=next((r for j,r in data if j['name']==job['name']),None)
            if match:
                rows=match['steps'];d=[r['depth_mm'] for r in rows]
                upper.plot(d,[r['reaction_fz_n'] for r in rows],color='#0072B2',label='Audited FE ('+job['status']+')')
                offset=[r.get('tip_contact',{}).get('reference_offset_mm') for r in rows]
                lower.plot(d,[np.nan if v is None else v for v in offset],color='#0072B2')
            else:upper.text(.03,.84,job['status'],transform=upper.transAxes,va='top')
            upper.set(title=f'{angle} degrees, mu=0.35',ylabel='Fz (N)')
            if upper.get_legend_handles_labels()[0]:upper.legend(fontsize=8)
            lower.set(xlabel='Laboratory downward depth (mm)',ylabel='Pole reference migration (mm)')
            lower.text(.03,.96,'Gaps: no unique pole contact',transform=lower.transAxes,va='top',fontsize=8)
            for ax in (upper,lower):ax.set_xlim(0,manifest['target_mm']);ax.grid(alpha=.2)
        fig.suptitle('Analytic circle + common finite Maxwell | no fracture | accepted data only')
        fig.savefig(out/'comparison.png',dpi=160,facecolor='white');plt.close(fig)
    write_json(out/'figure_provenance.json',dict(data='Per-case result.json, audit.json and manifest experiments',
        transforms='No force re-zero/rescale/smoothing; lines connect accepted depths; missing pole values NaN',
        uncertainty='33-degree manual visual ranges, not statistical confidence intervals',
        review='Automatic terminal figure; pending manual visual QA',
        alt_text='Three requested angles: intact FE force and available matching experiment above, reference contact migration below. Missing results remain absent.'))


def preflight_angles(out):
    """Actual fresh 0--.01 paths for each NEW angle, then exact own-history resume."""
    manifest=read(out/'manifest.json');verify_manifest(manifest)
    if manifest.get('kind')!='angle_queue':raise ValueError('Not an angle queue')
    if (out/'launch.json').exists():raise RuntimeError('Cannot modify launched queue')
    folder=out/'preflight';folder.mkdir(exist_ok=False)
    short=copy.deepcopy(manifest);short['target_mm']=.01
    for job in short['jobs']:
        job.update(target_mm=.01,seconds=180.,audit_allowance_s=180.,output=str(folder/job['name']),resume_pilot=False)
    write_json(folder/'manifest.json',short);worker(folder)
    state=read(folder/'queue_status.json')
    if not state['all_cases_completed'] or state.get('postprocessing_errors'):
        raise RuntimeError('Fresh angle preflight failed; no full queue launch')
    for checked in state['jobs']:
        report=read(Path(checked['output'])/'result.json');states=[]
        job=next(j for j in manifest['jobs'] if j['angle']==checked['angle'])
        for i,row in enumerate(report['steps']):
            path=Path(checked['output'])/'states'/f'{i:04d}.npz';digest=sha(path)
            if digest!=report['checkpoint_sha256'][path.name]:raise RuntimeError('Preflight checkpoint changed')
            states.append(dict(path=str(path),sha256=digest));manifest['input_sha256'][str(path)]=digest
        manifest['pilots'][str(job['angle'])]=dict(signature=report['signature'],steps=report['steps'],
            states=states,elapsed_s=report['elapsed_s'])
        job['resume_pilot']=True
    manifest['preflight']=dict(status='passed',folder=str(folder),depth_mm=.01,
        interpretation='Every angle starts fresh; only its OWN two-state prefix is resumed')
    write_json(out/'manifest.json',manifest);verify_manifest(manifest)
    print('PREFLIGHT_PASSED all requested angles, exact own histories through .01 mm',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot-root',type=Path,default=ROOT/'outputs/curve_contact_20261015')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--worker',action='store_true');mode.add_argument('--case-index',type=int)
    mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--launch-prepared',action='store_true');mode.add_argument('--preflight',action='store_true')
    parser.add_argument('--depth',type=float,default=None)
    parser.add_argument('--angles',type=int,nargs='+',help='Explicit fresh angle queue, separate from the six-case diagnostic')
    args=parser.parse_args();out=args.output.resolve()
    if args.preflight:
        if read(out/'manifest.json').get('kind')=='angle_queue':preflight_angles(out)
        else:preflight(out)
    elif args.worker:worker(out)
    elif args.case_index is not None:solve_case(out,args.case_index)
    else:
        if args.launch_prepared:manifest=read(out/'manifest.json')
        elif args.angles is not None:manifest=prepare_angles(out,args.angles,2. if args.depth is None else args.depth)
        else:manifest=prepare(out,args.pilot_root.resolve(),.5 if args.depth is None else args.depth)
        verify_manifest(manifest)
        if args.prepare_only:print('PREPARED '+str(out))
        else:
            if not args.launch_prepared:
                if manifest.get('kind')=='angle_queue':preflight_angles(out)
                else:preflight(out)
                manifest=read(out/'manifest.json')
            print(json.dumps(launch(out,manifest),indent=2))


if __name__=='__main__':main()
