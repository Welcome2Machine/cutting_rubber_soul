"""Native P1 matched-normal paths and bounded, evidence-gated continuations.

Same intact common material/contact as the original native runs, fresh histories.
P1 is a computational reference, not a claim of converged contact migration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from html import escape

import numpy as np
from prototype.inclined_loading import NativeInclinedPathProblem
from run_support import write_json
from tools import run_angle_path_study as high
from tools import check_native_mesh as runner
from tools.run_material_candidate import signature as material_signature
from tools import run_material_queue as queue
from tools.run_cutting_discretization import verify_case

ROOT = Path(__file__).resolve().parents[1]
CASES = (('B33', 33, 'normal'), ('C33', 33, 'oblique'),
         ('B57', 57, 'normal'), ('C57', 57, 'oblique'))


def make_problem(angle, mode, *, coarse=False, frame_rotation=0., material_batch_size=0):
    p = NativeInclinedPathProblem(specimen_angle=angle, path_mode=mode,
        normal_speed=.1, frame_rotation=frame_rotation,
        maxwell_shear=1.24, relaxation_s=1.7, bulk_modulus=23.34333333333331,
        case='oblique', mu=.35, shear_modulus=.14, ogden_alpha=2., dhat=.0025,
        local_spacing=.25 if coarse else .05, surface_spacing=.025 if coarse else .005,
        nose_facets=64, tip_radius=.02, fine_halfwidth=1. if coarse else 3.,
        max_iterations=500, predictor='secant', hessian_mode='regularized', element='p1')
    p.stagnation_window = 40
    p.material_batch_size = material_batch_size
    return p


def signature(p):
    return dict(material_signature(p), loading_path=p.motion_signature(),
        kinematics='plane_strain', discretization='native-P1-one-history-per-triangle-v1',
        material_batch_size=p.material_batch_size)


def snapshot(out):
    hashes = high.snapshot(out)
    for name in ('tools/run_native_path_study.py','tools/render_solver_dashboard.py'):
        dest = out/'source'/name
        shutil.copy2(ROOT/name, dest)
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    return hashes


def case(args):
    if args.angle not in (33,57) or args.mode not in ('normal','oblique'):
        raise ValueError('This study only authorizes paired 33/57 paths')
    if not (0 < args.depth <= .5*np.cos(np.deg2rad(args.angle))+1e-12):
        raise ValueError('Normal depth exceeds the .5 mm oblique vertical equivalent')
    args.level = 0
    args.radius = .5  # Metadata only; this native element has NO enrichment.
    args.order = 3   # Export compatibility only; P1 uses one exact material point.
    holder = {}
    def factory(angle, level, radius):
        p = make_problem(angle, args.mode,material_batch_size=args.material_batch_size)
        p.ipc.set_num_threads(args.threads)
        holder['problem'] = p
        return p
    r = runner.run(args, problem_factory=factory, geometry_checker=high.geometry_audit,
        source_snapshot=snapshot, problem_signature=signature,
        diagnostic_metadata=dict(name='native-matched-normal-v1', angle=args.angle,
            mode=args.mode, normal_speed=.1, prezero='shared-normal',
            single_material_point_per_triangle=True, no_P2_history_transfer=True))
    if r['steps']:
        high.export_case(args.output, r, problem=holder['problem'])
    return r


def benchmark(out):
    """Interleaved CPU/thread and sparse-order tests on one real saved field.

    Does not integrate a new history or compare failed trial forces. Every
    linear direction is checked against the same current nonsymmetric matrix.
    """
    import cProfile
    import pstats
    import io
    import platform
    import scipy
    from scipy.sparse import diags
    from scipy.sparse.linalg import splu
    from threadpoolctl import threadpool_limits, threadpool_info
    from tools.run_material_candidate import make_problem as vertical
    from prototype.ipc_reference import BASELINE
    from run_support import compute_process_policy
    source=ROOT/'outputs/native_angles_2mm_20261016/02_angle57'
    r=json.loads((source/'result.json').read_text(encoding='utf-8'))
    audit=json.loads((source/'audit.json').read_text(encoding='utf-8'))
    index=next(i for i,x in enumerate(r['steps']) if abs(x['depth_mm']-.5)<1e-12)
    f=source/'states'/f'{index:04d}.npz'
    digest=hashlib.sha256(f.read_bytes()).hexdigest()
    if digest!=audit['rows'][index]['checkpoint_sha256'] or not audit['rows'][index]['passed']:
        raise ValueError('Benchmark checkpoint changed or not audited')
    s=r['settings']
    with compute_process_policy() as policy, threadpool_limits(2):
        p=vertical(**{k:s[k] for k in ('angle','equilibrium_shear','maxwell_shear',
            'relaxation_s','loading_rate','bulk_modulus','mu')})
        with np.load(f) as z:
            state={k:z[k].copy() for k in z.files}
        p.audit_state(state,r['steps'][index])
        p.previous=state['previous_positions'].copy()
        p.previous_material_metric=state['previous_material_metric'].copy()
        p.material_dt=float(state['material_dt_s']); x=state['positions']
        lag=p.lag(x)
        _,g,H,_,metric=p.evaluate(x,lag,projected=False,with_metric=True)
        reference=(g.copy(),H.copy(),metric.copy())
        rows=[]
        # Reverse ordering reduces warm-up/thermal ordering bias; not a CI.
        for n in (1,2,4,8,8,4,2,1):
            with threadpool_limits(n):
                p.ipc.set_num_threads(n)
                t=time.perf_counter(); lag=p.lag(x)
                _,g,H,_,metric=p.evaluate(x,lag,projected=False,with_metric=True)
                A=p.coupled_friction_tangent(x,lag,H)[p.free][:,p.free]
                seconds=time.perf_counter()-t
            differences=[]
            for a,b in zip((g,H,metric),reference):
                d=a-b; differences.append(float(np.max(abs(d.data))) if hasattr(d,'tocsc') and d.nnz else
                    (0. if hasattr(d,'tocsc') else float(np.max(abs(d)))))
            if differences[0]>1e-10 or max(differences[1:])>1e-8:
                raise ValueError('Thread sweep changed assembled equations')
            rows.append(dict(threads=n,assembly_and_friction_jacobian_s=seconds,
                maximum_residual_difference=differences[0],maximum_matrix_difference=max(differences[1:])))
        med={n:float(np.median([z['assembly_and_friction_jacobian_s'] for z in rows if z['threads']==n]))
             for n in (1,2,4,8)}
        best=min(med,key=med.get); chosen=best if med[best]<.9*med[2] else 2
        systems=[]; matrix=A+.1*(metric[p.free][:,p.free]+diags(np.full(len(p.free),BASELINE['hessian_regularization_n_mm'])))
        # Original accepted residual; timing measures factorization, not convergence.
        rhs=-g[p.free]; limit=max(1e-10*np.linalg.norm(rhs),1e-12)
        with threadpool_limits(chosen):
            p.ipc.set_num_threads(chosen)
            for order in ('COLAMD','MMD_AT_PLUS_A','MMD_ATA','MMD_ATA','MMD_AT_PLUS_A','COLAMD'):
                t=time.perf_counter();lu=splu(matrix.tocsc(),permc_spec=order)
                factor_s=time.perf_counter()-t;t=time.perf_counter();du=lu.solve(rhs)
                error=float(np.linalg.norm(matrix@du-rhs))
                systems.append(dict(order=order,factor_s=factor_s,solve_s=time.perf_counter()-t,
                    factor_nonzeros=int(lu.L.nnz+lu.U.nnz),true_residual_n=error,limit_n=limit,passed=error<=limit))
                del lu
            profile=cProfile.Profile();profile.runcall(p.evaluate,x,p.lag(x),projected=False,with_metric=True)
            stream=io.StringIO();pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(25)
    out.mkdir(parents=True,exist_ok=True)
    (out/'assembly_profile.txt').write_text(stream.getvalue(),encoding='utf-8')
    result=dict(scope='same-field timings, not full-path speedup',checkpoint=str(f),checkpoint_sha256=digest,
        source_field_depth_mm=.5,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        threadpools=threadpool_info(),process_policy=policy,thread_rows=rows,thread_medians_s=med,
        selected_threads=chosen,linear_rows=systems,no_new_load_solve=True)
    write_json(out/'benchmark.json',result)
    print(json.dumps(dict(selected_threads=chosen,thread_medians_s=med,linear_rows=systems)))
    return result


def compare_high(candidate, reference, end=.04):
    """Same-history nominal anchors only; absent pole identities remain absent."""
    a,b=Path(candidate),Path(reference)
    ra,rb=queue.read(a/'result.json'),queue.read(b/'result.json')
    r=high.compare_control(b,a,end)
    contacts=[]
    for x in rb['steps']:
        q=x['depth_mm']
        if q>end+1e-12:continue
        y=next((y for y in ra['steps'] if abs(y['depth_mm']-q)<1e-12),None)
        if y is None:continue
        u,v=x['tip_contact'],y['tip_contact']
        contacts.append(dict(q_mm=q,high_status=u['status'],native_status=v['status'],
            identity_difference_mm=(abs(u['reference_offset_mm']-v['reference_offset_mm'])
                if u['status']==v['status']=='unique' else None),
            high_nose_intervals=u.get('nose_reference_intervals_mm'),
            native_nose_intervals=v.get('nose_reference_intervals_mm')))
    # An adaptive cutback changes the integration history, even at shared anchors.
    aq=[x['depth_mm'] for x in ra['steps'] if x['depth_mm']<=end+1e-12]
    bq=[x['depth_mm'] for x in rb['steps'] if x['depth_mm']<=end+1e-12]
    same=len(aq)==len(bq) and np.allclose(aq,bq,atol=1e-12,rtol=0)
    r.update(same_accepted_schedule=bool(same),passed=bool(r['passed'] and same),
        contact_observations=contacts,unique_contact_qualified=False,
        scope='P1 versus local P2 through q=.04 only; no global field convergence or unique-slip qualification')
    return r


def pair_check(b,c):
    """Require same discrete q/time history and the same zero-depth preparation."""
    rb,rc=queue.read(Path(b)/'result.json'),queue.read(Path(c)/'result.json')
    for key in ('config','material','loading_rate','mesh_sha256','contact_formulation',
                'dhat','discretization','material_batch_size'):
        if rb['signature'][key]!=rc['signature'][key]:raise ValueError('Pair mismatch: '+key)
    if rb['settings']['mode']!='normal' or rc['settings']['mode']!='oblique':
        raise ValueError('Pair must be normal / oblique')
    qb=[x['depth_mm'] for x in rb['steps']];qc=[x['depth_mm'] for x in rc['steps']]
    if len(qb)!=len(qc) or not np.allclose(qb,qc,rtol=0,atol=1e-12):
        return dict(passed=False,reason='Different accepted q/time schedules; no silent interpolation')
    np.testing.assert_allclose([x['material']['dt_s'] for x in rb['steps']],
        [x['material']['dt_s'] for x in rc['steps']],rtol=0,atol=1e-12)
    with np.load(Path(b)/'states/0000.npz') as x,np.load(Path(c)/'states/0000.npz') as y:
        np.testing.assert_array_equal(x['reference_xy'],y['reference_xy'])
        dx=float(np.max(abs(x['positions']-y['positions'])))
        dm=float(np.max(abs(x['material_metric']-y['material_metric'])))
    return dict(passed=dx<=2e-10 and dm<=2e-10,zero_position_difference_mm=dx,
        zero_metric_difference=dm,same_accepted_schedule=True)


def depth_gate(out,manifest,state):
    jobs={j['case']:j for j in state['jobs'] if j['stage']=='shallow'}
    checks=[];pairs=[]
    for name,angle,mode in CASES:
        j=jobs[name]
        if not (j.get('mechanics_completed') and j.get('evidence_verified')):
            checks.append(dict(case=name,passed=False,reason='No complete freshly audited .1 path'));continue
        try:
            comparison=compare_high(j['output'],manifest['references'][name])
            cost=j['elapsed_s']+j.get('prior_elapsed_s',0.)<=600.
            checks.append(dict(case=name,passed=bool(comparison['passed'] and cost),
                cost_gate=cost,elapsed_with_prefix_s=j['elapsed_s']+j.get('prior_elapsed_s',0.),
                comparison=comparison))
        except Exception as exc:checks.append(dict(case=name,passed=False,reason=repr(exc)))
    for angle in (33,57):
        try:
            if not all(jobs[k+str(angle)].get('mechanics_completed') for k in ('B','C')):
                raise ValueError('Both complete paths required')
            pairs.append(dict(angle=angle,**pair_check(jobs['B'+str(angle)]['output'],jobs['C'+str(angle)]['output'])))
        except Exception as exc:pairs.append(dict(angle=angle,passed=False,reason=repr(exc)))
    return dict(passed=all(x['passed'] for x in checks+pairs),checks=checks,pairs=pairs,
        force_fraction=.02,force_denominator_floor_n=.1,surface_mm=.002,
        validated_common_q_mm=.04,cost_limit_s_per_shallow_path=600,
        unique_contact_qualified=False,experimental_validation=False)


def prepare(out):
    if (out/'manifest.json').exists():raise RuntimeError('Already prepared; do not overwrite')
    bench=queue.read(out/'batch_benchmark.json');size=bench['selected_batch_size']
    frozen=snapshot(out);inputs={};refs={};jobs=[]
    for name,angle,mode in CASES:
        pilot=out/(name+'_pilot');r=queue.read(pilot/'result.json')
        if not r['completed'] or abs(r['last_depth_mm']-.005)>1e-12:
            raise ValueError('Successful own .005 pilot required: '+name)
        if r['source_sha256']!=frozen or r['settings']['material_batch_size']!=size:
            raise ValueError('Pilot source or batch policy differs')
        inputs.update(verify_case(pilot))
        ref=ROOT/'outputs/angle_paths_20261018'/(name+'_base')
        refs[name]=str(ref);inputs.update(verify_case(ref))
        # Existing time controls are evidence, never restart seeds.
        half=ref.parent/(name+'_halfstep');inputs.update(verify_case(half))
        time_check=high.compare_control(ref,half,end=.04)
        if not time_check['passed']:raise ValueError('Existing shallow time control failed: '+name)
        jobs.append(dict(name=name+'_shallow',case=name,stage='shallow',angle=angle,mode=mode,
            target_mm=.1,increment=.005,solve_seconds=1200.,status='pending',mu=.35,
            resume=str(pilot),output=str(out/(name+'_shallow'))))
    for name,angle,mode in CASES:
        jobs.append(dict(name=name+'_depth',case=name,stage='depth',angle=angle,mode=mode,
            target_mm=float(.5*np.cos(np.deg2rad(angle))),increment=.005,solve_seconds=3600.,
            status='gated',mu=.35,resume=str(out/(name+'_shallow')),output=str(out/(name+'_depth'))))
    for name in ('benchmark.json','batch_benchmark.json','tests.log'):
        f=out/name
        if not f.is_file():raise ValueError('Missing qualification evidence: '+name)
        inputs[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    manifest=dict(version=1,source=str(out/'source'),source_sha256=frozen,input_sha256=inputs,
        references=refs,python=sys.executable,material_batch_size=size,threads=2,max_workers=2,
        runtime_path=os.pathsep.join((str(ROOT/'.runtime/ipc-python'),str(out/'source'))),
        setup_and_audit_allowance_s=300.,no_fracture=True,physical_acceptance_unchanged=True,
        auto_extension='Only four own prefixes after joint shallow force/surface/time/cost gates; max .5 mm oblique lab equivalent')
    write_json(out/'manifest.json',manifest)
    state=dict(status='prepared',jobs=jobs,created_utc=queue.now(),depth_gate=None)
    write_json(out/'queue_status.json',state);report(out,manifest,state)
    return manifest


def job_command(m,j):
    if j['angle'] not in (33,57) or j['target_mm']>.5*np.cos(np.deg2rad(j['angle']))+1e-12:
        raise ValueError('Unapproved angle/depth')
    command=[m['python'],'-X','utf8','-u','-m','tools.run_native_path_study','--phase','case',
        '--output',j['output'],'--angle',str(j['angle']),'--mode',j['mode'],
        '--depth',str(j['target_mm']),'--increment',str(j['increment']),
        '--seconds',str(j['solve_seconds']),'--threads',str(m['threads']),
        '--material-batch-size',str(m['material_batch_size'])]
    if j.get('resume'):command+=['--resume',j['resume']]
    return command


def worker(out):
    import psutil
    m=queue.read(out/'manifest.json');queue.verify_inputs(m)
    for name,h in m['source_sha256'].items():
        if hashlib.sha256((Path(m['source'])/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),started_utc=queue.now()),f)
    state=queue.read(out/'queue_status.json');state.update(status='running',worker_pid=os.getpid())
    env=os.environ.copy();env['PYTHONPATH']=m['runtime_path']
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[key]=str(m['threads'])
    active={};last_html=0.
    try:
        while True:
            for name,(process,j,started,stdout,stderr) in list(active.items()):
                forced=None
                if process.poll() is None and time.monotonic()-started>j['solve_seconds']+m['setup_and_audit_allowance_s']:
                    process.terminate();process.wait(timeout=30);forced='process_wall_timeout'
                if process.poll() is None:continue
                stdout.close();stderr.close()
                j.update(queue.summarize_case(j['output'],j,process.returncode,forced_status=forced))
                j['wall_seconds']=time.monotonic()-started;j['finished_utc']=queue.now()
                try:
                    evidence=verify_case(j['output']);r=queue.read(Path(j['output'])/'result.json')
                    if r['source_sha256']!=m['source_sha256']:raise ValueError('Worker physical source changed')
                    if (r['settings']['mode']!=j['mode'] or r['settings']['increment']!=j['increment']
                            or r['signature']['material_batch_size']!=m['material_batch_size']
                            or r['signature']['loading_path']['normal_speed_mm_s']!=.1):
                        raise ValueError('Requested loading/discretization policy mismatch')
                    j['evidence_verified']=True;j['prior_elapsed_s']=r.get('prior_elapsed_s',0.)
                    j['elapsed_s']=r['elapsed_s']
                    write_json(Path(j['output'])/'evidence_hashes.json',evidence)
                except Exception as exc:j.update(evidence_verified=False,evidence_error=repr(exc))
                del active[name];last_html=0.
            shallow=[j for j in state['jobs'] if j['stage']=='shallow']
            if state['depth_gate'] is None and all(j['status'] not in ('pending','running') for j in shallow):
                state['depth_gate']=depth_gate(out,m,state)
                for j in state['jobs']:
                    if j['stage']=='depth':
                        j['status']='pending' if state['depth_gate']['passed'] else 'skipped_gate_failed'
                last_html=0.
            for j in state['jobs']:
                if len(active)>=m['max_workers']:break
                if j['status']!='pending':continue
                # Never evict unrelated work or saturate memory for throughput.
                if psutil.virtual_memory().available<4*1024**3:
                    if active:break
                    j.update(status='skipped_memory_budget',error='Available RAM below 4 GiB');continue
                command=job_command(m,j);stdout=(out/(j['name']+'.stdout.log')).open('wb')
                stderr=(out/(j['name']+'.stderr.log')).open('wb')
                extra=dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=='nt' else {}
                process=subprocess.Popen(command,cwd=m['source'],env=env,stdin=subprocess.DEVNULL,
                    stdout=stdout,stderr=stderr,**extra)
                j.update(status='running',solver_pid=process.pid,command=command,started_utc=queue.now())
                active[j['name']]=(process,j,time.monotonic(),stdout,stderr);last_html=0.
            write_json(out/'queue_status.json',state)
            if time.monotonic()-last_html>15:
                report(out,m,state);last_html=time.monotonic()
            if not active and all(j['status'] not in ('pending','gated') for j in state['jobs']):break
            time.sleep(2)
        state.update(status='finished',finished_utc=queue.now())
    except Exception as exc:
        state.update(status='worker_failed',error=repr(exc))
        raise
    finally:
        # Only this worker's Popen children; historical PIDs are never targeted.
        for process,j,started,stdout,stderr in active.values():
            if process.poll() is None:process.terminate();process.wait(timeout=30)
            stdout.close();stderr.close()
            j.update(status='interrupted_by_worker',wall_seconds=time.monotonic()-started)
        write_json(out/'queue_status.json',state);report(out,m,state,terminal=True)


def launch(out):
    m=queue.read(out/'manifest.json');queue.verify_inputs(m)
    if (out/'launch.json').exists() or (out/'worker_started.json').exists():raise RuntimeError('Already launched')
    for name,h in m['source_sha256'].items():
        if hashlib.sha256((Path(m['source'])/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
    cmd=[m['python'],'-X','utf8','-u','-m','tools.run_native_path_study','--phase','worker','--output',str(out)]
    env=os.environ.copy();env['PYTHONPATH']=m['runtime_path'];env['OPENBLAS_NUM_THREADS']='2';env['OMP_NUM_THREADS']='2'
    extra={}
    if os.name=='nt':
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=subprocess.SW_HIDE
        extra=dict(startupinfo=startup,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        p=subprocess.Popen(cmd,cwd=m['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            start_new_session=os.name!='nt',**extra)
    row=dict(worker_pid=p.pid,command=cmd,started_utc=queue.now());write_json(out/'launch.json',row)
    print(json.dumps(row))


def report(out,m,state,terminal=False):
    """Offline HTML status and complete raw-log access, refreshed by the worker."""
    from tools.render_solver_dashboard import shell,atomic_write,stamp,tail
    body=f'<header><h1>原生P1成对路径与效率验证</h1><p>{stamp()} · {escape(state["status"])} · 最多两组并行</p></header>'
    body+='<div class="notice">q是参考法向深度，不是实验室下行深度。匹配法向速度0.1 mm/s及共同预备历史；与原实验竖直速度0.2 mm/s不同，不能当作新实验拟合曲线。无断裂、无按角度调参。P1作为计算参照，局部接触迁移仍未获得精度认证。</div>'
    body+='<section><h2>运行状态</h2><table><tr><th>路径</th><th>状态</th><th>已接受q / mm</th><th>态数</th><th>Fz / N</th><th>求解记录与限制</th></tr>'
    for j in state['jobs']:
        folder=Path(j['output']);r=queue.read(folder/'result.json') if (folder/'result.json').is_file() else {}
        rows=r.get('steps',[]);last=rows[-1] if rows else {};name=j['name']
        body+=f'<tr><td>{name}</td><td>{escape(j["status"])}</td><td>{last.get("depth_mm","—")}</td><td>{len(rows)}</td><td>{last.get("reaction_fz_n","—")}</td><td>预算{j["solve_seconds"]:.0f}s + 300s建模/审计<br><a href="{name}_log.html">HTML日志</a></td></tr>'
        if (folder/'run.log').exists():
            # Bounded live tail; preserve the full original file and wrap all at terminal.
            log=(folder/'run.log').read_text(encoding='utf-8') if j['status']!='running' else tail(folder/'run.log',100)
            page=f'<h1>{name} 日志</h1><p>运行时展示末100行，结束后展示完整日志。<a href="{name}/run.log">原始日志</a></p><pre>{escape(log)}</pre><h2>标准错误</h2><pre>{escape(tail(out/(name+".stderr.log")))}</pre>'
            atomic_write(out/(name+'_log.html'),shell(name,page,refresh=j['status']=='running'))
    body+='</table></section>'
    gate=state.get('depth_gate')
    body+='<section><h2>加深门槛</h2><p>四条路径全部完成q=.1并通过fresh审计；与原高阶路径共同q=0–.04的反力分量差≤2%、整顶面位移差≤2 μm；同角度B/C实际时间表一致，零态位置与材料历史一致；每条浅层累计记录耗时≤600 s。全部通过才续各自前缀，最多到斜向竖直行程.5 mm对应的q。任一门槛失败，四项加深全部跳过，不加预算。</p>'
    if gate:
        body+=f'<p><strong>联合门槛：{"通过" if gate["passed"] else "未通过；不启动加深"}</strong></p><table><tr><th>路径</th><th>力分量差</th><th>顶面差 / μm</th><th>累计记录秒</th><th>结论</th></tr>'
        for c in gate['checks']:
            comp=c.get('comparison',{});f=comp.get('max_force_fraction');u=comp.get('max_surface_mm')
            body+=f'<tr><td>{c["case"]}</td><td>{f"{100*f:.3f}%" if f is not None else "—"}</td><td>{f"{1000*u:.3f}" if u is not None else "—"}</td><td>{c.get("elapsed_with_prefix_s","—")}</td><td>{c["passed"]} {escape(c.get("reason",""))}</td></tr>'
        body+='</table><p>接触身份／鼻部邻近区间见机器报告；缺失不是零。共同浅层比较不证明深部空间收敛。</p>'
    body+='</section>'
    bench=queue.read(out/'batch_benchmark.json') if (out/'batch_benchmark.json').is_file() else {}
    body+='<section><h2>本轮效率选择</h2><p>同一已接受57°真实场测试：1/2/4/8线程约0.78–0.80 s，保留两线程；SuperLU原COLAMD分解约0.214 s，替代MMD_AT_PLUS_A约9.43 s，未采用。新材料分批不改本构或残差限，在三个准确历史场上核对力、切线、内部变量和耗散。</p>'
    body+=f'<p>选用每批{m["material_batch_size"]}个材料点；配对装配中位秒：{escape(str(bench.get("medians_s",{})))}。这是固定场测量，整程收益取决于迭代次数，不能外推。P1保留原接触边界与体网格，移除额外内部P2自由度／积分点；这种降阶必须经过上面的误差门槛。</p>'
    body+='<p>CPU：i9-13900HX，24核/32逻辑处理器；内存31.72 GiB；GPU：RTX 5070 Ti Laptop、12 GiB。GPU加速不覆盖当前CPU接触、回退和全部材料核，暂未接入生产。最多两进程，每进程BLAS/IPC两线程，余量低于4 GiB不新开组。</p>'
    body+='<p><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html">SciPy/SuperLU排序与CSC接口</a> · <a href="https://docs.nvidia.com/cuda/cudss/latest/index.html">cuDSS GPU稀疏直接求解</a> · <a href="https://petsc.org/release/manual/snes/">PETSc非线性求解框架</a>。这些是候选计算组件，不能替代本任务的接触、历史及物理验收；本轮未迁移到新FE库。</p></section>'
    body+='<section><h2>证据与复现</h2><p><a href="tests.html">测试日志</a> · <a href="benchmark.json">线程与稀疏库实测</a> · <a href="batch_benchmark.json">材料分批实测</a> · <a href="study_report.json">对照与时间口径</a> · <a href="manifest.json">冻结源码与输入</a></p></section>'
    # Pair mechanism data only from exact matching accepted anchors. Histories are
    # checked separately; no interpolation or extrapolated completion is allowed.
    pairs=[]
    for stage in ('shallow','depth'):
        for angle in (33,57):
            paths=[out/f'{k}{angle}_{stage}' for k in ('B','C')]
            if not all((p/'result.json').exists() for p in paths):continue
            rb,rc=[queue.read(p/'result.json') for p in paths];rows=[]
            for b in rb['steps']:
                c=next((c for c in rc['steps'] if abs(c['depth_mm']-b['depth_mm'])<1e-12),None)
                if c is None:continue
                rows.append(dict(q_mm=b['depth_mm'],normal_only_N=b['motion']['normal_reaction_n'],
                    oblique_N=c['motion']['normal_reaction_n'],normal_only_T=b['motion']['tangent_resistance_n'],
                    oblique_T=c['motion']['tangent_resistance_n'],normal_only_Fz=b['reaction_fz_n'],
                    oblique_Fz=c['reaction_fz_n'],b_minJ=b['min_j'],c_minJ=c['min_j']))
            pairs.append(dict(angle=angle,stage=stage,rows=rows,complete=bool(rb['completed'] and rc['completed']),
                schedule_check=pair_check(*paths)))
    write_json(out/'study_report.json',dict(status=state['status'],gate=gate,pairs=pairs,
        experimental_validation=False,contact_migration_qualified=False,
        time_scope='elapsed_s includes setup/resume/solve; prior_elapsed_s prefix added; terminal audit/export additional'))
    if terminal and pairs:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with plt.rc_context({'font.size':10}):
            fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
            for pair in pairs:
                if pair['stage']!='shallow' or not pair['rows']:continue
                col=0 if pair['angle']==33 else 1;rows=pair['rows'];q=[x['q_mm'] for x in rows]
                for ax,component in zip(axs[:,col],('N','T')):
                    for prefix,label,style in [('normal_only','Normal path','--'),('oblique','Oblique path','-')]:
                        ax.plot(q,[x[prefix+'_'+component] for x in rows],style,label=label)
                    ax.set(xlabel='Reference normal depth q (mm)',ylabel=component+' (N)',title=str(pair['angle'])+' deg')
                    ax.grid(alpha=.2);ax.legend()
            fig.savefig(out/'paired_forces.png',dpi=160);plt.close(fig)
        body+='<section><h2>共同接受范围的法向／切向反力</h2><img src="paired_forces.png" alt="33和57度的匹配法向路径反力；只显示共同接受前缀" style="max-width:100%"><p>不延长失败曲线；该图不代表实验拟合或局部场收敛。</p></section>'
    atomic_write(out/'index.html',shell('原生P1成对路径',body,refresh=state['status']=='running'))
    if (out/'tests.log').exists():
        atomic_write(out/'tests.html',shell('回归测试','<h1>测试原始记录</h1><pre>'+escape((out/'tests.log').read_text(encoding='utf-8'))+'</pre>'))


def batch_benchmark(out):
    """Benchmark optional constitutive batches on original exact-history fields."""
    from threadpoolctl import threadpool_limits
    from tools.run_material_candidate import make_problem as vertical
    from run_support import compute_process_policy
    source=ROOT/'outputs/native_angles_2mm_20261016/02_angle57'
    r=json.loads((source/'result.json').read_text(encoding='utf-8'))
    audits=json.loads((source/'audit.json').read_text(encoding='utf-8'))
    s=r['settings'];rows=[];inputs={}
    with compute_process_policy(),threadpool_limits(2):
        p=vertical(**{k:s[k] for k in ('angle','equilibrium_shear','maxwell_shear',
            'relaxation_s','loading_rate','bulk_modulus','mu')});p.ipc.set_num_threads(2)
        for depth in (.05,.5,1.):
            i=next(i for i,x in enumerate(r['steps']) if abs(x['depth_mm']-depth)<1e-12)
            f=source/'states'/f'{i:04d}.npz'; h=hashlib.sha256(f.read_bytes()).hexdigest()
            if h!=audits['rows'][i]['checkpoint_sha256']:raise ValueError('Changed saved field')
            inputs[str(f)]=h
            with np.load(f) as z:
                x=z['positions'].copy();p.previous=z['previous_positions'].copy()
                p.previous_material_metric=z['previous_material_metric'].copy();p.material_dt=float(z['material_dt_s'])
            p.material_batch_size=0;ref=p.material_response(x,tangent=True);lag=p.lag(x)
            baseline=p.evaluate(x,lag,projected=False,with_metric=True)
            for size in (0,2048,8192,8192,2048,0):
                p.material_batch_size=size;t=time.perf_counter()
                result=p.evaluate(x,lag,projected=False,with_metric=True);elapsed=time.perf_counter()-t
                q=p.material_response(x,tangent=True)
                errors={key:float(np.max(abs(q[key]-ref[key]))/max(1.,np.max(abs(ref[key]))))
                    for key in ('P','tangent','metric','viscous_rate','incremental_potential')}
                gd=float(np.linalg.norm(result[1]-baseline[1]))
                hd=float(np.max(abs((result[2]-baseline[2]).data))) if (result[2]-baseline[2]).nnz else 0.
                ok=max(errors.values())<=1e-8 and gd<=1e-7 and hd<=1e-5
                rows.append(dict(depth_mm=depth,batch_size=size,paired_assembly_s=elapsed,
                    normalized_material_errors=errors,force_difference_n=gd,tangent_max_difference=hd,passed=bool(ok)))
    med={size:float(np.median([z['paired_assembly_s'] for z in rows if z['batch_size']==size])) for size in (0,2048,8192)}
    best=min(med,key=med.get)
    chosen=best if med[best]<.9*med[0] and all(z['passed'] for z in rows if z['batch_size']==best) else 0
    result=dict(scope='same exact-history field; not whole-path speedup',input_sha256=inputs,rows=rows,
        medians_s=med,selected_batch_size=chosen,physical_acceptance_unchanged=True)
    write_json(out/'batch_benchmark.json',result);print(json.dumps(dict(medians_s=med,selected_batch_size=chosen,all_passed=all(x['passed'] for x in rows))))
    return result


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--phase', choices=('case','benchmark','batch-benchmark','prepare','launch','worker','report'), default='case')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--angle', type=float, choices=(33.,57.), default=57.)
    ap.add_argument('--mode', choices=('normal','oblique'), default='oblique')
    ap.add_argument('--depth', type=float, default=.005)
    ap.add_argument('--increment', type=float, default=.005)
    ap.add_argument('--seconds', type=float, default=300.)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--material-batch-size',type=int,default=0)
    a = ap.parse_args(); a.output = a.output.resolve()
    if not all(np.isfinite(x) and x > 0 for x in (a.depth,a.increment,a.seconds)) or a.threads<1:
        ap.error('Positive finite depth, increment, budget and thread count required')
    if a.phase=='benchmark': benchmark(a.output)
    elif a.phase=='batch-benchmark': batch_benchmark(a.output)
    elif a.phase=='prepare':prepare(a.output)
    elif a.phase=='launch':launch(a.output)
    elif a.phase=='worker':worker(a.output)
    elif a.phase=='report':report(a.output,queue.read(a.output/'manifest.json'),queue.read(a.output/'queue_status.json'))
    else: case(a)


if __name__ == '__main__': main()
