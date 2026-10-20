"""Bounded AL contact qualification; HTML and exact same-path continuation.

This runner never revives an old IPC history. The first candidate retains the
local P2 bulk. Native IPC pole proximity is NOT exported as measured AL contact.
"""
import argparse
import contextlib
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from prototype.augmented_blade import AugmentedBladeProblem
from prototype.ipc_reference import adaptive_load_steps
from run_support import write_json,save_npz,source_hashes,compute_process_policy
from tools.run_material_candidate import audit,signature
from tools.render_solver_dashboard import shell,atomic_write,stamp

ROOT=Path(__file__).resolve().parents[1]


def make_problem(angle,*,coarse=False,spacing=.00125,material_order=3,material_refinement=0,surface_spacing=.005):
    return AugmentedBladeProblem(specimen_angle=angle,path_mode='oblique',normal_speed=.1,
        maxwell_shear=1.24,relaxation_s=1.7,bulk_modulus=23.34333333333331,
        case='oblique',mu=.35,shear_modulus=.14,ogden_alpha=2.,dhat=.0025,
        local_spacing=.25 if coarse else .05,surface_spacing=surface_spacing,
        nose_facets=64,tip_radius=.02,fine_halfwidth=1. if coarse else 3.,
        max_iterations=500,predictor='affine',hessian_mode='regularized',element='p1',
        kinematics='plane_strain',enrichment_radius=.5,quadrature_order=material_order,
        quadrature_refinement=material_refinement,
        contact_spacing=spacing)


def snapshot(out):
    hashes=source_hashes(packages=('prototype','mechanics'),snapshot=out/'source')
    for name in ('tools/run_augmented_blade.py','tools/run_material_candidate.py',
                 'tools/render_solver_dashboard.py','prototype/ipc_baseline.json'):
        dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dst);hashes[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
    for name in ('tools/run_al_depth.py','tools/run_al_performance.py','tools/profile_al_resources.py',
                 'tools/run_al_integration.py','tools/replay_continuous_contact.py',
                 'tools/run_separated_contact.py','tools/run_geometric_contact.py'):
        if (ROOT/name).exists():
            dst=out/'source'/name;shutil.copy2(ROOT/name,dst);hashes[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
    return hashes


def restore_prefix(p,source,out,report):
    """Resume this exact AL path; no mesh, material or contact history transfer."""
    source=Path(source).resolve();old=json.loads((source/'result.json').read_text(encoding='utf-8'))
    if old['signature']!=json.loads(json.dumps(report['signature'])):
        raise ValueError('AL resume requires the exact physical/discretization signature')
    changed={k for k in set(old['source_sha256'])|set(report['source_sha256'])
             if old['source_sha256'].get(k)!=report['source_sha256'].get(k)}
    if changed-{'tools/run_augmented_blade.py','tools/run_al_depth.py'}:
        raise ValueError('AL resume physical source changed: '+str(changed))
    if not old.get('completed') or not old['steps']:raise ValueError('Require a completed qualified prefix')
    evidence=verify_case(source,require_independent=True)
    print(f'RESUME_AUDIT source={source} states={len(old["steps"])}',flush=True)
    audit(p,source,old,destination=out/'resume_audit.json')
    report.update(resumed_from=str(source),resume_input_sha256=evidence,
        resume_changed_runner_files=sorted(changed),steps=old['steps'].copy(),
        rejected_load_attempts=old['rejected_load_attempts'].copy(),
        prior_elapsed_s=old.get('prior_elapsed_s',0.)+old.get('elapsed_s',0.))
    for i in range(len(old['steps'])):
        f=source/'states'/f'{i:04d}.npz';shutil.copy2(f,out/'states'/f.name)
    with np.load(source/'states'/f'{len(old["steps"])-1:04d}.npz') as s:
        x=s['positions'].copy();previous=s['previous_positions'].copy()
        p.material_metric=s['material_metric'].copy()
        p.previous_material_metric=s['previous_material_metric'].copy()
        p.material_dt=float(s['material_dt_s'])
        p.material_dissipation=float(s['cumulative_viscous_dissipation_n_mm'])
        p.previous=previous.copy()
        p._set_quadrature(s['al_edges'],s['al_fractions'],s['al_weights'])
        p.multipliers=s['al_multipliers'].copy()
    last=old['steps'][-1]['depth_mm']
    history=([(old['steps'][-2]['depth_mm'],previous)] if len(old['steps'])>1 else [])
    p.seed_predictor_history(history+[(last,x)])
    report['last_depth_mm']=last
    print(f'RESUME_READY q_mm={last:.9g} states={len(old["steps"])} exact_dual_history=True',flush=True)
    return x,last


def report_html(out,r):
    steps=r.get('steps',[])
    basis=r.get('settings',{}).get('contact_basis','P0')
    description=('连续P1节点乘子，采用正形函数与参考面积加权间隙／滑移' if basis=='P1'
                 else '每条原始边一个P0乘子，采用边平均间隙／滑移')
    rows=''.join('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in
        (z['depth_mm'],z['reaction_fz_n'],z['residual_n'],z['min_j'],
         z['al_geometry']['penetration_bound_mm']*1000,z['al_constraints']['integrated_projection_defect_n']))+'</tr>' for z in steps)
    body=f'<header><h1>新面接触／增广拉格朗日验证</h1><p>{stamp()} · {escape(r["status"])}</p></header>'
    lab_speed=r.get('settings',{}).get('lab_speed')
    path=('q为参考法向深度，速度0.1 mm/s；不是实验竖直下行深度。' if lab_speed is None
          else f'q为法向深度，d=q/cosθ；首触后实验室竖直速度{lab_speed} mm/s。首触前仍为法向无载接近。')
    self_policy=('仅限未出现非相邻自接触；排除同一边端点排斥，遇其他自接触立即停止，全部自身CCD保留。'
                 if r.get('settings',{}).get('self_contact_policy')=='separated-only-v1' else '材料自接触采用原IPC。')
    if r.get('settings',{}).get('self_contact_policy','').startswith('oriented-gcp2'):
        self_policy='材料采用定向GCP法向自接触及本项目增量正则库仑自摩擦，完整自身CCD保留；固定加载步，失败停止，不自动缩步。'
    body+=f'<div class="notice">独立候选，不是原IPC的继续运行。共同材料与局部P2不变。{path}{self_policy}刀具允许核验的有限穿透≤0.2 μm，材料正J与自接触CCD保留。接触方程改变，不能只凭与旧力接近认证。</div>'
    body+=f'<section><h2>终态／当前接受前缀</h2><p>目标q={r["target_mm"]} mm；状态 {escape(r["status"])}；记录耗时 {r.get("elapsed_s","—")} s；完整完成 {r.get("completed",False)}。</p><p>{escape(r.get("error", ""))}</p><p><a href="log.html">完整HTML日志</a> · <a href="result.json">参数、历史与逐态记录</a> · <a href="audit.json">重新装配审计</a> · <a href="force_motion.csv">力与实际位移</a></p></section>'
    body+=f'<section><h2>平衡与弱积分接触约束分别验收</h2><p>力残差≤1e−4 N；积分牵引投影缺陷≤5e−5 N；摩擦锥、非负总耗散、整边间隙、固定底面及刀具位置独立检查。{description}；这不是逐点互补证明。位移、法向与切向乘子联立求解，材料历史只在接受后提交。没有唯一刀尖身份的资格结果；所有顶面材料节点轨迹另存。</p><table><tr><th>q/mm</th><th>Fz/N</th><th>残差/N</th><th>minJ</th><th>穿透上界/μm</th><th>接触缺陷/N</th></tr>'+rows+'</table></section>'
    if r.get('audit_progress'):
        ap=r['audit_progress']
        body+=f'<section><h2>独立复核进度</h2><p>{ap["accepted_states"]}/{ap["total_states"]} 状态；{ap["elapsed_s"]:.2f} s。求解、复核分别计时。每个状态仍检查两套整边间隙界和准确双历史。</p></section>'
    atomic_write(out/'index.html',shell('AL刀具接触验证',body))
    if (out/'run.log').exists():
        atomic_write(out/'log.html',shell('AL运行日志','<h1>新接触候选运行日志</h1><a href="index.html">返回状态</a><pre>'+escape((out/'run.log').read_text(encoding='utf-8',errors='replace'))+'</pre>'))


def fixed_load_step(p,x,old,target,deadline,on_reject):
    """One physical increment, no load bisection or alternative retries."""
    p.attempted_depth=float(target);p.failed_previous=x.copy();tick=time.monotonic()
    try:value,record=p.step(x.copy(),old,target,deadline)
    except Exception as exc:
        on_reject(dict(previous_depth_mm=old,target_depth_mm=target,increment_mm=target-old,
            policy='fixed-no-cutback',elapsed_s=time.monotonic()-tick,error=repr(exc)))
        raise
    record['load_step']=dict(previous_depth_mm=old,increment_mm=target-old,
        nominal_target_mm=target,subdivision_level=0,policy='fixed-no-cutback')
    yield value,record


def run(args,*,problem_factory=None,audit_factory=None):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'states').mkdir()
    r=dict(status='initializing',completed=False,target_mm=args.depth,steps=[],
           rejected_load_attempts=[],source_sha256=snapshot(out),
           settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    write_json(out/'result.json',r);report_html(out,r);p=None;x=None;began=time.monotonic()
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(getattr(args,'blas_threads',2)),compute_process_policy() as policy:
        r['process_policy']=policy
        try:
            p=(problem_factory or make_problem)(args.angle,coarse=args.coarse,spacing=args.spacing);p.ipc.set_num_threads(getattr(args,'ipc_threads',2))
            r['compute_settings']=getattr(p,'compute_settings',{'backend':'original'})
            r['signature']=dict(signature(p),motion=p.motion_signature(),
                contact_spacing_mm=p.contact_spacing,penalty_n_mm3=p.penalty,
                penetration_tolerance_mm=p.penetration_tolerance,
                local_P2_radius_mm=.5,material_order=p.enrichment.order)
            if hasattr(p,'geometric_self'):r['signature']['geometric_self_contact']=p.geometric_self.spec
            if p.enrichment.quadrature_refinement:
                r['signature']['material_quadrature_refinement']=p.enrichment.quadrature_refinement
            x=p.rest.copy();old=p.start
            if getattr(args,'resume',None):
                r['status']='restoring_and_auditing_prefix';write_json(out/'result.json',r);report_html(out,r)
                x,old=restore_prefix(p,args.resume,out,r)
            if args.depth<=old+1e-12:raise ValueError('Continuation target must exceed accepted depth')
            # Setup/resume audit is reported separately from the solve budget.
            r['setup_elapsed_s']=time.monotonic()-began;deadline=time.monotonic()+args.seconds
            r['status']='running'
            def reject(event):
                r['rejected_load_attempts'].append(event);write_json(out/'result.json',r)
            targets=np.r_[0.,np.arange(args.increment,args.depth-1e-12,args.increment),args.depth]
            for target in targets[targets>old+1e-12]:
                print(f'LOAD_TARGET q_mm={target:.9g} previous_q_mm={old:.9g}',flush=True)
                attempts=(fixed_load_step(p,x,old,float(target),deadline,reject)
                    if getattr(args,'fixed_load_steps',False) else
                    adaptive_load_steps(p,x,old,float(target),deadline,.000625,reject))
                for x,record in attempts:
                    old=record['depth_mm'];index=len(r['steps'])
                    save_npz(out/'states'/f'{index:04d}.npz',reference_xy=p.rest,
                        triangles=p.mesh.triangles,positions=x,previous_positions=p.previous,
                        depth_mm=old,body_nodes=p.n,**p.checkpoint_data())
                    r['steps'].append(record);r['last_depth_mm']=old;r['elapsed_s']=time.monotonic()-began
                    write_json(out/'result.json',r);report_html(out,r)
                    print(f'ACCEPT q_mm={old:.9g} Fz_N={record["reaction_fz_n"]:.9g} residual_N={record["residual_n"]:.9g}',flush=True)
            r.update(status='completed',completed=True)
        except Exception as exc:
            r.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',
                     error=f'{type(exc).__name__}: {exc}',completed=False)
            traceback.print_exc(file=log)
            if p is not None and p.material_dt is not None:
                rejected=getattr(p,'rejected_al_state',None)
                if rejected is None:
                    rejected=dict(positions=getattr(p,'trial',x),previous_positions=p.previous,**p.checkpoint_data())
                save_npz(out/'failed_trial.npz',**rejected)
        r['elapsed_s']=time.monotonic()-began
        r['solve_elapsed_s']=r['elapsed_s']-r.get('setup_elapsed_s',0.)
        if p is not None and r['steps']:
            final_status=r['status'];audit_began=time.monotonic()
            r['status']='auditing';r['solve_status']=final_status
            write_json(out/'result.json',r);report_html(out,r)
            def progress(row,count,total):
                r['audit_progress']=dict(accepted_states=count,total_states=total,
                    last_q_mm=row['depth_mm'],elapsed_s=time.monotonic()-audit_began)
                write_json(out/'result.json',r)
                if count%5==0 or count==total:report_html(out,r)
            try:
                auditor=audit_factory(args.angle,coarse=args.coarse,spacing=args.spacing) if audit_factory else p
                auditor.ipc.set_num_threads(getattr(args,'ipc_threads',2))
                audit(auditor,out,r,progress=progress,deadline=audit_began+getattr(args,'audit_seconds',900.))
                r['status']=final_status
                if audit_factory:r['original_kernel_audit_passed']=True
            except Exception as exc:
                r.update(status='audit_failed',completed=False,audit_error=repr(exc));traceback.print_exc(file=log)
            r['audit_elapsed_s']=time.monotonic()-audit_began
        r['total_elapsed_s']=time.monotonic()-began
    if p is not None and r['steps']:
        nodes=np.flatnonzero(np.isclose(p.mesh.local_sn[:p.base_nodes,1],p.config.height_mm,atol=1e-10))
        with (out/'surface_material_motion.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['q_mm','reference_node_id','reference_s_offset_mm','u_t_mm','u_n_mm','x_mm','y_mm'])
            for i,z in enumerate(r['steps']):
                with np.load(out/'states'/f'{i:04d}.npz') as state:
                    xy=state['positions'][nodes];u=xy-p.rest[nodes]
                    w.writerows((z['depth_mm'],int(n),p.mesh.local_sn[n,0]-p.config.drop_s_mm,
                        float(du@p.es),float(du@p.en),*v) for n,du,v in zip(nodes,u,xy))
    with (out/'force_motion.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['normal_depth_mm','laboratory_downward_mm','fz_n','rx_n','normal_reaction_n','tangent_resistance_n'])
        for z in r['steps']:
            m=z['motion'];w.writerow([z['depth_mm'],-m['laboratory_translation_mm'][1],z['reaction_fz_n'],z['reaction_x_n'],m['normal_reaction_n'],m['tangent_resistance_n']])
    write_json(out/'result.json',r);write_json(out/'outcome.json',{k:r.get(k) for k in ('status','completed','last_depth_mm','elapsed_s','error','audit_error')})
    report_html(out,r)
    print(json.dumps({k:r.get(k) for k in ('status','last_depth_mm','elapsed_s','error','audit_error')}))
    return r


def verify_case(folder,*,require_independent=False):
    folder=Path(folder);r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    a=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
    if not a['passed'] or a['states']!=len(r['steps']):raise ValueError('Incomplete AL audit')
    files={}
    for name,digest in r['source_sha256'].items():
        f=folder/'source'/name
        if hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Changed source '+name)
        files[str(f)]=digest
    for i,row in enumerate(a['rows']):
        f=folder/'states'/f'{i:04d}.npz';digest=hashlib.sha256(f.read_bytes()).hexdigest()
        if not row['passed'] or digest!=row['checkpoint_sha256']:raise ValueError('Changed accepted state')
        if require_independent:
            for key in ('geometry','geometry_lipschitz'):
                geometry=row[key]
                if (geometry['penetration_bound_mm']>.0002 or
                    geometry['minimum_signed_gap_lower_mm']>geometry['sampled_minimum_gap_upper_mm']+1e-12):
                    raise ValueError('Invalid independent geometry audit')
        files[str(f)]=digest
    for name in ('result.json','audit.json'):
        f=folder/name;files[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    return files


def compare(first,second,end):
    a,b=map(Path,(first,second));ra,rb=[json.loads((x/'result.json').read_text(encoding='utf-8')) for x in (a,b)]
    for key in ('config','material','loading_rate','contact_formulation','motion','mesh_sha256'):
        if ra['signature'][key]!=rb['signature'][key]:raise ValueError('AL comparison physical mismatch: '+key)
    theta=np.deg2rad(ra['settings']['angle']);en=np.array([np.sin(theta),np.cos(theta)])
    with np.load(a/'states/0000.npz') as s:
        n=int(s['base_body_nodes']);ref=s['reference_xy'][:n]
        top=np.flatnonzero(np.isclose(ref@en,ra['signature']['config']['height_mm'],atol=1e-10))
    rows=[]
    for i,x in enumerate(ra['steps']):
        if x['depth_mm']>end+1e-12:continue
        match=next(((j,y) for j,y in enumerate(rb['steps']) if abs(y['depth_mm']-x['depth_mm'])<1e-12),None)
        if match is None:continue
        j,y=match
        with np.load(a/'states'/f'{i:04d}.npz') as u,np.load(b/'states'/f'{j:04d}.npz') as v:
            np.testing.assert_array_equal(u['reference_xy'],v['reference_xy'])
            error=float(np.linalg.norm(u['positions'][top]-v['positions'][top],axis=1).max())
        differences={k:abs(x['motion'][k]-y['motion'][k])/max(.1,abs(x['motion'][k]))
            for k in ('normal_reaction_n','tangent_resistance_n','generalized_resistance_n')}
        differences['fz_n']=abs(x['reaction_fz_n']-y['reaction_fz_n'])/max(.1,abs(x['reaction_fz_n']))
        rows.append(dict(q_mm=x['depth_mm'],force_fractions=differences,whole_top_max_difference_mm=error))
    cover=bool(rows and abs(rows[0]['q_mm'])<1e-12 and abs(rows[-1]['q_mm']-end)<1e-12
        and len(rows)==sum(z['depth_mm']<=end+1e-12 for z in ra['steps']))
    maximum=max((max(z['force_fractions'].values()) for z in rows),default=float('inf'))
    surface=max((z['whole_top_max_difference_mm'] for z in rows),default=float('inf'))
    return dict(passed=bool(cover and maximum<=.02 and surface<=.002),complete_comparison=cover,
        max_force_fraction=maximum,max_surface_mm=surface,rows=rows,
        scope='Same candidate integration/time control only; not contact-space, unique slip or experimental validation')


def queue_report(out,state):
    precision=state.get('study')=='precision01'
    body=f'<header><h1>新接触路线：完整浅层验证</h1><p>{stamp()} · {escape(state["status"])}</p></header>'
    if precision:
        body+='<div class="notice">本批仅57°：加密积分和步长减半均从静止独立运行到法向q=.1 mm（竖直约.183608 mm）。复用经修正算法重新组装核查的已完成基础路径，不续接其历史。原600 s成本门槛未过的事实保留；本次精度控制各1800 s，另300 s建模／审计。没有自动加深1／2 mm。</div>'
        body+='<p><a href="../index.html">修复、预检和基准重新审查</a> · <a href="../reference_audit.json">基准修正后审计</a></p>'
    else:
        body+='<div class="notice">只运行这四项独立路径，不续接旧历史、不自动延伸1／2 mm。局部P2与共同材料保留，刚性刀具改为弱积分接触。先前逐积分点增广预检失败记录保留；联立弱积分方案通过两项0.005 mm预检，不代表完整路径已经成功。</div>'
    body+='<section><h2>串行队列与HTML日志</h2><table><tr><th>案例</th><th>状态</th><th>目标q/mm</th><th>接受q/mm</th><th>预算/s</th></tr>'
    for job in state['jobs']:
        folder=out/job['name'];result=json.loads((folder/'result.json').read_text(encoding='utf-8')) if (folder/'result.json').exists() else {}
        link=f'<a href="{job["name"]}/index.html">{job["name"]}</a>' if result else job['name']
        body+=f'<tr><td>{link}</td><td>{escape(job["status"])}</td><td>{job["depth"]}</td><td>{result.get("last_depth_mm","—")}</td><td>{job["seconds"]}+{300 if precision else 180}审计</td></tr>'
        if result:report_html(folder,result)
    criteria=('两项完整q=.1并逐态重新装配通过；与已完成基础路径在全部共同锚点比较，力分量差≤2%、整顶面运动差≤2 μm。实际时间步／缩步另行记录；独立Lipschitz整边核查每个保存态。' if precision else
        'C57完整q=.1并鲜算通过，记录耗时≤600 s；独立积分／半步到q=.02，力分量差≤2%、整顶面位移差≤2 μm；C33短路径到q=.02。')
    body+='</table></section><section><h2>预声明判据</h2><p>'+criteria+'整边穿透≤0.2 μm、力≤1e−4 N、积分投影缺陷≤5e−5 N及耗散／历史均需通过。这不是接触乘子空间、唯一滑移点或实验验证。</p><p>每项成功、失败或超时均推进其余独立项。完成后停止，不增加预算。<a href="manifest.json">冻结清单</a> · <a href="queue_status.json">机器状态</a> · <a href="analysis.json">自动终态分析</a></p></section>'
    if state.get('analysis'):
        if precision:
            a=state['analysis']
            body+='<section><h2>完整0–0.1 mm精度结论</h2><p>精度门槛：<strong>'+('通过' if a['passed'] else '未通过／未完成')+'</strong>。原基础路径600 s成本门槛仍未通过，不以精度控制预算覆盖原记录。</p><table><tr><th>对照</th><th>共同区间完整</th><th>最大力分量差/%</th><th>整顶面差/μm</th><th>判据</th></tr>'
            for c in a['controls']:
                body+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in (c['case'],c.get('complete_comparison',False),100*c.get('max_force_fraction',float('nan')),1000*c.get('max_surface_mm',float('nan')),c['passed']))+'</tr>'
            body+='</table><p>终态也保留失败前缀与原始日志；通过不代表实验曲率、33°偏差或唯一接触点已经解决。</p></section>'
        body+='<section><h2>终态结论</h2><pre>'+escape(json.dumps(state['analysis'],ensure_ascii=False,indent=2))+'</pre></section>'
    atomic_write(out/'index.html',shell('AL浅层验证队列',body,refresh=state['status']=='running'))


def worker(out):
    m=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid()},f)
    for name,digest in m['source_sha256'].items():
        if hashlib.sha256((Path(m['source'])/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source differs')
    for name,digest in m.get('inputs_sha256',{}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed qualification input')
    precision=m.get('study')=='precision01'
    state=dict(status='running',jobs=m['jobs'],study=m.get('study','initial'));write_json(out/'queue_status.json',state)
    env=os.environ.copy();env.update(PYTHONPATH=m['runtime_path'],OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    for job in state['jobs']:
        cmd=[sys.executable,'-X','utf8','-u','-m','tools.run_augmented_blade','--phase','run',
             '--output',str(out/job['name']),'--angle',str(job['angle']),'--depth',str(job['depth']),
             '--increment',str(job['increment']),'--spacing',str(job['spacing']),'--seconds',str(job['seconds'])]
        extra=dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=='nt' else {}
        with (out/(job['name']+'.stdout.log')).open('wb') as stdout,(out/(job['name']+'.stderr.log')).open('wb') as stderr:
            p=subprocess.Popen(cmd,cwd=m['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**extra)
            job.update(status='running',pid=p.pid,command=cmd,started=stamp());began=time.monotonic()
            write_json(out/'queue_status.json',state);queue_report(out,state)
            forced=False
            while p.poll() is None:
                if time.monotonic()-began>job['seconds']+(300 if precision else 180):
                    p.terminate();p.wait(timeout=30);forced=True;break
                queue_report(out,state);time.sleep(10)
        job.update(status='process_timeout' if forced else 'failed',returncode=p.returncode,wall_seconds=time.monotonic()-began)
        try:
            folder=out/job['name'];r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            if r['source_sha256']!=m['source_sha256']:raise ValueError('Case source differs from frozen queue')
            job['evidence']=verify_case(folder,require_independent=precision);job['audit_verified']=True
            if not forced:job['status']=r['status']
            job.update(completed=bool(r['completed'] and not forced),elapsed_s=r['elapsed_s'],last_depth_mm=r.get('last_depth_mm'))
        except Exception as exc:job.update(audit_verified=False,verification_error=repr(exc))
        write_json(out/'queue_status.json',state);queue_report(out,state)
    checks=[]
    reference=Path(m['reference']) if precision else out/'C57_base'
    for name in ('C57_quadrature','C57_halfstep'):
        try:
            for path,digest in m.get('inputs_sha256',{}).items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed qualification input')
            checks.append(dict(case=name,**compare(reference,out/name,.1 if precision else .02)))
        except Exception as exc:checks.append(dict(case=name,passed=False,error=repr(exc)))
    basic=all(j.get('completed') and j.get('audit_verified') for j in state['jobs'])
    cost=(m['reference_elapsed_s'] if precision else state['jobs'][0].get('elapsed_s',float('inf')))<=600
    state['analysis']=dict(passed=bool(basic and (precision or cost) and all(z['passed'] for z in checks)),
        all_cases_completed_and_audited=bool(basic),c57_cost_gate=cost,controls=checks,
        gate_scope='0--.1 mm time/integration accuracy only' if precision else 'initial joint cost/accuracy',
        no_automatic_extension=True,experimental_validation=False,unique_contact_migration_qualified=False)
    if precision:
        schedules={}
        for path in (reference,*(out/n for n in ('C57_quadrature','C57_halfstep'))):
            try:
                r=json.loads((path/'result.json').read_text(encoding='utf-8'))
                schedules[path.name]=dict(increments_mm=[s['load_step']['increment_mm'] for s in r['steps'] if 'load_step' in s],
                    dt_s=[s['material']['dt_s'] for s in r['steps']],rejections=len(r['rejected_load_attempts']))
            except Exception as exc:schedules[path.name]=dict(error=repr(exc))
        state['analysis']['actual_schedules']=schedules
    state.update(status='finished',finished=stamp());write_json(out/'analysis.json',state['analysis'])
    write_json(out/'queue_status.json',state);queue_report(out,state)


def prepare_precision(out,reference):
    """Reaudit the completed reference and run actual fine-mesh short pilots.

    This phase writes only NEW evidence; no reference states or histories are
    overwritten. Only the bound/checking implementation and this runner may
    differ from the old frozen reference sources.
    """
    out.mkdir(parents=True,exist_ok=False);current=snapshot(out)
    gate=dict(study='precision01',passed=False,current_sources=current,
              reference=str(reference),evidence_sha256={},audits=[],pilots=[])
    old=json.loads((reference/'result.json').read_text(encoding='utf-8'))
    if not (old['completed'] and abs(old['last_depth_mm']-.1)<1e-12 and old['settings']['angle']==57):
        raise ValueError('A completed 57-degree q=.1 reference is required')
    gate['reference_elapsed_s']=old['elapsed_s']
    def page():
        body='<h1>修复整边检查，补齐57°法向0.1 mm精度控制</h1><p>'+stamp()+'</p>'
        body+='<div class="notice">物理参数、局部P2与边P0接触乘子不变。支撑线最小值检查加入两端点；终态另用独立1-Lipschitz界复核。旧600 s成本门槛未过的事实保留。本次只验证时间和积分精度，未证明接触空间或实验收敛。</div>'
        body+='<section><h2>准备与重新审查</h2><p>准备门槛：'+('通过' if gate['passed'] else '进行中／未通过')+'</p><pre>'+escape(json.dumps({k:v for k,v in gate.items() if k not in ('current_sources','evidence_sha256')},ensure_ascii=False,indent=2))+'</pre></section>'
        body+='<section><h2>本批计划</h2><p>独立积分长度.000625 mm、独立步长.0025 mm，两条路径均从静止到q=.1（57°竖直.183608 mm）。每项1800 s求解，另300 s建模／审计，串行两线程。与重新审查通过的旧21态基础路径逐共同锚点比较：反力分量差≤2%、整顶面差≤2 μm；独立几何门槛仍≤.2 μm、力残差仍≤1e−4 N。</p><p>各项成功、失败、超时均继续下一项，结束后停止；不自动加深1／2 mm。</p><a href="qualification/index.html">后台队列HTML与终态对照</a> · <a href="../al_contact_20261019/terminal_review.html">前批结果与错误定位</a></section>'
        atomic_write(out/'index.html',shell('接触检查修复及0.1 mm精度验证',body))
        write_json(out/'preflight_verification.json',gate)
    page()
    try:
        with threadpool_limits(2),compute_process_policy():
            for folder in sorted(reference.parent.glob('C*')):
                if not (folder/'result.json').is_file():continue
                r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
                changed={n for n in set(current)|set(r['source_sha256']) if current.get(n)!=r['source_sha256'].get(n)}
                if changed-{'prototype/augmented_blade.py','tools/run_augmented_blade.py'}:
                    raise ValueError('Unqualified physical source changes: '+str(changed))
                gate['evidence_sha256'].update(verify_case(folder))
                p=make_problem(r['settings']['angle'],spacing=r['settings']['spacing']);p.ipc.set_num_threads(2)
                # Stored JSON converts material tuples to arrays. Compare the
                # same serialized representation, not Python container types.
                for key,value in json.loads(json.dumps(signature(p))).items():
                    if r['signature'][key]!=value:raise ValueError('Reference physical signature mismatch: '+key)
                destination=out/('reference_audit.json' if folder==reference else folder.name+'_reaudit.json')
                fresh=audit(p,folder,r,destination=destination)
                gate['evidence_sha256'][str(destination)]=hashlib.sha256(destination.read_bytes()).hexdigest()
                gate['audits'].append(dict(case=folder.name,states=fresh['states'],passed=fresh['passed'],
                    corrected_penetration_max_um=1000*max(z['geometry']['penetration_bound_mm'] for z in fresh['rows']),
                    independent_penetration_bound_max_um=1000*max(z['geometry_lipschitz']['penetration_bound_mm'] for z in fresh['rows'])))
                print('REAUDIT',folder.name,fresh['states'],'passed',flush=True);page()
            for name,increment,spacing in (('pilot_quadrature',.005,.000625),('pilot_halfstep',.0025,.00125)):
                args=argparse.Namespace(output=out/name,phase='run',study='precision01',angle=57.,
                    depth=.005,increment=increment,seconds=180.,spacing=spacing,coarse=False)
                r=run(args)
                if not r['completed']:raise ValueError('Actual runner preflight failed: '+name)
                gate['evidence_sha256'].update(verify_case(out/name,require_independent=True))
                if r['source_sha256']!=current:raise ValueError('Preflight evaluator changed')
                check=compare(reference,out/name,.005)
                gate['pilots'].append(dict(case=name,states=len(r['steps']),elapsed_s=r['elapsed_s'],comparison=check))
                if not check['passed']:raise ValueError('Preflight/reference force or motion gate failed')
                page()
        for name,digest in current.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source changed during preflight')
        gate['passed']=True;page()
    except Exception as exc:
        gate.update(error=repr(exc));page();raise
    return gate


def launch(out,study='initial'):
    preflight=out.parent/'preflight_verification.json'
    gate=json.loads(preflight.read_text(encoding='utf-8'))
    if not gate.get('passed'):raise ValueError('Preflight qualification has not passed')
    out.mkdir(parents=True,exist_ok=False);hashes=snapshot(out)
    if hashes!=gate['current_sources']:raise ValueError('Source changed after final preflight audit')
    if study=='precision01' and gate.get('study')!=study:raise ValueError('Wrong precision preflight')
    specifications=(('C57_quadrature',57,.1,.005,.000625,1800),('C57_halfstep',57,.1,.0025,.00125,1800)) if study=='precision01' else (
            ('C57_base',57,.1,.005,.00125,600),('C57_quadrature',57,.02,.005,.000625,240),
            ('C57_halfstep',57,.02,.0025,.00125,240),('C33_short',33,.02,.005,.00125,240))
    jobs=[dict(name=n,angle=a,depth=d,increment=inc,spacing=spacing,seconds=seconds,status='pending')
        for n,a,d,inc,spacing,seconds in specifications]
    m=dict(source=str(out/'source'),source_sha256=hashes,jobs=jobs,study=study,
           runtime_path=str(ROOT/'.runtime/ipc-python')+os.pathsep+str(out/'source'),
           preflight_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest())
    if study=='precision01':
        m.update(reference=gate['reference'],reference_elapsed_s=gate['reference_elapsed_s'],
            inputs_sha256=dict(gate['evidence_sha256'],**{str(preflight):hashlib.sha256(preflight.read_bytes()).hexdigest()}))
        for path,digest in m['inputs_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed precision input')
    write_json(out/'manifest.json',m);state=dict(status='prepared',jobs=jobs,study=study)
    write_json(out/'queue_status.json',state);queue_report(out,state)
    env=os.environ.copy();env.update(PYTHONPATH=m['runtime_path'],OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    cmd=[sys.executable,'-X','utf8','-u','-m','tools.run_augmented_blade','--phase','worker','--output',str(out)]
    extra={}
    if os.name=='nt':
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=subprocess.SW_HIDE
        extra=dict(startupinfo=startup,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        p=subprocess.Popen(cmd,cwd=m['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                           start_new_session=os.name!='nt',**extra)
    record=dict(pid=p.pid,command=cmd,started=stamp());write_json(out/'launch.json',record);print(json.dumps(record))


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',required=True,type=Path)
    a.add_argument('--phase',choices=('run','launch','worker','prepare'),default='run')
    a.add_argument('--study',choices=('initial','precision01'),default='initial')
    a.add_argument('--reference',type=Path)
    a.add_argument('--resume',type=Path)
    a.add_argument('--angle',type=float,choices=(33.,57.),default=57.)
    a.add_argument('--depth',type=float,default=.005);a.add_argument('--increment',type=float,default=.005)
    a.add_argument('--seconds',type=float,default=180.);a.add_argument('--spacing',type=float,default=.00125)
    a.add_argument('--coarse',action='store_true');args=a.parse_args()
    if not (0<args.depth<=.25 and 0<args.increment<=.005 and 0<args.seconds<=3600 and 0<args.spacing<=.0025):
        a.error('Bounded AL diagnostic requires q<=.25, increment<=.005, budget<=3600, spacing<=.0025')
    if args.phase=='prepare':
        if args.study!='precision01' or args.reference is None:a.error('Precision preparation requires --reference')
        prepare_precision(args.output.resolve(),args.reference.resolve())
    elif args.phase=='launch':launch(args.output.resolve(),args.study)
    elif args.phase=='worker':worker(args.output.resolve())
    else:run(args)


if __name__=='__main__':main()
