"""Bounded 57-degree AL continuation and conditional contact-space checks.

The production body/contact equations remain in augmented_blade. The split
multiplier variant below is ONLY a fixed-previous-history diagnostic. It never
seeds a production continuation or claims a newly integrated loading history.
"""
import argparse
import contextlib
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from scipy.sparse import coo_matrix,diags
from threadpoolctl import threadpool_limits
from tools import run_augmented_blade as base
from tools.run_material_candidate import load_force_csv
from tools.render_solver_dashboard import shell,atomic_write,stamp
from run_support import write_json,save_npz,compute_process_policy


class SplitContactProblem(base.AugmentedBladeProblem):
    """Two P0 pieces per original edge, split ONLY at existing Gauss-cell ends.

    Bulk, boundary interpolation, normals, quadrature locations and weights are
    identical. If an edge has only one cell it is left unsplit. More multiplier
    constraints may be rank deficient: bounded failure is reported, not cured
    by relaxing physical or geometric acceptance.
    """
    def _set_quadrature(self,edges,fractions,weights):
        super()._set_quadrature(edges,fractions,weights)
        labels=2*self.qedges.copy()
        for e in np.unique(self.qedges):
            ids=np.flatnonzero(self.qedges==e);cells=len(ids)//3
            if cells>=2:labels[ids]+=self.qfraction[ids]>=(cells//2)/cells
        self.contact_edges,inverse=np.unique(labels,return_inverse=True)
        self.edgeQ=coo_matrix((np.ones(len(labels)),(np.arange(len(labels)),inverse)),
            shape=(len(labels),len(self.contact_edges))).tocsr()
        self.edge_area=np.asarray(self.edgeQ.T@self.qweights).ravel()
        self.average=diags(1/self.edge_area)@self.edgeQ.T@diags(self.qweights)


def split_problem(**kwargs):
    # Factory parameters are identical; the diagnostic overrides only the
    # multiplier aggregation on the already qualified boundary quadrature.
    p=base.make_problem(**kwargs)
    p.__class__=SplitContactProblem
    p.contact_formulation='weak-P0-subedge-conditional-v1'
    p._set_quadrature(p.qedges,p.qfraction,p.qweights)
    return p


def probe(out,source,depth,seconds=180):
    out.mkdir(parents=True,exist_ok=False);(out/'states').mkdir()
    result=dict(status='initializing',completed=False,conditional=True,target_mm=depth,steps=[],
        source_sha256=base.snapshot(out),scope='One increment with exact fixed previous material/friction histories; not a full refined path')
    write_json(out/'result.json',result);began=time.monotonic()
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(2),compute_process_policy() as policy:
        result['process_policy']=policy
        try:
            parent=json.loads((source/'result.json').read_text(encoding='utf-8'))
            candidates=[(i,r) for i,r in enumerate(parent['steps']) if abs(r['depth_mm']-depth)<1e-12]
            if not candidates:
                result['status']='skipped_missing_checkpoint'
            else:
                i,old=candidates[0];evidence=base.verify_case(source,require_independent=True)
                result['reference_evidence_sha256']=evidence
                result['reference']=dict(folder=str(source),index=i,depth_mm=depth)
                p=split_problem(angle=parent['settings']['angle'],coarse=parent['settings']['coarse'],spacing=parent['settings']['spacing'])
                p.ipc.set_num_threads(2)
                for key,value in json.loads(json.dumps(base.signature(p))).items():
                    if key!='contact_formulation' and parent['signature'][key]!=value:
                        raise ValueError('Conditional check physical signature differs: '+key)
                if parent['signature']['contact_formulation']!='weak-P0-AL-rigid-blade-v2':
                    raise ValueError('Conditional check needs the qualified original P0 path')
                changed={n for n in set(parent['source_sha256'])|set(result['source_sha256'])
                    if parent['source_sha256'].get(n)!=result['source_sha256'].get(n)}
                if changed-{'tools/run_augmented_blade.py','tools/run_al_depth.py'}:
                    raise ValueError('Conditional physical sources differ')
                with np.load(source/'states'/f'{i:04d}.npz') as z:s={k:z[k].copy() for k in z.files}
                np.testing.assert_array_equal(s['reference_xy'],p.rest)
                p._set_quadrature(s['al_edges'],s['al_fractions'],s['al_weights'])
                original=np.unique(s['al_edges']);mapping={int(e):j for j,e in enumerate(original)}
                p.multipliers=np.array([s['al_multipliers'][mapping[int(label)//2]] for label in p.contact_edges])
                p.previous=s['previous_positions'].copy()
                p.previous_material_metric=s['previous_material_metric'].copy()
                p.material_metric=s['previous_material_metric'].copy();p.material_dt=float(s['material_dt_s'])
                held=(p.previous.copy(),p.previous_material_metric.copy(),p.material_dt)
                x=s['positions'].copy();result['original_multiplier_pairs']=len(original)
                result['refined_multiplier_pairs']=len(p.multipliers)
                result['initial_projection_defect_n']=p.constraint_audit(x)[1]['integrated_projection_defect_n']
                x,it=p._equilibrate(x,time.monotonic()+seconds)
                np.testing.assert_array_equal(p.previous,held[0]);np.testing.assert_array_equal(p.previous_material_metric,held[1])
                assert p.material_dt==held[2]
                record=p.acceptance(x,depth,it,1)
                p.material_metric=p.material_response(x)['metric'].copy()
                p.material_dissipation=(float(s['cumulative_viscous_dissipation_n_mm'])-
                    old['material']['viscous_dissipation_increment_n_mm']+record['material']['viscous_dissipation_increment_n_mm'])
                record['material']['cumulative_viscous_dissipation_n_mm']=p.material_dissipation
                data=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,
                    previous_positions=p.previous,depth_mm=depth,body_nodes=p.n,**p.checkpoint_data())
                save_npz(out/'states/0000.npz',**data)
                fresh=p.audit_state(data,record)
                fresh['checkpoint_sha256']=hashlib.sha256((out/'states/0000.npz').read_bytes()).hexdigest()
                write_json(out/'audit.json',dict(passed=True,states=1,rows=[fresh],scope=result['scope']))
                top=np.flatnonzero(np.isclose(p.mesh.local_sn[:p.base_nodes,1],p.config.height_mm,atol=1e-10))
                differences={k:abs(record['motion'][k]-old['motion'][k])/max(.1,abs(old['motion'][k]))
                    for k in ('normal_reaction_n','tangent_resistance_n','generalized_resistance_n')}
                differences['fz_n']=abs(record['reaction_fz_n']-old['reaction_fz_n'])/max(.1,abs(old['reaction_fz_n']))
                surface=float(np.linalg.norm(x[top]-s['positions'][top],axis=1).max())
                result.update(status='completed',completed=True,steps=[record],last_depth_mm=depth,
                    force_fractions=differences,whole_top_max_difference_mm=surface,
                    conditional_screen_passed=bool(max(differences.values())<=.02 and surface<=.002))
        except Exception as exc:
            result.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',error=repr(exc))
            traceback.print_exc(file=log)
        result['elapsed_s']=time.monotonic()-began
    write_json(out/'result.json',result);probe_html(out,result)
    return result


def probe_html(out,r):
    body='<h1>接触乘子空间条件检查</h1><div class="notice">只改变同一积分布局上的乘子分段，固定准确前步材料与位置历史。不是完整加载路径，也不提供可接续的生产历史。失败或超时没有可比较的平衡力。</div>'
    body+='<p>'+escape(r['status'])+'</p><pre>'+escape(json.dumps({k:v for k,v in r.items() if k not in ('source_sha256','reference_evidence_sha256','steps')},ensure_ascii=False,indent=2))+'</pre>'
    if (out/'run.log').exists():
        atomic_write(out/'log.html',shell('接触空间检查日志','<pre>'+escape((out/'run.log').read_text(encoding='utf-8'))+'</pre>'))
    body+='<a href="log.html">HTML日志</a> · <a href="result.json">完整记录</a>'
    atomic_write(out/'index.html',shell('接触空间条件检查',body))


def observe(source):
    """Pressure-weighted contact REGION, not unique tip material identity."""
    r=json.loads((source/'result.json').read_text(encoding='utf-8'))
    p=base.make_problem(r['settings']['angle'],coarse=r['settings']['coarse'],spacing=r['settings']['spacing'])
    np.testing.assert_allclose(p.rest[:p.base_nodes]@p.es,p.mesh.local_sn[:p.base_nodes,0],rtol=0,atol=1e-10)
    rows=[]
    for i,rec in enumerate(r['steps']):
        with np.load(source/'states'/f'{i:04d}.npz') as s:
            p._set_quadrature(s['al_edges'],s['al_fractions'],s['al_weights'])
            p.multipliers=s['al_multipliers'].copy();p.previous=s['previous_positions']
            x=s['positions'];c=p.contact_data(x)
        positive=p.multipliers[:,0]>1e-6
        original=p.body_edges[p.contact_edges[positive]]
        region_s=(p.rest[original]@p.es-p.config.drop_s_mm).ravel()
        weights=np.maximum(c['pressure'],0)*p.qweights
        ref_s=(p.Q@p.rest)@p.es-p.config.drop_s_mm
        norm=float(weights.sum())
        rows.append(dict(q_mm=rec['depth_mm'],laboratory_depth_mm=-rec['motion']['laboratory_translation_mm'][1],
            contact_reference_min_mm=float(region_s.min()) if len(region_s) else None,
            contact_reference_max_mm=float(region_s.max()) if len(region_s) else None,
            pressure_weighted_reference_centroid_mm=float(weights@ref_s/norm) if norm>1e-10 else None,
            pressure_weighted_abs_incremental_slip_mm=float(weights@abs(c['slip'])/norm) if norm>1e-10 else None,
            active_original_edges=int(positive.sum())))
    with (source/'contact_region.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows


def analyze(out,experiment,*,source=None):
    display_preflight=source is not None
    out.mkdir(parents=True,exist_ok=True)
    source=source or out/'C57_depth';r=json.loads((source/'result.json').read_text(encoding='utf-8'))
    evidence=base.verify_case(source,require_independent=True)
    region=observe(source);ed,ef,rate=load_force_csv(experiment)
    steps=r['steps'];q=np.array([s['depth_mm'] for s in steps]);z=np.array([-s['motion']['laboratory_translation_mm'][1] for s in steps]);f=np.array([s['reaction_fz_n'] for s in steps])
    anchors=[]
    for target in (0.,.05,.1,.15,.2,.25):
        ids=np.flatnonzero(abs(q-target)<1e-12)
        if len(ids):
            i=int(ids[0]);s=steps[i]
            anchors.append(dict(q_mm=target,vertical_mm=z[i],Fz_n=f[i],experiment_raw_n=float(np.interp(z[i],ed,ef)),
                normal_reaction_n=s['motion']['normal_reaction_n'],tangent_resistance_n=s['motion']['tangent_resistance_n'],
                min_j=s['min_j'],iterations=s['iterations']))
    secants=[dict(from_q_mm=a['q_mm'],to_q_mm=b['q_mm'],FE_n_per_vertical_mm=(b['Fz_n']-a['Fz_n'])/(b['vertical_mm']-a['vertical_mm']),
        experiment_n_per_vertical_mm=(b['experiment_raw_n']-a['experiment_raw_n'])/(b['vertical_mm']-a['vertical_mm'])) for a,b in zip(anchors,anchors[1:])]
    probes=[]
    for name in ('space_at_010','space_at_025'):
        p=out/name/'result.json'
        if p.exists():
            t=json.loads(p.read_text(encoding='utf-8'))
            if t.get('completed'):evidence.update(base.verify_case(p.parent,require_independent=True))
            probes.append({k:t.get(k) for k in ('status','target_mm','completed','conditional_screen_passed','force_fractions','whole_top_max_difference_mm','error')})
    summary=dict(status=r['status'],completed=r['completed'],source_target_q_mm=r['target_mm'],
        display_preflight_only=display_preflight,last_q_mm=float(q[-1]),
        last_vertical_mm=float(z[-1]),audited_states=len(steps),max_residual_n=max(s['residual_n'] for s in steps),
        path_min_j=min(s['min_j'] for s in steps),recorded_continuation_s=r['elapsed_s'],
        setup_s=r.get('setup_elapsed_s'),solve_s=r.get('solve_elapsed_s'),prior_s=r.get('prior_elapsed_s'),
        rejected_load_attempts=len(r['rejected_load_attempts']),anchors=anchors,secants=secants,
        contact_space_probes=probes,experiment_speed_mm_s=rate,fe_vertical_speed_mm_s=.1/np.cos(np.deg2rad(57)),
        raw_experiment_initial_force_n=float(ef[0]),experiment_sha256=hashlib.sha256(experiment.read_bytes()).hexdigest(),
        evidence_sha256=evidence,experimental_validation=False,unique_tip_migration_qualified=False,
        conditional_contact_space_is_not_full_path_convergence=True)
    write_json(out/'analysis.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':10,'axes.spines.right':False,'axes.spines.top':False}):
        fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        ax[0,0].plot(z,f,'o-',ms=3,color='#0072B2',label='FE: accepted states')
        use=ed<=z[-1];ax[0,0].plot(ed[use],ef[use],'--',color='#555555',label='Experiment: context only')
        ax[0,0].set(xlabel='Laboratory depth (mm)',ylabel='Fz (N)',title='Force vs actual downward depth',ylim=(0,None));ax[0,0].legend()
        ax[0,1].plot(q,[s['motion']['normal_reaction_n'] for s in steps],'-',label='N: total normal reaction')
        ax[0,1].plot(q,[s['motion']['tangent_resistance_n'] for s in steps],'--',label='T: total tangential resistance')
        ax[0,1].set(xlabel='Reference-normal depth q (mm)',ylabel='Reaction (N)',title='Specimen-frame reaction components');ax[0,1].legend()
        lo=np.array([s['contact_reference_min_mm'] if s['contact_reference_min_mm'] is not None else np.nan for s in region]);hi=np.array([s['contact_reference_max_mm'] if s['contact_reference_max_mm'] is not None else np.nan for s in region])
        centroid=[s['pressure_weighted_reference_centroid_mm'] if s['pressure_weighted_reference_centroid_mm'] is not None else np.nan for s in region]
        ax[1,0].fill_between(q,lo,hi,color='#0072B2',alpha=.2,label='Positive-pressure reference region')
        ax[1,0].plot(q,centroid,'--',color='#0072B2',label='Pressure-weighted reference centroid')
        ax[1,0].set(xlabel='Reference-normal depth q (mm)',ylabel='Original surface coordinate s (mm)',title='Contact REGION, not unique tip slip');ax[1,0].legend(fontsize=8)
        ax[1,1].plot(q,[s['min_j'] for s in steps],'o-',ms=3,color='#D55E00')
        ax[1,1].set(xlabel='Reference-normal depth q (mm)',ylabel='Local minimum J',title='Local compression along accepted path')
        for a in ax.flat:a.grid(alpha=.2)
        fig.suptitle('57° intact AL depth study: accepted prefix only')
        fig.savefig(out/'mechanics.png',dpi=150);plt.close(fig)
    body='<h1>57°加深试验：力学趋势与接触区域</h1><p>'+stamp()+'</p>'
    if display_preflight:body+='<div class="notice"><strong>本页仅为已完成0.105 mm续接预检的展示检查，不是0.25 mm运行结果。</strong></div>'
    body+='<div class="notice">只展示通过审查的接受前缀。q不是竖直下压深度；FE竖直速度约.1836 mm/s，实验.2 mm/s，原始.51 N初值保留。实验只作背景对照，不拟合、不平移深度、不重复换算传感器系数。N/T是相对初始材料表面的总反力分解，不是独立摩擦因果贡献。</div>'
    body+='<p>状态：'+escape(r['status'])+'；法向q='+str(q[-1])+' mm，竖直约'+f'{z[-1]:.6f}'+' mm；'+str(len(steps))+'态已审查。</p><img src="mechanics.png" style="max-width:100%;height:auto" alt="接受前缀的力与实验背景、材料坐标反力、受压接触参考区域、局部J；未绘制未完成区间。">'
    body+='<p>接触区域阈值为边压力&gt;1e−6 MPa；压力加权中心不是唯一刀尖材料点。增量滑移在当前切向上计算，不能当作同一材料点的累计滑程。完整顶面节点运动和受压区域CSV均保留。图中FE点直接连线，不作平滑；缺少接触的零态留空。实验锚点仅在原始深度范围内线性插值。</p>'
    body+='<h2>分段刚度（N/mm，竖直位移）</h2><table><tr><th>q区间/mm</th><th>FE</th><th>实验背景</th></tr>'
    for row in secants:body+=f'<tr><td>{row["from_q_mm"]}–{row["to_q_mm"]}</td><td>{row["FE_n_per_vertical_mm"]:.4f}</td><td>{row["experiment_n_per_vertical_mm"]:.4f}</td></tr>'
    body+='</table><h2>接触空间检查与范围限制</h2><p>子边乘子检查固定前一步准确历史，只是条件敏感性。若失败／超时，不报告候选平衡力；即使通过也不证明完整加载历史的接触空间收敛。本轮完成后停止，不自动加深1／2 mm。</p><pre>'+escape(json.dumps({k:v for k,v in summary.items() if k not in ('evidence_sha256','anchors','secants')},ensure_ascii=False,indent=2))+'</pre>'
    relative=os.path.relpath(source,out).replace(os.sep,'/')
    body+='<p><a href="'+relative+'/index.html">算例与HTML日志</a> · <a href="'+relative+'/force_motion.csv">力与位移CSV</a> · <a href="'+relative+'/surface_material_motion.csv">顶面材料节点运动</a> · <a href="'+relative+'/contact_region.csv">接触区域与增量滑移CSV</a> · <a href="analysis.json">机器证据</a></p>'
    atomic_write(out/'analysis.html',shell('57°加深结果分析',body))
    return summary


def dashboard(out,state):
    body='<h1>57°继续到法向0.25 mm</h1><p>'+stamp()+' · '+escape(state['status'])+'</p>'
    body+='<div class="notice">只续自己的积分加密双历史；法向q=.25对应竖直约.459 mm。主线目标是更深的力曲率、表面运动及效率；两个子边乘子检查只是固定前步历史的条件对照。主线不改材料／网格／摩擦／刀具／边界／验收。检查失败会记录并推进下一项；不自动延伸1／2 mm。</div><table><tr><th>任务</th><th>状态</th><th>接受q/mm</th><th>计算预算/s</th></tr>'
    for j in state['jobs']:
        folder=out/j['name'];r=json.loads((folder/'result.json').read_text(encoding='utf-8')) if (folder/'result.json').exists() else {}
        if r:
            if j['kind']=='run':base.report_html(folder,r)
            else:probe_html(folder,r)
        link=f'<a href="{j["name"]}/index.html">{j["name"]}</a>' if r else j['name']
        body+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in (link,j['status'],r.get('last_depth_mm','—'),j['seconds']))+'</tr>'
    body+='</table><p>主线另600 s建模／审计；条件检查各另120 s。q=.15/.25锚点与全前缀均自动保存，终态重新组装审查和实验背景对比。半径／材料参数是候选模型假设，本轮不标定。</p><p><a href="analysis.html">终态力学分析（完成后生成）</a> · <a href="manifest.json">冻结清单</a> · <a href="queue_status.json">机器状态</a> · <a href="../preflight_verification.json">实际续接预检</a></p>'
    if state.get('analysis_error'):body+='<pre>'+escape(state['analysis_error'])+'</pre>'
    atomic_write(out/'index.html',shell('AL有界深度试验',body,refresh=state['status']=='running'))


def prepare(out,reference):
    out.mkdir(parents=True,exist_ok=False);sources=base.snapshot(out)
    evidence=base.verify_case(reference,require_independent=True)
    # Actual continuation goes .1 -> .105 and audits the whole inherited prefix.
    args=argparse.Namespace(output=out/'pilot',angle=57.,coarse=False,spacing=.000625,
        depth=.105,increment=.005,seconds=180.,resume=reference,phase='run')
    result=base.run(args)
    gate=dict(passed=bool(result['completed']),reference=str(reference),pilot=str(args.output),
        current_sources=sources,evidence_sha256=evidence,pilot_elapsed_s=result['elapsed_s'])
    if gate['passed']:
        gate['evidence_sha256'].update(base.verify_case(args.output,require_independent=True))
        if result['source_sha256']!=sources:raise ValueError('Preflight source changed')
        analyze(out/'display_preflight',base.ROOT/'docs/1014n57a1__Fz__curve_0.csv',source=args.output)
        gate['display_preflight_passed']=True
    for name,digest in sources.items():
        if hashlib.sha256((base.ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Working source changed')
    write_json(out/'preflight_verification.json',gate)
    return gate


def launch(out):
    preflight=out.parent/'preflight_verification.json';g=json.loads(preflight.read_text(encoding='utf-8'))
    if not g['passed']:raise ValueError('Continuation preflight failed')
    out.mkdir(parents=True,exist_ok=False);sources=base.snapshot(out)
    if sources!=g['current_sources']:raise ValueError('Changed source after preflight')
    experiment=(base.ROOT/'docs/1014n57a1__Fz__curve_0.csv').resolve()
    inputs=dict(g['evidence_sha256']);inputs[str(preflight)]=hashlib.sha256(preflight.read_bytes()).hexdigest()
    inputs[str(experiment)]=hashlib.sha256(experiment.read_bytes()).hexdigest()
    jobs=[dict(name='space_at_010',kind='probe',source=g['reference'],depth=.1,seconds=180,status='pending'),
          dict(name='C57_depth',kind='run',source=g['pilot'],depth=.25,seconds=3600,status='pending'),
          dict(name='space_at_025',kind='probe',source=str(out/'C57_depth'),depth=.25,seconds=180,status='pending')]
    manifest=dict(source=str(out/'source'),source_sha256=sources,inputs_sha256=inputs,jobs=jobs,experiment=str(experiment),
        runtime_path=str(base.ROOT/'.runtime/ipc-python')+os.pathsep+str(out/'source'))
    for path,digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed input')
    write_json(out/'manifest.json',manifest);state=dict(status='prepared',jobs=jobs);write_json(out/'queue_status.json',state);dashboard(out,state)
    env=os.environ.copy();env.update(PYTHONPATH=manifest['runtime_path'],OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    cmd=[sys.executable,'-X','utf8','-u','-m','tools.run_al_depth','--phase','worker','--output',str(out)]
    extra={}
    if os.name=='nt':
        si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=subprocess.SW_HIDE
        extra=dict(startupinfo=si,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        p=subprocess.Popen(cmd,cwd=manifest['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=os.name!='nt',**extra)
    record=dict(pid=p.pid,command=cmd,started=stamp());write_json(out/'launch.json',record);print(json.dumps(record))


def worker(out):
    m=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid()),f)
    for path,digest in {**{str(Path(m['source'])/k):v for k,v in m['source_sha256'].items()},**m['inputs_sha256']}.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source/input mismatch')
    env=os.environ.copy();env.update(PYTHONPATH=m['runtime_path'],OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    state=dict(status='running',jobs=m['jobs'])
    for j in state['jobs']:
        if j['kind']=='run':
            cmd=[sys.executable,'-X','utf8','-u','-m','tools.run_augmented_blade','--phase','run',
                 '--resume',j['source'],'--angle','57','--spacing','.000625','--increment','.005']
        else:cmd=[sys.executable,'-X','utf8','-u','-m','tools.run_al_depth','--phase','probe','--reference',j['source']]
        cmd+=['--output',str(out/j['name']),'--depth',str(j['depth']),'--seconds',str(j['seconds'])]
        with (out/(j['name']+'.stdout.log')).open('wb') as stdout,(out/(j['name']+'.stderr.log')).open('wb') as stderr:
            p=subprocess.Popen(cmd,cwd=m['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                **(dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=='nt' else {}))
            j.update(status='running',pid=p.pid,command=cmd,started=stamp());began=time.monotonic();forced=False
            write_json(out/'queue_status.json',state);dashboard(out,state)
            while p.poll() is None:
                if time.monotonic()-began>j['seconds']+(600 if j['kind']=='run' else 120):
                    p.terminate();p.wait(timeout=30);forced=True;break
                dashboard(out,state);time.sleep(10)
        j.update(status='process_timeout' if forced else 'failed',wall_seconds=time.monotonic()-began,returncode=p.returncode)
        try:
            folder=out/j['name'];r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            if r['source_sha256']!=m['source_sha256']:raise ValueError('Case source mismatch')
            if r.get('steps'):j['verified_evidence']=base.verify_case(folder,require_independent=True)
            if not forced:j['status']=r['status']
            j.update(completed=bool(r.get('completed') and not forced),last_q_mm=r.get('last_depth_mm'))
        except Exception as exc:j['verification_error']=repr(exc)
        write_json(out/'queue_status.json',state);dashboard(out,state)
    try:
        for path,digest in m['inputs_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Input changed during calculation')
        a=analyze(out,Path(m['experiment']));state['main_completed']=a['completed']
    except Exception as exc:state['analysis_error']=repr(exc);traceback.print_exc()
    state.update(status='finished',finished=stamp());write_json(out/'queue_status.json',state);dashboard(out,state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','launch','worker','probe'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path)
    parser.add_argument('--depth',type=float,default=.1);parser.add_argument('--seconds',type=float,default=180.)
    args=parser.parse_args();out=args.output.resolve()
    if args.phase=='prepare':prepare(out,args.reference.resolve())
    elif args.phase=='launch':launch(out)
    elif args.phase=='worker':worker(out)
    elif 0<args.depth<=.25 and 0<args.seconds<=180:probe(out,args.reference.resolve(),args.depth,args.seconds)
    else:parser.error('Conditional checks require depth<=.25 and solve budget<=180 s')


if __name__=='__main__':main()
