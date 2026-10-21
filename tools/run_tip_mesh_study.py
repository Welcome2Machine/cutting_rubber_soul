"""Four bounded native surface/near-body refinement controls, no physical fit.

Each case starts at rest with its own Maxwell and friction histories. Changing
surface spacing also refines adjacent graded body elements. The coarse paths
are immutable comparison inputs, never initial guesses or resume prefixes.
"""
import argparse
import csv
from html import escape
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from prototype.material_case import signature, load_force_csv
from tools import run_common_material_study as host
from tools import run_tip_radius_study as radius_study
from tools.run_material_uncertainty import direction_trace
from tools.render_solver_dashboard import atomic_write, shell, stamp
from run_support import write_json

ROOT=Path(__file__).resolve().parents[1]
MODULE='tools.run_tip_mesh_study'
PARAMETERS=radius_study.PARAMETERS
COARSE=ROOT/'outputs/tip_radius_20261021'


def jobs():
    return [dict(name=f'r{round(1000*r):02d}_angle{a}',angle=a,radius_mm=r,
        parameters=list(PARAMETERS),surface_spacing_mm=.0025,stage='mesh',
        target_mm=.5,increment=.01,seconds=3600,status='pending')
        for r in (.02,.01) for a in (0,33)]


def make(angle,parameters,*,radius):
    if radius not in (.01,.02):raise ValueError('Refinement is restricted to radii10/20 um')
    return radius_study.make(angle,parameters,radius=radius,surface_spacing=.0025)


def geometry_check(fine,coarse):
    """Check the intended sole discretization change before any loading."""
    for key in ('config','material','loading_rate','contact_formulation','dhat',
                'max_iterations','stagnation_window','start','nose_facets'):
        if getattr(fine,key)!=getattr(coarse,key):raise ValueError('Changed '+key)
    np.testing.assert_array_equal(fine.rest[fine.n:],coarse.rest[coarse.n:])
    np.testing.assert_array_equal(fine.previous,fine.rest)
    np.testing.assert_array_equal(fine.material_metric,np.tile(np.eye(3),(len(fine.mesh.triangles),1,1)))
    if fine.n<=coarse.n:raise ValueError('No actual refinement')
    for p,h in ((fine,.0025),(coarse,.005)):
        area=p.mesh.bulk_data[0]
        if np.any(area<=0):raise ValueError('Nonpositive rest element area')
        np.testing.assert_allclose(area.sum(),p.config.length_mm*p.config.height_mm,rtol=1e-12)
        local=p.mesh.local_sn
        np.testing.assert_allclose(local.min(axis=0),(0,0),atol=1e-12)
        np.testing.assert_allclose(local.max(axis=0),(p.config.length_mm,p.config.height_mm),atol=1e-12)
        np.testing.assert_array_equal(p.mesh.bottom_nodes,np.flatnonzero(np.isclose(local[:,1],0)))
        top=local[np.isclose(local[:,1],p.config.height_mm)&(abs(local[:,0]-p.config.drop_s_mm)<1),0]
        np.testing.assert_allclose(np.diff(np.sort(top)),h,rtol=1e-10,atol=1e-12)
        # Plane-strain affine patch: mesh-independent area integral and J.
        F=np.array([[1.15,.24],[-.06,.87]])
        X=p.mesh.reference_xy;z=(X@F.T)[p.mesh.triangles]
        J=np.linalg.det(np.stack((z[:,1]-z[:,0],z[:,2]-z[:,0]),axis=2))/(2*area)
        np.testing.assert_allclose(J,np.linalg.det(F),rtol=1e-10,atol=1e-11)
    return dict(passed=True,coarse_nodes=coarse.n,fine_nodes=fine.n,
        coarse_elements=len(coarse.mesh.triangles),fine_elements=len(fine.mesh.triangles),
        coarse_surface_spacing_mm=.005,fine_surface_spacing_mm=.0025,
        same_blade_material_boundary_domain_time=True,affine_geometry_patch=True,
        independent_rest_histories=True,scope='Surface AND adjacent graded body; no space-convergence claim')


def case(out,job):
    if job.get('resume'):raise ValueError('This study starts every mesh from rest')
    if job['surface_spacing_mm']!=.0025:raise ValueError('Unexpected mesh')
    factory=lambda a,p:make(a,p,radius=job['radius_mm'])
    targets=np.arange(51,dtype=float)/100
    with direction_trace(out,job):return host.case(out,job,factory=factory,targets=targets)


def render(out,state):
    body=f'<h1>刀尖附近网格细化：四组成对路径</h1><p>{stamp()} · {escape(state["status"])}</p>'
    body+='<p>0/33° × 半径20/10 μm；表面5→2.5 μm，同时改变邻近渐变体网格。原生P1/完整自接触、共同材料、μ=.35、dhat=2.5 μm、64刀鼻面片、Δd=.01 mm保持。</p>'
    body+='<p><a href="analysis.html">细/粗网格与实验对照</a> · <a href="plan.html">方案与预算</a> · <a href="qualification.json">几何资格与旧场重审</a> · <a href="tests.html">回归</a> · <a href="manifest.json">冻结源/输入</a></p>'
    body+='<table><tr><th>算例</th><th>状态</th><th>接受mm</th><th>Fz N</th><th>记录</th></tr>'
    for j in state['jobs']:
        folder=out/j['name'];r=host.read(folder/'result.json') if (folder/'result.json').exists() else {}
        last=(r.get('steps') or [{}])[-1]
        if r:host.case_page(folder)
        body+=f'<tr><td>{j["name"]}</td><td>{escape(j["status"])}</td><td>{last.get("depth_mm","—")}</td><td>{last.get("reaction_fz_n","—")}</td><td>'
        if r:body+=f'<a href="{j["name"]}/index.html">HTML日志</a> · <a href="{j["name"]}.ptc.jsonl">实际PTC方向</a>'
        body+='</td></tr>'
    body+='</table><p>每组独立从静止至0.5 mm，3600 s求解+900 s准备/审查；失败/超时转下一组，不重试、缩步、加预算或换历史。500 PTC/40停滞、原真实残差/J/CCD/耗散验收。PARDISO8/IPC4/BLAS1，顺序运行。</p>'
    atomic_write(out/'index.html',shell('局部网格细化',body,refresh=state['status']=='running'))
    write_json(out/'queue_status.json',state)


def supervise(out,m,job,state):
    previous=host.MODULE,host.render;host.MODULE,host.render=MODULE,render
    try:return host.supervise(out,m,job,state)
    finally:host.MODULE,host.render=previous


def paired_summary(records):
    complete={(r['angle'],r['radius_mm'],r['mesh']):r for r in records if r['complete']}
    rows=[]
    for a in (0,33):
        if not all((a,r,h) in complete for r in (.01,.02) for h in ('fine','coarse')):continue
        gap={h:complete[a,.01,h]['fz05_n']-complete[a,.02,h]['fz05_n'] for h in ('fine','coarse')}
        deltas={str(r):complete[a,r,'fine']['fz05_n']-complete[a,r,'coarse']['fz05_n'] for r in (.01,.02)}
        rows.append(dict(angle=a,radius_signal_n=gap,mesh_change_n=deltas,
            mesh_change_exceeds_old_radius_signal=any(abs(x)>abs(gap['coarse']) for x in deltas.values()),
            converged=False,promotion=False))
    return rows


def analysis(out,m,state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[];table=[]
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharex=True,sharey=True,layout='constrained')
        for ax,angle in zip(axes,(0,33)):
            ed,ef,_=load_force_csv(out/f'experiment{angle}.csv');sel=ed<=.5+1e-12
            ax.plot(ed[sel],ef[sel],color='#333333',label='Experiment (original zero)',lw=2)
            for radius,color in ((.02,'#0072B2'),(.01,'#D55E00')):
                name=f'r{round(1000*radius):02d}_angle{angle}'
                for mesh,folder,style in (('coarse',Path(m['coarse_cases'][name]),'--'),('fine',out/name,'-')):
                    if not (folder/'audit.json').exists():continue
                    r=host.read(folder/'result.json');audit=host.read(folder/'audit.json')
                    if not audit['passed'] or audit['states']!=len(r['steps']) or not r['steps']:continue
                    host.verify_case(folder)
                    d=np.array([s['depth_mm'] for s in r['steps']]);f=np.array([s['reaction_fz_n'] for s in r['steps']])
                    complete=bool(r.get('completed') and d[-1]>=.5-1e-12)
                    row=dict(name=name,angle=angle,radius_mm=radius,mesh=mesh,complete=complete,
                        accepted_mm=float(d[-1]),last_fz_n=float(f[-1]),min_j=min(s['min_j'] for s in r['steps']))
                    if complete:
                        grid=np.arange(10,51)/100
                        row.update(fz05_n=float(f[-1]),rmse_01_05_n=float(np.sqrt(np.mean((np.interp(grid,d,f)-np.interp(grid,ed,ef))**2))))
                    records.append(row)
                    ax.plot(d,f,ls=style,color=color,marker='o' if mesh=='fine' else None,markevery=10,ms=3,
                        label=f'R={radius*1000:g} um | h={2.5 if mesh=="fine" else 5:g} um'+(' [partial]' if not complete else ''))
                    for x,y in zip(d,f):table.append(dict(case=name,mesh=mesh,depth_mm=float(x),fz_n=float(y)))
            ax.set(title=f'{angle} degrees',xlabel='Vertical blade travel (mm)',xlim=(0,.5),ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0].set_ylabel('Fz (N)');fig.suptitle('Native intact model | matched radii and physics | no rescaling')
        fig.savefig(out/'force_comparison.png',dpi=160);plt.close(fig)
    result=dict(created=stamp(),records=records,paired=paired_summary(records),space_converged=False,
        limits='One refinement only; native contact limitations and unmeasured nose radii remain. Failed prefixes are not fracture predictions.')
    write_json(out/'analysis.json',result)
    with (out/'force_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case','mesh','depth_mm','fz_n']);w.writeheader();w.writerows(table)
    body='<h1>实际接受路径：网格差与半径差</h1><p>'+stamp()+'</p><img src="force_comparison.png" alt="0和33度的粗细网格、两个刀尖半径与各自实验曲线" style="width:100%">'
    body+='<p>仅绘制有完整前缀审查的曲线；新算例运行中尚未完成末尾审查时，不提前冒充通过。进度看状态页。缺失不外推；实验只转换μm→mm，N和零点保留。两网格的接触材料身份/表面运动另见各组保存态与接触CSV。</p>'
    body+='<pre>'+escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre><p><a href="index.html">状态与日志</a> · <a href="force_comparison.csv">CSV</a></p>'
    atomic_write(out/'analysis.html',shell('细化对照分析',body))


def prepare(out):
    if (out/'manifest.json').exists():raise ValueError('Refuse existing manifest')
    tests=host.read(out/'tests.json')
    if not tests['passed'] or tests['skipped']:raise ValueError('Unqualified regressions')
    inputs={str(out/'tests.json'):host.sha(out/'tests.json')};qual=[];anchors=[];coarse_cases={}
    with threadpool_limits(1):
        for j in jobs():
            folder=COARSE/j['name'];inputs.update(host.verify_case(folder));r=host.read(folder/'result.json')
            if not r.get('completed') or r['steps'][-1]['depth_mm']<.5:raise ValueError('Coarse reference incomplete')
            # Do not silently compare different physical source versions.
            for name,digest in r['source_sha256'].items():
                if name.startswith(('prototype/','mechanics/')) or name=='run_support.py':
                    if host.sha(ROOT/name)!=digest:raise ValueError('Changed physical source '+name)
            coarse=radius_study.make(j['angle'],PARAMETERS,radius=j['radius_mm'])
            if json.loads(json.dumps(signature(coarse)))!=r['signature']:raise ValueError('Coarse physics mismatch')
            fine=make(j['angle'],PARAMETERS,radius=j['radius_mm'])
            qual.append(dict(name=j['name'],**geometry_check(fine,coarse)))
            for idx in (0,50):
                with np.load(folder/'states'/f'{idx:04d}.npz',allow_pickle=False) as z:
                    anchors.append(dict(name=j['name'],**coarse.audit_state(dict(z),r['steps'][idx])))
            coarse_cases[j['name']]=str(folder)
    for a in (0,33):
        shutil.copy2(COARSE/f'experiment{a}.csv',out/f'experiment{a}.csv')
        inputs[str(out/f'experiment{a}.csv')]=host.sha(out/f'experiment{a}.csv')
    write_json(out/'qualification.json',dict(passed=True,geometry=qual,fresh_coarse_anchors=anchors))
    inputs[str(out/'qualification.json')]=host.sha(out/'qualification.json')
    source=host.snapshot(out)
    for name in ('tools/run_tip_mesh_study.py','tools/run_tip_radius_study.py','tools/run_material_uncertainty.py',
                 'tests/test_tip_mesh_study.py','tests/test_tip_radius_study.py'):
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target);source[name]=host.sha(target)
    m=dict(created=stamp(),python=sys.executable,runtime_path=str(ROOT/'.runtime/ipc-python'),
        pardiso_runtime=str(ROOT/'.runtime/pardiso'),source_sha256=source,inputs=inputs,
        jobs=jobs(),coarse_cases=coarse_cases,policy='Fresh independent rest histories; no retry/cutback/automatic extension')
    write_json(out/'manifest.json',m)
    atomic_write(out/'plan.html',shell('局部细化方案','<h1>仅改变表面及邻近体网格</h1><p>用户明确授权四组0/33°×R10/20 μm至0.5 mm；h从5降到2.5 μm。共同材料G∞=.14/Gv=1.24/τ=1.7/K=23.343333，μ=.35，竖直速度.2 mm/s、dhat=.0025 mm、εu=1e-5 mm、23°楔角/.245 mm厚度/64面片均不变。</p><p>四组几何/仿射映射资格、四个粗网格的起点和末态重审通过后，从各自静止态直接加载。保留相同起始距离和准备时间，无粗→细历史插值。步长.01 mm，每组3600 s求解+900 s准备审查，500 PTC/40停滞及全部物理验收，任何结果后进入下一组。禁止增加预算或自动重跑。先两个20 μm，再两个10 μm。</p><p>每个接受增量保存准确材料/摩擦历史，终态审查整个接受前缀。末力比较：网格差是否超过此前约0.18–0.21 N半径信号；同时比较全曲线、minJ、接触身份及表面位移。一次细化不证明空间收敛，降力不等于更正确。原生有限间隙摩擦局限保留，57°与解析刀具事件候选不在本轮。</p>'))
    state=dict(status='prepared',jobs=jobs());analysis(out,m,state);render(out,state)
    print(json.dumps(dict(prepared=True,sources=len(source),inputs=len(inputs),geometry=qual)),flush=True)


def worker(out):
    m=host.read(out/'manifest.json');host.check(m,out);state=host.read(out/'queue_status.json')
    if state['status']!='prepared':raise ValueError('Not prepared')
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),time=stamp()),f)
    state['status']='running';render(out,state)
    try:
        for j in state['jobs']:
            host.check(m,out)
            try:supervise(out,m,j,state)
            except Exception as exc:j.update(status='worker_case_failed',error=repr(exc));traceback.print_exc()
            try:analysis(out,m,state)
            except Exception as exc:
                state.setdefault('analysis_errors',[]).append(dict(case=j['name'],error=repr(exc)));traceback.print_exc()
            render(out,state)
        state.update(status='ended',ended=stamp())
    except Exception as exc:
        state.update(status='worker_failed',error=repr(exc));raise
    finally:render(out,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','worker','case','report'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--job-json');a=p.parse_args();out=a.output.resolve()
    if a.mode=='case':case(out,json.loads(a.job_json))
    elif a.mode=='prepare':prepare(out)
    elif a.mode=='worker':worker(out)
    else:analysis(out,host.read(out/'manifest.json'),host.read(out/'queue_status.json'))


if __name__=='__main__':main()
