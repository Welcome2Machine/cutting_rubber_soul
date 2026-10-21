"""Three independent cutting paths isolating the transverse constraint.

New user-authorized structural diagnostic, not a restart of failed PE/57 paths.
Every angle has the same material and reference width. No fitted angle weights,
no new retries, no relaxed physical acceptance, no automatic model promotion.
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
from prototype.transverse_control import PlaneStressControl
from prototype.material_case import signature,load_force_csv
from tools import run_common_material_study as host
from tools.run_material_uncertainty import direction_trace
from tools.render_solver_dashboard import atomic_write,shell,stamp
from run_support import write_json

ROOT=Path(__file__).resolve().parents[1]
MODULE='tools.run_transverse_study'


class PlaneStressMaterial(PlaneStressControl,host.ComputeMaterial):
    pass


def make(angle,parameters):
    p=host.make(angle,parameters);p.__class__=PlaneStressMaterial
    return p


def jobs():
    return [dict(name=f'ps_{a}',angle=a,parameters=host.BASE.tolist(),stage='constraint',
        target_mm=.5,increment=.01,seconds=3600,status='pending') for a in (0,33,57)]


def case(out,job):
    with direction_trace(out,job):return host.case(out,job,factory=make)


def render(out,state):
    body=f'<h1>小角度幅值偏差：面外约束独立加载</h1><p>{stamp()} · {escape(state["status"])}</p>'
    body+='<p>共同材料G∞=.14/Gv=1.24 MPa，τ=1.7 s，K=23.343333 MPa、μ=.35；仅把F33=1改为每个材料点总σ33=0。保留同一P1网格、20 mm参考宽度、刀具、原生接触、自接触、竖直速度和原验收。PS是面外自由的诊断极限，不是有限宽三维解或已确认的实际边界。</p>'
    body+='<p><a href="analysis.html">已审查力曲线/实验误差与57°保持门槛</a> · <a href="screen/index.html">45个共同参数条件方向</a> · <a href="plan.html">依据/限制/预算</a> · <a href="tests.html">回归</a> · <a href="manifest.json">冻结输入</a></p>'
    body+='<table><tr><th>组别</th><th>状态</th><th>接受mm</th><th>Fz N</th><th>记录</th></tr>'
    for j in state['jobs']:
        folder=out/j['name'];r=host.read(folder/'result.json') if (folder/'result.json').exists() else {}
        s=(r.get('steps') or [{}])[-1]
        body+=f'<tr><td>{j["name"]}</td><td>{escape(j["status"])}</td><td>{s.get("depth_mm","—")}</td><td>{s.get("reaction_fz_n","—")}</td><td>'
        if r:
            case_page(folder);body+=f'<a href="{j["name"]}/index.html">HTML日志</a> · <a href="{j["name"]}.ptc.jsonl">PTC探针</a>'
        body+='</td></tr>'
    body+='</table><p>三条各自静止→.01 mm预检（180 s求解+900 s准备审查），通过后只续自己的准确双历史至.5 mm（3600+900 s）。500 PTC/40停滞/1e−4 N验收不变；失败或超时继续下一组，不缩步、不重试、不增加预算。57°为新面外约束问题，未恢复此前失败PE路径。</p>'
    atomic_write(out/'index.html',shell('面外约束因素验证',body,refresh=state['status']=='running'))
    write_json(out/'queue_status.json',state)


def case_page(folder):
    host.case_page(folder)
    path=folder/'index.html'
    content=path.read_text(encoding='utf-8').replace('共同材料算例日志','平面应力诊断日志')
    content=content.replace('</h1>','</h1><p>独立面外自由诊断：消去总σ33；不是三维模型。材料与所有其他角度共用；参考宽度20 mm不按变形重新乘力。物理残差、CCD/J、耗散及额外平面应力验收不变。</p>',1)
    atomic_write(path,content)


def supervise(out,m,job,state):
    old=host.MODULE,host.render;host.MODULE,host.render=MODULE,render
    try:return host.supervise(out,m,job,state)
    finally:host.MODULE,host.render=old


def qualify_pilot(out,job,pilot):
    """Check assembled condensed material tangent at a real accepted FE state.

    Surface perturbations are zero: this isolates the changed bulk formulation;
    it does not requalify the known native contact-kinematics limitations.
    """
    folder=out/pilot['name'];host.verify_case(folder);r=host.read(folder/'result.json')
    p=make(job['angle'],job['parameters']);i=len(r['steps'])-1
    with np.load(folder/'states'/f'{i:04d}.npz',allow_pickle=False) as z:s=dict(z)
    fresh=p.audit_state(s,r['steps'][i]);p.previous=s['previous_positions']
    p.previous_material_metric=s['previous_material_metric'];p.material_dt=float(s['material_dt_s'])
    x=s['positions'];lag=p.lag(x);_,g,H,_=p.evaluate(x,lag,projected=False)
    A=p.coupled_friction_tangent(x,lag,H)
    v=np.zeros_like(x);coord=p.mesh.local_sn
    v[:p.n,0]=np.sin(1.37*coord[:,0]+.23)*np.sin(.77*coord[:,1])
    v[:p.n,1]=np.cos(.91*coord[:,0]-.11)*np.sin(1.13*coord[:,1])
    v[p.surface]=0.;flat=v.ravel();allowed=np.zeros(p.total,dtype=bool);allowed[p.free]=True;flat[~allowed]=0.
    v/=max(float(np.max(abs(v))),1e-30);expected=A@v.ravel();trials=[]
    for eps in (1e-7,3e-8):
        plus=p.evaluate(x+eps*v,p.lag(x+eps*v),tangent=False)[1]
        minus=p.evaluate(x-eps*v,p.lag(x-eps*v),tangent=False)[1]
        relative=float(np.linalg.norm((plus-minus)/(2*eps)-expected)/max(np.linalg.norm(expected),1e-30))
        trials.append(dict(displacement_scale_mm=eps,relative_error=relative,passed=bool(relative<2e-4)))
    result=dict(passed=all(t['passed'] for t in trials),fresh_audit=fresh,trials=trials,
        scope='Real nonuniform FE condensed material tangent; surface fixed, not contact objectivity certification')
    write_json(out/(job['name']+'.pilot_qualification.json'),result)
    return result['passed']


def curve(folder):
    r=host.read(folder/'result.json');a=host.read(folder/'audit.json')
    if not a['passed'] or a['states']!=len(r['steps']):raise ValueError('Incomplete audit')
    return r,np.array([s['depth_mm'] for s in r['steps']]),np.array([s['reaction_fz_n'] for s in r['steps']])


def features(d,f,ed,ef):
    if d[-1]<.5-1e-10:return None
    grid=np.arange(10,51)/100;err=np.interp(grid,d,f)-np.interp(grid,ed,ef)
    z=np.array([.1,.25,.5]);fs=np.diff(np.interp(z,d,f))/np.diff(z);es=np.diff(np.interp(z,ed,ef))/np.diff(z)
    return dict(rmse_n=float(np.sqrt(np.mean(err**2))),bias_n=float(err.mean()),fz05_n=float(np.interp(.5,d,f)),
                fe_secants_n_mm=fs.tolist(),experiment_secants_n_mm=es.tolist(),
                max_secant_relative_error=float(np.max(abs(fs/es-1))))


def analysis(out,m,state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[];raw=[];curves={}
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(1,3,figsize=(13,4.7),sharex=True,sharey=True,layout='constrained')
        for ax,angle in zip(axes,(0,33,57)):
            ed,ef,_=load_force_csv(out/f'experiment{angle}.csv');mask=ed<=.5+1e-12
            ax.plot(ed[mask],ef[mask],color='#333333',lw=2,label='Experiment (original zero)')
            for label,folder,color,style in (('PE reference',Path(m['baselines'][str(angle)]),'#0072B2','--'),
                    ('PS diagnostic',out/f'ps_{angle}','#D55E00','-')):
                if not (folder/'audit.json').exists():continue
                r,d,f=curve(folder);mask=d<=.5+1e-12;d=d[mask];f=f[mask]
                complete=bool(d[-1]>=.5-1e-10)
                metrics=features(d,f,ed,ef);curves[angle,label]=(d,f)
                row=dict(angle=angle,model=label,status=r['status'],complete_window=complete,
                    accepted_mm=float(d[-1]),metrics=metrics,transverse=r['steps'][len(d)-1].get('transverse'))
                records.append(row)
                ax.plot(d,f,color=color,ls=style,marker='o' if label.startswith('PS') else None,
                    markevery=10,ms=3,label=label+(' [partial]' if not complete else ''))
                raw.extend(dict(angle=angle,model=label,depth_mm=float(x),fz_n=float(y)) for x,y in zip(d,f))
            ax.set(title=f'{angle} degrees',xlabel='Vertical blade travel (mm)',xlim=(0,.5),ylim=(0,None))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0].set_ylabel('Fz (N)');fig.suptitle('Same material and reference width | transverse-constraint diagnostic')
        fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    by={(r['angle'],r['model']):r for r in records};gate=dict(evaluable=False,promotion=False)
    if all((a,'PS diagnostic') in by and by[a,'PS diagnostic']['complete_window'] for a in (0,33,57)):
        grid=np.arange(10,51)/100
        d,f=curves[57,'PS diagnostic'];bd,bf=curves[57,'PE reference']
        change=float(np.max(abs(np.interp(grid,d,f)-np.interp(grid,bd,bf))))
        baseline_scale=float(np.interp(.5,bd,bf));tolerance=max(.5,.1*baseline_scale)
        small=all(by[a,'PS diagnostic']['metrics']['rmse_n']<=.8*by[a,'PE reference']['metrics']['rmse_n'] for a in (0,33))
        large=bool(change<=tolerance and by[57,'PS diagnostic']['metrics']['rmse_n']<=by[57,'PE reference']['metrics']['rmse_n']+.25)
        gate=dict(evaluable=True,small_angle_rmse_improves_20_percent=small,
            large_angle_max_curve_change_n=change,large_angle_change_limit_n=tolerance,
            large_angle_preserved=large,joint_signal=bool(small and large),promotion=False)
    result=dict(created=stamp(),records=records,gate=gate,
        scope='Existing audited PE references vs independently loaded PS. Later fracture unknown; no claim of 3D validity.',
        transformations='Only um to mm and linear interpolation within observed coverage; no zero shift, force rescale, extrapolation or curve smoothing.')
    write_json(out/'analysis.json',result)
    with (out/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['angle','model','depth_mm','fz_n']);writer.writeheader();writer.writerows(raw)
    body='<h1>面外约束：幅值、形状和大角度保持检查</h1><p>'+stamp()+'</p><img src="comparison.png" style="width:100%" alt="三个角度分别比较实验、原平面应变和新平面应力接受路径，共同坐标轴。未完成的路径不延长。">'
    body+='<p>只绘制已经完整前缀审查的结果；运行进度在状态页。PS容许厚度方向自由变形，是诊断极限，不自动代表20 mm宽试样。Fz仍按20 mm参考宽度积分，没有缩放有效宽度。57°参与过原材料选择，保留它是本轮约束，不是独立验证证明。</p>'
    body+='<p>预设门槛：0/33°的.1–.5 mm RMSE各改善至少20%；57°整段相对原参考最大改变量≤max(.5 N, 参考末力10%)且实验RMSE增量≤.25 N。未完成57°时不能判定保持成功。曲率、面外伸长、体积J和表面接触身份分别保留，不由一个终点选模型。</p>'
    body+='<pre>'+escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre><p><a href="index.html">状态/日志</a> · <a href="comparison.csv">原始接受力表</a></p>'
    atomic_write(out/'analysis.html',shell('面外约束实验对照',body))


def prepare(out):
    if (out/'manifest.json').exists():raise ValueError('Refuse existing manifest')
    tests=host.read(out/'tests.json')
    if not tests['passed'] or tests['skipped']:raise ValueError('Tests not qualified')
    screen=host.read(out/'screen/probe.json')
    if screen['status']!='completed' or len(screen['fields'])!=9:raise ValueError('Parameter screen incomplete')
    inputs={str(out/'tests.json'):host.sha(out/'tests.json'),str(out/'screen/probe.json'):host.sha(out/'screen/probe.json')}
    baselines={};checks=[]
    with threadpool_limits(1):
        for a in (0,33,57):
            folder=ROOT/host.REFERENCES[a];inputs.update(host.verify_case(folder));r=host.read(folder/'result.json')
            base=host.make(a,host.BASE);new=make(a,host.BASE)
            s=signature(new);s.pop('material_kinematics')
            if json.loads(json.dumps(s))!=r['signature']:raise ValueError('Changed nontransverse physics')
            np.testing.assert_array_equal(base.rest,new.rest);np.testing.assert_array_equal(base.mesh.triangles,new.mesh.triangles)
            np.testing.assert_array_equal(base.free,new.free)
            idx=next(i for i,s in enumerate(r['steps']) if abs(s['depth_mm']-.5)<1e-10)
            with np.load(folder/'states'/f'{idx:04d}.npz',allow_pickle=False) as z:
                checks.append(dict(angle=a,fresh_baseline=base.audit_state(dict(z),r['steps'][idx]),
                    same_geometry_material_contact_boundary=True,independent_rest_histories=True))
            baselines[str(a)]=str(folder)
            src=ROOT/'docs'/host.EXPERIMENTS[a];shutil.copy2(src,out/f'experiment{a}.csv')
            inputs[str(out/f'experiment{a}.csv')]=host.sha(out/f'experiment{a}.csv')
    write_json(out/'qualification.json',dict(passed=True,checks=checks))
    inputs[str(out/'qualification.json')]=host.sha(out/'qualification.json')
    source=host.snapshot(out)
    for name in ('tools/run_transverse_study.py','tools/run_material_uncertainty.py','tools/probe_angle_factors.py','tests/test_transverse_control.py'):
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest);source[name]=host.sha(dest)
    m=dict(created=stamp(),python=sys.executable,runtime_path=str(ROOT/'.runtime/ipc-python'),pardiso_runtime=str(ROOT/'.runtime/pardiso'),
        source_sha256=source,inputs=inputs,jobs=jobs(),baselines=baselines,
        policy='Explicit new structural diagnostic; own histories only; no retry/cutback/promotion')
    write_json(out/'manifest.json',m);state=dict(status='prepared',jobs=jobs());analysis(out,m,state);render(out,state)
    print(json.dumps(dict(prepared=True,sources=len(source),inputs=len(inputs),checks=checks)),flush=True)


def worker(out):
    m=host.read(out/'manifest.json');host.check(m,out);state=host.read(out/'queue_status.json')
    if state['status']!='prepared':raise ValueError('Not prepared')
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),time=stamp()),f)
    state['status']='running';render(out,state)
    try:
        for j in list(state['jobs']):
            host.check(m,out)
            try:
                pilot=dict(j,name='pilot_'+j['name'],target_mm=.01,seconds=180,stage='preflight')
                state['jobs'].append(pilot)
                if supervise(out,m,pilot,state) and qualify_pilot(out,j,pilot):
                    j['resume']=str(out/pilot['name']);supervise(out,m,j,state)
                else:j.update(status='not_qualified',reason='Own plane-stress preflight or assembled tangent failed')
            except Exception as exc:j.update(status='worker_case_failed',error=repr(exc));traceback.print_exc()
            try:analysis(out,m,state)
            except Exception as exc:state.setdefault('analysis_errors',[]).append(dict(case=j['name'],error=repr(exc)));traceback.print_exc()
            render(out,state)
        state.update(status='ended',ended=stamp())
    except Exception as exc:state.update(status='worker_failed',error=repr(exc));raise
    finally:render(out,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','worker','case','report'));p.add_argument('--output',type=Path,required=True);p.add_argument('--job-json')
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='case':case(out,json.loads(a.job_json))
    elif a.mode=='prepare':prepare(out)
    elif a.mode=='worker':worker(out)
    else:analysis(out,host.read(out/'manifest.json'),host.read(out/'queue_status.json'))


if __name__=='__main__':main()
