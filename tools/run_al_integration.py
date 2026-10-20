"""Bounded independent P2 integration/time paths and gated deeper calculation."""
import argparse
import contextlib
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_support import write_json
from tools import run_augmented_blade as base
from tools.run_material_candidate import audit
from tools.render_solver_dashboard import shell,atomic_write,stamp
from prototype.al_globalization import WarmGlobalizedAugmentedBladeProblem


def factory(angle,*,coarse=False,spacing=.000625,order=5,refinement=0,fast=True,contact_basis='P0',surface_spacing=.005,
            material_workers=2,linear_threads=8):
    p=base.make_problem(angle,coarse=coarse,spacing=spacing,material_order=order,material_refinement=refinement,surface_spacing=surface_spacing)
    if fast:
        p.__class__=WarmGlobalizedAugmentedBladeProblem
        p.configure_compute(material_workers=material_workers,linear_threads=linear_threads)
    if contact_basis=='P1':
        from prototype.continuous_al import enable_continuous_contact
        enable_continuous_contact(p,fast=fast)
    elif contact_basis!='P0':raise ValueError('Unknown contact multiplier basis')
    return p


def run_case(args):
    basis=getattr(args,'contact_basis','P0');surface=getattr(args,'surface_spacing',.005)
    if args.resume:
        previous=json.loads((args.resume/'result.json').read_text(encoding='utf-8'))
        if (previous['settings'].get('order')!=args.order or previous['settings'].get('refinement',0)!=args.refinement
                or previous['settings'].get('contact_basis','P0')!=basis or previous['settings'].get('surface_spacing',.005)!=surface):
            raise ValueError('New material integration requires independent history, never transfer')
    made=[]
    def make(*a,**kw):
        p=factory(*a,**kw,order=args.order,refinement=args.refinement,contact_basis=basis,surface_spacing=surface);made.append(p);return p
    def original(*a,**kw):
        return factory(*a,**kw,order=args.order,refinement=args.refinement,contact_basis=basis,surface_spacing=surface,fast=False)
    try:
        # ONE independent original-kernel audit of every state replaces two
        # repeated complete audits. Both geometry algorithms/history chains stay.
        r=base.run(args,problem_factory=make,audit_factory=original)
        write_json(args.output/'result.json',r)
        write_json(args.output/'outcome.json',{k:r.get(k) for k in ('status','completed','last_depth_mm','elapsed_s','error','audit_error','original_kernel_audit_passed')})
        base.report_html(args.output,r)
        return r
    finally:
        for p in made:p.close_compute()


def check_sources(m):
    for name,digest in m['source_sha256'].items():
        if hashlib.sha256((Path(m['source'])/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed frozen source '+name)
    for name,digest in m.get('input_sha256',{}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed qualified input '+name)


def gate(out,folders,end):
    result=dict(passed=False,target_q_mm=end,scope='Integration/time engineering screen, not mesh/contact-space or experimental validation')
    try:
        inputs={};results=[]
        for name in folders:
            p=out/name;r=json.loads((p/'result.json').read_text(encoding='utf-8'))
            if not r['completed'] or not r.get('original_kernel_audit_passed'):raise ValueError('Incomplete or unaudited '+name)
            inputs.update(base.verify_case(p,require_independent=True));results.append(r)
        comparisons=[base.compare(out/folders[0],out/name,end) for name in folders[1:]]
        minj=min(z['min_j'] for r in results for z in r['steps'])
        result.update(comparisons=comparisons,min_j=minj,input_sha256=inputs,
                      passed=bool(all(c['passed'] for c in comparisons) and minj>=.05))
    except Exception as exc:result['error']=repr(exc)
    return result


def dashboard(out,state):
    body=f'<h1>接触／积分离散：独立路径与自动加深</h1><p>{stamp()} · {escape(state["status"])}</p>'
    body+='<p>q 为参考法向深度。57°下 q=0.3 / 0.5 mm 对应竖直 0.550824 / 0.918039 mm。保持同一材料、刀具、摩擦、底部约束和原物理验收；不预测裂纹。</p>'
    body+='<p>三条57°路径：5阶积分、8阶积分、5阶半步，各有从静止开始的独立积分点历史；只续接自己的合格预检。先全部到 q=0.3；反力分量差≤2%、整顶面差≤2 μm、全路径 minJ≥0.05 且每态原核复核通过，才自动到 q=0.5，并做33° q=0.3。没有通过就停止依赖项，继续独立对照。</p>'
    body+='<p>积分不收敛、局部退化或超时均保留失败前缀；不自动加预算或放宽物理门槛。成功求解不等于实验验证。<a href="../index.html">诊断与本轮说明</a></p><table><tr><th>任务</th><th>状态</th><th>目标q/mm</th><th>已接受q/mm</th><th>计算预算/s</th></tr>'
    for j in state['jobs']:
        folder=out/j['name'];depth=j.get('last_depth_mm','—');status=j['status']
        if (folder/'result.json').exists():
            try:
                r=json.loads((folder/'result.json').read_text(encoding='utf-8'));depth=r.get('last_depth_mm','—')
                if status=='running':status=r['status']
            except (OSError,json.JSONDecodeError):pass
        body+='<tr>'+f'<td><a href="{j["name"]}/index.html">{j["name"]}</a></td>'+''.join(f'<td>{escape(str(v))}</td>' for v in (status,j['depth'],depth,j['seconds']))+'</tr>'
    body+='</table><p>每个任务有HTML日志、力/位移CSV、准确历史检查点和终态审查。</p>'
    for name,g in state.get('gates',{}).items():
        body+=f'<h2>{escape(name)}</h2><pre>'+escape(json.dumps({k:v for k,v in g.items() if k!='input_sha256'},ensure_ascii=False,indent=2))+'</pre>'
    atomic_write(out/'index.html',shell('积分退化批量验证',body,refresh=state['status']=='running'))


def execute_job(out,m,j):
    folder=out/j['name'];cmd=[sys.executable,'-X','utf8','-u','-m','tools.run_al_integration','--phase','case','--output',str(folder),
        '--angle',str(j['angle']),'--order',str(j['order']),'--depth',str(j['depth']),'--increment',str(j['increment']),'--seconds',str(j['seconds'])]
    if j.get('resume'):cmd+=['--resume',str(Path(j['resume']))]
    for key,flag in (('contact_basis','--contact-basis'),('surface_spacing','--surface-spacing'),('spacing','--spacing')):
        if key in j:cmd+=[flag,str(j[key])]
    env=os.environ.copy();env.update(PYTHONPATH=m['runtime_path'],SOFTMODELING_PARDISO_RUNTIME=m['pardiso_runtime'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    start=time.monotonic();j.update(status='running',started=stamp(),command=cmd)
    with (out/(j['name']+'.stdout.log')).open('wb') as stdout,(out/(j['name']+'.stderr.log')).open('wb') as stderr:
        proc=subprocess.Popen(cmd,cwd=m['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
        j['pid']=proc.pid
        while proc.poll() is None:
            if time.monotonic()-start>j['seconds']+900:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=15)
                j.update(status='supervisor_timeout',completed=False);break
            state=json.loads((out/'queue_status.json').read_text(encoding='utf-8'))
            for i,item in enumerate(state['jobs']):
                if item['name']==j['name']:state['jobs'][i]=dict(j)
            write_json(out/'queue_status.json',state);dashboard(out,state)
            if (folder/'run.log').exists():
                # Own filename: never race the solver's index/log atomic writes.
                text=(folder/'run.log').read_text(encoding='utf-8',errors='replace')
                atomic_write(folder/'live_log.html',shell('实时计算日志','<a href="index.html">状态</a><pre>'+escape(text)+'</pre>',refresh=True))
            time.sleep(20)
    j.update(returncode=proc.returncode,wall_elapsed_s=time.monotonic()-start,finished=stamp())
    if j['status']!='supervisor_timeout':
        try:
            r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            j.update({k:r.get(k) for k in ('status','completed','last_depth_mm','solve_elapsed_s','error')})
            base.verify_case(folder,require_independent=True);j['audit_verified']=True
        except Exception as exc:j.update(status='terminal_verification_failed',completed=False,verification_error=repr(exc))
    return j


def worker(out):
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),time=stamp()),f)
    m=json.loads((out/'manifest.json').read_text(encoding='utf-8'));check_sources(m)
    state=dict(status='running',jobs=m['jobs'],gates={});write_json(out/'queue_status.json',state);dashboard(out,state)
    for index,j in enumerate(state['jobs']):
        if index==3:
            state['gates']['normal_030']=gate(out,['C57_order5_030','C57_order8_030','C57_half_030'],.3)
        if index>=3 and not state['gates']['normal_030']['passed']:
            j.update(status='skipped_accuracy_gate',completed=False)
        else:
            try:check_sources(m);execute_job(out,m,j)
            except Exception as exc:j.update(status='worker_error',completed=False,error=repr(exc))
        write_json(out/'queue_status.json',state);dashboard(out,state)
    if state['gates'].get('normal_030',{}).get('passed'):
        state['gates']['normal_050']=gate(out,['C57_order5_050','C57_order8_050','C57_half_050'],.5)
    state.update(status='finished',finished=stamp());write_json(out/'queue_status.json',state);dashboard(out,state)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--phase',choices=('case','worker'),default='case');a.add_argument('--output',type=Path,required=True)
    a.add_argument('--angle',type=float,choices=(33.,57.),default=57.);a.add_argument('--order',type=int,choices=(3,5,8),default=5);a.add_argument('--refinement',type=int,choices=(0,1,2),default=0)
    a.add_argument('--depth',type=float,default=.01);a.add_argument('--increment',type=float,default=.005);a.add_argument('--seconds',type=float,default=300)
    a.add_argument('--spacing',type=float,default=.000625);a.add_argument('--resume',type=Path);a.add_argument('--coarse',action='store_true');a.add_argument('--blas-threads',type=int,default=1)
    a.add_argument('--contact-basis',choices=('P0','P1'),default='P0');a.add_argument('--surface-spacing',type=float,default=.005)
    a.add_argument('--audit-seconds',type=float,default=900.)
    args=a.parse_args();args.output=args.output.resolve()
    if not (0<args.depth<=.5 and 0<args.increment<=.005 and 0<args.seconds<=7200
            and 0<args.surface_spacing<=.005 and 0<args.audit_seconds<=1800):a.error('Bounded .5 mm diagnostic only')
    if args.phase=='worker':worker(args.output)
    else:run_case(args)


if __name__=='__main__':main()
