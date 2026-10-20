"""Bounded exact-history AL repair replays; original independent acceptance."""
import argparse
import contextlib
import hashlib
from html import escape
import json
from pathlib import Path
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from run_support import write_json,save_npz,compute_process_policy
from tools import run_augmented_blade as base
from tools.run_al_performance import fast_factory
from tools.diagnose_al_failure import restore
from tools.render_solver_dashboard import shell,atomic_write
from prototype.al_globalization import (GlobalizedAugmentedBladeProblem,WarmStartedAugmentedBladeProblem,
                                       WarmGlobalizedAugmentedBladeProblem)
from prototype.ipc_reference import adaptive_load_steps


def audit_conditional(p,out,report):
    """Verify the imported endpoint identity, then every exact history link.

    The inherited origin is verified against its existing full audit. Never
    substitute rest/identity histories for a nonzero-depth imported endpoint.
    """
    source=Path(report['inherited_prefix']);evidence=base.verify_case(source,require_independent=True)
    origin=json.loads((source/'result.json').read_text(encoding='utf-8'))
    first=report['steps'][0]['depth_mm']
    index=next(i for i,r in enumerate(origin['steps']) if abs(r['depth_mm']-first)<1e-12)
    inherited=source/'states'/f'{index:04d}.npz'
    rows=[];prior_x=None;prior_metric=None;prior_depth=None;total=None
    for i,record in enumerate(report['steps']):
        file=out/'states'/f'{i:04d}.npz'
        with np.load(file,allow_pickle=False) as state:
            if i==0:
                with np.load(inherited,allow_pickle=False) as original:
                    if set(state.files)!=set(original.files):raise ValueError('Imported endpoint fields differ')
                    for key in state.files:np.testing.assert_array_equal(state[key],original[key],err_msg=key)
                if record!=origin['steps'][index]:raise ValueError('Imported endpoint record differs')
                total=float(state['cumulative_viscous_dissipation_n_mm'])
            else:
                np.testing.assert_array_equal(state['previous_positions'],prior_x)
                np.testing.assert_array_equal(state['previous_material_metric'],prior_metric)
                np.testing.assert_allclose(float(state['material_dt_s']),
                    (record['depth_mm']-prior_depth)/p.loading_rate,rtol=1e-12,atol=1e-12)
                total+=record['material']['viscous_dissipation_increment_n_mm']
            np.testing.assert_allclose(float(state['depth_mm']),record['depth_mm'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(float(state['cumulative_viscous_dissipation_n_mm']),total,rtol=1e-10,atol=1e-10)
            row=p.audit_state(state,record)
            prior_x=state['positions'].copy();prior_metric=state['material_metric'].copy();prior_depth=record['depth_mm']
        row['checkpoint_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();rows.append(row)
    result=dict(passed=True,states=len(rows),rows=rows,origin_evidence_sha256=evidence,
                inherited_endpoint_sha256=hashlib.sha256(inherited.read_bytes()).hexdigest(),
                audit_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Exact inherited endpoint plus original full geometry/force/material and every subsequent history/time/dissipation link')
    write_json(out/'audit.json',result);return result


def run(source,out,*,mode='globalized',target=.185,seconds=240,failed=False,start=None):
    out.mkdir(parents=True,exist_ok=False)
    r=json.loads((source/'result.json').read_text(encoding='utf-8'))
    result=dict(status='running',completed=False,target_mm=target,mode=mode,failed_seed=failed,
        source_sha256=base.snapshot(out),input_sha256=base.verify_case(source,require_independent=True),
        scope='Conditional exact-history increment, not a new independent loading path')
    for script in ('tools/replay_al_globalization.py','tools/diagnose_al_failure.py'):
        f=Path(script);dst=out/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(f.read_bytes())
        result['source_sha256'][script]=hashlib.sha256(f.read_bytes()).hexdigest()
    p=None;began=time.perf_counter()
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy():
        try:
            p=fast_factory(r['settings']['angle'],coarse=r['settings']['coarse'],spacing=r['settings']['spacing'])
            if mode=='globalized':p.__class__=GlobalizedAugmentedBladeProblem
            if mode=='warm':p.__class__=WarmStartedAugmentedBladeProblem
            if mode=='warm_globalized':p.__class__=WarmGlobalizedAugmentedBladeProblem
            p.ipc.set_num_threads(2)
            if failed:
                file=source/'failed_trial.npz'
                with np.load(file,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
                result['input_sha256'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
                x=restore(p,s)
                # Reconstruct target from prescribed tool translation, not an assumed step index.
                target=float((x[p.n]-p.blade_at_zero[0])@p.loading_direction/(p.loading_direction@p.loading_direction))
                result['target_mm']=target;old=target-p.material_dt*p.loading_rate
                tick=time.perf_counter();value,it=p._equilibrate(x,time.monotonic()+seconds)
                record=p.acceptance(value,target,it,1)
                p.material_metric=p.material_response(value)['metric'].copy()
                p.material_dissipation+=record['material']['viscous_dissipation_increment_n_mm']
            else:
                i=next(i for i,s in enumerate(r['steps']) if abs(s['depth_mm']-start)<1e-12) if start is not None else len(r['steps'])-1
                with np.load(source/'states'/f'{i:04d}.npz') as z:s={k:z[k].copy() for k in z.files}
                x=restore(p,s);old=float(s['depth_mm'])
                # Original last accepted CURRENT metric is the new step's previous history.
                tick=time.perf_counter();value,record=p.step(x,old,target,time.monotonic()+seconds)
            result['solve_s']=time.perf_counter()-tick
            record['material']['cumulative_viscous_dissipation_n_mm']=p.material_dissipation
            state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=value,previous_positions=p.previous,
                depth_mm=target,body_nodes=p.n,**p.checkpoint_data())
            save_npz(out/'state.npz',**state)
            original=base.make_problem(r['settings']['angle'],coarse=r['settings']['coarse'],spacing=r['settings']['spacing'])
            audit=original.audit_state(state,record)
            result.update(status='completed',completed=True,record=record,original_fresh_audit=audit,from_mm=old)
        except Exception as exc:
            result.update(status='timeout' if isinstance(exc,TimeoutError) else 'failed',error=repr(exc));traceback.print_exc(file=log)
            if p is not None:
                data=getattr(p,'rejected_al_state',None)
                if data is None:data=dict(positions=getattr(p,'trial',p.rest),previous_positions=p.previous,**p.checkpoint_data())
                save_npz(out/'failed_trial.npz',**data)
        finally:
            if p is not None:p.close_compute()
    result['elapsed_s']=time.perf_counter()-began
    write_json(out/'result.json',result)
    shown={k:v for k,v in result.items() if k not in ('source_sha256','input_sha256')}
    atomic_write(out/'index.html',shell('AL有界修复验证','<h1>准确历史的有界修复验证</h1><p>保持材料、摩擦、边界、空间离散及原验收。不是新完整路径。</p><pre>'+escape(json.dumps(shown,ensure_ascii=False,indent=2))+'</pre><a href="log.html">完整日志</a>'))
    atomic_write(out/'log.html',shell('AL修复验证日志','<pre>'+escape((out/'run.log').read_text(encoding='utf-8'))+'</pre>'))
    print(json.dumps({k:result.get(k) for k in ('status','target_mm','solve_s','elapsed_s','error')}))
    return result


def sequence(source,out,*,start=.18,target=.20,increment=.005,seconds=300):
    """New conditional continuation from an audited exact accepted history.

    No old queue is resumed or overwritten. The original full prefix remains
    an input, with an independent audit under the unchanged physical equations.
    This is NOT an independent rest-to-target discretization experiment.
    """
    out.mkdir(parents=True,exist_ok=False);(out/'states').mkdir()
    origin=json.loads((source/'result.json').read_text(encoding='utf-8'))
    r=dict(status='initializing',completed=False,target_mm=target,steps=[],rejected_load_attempts=[],
        source_sha256=base.snapshot(out),input_sha256=base.verify_case(source,require_independent=True),
        settings=dict(angle=origin['settings']['angle'],coarse=origin['settings']['coarse'],
                      spacing=origin['settings']['spacing'],increment=increment,seconds=seconds),
        scope='Conditional exact-history continuation with solver-only changes; not a new independent path',
        inherited_prefix=str(source),inherited_depth_mm=start)
    for name in ('tools/replay_al_globalization.py','tools/diagnose_al_failure.py'):
        f=Path(name);dst=out/'source'/f;dst.write_bytes(f.read_bytes());r['source_sha256'][name]=hashlib.sha256(dst.read_bytes()).hexdigest()
    p=None;began=time.monotonic();logfile=out/'run.log'
    with logfile.open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy():
        try:
            settings=r['settings'];p=fast_factory(settings['angle'],coarse=settings['coarse'],spacing=settings['spacing'])
            p.__class__=WarmGlobalizedAugmentedBladeProblem;p.ipc.set_num_threads(2)
            original=base.make_problem(settings['angle'],coarse=settings['coarse'],spacing=settings['spacing'])
            # Full original-source evidence is already checked; reassemble the
            # actual imported endpoint again and compare exact reference mesh.
            i=next(i for i,rec in enumerate(origin['steps']) if abs(rec['depth_mm']-start)<1e-12)
            with np.load(source/'states'/f'{i:04d}.npz') as z:state={k:z[k].copy() for k in z.files}
            np.testing.assert_array_equal(p.rest,state['reference_xy'])
            original.audit_state(state,origin['steps'][i]);x=restore(p,state);old=start
            r['signature']=dict(origin['signature']);r['compute_settings']=p.compute_settings
            save_npz(out/'states/0000.npz',**state);r['steps'].append(origin['steps'][i]);r['last_depth_mm']=old
            r['setup_elapsed_s']=time.monotonic()-began;deadline=time.monotonic()+seconds;r['status']='running'
            def reject(event):
                r['rejected_load_attempts'].append(event);write_json(out/'result.json',r);base.report_html(out,r)
            targets=np.r_[np.arange(start+increment,target-1e-12,increment),target]
            for endpoint in targets:
                print(f'REPAIR_TARGET q_mm={endpoint:.9g}',flush=True)
                for x,record in adaptive_load_steps(p,x,old,float(endpoint),deadline,.000625,reject):
                    old=record['depth_mm'];index=len(r['steps'])
                    state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,
                        previous_positions=p.previous,depth_mm=old,body_nodes=p.n,**p.checkpoint_data())
                    save_npz(out/'states'/f'{index:04d}.npz',**state)
                    original.audit_state(state,record)
                    r['steps'].append(record);r['last_depth_mm']=old;r['elapsed_s']=time.monotonic()-began
                    write_json(out/'result.json',r);base.report_html(out,r)
                    print(f'ACCEPT q_mm={old:.9g} Fz_N={record["reaction_fz_n"]:.9g}',flush=True)
            r.update(status='completed',completed=True)
        except Exception as exc:
            r.update(status='timeout' if isinstance(exc,TimeoutError) else 'failed',error=repr(exc));traceback.print_exc(file=log)
            if p is not None and hasattr(p,'rejected_al_state'):
                save_npz(out/'failed_trial.npz',**p.rejected_al_state)
        finally:
            r['elapsed_s']=time.monotonic()-began;r['solve_elapsed_s']=r['elapsed_s']-r.get('setup_elapsed_s',0)
            if p is not None and r['steps']:
                try:audit_conditional(original,out,r)
                except Exception as exc:r.update(status='audit_failed',completed=False,audit_error=repr(exc));traceback.print_exc(file=log)
            if p is not None:p.close_compute()
    write_json(out/'result.json',r);base.report_html(out,r)
    print(json.dumps({k:r.get(k) for k in ('status','last_depth_mm','solve_elapsed_s','error','audit_error')}))
    return r


def qualify(source,out):
    out.mkdir(parents=True,exist_ok=False)
    summary=dict(status='running',passed=False,scope='Conditional solver and time checks over .1775--.19 mm, not full-path convergence')
    def write():
        write_json(out/'summary.json',summary)
        body='<h1>C57修复资格：准确历史与半步对照</h1><p>原方程、原力/J/接触门槛。比较窗口为法向 .18–.19 mm；不是从静止开始的全路径时间验证。</p>'
        body+='<p>'+ ' · '.join(f'<a href="{n}/index.html">{n}</a>' for n in ('accepted_anchor','same_failed_depth','base','half'))+'</p>'
        body+='<pre>'+escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
        atomic_write(out/'index.html',shell('C57修复资格',body,refresh=summary['status']=='running'))
    write()
    anchor=run(source,out/'accepted_anchor',mode='warm_globalized',start=.1775,target=.18,seconds=180)
    origin=json.loads((source/'result.json').read_text(encoding='utf-8'))
    if anchor['completed']:
        index=next(i for i,r in enumerate(origin['steps']) if abs(r['depth_mm']-.18)<1e-12)
        with np.load(source/'states'/f'{index:04d}.npz') as a,np.load(out/'accepted_anchor/state.npz') as b:
            n=int(a['base_body_nodes']);error=float(np.linalg.norm(a['positions'][:n]-b['positions'][:n],axis=1).max())
            metric=float(np.max(abs(a['material_metric']-b['material_metric'])))
        force=abs(anchor['record']['reaction_fz_n']-origin['steps'][index]['reaction_fz_n'])/abs(origin['steps'][index]['reaction_fz_n'])
        summary['anchor']=dict(passed=bool(force<=.002 and error<=.0002 and metric<=1e-6),force_fraction=force,max_body_mm=error,metric_max=metric)
    else:summary['anchor']=dict(passed=False,status=anchor['status'])
    write()
    same=run(source,out/'same_failed_depth',mode='warm_globalized',start=.18,target=.18125,seconds=180)
    summary['same_failed_depth']=dict(completed=same['completed'],status=same['status'])
    if not summary['anchor']['passed'] or not same['completed']:
        summary['status']='failed_preflight';write();return summary
    first=sequence(source,out/'base',start=.18,target=.19,increment=.005,seconds=240)
    second=sequence(source,out/'half',start=.18,target=.19,increment=.0025,seconds=300)
    summary['base_completed']=first['completed'];summary['half_completed']=second['completed']
    if first['completed'] and second['completed']:
        c=base.compare(out/'base',out/'half',.19)
        c['scope']='Conditional time control ONLY over .18--.19 mm, with an identical imported .18 history'
        c['complete_comparison']=bool(len(c['rows'])==len(first['steps']) and abs(c['rows'][0]['q_mm']-.18)<1e-12 and abs(c['rows'][-1]['q_mm']-.19)<1e-12)
        c['passed']=bool(c['complete_comparison'] and c['max_force_fraction']<=.02 and c['max_surface_mm']<=.002)
        summary['time_control']=c;summary['passed']=c['passed']
    summary['status']='completed';write();return summary


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--mode',choices=('original','globalized','warm','warm_globalized'),default='globalized')
    a.add_argument('--target',type=float,default=.185);a.add_argument('--seconds',type=float,default=240)
    a.add_argument('--failed',action='store_true');a.add_argument('--start',type=float)
    a.add_argument('--sequence',action='store_true');a.add_argument('--increment',type=float,default=.005)
    a.add_argument('--qualify',action='store_true')
    args=a.parse_args()
    if not 0<args.seconds<=1800 or not 0<args.target<=.25:a.error('Bounded diagnostics require target <= .25 and seconds <= 1800')
    if args.qualify:qualify(args.source.resolve(),args.output.resolve())
    elif args.sequence:
        if args.start is None or not 0<=args.start<args.target or args.increment<=0:a.error('Sequence requires start < target and positive increment')
        sequence(args.source.resolve(),args.output.resolve(),start=args.start,target=args.target,increment=args.increment,seconds=args.seconds)
    else:run(args.source.resolve(),args.output.resolve(),mode=args.mode,target=args.target,seconds=args.seconds,failed=args.failed,start=args.start)
