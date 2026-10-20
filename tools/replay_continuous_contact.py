"""Bounded new contact-space replay with EXACT old physical histories.

This conditional test isolates the multiplier-space change. It is never a
qualified independent new loading path, and cannot seed one.
"""
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
from tools.run_al_integration import factory
from tools.render_solver_dashboard import shell,atomic_write


def run(source,out,start=.255,target=.26,seconds=240):
    out.mkdir(parents=True,exist_ok=False)
    r=json.loads((source/'result.json').read_text(encoding='utf-8'));s=r['settings']
    index=next(i for i,z in enumerate(r['steps']) if abs(z['depth_mm']-start)<1e-12)
    file=source/'states'/f'{index:04d}.npz'
    with np.load(file,allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    result=dict(status='running',completed=False,from_q_mm=start,target_q_mm=target,
        scope='Conditional new-contact-space step, old exact physical histories; NOT independent P1 contact path',
        input_sha256={str(file.resolve()):hashlib.sha256(file.read_bytes()).hexdigest()},source_sha256=base.snapshot(out))
    p=None;began=time.perf_counter()
    def make(fast):return factory(s['angle'],coarse=s['coarse'],spacing=s['spacing'],order=s['order'],refinement=s.get('refinement',0),fast=fast,contact_basis='P1')
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy():
        try:
            p=make(True);p.ipc.set_num_threads(2)
            np.testing.assert_array_equal(p.rest,state['reference_xy'])
            x=state['positions'].copy();p.previous=state['previous_positions'].copy()
            p.material_metric=state['material_metric'].copy();p.previous_material_metric=state['previous_material_metric'].copy()
            p.material_dt=float(state['material_dt_s']);p.material_dissipation=float(state['cumulative_viscous_dissipation_n_mm'])
            p._set_quadrature(state['al_edges'],state['al_fractions'],state['al_weights'])
            # Positive reference-area averaging is ONLY an algebraic initial
            # guess, preserving cone membership. No physical history is fitted.
            old_edges=np.unique(state['al_edges']);lookup=np.searchsorted(old_edges,p.qedges)
            point_traction=state['al_multipliers'][lookup]
            p.multipliers=np.asarray(p.average@point_traction)
            tick=time.perf_counter();value,record=p.step(x,start,target,time.monotonic()+seconds)
            result['solve_s']=time.perf_counter()-tick
            record['material']['cumulative_viscous_dissipation_n_mm']=p.material_dissipation
            new=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=value,previous_positions=p.previous,
                     depth_mm=target,body_nodes=p.n,**p.checkpoint_data())
            np.testing.assert_array_equal(new['previous_positions'],state['positions'])
            np.testing.assert_array_equal(new['previous_material_metric'],state['material_metric'])
            np.testing.assert_allclose(new['material_dt_s'],(target-start)/p.loading_rate,rtol=1e-12,atol=1e-12)
            save_npz(out/'state.npz',**new)
            original=make(False);original.ipc.set_num_threads(2);tick=time.perf_counter()
            fresh=original.audit_state(new,record)
            result.update(status='completed',completed=True,record=record,original_fresh_audit=fresh,audit_s=time.perf_counter()-tick)
        except Exception as exc:
            result.update(status='timeout' if isinstance(exc,TimeoutError) else 'failed',error=repr(exc));traceback.print_exc(file=log)
            if p is not None and hasattr(p,'rejected_al_state'):save_npz(out/'failed_trial.npz',**p.rejected_al_state)
        finally:
            if p is not None:p.close_compute()
    result['elapsed_s']=time.perf_counter()-began;write_json(out/'result.json',result)
    shown={k:v for k,v in result.items() if k not in ('source_sha256','input_sha256')}
    atomic_write(out/'index.html',shell('接触离散修复：条件重放','<h1>接触离散修复：条件重放</h1><p>原验收门槛；不是完整新路径。</p><pre>'+escape(json.dumps(shown,ensure_ascii=False,indent=2))+'</pre><a href="log.html">日志</a>'))
    atomic_write(out/'log.html',shell('条件重放日志','<pre>'+escape((out/'run.log').read_text(encoding='utf-8'))+'</pre>'))
    print(json.dumps({k:result.get(k) for k in ('status','solve_s','audit_s','elapsed_s','error')}));return result


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--start',type=float,default=.255);a.add_argument('--target',type=float,default=.26);a.add_argument('--seconds',type=float,default=240)
    z=a.parse_args();run(z.source,z.output,z.start,z.target,z.seconds)
