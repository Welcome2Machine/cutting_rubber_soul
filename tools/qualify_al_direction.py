"""Bounded exact-history direction-policy replay with an ORIGINAL-kernel audit.

This is a conditional numerical control, not a new experimental force curve.
"""
import argparse
import contextlib
from html import escape
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from run_support import compute_process_policy, save_npz, write_json
from tools.run_al_integration import factory
from tools.diagnose_al_failure import restore
from tools.run_augmented_blade import snapshot
from tools.render_solver_dashboard import shell, atomic_write


def run(source,out,*,start=None,target=.07,seconds=240,policy='adaptive',separated_self=False):
    out.mkdir(parents=True,exist_ok=False)
    origin=json.loads((source/'result.json').read_text(encoding='utf-8'));s=origin['settings']
    failed=start is None
    i=None if failed else next(i for i,r in enumerate(origin['steps']) if abs(r['depth_mm']-start)<1e-12)
    file=source/'failed_trial.npz' if failed else source/'states'/f'{i:04d}.npz'
    with np.load(file,allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    result=dict(status='running',completed=False,policy=policy,failed_seed=failed,
        scope='Conditional same exact physical histories; NOT a fresh experimental path',
        settings=s,source_sha256=snapshot(out),input_sha256={str(file.resolve()):hashlib.sha256(file.read_bytes()).hexdigest()})
    name='tools/qualify_al_direction.py';script=Path(__file__).read_bytes()
    (out/'source'/name).write_bytes(script)
    result['source_sha256'][name]=hashlib.sha256(script).hexdigest()
    result['separated_self_contact_diagnostic']=separated_self
    def make(fast):
        p=factory(s['angle'],coarse=s['coarse'],spacing=s['spacing'],order=s['order'],
            contact_basis=s['contact_basis'],surface_spacing=s['surface_spacing'],fast=fast)
        if separated_self:
            from prototype.separated_self_contact import enable_separated_self_contact
            enable_separated_self_contact(p)
        return p
    p=None;began=time.perf_counter()
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy():
        try:
            p=make(True);p.ipc.set_num_threads(2);p.direction_policy=policy
            x=restore(p,state);tick=time.perf_counter()
            if failed:
                target=float(state['attempted_depth_mm'])
                x,it=p._equilibrate(x,time.monotonic()+seconds)
                record=p.acceptance(x,target,it,1)
                p.material_metric=p.material_response(x)['metric'].copy()
                p.material_dissipation+=record['material']['viscous_dissipation_increment_n_mm']
            else:x,record=p.step(x,start,target,time.monotonic()+seconds)
            result['solve_s']=time.perf_counter()-tick
            record['material']['cumulative_viscous_dissipation_n_mm']=p.material_dissipation
            new=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,previous_positions=p.previous,
                     depth_mm=target,body_nodes=p.n,**p.checkpoint_data())
            if not failed:
                np.testing.assert_array_equal(new['previous_positions'],state['positions'])
                np.testing.assert_array_equal(new['previous_material_metric'],state['material_metric'])
                np.testing.assert_allclose(new['material_dt_s'],(target-start)/p.loading_rate,rtol=1e-12)
            save_npz(out/'state.npz',**new)
            original=make(False);original.ipc.set_num_threads(2)
            fresh=original.audit_state(new,record)
            result.update(status='completed',completed=True,record=record,original_fresh_audit=fresh)
        except Exception as exc:
            result.update(status='timeout' if isinstance(exc,TimeoutError) else 'failed',error=repr(exc));traceback.print_exc(file=log)
            if p is not None:
                if hasattr(p,'rejected_al_state'):save_npz(out/'failed_trial.npz',**p.rejected_al_state)
                elif hasattr(p,'trial'):save_npz(out/'failed_trial.npz',positions=p.trial,previous_positions=p.previous,
                    attempted_depth_mm=target,**p.checkpoint_data())
        finally:
            if p is not None:p.close_compute()
    result['elapsed_s']=time.perf_counter()-began;write_json(out/'result.json',result)
    display={k:v for k,v in result.items() if k not in ('source_sha256','input_sha256')}
    atomic_write(out/'index.html',shell('方向策略条件验证','<h1>方向策略条件验证</h1><pre>'+escape(json.dumps(display,ensure_ascii=False,indent=2))+'</pre><a href="log.html">日志</a>'))
    atomic_write(out/'log.html',shell('条件验证日志','<pre>'+escape((out/'run.log').read_text(encoding='utf-8'))+'</pre>'))
    print(json.dumps({k:result.get(k) for k in ('status','solve_s','elapsed_s','error')}))
    return result


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--start',type=float);a.add_argument('--target',type=float,default=.07)
    a.add_argument('--seconds',type=float,default=240);a.add_argument('--policy',choices=('first','adaptive'),default='adaptive')
    a.add_argument('--separated-self',action='store_true',help='Conditional no-nonadjacent-self-contact candidate, not original equations')
    z=a.parse_args();run(z.source,z.output,start=z.start,target=z.target,seconds=z.seconds,policy=z.policy,separated_self=z.separated_self)
