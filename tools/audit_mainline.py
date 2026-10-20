"""Bounded, fixed-history audit of a stalled AL state; never advance a load.

Separates constitutive/assembly cost, mixed tangent accuracy, contact branches,
geometric step bounds, and residual backtracking on EXACTLY the same field.
Results are diagnostics, never accepted FE reactions or a replacement history.
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
from scipy.sparse import diags
from threadpoolctl import threadpool_limits
from run_support import write_json, compute_process_policy
from tools.run_al_integration import factory
from tools.run_augmented_blade import snapshot
from tools.diagnose_al_failure import restore, system, tangent_check, constraint_rank
from tools.render_solver_dashboard import shell, atomic_write
from prototype.al_performance import ParallelMaterial
from prototype.curve_contact import blocks


def branch(p, q):
    pressure = np.maximum(0., p.multipliers[:, 0]-p.penalty*q['gap_average'])
    trial = p.multipliers[:, 1]+p.penalty*q['slip_average']
    active = pressure > 0
    stick = active & (abs(trial) < p.config.friction_coefficient*pressure)
    return np.where(~active, 0, np.where(stick, 1, np.where(trial >= 0, 2, 3)))


def nonlinear_split(p,x,q,h,d,alpha):
    """Component norms of R(x+a*d)-R(x)-a*J*d, not a force attribution."""
    nf=len(p.free);direction=np.zeros(p.total);direction[p.free]=d[:nf]
    direction=direction.reshape(-1,2);dl=d[nf:].reshape(-1,2);lam=p.multipliers.copy()
    def forces(xx):
        re,_,dofs,_,_=p.material_bulk((xx[:p.n]-p.rest[:p.n]).ravel(),with_tangent=False)
        bulk=np.zeros(p.total);np.add.at(bulk,dofs.ravel(),re.ravel())
        c=p.contact_data(xx);self_force=np.zeros(p.total);self_force[p.self_dofs]=p._self_terms(xx,False)[1]
        normal=p.D.T@(-p.qweights[:,None]*c['pressure'][:,None]*c['normal']).ravel()
        friction=p.D.T@(p.qweights[:,None]*c['shear'][:,None]*c['tangent']).ravel()
        return bulk,self_force,normal,friction
    base=forces(x);self_h=p._self_terms(x,True)[2]
    normal_h=p.D.T@blocks(-p.qweights[:,None,None]*q['pressure'][:,None,None]*p.distance(p.Q@x,x)[2])@p.D
    contact_h=p.D.T@blocks(q['jacobian'])@p.D
    bulk_h=h-self_h-contact_h
    dn=np.zeros_like(dl);dn[:,0]=dl[:,0];dt=np.zeros_like(dl);dt[:,1]=dl[:,1]
    predicted=[bulk_h@direction.ravel(),self_h@direction.ravel(),
        normal_h@direction.ravel()+q['force_lambda']@dn.ravel(),
        (contact_h-normal_h)@direction.ravel()+q['force_lambda']@dt.ravel()]
    try:
        p.multipliers=lam+alpha*dl;trial=forces(x+alpha*direction)
    finally:p.multipliers=lam
    return {name:float(np.linalg.norm((b-a-alpha*z)[p.free]))
            for name,a,b,z in zip(('bulk','body_self_contact','blade_normal','blade_friction'),base,trial,predicted)}


def run(source, out, seconds=600,failed_file=None):
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((source/'result.json').read_text(encoding='utf-8'))
    settings = original['settings']
    file = failed_file or source/'failed_trial.npz'
    with np.load(file, allow_pickle=False) as z:
        state = {k: z[k].copy() for k in z.files}
    result = dict(status='running', scope='Fixed failed field, exact histories; no FE load advance',
                  input_sha256={str(file.resolve()): hashlib.sha256(file.read_bytes()).hexdigest()},
                  settings=settings,source_sha256=snapshot(out))
    script=Path(__file__);name='tools/audit_mainline.py'
    (out/'source'/name).write_bytes(script.read_bytes())
    result['source_sha256'][name]=hashlib.sha256(script.read_bytes()).hexdigest()
    p = None
    began = time.perf_counter()
    deadline = time.monotonic()+seconds
    with (out/'run.log').open('w', encoding='utf-8', buffering=1) as log, contextlib.redirect_stdout(log), threadpool_limits(1), compute_process_policy():
        try:
            p = factory(settings['angle'], coarse=settings['coarse'], spacing=settings['spacing'],
                        order=settings['order'], contact_basis=settings['contact_basis'],
                        surface_spacing=settings['surface_spacing'])
            p.ipc.set_num_threads(2)
            with np.load(source/'states'/'0000.npz', allow_pickle=False) as origin:
                np.testing.assert_array_equal(p.rest, origin['reference_xy'])
            x = restore(p, state)
            target=float(state['attempted_depth_mm'])
            np.testing.assert_allclose(x[p.n:],p.prescribed_blade_positions(target),rtol=0,
                                       atol=p.blade_boundary_roundoff_tolerance(target))
            np.testing.assert_array_equal(x[p.mesh.bottom_nodes],p.rest[p.mesh.bottom_nodes])
            tick = time.perf_counter(); A, r, h, q, checks, j = system(p, x)
            nf = len(p.free); w = np.repeat(p.edge_area, 2)
            initial_branch = branch(p, q)
            result.update(assembly_s=time.perf_counter()-tick, matrix_shape=list(A.shape), nnz=A.nnz,
                          force_residual_n=float(np.linalg.norm(r[:nf])), checks=checks, min_j=j,
                          attempted_q_mm=float(state['attempted_depth_mm']), dt_s=p.material_dt,
                          branch_counts=np.bincount(initial_branch, minlength=4).tolist(),
                          counts=dict(body_nodes=p.n, material_points=len(p.integration_weights),
                                      multiplier_nodes=len(p.multipliers), contact_points=len(p.qweights)))
            # Both histories must identify an ACTUAL accepted state, even though
            # the trial itself is not accepted. The constitutive increment must
            # match its prescribed target; no rest-history substitution.
            previous_matches = []
            for i, record in enumerate(original['steps']):
                if abs(record['depth_mm']-(result['attempted_q_mm']-p.material_dt*p.loading_rate)) > 1e-10:
                    continue
                with np.load(source/'states'/f'{i:04d}.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(state['previous_positions'], z['positions'])
                    np.testing.assert_array_equal(state['previous_material_metric'], z['material_metric'])
                previous_matches.append(i)
            if len(previous_matches) != 1:
                raise ValueError('Failed field has no unique exact accepted predecessor')
            result['exact_predecessor_index'] = previous_matches[0]
            result['tangent'] = tangent_check(p, x, A, r)
            result['constraint_rank'] = constraint_rank(p, q)
            write_json(out/'result.json', result)

            # Actual field: same kernel, histories and dtype, independent points.
            F = p._deformation(x[:p.n]-p.rest[:p.n]); material_rows = []; control = None
            for workers in (1, 2, 4):
                kernel = ParallelMaterial(workers, 4096)
                try:
                    tick = time.perf_counter()
                    answer = kernel.evaluate(F, p.previous_material_metric, p.material_dt,
                                             p.material, tangent=True, kinematics=p.kinematics)
                    elapsed = time.perf_counter()-tick
                    if control is None: control = answer
                    errors = {k: float(np.max(abs(answer[k]-control[k]))) for k in answer}
                    for k in answer: np.testing.assert_allclose(answer[k], control[k], rtol=1e-12, atol=1e-12)
                    material_rows.append(dict(workers=workers, seconds=elapsed, max_absolute_errors=errors))
                finally: kernel.close()
            result['material_timing'] = material_rows
            del control, answer
            scale = np.r_[np.maximum(abs(h.diagonal()[p.free]), 1.), np.zeros(len(w))]
            lam = p.multipliers.copy(); result['directions'] = []
            for shift in (0., 1e-5, 1e-3, .01, .1, 1.):
                if time.monotonic() > deadline: raise TimeoutError('Fixed-field diagnostic budget')
                M = A+diags(shift*scale+np.r_[np.full(nf, 1e-9), np.zeros(len(w))])
                tick = time.perf_counter(); d = p.linear_direction(M, -r)
                row = dict(shift=shift, solve_s=time.perf_counter()-tick,
                           linear_residual_n=float(np.linalg.norm(M@d+r)),
                           extent_mm=float(abs(d[:nf]).max()), trials=[])
                result['directions'].append(row)
                if row['extent_mm'] > .01:
                    row['rejected'] = 'Original displacement direction radius .01 mm'; continue
                direction = np.zeros(p.total); direction[p.free] = d[:nf]; direction = direction.reshape(-1, 2)
                row['jacobian_bound'] = p.enrichment.safe_step(x[:p.n]-p.rest[:p.n], direction[:p.n])
                row['self_ccd_bound'] = float(p.ipc.compute_collision_free_stepsize(p.self_mesh,
                    x[p.body_surface], (x+direction)[p.body_surface]))
                alpha = min(1., row['jacobian_bound'], row['self_ccd_bound'])
                if shift in (0., .1):
                    row['nonlinear_error_split_n']=nonlinear_split(p,x,q,h,d,alpha)
                    row['split_alpha']=alpha
                Ad = A@d; slope = float(r@Ad); row['slope'] = slope
                for back in range(9):
                    if time.monotonic() > deadline: raise TimeoutError('Fixed-field trial budget')
                    if alpha < 1/256: break
                    trial = x+alpha*direction
                    p.multipliers = lam+alpha*d[nf:].reshape(-1, 2)
                    tick = time.perf_counter()
                    try:
                        g = p.evaluate(trial, tangent=False)[1]; qt = p.contact_data(trial)
                        rt = np.r_[g[p.free], w*qt['constraint'].ravel()]
                        actual = float((r@r-rt@rt)/2)
                        model = float(-alpha*slope-.5*alpha*alpha*(Ad@Ad))
                        ratio = actual/model if model > 0 else -1.
                        ok = bool(actual > max(1e-24, 1e-6*float(r@r)) and ratio >= .1
                                  and actual >= -1e-4*alpha*slope)
                        switches = int(np.count_nonzero(branch(p, qt) != initial_branch))
                        row['trials'].append(dict(alpha=alpha, force_residual_n=float(np.linalg.norm(rt[:nf])),
                            joint_residual_n=float(np.linalg.norm(rt)), actual=actual, ratio=ratio,
                            switches=switches, original_merit_accepts=ok, seconds=time.perf_counter()-tick))
                    finally: p.multipliers = lam.copy()
                    if ok: break
                    alpha *= .5
                write_json(out/'result.json', result)
            result['status'] = 'completed'
        except Exception as exc:
            result.update(status='failed', error=repr(exc)); traceback.print_exc(file=log)
        finally:
            if p is not None: p.close_compute()
    result['elapsed_s'] = time.perf_counter()-began
    write_json(out/'result.json', result)
    atomic_write(out/'index.html', shell('主线固定场核查', '<h1>主线固定场核查</h1><p>失败场不是平衡解；不推进载荷。</p><pre>'+escape(json.dumps(result, ensure_ascii=False, indent=2))+'</pre>'))
    atomic_write(out/'log.html', shell('核查日志', '<pre>'+escape((out/'run.log').read_text(encoding='utf-8'))+'</pre>'))
    print(json.dumps({k: result.get(k) for k in ('status', 'elapsed_s', 'error')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--failed-file',type=Path)
    args = parser.parse_args(); run(args.source, args.output, args.seconds,args.failed_file)
