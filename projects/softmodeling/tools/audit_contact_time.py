"""Independent geometric-partition contact work and impulse along accepted chords.

Integrates the unchanged velocity law on the discrete trajectory, not an exact
continuous-time solution. Error estimates are empirical, never rigorous bounds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad_vec
from numerical_runtime import runtime_entry, add_thread_arguments, thread_arguments
from prototype.ipc_reference import IPCProblem
from stiff_contact import install
from virtual_path_geometry import geometry_partitions
from .dynamic_release import sources, arrays
from . import run_zero_gap_cut as driver
from .frozen_scene import digest
from run_support import write_json


def integrate(p, state, deadline):
    start, end, du = state['previous_positions'], state['positions'], state['increment']
    cuts, _ = geometry_partitions(p, start, end)
    relative = start.astype(np.longdouble)-start[p.surface[0]].astype(np.longdouble)
    d = du.astype(np.longdouble)
    sd = du[p.surface]; blade = p.surface >= p.n
    values = []; forces = []; orders = []; converged = False
    for order in (4, 8, 16, 32, 64):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        total = np.zeros(5); accumulated = np.zeros((2, sd.size))
        for left, right in zip(cuts[:-1], cuts[1:]):
            for z, w in zip(nodes, weights):
                if time.monotonic() >= deadline: raise TimeoutError('Independent contact work budget')
                s = .5*(left+right+(right-left)*z); weight = .5*(right-left)*w
                x = np.asarray(relative+np.longdouble(s)*d, float)
                lag = IPCProblem.lag(p, x)
                gf = np.zeros(sd.size) if lag is None else p.friction.gradient(lag, p.cmesh, sd)
                gn = p.barrier.gradient(p.collisions(x), p.cmesh, x[p.surface])
                f, n = gf.reshape(-1,2), gn.reshape(-1,2)
                row = np.array([2*float(gf@sd.ravel()),
                    2*float(np.sum(f[blade]*sd[blade])), 2*float(np.sum(n[blade]*sd[blade])),
                    -2*p.material_dt*float(f[blade,1].sum()), -2*p.material_dt*float(n[blade,1].sum())])
                if not np.isfinite(row).all() or row[0] < -1e-12: raise ValueError('Invalid contact power')
                total += weight*row; accumulated += weight*np.stack((gf, gn))
        values.append(total); forces.append(accumulated); orders.append(order)
        if len(values) >= 3:
            error = 2*np.max(np.abs(np.diff(values[-3:], axis=0)), axis=0)
            impulse_error = 2*p.material_dt*np.max(np.abs(np.diff(forces[-3:], axis=0)), axis=0)
            allowance = np.array([1e-7, 1e-9, 1e-9, 1e-10, 1e-10])+1e-5*np.abs(values[-1])
            converged = bool(np.all(error <= allowance) and np.max(impulse_error) <= 1e-10)
            if converged: break
    adaptive = None
    if not converged:
        # Different quadrature, same native law and the same primary chord.
        # Scaled max-norm error controls every force/impulse component.
        scales = np.r_[[1e-7, 1e-9, 1e-9, 1e-10, 1e-10], np.full(2*sd.size, 1e-10)]
        def power(s):
            if time.monotonic() >= deadline: raise TimeoutError('Adaptive contact work budget')
            x = np.asarray(relative+np.longdouble(s)*d, float)
            lag = IPCProblem.lag(p, x)
            gf = np.zeros(sd.size) if lag is None else p.friction.gradient(lag, p.cmesh, sd)
            gn = p.barrier.gradient(p.collisions(x), p.cmesh, x[p.surface])
            f, n = gf.reshape(-1,2), gn.reshape(-1,2)
            v = np.r_[[2*float(gf@sd.ravel()),
                2*float(np.sum(f[blade]*sd[blade])), 2*float(np.sum(n[blade]*sd[blade])),
                -2*p.material_dt*float(f[blade,1].sum()), -2*p.material_dt*float(n[blade,1].sum())],
                p.material_dt*gf, p.material_dt*gn]
            if not np.isfinite(v).all() or v[0] < -1e-12: raise ValueError('Invalid contact power')
            return v/scales
        val, err, info = quad_vec(power, 0., 1., epsabs=.1, epsrel=0., norm='max',
            limit=2000, points=cuts[1:-1], quadrature='gk21', workers=1, full_output=True)
        adaptive = dict(method='independent_vector_GK21', scaled_empirical_error=float(err),
            success=bool(info.success), evaluations=int(info.neval), message=info.message,
            preceding_GL_integral=values[-1].tolist(), preceding_GL_error=error.tolist())
        values[-1] = val[:5]*scales[:5]
        forces[-1] = (val[5:]*scales[5:]/p.material_dt).reshape(2,-1)
        error = err*scales[:5]; impulse_error = err*scales[5:]
        converged = bool(info.success and err <= 1.)
    fields = ('friction_n_mm', 'blade_friction_work_n_mm', 'blade_normal_work_n_mm',
              'blade_friction_impulse_z_n_s', 'blade_normal_impulse_z_n_s')
    allowance = np.array([1e-7, 1e-9, 1e-9, 1e-10, 1e-10])+1e-5*np.abs(values[-1])
    return dict(integral=dict(zip(fields, map(float, values[-1]))),
        empirical_error=dict(zip(fields, map(float, error))),
        converged=converged,
        half_domain_impulse_error_max_n_s=float(np.max(impulse_error)),
        half_domain_friction_impulse_difference_n_s=float(np.linalg.norm(
            -p.material_dt*forces[-1][0]-state['impulse_friction'][p.surface].ravel())),
        half_domain_normal_impulse_difference_n_s=float(np.linalg.norm(
            -p.material_dt*forces[-1][1]-state['impulse_normal'][p.surface].ravel())),
        orders=orders, partitions=len(cuts)-1, adaptive=adaptive)


@runtime_entry
def inspect(directory, seconds, *, state_start=0, state_stop=None, runtime=None):
    if not np.isfinite(seconds) or not 0 < seconds <= 600: raise ValueError('Audit requires 0 < seconds <= 600')
    directory = Path(directory).resolve(); started = time.monotonic()
    raw = json.loads((directory/'result.json').read_text())
    stop = len(raw['steps']) if state_stop is None else state_stop
    if not 0 <= state_start < stop <= len(raw['steps']): raise ValueError('Invalid audit state range')
    report = dict(schema='softcut-independent-contact-time/1', passed=False, chunk_passed=False, audits=[],
        state_start=state_start, state_stop=stop, total_states=len(raw['steps']),
        full_path_coverage=(state_start == 0 and stop == len(raw['steps'])),
        result_sha256=digest(directory/'result.json'), source_hashes=sources(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method='geometry_partitioned_GL4_8_16_32_64_then_GK21_on_primary_temporal_chords',
        physical_time_convergence_certified=False, uncertainty_is_rigorous_bound=False)
    try:
        if sources() != raw['source_hashes']: raise ValueError('Numerical producer source differs')
        p = install(driver.problem(raw['config']), raw['settings'].get('integrator', 'midpoint'))
        for index in range(state_start, stop):
            rec = raw['steps'][index]
            path = directory/rec['state_file']
            if digest(path) != rec['state_sha256']: raise ValueError('State hash differs')
            state = arrays(path); audit = driver.audit(p, state, rec)
            if not audit['accepted_chord_ccd_safe']: raise ValueError('Temporal CCD failed')
            row = integrate(p, state, started+seconds)
            row.update(state_index=index, state_sha256=rec['state_sha256'], local_time_s=rec['local_time_s'], dt_s=p.material_dt,
                producer_friction_n_mm=rec['energy']['friction_work_n_mm'],
                producer_blade_work_n_mm=rec['energy']['external_work_n_mm'])
            report['audits'].append(row)
        report['chunk_passed'] = len(report['audits']) == stop-state_start and all(q['converged'] for q in report['audits'])
    except (Exception, TimeoutError) as exc:
        report['error'] = str(exc)
    report['source_unchanged'] = sources() == report['source_hashes']
    report['chunk_passed'] &= report['source_unchanged']
    report['passed'] = report['chunk_passed'] and report['full_path_coverage']
    report.update(elapsed_s=time.monotonic()-started)
    if report['audits']:
        report['sum_integral'] = {k:sum(q['integral'][k] for q in report['audits']) for k in report['audits'][0]['integral']}
        report['sum_empirical_error'] = {k:sum(q['empirical_error'][k] for q in report['audits']) for k in report['audits'][0]['empirical_error']}
    return report


def combine(directory, paths):
    """Certify exact full coverage from bounded, independently computed chunks."""
    directory = Path(directory).resolve()
    raw = json.loads((directory/'result.json').read_text())
    result_sha = digest(directory/'result.json'); own_sha = digest(Path(__file__))
    if raw['source_hashes'] != sources(): raise ValueError('Numerical producer source differs')
    rows=[]; inputs={}; indices=[]
    for path in paths:
        path=Path(path).resolve(); report=json.loads(path.read_text())
        if (not report.get('chunk_passed') or not report.get('source_unchanged') or
                report['result_sha256'] != result_sha or report['source_hashes'] != sources() or
                report['audit_source_sha256'] != own_sha):
            raise ValueError('Unqualified or differently bound audit chunk')
        chunk=report['audits']
        expected=list(range(report['state_start'],report['state_stop']))
        if [r['state_index'] for r in chunk] != expected or not all(r['converged'] for r in chunk):
            raise ValueError('Incomplete audit chunk')
        rows.extend(chunk); indices.extend(expected); inputs[str(path)]=digest(path)
    if sorted(indices) != list(range(len(raw['steps']))):
        raise ValueError('Audit coverage has a gap or duplicate state')
    rows.sort(key=lambda r:r['state_index'])
    for rec,row in zip(raw['steps'],rows):
        path=(directory/rec['state_file']).resolve()
        if path.parent != directory or digest(path) != rec['state_sha256'] or row['state_sha256'] != rec['state_sha256']:
            raise ValueError('Covered state hash differs')
    return dict(schema='softcut-independent-contact-time/1',passed=True,chunk_passed=True,
        full_path_coverage=True,state_start=0,state_stop=len(rows),total_states=len(rows),
        result_sha256=result_sha,source_hashes=sources(),source_unchanged=True,audit_source_sha256=own_sha,
        method='exact_coverage_union_of_bounded_independent_contact_audits',chunk_inputs_sha256=inputs,
        physical_time_convergence_certified=False,uncertainty_is_rigorous_bound=False,audits=rows,
        sum_integral={k:sum(q['integral'][k] for q in rows) for k in rows[0]['integral']},
        sum_empirical_error={k:sum(q['empirical_error'][k] for q in rows) for k in rows[0]['empirical_error']})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--seconds',type=float,default=540.)
    p.add_argument('--state-start',type=int,default=0);p.add_argument('--state-stop',type=int)
    p.add_argument('--combine',nargs='+',type=Path,help='Combine fully qualified bounded chunks with exact coverage')
    add_thread_arguments(p); args=p.parse_args()
    if args.out.exists(): raise FileExistsError('Preserve existing audit')
    result=(combine(args.directory,args.combine) if args.combine else inspect(args.directory,args.seconds,
        state_start=args.state_start,state_stop=args.state_stop,**thread_arguments(args)))
    write_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('audits','source_hashes','thread_runtime')}),flush=True)


if __name__ == '__main__': main()
