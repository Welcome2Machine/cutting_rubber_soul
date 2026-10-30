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
    values = []; forces = []
    for order in (4, 8, 16):
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
        values.append(total); forces.append(accumulated)
    error = 2*np.max(np.abs(np.diff(values, axis=0)), axis=0)
    impulse_error = 2*p.material_dt*np.max(np.abs(np.diff(forces, axis=0)), axis=0)
    fields = ('friction_n_mm', 'blade_friction_work_n_mm', 'blade_normal_work_n_mm',
              'blade_friction_impulse_z_n_s', 'blade_normal_impulse_z_n_s')
    allowance = np.array([1e-7, 1e-9, 1e-9, 1e-10, 1e-10])+1e-5*np.abs(values[-1])
    return dict(integral=dict(zip(fields, map(float, values[-1]))),
        empirical_error=dict(zip(fields, map(float, error))),
        converged=bool(np.all(error <= allowance) and np.max(impulse_error) <= 1e-10),
        half_domain_impulse_error_max_n_s=float(np.max(impulse_error)),
        half_domain_friction_impulse_difference_n_s=float(np.linalg.norm(
            -p.material_dt*forces[-1][0]-state['impulse_friction'][p.surface].ravel())),
        half_domain_normal_impulse_difference_n_s=float(np.linalg.norm(
            -p.material_dt*forces[-1][1]-state['impulse_normal'][p.surface].ravel())),
        orders=[4,8,16], partitions=len(cuts)-1)


@runtime_entry
def inspect(directory, seconds, *, runtime=None):
    if not np.isfinite(seconds) or not 0 < seconds <= 600: raise ValueError('Audit requires 0 < seconds <= 600')
    directory = Path(directory).resolve(); started = time.monotonic()
    raw = json.loads((directory/'result.json').read_text())
    report = dict(schema='softcut-independent-contact-time/1', passed=False, audits=[],
        result_sha256=digest(directory/'result.json'), source_hashes=sources(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method='geometry_partitioned_GL4_8_16_on_primary_temporal_chords',
        physical_time_convergence_certified=False, uncertainty_is_rigorous_bound=False)
    try:
        if sources() != raw['source_hashes']: raise ValueError('Numerical producer source differs')
        p = install(driver.problem(raw['config']), raw['settings'].get('integrator', 'midpoint'))
        for rec in raw['steps']:
            path = directory/rec['state_file']
            if digest(path) != rec['state_sha256']: raise ValueError('State hash differs')
            state = arrays(path); audit = driver.audit(p, state, rec)
            if not audit['accepted_chord_ccd_safe']: raise ValueError('Temporal CCD failed')
            row = integrate(p, state, started+seconds)
            row.update(local_time_s=rec['local_time_s'], dt_s=p.material_dt,
                producer_friction_n_mm=rec['energy']['friction_work_n_mm'],
                producer_blade_work_n_mm=rec['energy']['external_work_n_mm'])
            report['audits'].append(row)
        report['passed'] = bool(raw['steps']) and all(q['converged'] for q in report['audits'])
    except (Exception, TimeoutError) as exc:
        report['error'] = str(exc)
    report['source_unchanged'] = sources() == report['source_hashes']
    report['passed'] &= report['source_unchanged']
    report.update(elapsed_s=time.monotonic()-started)
    if report['audits']:
        report['sum_integral'] = {k:sum(q['integral'][k] for q in report['audits']) for k in report['audits'][0]['integral']}
        report['sum_empirical_error'] = {k:sum(q['empirical_error'][k] for q in report['audits']) for k in report['audits'][0]['empirical_error']}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--seconds',type=float,default=540.)
    add_thread_arguments(p); args=p.parse_args()
    if args.out.exists(): raise FileExistsError('Preserve existing audit')
    result=inspect(args.directory,args.seconds,**thread_arguments(args));write_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('audits','source_hashes','thread_runtime')}),flush=True)


if __name__ == '__main__': main()
