"""Audit the prescribed virtual process, contact work and required auxiliary work.

No trajectory is selected by these diagnostics. The production straight path,
endpoints, frozen histories and physical fields remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numerical_runtime import numerical_runtime, add_thread_arguments, thread_arguments
from run_support import write_json
from .frozen_scene import load_scene
from .virtual_work_audit import audit_work
from virtual_path_geometry import geometry_partitions
from virtual_friction_work import integrate_power, contact_power_function, interval_decision


def contact_partition(v, lag, displacement):
    """Sum native SINGLE-stencil work, then verify against the complete native set."""
    groups = dict(blade=0., body_self=0., blade_self=0.)
    loads = dict.fromkeys(groups, 0.)
    details = []
    for kind in ('vv_collisions', 'ev_collisions'):
        for q in getattr(lag, kind):
            local = np.asarray([i for i in q.vertex_ids(v.cmesh.edges, v.cmesh.faces) if i >= 0])
            nodes = v.surface[local]
            group = 'blade_self' if np.all(nodes >= v.n) else 'blade' if np.any(nodes >= v.n) else 'body_self'
            d = displacement[local].ravel()
            work = 2*float(v.friction.gradient(q, d)@d)
            groups[group] += work
            loads[group] += 2*float(q.normal_force_magnitude)
            details.append(dict(kind=kind, nodes=nodes.tolist(), group=group,
                power_n_mm=work, normal_load_n=2*float(q.normal_force_magnitude)))
    total = 2*float(v.friction.gradient(lag, v.cmesh, displacement)@displacement.ravel())
    np.testing.assert_allclose(sum(groups.values()), total, rtol=1e-10, atol=1e-10)
    return dict(power_n_mm=total, power_by_contact_n_mm=groups,
        normal_load_by_contact_n=loads, contacts=details)


def inspect(case, output, seconds, endpoint='production'):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started+seconds
    p, base, v, before, after, scene = load_scene(case, endpoint)
    if (scene['endpoint']['residual_n'] > 1e-4 or scene['before_residual_n'] > 1e-4
            or not scene['endpoint']['chord_safe']):
        raise ValueError('Fixed endpoint pair lacks original force/whole-chord certification')
    cuts, geometry = geometry_partitions(v, before, after)
    _, arithmetic = contact_power_function(v, before, after, geometry)
    delta = after-before; origin = np.asarray(arithmetic['origin_mm'])
    relative = before-origin; sd = delta[v.surface]; trace = {}
    def sampled(s):
        s = float(s)
        if time.monotonic() >= deadline:
            raise TimeoutError('Virtual process audit budget')
        if s not in trace:
            point = relative+s*delta
            lag = v.lag(point)
            q = contact_partition(v, lag, sd)
            gaps = []
            for contact in q['contacts']:
                coords = point[contact['nodes']]
                if len(coords) == 2:
                    gap = np.linalg.norm(coords[0]-coords[1])
                else:
                    a,b,c = coords; edge = c-b
                    beta = np.clip((a-b)@edge/(edge@edge), 0, 1)
                    gap = np.linalg.norm(a-b-beta*edge)
                gaps.append(float(gap))
            q.update(s=s, min_gap_mm=min(gaps, default=None))
            trace[s] = q
        return trace[s]
    release = base.stored(before)-v.stored(after)
    cost = p.toughness*float(p.seam_area[v.virtual_removed].sum())
    report = dict(schema='softcut-virtual-process-audit/2', scene=scene,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry=geometry, coordinate_evaluation=arithmetic,
        stored_release_n_mm=release, required_fracture_n_mm=cost,
        completed=False, unchanged_straight_path=True, physical_history_committed=False)
    try:
        total = integrate_power(lambda s:sampled(s)['power_n_mm'], cuts, deadline=deadline)
        report['production_rule'] = total
        # Different quadrature family and extended-precision coordinate construction.
        independent = audit_work(v, before, after, deadline)
        report['independent'] = independent
        report['work_agreement'] = bool(total['converged'] and independent['converged'] and
            abs(total['work_n_mm']-independent['work_n_mm']) <=
            total['empirical_uncertainty_n_mm']+independent['empirical_uncertainty_n_mm']+1e-7)
        report['contact_integrals'] = {}
        for name in ('blade','body_self','blade_self'):
            report['contact_integrals'][name] = integrate_power(
                lambda s:sampled(s)['power_by_contact_n_mm'][name], cuts, deadline=deadline)
        report['partition_integral_sum_n_mm'] = sum(q['work_n_mm'] for q in report['contact_integrals'].values())
        report['decision'] = interval_decision(release, cost, total['work_n_mm'],
            total['empirical_uncertainty_n_mm'], report['work_agreement'])
        report['prescribed_process_ledger'] = dict(
            external_blade_support_work_n_mm=0., kinetic_change_n_mm=0., maxwell_dissipation_n_mm=0.,
            frozen_storage_change_n_mm=-release, friction_dissipation_n_mm=total['work_n_mm'],
            required_auxiliary_work_without_fracture_n_mm=total['work_n_mm']-release,
            required_auxiliary_work_with_gc_n_mm=total['work_n_mm']+cost-release,
            auxiliary_work_is_balance_requirement_not_measured_force_integral=True,
            zero_kinetic_and_frozen_history_are_virtual_assumptions=True,
            interpretation='A prescribed path requiring positive auxiliary work is not an unforced quasistatic release. This identity alone does not define an admissible alternative process.')
        panels = sorted(total['panels'], key=lambda q:q['work_n_mm'], reverse=True)
        running = 0.; dominant = []
        for q in panels:
            dominant.append(q); running += q['work_n_mm']
            if running >= .99*total['work_n_mm']:break
        report['panels_carrying_99_percent'] = dict(panels=dominant,
            total_parameter_width=sum(q['b']-q['a'] for q in dominant), work_n_mm=running)
        # Fixed endpoints: regularization sensitivity is not a qualified root
        # under another model; no changed result is used in crack selection.
        original = v.friction
        report['fixed_endpoint_friction_regularization'] = []
        try:
            for factor in (.1, 1., 10.):
                v.friction = v.ipc.FrictionPotential(v.virtual_friction_epsilon*factor)
                power, _ = contact_power_function(v, before, after, geometry)
                q = integrate_power(power, cuts, deadline=deadline)
                v.set_primary_increment(after, after-v.previous)
                _,g,_,j = v.evaluate(after,v.lag(after),tangent=False)
                v.update_constraint_free(after,g)
                report['fixed_endpoint_friction_regularization'].append(dict(factor=factor,
                    original_endpoint_residual_n=float(np.sqrt(2)*np.linalg.norm(g[v.free])),
                    min_j=float(j), work=q, fixed_endpoints_only=True,
                    changed_regularization_not_used_for_production=True))
        finally:
            v.friction = original
        report['completed'] = True
    except (TimeoutError,ValueError,AssertionError,RuntimeError) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['elapsed_s'] = time.monotonic()-started
        points = sorted(trace.values(),key=lambda q:q['s'])
        report['evaluations'] = len(points)
        report['peak'] = max(points,key=lambda q:q['power_n_mm']) if points else None
        write_json(output/'power_trace.json',points)
        write_json(output/'diagnostic.json',report)
        np.savez_compressed(output/'endpoints.npz',before=before,after=after)
        if points:
            import matplotlib;matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(3,1,sharex=True,figsize=(10,9))
            ss=[q['s'] for q in points]
            for name in ('blade','body_self'):
                axes[0].plot(ss,[q['power_by_contact_n_mm'][name] for q in points],label=name)
                axes[2].plot(ss,[q['normal_load_by_contact_n'][name] for q in points],label=name)
            axes[1].plot(ss,[q['min_gap_mm'] for q in points])
            for ax,label in zip(axes,('Friction power [N mm]','Numerical gap [mm]','Native normal load sum [N]')):
                ax.set_ylabel(label);ax.set_yscale('symlog',linthresh=1e-10);ax.grid()
            axes[0].legend();axes[-1].set_xlabel('Prescribed virtual path parameter s')
            fig.tight_layout();fig.savefig(output/'power.png');plt.close(fig)
            (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Fixed virtual process audit</h1><p>Same certified endpoints. No physical continuation. Numerical gaps have no calibrated microscopic meaning.</p><img src="power.png"><p><a href="diagnostic.json">Energy, contact decomposition and provenance</a></p>')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--endpoint',choices=['production','diagnostic'],default='production')
    parser.add_argument('--seconds',type=float,default=480)
    add_thread_arguments(parser);args=parser.parse_args()
    if not 1<=args.seconds<=540:raise ValueError('Independent work diagnosis <=540 s')
    with numerical_runtime(**thread_arguments(args)) as runtime:
        result=inspect(args.case,args.output,args.seconds,args.endpoint);result['thread_runtime']=runtime
    write_json(args.output/'diagnostic.json',result)
    print(json.dumps(dict(completed=result['completed'],elapsed_s=result['elapsed_s'],error=result.get('error'))))


if __name__=='__main__':main()
