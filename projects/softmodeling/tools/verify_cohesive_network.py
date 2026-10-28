"""Small finite-strain network-fracture equilibrium benchmark with full IPC/CCD.

All internal edges are cohesive candidates. Opposite loading directions may
select different edges; no candidate is seeded, weakened or forced to fail.
This four-triangle example is a discretization verification, not a cutting
prediction or evidence of mesh-independent crack paths.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.cohesive_mixed import MixedModeLaw, interface_response
from prototype.cohesive_network import network_mesh
from prototype.ipc_reference import determinant_step, toolkit
from prototype.material import bulk
from prototype.model_config import FEMConfig, FEMMesh


PARAMETERS = dict(normal_strength_mpa=.02, shear_strength_mpa=.03,
                  normal_stiffness_mpa_per_mm=50., toughness_n_per_mm=.0002)


def _check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Network benchmark case execution budget exhausted')


class NetworkPatch:
    """Four independent triangles, eight free center DOFs, prescribed exterior."""
    def __init__(self, direction_sign=1, gap_mm=.002):
        if direction_sign not in (-1, 1):
            raise ValueError('direction_sign must be +1 or -1')
        self.config = FEMConfig(length_mm=1., height_mm=1., width_mm=1.,
                                inclination_deg=0., shear_modulus_n_mm2=4., poisson_ratio=.25)
        coordinates = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [.5, .5]])
        triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        original = FEMMesh(coordinates, coordinates.copy(), triangles, np.empty((0, 2), int),
                           np.empty(0), np.array([0, 1]), np.empty(0, int), np.zeros(5, int))
        self.mesh, self.topology = network_mesh(original, self.config, gap_mm)
        self.reference = self.mesh.reference_xy.copy()
        self.law = MixedModeLaw(**PARAMETERS)
        self.history = np.zeros((len(self.topology.segments), 2))
        self.fixed_nodes = np.flatnonzero(np.isin(self.topology.original_vertex_ids, self.topology.exterior_nodes))
        self.fixed = (2*self.fixed_nodes[:, None]+np.arange(2)).ravel()
        self.free = np.setdiff1d(np.arange(self.reference.size), self.fixed)
        self.direction = np.array([1., float(direction_sign)])/np.sqrt(2.)
        # A tensile isotropic component keeps the orthogonal candidates open;
        # the larger directional component supplies the unchanged mirror bias.
        self.loading = (self.reference-.5)@(.35*np.eye(2)+np.outer(self.direction, self.direction))
        self.ipc = toolkit()
        tri = self.mesh.triangles
        edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
        self.cmesh = self.ipc.CollisionMesh(self.reference, edges)
        self.cmesh.can_collide = self.ipc.make_static_obstacle_filter(len(self.reference))
        self.dhat = gap_mm/8.
        self.barrier = self.ipc.BarrierPotential(self.dhat, 6.6, True)
        if self.ipc.has_intersections(self.cmesh, self.reference):
            raise ValueError('Reference inset network intersects')
        initial = self.evaluate(self.reference, self.history)
        if np.linalg.norm(initial['gradient']) > 1e-10:
            raise ValueError('Reference network is not stress free')
        self.original_coordinates = coordinates

    def collisions(self, positions):
        collisions = self.ipc.NormalCollisions()
        collisions.use_area_weighting = True
        collisions.collision_set_type = self.ipc.NormalCollisions.IMPROVED_MAX_APPROX
        collisions.build(self.cmesh, positions, self.dhat)
        return collisions

    def evaluate(self, positions, old, tangent=True):
        residual, stiffness, dofs, energy, minimum_j = bulk(
            self.mesh, (positions-self.reference).ravel(), self.config, with_tangent=tangent)
        gradient = np.zeros(positions.size)
        np.add.at(gradient, dofs.ravel(), residual.ravel())
        response = interface_response(positions, self.topology.segments, self.topology.reference_gaps,
                                      self.topology.reference_areas, old, self.law, tangent=tangent,
                                      reference_slips=self.topology.reference_slips)
        collisions = self.collisions(positions)
        barrier_energy = float(self.barrier(collisions, self.cmesh, positions))
        gradient += response['gradient']+self.barrier.gradient(collisions, self.cmesh, positions)
        hessian = None
        if tangent:
            hessian = np.zeros((positions.size, positions.size))
            for local, ids in zip(stiffness, dofs):
                hessian[np.ix_(ids, ids)] += local
            hessian += response['hessian'].toarray()
            hessian += self.barrier.hessian(collisions, self.cmesh, positions,
                                           self.ipc.PSDProjectionMethod.NONE).toarray()
        return dict(potential=float(energy+response['incremental_potential_n_mm']+barrier_energy),
                    bulk_stored=float(energy), cohesive=response, barrier_stored=barrier_energy,
                    gradient=gradient, hessian=hessian, min_j=float(minimum_j),
                    ev_contacts=len(collisions.ev_collisions), vv_contacts=len(collisions.vv_collisions))

    def safe_step(self, positions, increment):
        gradients = self.mesh.bulk_data[1]
        deformation = np.eye(2)+np.einsum('eai,eaJ->eiJ',
                                         (positions-self.reference)[self.mesh.triangles], gradients)
        change = np.einsum('eai,eaJ->eiJ', increment[self.mesh.triangles], gradients)
        return min(determinant_step(deformation, change),
                   float(self.ipc.compute_collision_free_stepsize(self.cmesh, positions, positions+increment)))

    def solve(self, accepted, previous_load, load, deadline):
        """Pure trial solve. The caller alone commits history after acceptance."""
        trial = accepted+(load-previous_load)*self.loading
        if self.safe_step(accepted, trial-accepted) < 1.-1e-12:
            raise RuntimeError('Boundary increment violates full CCD/J bound; refine loading step')
        trial[self.fixed_nodes] = (self.reference+load*self.loading)[self.fixed_nodes]
        old = self.history.copy()
        line_search_evaluations = 0
        for iteration in range(100):
            self.last_trial_positions = trial.copy()
            _check_time(deadline)
            response = self.evaluate(trial, old)
            residual = response['gradient'][self.free]
            norm = float(np.linalg.norm(residual))
            exact = response['hessian'][np.ix_(self.free, self.free)]
            minimum_eigenvalue = float(np.linalg.eigvalsh(exact).min())
            if norm < 1e-8:
                if response['min_j'] <= 0. or self.ipc.has_intersections(self.cmesh, trial):
                    raise RuntimeError('Equilibrium has invalid geometry')
                return trial, response, dict(iterations=iteration, residual_n=norm,
                                              minimum_free_hessian_eigenvalue_n_per_mm=minimum_eigenvalue,
                                              line_search_evaluations=line_search_evaluations)
            shift = max(0., 1e-8-minimum_eigenvalue)
            direction = np.zeros(trial.size)
            direction[self.free] = np.linalg.solve(exact+shift*np.eye(len(self.free)), -residual)
            direction = direction.reshape(trial.shape)
            alpha = self.safe_step(trial, direction)
            slope = float(response['gradient']@direction.ravel())
            for search in range(30):
                _check_time(deadline)
                candidate = trial+alpha*direction
                candidate_response = self.evaluate(candidate, old, tangent=False)
                line_search_evaluations += 1
                # Near equilibrium the elastic energy difference loses digits;
                # a fresh residual decrease within roundoff also globalizes.
                residual_decrease = (np.linalg.norm(candidate_response['gradient'][self.free]) < .5*norm
                                     and candidate_response['potential'] <= response['potential']
                                     +64*np.finfo(float).eps*max(1., abs(response['potential'])))
                if (candidate_response['potential'] <= response['potential']+1e-4*alpha*slope+1e-16
                        or residual_decrease):
                    trial = candidate
                    break
                alpha *= .5
            else:
                raise RuntimeError('Network Newton line search failed')
        raise RuntimeError('Network Newton iteration limit reached')


def _row(load, response, diagnostics):
    cohesive = response['cohesive']
    return dict(load_parameter=float(load), bulk_stored_n_mm=response['bulk_stored'],
                cohesive_stored_n_mm=cohesive['stored_energy_n_mm'],
                fracture_dissipation_n_mm=cohesive['fracture_dissipation_n_mm'],
                ipc_stored_n_mm=response['barrier_stored'], min_j=response['min_j'],
                maximum_damage=float(np.max(cohesive['damage'])),
                failed_interface_ids=np.flatnonzero(np.all(cohesive['damage'] >= 1.-1e-10, axis=1)).tolist(),
                maximum_history_mm=float(np.max(cohesive['history'])),
                ipc_ev_contacts=response['ev_contacts'], ipc_vv_contacts=response['vv_contacts'], **diagnostics)


def run_case(direction_sign=1, *, steps=80, target=.08, seconds=120., gap_mm=.002):
    if direction_sign not in (-1, 1) or type(steps) is not int or steps < 4:
        raise ValueError('Require direction_sign +/-1 and integer steps >=4')
    if not np.isfinite([target, seconds, gap_mm]).all() or not 0 < target <= .2 or not 1 <= seconds <= 120 or gap_mm <= 0:
        raise ValueError('Require target (0,.2], case seconds [1,120], positive gap')
    started = time.monotonic(); deadline = started+seconds
    patch = NetworkPatch(direction_sign, gap_mm)
    positions, history = [patch.reference.copy()], [patch.history.copy()]
    response = patch.evaluate(positions[0], patch.history)
    rows = [_row(0., response, dict(iterations=0, residual_n=float(np.linalg.norm(response['gradient'][patch.free])),
                                  minimum_free_hessian_eigenvalue_n_per_mm=float(np.linalg.eigvalsh(
                                      response['hessian'][np.ix_(patch.free, patch.free)]).min()),
                                  line_search_evaluations=0))]
    rows[0]['generalized_reaction_n_mm'] = float(response['gradient'][patch.fixed]@patch.loading.ravel()[patch.fixed])
    failures = []
    failure_positions, failure_previous, failure_history = [], [], []
    completed = False
    try:
        for load in np.linspace(0., target, steps+1)[1:]:
            trial, response, diagnostics = patch.solve(positions[-1], rows[-1]['load_parameter'], float(load), deadline)
            if np.any(response['cohesive']['history'] < patch.history-1e-14):
                raise RuntimeError('Trial history decreased')
            # Only this complete accepted equilibrium commits irreversible history.
            patch.history = response['cohesive']['history'].copy()
            positions.append(trial.copy()); history.append(patch.history.copy())
            row = _row(load, response, diagnostics)
            row['generalized_reaction_n_mm'] = float(response['gradient'][patch.fixed]@patch.loading.ravel()[patch.fixed])
            rows.append(row)
        completed = True
    except Exception as error:
        failure_positions.append(getattr(patch, 'last_trial_positions', positions[-1]).copy())
        failure_previous.append(positions[-1].copy())
        failure_history.append(history[-1].copy())
        failures.append(dict(attempted_load_parameter=float(load), type=type(error).__name__, error=str(error),
                             accepted_states=len(rows),
                             accepted_history_unchanged=bool(np.array_equal(patch.history, history[-1])),
                             failure_snapshot_index=len(failure_positions)-1))
    # An independent evaluation of every saved equilibrium and history transition.
    maximum_audit_residual, maximum_history_error, minimum_j = 0., 0., float(rows[0]['min_j'])
    audited = False
    audit_error = None
    try:
        for index, (x, accepted_history) in enumerate(zip(positions, history)):
            _check_time(deadline)
            previous = history[index-1] if index else np.zeros_like(accepted_history)
            audited_response = patch.evaluate(x, previous, tangent=False)
            maximum_audit_residual = max(maximum_audit_residual, float(np.linalg.norm(audited_response['gradient'][patch.free])))
            maximum_history_error = max(maximum_history_error, float(np.max(np.abs(audited_response['cohesive']['history']-accepted_history))))
            minimum_j = min(minimum_j, audited_response['min_j'])
            if patch.ipc.has_intersections(patch.cmesh, x):
                raise RuntimeError('Independent audit found an intersection')
        audited = maximum_audit_residual < 1e-8 and maximum_history_error < 1e-12 and minimum_j > 0.
    except Exception as error:
        audit_error = f'{type(error).__name__}: {error}'
    load = np.array([r['load_parameter'] for r in rows])
    reaction = np.array([r['generalized_reaction_n_mm'] for r in rows])
    external = np.r_[0., np.cumsum(.5*(reaction[1:]+reaction[:-1])*np.diff(load))]
    accounted = np.array([r['bulk_stored_n_mm']+r['cohesive_stored_n_mm']+r['fracture_dissipation_n_mm']+r['ipc_stored_n_mm'] for r in rows])
    residual = external-(accounted-accounted[0])
    for row, work, remainder in zip(rows, external, residual):
        row.update(external_work_n_mm=float(work), unaccounted_energy_n_mm=float(remainder))
    relative = float(np.max(np.abs(residual))/max(np.max(np.abs(external)), 1e-14))
    result = dict(direction_sign=direction_sign, prescribed_loading_direction=patch.direction.tolist(),
                  prescribed_boundary_gradient=(.35*np.eye(2)+np.outer(patch.direction, patch.direction)).tolist(),
                  boundary_rule='x = X + load_parameter * (0.35 I + a tensor a) * (X - [0.5,0.5])',
                  steps_requested=steps, target_load_parameter=target, completed=completed, accepted_states=len(rows),
                  elapsed_s=time.monotonic()-started, case_budget_s=seconds, topology=patch.topology.to_dict(),
                  free_dofs=len(patch.free), fixed_dofs=len(patch.fixed), no_seeded_or_weakened_interfaces=True,
                  final_failed_interface_ids=rows[-1]['failed_interface_ids'], rows=rows, failures=failures,
                  audit=dict(passed=audited, maximum_residual_n=maximum_audit_residual,
                             maximum_history_error_mm=maximum_history_error, min_j=minimum_j, error=audit_error),
                  energy=dict(maximum_absolute_residual_n_mm=float(np.max(np.abs(residual))),
                              maximum_relative_residual=relative,
                              qualified=bool(completed and audited and relative < .002),
                              unexplained_remainder_is_not_dissipation=True),
                  stable_accepted_branch=all(r['minimum_free_hessian_eigenvalue_n_per_mm'] > 0. for r in rows),
                  contact=dict(kind='native_full_EV_VV_improved_max', full_ccd=True, frictionless=True,
                               dhat_mm=patch.dhat, reference_gap_mm=gap_mm, ipc_version=patch.ipc.__version__))
    fields = dict(reference=patch.reference, positions=np.asarray(positions), history=np.asarray(history),
                  triangles=patch.mesh.triangles, segments=patch.topology.segments,
                  original_edges=patch.topology.original_edges, original_coordinates=patch.original_coordinates,
                  load_parameter=load,
                  failed_positions=np.asarray(failure_positions).reshape(-1, *patch.reference.shape),
                  failed_previous=np.asarray(failure_previous).reshape(-1, *patch.reference.shape),
                  failed_previous_history=np.asarray(failure_history).reshape(-1, *patch.history.shape))
    return result, fields


def _render(output, report, fields):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    law = MixedModeLaw(**PARAMETERS)
    for axis, case, data in zip(axes[:2], report['cases'][:2], fields[:2]):
        x = data['positions'][-1]
        axis.triplot(x[:, 0], x[:, 1], data['triangles'], color='#68737d', lw=.8)
        damage = law.history_response(data['history'][-1])['damage'].max(axis=1)
        for segment, value in zip(data['segments'], damage):
            points = x[segment].reshape(2, 2, 2).mean(axis=1)
            axis.plot(points[:, 0], points[:, 1], color=plt.cm.inferno(value), lw=4)
        axis.set_aspect('equal'); axis.set_title(f"Loading sign {case['direction_sign']:+d}; failed {case['final_failed_interface_ids']}")
        axis.set(xlabel='x (mm)', ylabel='y (mm)')
    for case in report['cases']:
        axes[2].plot([r['load_parameter'] for r in case['rows']], [r['generalized_reaction_n_mm'] for r in case['rows']],
                     label=f"sign {case['direction_sign']:+d}, {case['steps_requested']} steps")
    axes[2].set(xlabel='Prescribed boundary strain parameter', ylabel='Generalized reaction (N mm)'); axes[2].legend()
    fig.suptitle('All four internal edges compete; center nodes solve equilibrium; numerical example')
    fig.tight_layout(); fig.savefig(output/'network.svg'); plt.close(fig)
    table = ''.join('<tr><td>'+html.escape(g['name'])+'</td><td>'+('PASS' if g['passed'] else 'FAIL')+'</td></tr>' for g in report['gates'])
    text = ('<!doctype html><meta charset="utf-8"><title>Network free-path verification</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 24px}img{max-width:100%}'
            'td{padding:6px 15px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
            '<h1>自由选路网络：二维平衡验证</h1><p>同一有限变形体材料、同一内聚参数，所有内部边都候选。'
            '外边施加镜像倾斜拉伸；中心节点自由求力平衡，没有指定失效边或弱化路径。'
            '完整 EV/VV IPC、CCD、正 J 和全历史独立审查保留。</p>'
            '<p>这是四三角形最小验证，裂纹只能沿网格边选择，不能证明真实裂路、角度或网格无关性。'
            '初始有限间距导致材料面积损失与附加柔度；它不是完整体等价离散。示例参数没有读取实验力数据。</p>'
            '<pre>'+html.escape(json.dumps({key: report.get(key) for key in ('status', 'passed', 'parameters')}, ensure_ascii=False, indent=2))+'</pre>'
            '<img src="network.svg"><table>'+table+'</table><p><a href="verification.json">完整参数、残量、能量和失败记录</a></p>')
    (output/'index.html').write_text(text, encoding='utf-8')


def run(output, *, seconds=120., steps=80, target=.08):
    if not np.isfinite(seconds) or not 1 <= seconds <= 120:
        raise ValueError('seconds must be in [1,120] per case')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    names = ['prototype/cohesive_network.py', 'prototype/cohesive_mixed.py', 'prototype/material.py',
             'prototype/model_config.py', 'prototype/ipc_reference.py', 'mechanics/finite_strain.py',
             'tools/verify_cohesive_network.py']
    hashes = {name: hashlib.sha256((project/name).read_bytes()).hexdigest() for name in names}
    report = dict(schema='softcut-cohesive-network-verification/1', status='running', passed=False,
                  parameter_status='theory_verification_only', no_inverse_fitting=True, experimental_inputs_used=[],
                  physical_onset_depth_prediction=None, interpreter=sys.executable,
                  started_utc=datetime.now(timezone.utc).isoformat(), case_execution_budget_s=seconds,
                  maximum_case_count=4, source_sha256=hashes, parameters=PARAMETERS,
                  cases=[], gates=[], limitations=['Only four original mesh-edge candidates; not mesh-direction convergence.',
                                                   'Finite numerical voids and intrinsic interface compliance.',
                                                   'Finite-strain elastic bulk; no material viscosity or inertia.',
                                                   'Quasistatic branch energy is qualified independently of equilibrium.'])
    fields = []
    with threadpool_limits(limits=1):
        for sign, count in ((1, steps), (-1, steps), (1, 2*steps), (-1, 2*steps)):
            case, data = run_case(sign, steps=count, target=target, seconds=seconds)
            report['cases'].append(case); fields.append(data)
            np.savez_compressed(output/f'case_{len(fields)}.npz', **data)
            (output/'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    def gate(name, passed, **metrics):
        report['gates'].append(dict(name=name, passed=bool(passed), **metrics))
    for index, case in enumerate(report['cases']):
        gate(f'case_{index+1}_complete_equilibrium_and_full_contact', case['completed'] and case['audit']['passed'])
        gate(f'case_{index+1}_energy_balance', case['energy']['qualified'], **case['energy'])
        gate(f'case_{index+1}_stable_branch', case['stable_accepted_branch'])
    first, second, fine_first, fine_second = report['cases']
    selected = [set(case['final_failed_interface_ids']) for case in report['cases']]
    gate('opposite_loading_selects_different_nonempty_paths', bool(selected[0]) and bool(selected[1]) and selected[0] != selected[1])
    gate('selected_paths_reproduce_under_step_halving', selected[0] == selected[2] and selected[1] == selected[3])
    for coarse, fine in ((first, fine_first), (second, fine_second)):
        gate(f"sign_{coarse['direction_sign']}_external_work_refinement",
             fine['energy']['maximum_absolute_residual_n_mm'] <= .6*coarse['energy']['maximum_absolute_residual_n_mm']+1e-13)
    final_hashes = {name: hashlib.sha256((project/name).read_bytes()).hexdigest() for name in names}
    report['source_sha256_at_end'] = final_hashes
    gate('numerical_sources_unchanged', hashes == final_hashes)
    report['passed'] = all(g['passed'] for g in report['gates'])
    report['status'] = 'passed' if report['passed'] else 'failed_gates'
    _render(output, report, fields)
    (output/'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return 0 if report['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=120., help='Per-case budget, at most 120 seconds; four cases')
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--target', type=float, default=.08, help='Dimensionless boundary strain, not blade depth')
    arguments = parser.parse_args(argv)
    return run(arguments.output, seconds=arguments.seconds, steps=arguments.steps, target=arguments.target)


if __name__ == '__main__':
    raise SystemExit(main())
