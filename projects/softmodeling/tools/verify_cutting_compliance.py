"""Bounded actual-blade intact/network compliance comparisons; no experimental inputs.

This tool changes no solver. A prepared intact subclass rebuilds the blade and
collision reference consistently, and borrows the network's tightened CCD/J
method. Both branches use the maintained adaptive controller and fresh-history
audit. A complete small-depth comparison is not full-cutting qualification.
"""
import argparse
import contextlib
import copy
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.cohesive_cut import CohesiveCutProblem, make_cohesive_problem
from prototype.geometry import intact_mesh
from prototype.ipc_reference import BASELINE, adaptive_load_steps
from prototype.ipc_viscoelastic import ViscoelasticProblem
from prototype.material_case import audit, failure_checkpoint, signature
from run_support import background_child_options, save_npz, source_hashes, write_json
from tools.run_cohesive_cut import read_config

PROJECT = Path(__file__).resolve().parents[1]
FORCE_TOLERANCE_N = 1e-4
REACTION_RESOLUTION_N = 10 * FORCE_TOLERANCE_N
MATERIAL = dict(equilibrium_shear_mpa=.14, maxwell_shear_mpa=1.24,
                relaxation_s=1.7, bulk_mpa=23.34333333333331)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def record_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode('utf-8'))


def numerical_hashes():
    result = source_hashes()
    for name in ('tools/verify_cutting_compliance.py', 'tools/run_cohesive_cut.py',
                 'prototype/ipc_baseline.json'):
        result[name] = digest((PROJECT/name).read_bytes())
    return result


def _positive(value, name, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value <= 0 or maximum is not None and value > maximum):
        raise ValueError('Positive finite '+name+' required within the stated range')
    return float(value)


def build_matrix(*, gap_mm, normal_stiffness_mpa_per_mm, compression_stiffness_mpa_per_mm,
                 surfaces=(.02, .01, .005), dhat_mm=2.5e-7, ccd_tolerance_mm=1e-9,
                 target_mm=.01, increment_mm=.00125, minimum_increment_mm=.000625,
                 preparation_clearance_mm=.02, compression_regularization='gap_bounded',
                 local_spacing_mm=.75, coarse_spacing_mm=1.5, fine_halfwidth_mm=.2,
                 nose_facets=24, angle_deg=0., ptc_model_backtracks=12,
                 ptc_residual_guard=False, ptc_shift_policy='residual_ratio'):
    """Same physics/base mesh within each pair; only surface spacing across pairs."""
    for name, value in dict(gap=gap_mm, Kn=normal_stiffness_mpa_per_mm,
            Kc=compression_stiffness_mpa_per_mm, dhat=dhat_mm, ccd=ccd_tolerance_mm,
            target=target_mm, increment=increment_mm, minimum=minimum_increment_mm,
            preparation=preparation_clearance_mm, local=local_spacing_mm,
            coarse=coarse_spacing_mm, halfwidth=fine_halfwidth_mm).items():
        _positive(value, name)
    if not 0 < target_mm <= .05 or not .000625 <= increment_mm <= .05:
        raise ValueError('This pre-cut tool allows target <= .05 mm and increment [.000625,.05] mm')
    if minimum_increment_mm > increment_mm or gap_mm <= 2*dhat_mm:
        raise ValueError('Require minimum <= increment and gap > 2*dhat')
    if ccd_tolerance_mm > min(1e-6, gap_mm/10, dhat_mm/10):
        raise ValueError('CCD tolerance must resolve the gap/dhat and tighten the native default')
    if (compression_regularization not in ('linear', 'gap_bounded')
            or isinstance(angle_deg, bool) or not np.isfinite(angle_deg) or not 0 <= angle_deg < 90
            or type(nose_facets) is not int or nose_facets < 4 or nose_facets % 2
            or type(ptc_model_backtracks) is not int or not 0 <= ptc_model_backtracks <= 30
            or type(ptc_residual_guard) is not bool or ptc_residual_guard and not ptc_model_backtracks
            or ptc_shift_policy not in ('residual_ratio', 'backtrack_aware')
            or ptc_shift_policy == 'backtrack_aware' and not ptc_model_backtracks):
        raise ValueError('Invalid regularization, blade geometry or numerical policy')
    surfaces = tuple(_positive(v, 'surface spacing') for v in surfaces)
    if not surfaces or len(set(surfaces)) != len(surfaces):
        raise ValueError('Nonempty distinct surface spacings required')
    base = dict(schema='softcut-cohesive/3', label='Actual blade damage-disabled compliance control',
        cohesive_kind='mixed_mode_energy', fracture_topology='edge_network', angle_deg=float(angle_deg),
        loading_rate_mm_s=.2, target_mm=target_mm, increment_mm=increment_mm,
        minimum_increment_mm=minimum_increment_mm, preparation_clearance_mm=preparation_clearance_mm,
        ccd_tolerance_mm=ccd_tolerance_mm, reference_gap_mm=gap_mm, dhat_mm=dhat_mm,
        tip_radius_mm=.02, friction_coefficient=.35, network_metric_floor_n_per_mm=1.,
        network_predictor='safe_affine_blend', ptc_model_backtracks=ptc_model_backtracks,
        ptc_residual_guard=ptc_residual_guard, ptc_shift_policy=ptc_shift_policy,
        closure_regularization=('bounded_reversible_compression_penalty'
            if compression_regularization == 'gap_bounded' else 'reversible_compression_penalty'),
        parameter_status='demonstration_only_not_experimentally_calibrated',
        no_experimental_force_fitting=True, material=dict(MATERIAL),
        cohesive=dict(normal_strength_mpa=.2, shear_strength_mpa=.25,
            normal_stiffness_mpa_per_mm=normal_stiffness_mpa_per_mm,
            compression_stiffness_mpa_per_mm=compression_stiffness_mpa_per_mm,
            compression_regularization_mm=gap_mm if compression_regularization == 'gap_bounded' else None,
            toughness_n_per_mm=.006, damage_enabled=False),
        mesh=dict(local_spacing_mm=local_spacing_mm, coarse_spacing_mm=coarse_spacing_mm,
            fine_halfwidth_mm=fine_halfwidth_mm, nose_facets=nose_facets, seam_spacing_mm=.1))
    result = []
    for i, surface in enumerate(surfaces):
        config = copy.deepcopy(base)
        config['mesh']['surface_spacing_mm'] = surface
        for model in ('intact', 'network'):
            result.append(dict(id=f'mesh{i}_{model}', model=model, configuration=copy.deepcopy(config)))
    return result


class PreparedIntactProblem(ViscoelasticProblem):
    """Consistent explicit-clearance intact reference; original bulk/contact solver.

    No history is transplanted. The blade translation is applied before rebuilding
    CollisionMesh, and ``previous`` is reset to that same stress-free reference.
    ``safe_step`` is exactly the maintained tightened network CCD/J method.
    """
    safe_step = CohesiveCutProblem.safe_step

    def __init__(self, config):
        material, mesh = config['material'], config['mesh']
        super().__init__(maxwell_shear=material['maxwell_shear_mpa'],
            relaxation_s=material['relaxation_s'], loading_rate=config['loading_rate_mm_s'],
            bulk_modulus=material['bulk_mpa'], shear_modulus=material['equilibrium_shear_mpa'],
            inclination=config['angle_deg'], mu=config['friction_coefficient'], dhat=config['dhat_mm'],
            local_spacing=mesh['local_spacing_mm'], surface_spacing=mesh['surface_spacing_mm'],
            coarse_spacing=mesh['coarse_spacing_mm'], nose_facets=mesh['nose_facets'],
            tip_radius=config['tip_radius_mm'], fine_halfwidth=mesh['fine_halfwidth_mm'],
            max_iterations=500, predictor='secant', hessian_mode='regularized')
        original_start = self.start
        self.start = -config['preparation_clearance_mm']
        self.rest = self.rest.copy()
        self.rest[self.n:, 1] += original_start-self.start
        self.previous = self.rest.copy()
        edges = np.asarray(self.cmesh.edges).copy()
        self.cmesh = self.ipc.CollisionMesh(self.rest[self.surface], edges)
        self.cmesh.can_collide = self.ipc.make_static_obstacle_filter(
            int(np.count_nonzero(self.surface < self.n)))
        self._connected_boundary_pairs = {tuple(sorted(map(int, e))) for e in edges}
        if self.ipc.has_intersections(self.cmesh, self.rest[self.surface]):
            raise ValueError('Intersecting explicitly prepared intact reference')
        self.network_narrow_phase_ccd = self.ipc.TightInclusionCCD(
            tolerance=config['ccd_tolerance_mm'],
            max_iterations=self.ipc.TightInclusionCCD.DEFAULT_MAX_ITERATIONS,
            conservative_rescaling=self.ipc.TightInclusionCCD.DEFAULT_CONSERVATIVE_RESCALING)
        self.reference_contact_check = self.check_reference_contact()
        self.stagnation_window = 40
        self.ptc_shift_policy = config.get('ptc_shift_policy', 'residual_ratio')
        self.ptc_model_backtracks = config.get('ptc_model_backtracks', 0)
        self.ptc_residual_guard = config.get('ptc_residual_guard', False)

    def _audit_extra_acceptance(self, fresh, record):
        # The base audit already re-integrates the exact Maxwell history. Bind
        # the energy diagnostics as well as its checked vertical reaction.
        material_keys = ('dt_s', 'stored_energy_n_mm', 'equilibrium_energy_n_mm',
                         'overstress_energy_n_mm', 'viscous_dissipation_increment_n_mm')
        for key in material_keys:
            np.testing.assert_allclose(fresh['material'][key], record['material'][key],
                                       rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(fresh['friction_work_n_mm'], record['friction_work_n_mm'],
                                   rtol=1e-10, atol=1e-12)
        return dict(material={key: fresh['material'][key] for key in material_keys},
                    friction_work_n_mm=fresh['friction_work_n_mm'])


def make_problem(spec):
    config = spec['configuration']
    if config['cohesive']['damage_enabled'] is not False:
        raise ValueError('Compliance comparison requires explicitly disabled damage')
    if spec['model'] == 'network':
        return make_cohesive_problem(config)
    if spec['model'] == 'intact':
        return PreparedIntactProblem(config)
    raise ValueError('Choose intact or network')


def describe_problem(problem):
    """Fingerprints include shared pre-inset mesh and actual zero-depth blade."""
    mesh, _ = intact_mesh(problem.config, problem.options, fine_halfwidth_mm=problem.fine_halfwidth)
    points = mesh.reference_xy[mesh.triangles]
    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=1)-points, axis=2)
    heights = 2*mesh.bulk_data[0][:, None]/edge_lengths
    topology = getattr(problem, 'network_topology', None)
    g0 = sum(problem.material.mu_ov)+problem.config.shear_modulus_n_mm2
    bulk = problem.config.bulk_modulus_n_mm2
    young = 9*bulk*g0/(3*bulk+g0)
    poisson = (3*bulk-2*g0)/(2*(3*bulk+g0))
    result = dict(body_nodes=problem.n, elements=len(problem.mesh.triangles),
        base_mesh_sha256=digest(mesh.reference_xy.tobytes()+mesh.triangles.tobytes()),
        base_triangles=len(mesh.triangles), base_nodes=mesh.node_count,
        initial_blade_sha256=digest(problem.rest[problem.n:].tobytes()),
        zero_depth_blade_sha256=digest(problem.prescribed_blade_positions(0.).tobytes()),
        initial_depth_mm=float(problem.start), loading_rate_mm_s=problem.loading_rate,
        base_minimum_altitude_mm=float(heights.min()),
        surface_spacing_over_tip_radius=problem.options.surface_spacing_mm/problem.options.tip_radius_mm,
        area_ratio_to_base=1. if topology is None else topology.bulk_area_ratio,
        initial_young_modulus_mpa=young, initial_plane_strain_modulus_mpa=young/(1-poisson**2))
    if topology is not None:
        result.update(topology=topology.to_dict(),
            max_E_plane_strain_over_Kn_h_alt=result['initial_plane_strain_modulus_mpa']/
                (problem.cohesive_law.normal_stiffness_mpa_per_mm*heights.min()),
            cohesive_law=problem.cohesive_law.to_dict())
    return result


def targets(config):
    return np.r_[0., np.arange(config['increment_mm'], config['target_mm']-1e-12,
                              config['increment_mm']), config['target_mm']]


def certify_saved_transition(problem, previous, positions, *, deadline=None, maximum_pieces=64):
    """Cover a saved straight path using unchanged conservative CCD/J queries.

    TightInclusionCCD may stop at its artificial positive separation buffer,
    even when no true collision occurs. A fraction below one is inconclusive,
    not permission to bypass CCD. Advance through a conservative safe prefix,
    recheck that whole prefix, and query the remaining straight path. Every
    piece must pass full native CCD/J and endpoint intersection tests. These
    are geometry queries only: no equilibrium, dt, or history is advanced.
    """
    if type(maximum_pieces) is not int or not 1 <= maximum_pieces <= 256:
        raise ValueError('Geometric coverage budget must be an integer in [1,256]')
    result = dict(passed=False, covered_fraction=0., initial_query_fraction=None,
        pieces=[], maximum_pieces=maximum_pieces,
        rule='unchanged complete CCD/J; rechecked conservative prefixes covering the same saved straight path',
        physical_substeps_added=False, histories_modified=False)
    previous, positions = np.asarray(previous), np.asarray(positions)
    if (previous.shape != positions.shape or not np.isfinite(previous).all()
            or not np.isfinite(positions).all()):
        result['reason'] = 'Nonfinite or mismatched saved geometry'
        return result
    span = positions-previous

    def geometry(point):
        F = np.eye(2)+np.einsum('eai,eaJ->eiJ',
            (point[:problem.n]-problem.mesh.reference_xy)[problem.mesh.triangles], problem.grad)
        minimum = float(np.linalg.det(F).min())
        intersection = bool(problem.ipc.has_intersections(problem.cmesh, point[problem.surface]))
        if not np.isfinite(minimum) or minimum <= 0. or intersection:
            raise RuntimeError('Invalid endpoint geometry in saved-path coverage')
        return minimum

    alpha, current = 0., previous
    try:
        minimum_start = geometry(current)
        for _ in range(maximum_pieces):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError('Independent saved-path geometry coverage budget')
            fraction = float(problem.safe_step(current, positions-current))
            if result['initial_query_fraction'] is None:
                result['initial_query_fraction'] = fraction if np.isfinite(fraction) else None
            if not np.isfinite(fraction) or not 0. < fraction <= 1.:
                result['reason'] = 'No positive finite safe advance from the native CCD/J query'
                return result
            next_alpha = 1. if fraction == 1. else alpha+.9*fraction*(1.-alpha)
            endpoint = positions if next_alpha == 1. else previous+next_alpha*span
            if (not alpha < next_alpha <= 1. or np.array_equal(current, endpoint)
                    and not np.array_equal(current, positions)):
                result['reason'] = 'No representable geometric progress'
                return result
            # Never assume a returned fraction alone is a certificate for a
            # newly rounded intermediate endpoint. Independently recheck it.
            piece_fraction = float(problem.safe_step(current, endpoint-current))
            if piece_fraction != 1.:
                result['reason'] = 'Conservative geometric prefix did not fully pass its independent CCD/J query'
                return result
            minimum_end = geometry(endpoint)
            result['pieces'].append(dict(alpha0=alpha, alpha1=next_alpha,
                remaining_query_fraction=fraction, subinterval_recheck_fraction=piece_fraction,
                start_intersection=False, end_intersection=False,
                min_j_start=minimum_start, min_j_end=minimum_end))
            alpha, current, minimum_start = next_alpha, endpoint, minimum_end
            result['covered_fraction'] = alpha
            if alpha == 1.:
                result['passed'] = True
                result['reason'] = 'Complete contiguous geometric coverage; all original CCD/J queries passed'
                return result
        result['reason'] = 'Geometric piece budget exhausted before complete coverage'
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        result['reason'] = f'{type(error).__name__}: {error}'
    return result


def coverage_evidence_valid(coverage):
    """Authenticate coverage topology as well as each piece's checked gates."""
    try:
        pieces = coverage['pieces']
        if (coverage.get('passed') is not True or coverage['covered_fraction'] != 1.
                or coverage.get('physical_substeps_added') is not False
                or coverage.get('histories_modified') is not False
                or not 1 <= len(pieces) <= coverage['maximum_pieces'] <= 256):
            return False
        alpha = 0.
        for piece in pieces:
            if (piece['alpha0'] != alpha or not alpha < piece['alpha1'] <= 1.
                    or piece['subinterval_recheck_fraction'] != 1.
                    or piece['start_intersection'] is not False or piece['end_intersection'] is not False
                    or not np.isfinite([piece['min_j_start'], piece['min_j_end']]).all()
                    or min(piece['min_j_start'], piece['min_j_end']) <= 0.):
                return False
            alpha = piece['alpha1']
        return alpha == 1.
    except (KeyError, TypeError, ValueError):
        return False


def path_audit(problem, output, report, deadline):
    """Maintained fresh dual-history audit plus independent saved-transition CCD/J."""
    checked = audit(problem, output, report, destination=output/'audit.pending.json', deadline=deadline)
    previous_cohesive = getattr(problem, 'cohesive_maximum', None)
    if previous_cohesive is not None:
        previous_cohesive = np.zeros_like(previous_cohesive)
    for i, (record, row) in enumerate(zip(report['steps'], checked['rows'])):
        if time.monotonic() > deadline:
            raise TimeoutError('Independent path CCD/history audit wall budget')
        with np.load(output/'states'/f'{i:04d}.npz', allow_pickle=False) as saved:
            x, previous = saved['positions'], saved['previous_positions']
            coverage = certify_saved_transition(problem, previous, x, deadline=deadline)
            if not coverage_evidence_valid(coverage):
                raise RuntimeError('Saved accepted transition fails complete CCD/J coverage audit: '+coverage['reason'])
            observation = _accepted_observations(problem, x)
            for key, value in observation.items():
                np.testing.assert_allclose(record['compliance_observations'][key], value,
                                           rtol=1e-10, atol=1e-12)
            if previous_cohesive is not None:
                np.testing.assert_array_equal(saved['previous_cohesive_maximum'], previous_cohesive)
                previous_cohesive = saved['cohesive_maximum'].copy()
                history = problem.cohesive_law.history_response(previous_cohesive)
                if np.any(history['damage'] != 0) or np.any(history['dissipated'] != 0):
                    raise RuntimeError('Damage-disabled path contains damage or fracture dissipation')
            row.update(transition_safe_fraction=coverage['initial_query_fraction'],
                       transition_path_coverage=coverage, accepted_record_sha256=record_digest(record),
                       physical_time_including_preparation_s=(record['depth_mm']-problem.start)/problem.loading_rate)
    checked.update(full_saved_transition_ccd_j=True, cohesive_history_chain_passed=True,
        damage_disabled_verified=True, force_tolerance_n=FORCE_TOLERANCE_N)
    write_json(output/'audit.json', checked)
    (output/'audit.pending.json').unlink(missing_ok=True)
    return checked


def _accepted_observations(problem, x):
    collisions = problem.collisions(x)
    return dict(ipc_ev_contacts=len(collisions.ev_collisions), ipc_vv_contacts=len(collisions.vv_collisions),
        ipc_barrier_stored_energy_n_mm=float(problem.barrier(collisions, problem.cmesh, x[problem.surface])))


def energy_ledger(steps):
    """Diagnostic trapezoidal blade work; compression is counted inside cohesive once."""
    rows = []
    for before, after in zip(steps[:-1], steps[1:]):
        work = .5*(before['reaction_fz_n']+after['reaction_fz_n'])*(after['depth_mm']-before['depth_mm'])
        stored = after['material']['stored_energy_n_mm']-before['material']['stored_energy_n_mm']
        cohesive = (after.get('fracture', {}).get('cohesive_stored_energy_n_mm', 0.)
                    -before.get('fracture', {}).get('cohesive_stored_energy_n_mm', 0.))
        viscous = after['material']['viscous_dissipation_increment_n_mm']
        friction = after['friction_work_n_mm']
        barrier = (after['compliance_observations']['ipc_barrier_stored_energy_n_mm']
                   -before['compliance_observations']['ipc_barrier_stored_energy_n_mm'])
        rows.append(dict(from_depth_mm=before['depth_mm'], to_depth_mm=after['depth_mm'],
            blade_work_n_mm=work, bulk_stored_change_n_mm=stored, cohesive_stored_change_n_mm=cohesive,
            viscous_dissipation_n_mm=viscous, friction_work_n_mm=friction, barrier_stored_change_n_mm=barrier,
            unaccounted_energy_n_mm=work-stored-cohesive-viscous-friction-barrier))
    return dict(qualified=False, scope='diagnostic trapezoidal work from first accepted state; preparation excluded; no energy-convergence claim', rows=rows)


def run_case(spec, output, *, seconds=120.):
    """One independent worker, with construction/loading and reserved audit budgets."""
    _positive(seconds, 'case budget', 3600.)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'states').mkdir()
    write_json(output/'config.json', spec['configuration'])
    config = read_config(output/'config.json')
    if config['target_mm'] > .05 or config['cohesive']['damage_enabled'] is not False:
        raise ValueError('Small-depth damage-disabled compliance scope required')
    if not np.isclose(BASELINE['force_tolerance_n'], FORCE_TOLERANCE_N, rtol=0, atol=0):
        raise ValueError('Native force tolerance differs from declared qualification')
    source = numerical_hashes()
    reserve = min(60., max(5., .25*seconds), .5*seconds)
    report = dict(schema='actual-blade-compliance-case/1', **copy.deepcopy(spec),
        status='initializing', completed=False, qualified_for_comparison=False, steps=[],
        rejected_load_attempts=[], interpreter=sys.executable, source_sha256=source,
        input_sha256=digest((output/'config.json').read_bytes()), execution_budget_s=seconds,
        solve_budget_s=seconds-reserve, reserved_audit_budget_s=reserve,
        contact=dict(full_native_EV_VV=True, full_CCD=True, positive_J=True,
            ccd_tolerance_mm=config['ccd_tolerance_mm'], conservative_rescaling=.8),
        physical_time=dict(preparation_s=config['preparation_clearance_mm']/.2,
            total_requested_s=(config['preparation_clearance_mm']+config['target_mm'])/.2,
            rule='actual accepted dt = depth increment / .2; preparation included in history'),
        solver_policy=dict(intact='maintained secant, affine retry, plateau 40',
            network='maintained explicit safe_affine_blend and recorded network PTC controls',
            shared_ptc_shift_policy=config.get('ptc_shift_policy', 'residual_ratio'),
            shared_ptc_model_backtracks=config.get('ptc_model_backtracks', 0),
            shared_ptc_residual_guard=config.get('ptc_residual_guard', False),
            physical_acceptance_unchanged=True), experimental_inputs_used=[])
    write_json(output/'result.json', report)
    started = time.monotonic()
    problem, x = None, None
    with (output/'run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), threadpool_limits(limits=1):
        try:
            problem = make_problem(spec)
            problem.ipc.set_num_threads(1)
            report.update(signature=signature(problem), geometry=describe_problem(problem),
                          reference_contact_check=problem.reference_contact_check)
            x, old = problem.rest.copy(), problem.start
            report['status'] = 'running'
            def reject(event):
                report['rejected_load_attempts'].append(event)
                write_json(output/'result.json', report)
            for depth in targets(config):
                for x, record in adaptive_load_steps(problem, x, old, float(depth),
                        started+seconds-reserve, config['minimum_increment_mm'], reject):
                    old = float(record['depth_mm'])
                    record['compliance_observations'] = _accepted_observations(problem, x)
                    record['compliance_solver_observations'] = dict(
                        ptc_shift_summary=copy.deepcopy(getattr(problem, 'ptc_shift_summary', None)),
                        ptc_model_backtrack_summary=copy.deepcopy(getattr(problem, 'ptc_model_backtrack_summary', None)))
                    record['physical_time_including_preparation_s'] = (old-problem.start)/problem.loading_rate
                    save_npz(output/'states'/f'{len(report["steps"]):04d}.npz',
                        reference_xy=problem.rest, triangles=problem.mesh.triangles, positions=x,
                        previous_positions=problem.previous, depth_mm=old, body_nodes=problem.n,
                        **problem.checkpoint_data())
                    report['steps'].append(record)
                    report['last_depth_mm'] = old
                    write_json(output/'result.json', report)
                    print(f'ACCEPT depth={old:.9g} Fz={record["reaction_fz_n"]:.9g}', flush=True)
            report.update(status='completed', completed=True)
        except Exception as error:
            report.update(status='budget_exhausted' if isinstance(error, TimeoutError) else 'failed',
                          error=f'{type(error).__name__}: {error}')
            traceback.print_exc(file=log)
            if problem is not None and x is not None and problem.material_dt is not None:
                failed, context = failure_checkpoint(problem, x)
                save_npz(output/'failed_trial.npz', **failed)
                report['failed_attempt'] = context
        if report['steps']:
            try:
                fresh = make_problem(spec)
                checked = path_audit(fresh, output, report, started+seconds)
                report['audit_passed'] = checked['passed']
            except Exception as error:
                report.update(audit_passed=False, audit_error=f'{type(error).__name__}: {error}')
                write_json(output/'audit.json', dict(passed=False, rows=[], error=report['audit_error']))
                traceback.print_exc(file=log)
    report.update(elapsed_s=time.monotonic()-started, source_sha256_at_end=numerical_hashes(),
                  energy_accounting=energy_ledger(report['steps']))
    report['qualified_for_comparison'] = bool(report['completed'] and report.get('audit_passed')
        and report['source_sha256'] == report['source_sha256_at_end']
        and report['elapsed_s'] <= seconds
        and abs(report.get('last_depth_mm', -1)-config['target_mm']) <= 1e-12)
    report['qualified_prefix_states'] = len(report['steps']) if report.get('audit_passed') else 0
    report['complete_cutting_qualified'] = False
    write_json(output/'result.json', report)
    return report


def authenticate_case(directory, report):
    """Check worker artifacts before using any curve in a comparison."""
    try:
        directory = Path(directory)
        checked = json.loads((directory/'audit.json').read_text(encoding='utf-8'))
        steps, rows = report['steps'], checked['rows']
        if (not steps or len(steps) != len(rows) or not checked.get('passed')
                or not checked.get('full_saved_transition_ccd_j') or not checked.get('damage_disabled_verified')
                or report['source_sha256'] != report['source_sha256_at_end']
                or digest((directory/'config.json').read_bytes()) != report['input_sha256']
                or json.loads((directory/'config.json').read_text(encoding='utf-8')) != report['configuration']):
            return False
        for i, (step, row) in enumerate(zip(steps, rows)):
            if (row['checkpoint_sha256'] != digest((directory/'states'/f'{i:04d}.npz').read_bytes())
                    or row['accepted_record_sha256'] != record_digest(step) or row.get('passed') is not True
                    or not 0 <= row['residual_n'] <= FORCE_TOLERANCE_N or not row['min_j'] > 0
                    or (not coverage_evidence_valid(row['transition_path_coverage'])
                        if 'transition_path_coverage' in row else row['transition_safe_fraction'] != 1.)):
                return False
        return True
    except (KeyError, OSError, ValueError, TypeError):
        return False


def compare_cases(candidate, reference, *, kind='network_to_intact'):
    result = dict(candidate=candidate['id'], reference=reference['id'], kind=kind,
        status='inconclusive', comparison_available=False, terminal_force_ratio=None,
        maximum_absolute_force_difference_n=None, relative_l2_force_difference=None,
        complete_cutting_qualified=False)
    if not all(c.get('qualified_for_comparison') and c.get('artifacts_authenticated')
               and completed_path_evidence(c) for c in (candidate, reference)):
        result['reason'] = 'Both complete small-depth paths and independent authenticated audits are required; prefixes are not compared'
        return result
    a, b = copy.deepcopy(candidate['configuration']), copy.deepcopy(reference['configuration'])
    if kind == 'surface_mesh':
        a['mesh'].pop('surface_spacing_mm'); b['mesh'].pop('surface_spacing_mm')
    elif kind != 'network_to_intact':
        raise ValueError('Unknown comparison kind')
    shared_geometry = ('initial_blade_sha256', 'zero_depth_blade_sha256', 'initial_depth_mm', 'loading_rate_mm_s')
    if kind == 'network_to_intact':
        shared_geometry += ('base_mesh_sha256',)
    if (a != b or candidate['source_sha256'] != reference['source_sha256']
            or any(candidate['geometry'][k] != reference['geometry'][k] for k in shared_geometry)):
        result['reason'] = 'Common physical inputs, base mesh, blade preparation or source fingerprints differ'
        return result
    depth_a = np.array([s['depth_mm'] for s in candidate['steps']])
    depth_b = np.array([s['depth_mm'] for s in reference['steps']])
    dt_a = np.array([s['material']['dt_s'] for s in candidate['steps']])
    dt_b = np.array([s['material']['dt_s'] for s in reference['steps']])
    if (depth_a.shape != depth_b.shape or not np.allclose(depth_a, depth_b, rtol=0, atol=1e-12)
            or not np.allclose(dt_a, dt_b, rtol=1e-12, atol=1e-12)):
        result['reason'] = 'Accepted time/depth schedules differ; no interpolation of histories or truncated-prefix comparison'
        return result
    fa = np.array([s['reaction_fz_n'] for s in candidate['steps']])
    fb = np.array([s['reaction_fz_n'] for s in reference['steps']])
    if not np.isfinite(np.r_[fa, fb]).all() or abs(fb[-1]) <= REACTION_RESOLUTION_N:
        result['reason'] = 'Reference terminal force unresolved at ten times the force residual tolerance'
        return result
    def square_integral(values):
        # Exact integral of the square of the piecewise-linear force curve.
        return float(np.sum((values[:-1]**2+values[:-1]*values[1:]+values[1:]**2)*np.diff(depth_b)/3.))
    norm = np.sqrt(square_integral(fb))
    difference = np.sqrt(square_integral(fa-fb))
    result.update(status='complete_small_depth_comparison', comparison_available=True,
        actual_depth_interval_mm=[float(depth_b[0]), float(depth_b[-1])],
        terminal_force_ratio=float(fa[-1]/fb[-1]),
        terminal_force_difference_n=float(fa[-1]-fb[-1]),
        maximum_absolute_force_difference_n=float(np.max(np.abs(fa-fb))),
        relative_l2_force_difference=float(difference/norm) if norm > 0 else None,
        reaction_resolution_n=REACTION_RESOLUTION_N, mesh_convergence_proven=False,
        reason='Numerical reaction comparison at the declared small target only; damage is disabled')
    return result


def completed_path_evidence(case):
    """Recheck actual extent/clock independently of convenience qualification flags."""
    try:
        config = case['configuration']
        steps = case['steps']
        depth = np.asarray([row['depth_mm'] for row in steps], dtype=float)
        dt = np.asarray([row['material']['dt_s'] for row in steps], dtype=float)
        elapsed = np.asarray([row['physical_time_including_preparation_s'] for row in steps], dtype=float)
        start, speed = -config['preparation_clearance_mm'], config['loading_rate_mm_s']
        return bool(case.get('completed') is True and case.get('status') == 'completed'
            and config['cohesive']['damage_enabled'] is False and speed == .2
            and len(depth) >= 2 and np.isfinite(np.r_[depth, dt, elapsed]).all()
            and abs(depth[0]) <= 1e-12 and abs(depth[-1]-config['target_mm']) <= 1e-12
            and np.all(np.diff(depth) > 0.) and np.all(dt > 0.)
            and np.allclose(dt, np.diff(np.r_[start, depth])/speed, rtol=1e-12, atol=1e-12)
            and np.allclose(elapsed, (depth-start)/speed, rtol=1e-12, atol=1e-12)
            and all(np.any(np.isclose(depth, target, rtol=0, atol=1e-12)) for target in targets(config))
            and abs(case['geometry']['initial_depth_mm']-start) <= 1e-12)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def save_report(output, report):
    write_json(output/'verification.json', report)
    columns = ('candidate', 'reference', 'kind', 'status', 'comparison_available',
               'terminal_force_ratio', 'terminal_force_difference_n',
               'maximum_absolute_force_difference_n', 'relative_l2_force_difference', 'reason')
    with (output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(report['comparisons'])
    with (output/'curves.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'model', 'surface_mm', 'depth_mm', 'physical_time_including_preparation_s',
                         'fz_n', 'residual_n', 'min_j', 'audited_prefix', 'complete_comparison_qualified'])
        for case in report['cases']:
            for i, row in enumerate(case.get('steps', [])):
                writer.writerow([case['id'], case['model'], case['configuration']['mesh']['surface_spacing_mm'],
                    row['depth_mm'], row['physical_time_including_preparation_s'], row['reaction_fz_n'],
                    row['residual_n'], row['min_j'], i < case.get('qualified_prefix_states', 0),
                    case.get('qualified_for_comparison', False)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for case in report['cases']:
        rows = case.get('steps', [])
        if rows:
            qualified = case.get('qualified_for_comparison', False)
            ax.plot([r['depth_mm'] for r in rows], [r['reaction_fz_n'] for r in rows],
                '-' if qualified else ':', marker='.', label=case['id']+('' if qualified else ' (prefix/unqualified)'))
    ax.set(xlabel='Actual accepted blade travel (mm)', ylabel='Unshifted reaction Fz (N)',
           title='Actual-blade damage-disabled small-depth control')
    if ax.lines:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'curves.svg'); plt.close(fig)
    overview = [{k: c.get(k) for k in ('id', 'model', 'status', 'completed', 'qualified_for_comparison',
                   'qualified_prefix_states', 'last_depth_mm', 'error', 'audit_error', 'geometry')} for c in report['cases']]
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Actual-blade compliance</title>'
    page += '<style>body{font:15px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}pre{white-space:pre-wrap}img{max-width:100%}</style>'
    page += '<h1>Actual-blade pre-cut compliance</h1><p>Independent intact and damage-disabled network paths. '
    page += 'Identical blade, material, preparation history and CCD precision. Incomplete paths remain inconclusive. '
    page += 'This is a small-depth numerical comparison, not full cutting, fracture onset, mesh convergence or experimental validation.</p>'
    page += '<p><a href="verification.json">JSON evidence</a> · <a href="summary.csv">Comparisons CSV</a> · <a href="curves.csv">Actual curves CSV</a></p><img src="curves.svg" alt="Actual accepted force paths">'
    page += '<h2>Comparisons</h2><pre>'+html.escape(json.dumps(report['comparisons'], indent=2))+'</pre>'
    page += '<h2>Actual cases</h2><pre>'+html.escape(json.dumps(overview, indent=2))+'</pre></html>'
    (output/'index.html').write_text(page, encoding='utf-8')


def run(output, specs, *, seconds=120.):
    _positive(seconds, 'per-case budget', 3600.)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = numerical_hashes()
    report = dict(schema='actual-blade-compliance-verification/1',
        created_utc=datetime.now(timezone.utc).isoformat(), interpreter=sys.executable,
        source_sha256=source, cases=[], comparisons=[], per_case_budget_s=seconds,
        hard_worker_allowance_s=10., nominal_total_case_budget_s=len(specs)*(seconds+10.),
        experimental_inputs_used=[], no_experimental_force_fitting=True,
        scope='actual-blade damage-disabled small-depth mesh/compliance control',
        complete_cutting_qualified=False, mesh_convergence_proven=False)
    write_json(output/'manifest.json', dict(cases=specs, seconds=seconds, source_sha256=source))
    for spec in specs:
        spec_path = output/f'{spec["id"]}.spec.json'
        write_json(spec_path, spec)
        destination = output/spec['id']
        args = [sys.executable, '-u', '-m', 'tools.verify_cutting_compliance',
                '--case-json', str(spec_path), '--output', str(destination), '--seconds', str(seconds)]
        process_error = None
        with (output/f'{spec["id"]}.worker.log').open('w', encoding='utf-8') as stream:
            try:
                child = subprocess.run(args, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=seconds+10., check=False, **background_child_options())
                if child.returncode:
                    process_error = f'Worker exit {child.returncode}'
            except subprocess.TimeoutExpired:
                process_error = 'Hard case budget exhausted; worker terminated'
        result_path = destination/'result.json'
        if result_path.is_file():
            case = json.loads(result_path.read_text(encoding='utf-8'))
        else:
            case = dict(**spec, status='no_result', completed=False, qualified_for_comparison=False, steps=[])
        case['execution'] = dict(args=args, hard_wall_budget_s=seconds+10., process_error=process_error)
        case['artifacts_authenticated'] = authenticate_case(destination, case)
        case['qualified_for_comparison'] = bool(case.get('qualified_for_comparison') and not process_error
            and case['artifacts_authenticated'] and completed_path_evidence(case)
            and case.get('source_sha256') == source)
        if not case['artifacts_authenticated']:
            case['qualified_prefix_states'] = 0
        report['cases'].append(case)
        write_json(output/'verification.json', report)
        print(f'{spec["id"]}: {case["status"]}; accepted={len(case["steps"])}; comparison_qualified={case["qualified_for_comparison"]}', flush=True)
    by_id = {c['id']: c for c in report['cases']}
    for case in report['cases']:
        if case['model'] == 'network':
            reference = by_id.get(case['id'].replace('_network', '_intact'))
            if reference:
                report['comparisons'].append(compare_cases(case, reference))
    for model in ('intact', 'network'):
        group = sorted([c for c in report['cases'] if c['model'] == model],
                       key=lambda c: c['configuration']['mesh']['surface_spacing_mm'])
        for case in group[1:]:
            report['comparisons'].append(compare_cases(case, group[0], kind='surface_mesh'))
    report['source_sha256_at_end'] = numerical_hashes()
    if source != report['source_sha256_at_end']:
        for comparison in report['comparisons']:
            comparison.update(status='inconclusive', comparison_available=False, reason='Sources changed during study',
                terminal_force_ratio=None, terminal_force_difference_n=None,
                maximum_absolute_force_difference_n=None, relative_l2_force_difference=None)
    report['all_comparisons_available'] = bool(report['comparisons'] and all(c['comparison_available'] for c in report['comparisons']))
    save_report(output, report)
    return 0 if report['all_comparisons_available'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=120., help='Per-case construction/loading/audit budget, max 3600 s')
    parser.add_argument('--case-json', help=argparse.SUPPRESS)
    parser.add_argument('--gap-mm', type=float)
    parser.add_argument('--normal-stiffness', type=float)
    parser.add_argument('--compression-stiffness', type=float)
    parser.add_argument('--surfaces', type=float, nargs='+', default=[.02, .01, .005])
    parser.add_argument('--dhat-mm', type=float, default=2.5e-7)
    parser.add_argument('--ccd-tolerance-mm', type=float, default=1e-9)
    parser.add_argument('--target-mm', type=float, default=.01)
    parser.add_argument('--increment-mm', type=float, default=.00125)
    parser.add_argument('--minimum-increment-mm', type=float, default=.000625)
    parser.add_argument('--preparation-clearance-mm', type=float, default=.02)
    parser.add_argument('--compression-regularization', choices=('linear', 'gap_bounded'), default='gap_bounded')
    parser.add_argument('--local-spacing-mm', type=float, default=.75)
    parser.add_argument('--coarse-spacing-mm', type=float, default=1.5)
    parser.add_argument('--fine-halfwidth-mm', type=float, default=.2)
    parser.add_argument('--nose-facets', type=int, default=24)
    parser.add_argument('--angle-deg', type=float, default=0.)
    parser.add_argument('--ptc-model-backtracks', type=int, default=12)
    parser.add_argument('--ptc-residual-guard', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ptc-shift-policy', choices=('residual_ratio', 'backtrack_aware'), default='residual_ratio')
    parser.add_argument('--prepare-only', action='store_true', help='Write manifest only; do not construct or load FE')
    args = parser.parse_args(argv)
    _positive(args.seconds, 'per-case budget', 3600.)
    if args.case_json:
        run_case(json.loads(Path(args.case_json).read_text(encoding='utf-8')), args.output, seconds=args.seconds)
        return 0
    if any(v is None for v in (args.gap_mm, args.normal_stiffness, args.compression_stiffness)):
        parser.error('Declare --gap-mm, --normal-stiffness and --compression-stiffness explicitly')
    options = vars(args).copy()
    for key in ('output', 'seconds', 'case_json', 'prepare_only'):
        options.pop(key)
    options['normal_stiffness_mpa_per_mm'] = options.pop('normal_stiffness')
    options['compression_stiffness_mpa_per_mm'] = options.pop('compression_stiffness')
    specs = build_matrix(**options)
    if args.prepare_only:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'manifest.json', dict(cases=specs, seconds=args.seconds,
            source_sha256=numerical_hashes(), loading_started=False))
        return 0
    return run(args.output, specs, seconds=args.seconds)


if __name__ == '__main__':
    raise SystemExit(main())
