"""Continuous experimental directional fracture / IPC loading with explicit limits.

This runner attempts the entire requested displacement. It stops honestly if
diffuse damage requires new explicit crack/knife contact, or if energy/force
acceptance fails. A stopped or completed attempt is not experimental validation.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from run_support import source_hashes, save_npz, write_json
from .ipc_reference import IPCProblem, ROOT, BASELINE, BASELINE_PATH
from .material import response
from .directional_fracture import (RationalAT1, relaxed_response, strength_calibration,
                                  best_orientation, CONTINUOUS_ORIENTATION_METHOD)


class ModelBoundaryReached(RuntimeError):
    """Unsupported physical regime, distinct from nonlinear solver failure."""


class EnergyAuditError(RuntimeError):
    pass


class DirectionalFractureProblem(IPCProblem):
    def __init__(self, *args, ell_mm=.04, orientation_samples=16,
                 refine_orientations=False, orientation_lock_damage=1e-8, **kwargs):
        if kwargs.get('volumetric_discretization', 'element') != 'element':
            raise ValueError('Directional fracture currently requires element volumetric discretization; '
                             'mixed damage/pressure coupling is not implemented')
        kwargs.setdefault('friction_linear_solver', 'retry-gmres')
        super().__init__(*args, **kwargs)
        if orientation_samples < 4: raise ValueError('At least four crack orientations required')
        if not np.isfinite(orientation_lock_damage) or not 0 <= orientation_lock_damage <= 1:
            raise ValueError('Orientation lock damage must be in [0, 1]')
        self.orientation_samples = orientation_samples
        self.refine_orientations = bool(refine_orientations)
        self.orientation_lock_damage = float(orientation_lock_damage)
        self.calibration = strength_calibration(self.config)
        self.phase = RationalAT1(self.mesh.reference_xy, self.mesh.triangles, self.config.width_mm,
                                self.config.fracture_energy_n_mm, ell_mm,
                                critical_release_mpa=self.calibration['critical_release_mpa'])
        self.damage = np.zeros(self.n)
        self.normals = np.tile([1., 0.], (len(self.mesh.triangles), 1))
        self.locked = np.zeros(len(self.mesh.triangles), bool)

    def deformation(self, x):
        return np.eye(2)+np.einsum('eai,eaJ->eiJ', (x[:self.n]-self.mesh.reference_xy)[self.mesh.triangles], self.grad)

    def fields(self, x, *, tangent=False):
        F = self.deformation(x)
        q = response(F, self.config, tangent=tangent)
        weights = self.phase.degradation(self.damage)
        damaged = weights < 1.
        if np.any(damaged):
            relaxed = relaxed_response(F[damaged], self.normals[damaged], self.config, tangent=tangent)
            for name in ('energy', 'P', 'cauchy', *(['tangent'] if tangent else [])):
                w = weights[damaged].reshape((-1,)+(1,)*(q[name].ndim-1))
                q[name][damaged] = w*q[name][damaged]+(1-w)*relaxed[name]
        return q

    def bulk_terms(self, x, tangent):
        if not np.any(self.damage): return super().bulk_terms(x, tangent)
        q = self.fields(x, tangent=tangent); V = self.phase.volumes
        r = np.einsum('e,eiJ,eaJ->eai', V, q['P'], self.grad).reshape(-1, 6)
        k = None
        if tangent:
            C = q['tangent'].reshape(-1, 2, 2, 2, 2)
            k = np.einsum('e,eaJ,eiJkL,ebL->eaibk', V, self.grad, C, self.grad, optimize=True).reshape(-1, 6, 6)
        return r, k, self.dofs, float(V@q['energy']), float(q['J'].min())

    def _best_orientation(self, F, cauchy, normals, effective):
        """Identical angular/principal candidates for driving and certification."""
        options = dict(refine=True) if self.refine_orientations else {}
        return best_orientation(F, cauchy, normals, effective, self.config,
                                orientation_samples=self.orientation_samples, **options)

    def release(self, x, *, choose=False):
        F = self.deformation(x); intact = response(F, self.config, tangent=False)
        effective = relaxed_response(F, self.normals, self.config, tangent=False)['energy']
        if choose:
            free = ~self.locked
            if np.any(free):
                # Reference-angle search is global over the BODY, not a blade-line seed.
                start = time.perf_counter()
                best, selected = self._best_orientation(F[free], intact['cauchy'][free],
                                                       self.normals[free], effective[free])
                counts = self.work_counters
                counts['orientation_searches'] = counts.get('orientation_searches', 0)+1
                counts['orientation_search_elements'] = counts.get('orientation_search_elements', 0)+int(free.sum())
                counts['orientation_search_seconds'] = counts.get('orientation_search_seconds', 0.)+time.perf_counter()-start
                self.normals[free] = selected; effective[free] = best
        Y = intact['energy']-effective
        if np.min(Y) < -1e-7: raise RuntimeError('Relaxed energy exceeds intact energy')
        return np.maximum(Y, 0.)

    def orientation_error(self, x, release):
        """Exact maximum of (1-g)*abs(best_release-release), without mutation.

        Driving forces still search the entire unlocked body in ``release``.
        Only this final certification omits intact elements (weight exactly
        zero) and locked elements (no available orientation change).
        ``release`` must be the fixed-orientation release at this same x.
        """
        start = time.perf_counter()
        weights = self.phase.degradation(self.damage)
        active = (~self.locked) & (weights < 1.)
        counts = self.work_counters
        counts['orientation_certifications'] = counts.get('orientation_certifications', 0)+1
        counts['orientation_certification_elements'] = counts.get('orientation_certification_elements', 0)+int(active.sum())
        error = 0.
        if np.any(active):
            triangles = self.mesh.triangles[active]
            F = np.eye(2)+np.einsum('eai,eaJ->eiJ',
                (x[:self.n]-self.mesh.reference_xy)[triangles], self.grad[active])
            intact = response(F, self.config, tangent=False)
            effective = relaxed_response(F, self.normals[active], self.config, tangent=False)['energy']
            best, _ = self._best_orientation(F, intact['cauchy'], self.normals[active], effective)
            best_release = intact['energy']-best
            if np.min(best_release) < -1e-7: raise RuntimeError('Relaxed energy exceeds intact energy')
            error = float(np.max((1-weights[active])*np.abs(np.maximum(best_release, 0.)-release[active])))
        counts['orientation_certification_seconds'] = counts.get('orientation_certification_seconds', 0.)+time.perf_counter()-start
        return error

    def snapshot_history(self):
        return tuple(v.copy() for v in (self.damage, self.normals, self.locked, self.previous))

    def restore_history(self, history):
        self.damage, self.normals, self.locked, self.previous = (v.copy() for v in history)

    def advance(self, x, old, depth, deadline, *, max_outer=40, phase_tolerance=1e-6,
                previous_accepted=None):
        saved = self.snapshot_history(); lower = self.damage.copy(); physical_previous = x.copy()
        self.coupling_history = []
        for name in ('failed_damage', 'failed_normals'):
            if hasattr(self, name): delattr(self, name)
        try:
            current, balance = super().step(x, old, depth, deadline, previous_accepted=previous_accepted)
            intact_onset = None
            # An upper bound on releasable energy avoids orientation searches in
            # the undamaged, strictly subcritical regime. This is not a spatial mask.
            intact_energy = response(self.deformation(current), self.config, tangent=False)['energy']
            subcritical = bool(not np.any(lower) and intact_energy.max() < self.phase.critical_release)
            for outer in range(max_outer):
                if time.monotonic() >= deadline: raise TimeoutError('Coupled loading wall budget')
                if subcritical:
                    # The actual KKT is nonnegative at the lower bound since Y <= W < Yc.
                    Y = np.zeros(len(self.mesh.triangles)); kkt = 0.; direction_error = 0.
                else:
                    Y = self.release(current, choose=True)
                    if outer == 0 and not np.any(lower):
                        stability = self.phase.intact_stability(Y)
                        peak = int(np.argmax(stability['ratio']))
                        tip = current[self.n:][np.argmin(current[self.n:, 1])]
                        support = np.flatnonzero(np.any(self.mesh.triangles == peak, axis=1))
                        strongest = int(support[np.argmax(Y[support])])
                        deformation = self.deformation(current)
                        mode = relaxed_response(deformation[support], self.normals[support], self.config, tangent=False)
                        intact_onset = dict(
                            definition='first admissible variation at d=0, not a visible crack',
                            admissible_descent=bool(stability['kkt_residual_mpa'] > phase_tolerance),
                            phase_tolerance_mpa=phase_tolerance,
                            phase_kkt_residual_at_zero_mpa=stability['kkt_residual_mpa'],
                            peak_ratio=float(stability['ratio'][peak]), peak_node=peak,
                            reference_xz_mm=self.rest[peak].tolist(),
                            current_xz_mm=current[peak].tolist(),
                            offset_from_tip_xz_mm=(current[peak]-tip).tolist(),
                            support_reference_area_mm2=float(self.phase.volumes[support].sum()/self.config.width_mm),
                            support_min_J=float(mode['J'].min()), contributing_elements=support.tolist(),
                            reference_normal_xz=self.normals[strongest].tolist(),
                            closed_mode_elements=int(np.count_nonzero(~mode['opened'])),
                            direction_note='strongest incident element; not a unique crack path',
                            closed_crack_shear_assumption='frictionless relaxation')
                    d, info = self.phase.solve(Y, old=lower, initial=self.damage,
                                               tolerance_mpa=phase_tolerance, seconds=min(120., deadline-time.monotonic()))
                    before_energy = self.phase.terms(self.damage, Y, tangent=False)[0]
                    if info['energy_change_n_mm'] > before_energy+1e-8*max(1., abs(before_energy)):
                        raise RuntimeError('Phase subproblem increased energy')
                    self.damage = d
                    if np.any(d):
                        current, iterations = self.solve_friction(current, depth, deadline)
                        balance = self.acceptance(current, depth, iterations+1, outer+1)
                    # Re-evaluate driving forces and orientation stationarity at the
                    # SAME displacement. Never commit trial damage as physical history.
                    Y = self.release(current)
                    kkt = self.phase.kkt_residual(self.damage, Y, lower)
                    direction_error = self.orientation_error(current, Y)
                record = dict(outer=outer, depth_mm=float(depth), max_damage=float(self.damage.max()),
                              phase_kkt_residual_mpa=kkt, orientation_energy_error_mpa=direction_error,
                              residual_n=balance['residual_n'], subcritical_bound=subcritical)
                self.coupling_history.append(record)
                print('COUPLING '+json.dumps(record), flush=True)
                if kkt <= phase_tolerance and direction_error <= phase_tolerance:
                    np.testing.assert_array_equal(self.previous, physical_previous)
                    balance.update(record)
                    balance['intact_onset'] = intact_onset
                    balance.update(self.energy_state(current, Y))
                    balance.update(self.incremental_work(current))
                    self.locked |= self.damage[self.mesh.triangles].max(axis=1) > self.orientation_lock_damage
                    return current, balance
            raise RuntimeError('Directional fracture coupling budget')
        except Exception:
            self.failed_damage = self.damage.copy(); self.failed_normals = self.normals.copy()
            self.restore_history(saved)
            raise

    def energy_state(self, x, release=None):
        elastic = self.bulk_terms(x, False)[3]
        barrier = float(self.barrier(self.collisions(x), self.cmesh, x[self.surface]))
        if release is None: release = self.release(x)
        fracture = self.phase.energy(self.damage, release)[1]
        return dict(elastic_energy_n_mm=elastic, barrier_energy_n_mm=barrier,
                    fracture_functional_n_mm=fracture, stored_plus_fracture_n_mm=elastic+barrier+fracture)

    def incremental_work(self, x):
        """Independent endpoint virtual work; no pseudo-time or friction energy."""
        delta = (x-self.previous).ravel()
        conservative = self.evaluate(x, tangent=False)[1]
        friction = np.zeros(self.total)
        lag = self.lag(x)
        if lag is not None:
            friction[self.surface_dofs] = self.friction.gradient(
                lag, self.cmesh, (x-self.previous)[self.surface])
        total = conservative+friction
        return dict(conservative_endpoint_work_n_mm=float(conservative@delta),
                    free_residual_work_n_mm=float(total[self.free]@delta[self.free]))


def check_step_budget(previous, current, increment, *, tolerance=.05, absolute=1e-5):
    """Audit endpoint equilibrium and a one-sided discrete energy inequality.

    IPC uses endpoint friction forces, not a time-centred work integrator.
    Even a convex elastic spring has grad(U)_1.dx - delta(U) != 0.
    Do not reject that surplus as energy creation or count it as physical heat.
    The old trapezoidal balance remains an explicitly unresolved diagnostic.
    A successful audit is NOT a time-step convergence or physical energy proof.
    """
    work = .5*(previous['reaction_fz_n']+current['reaction_fz_n'])*increment
    endpoint = current['reaction_fz_n']*increment
    change = current['stored_plus_fracture_n_mm']-previous['stored_plus_fracture_n_mm']
    friction = current['friction_work_n_mm']
    conservative = current['conservative_endpoint_work_n_mm']
    residual = current['free_residual_work_n_mm']
    if (not np.isfinite([work, endpoint, change, friction, conservative, residual,
                         increment, tolerance, absolute]).all()
            or increment < 0 or tolerance < 0 or absolute <= 0):
        raise EnergyAuditError('Non-finite or invalid incremental energy inputs')
    if friction < -1e-12:
        raise EnergyAuditError('Negative friction incremental work')
    # g_free.dx_free + g_prescribed.dx_prescribed = g_con.dx + D.
    virtual_defect = endpoint+residual-conservative-friction
    virtual_limit = 1e-9+1e-8*max(abs(endpoint), abs(conservative), abs(friction))
    if abs(virtual_defect) > virtual_limit:
        raise EnergyAuditError(f'Endpoint virtual work mismatch: defect={virtual_defect:g} Nmm, allowance={virtual_limit:g} Nmm')
    defect = work-change-friction
    endpoint_defect = endpoint-change-friction
    scale = max(abs(endpoint), abs(change), abs(friction), absolute)
    allowance = absolute+tolerance*scale
    if endpoint_defect < -allowance:
        raise EnergyAuditError(f'Increment energy creation: defect={endpoint_defect:g} Nmm, allowance={allowance:g} Nmm')
    return dict(external_work_increment_n_mm=work, energy_balance_defect_n_mm=defect,
                external_endpoint_work_n_mm=endpoint, endpoint_virtual_work_defect_n_mm=virtual_defect,
                conservative_endpoint_excess_n_mm=conservative-change,
                endpoint_energy_surplus_n_mm=endpoint_defect,
                energy_balance_resolved=bool(abs(defect) <= allowance),
                energy_audit='endpoint-work-and-one-sided-energy-v2',
                energy_balance_relative_tolerance=tolerance, energy_balance_absolute_n_mm=absolute)


def bisected_target(old, target, minimum):
    """Reach the specified minimum increment despite binary roundoff."""
    half = (target-old)/2
    roundoff = 16*np.finfo(float).eps*max(1., abs(old), abs(target))
    if half < minimum-roundoff:
        return None
    return old+max(half, minimum)


def check_model_boundary(damage, threshold=.95):
    if np.max(damage) >= threshold:
        raise ModelBoundaryReached('Diffuse-damage limit reached: explicit new crack/knife contact is not implemented; refusing post-fracture continuation')


def save_state(path, p, x, previous, old_damage, depth):
    fields = p.fields(x)
    save_npz(path, reference_xy=p.rest, triangles=p.mesh.triangles, positions=x,
             previous_positions=previous, body_nodes=p.n, depth_mm=depth,
             damage=p.damage, old_damage=old_damage, reference_normals=p.normals,
             orientation_locked=p.locked, cauchy=fields['cauchy'], J=fields['J'])
    peak = int(p.damage.argmax())
    stress = np.maximum(np.linalg.eigvalsh(fields['cauchy'])[:, -1], 0.)
    element = int(stress.argmax())
    tip = x[p.n:][np.argmin(x[p.n:, 1])]
    return dict(damage_diagnostics=damage_diagnostics(p, old_damage),
                peak_damage_node=peak if p.damage.max() > 1e-12 else None,
                peak_damage_reference_xz_mm=p.rest[peak].tolist() if p.damage.max() > 1e-12 else None,
                peak_damage_current_xz_mm=x[peak].tolist() if p.damage.max() > 1e-12 else None,
                peak_damage_offset_from_tip_mm=(x[peak]-tip).tolist() if p.damage.max() > 1e-12 else None,
                maximum_principal_cauchy_mpa=float(stress[element]), peak_stress_element=element,
                out_of_plane_dominant_at_peak_stress=bool(fields['cauchy'][element, 2, 2] > np.linalg.eigvalsh(fields['cauchy'][element, :2, :2])[-1]),
                peak_stress_reference_xz_mm=p.rest[p.mesh.triangles[element]].mean(axis=0).tolist(),
                peak_stress_current_xz_mm=x[p.mesh.triangles[element]].mean(axis=0).tolist())


def damage_diagnostics(p, old_damage):
    """Reference-material measures; thresholds label damage, never crack faces."""
    d = p.phase.validate_damage(p.damage)
    old = p.phase.validate_damage(old_damage)
    if np.any(d < old):
        raise ValueError('Accepted damage must be irreversible')
    tri = p.mesh.triangles
    areas = p.phase.volumes/p.config.width_mm
    measures = []
    for threshold in (.01, .1, .5):
        # Exact area of the union of selected reference elements. This is a
        # reproducible resolution-dependent diagnostic, not a physical length.
        mask = d[tri].mean(axis=1) >= threshold
        previous_mask = old[tri].mean(axis=1) >= threshold
        nodes = np.unique(tri[mask])
        box = None if not len(nodes) else [p.rest[nodes].min(axis=0).tolist(),
                                           p.rest[nodes].max(axis=0).tolist()]
        measures.append(dict(mean_nodal_damage_threshold=threshold,
                             reference_area_mm2=float(areas[mask].sum()),
                             newly_crossed_reference_area_mm2=float(areas[mask & ~previous_mask].sum()),
                             reference_bbox_xz_mm=box, elements=int(mask.sum())))
    return dict(max_damage=float(d.max()),
                damage_weighted_reference_area_mm2=float(p.phase.lumped@d/p.config.width_mm),
                damage_increment_reference_area_mm2=float(p.phase.lumped@(d-old)/p.config.width_mm),
                threshold_regions=measures, material_coordinates=True,
                definition='Diffuse-damage extent; not crack length, opening, or validated nucleation',
                crack_opening_available=False)


def import_intact_window(args, report, deadline):
    """Verify a complete intact prefix before introducing zero damage.

    No re-equilibration at the anchor: that would erase its friction increment.
    Earlier states are screened with the SAME direction rule as this window.
    Any earlier local instability rejects the late zero-damage initialization.
    """
    source = args.source.resolve()
    payload = (source/'result.json').read_bytes()
    origin = json.loads(payload)
    if (origin.get('status') != 'completed' or not origin.get('completed')
            or origin.get('simplifications', {}).get('fracture', True)):
        raise ValueError('Window source must be a completed intact run')
    if origin.get('volumetric_discretization', 'element') != 'element':
        raise ValueError('Window source requires element volumetric discretization')
    increment, target_depth = origin.get('loading_increment_mm'), origin.get('target_mm')
    if (increment is None or target_depth is None or not np.isfinite([increment, target_depth]).all()
            or increment <= 0 or target_depth <= 0):
        raise ValueError('Source must declare positive loading increment and target')
    expected = np.r_[0., np.arange(increment, target_depth-1e-12, increment), target_depth]
    depths = np.array([s['depth_mm'] for s in origin['steps']], float)
    if len(depths) != len(expected) or not np.allclose(depths, expected, atol=1e-12, rtol=0):
        raise ValueError('Source history must be a complete consecutive loading prefix')
    if (abs(origin['last_depth_mm']-depths[-1]) > 1e-12
            or origin['last_checkpoint'] != f'states/{len(depths)-1:04d}.npz'):
        raise ValueError('Source completion metadata mismatch')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest(BASELINE_PATH) != origin['baseline_config_sha256']:
        raise ValueError('Archived IPC baseline settings mismatch')
    for name in ('prototype/material.py', 'mechanics/finite_strain.py'):
        if digest(ROOT/name) != origin['source_sha256'][name]:
            raise ValueError('Material source mismatch: '+name)
    indices = [i for i, s in enumerate(origin['steps']) if abs(s['depth_mm']-args.start_depth) < 1e-12]
    if len(indices) != 1:
        raise ValueError('Start depth must identify exactly one accepted source state')
    anchor = indices[0]
    mesh, sim = origin['mesh_settings'], origin['simplifications']
    if sim.get('friction_smoothing_mm') != BASELINE['friction_smoothing_mm']:
        raise ValueError('Source friction regularization mismatch')
    p = DirectionalFractureProblem(case=origin['case'], mu=origin['config']['friction_coefficient'],
        dhat=origin['dhat_mm'], stiffness=origin['barrier_stiffness_mpa'],
        max_iterations=args.max_iterations, ell_mm=args.ell,
        local_spacing=mesh['local_spacing_mm'], surface_spacing=mesh['surface_spacing_mm'],
        fine_halfwidth=mesh['fine_halfwidth_mm'], nose_facets=sim['nose_facets'],
        tip_radius=sim['tip_radius_mm'], inclination=origin['config']['inclination_deg'],
        hessian_mode=origin['hessian_mode'],
        friction_linear_solver=getattr(args, 'friction_linear_solver', 'direct'),
        orientation_samples=getattr(args, 'orientation_samples', 16),
        refine_orientations=getattr(args, 'refine_orientations', False),
        orientation_lock_damage=getattr(args, 'orientation_lock_damage', 1e-8))
    p.ipc.set_num_threads(args.threads)
    if digest(p.ipc.__file__) != origin['ipc_binary_sha256']:
        raise ValueError('IPC binary mismatch')
    for key, value in origin['config'].items():
        if getattr(p.config, key) != value:
            raise ValueError('Source configuration mismatch: '+key)
    provenance = dict(path=str(source), result_sha256=hashlib.sha256(payload).hexdigest(),
                      source_sha256=origin['source_sha256'], checkpoint_sha256={},
                      config=origin['config'], mesh_settings=mesh,
                      contact_settings=dict(dhat_mm=origin['dhat_mm'],
                          barrier_stiffness_mpa=origin['barrier_stiffness_mpa'],
                          friction_smoothing_mm=sim['friction_smoothing_mm'],
                          tip_radius_mm=sim['tip_radius_mm'], nose_facets=sim['nose_facets'],
                          hessian_mode=origin['hessian_mode']),
                      source_target_mm=origin.get('target_mm'), anchor_index=anchor,
                      physical_and_mesh_parameters='inherited exactly from source',
                      prefix_screen=[], prefix_screen_completed=False,
                      history='intact accepted prefix; zero initial damage; original anchor slip retained')
    report['intact_source'] = provenance
    previous, previous_depth = p.rest.copy(), None
    for i, step in enumerate(origin['steps'][:anchor+1]):
        if time.monotonic() >= deadline:
            raise TimeoutError('Intact prefix screening wall budget')
        depth = float(step['depth_mm'])
        if (not np.isfinite(depth) or (i == 0 and abs(depth) > 1e-12)
                or (previous_depth is not None and depth <= previous_depth)):
            raise ValueError('Source must contain an ordered prefix starting at zero')
        path = source/'states'/f'{i:04d}.npz'
        provenance['checkpoint_sha256'][path.relative_to(source).as_posix()] = digest(path)
        with np.load(path, allow_pickle=False) as state:
            np.testing.assert_array_equal(state['reference_xy'], p.rest)
            np.testing.assert_array_equal(state['triangles'], p.mesh.triangles)
            np.testing.assert_array_equal(state['previous_positions'], previous)
            if int(state['body_nodes']) != p.n or abs(float(state['depth_mm'])-depth) > 1e-12:
                raise ValueError('Source checkpoint geometry/depth mismatch')
            if 'damage' in state.files:
                raise ValueError('Intact source must not contain a damage history')
            x = state['positions'].copy()
        if x.shape != p.rest.shape or not np.isfinite(x).all():
            raise ValueError('Invalid accepted positions')
        fixed = np.setdiff1d(np.arange(2*p.n), p.free)
        knife = p.rest[p.n:].copy()
        knife[:, 1] -= depth-p.start
        if (not np.allclose(x.ravel()[fixed], p.rest.ravel()[fixed], rtol=0, atol=2e-10)
                or not np.allclose(x[p.n:], knife, rtol=0, atol=2e-10)):
            raise ValueError('Source violates fixed boundary or prescribed blade trajectory')
        if not 0 <= step['residual_n'] <= BASELINE['force_tolerance_n']:
            raise ValueError('Source contains an unaccepted force residual')
        F = p.deformation(x)
        J = np.linalg.det(F)
        if J.min() <= 0 or abs(J.min()-step['min_j']) > 1e-8:
            raise ValueError('Source accepted Jacobian mismatch')
        W = response(F, p.config, tangent=False)['energy']
        upper = p.phase.intact_stability(W)
        bounded = bool(W.max() < p.phase.critical_release)
        stability = upper if bounded else p.phase.intact_stability(p.release(x, choose=True))
        record = dict(depth_mm=depth, peak_ratio=float(stability['ratio'].max()),
                      ratio_is_upper_bound=bool(bounded), min_J=float(J.min()),
                      phase_kkt_residual_at_zero_mpa=stability['kkt_residual_mpa'])
        provenance['prefix_screen'].append(record)
        print('PREFIX '+json.dumps(record), flush=True)
        if stability['kkt_residual_mpa'] > 1e-6:
            provenance['earlier_local_instability'] = record
            raise ValueError(f'Zero-damage restart would skip an earlier local instability at {depth:g} mm; choose an earlier start depth')
        if i == anchor:
            p.previous = previous.copy()
            balance = p.acceptance(x, depth, 0, 0)
            np.testing.assert_allclose(balance['reaction_fz_n'], step['reaction_fz_n'], atol=1e-8, rtol=1e-8)
            balance.update(p.energy_state(x), max_damage=0.)
            balance['source_anchor'] = True
            if (source/'result.json').read_bytes() != payload:
                raise ValueError('Source report changed while reading')
            provenance.update(prefix_screen_completed=True, anchor_balance=balance)
            return p, x, depth, balance, (previous_depth, previous) if i else None
        previous, previous_depth = x, depth
    raise ValueError('Source anchor not found')


def run_case(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'states').mkdir()
    window = getattr(args, 'source', None) is not None
    parameters = {key: str(value.resolve()) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
    report = dict(status='initializing', completed=False, completed_full_case=False,
                  target_mm=args.depth, window_start_mm=getattr(args, 'start_depth', None),
                  run_kind='nucleation_window' if window else 'continuous_loading', framework='directional-rational-at1-v1',
                  pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(), parameters=parameters,
                  source_sha256=source_hashes(snapshot=output/'source'), steps=[], retries=[],
                  experimental_validation=False, new_crack_contact=False,
                  scope='continuous diffuse nucleation/process-zone attempt; stops at explicit-crack boundary',
                  diffuse_damage_limit=.95,
                  orientation_lock_damage=getattr(args, 'orientation_lock_damage', 1e-8),
                  orientation_lock_status='numerical history threshold, not physical crack threshold',
                  work_accumulation='window increments only' if window else 'loading increments')
    report['parameters']['output'] = str(output)
    for name in ('prototype/ipc_baseline.json', 'tools/visualize_phase_probe.py', 'requirements.txt'):
        target = output/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        report['source_sha256'][name] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(output/'result.json', report)
    start = time.monotonic(); p = None; x = None; target = None
    with (output/'run.log').open('w', encoding='utf-8', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), threadpool_limits(limits=args.threads):
        try:
            deadline = start+args.seconds if args.seconds > 0 else float('inf')
            if window:
                p, x, old, prev, previous_accepted = import_intact_window(args, report, deadline)
                target = min(args.depth, old+args.increment)
                save_npz(output/'anchor.npz', reference_xy=p.rest, triangles=p.mesh.triangles,
                         positions=x, previous_positions=p.previous, body_nodes=p.n, depth_mm=old,
                         damage=p.damage, reference_normals=p.normals, orientation_locked=p.locked)
                report.update(last_depth_mm=old, last_checkpoint='anchor.npz')
                report['parameters'].update(case=p.case, inclination=p.config.inclination_deg,
                    friction=p.config.friction_coefficient,
                    local_spacing=p.options.refinement_mm)
            else:
                p = DirectionalFractureProblem(case=args.case, mu=args.friction, inclination=args.inclination,
                    tip_radius=.02, dhat=.0025, local_spacing=args.local_spacing,
                    surface_spacing=.005 if args.case == 'oblique' else .1,
                    fine_halfwidth=3., nose_facets=64, hessian_mode='regularized',
                    max_iterations=args.max_iterations, ell_mm=args.ell,
                    orientation_samples=getattr(args, 'orientation_samples', 16),
                    refine_orientations=getattr(args, 'refine_orientations', False),
                    orientation_lock_damage=getattr(args, 'orientation_lock_damage', 1e-8),
                    friction_linear_solver=getattr(args, 'friction_linear_solver', 'retry-gmres'))
                x = p.rest.copy(); old = p.start
                prev = dict(reaction_fz_n=0., stored_plus_fracture_n_mm=p.energy_state(x)['stored_plus_fracture_n_mm'])
                target = 0.; previous_accepted = None
            p.ipc.set_num_threads(args.threads)
            report.update(status='running', config=p.config.to_dict(), calibration=p.calibration,
                          physical_target_mm=p.config.max_indentation_mm,
                          orientation_method=(CONTINUOUS_ORIENTATION_METHOD if p.refine_orientations
                                              else 'discrete_reference_grid_and_retained_principal_seeds'),
                          degradation_a=p.phase.a, ipc_version=p.ipc.__version__,
                          ipc_binary_sha256=hashlib.sha256(Path(p.ipc.__file__).read_bytes()).hexdigest(),
                          nodes=p.n, elements=len(p.mesh.triangles), force_tolerance_n=BASELINE['force_tolerance_n'])
            write_json(output/'config.json', dict(material_and_geometry=p.config.to_dict(), numerical=report['parameters'],
                                                fracture_calibration=p.calibration, assumptions=report['scope'],
                                                source_settings=({key:report['intact_source'][key]
                                                    for key in ('config','mesh_settings','contact_settings')}
                                                    if window else None)))
            write_json(output/'result.json', report)
            print('START '+json.dumps({k:report[k] for k in ('target_mm','nodes','elements','calibration','degradation_a')}), flush=True)
            while old < args.depth-1e-12:
                history = p.snapshot_history(); previous = x.copy(); old_damage = p.damage.copy()
                trial = None; attempt_work = {}; work_before = p.work_counters.copy()
                attempt_start = time.perf_counter()
                try:
                    deadline = start+args.seconds if args.seconds > 0 else float('inf')
                    predictor_history = previous_accepted if getattr(args, 'predictor', 'secant') == 'secant' else None
                    trial, balance = p.advance(x, old, target, deadline, previous_accepted=predictor_history)
                    balance.update(work=attempt_work, predictor=p.last_predictor.copy())
                    try:
                        check_model_boundary(p.damage)
                    except ModelBoundaryReached:
                        balance.update(save_state(output/'model_boundary_state.npz', p, trial, previous, old_damage, target))
                        report['model_boundary_balance'] = balance
                        raise
                    audit = check_step_budget(prev, balance, target-old)
                    balance.update(audit)
                except ModelBoundaryReached:
                    p.restore_history(history)
                    raise
                except (RuntimeError, FloatingPointError, TimeoutError) as exc:
                    failed_damage = getattr(p, 'failed_damage', p.damage).copy()
                    failed_normals = getattr(p, 'failed_normals', p.normals).copy()
                    save_npz(output/'last_rejected_trial.npz', positions=trial if trial is not None else getattr(p, 'trial', x),
                             damage=failed_damage, reference_normals=failed_normals,
                             previous_positions=previous, old_damage=old_damage, depth_mm=target,
                             reference_xy=p.rest, triangles=p.mesh.triangles, body_nodes=p.n)
                    p.restore_history(history)
                    report['retries'].append(dict(from_depth_mm=old, attempted_depth_mm=target,
                        error=f'{type(exc).__name__}: {exc}', work=attempt_work,
                        predictor=getattr(p, 'last_predictor', {}).copy()))
                    next_target = bisected_target(old, target, args.min_increment)
                    if isinstance(exc, TimeoutError) or next_target is None:
                        raise
                    target = next_target
                    print(f'RETRY depth_mm={target:.9g} previous_depth_mm={old:.9g} reason={exc}', flush=True)
                    continue
                finally:
                    attempt_work.update({k:v-work_before.get(k, 0) for k,v in p.work_counters.items()})
                    attempt_work['wall_seconds'] = time.perf_counter()-attempt_start
                    report['work_totals'] = p.work_counters.copy()
                    print('WORK '+json.dumps(dict(from_depth_mm=old, **attempt_work)), flush=True)
                    write_json(output/'result.json', report)
                index = len(report['steps']); x = trial
                checkpoint = output/'states'/f'{index:04d}.npz'
                balance.update(save_state(checkpoint, p, x, previous, old_damage, target))
                onset = balance.get('intact_onset')
                if onset and onset['admissible_descent'] and 'first_local_damage_instability' not in report:
                    report['first_local_damage_instability'] = dict(
                        depth_bracket_mm=[None if old < 0 else old, target], evidence=onset,
                        physical_crack_validated=False)
                if balance['max_damage'] > 1e-4 and 'first_damage_interval_mm' not in report:
                    report['first_damage_interval_mm'] = [old, target]
                    report['first_damage_note'] = 'Numerical diffuse-damage event, not an experimentally visible crack'
                report['steps'].append(balance)
                report.update(last_depth_mm=target, last_checkpoint=checkpoint.relative_to(output).as_posix())
                report['cumulative_external_work_n_mm'] = report.get('cumulative_external_work_n_mm', 0.)+audit['external_work_increment_n_mm']
                report['cumulative_friction_work_n_mm'] = report.get('cumulative_friction_work_n_mm', 0.)+balance['friction_work_n_mm']
                report['cumulative_endpoint_energy_surplus_n_mm'] = report.get('cumulative_endpoint_energy_surplus_n_mm', 0.)+audit['endpoint_energy_surplus_n_mm']
                report['unresolved_energy_steps'] = report.get('unresolved_energy_steps', 0)+int(not audit['energy_balance_resolved'])
                report['energy_balance_resolved'] = report['unresolved_energy_steps'] == 0
                print('ENERGY '+json.dumps(audit), flush=True)
                write_json(output/'result.json', report)
                print(f'ACCEPT depth_mm={target:.6f} Fz_N={balance["reaction_fz_n"]:.9g} max_damage={balance["max_damage"]:.6g} force_residual_N={balance["residual_n"]:.6g} phase_KKT_MPa={balance["phase_kkt_residual_mpa"]:.6g}', flush=True)
                # Commit predictor history only after mechanics, phase, geometry,
                # model-boundary and energy gates have all accepted this increment.
                if old >= 0.: previous_accepted = (old, previous)
                old = target; prev = balance
                target = min(args.depth, 0. if old < -1e-12 else old+args.increment)
            report.update(status='completed_window' if window else 'completed', completed=True,
                          completed_full_case=bool(not window and abs(args.depth-p.config.max_indentation_mm)<1e-12))
        except Exception as exc:
            report.update(status='model_boundary' if isinstance(exc, ModelBoundaryReached) else 'failed',
                          error=f'{type(exc).__name__}: {exc}', attempted_depth_mm=locals().get('target'))
            traceback.print_exc()
        report['elapsed_s'] = time.monotonic()-start
        write_json(output/'result.json', report)
        if report['steps'] and not args.no_visualize:
            try:
                command = [sys.executable, str(output/'source/tools/visualize_phase_probe.py'), '--source', str(output), '--output', str(output/'visualization')]
                with (output/'visualization.log').open('w', encoding='utf-8') as visual_log:
                    subprocess.run(command, stdout=visual_log, stderr=visual_log, check=True)
                report['visualization'] = dict(status='completed')
            except Exception as exc:
                report['visualization'] = dict(status='failed', error=str(exc))
            write_json(output/'result.json', report)
    print(json.dumps({k:report.get(k) for k in ('status','last_depth_mm','error')}), flush=True)
    return report['completed']


def supervise(output):
    """Detached controller records even a native worker crash; no agent polling."""
    output = Path(output)
    spec = json.loads((output/'launch.json').read_text(encoding='utf-8'))
    state_path = output/'job.json'
    if state_path.exists(): raise ValueError('Supervisor has already started')
    state = dict(status='launching', supervisor_pid=os.getpid(), target_mm=spec['target_mm'])
    write_json(state_path, state)
    try:
        with (output/'stdout.log').open('wb') as stdout, (output/'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(spec['command'], cwd=spec['source_root'],
                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                        start_new_session=os.name != 'nt')
            state.update(status='running', worker_pid=process.pid)
            write_json(state_path, state)
            code = process.wait()
        state['exit_code'] = code
        path = output/'case/result.json'
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        status = result.get('status')
        if status == 'model_boundary': state['status'] = status
        elif code == 0 and status in ('completed', 'completed_window') and result.get('completed'): state['status'] = status
        else: state['status'] = 'failed'
        state.update(last_depth_mm=result.get('last_depth_mm'), error=result.get('error'))
        if status not in ('completed','completed_window','failed','model_boundary') or (code != 0 and status in ('completed','completed_window')):
            state['error'] = f'Worker exited with code {code} without consistent completion status'
            if path.exists():
                result.update(status='failed', completed=False, error=state['error'], worker_exit_code=code)
                write_json(path, result)
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    state['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path,
                        help='Completed intact run; inherit its exact physical/mesh settings and verify the prefix')
    parser.add_argument('--start-depth', type=float,
                        help='Accepted source depth before any local damage instability; requires --source')
    parser.add_argument('--inclination', type=float, default=60.)
    parser.add_argument('--depth', type=float, default=3.)
    parser.add_argument('--friction', type=float, default=.35)
    parser.add_argument('--ell', type=float, default=.04)
    parser.add_argument('--orientation-samples', type=int, default=16)
    parser.add_argument('--refine-orientations', action='store_true',
                        help='Add continuous QR energy candidates; does not certify a unique physical crack direction')
    parser.add_argument('--orientation-lock-damage', type=float, default=1e-8,
                        help='Numerical direction-history threshold, retained for sensitivity studies')
    parser.add_argument('--local-spacing', type=float, default=.02)
    parser.add_argument('--increment', type=float, default=.01)
    parser.add_argument('--min-increment', type=float, default=.000625)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-iterations', type=int, default=500)
    parser.add_argument('--predictor', choices=('affine', 'secant'), default='secant',
                        help='Accepted-state secant initial guess with geometry/residual fallback')
    parser.add_argument('--friction-linear-solver', choices=('direct', 'retry-gmres'), default='retry-gmres',
                        help='Optional LU preconditioning within same-state shift retries')
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--case', choices=('oblique','patch'), default='oblique')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-visualize', action='store_true')
    args = parser.parse_args()
    if args.supervise:
        supervise(args.output.resolve())
        return
    if (not all(np.isfinite(v) and v > 0 for v in (args.depth,args.ell,args.local_spacing,args.increment,args.min_increment))
            or args.threads < 1 or args.max_iterations < 1 or not np.isfinite(args.seconds) or args.seconds < 0
            or args.min_increment > args.increment or not 0 <= args.inclination < 90
            or not np.isfinite(args.friction) or args.friction < 0
            or args.orientation_samples < 4 or not np.isfinite(args.orientation_lock_damage)
            or not 0 <= args.orientation_lock_damage <= 1):
        parser.error('Invalid physical or numerical parameters')
    if ((args.source is None) != (args.start_depth is None)
            or (args.source is not None and (not np.isfinite(args.start_depth) or not 0 <= args.start_depth < args.depth))):
        parser.error('--source and --start-depth must be used together, with 0 <= start < depth')
    if args.source is not None:
        args.source = args.source.resolve()
    if args.output.exists(): parser.error('Output must not exist')
    if args.detach:
        output = args.output.resolve(); output.mkdir(parents=True)
        frozen = output/'source'; fingerprints = source_hashes(snapshot=frozen)
        for name in ('prototype/ipc_baseline.json','tools/visualize_phase_probe.py','requirements.txt'):
            path = frozen/name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, path); fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        command = [sys.executable, '-u', '-m', 'prototype.ipc_cutting', '--output', str(output/'case')]
        for name, value in vars(args).items():
            if name not in ('output','detach','supervise','no_visualize','refine_orientations') and value is not None:
                command += ['--'+name.replace('_','-'), str(value)]
        if args.no_visualize: command += ['--no-visualize']
        if args.refine_orientations: command += ['--refine-orientations']
        env = os.environ.copy(); env['PYTHONPATH'] = os.pathsep.join([str(frozen), str(ROOT/'.runtime/ipc-python')])
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): env[key] = str(args.threads)
        spec = dict(command=command, source_sha256=fingerprints, source_root=str(frozen), target_mm=args.depth, status='prepared')
        write_json(output/'launch.json', spec)
        control = [sys.executable, '-u', '-m', 'prototype.ipc_cutting', '--output', str(output), '--supervise']
        with (output/'supervisor.log').open('wb') as stdout:
            process = subprocess.Popen(control, cwd=frozen, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stdout,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                       start_new_session=os.name != 'nt')
        spec.update(pid=process.pid, status='launched', supervisor_command=control)
        write_json(output/'launch.json', spec)
        print(json.dumps(dict(pid=process.pid, output=str(output))))
    else:
        sys.exit(0 if run_case(args) else 1)


if __name__ == '__main__': main()
