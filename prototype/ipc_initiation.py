"""Whole-domain initiation screening of accepted IPC states; no crack growth.

Uses the saved intact equilibrium history without refitting material parameters.
Local peaks and reference-area averages are reported separately. No averaged
criterion is silently chosen as a calibrated material strength law.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from .fracture_criteria import StrengthSurface, principal_tensile_measures
from .material import response
from .model_config import FEMConfig, _rotation_vectors
from run_support import ROOT, save_npz, source_hashes, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InitiationScreen:
    def __init__(self, reference, triangles, config, radii=(.05, .1, .2), strength_surface=None):
        self.reference = np.asarray(reference, float)
        self.triangles = np.asarray(triangles, int)
        self.config, self.strength_surface = config, strength_surface
        X = self.reference[self.triangles]
        edges = np.stack((X[:, 1]-X[:, 0], X[:, 2]-X[:, 0]), axis=-1)
        self.inverse = np.linalg.inv(edges)
        self.areas = np.abs(np.linalg.det(edges))/2
        if np.any(self.areas <= 0):
            raise ValueError('Degenerate reference mesh')
        self.centroids = X.mean(axis=1)
        self.sizes = np.max(np.linalg.norm(X-np.roll(X, 1, axis=1), axis=-1), axis=1)
        es, en = _rotation_vectors(config.inclination_deg)
        self.local = self.centroids @ np.column_stack((es, en))
        tree = cKDTree(self.centroids)
        self.averages = {}
        for radius in radii:
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError('Averaging radii must be finite and positive')
            pairs = tree.sparse_distance_matrix(tree, radius, output_type='coo_matrix')
            weights = self.areas[pairs.col]*(1-(pairs.data/radius)**2)**2
            A = coo_matrix((weights, (pairs.row, pairs.col)), shape=pairs.shape).tocsr()
            self.averages[float(radius)] = diags(1/np.asarray(A.sum(axis=1)).ravel()) @ A

    def fields(self, positions):
        x = np.asarray(positions, float)[self.triangles]
        F2 = np.stack((x[:, 1]-x[:, 0], x[:, 2]-x[:, 0]), axis=-1) @ self.inverse
        material = response(F2, self.config, tangent=False)
        F = np.tile(np.eye(3), (len(x), 1, 1)); F[:, :2, :2] = F2
        f = principal_tensile_measures(F, material['cauchy'])
        f.update(F=F, J=material['J'], cauchy_mpa=material['cauchy'],
                 current_centroids=x.mean(axis=1), energy_density_mpa=material['energy'])
        f['criteria'] = dict(nominal=f['nominal_tensile_mpa']/self.config.tensile_strength_n_mm2,
                             stretch=f['principal_stretches'].max(axis=1)/4.2)
        if self.strength_surface is not None:
            f['criteria']['multiaxial'] = self.strength_surface.index(f['second_piola_mpa'])
        return f

    def summarize(self, fields, values, positions, body_nodes):
        i = int(np.argmax(values)); current = fields['current_centroids'][i]
        knife = positions[body_nodes:]; tip = knife[np.argmin(knife[:, 1])]
        normal = fields['current_normal'][i]
        supported = bool(fields['direction_unique'][i] and abs(normal[2]) < 1e-8)
        # Do not project an out-of-plane failure mode into an arbitrary 2-D crack.
        direction = dict(kind='local tensile-plane diagnostic, not selected crack path',
                         unique=bool(fields['direction_unique'][i]),
                         representable_in_2d=supported,
                         current_normal_xzy=normal.tolist(),
                         reference_normal_xzy=fields['reference_normal'][i].tolist(),
                         current_tangent_xz=([-float(normal[1]), float(normal[0])] if supported else None))
        if not fields['direction_unique'][i]:
            direction.update(current_normal_xzy=None, reference_normal_xzy=None)
        sn = self.local[i]; h = self.sizes[i]
        boundary_distance = min(sn[0], self.config.length_mm-sn[0], sn[1], self.config.height_mm-sn[1])
        return dict(peak_ratio=float(values[i]), element=i,
                    reference_xz_mm=self.centroids[i].tolist(), reference_sn_mm=sn.tolist(),
                    current_xz_mm=current.tolist(), tip_current_xz_mm=tip.tolist(),
                    offset_from_tip_xz_mm=(current-tip).tolist(),
                    element_max_edge_mm=float(h), near_fixed_boundary=bool(sn[1] <= h),
                    near_external_boundary=bool(boundary_distance <= h),
                    J=float(fields['J'][i]),
                    cauchy_principal_mpa=fields['cauchy_principal_mpa'][i].tolist(),
                    nominal_principal_mpa=fields['nominal_principal_mpa'][i].tolist(),
                    principal_plane_area_factors=fields['principal_plane_area_factors'][i].tolist(),
                    principal_stretches=fields['principal_stretches'][i].tolist(),
                    direction=direction, supercritical_count=int(np.count_nonzero(values >= 1.)))

    def side_comparison(self, fields, positions, body_nodes, radius=.2):
        """Symmetric current-coordinate regions; no side is preferred or excluded."""
        knife = positions[body_nodes:]; tip = knife[np.argmin(knife[:, 1])]
        relative = fields['current_centroids']-tip
        metrics = dict(nominal_mpa=fields['nominal_tensile_mpa'],
                       cauchy_tension_mpa=np.maximum(fields['cauchy_principal_mpa'][:, -1], 0.),
                       max_stretch=fields['principal_stretches'].max(axis=1),
                       elastic_energy_density_mpa=fields['energy_density_mpa'])
        output = dict(tip_radius_current_mm=radius, regions={})
        for region, near in [('whole', np.ones(len(relative), bool)),
                             ('near_tip', np.linalg.norm(relative, axis=1) <= radius)]:
            for side, mask in [('left', relative[:, 0] < 0), ('right', relative[:, 0] >= 0)]:
                ids = np.flatnonzero(near & mask)
                summary = dict(element_count=len(ids), metrics={})
                for name, values in metrics.items():
                    if not len(ids):
                        summary['metrics'][name] = None
                        continue
                    i = int(ids[np.argmax(values[ids])])
                    summary['metrics'][name] = dict(peak=float(values[i]), element=i,
                        reference_area_mean=float(np.average(values[ids], weights=self.areas[ids])),
                        offset_from_tip_xz_mm=relative[i].tolist(),
                        reference_sn_mm=self.local[i].tolist())
                output['regions'][f'{region}_{side}'] = summary
        return output


def first_event(previous_depth, depth, evidence):
    if evidence['peak_ratio'] < 1:
        return None
    return dict(kind='unvalidated strength/stretch threshold candidate',
                depth_bracket_mm=[previous_depth, depth],
                already_exceeded_at_first_frame=previous_depth is None,
                evidence=evidence)


class DirectionalOnset:
    """First admissible damage variation of the current rational AT1 model.

    This tests local stability of d=0, not global fracture minimization or a
    visible crack. The P1 nodal support is fixed by the discretization; it is
    not an adjustable spatial averaging radius or a prescribed blade seed.
    """
    def __init__(self, screen, ell_mm=.04, orientation_samples=16):
        from .directional_fracture import RationalAT1, strength_calibration
        self.screen = screen
        self.calibration = strength_calibration(screen.config)
        self.phase = RationalAT1(screen.reference, screen.triangles, screen.config.width_mm,
                                screen.config.fracture_energy_n_mm, ell_mm,
                                critical_release_mpa=self.calibration['critical_release_mpa'])
        if orientation_samples < 4:
            raise ValueError('At least four orientation samples required')
        self.orientation_samples = orientation_samples
        self.normals = np.tile([1., 0.], (len(screen.triangles), 1))

    def evaluate(self, fields, positions, body_nodes):
        from .directional_fracture import best_orientation, relaxed_response
        W = fields['energy_density_mpa']
        # Relaxed energy is nonnegative: nodal W is a rigorous upper bound
        # on the release. Use the coupled solver's same element-bound skip
        # so its retained orientation candidates evolve identically.
        upper = self.phase.intact_stability(W)
        bounded = bool(np.max(W) < self.phase.critical_release)
        if bounded:
            release, stability = W, upper
        else:
            F = fields['F'][:, :2, :2]
            effective = relaxed_response(F, self.normals, self.screen.config, tangent=False)['energy']
            effective, self.normals = best_orientation(
                F, fields['cauchy_mpa'], self.normals, effective, self.screen.config,
                orientation_samples=self.orientation_samples)
            release = W-effective
            if np.min(release) < -1e-7:
                raise ValueError('Relaxed energy exceeds intact energy')
            release = np.maximum(release, 0.)
            stability = self.phase.intact_stability(release)
        i = int(np.argmax(stability['ratio']))
        tri = self.screen.triangles
        incident = np.flatnonzero(np.any(tri == i, axis=1))
        e = int(incident[np.argmax(release[incident])])
        tip = positions[body_nodes:][np.argmin(positions[body_nodes:, 1])]
        es, en = _rotation_vectors(self.screen.config.inclination_deg)
        sn = self.screen.reference[i] @ np.column_stack((es, en))
        result = dict(peak_ratio=float(stability['ratio'][i]), ratio_is_upper_bound=bounded,
            admissible_descent=bool(not bounded and stability['kkt_residual_mpa'] > 1e-6),
            phase_kkt_residual_at_zero_mpa=stability['kkt_residual_mpa'],
            minimum_normalized_gradient_mpa=float(np.min(stability['normalized_gradient_mpa'])),
            peak_node=i, reference_xz_mm=self.screen.reference[i].tolist(), reference_sn_mm=sn.tolist(),
            current_xz_mm=positions[i].tolist(), offset_from_tip_xz_mm=(positions[i]-tip).tolist(),
            nodal_release_mpa=float(stability['nodal_release_mpa'][i]),
            contributing_elements=incident.tolist(), strongest_contributing_element=e,
            support_reference_area_mm2=float(self.screen.areas[incident].sum()),
            reference_normal_xz=None if bounded else self.normals[e].tolist(),
            direction_note='normal of strongest incident element; not a unique nodal crack direction',
            orientation_scope='enumerated angles plus principal and retained normals; not a continuous-angle certificate',
            near_fixed_boundary=bool(abs(sn[1]) < 1e-8))
        return result, release, stability['ratio']


class IntactInitiationMonitor:
    """Read-only first-onset diagnostic attached to accepted intact states.

    No damage solve, material update, contact update or stopping decision is
    exposed by this object. Once the directional event has been recorded its
    expensive orientation search is retired; inexpensive full-domain strength
    and stretch diagnostics continue on the counterfactual intact branch.
    Call only after mechanics has accepted a state, never for Newton trials.
    """
    def __init__(self, reference, triangles, config, output, *, radii=(),
                 directional_onset=True, ell_mm=.04, orientation_samples=16,
                 strength_surface=None, volumetric_discretization='element'):
        if volumetric_discretization != 'element':
            raise ValueError('Online initiation requires element pressure; mixed pressure is unsupported')
        if not np.isfinite(config.speed_mm_s) or config.speed_mm_s <= 0:
            raise ValueError('Positive finite loading speed required for the model clock')
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError('Initiation output must be empty')
        self.screen = InitiationScreen(np.array(reference, float, copy=True),
                                       np.array(triangles, int, copy=True), config,
                                       radii, strength_surface)
        self.onset = DirectionalOnset(self.screen, ell_mm, orientation_samples) if directional_onset else None
        self.primary_key = 'directional_at1_local_stability' if self.onset else 'nominal_local'
        self.previous_depth = None
        self.report = dict(
            status='running', completed=False, mechanics_feedback=False, fracture_evolution=False,
            physical_nucleation_validated=False, primary_criterion=self.primary_key,
            search_domain='all body elements/nodal supports; no prescribed crack or blade seed',
            post_onset_policy='continue unchanged intact mechanics; later forces are counterfactual',
            depth_definition='accepted solver displacement; initial clearance is not subtracted',
            model_clock=dict(speed_mm_s=float(config.speed_mm_s),
                definition='accepted solver displacement / prescribed speed',
                provenance='repository loading speed, not independently verified from experiment',
                experimental_contact_time_zero_verified=False),
            nominal_strength_mpa=float(config.tensile_strength_n_mm2),
            nominal_strength_status='catalogue uniaxial; multiaxial extension assumed',
            stretch_threshold=4.2,
            stretch_status='catalogue 320% elongation; independent diagnostic, not an AND gate',
            multiaxial_strength=strength_surface.to_dict() if strength_surface else None,
            averaging_radii_reference_mm=list(radii),
            averaging_status='diagnostic sensitivity, not calibrated process-zone lengths',
            directional_onset_enabled=bool(self.onset),
            directional_onset_definition=('first loss of d=0 local one-sided stability on the intact branch; '
                'P1 assembled variation for enumerated orientations, KKT tolerance 1e-6 MPa; '
                'conditional diffuse onset, not visible crack or propagation') if self.onset else None,
            directional_ell_mm=float(ell_mm) if self.onset else None,
            orientation_samples=int(orientation_samples) if self.onset else None,
            directional_calibration=self.onset.calibration if self.onset else None,
            events={}, steps=[], diagnostic_seconds=0.)
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output/'result.json', self.report)

    def observe(self, depth_mm, positions, body_nodes, *, checkpoint=None):
        """Record an accepted state; return a detached, JSON-ready summary.

        The input arrays are never mutated. Depths must increase, so duplicated
        accepted frames cannot silently corrupt the first-crossing interval.
        """
        started = time.perf_counter()
        depth = float(depth_mm)
        if self.report['status'] != 'running':
            raise ValueError('Cannot observe a finished initiation monitor')
        if not np.isfinite(depth) or depth < 0 or (self.previous_depth is not None and depth <= self.previous_depth):
            raise ValueError('Accepted diagnostic depths must be finite, nonnegative and strictly increasing')
        x = np.asarray(positions, float)
        n = int(body_nodes)
        if (n != len(self.screen.reference) or x.ndim != 2 or x.shape[1] != 2
                or len(x) <= n or not np.isfinite(x).all()):
            raise ValueError('Expected finite body and blade positions on the original mesh')
        f = self.screen.fields(x[:n])
        if np.min(f['J']) <= 0:
            raise ValueError('Nonpositive J in accepted initiation state')
        summaries, new_events = {}, {}
        arrays = {}
        for criterion, raw in f['criteria'].items():
            variants = {'local': raw}
            variants.update({f'average_r{radius:g}': A@raw for radius, A in self.screen.averages.items()})
            for variant, values in variants.items():
                key = f'{criterion}_{variant}'
                evidence = self.screen.summarize(f, values, x, n)
                evidence['averaged_peak_direction_note'] = 'direction and stress at peak element are local, not averaged'
                summaries[key] = evidence
                event = first_event(self.previous_depth, depth, evidence)
                if event and key not in self.report['events']:
                    event['supercritical_reference_area_mm2'] = float(self.screen.areas[values >= 1].sum())
                    new_events[key] = event
                    arrays[f'{key}_ratio'] = values
        onset_evidence = None
        if self.onset is not None and self.primary_key not in self.report['events']:
            onset_evidence, release, nodal_ratio = self.onset.evaluate(f, x, n)
            if onset_evidence['admissible_descent']:
                new_events[self.primary_key] = dict(
                    kind='conditional diffuse onset: intact branch loses local damage stability',
                    depth_bracket_mm=[self.previous_depth, depth],
                    already_exceeded_at_first_frame=self.previous_depth is None, evidence=onset_evidence)
                arrays.update(release_mpa=release, nodal_ratio=nodal_ratio,
                              reference_normals=self.onset.normals)
        if new_events:
            candidate_file = f'candidate_{len(self.report["steps"]):04d}.npz'
            save_npz(self.output/candidate_file, reference_xy=self.screen.reference,
                     triangles=self.screen.triangles, positions=x, body_nodes=n,
                     current_centroids=f['current_centroids'], depth_mm=depth, **arrays)
            for key, event in new_events.items():
                event.update(checkpoint=checkpoint, fields_file=candidate_file,
                    model_time_bracket_s=[None if d is None else d/self.screen.config.speed_mm_s
                                          for d in event['depth_bracket_mm']])
                self.report['events'][key] = copy.deepcopy(event)
                print(f'INITIATION_CANDIDATE criterion={key} bracket_mm={event["depth_bracket_mm"]} '
                      f'ratio={event["evidence"]["peak_ratio"]:.6g} mechanics_unchanged=True', flush=True)
        primary = self.report['events'].get(self.primary_key)
        row = dict(depth_mm=depth, model_time_s=depth/self.screen.config.speed_mm_s,
                   checkpoint=checkpoint, min_J=float(f['J'].min()), criteria=summaries,
                   directional_onset=onset_evidence,
                   directional_evaluation_performed=onset_evidence is not None,
                   directional_orientation_search_performed=bool(onset_evidence is not None
                       and not onset_evidence['ratio_is_upper_bound']),
                   directional_search_retired=bool(self.onset and primary is not None),
                   branch='intact_counterfactual_after_candidate' if primary else 'intact_before_candidate',
                   side_comparison=self.screen.side_comparison(f, x, n),
                   new_event_keys=list(new_events))
        self.report['steps'].append(row)
        self.previous_depth = depth
        self.report.update(last_depth_mm=depth, primary_event=copy.deepcopy(primary))
        self.report['diagnostic_seconds'] += time.perf_counter()-started
        write_json(self.output/'result.json', self.report)
        return copy.deepcopy(row)

    def summary(self):
        """Compact status for the parent run report without duplicating history."""
        return copy.deepcopy(dict(status=self.report['status'],
            accepted_states_analyzed=len(self.report['steps']), last_depth_mm=self.previous_depth,
            primary_criterion=self.primary_key, primary_event=self.report.get('primary_event'),
            physical_nucleation_validated=False, mechanics_feedback=False,
            post_onset_policy=self.report['post_onset_policy'],
            diagnostic_seconds=self.report['diagnostic_seconds']))

    def finish(self, status, error=None):
        if status not in ('completed', 'failed', 'interrupted'):
            raise ValueError('Expected final mechanics status')
        self.report.update(status=status, completed=status == 'completed')
        if error is not None:
            self.report['source_error'] = str(error)
        write_json(self.output/'result.json', self.report)
        return self.summary()


def scan(source, output, radii=(.05, .1, .2), strength_surface=None, threads=2, *,
         allow_failed_prefix=False, energy_probe_depths=(), energy_lengths=(.05, .1, .2),
         directional_onset=False, ell_mm=.04, orientation_samples=16, stop_at_onset=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', completed=False, fracture_evolution=False,
                  physical_nucleation_validated=False, source=str(source), pid=os.getpid(),
                  threads=threads, averaging_radii_reference_mm=list(radii),
                  averaging_status='diagnostic sensitivity, not calibrated process-zone lengths',
                  nominal_strength_mpa=4.4, nominal_strength_status='catalogue uniaxial; multiaxial extension assumed',
                  stretch_threshold=4.2, stretch_status='catalogue 320% elongation; independent diagnostic, not an AND gate',
                  multiaxial_strength=(strength_surface.to_dict() if strength_surface else None),
                  source_sha256=source_hashes(snapshot=output/'source'), steps=[], events={}, checkpoints_sha256={})
    report.update(allow_failed_prefix=allow_failed_prefix, energy_probes=[],
                  energy_probe_depths_mm=list(energy_probe_depths), energy_lengths_mm=list(energy_lengths))
    report.update(directional_onset_enabled=directional_onset, stop_at_onset=stop_at_onset,
                  directional_ell_mm=ell_mm if directional_onset else None,
                  orientation_samples=orientation_samples if directional_onset else None)
    write_json(output/'result.json', report)
    with (output/'run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), threadpool_limits(limits=threads):
        try:
            payload = (source/'result.json').read_bytes()
            origin = json.loads(payload)
            prefix = not origin.get('completed')
            if prefix and not (allow_failed_prefix and origin.get('status') == 'failed'):
                raise ValueError('Scan requires a completed source, not an active/failed run')
            if not prefix and origin.get('status') != 'completed':
                raise ValueError('Inconsistent source completion status')
            if origin['simplifications']['fracture']:
                raise ValueError('Expected intact elastic history')
            if origin.get('volumetric_discretization', 'element') != 'element':
                raise ValueError('Initiation screening for mixed pressure histories is not implemented; '
                                 'element-local stress/energy would be inconsistent')
            for name in ('prototype/material.py', 'mechanics/finite_strain.py'):
                if digest(ROOT/name) != origin['source_sha256'][name]:
                    raise ValueError(f'Material source mismatch: {name}')
            report.update(source_result_sha256=hashlib.sha256(payload).hexdigest(),
                          source_completed=not prefix, source_status=origin['status'],
                          source_error=origin.get('error'), source_target_mm=origin['target_mm'],
                          coverage='accepted prefix only' if prefix else 'complete source history',
                          source_config=origin['config'], source_simplifications=origin['simplifications'],
                          depth_definition='source solver displacement; initial clearance is not subtracted',
                          time_status='not inferred: source does not establish experimental contact-time zero')
            (output/'source_result.json').write_bytes(payload)
            config = FEMConfig(**origin['config'])
            report['nominal_strength_mpa'] = config.tensile_strength_n_mm2
            expected = np.r_[0., np.arange(origin['loading_increment_mm'], origin['target_mm']-1e-12,
                                          origin['loading_increment_mm']), origin['target_mm']]
            if prefix:
                if not 0 < len(origin['steps']) < len(expected):
                    raise ValueError('Failed source must have a nonempty, incomplete accepted prefix')
                expected = expected[:len(origin['steps'])]
            np.testing.assert_allclose([s['depth_mm'] for s in origin['steps']], expected, rtol=0, atol=1e-12)
            for depth in energy_probe_depths:
                if not np.any(np.isclose(expected, depth, rtol=0, atol=1e-12)):
                    raise ValueError(f'Energy probe depth is not an accepted checkpoint: {depth}')
            previous_depth, previous_x, screen, onset = None, None, None, None
            for i, step in enumerate(origin['steps']):
                if not np.isfinite(step['residual_n']) or step['residual_n'] > origin['force_tolerance_n']:
                    raise ValueError('Source contains unaccepted force residual')
                file = source/'states'/f'{i:04d}.npz'
                report['checkpoints_sha256'][file.relative_to(source).as_posix()] = digest(file)
                with np.load(file) as state:
                    x, X = state['positions'].copy(), state['reference_xy'].copy()
                    tri, n = state['triangles'].copy(), int(state['body_nodes'])
                    np.testing.assert_allclose(float(state['depth_mm']), step['depth_mm'], rtol=0, atol=1e-12)
                    np.testing.assert_array_equal(state['previous_positions'], X if previous_x is None else previous_x)
                if not np.isfinite(x).all() or x.shape != X.shape:
                    raise ValueError('Invalid checkpoint positions')
                if screen is None:
                    screen = InitiationScreen(X[:n], tri, config, radii, strength_surface)
                    reference, triangles, count = X, tri, n
                    if directional_onset:
                        onset = DirectionalOnset(screen, ell_mm, orientation_samples)
                        report['directional_calibration'] = onset.calibration
                        report['directional_onset_definition'] = (
                            'first loss of d=0 local one-sided stability on the saved intact branch; '
                            'P1 assembled variation for enumerated orientations, KKT tolerance 1e-6 MPa; '
                            'conditional diffuse initiation, not physical crack or growth')
                else:
                    np.testing.assert_array_equal(X, reference)
                    np.testing.assert_array_equal(tri, triangles)
                    if n != count: raise ValueError('Body node count changed')
                f = screen.fields(x[:n])
                if np.min(f['J']) <= 0: raise ValueError('Nonpositive J')
                np.testing.assert_allclose(f['J'].min(), step['min_j'], rtol=1e-9, atol=1e-10)
                if any(abs(step['depth_mm']-d) < 1e-12 for d in energy_probe_depths):
                    from .phase_field import FixedDisplacementAT1
                    for ell in energy_lengths:
                        probe = FixedDisplacementAT1(X[:n], tri, config.width_mm,
                                                    config.fracture_energy_n_mm, ell)
                        damage, info = probe.solve(f['energy_density_mpa'])
                        peak = int(np.argmax(damage)) if damage.max() > 1e-12 else None
                        tip = x[n:][np.argmin(x[n:, 1])]
                        info.update(depth_mm=step['depth_mm'], ell_mm=ell,
                                    peak_node=peak, peak_current_xz_mm=x[peak].tolist() if peak is not None else None,
                                    peak_reference_xz_mm=X[peak].tolist() if peak is not None else None,
                                    peak_offset_from_tip_xz_mm=(x[peak]-tip).tolist() if peak is not None else None)
                        report['energy_probes'].append(info)
                        save_npz(output/f'energy_depth{step["depth_mm"]:g}_ell{ell:g}.npz',
                                 reference_xy=X, positions=x, triangles=tri, damage=damage,
                                 body_nodes=n, depth_mm=step['depth_mm'])
                        print(f'ENERGY_PROBE {json.dumps(info)}', flush=True)
                summaries = {}
                onset_evidence = None
                if onset is not None:
                    onset_evidence, release, nodal_ratio = onset.evaluate(f, x, n)
                    key = 'directional_at1_local_stability'
                    if onset_evidence['admissible_descent'] and key not in report['events']:
                        report['events'][key] = dict(
                            kind='conditional diffuse onset: intact branch loses local damage stability',
                            depth_bracket_mm=[previous_depth, step['depth_mm']],
                            already_exceeded_at_first_frame=previous_depth is None,
                            checkpoint=file.relative_to(source).as_posix(), evidence=onset_evidence)
                        save_npz(output/'directional_onset_candidate.npz', positions=x, reference_xy=X,
                                 triangles=tri, body_nodes=n, depth_mm=step['depth_mm'],
                                 release_mpa=release, nodal_ratio=nodal_ratio,
                                 reference_normals=onset.normals)
                        print(f'DIRECTIONAL_ONSET bracket_mm={[previous_depth, step["depth_mm"]]} '
                              f'node={onset_evidence["peak_node"]} ratio={onset_evidence["peak_ratio"]:.6g}', flush=True)
                for criterion, raw in f['criteria'].items():
                    variants = {'local': raw}
                    variants.update({f'average_r{radius:g}': A@raw for radius, A in screen.averages.items()})
                    for variant, values in variants.items():
                        key = f'{criterion}_{variant}'
                        evidence = screen.summarize(f, values, x, n)
                        evidence['averaged_peak_direction_note'] = 'direction and stress at peak element are local, not averaged'
                        summaries[key] = evidence
                        event = first_event(previous_depth, step['depth_mm'], evidence)
                        if event and key not in report['events']:
                            event['checkpoint'] = file.relative_to(source).as_posix()
                            event['supercritical_reference_area_mm2'] = float(screen.areas[values >= 1].sum())
                            report['events'][key] = event
                            # Save only first-crossing fields, not a second copy of the entire history.
                            save_npz(output/f'{key}_candidate.npz', positions=x, reference_xy=X,
                                     triangles=tri, ratio=values, raw_ratio=raw,
                                     supercritical_elements=np.flatnonzero(values >= 1),
                                     current_centroids=f['current_centroids'],
                                     reference_centroids=screen.centroids, depth_mm=step['depth_mm'])
                            print(f'CANDIDATE criterion={key} bracket_mm={event["depth_bracket_mm"]} '
                                  f'element={evidence["element"]} ratio={evidence["peak_ratio"]:.6g}', flush=True)
                report['steps'].append(dict(depth_mm=step['depth_mm'], reaction_fz_n=step['reaction_fz_n'],
                                            min_J=float(f['J'].min()), criteria=summaries,
                                            directional_onset=onset_evidence,
                                            side_comparison=screen.side_comparison(f, x, n)))
                previous_depth, previous_x = step['depth_mm'], x
                write_json(output/'result.json', report)
                print(f'SCAN depth_mm={previous_depth:.6f} nominal_peak={summaries["nominal_local"]["peak_ratio"]:.6g}', flush=True)
                if stop_at_onset and 'directional_at1_local_stability' in report['events']:
                    break
            report.update(status='completed', completed=True, last_depth_mm=previous_depth,
                          analyzed_entire_source=len(report['steps']) == len(origin['steps']),
                          note='Later intact states are counterfactual after candidate onset; no crack growth or unloading is computed.')
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            print(report['error'], flush=True)
        write_json(output/'result.json', report)
    return report['completed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radii', type=float, nargs='*', default=[.05, .1, .2])
    parser.add_argument('--strength-surface', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--allow-failed-prefix', action='store_true',
                        help='Analyze only accepted states from a failed run; never its failed trial')
    parser.add_argument('--energy-probe-depths', type=float, nargs='+', default=[])
    parser.add_argument('--energy-lengths', type=float, nargs='+', default=[.05, .1, .2])
    parser.add_argument('--directional-onset', action='store_true')
    parser.add_argument('--ell', type=float, default=.04)
    parser.add_argument('--orientation-samples', type=int, default=16)
    parser.add_argument('--stop-at-onset', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or not all(np.isfinite(r) and r > 0 for r in args.radii):
        parser.error('Require positive thread count and finite positive radii')
    if len(set(args.radii)) != len(args.radii): parser.error('Radii must be distinct')
    if not np.isfinite(args.ell) or args.ell <= 0 or args.orientation_samples < 4:
        parser.error('Positive ell and at least four orientation samples required')
    if args.stop_at_onset and not args.directional_onset:
        parser.error('--stop-at-onset requires --directional-onset')
    if (not all(np.isfinite(d) and d >= 0 for d in args.energy_probe_depths)
            or not all(np.isfinite(e) and e > 0 for e in args.energy_lengths)):
        parser.error('Invalid probe depths or lengths')
    if args.detach:
        if args.output.exists(): parser.error('Detached output must not exist')
        # Freeze the worker before launch, not after an intervening workspace edit.
        args.output.mkdir(parents=True)
        frozen = args.output.resolve()/'source'
        source_hashes(snapshot=frozen)
        output = args.output.resolve()
        command = [sys.executable, '-u', '-m', 'prototype.ipc_initiation',
                   '--source', str(args.source.resolve()), '--output', str(output/'analysis'),
                   '--threads', str(args.threads), '--radii', *map(str, args.radii)]
        if args.allow_failed_prefix: command += ['--allow-failed-prefix']
        if args.directional_onset:
            command += ['--directional-onset', '--ell', str(args.ell),
                        '--orientation-samples', str(args.orientation_samples)]
        if args.stop_at_onset: command += ['--stop-at-onset']
        if args.energy_probe_depths:
            command += ['--energy-probe-depths', *map(str, args.energy_probe_depths),
                        '--energy-lengths', *map(str, args.energy_lengths)]
        if args.strength_surface:
            shutil.copyfile(args.strength_surface, output/'strength_surface.json')
            command += ['--strength-surface', str(output/'strength_surface.json')]
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[key] = str(args.threads)
        with (output/'stdout.log').open('wb') as stdout, (output/'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, cwd=frozen, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                       start_new_session=os.name != 'nt')
        write_json(output/'launch.json', dict(pid=process.pid, command=command, source_root=str(frozen)))
        print(process.pid)
    else:
        strength = StrengthSurface.from_json(args.strength_surface) if args.strength_surface else None
        sys.exit(0 if scan(args.source, args.output, args.radii, strength, args.threads,
                          allow_failed_prefix=args.allow_failed_prefix,
                          energy_probe_depths=args.energy_probe_depths, energy_lengths=args.energy_lengths,
                          directional_onset=args.directional_onset, ell_mm=args.ell,
                          orientation_samples=args.orientation_samples, stop_at_onset=args.stop_at_onset) else 1)


if __name__ == '__main__':
    main()
