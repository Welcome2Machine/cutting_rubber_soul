"""Versioned MMT/OBT calculations over the same immutable material tracks.

MMT-v1 is the historical Cartesian + curved-map estimator. OBT-v1 is the
verified numerical repair: objective rotation, registered blade reference,
same-profile qualification, soft fixed-origin coupling and derivative recovery.
Both infer an original-surface address conditionally, not fracture onset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import numpy as np

from .boundary_contact_transport import BoundaryContactTransportConfig, BoundaryContactTransportObserver
from .boundary_material_contact import BoundaryMaterialContactConfig, project_to_path
from .contact_transport_observations import ContactTransportObserver
from .contact_transport_algorithms import (
    acceleration_scale, combine_models, fixed_parameter_fusion,
    legacy_qualification, pack_records, solve,
)
from .surface_identity_coupling import couple_joint_transport_profiles
from .contact_transport_filter import TransportPolicy, fuse_bounded, fuse_windowed


@dataclass(frozen=True)
class ContactMethod:
    key: str
    version: str
    chinese_name: str
    english_name: str
    color: str
    former_label: str

    @property
    def identifier(self):
        return f'{self.key.upper()}-{self.version}'


METHODS = (
    ContactMethod('mmt', 'v1', '混合模型输运法', 'Mixed-model transport', '#f691cb', 'Baseline'),
    ContactMethod('obt', 'v1', '客观边界输运法', 'Objective boundary transport', '#ffb358', 'Trial candidate'),
)


def method_manifest():
    return [dict(asdict(method), identifier=method.identifier) for method in METHODS]


def flattened_input(arrays):
    """Adapt the portable prepared-contact NPZ schema without copying targets."""
    result = dict(arrays)
    result['query'] = {k.removeprefix('query__'): v for k, v in arrays.items() if k.startswith('query__')}
    result['initial'] = {k.removeprefix('initial__'): v for k, v in arrays.items() if k.startswith('initial__')}
    result['paths'] = []
    for i in range(len(arrays['frames'])):
        lo, hi = arrays['path__offsets'][i:i+2]
        result['paths'].append(None if hi-lo < 2 else dict(
            xy=arrays['path__xy'][lo:hi], arc=arrays['path__arc'][lo:hi],
            tip_gap=float(arrays['path__gap'][i]), tip_projection=arrays['path__projection'][i]))
    return result


def path_at(geometry, i):
    lo, hi = geometry['path_offsets'][i:i+2]
    if hi-lo < 2:
        return None
    return dict(xy=geometry['path_xy'][lo:hi], arc=geometry['path_arc'][lo:hi],
                tip_projection=geometry['C_xy'][i], tip_gap=0.)


def validate_inputs(data, geometry, origin, motion):
    """Reject mixed runs/frame grids before any method can consume them."""
    frames = np.asarray(data['frames'])
    times = np.asarray(data['times'])
    if len(frames) < 2 or np.any(np.diff(frames) <= 0):
        raise ValueError('At least two increasing source frames are required')
    if len(times) != len(frames) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Physical times must match increasing source frames')
    if not np.isfinite(data['scale']) or data['scale'] <= 0:
        raise ValueError('Positive micrometres-per-pixel calibration is required')
    ids = data['query']['point_ids']
    if len(np.unique(ids)) != len(ids) or data['positions'].shape != (len(frames), len(ids), 2):
        raise ValueError('Positions must retain unique fixed material IDs')
    for label, other in (('geometry', geometry), ('origin', origin), ('motion', motion)):
        if not np.array_equal(other['frames'], frames):
            raise ValueError(f'{label} source frame grid differs from prepared input')
    if motion['source_kind'].shape != data['velocity_valid'].shape:
        raise ValueError('Recovered derivative material-ID grid differs')
    old = data['velocity_valid']
    if np.any((motion['source_kind'] == 2) & old):
        raise ValueError('Recovery cannot replace existing valid derivatives')
    np.testing.assert_allclose(motion['slope'][old], data['velocity'][old], rtol=1.e-9, atol=1.e-9)


def result_arrays(data, observations, solution, method):
    arrays = dict(observations, frames=data['frames'], frame_indices=data['frames'],
        depth_mm=data['depth'], times_s=data['times'], um_per_px=float(data['scale']),
        baseline_s_px=float(data['initial']['contact_s_px']),
        reference_s_px=solution['reference_s'], offset_um=solution['offset']*data['scale'],
        sigma_um=solution['offset_sigma']*data['scale'],
        method_key=np.array(method.key), method_version=np.array(method.version),
        method_identifier=np.array(method.identifier))
    arrays.update({'transport_'+k: v for k, v in solution.items() if isinstance(v, np.ndarray)})
    return arrays


class DualContactSolver:
    """Calculate each named recipe; never select/blend the two final curves."""

    legacy_qualification_compatibility = True

    def __init__(self, data, geometry, origin, motion):
        validate_inputs(data, geometry, origin, motion)
        self.data, self.geometry, self.origin, self.motion = data, geometry, origin, motion
        points, ids, initial, scale = (data['query']['seed_points'], data['query']['point_ids'],
                                       data['initial'], data['scale'])
        self.cartesian = ContactTransportObserver(points, ids, initial, scale)
        self.legacy = BoundaryContactTransportObserver(points, ids, initial, scale,
            config=BoundaryContactTransportConfig(velocity_model='legacy_projected'))
        self.objective = BoundaryContactTransportObserver(points, ids, initial, scale,
            config=BoundaryContactTransportConfig(velocity_model='objective_rotation'),
            boundary_config=BoundaryMaterialContactConfig(path_half_length_px=210.))
        self.legacy_raw = {}
        self.legacy_mmt = {}

    def _old_record(self, i, address):
        d = self.data
        kwargs = dict(velocity_sigma_px_s=d['velocity_sigma'][i],
            blade_velocity_sigma_px_s=d['blade_sigma'][i], branch_labels=d['branches'][i],
            contact_s_px=float(address), tip_uncertainty_px=d['uncertainty'][i])
        args = (d['positions'][i], d['support'][i], d['tips'][i], d['blade_velocity'][i],
                d['velocity'][i], d['velocity_valid'][i])
        cartesian = self.cartesian.estimate_frame(*args, **kwargs)
        path = d['paths'][i]
        if path is not None and i not in self.legacy_raw:
            raw = self.legacy.estimate_frame(*args, **kwargs, path=path, return_models=True)
            self.legacy_raw[i] = raw
            if self.legacy_qualification_compatibility:
                self.legacy_mmt[i] = legacy_qualification(raw)
        boundary = self.legacy_mmt if self.legacy_qualification_compatibility else self.legacy_raw
        return combine_models(cartesian, boundary.get(i, cartesian))

    def mixed_model(self, progress=None):
        d = self.data
        address = np.full(len(d['frames']), float(d['initial']['contact_s_px']))
        iterations = []
        for iteration in range(3):
            records = []
            for i in range(len(address)):
                records.append(self._old_record(i, address[i]))
                if progress and i % 80 == 0:
                    progress('mmt', iteration+1, i+1, len(address))
            observations = pack_records(records)
            solution, parameters = solve(d, observations)
            updated = solution['reference_s']
            iterations.append(dict(iteration=iteration+1,
                max_address_update_px=float(np.max(abs(updated-address))),
                median_address_update_px=float(np.median(abs(updated-address))), **parameters))
            address = updated
        arrays = result_arrays(d, observations, solution, METHODS[0])
        arrays['cartesian_compression_valid'] = np.array([
            [side.get('cartesian_compression_valid', False) for side in record['sides']] for record in records])
        return arrays, dict(iterations=iterations, parameters=parameters,
            cartesian_config=asdict(self.cartesian.config), boundary_config=asdict(self.legacy.config),
            legacy_absolute_qualification_compatibility=True,
            method_role='Frozen historical comparator; retains known legacy qualification behavior')

    def objective_boundary(self, progress=None):
        d, g, p, motion = self.data, self.geometry, self.origin, self.motion
        recovered = motion['source_kind'] == 2
        velocity = np.where(recovered[..., None], motion['slope'], d['velocity'])
        valid = d['velocity_valid'] | recovered
        sigma = np.where(recovered, motion['velocity_sigma'], d['velocity_sigma'])
        records, qualifications = [], []
        for i in range(len(d['frames'])):
            path = path_at(g, i)
            vv = valid[i].copy()
            vref = d['blade_velocity'][i]
            if not np.isfinite(vref).all():
                vv[:] = False
                vref = np.zeros(2)
            raw = self.objective.estimate_frame(d['positions'][i], d['support'][i], d['tips'][i],
                vref, velocity[i], vv, velocity_sigma_px_s=sigma[i],
                blade_velocity_sigma_px_s=float(d['blade_sigma'][i]) if np.isfinite(d['blade_sigma'][i]) else 1.e6,
                branch_labels=d['branches'][i], tip_uncertainty_px=float(d['uncertainty'][i]),
                path=path, return_models=True)
            lo, hi = g['path_offsets'][i:i+2]
            gate = bool(path is not None and g['C_state'][i] > 0 and g['path_edge_valid'][lo:hi-1].all())
            if path is None:
                raw['absolute_valid'] = False
                q = dict(profiles=[], selected=[], absolute_valid=False)
            else:
                arc, _, gap, alternative, _ = project_to_path(path['xy'], path['arc'], p['p0_xy'][i:i+1])
                eligible = bool(gap[0] <= 8 and alternative[0] > gap[0]+2 and p['p0_on_boundary'][i])
                current_arc = project_to_path(path['xy'], path['arc'],
                    np.nan_to_num(d['positions'][i], nan=-1.e6))[0]
                q = couple_joint_transport_profiles(raw, self.objective.boundary, current_arc,
                    float(d['initial']['contact_s_px']), float(arc[0]), float(p['p0_sigma_px'][i]),
                    int(p['p0_state'][i]) if eligible else 4,
                    float(p['p0_geometry_correction_px'][i]+gap[0]), geometry_qualified=gate)
                raw.update({k: q[k] for k in ('absolute_s_px', 'absolute_sigma_px', 'absolute_valid')})
            if not gate:
                raw['absolute_valid'] = False
                for side in raw['sides']:
                    side['flux_valid'] = False
                    side['absolute_valid'] = False
            records.append(raw)
            qualifications.append(dict(gate=gate, recovered_point_count=int(recovered[i].sum()),
                profiles=q['profiles'], selected=q['selected'], absolute_valid=bool(raw['absolute_valid'])))
            if progress and i % 80 == 0:
                progress('obt', 1, i+1, len(d['frames']))
        # Freeze the 20261008 recipe: derive common process parameters from
        # qualified old-boundary observations, never from desired endpoints.
        if len(self.legacy_raw) != len(d['frames']):
            raise ValueError('OBT process-scale control requires a valid old boundary at each frame')
        control = pack_records([self.legacy_raw[i] for i in range(len(d['frames']))])
        acceleration = acceleration_scale(d['times'], control['side_flux_px_s'],
            control['side_flux_valid'], control['side_flux_sigma_px_s'])
        first = fixed_parameter_fusion(d, control, acceleration)
        initial_sigma = float(first['initial_velocity_sigma'])
        observations = pack_records(records)
        solution = fixed_parameter_fusion(d, observations, acceleration, initial_sigma)
        arrays = result_arrays(d, observations, solution, METHODS[1])
        arrays.update(gate_mask=np.array([q['gate'] for q in qualifications]),
            acceleration_sigma=acceleration, initial_velocity_sigma=initial_sigma)
        parameters = dict(acceleration_sigma=acceleration, initial_velocity_sigma=initial_sigma,
            process_scale_source='qualified legacy curved control, same fixed-depth prefix',
            registered_blade_reference=True, derivative_recovery=True,
            boundary_config=asdict(self.objective.config), position_config=asdict(self.objective.boundary.config))
        return arrays, dict(parameters=parameters, qualification=qualifications,
            legacy_absolute_qualification_compatibility=False,
            method_role='Numerically repaired estimator; hidden surface identity remains conditional')

    def calculate(self, progress=None):
        mmt, mmt_meta = self.mixed_model(progress)
        obt, obt_meta = self.objective_boundary(progress)
        for key in ('frames', 'times_s', 'depth_mm', 'baseline_s_px', 'um_per_px'):
            np.testing.assert_array_equal(mmt[key], obt[key])
        return {'mmt': mmt, 'obt': obt}, {'mmt': mmt_meta, 'obt': obt_meta}


REPAIRED_METHODS = tuple(replace(method, version="v2") for method in METHODS)
RECOVERY_METHODS = tuple(replace(method, version="recovery-candidate") for method in REPAIRED_METHODS)

class RepairedDualContactSolver(DualContactSolver):
    legacy_qualification_compatibility = False
    fusion = staticmethod(fuse_bounded)
    def __init__(self, *args, policy=TransportPolicy(), **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def _qualified_fusion(self, obs, acceleration):
        bad = ~self.data['blade_observation_qualified']
        obs['side_flux_valid'] = obs['side_flux_valid'].copy()
        obs['absolute_valid'] = obs['absolute_valid'].copy()
        obs['side_flux_valid'][bad] = False
        obs['absolute_valid'][bad] = False
        if acceleration is None:
            acceleration = acceleration_scale(self.data['times'], obs['side_flux_px_s'],
                obs['side_flux_valid'], obs['side_flux_sigma_px_s'])
        solution = self.fusion(self.data, obs, acceleration, self.policy)
        return bad, acceleration, solution

    def _fuse(self, obs, method, acceleration=None):
        bad, acceleration, solution = self._qualified_fusion(obs, acceleration)
        arrays = result_arrays(self.data, obs, solution, method)
        return arrays, solution, dict(policy=asdict(self.policy), acceleration_sigma=float(acceleration),
            zero_velocity_deadband_px_s=self.policy.zero_velocity_px_s,
            original_geometry_rejected_frames=int(bad.sum()),
            legacy_absolute_qualification_compatibility=False,
            future_velocity_used_for_temporal_hold=False,
            unknown_position_restoration='requires an absolute address; velocity alone cannot restore lost displacement')

    def mixed_model(self, progress=None):
        d = self.data
        address = np.full(len(d['frames']), float(d['initial']['contact_s_px']))
        history = []
        for iteration in range(3):
            records = []
            for i in range(len(address)):
                records.append(self._old_record(i, address[i]))
                if progress and i % 80 == 0:
                    progress('mmt-v2', iteration+1, i+1, len(address))
            obs = pack_records(records)
            arrays, solution, parameters = self._fuse(obs, REPAIRED_METHODS[0])
            new = solution['search_reference_s']
            history.append(dict(iteration=iteration+1, max_guide_update_px=float(np.max(abs(new-address)))))
            address = new
        return arrays, dict(iterations=history, parameters=parameters,
            method_role='Cartesian/curved observations; corrected qualification and bounded causal fusion')

    def objective_boundary(self, progress=None):
        # Retain the objective-rotation spatial observer and common process
        # scale. Discard its historical temporal solution in its entirety.
        frozen, metadata = super().objective_boundary(progress)
        obs = {k: v for k, v in frozen.items() if k.startswith('side_') or k.startswith('absolute_')}
        control = pack_records([self.legacy_raw[i] for i in range(len(self.data['frames']))])
        control_valid = control['side_flux_valid'].copy()
        control_valid[~self.data['blade_observation_qualified']] = False
        acceleration = acceleration_scale(self.data['times'], control['side_flux_px_s'],
            control_valid, control['side_flux_sigma_px_s'])
        arrays, _, parameters = self._fuse(obs, REPAIRED_METHODS[1], acceleration)
        arrays['gate_mask'] = frozen['gate_mask']
        return arrays, dict(parameters=parameters,
            spatial_parameters=metadata['parameters'], qualification=metadata['qualification'],
            method_role='Objective boundary observations; bounded causal fusion')

def recover_derivatives(data, motion):
    """Fill absent derivatives only with already validated same-ID fits."""
    if not np.array_equal(data['frames'], motion['frames']):
        raise ValueError('Recovered derivatives must have the same source frames')
    original = np.asarray(data['velocity_valid'], bool)
    recovered = ((motion['source_kind'] == 2) & motion['valid'] & ~original
        & data['support'] & np.isfinite(motion['slope']).all(axis=2)
        & np.isfinite(motion['velocity_sigma']) & (motion['velocity_sigma'] > 0))
    result = dict(data)
    result['velocity'] = np.where(recovered[..., None], motion['slope'], data['velocity'])
    result['velocity_valid'] = original | recovered
    result['velocity_sigma'] = np.where(recovered, motion['velocity_sigma'], data['velocity_sigma'])
    result['material_velocity_recovered'] = recovered
    return result

class ObservationRecoveryContactSolver(RepairedDualContactSolver):
    fusion = staticmethod(fuse_windowed)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = recover_derivatives(self.data, self.motion)
        self.policy = TransportPolicy()

    def _fuse(self, obs, method, acceleration=None):
        obs = dict(obs)
        bad, acceleration, solution = self._qualified_fusion(obs, acceleration)
        candidate = next(item for item in RECOVERY_METHODS if item.key == method.key)
        arrays = result_arrays(self.data, obs, solution, candidate)
        arrays['material_velocity_recovered_count'] = self.data['material_velocity_recovered'].sum(axis=1)
        parameters = dict(policy=asdict(self.policy), acceleration_sigma=float(acceleration),
            spatial_qualification_relaxed=False, legacy_absolute_qualification_compatibility=False,
            original_geometry_rejected_frames=int(bad.sum()),
            derivative_recovery='Existing same-ID quadratic, cross-scale and held-out-position checks; MMT and OBT share it',
            total_recovered_point_derivatives=int(self.data['material_velocity_recovered'].sum()),
            weak_signed_motion_retained=True, missing_motion_is_zero=False,
            unknown_position_restoration='Still requires qualified absolute address; local segments retain their own origin',
            no_post_cutoff_samples=True, temporal_fusion_causal_given_observations=True,
            complete_pipeline_causal=False, physical_accuracy_validated=False)
        return arrays, solution, parameters

    def calculate(self, progress=None):
        results, reports = super().calculate(progress)
        for method in RECOVERY_METHODS:
            reports[method.key]['method_role'] = 'Candidate observation recovery; not adopted as default'
        return results, reports

@dataclass(frozen=True)
class ConvergencePolicy:
    maximum_iterations: int = 14
    damping: float = .5
    tolerance_px: float = .25
    consecutive_passes: int = 2

def convergence_metrics(guide, proposed, available, tolerance):
    guide, proposed = np.asarray(guide, float), np.asarray(proposed, float)
    active = np.asarray(available, bool)
    if guide.shape != proposed.shape or guide.shape != active.shape:
        raise ValueError('Matching guide and availability arrays required')
    if not np.isfinite(guide).all() or not np.isfinite(proposed).all():
        raise ValueError('Finite search guides required')
    delta = np.abs(proposed-guide)
    finite = delta[active]
    return dict(max_all_px=float(delta.max()),
        max_available_px=float(finite.max()) if len(finite) else None,
        p95_available_px=float(np.quantile(finite, .95)) if len(finite) else None,
        fraction_available_within_tolerance=float(np.mean(finite <= tolerance)) if len(finite) else 0.,
        converged=bool(len(finite) and delta.max() <= tolerance)), delta

class ConvergedContactSolver(ObservationRecoveryContactSolver):
    def mixed_model_converged(self, initial_guide, policy=ConvergencePolicy(), progress=None):
        if not 0 < policy.damping <= 1 or policy.maximum_iterations < 2 or policy.tolerance_px <= 0:
            raise ValueError('Invalid convergence policy')
        guide = np.asarray(initial_guide, float).copy()
        if guide.shape != self.data['frames'].shape:
            raise ValueError('Initial guide must match the prepared frame grid')
        history, consecutive = [], 0
        previous_gates = None
        for iteration in range(policy.maximum_iterations):
            evaluation_guide = guide.copy()
            records = [self._old_record(i, float(s)) for i, s in enumerate(guide)]
            obs = pack_records(records)
            arrays, solution, parameters = self._fuse(obs, RECOVERY_METHODS[0])
            proposed = solution['search_reference_s']
            metrics, residual = convergence_metrics(guide, proposed,
                arrays['transport_offset_available'], policy.tolerance_px)
            gates = np.column_stack((arrays['side_flux_valid'], arrays['absolute_valid']))
            changes = int(np.any(gates != previous_gates, axis=1).sum()) if previous_gates is not None else None
            passed = metrics['converged'] and changes == 0
            consecutive = consecutive+1 if passed else 0
            history.append(dict(iteration=iteration+1, **metrics, changed_gate_frames=changes))
            if progress:
                progress(history[-1])
            if consecutive >= policy.consecutive_passes:
                break
            previous_gates = gates
            guide += policy.damping*(proposed-guide)
        arrays['spatial_fixed_point_residual_px'] = residual
        arrays['spatial_fixed_point_resolved'] = residual <= policy.tolerance_px
        arrays['spatial_evaluation_guide_s_px'] = evaluation_guide
        # Keep the computed offset and its original uncertainty intact. A local
        # unresolved fixed point is separately masked only by the review layer.
        return arrays, dict(policy=asdict(policy), history=history,
            converged=consecutive >= policy.consecutive_passes,
            parameters=parameters, supplied_microscopy_values_used=False,
            previous_tracks_modified=False, no_post_cutoff_data=True,
            physical_accuracy_validated=False)
