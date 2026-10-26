"""Physics/provenance contracts for the two retained contact recipes."""
from copy import deepcopy

import numpy as np
import pytest

from src.analysis.dual_contact import (DualContactSolver, validate_inputs, recover_derivatives,
    convergence_metrics, RepairedDualContactSolver, flattened_input)
from src.analysis.validated_material_motion import validated_local_slopes
from pathlib import Path
from src.io import load_npz as load, slice_ragged
from src.workflows.contact_inputs import cutoffs_for


def rigid_scene():
    times = np.linspace(0., .4, 9)
    s, depth = np.meshgrid(np.arange(-100., 101., 5.), np.arange(0., 34., 3.))
    seed = np.column_stack((s.ravel(), depth.ravel()))
    surface_s = np.arange(-150., 151., 2.)
    surface = np.column_stack((surface_s, np.zeros(len(surface_s))))
    initial = dict(boundary_xy=surface, surface_s_px=surface_s, tangent=np.array([1., 0.]),
                   inward_normal=np.array([0., 1.]), contact_s_px=0.)
    displacement = np.column_stack((np.zeros(len(times)), 10*times))
    positions = seed[None]+displacement[:, None]
    velocity = np.broadcast_to([0., 10.], positions.shape).copy()
    support = np.ones(positions.shape[:2], bool)
    paths = [dict(xy=surface+offset, arc=surface_s, tip_gap=0., tip_projection=offset)
             for offset in displacement]
    data = dict(frames=np.arange(len(times)), times=times, depth=times/10., scale=1.,
        positions=positions, velocity=velocity, support=support, velocity_valid=support.copy(),
        velocity_sigma=np.full(support.shape, .25), tips=displacement,
        blade_velocity=np.broadcast_to([0., 10.], displacement.shape).copy(),
        blade_sigma=np.full(len(times), .25), uncertainty=np.full(len(times), 2.),
        branches=np.where(seed[:, 0] <= 0, 1, 2)[None].repeat(len(times), 0), paths=paths,
        initial=initial, query=dict(seed_points=seed, point_ids=np.arange(len(seed))))
    geometry = dict(frames=data['frames'], C_xy=displacement, C_state=np.full(len(times), 2),
        path_offsets=np.arange(len(times)+1)*len(surface), path_xy=np.concatenate([p['xy'] for p in paths]),
        path_arc=np.tile(surface_s, len(times)), path_edge_valid=np.tile(np.r_[np.ones(len(surface)-1, bool), False], len(times)))
    origin = dict(frames=data['frames'], p0_xy=displacement, p0_sigma_px=np.full(len(times), 2.),
        p0_on_boundary=np.ones(len(times), bool), p0_state=np.ones(len(times), int),
        p0_geometry_correction_px=np.zeros(len(times)))
    motion = dict(frames=data['frames'], source_kind=support.astype(np.int8), slope=velocity.copy(),
                  velocity_sigma=np.full(support.shape, .25))
    return data, geometry, origin, motion


def test_rigid_common_motion_gives_zero_slip_in_both_recipes_without_mutating_tracks():
    data, geometry, origin, motion = rigid_scene()
    positions = data['positions'].copy()
    support = data['support'].copy()
    results, reports = DualContactSolver(data, geometry, origin, motion).calculate()
    for key, identifier in (('mmt', 'MMT-v1'), ('obt', 'OBT-v1')):
        result = results[key]
        assert str(result['method_identifier']) == identifier
        np.testing.assert_allclose(result['offset_um'], 0., atol=1.e-7)
        assert result['transport_velocity_used'].any()
        assert result['offset_um'][0] == 0.
    assert reports['mmt']['legacy_absolute_qualification_compatibility']
    assert not reports['obt']['legacy_absolute_qualification_compatibility']
    np.testing.assert_array_equal(data['positions'], positions)
    np.testing.assert_array_equal(data['support'], support)


@pytest.mark.parametrize('index', [1, 2, 3])
def test_mixed_frame_grids_are_rejected_before_any_fitting(index):
    inputs = list(rigid_scene())
    inputs[index] = deepcopy(inputs[index])
    inputs[index]['frames'] = inputs[index]['frames']+1
    with pytest.raises(ValueError, match='frame grid'):
        validate_inputs(*inputs)


def test_recovery_cannot_replace_original_valid_derivative():
    data, geometry, origin, motion = rigid_scene()
    motion['source_kind'][0, 0] = 2
    with pytest.raises(ValueError, match='replace existing'):
        validate_inputs(data, geometry, origin, motion)


def test_unknown_case_needs_explicit_depth_instead_of_implicitly_assuming_fracture():
    with pytest.raises(ValueError, match='declared cutoff'):
        cutoffs_for(['0930l60a3'])
    assert cutoffs_for(['0930l60a3'], overrides=['0930l60a3=1.2']) == {'0930l60a3': 1.2}
    with pytest.raises(ValueError, match='unique'):
        cutoffs_for(['0930l60a1'], overrides=['0930l60a1=1.1', '0930l60a1=1.3'])


def test_endpoint_velocity_restored_from_position_history_without_future_samples():
    t=np.arange(50)*.0125
    p=np.stack((20+6*t+2*t*t, 30+4*t),axis=-1)[:,None,:]
    good=np.ones((len(t),1),bool)
    motion=validated_local_slopes(p,t,good)
    assert motion['source_kind'][-1,0]==2
    valid=motion['source_kind']==1
    data=dict(frames=np.arange(len(t)),support=good,velocity_valid=valid,
        velocity=np.where(valid[...,None],motion['slope'],np.nan),
        velocity_sigma=np.where(valid,motion['velocity_sigma'],np.nan))
    motion['frames']=data['frames']
    recovered=recover_derivatives(data,motion)
    assert recovered['velocity_valid'][-1,0]
    np.testing.assert_allclose(recovered['velocity'][-1,0],[6+4*t[-1],4.],atol=1e-8)
    np.testing.assert_array_equal(recovered['velocity'][valid],data['velocity'][valid])
    assert not data['velocity_valid'][-1,0]

def test_no_recovery_across_broken_material_identity_run():
    t=np.arange(50)*.0125
    p=np.stack((20+6*t,30+4*t),axis=-1)[:,None,:]
    good=np.ones((len(t),1),bool); edges=good.copy();edges[-3,0]=False
    motion=validated_local_slopes(p,t,good,motion_step_valid=edges)
    assert not motion['valid'][-1,0]
    valid=motion['source_kind']==1
    data=dict(frames=np.arange(len(t)),support=good,velocity_valid=valid,
        velocity=motion['slope'].copy(),velocity_sigma=motion['velocity_sigma'].copy())
    motion['frames']=data['frames']
    result=recover_derivatives(data,motion)
    assert not result['velocity_valid'][-1,0]

def test_recovery_cannot_override_a_valid_derivative_or_an_unsupported_point():
    data=dict(frames=np.arange(2),support=np.array([[True,False],[True,True]]),
        velocity_valid=np.array([[True,False],[False,False]]),
        velocity=np.zeros((2,2,2)),velocity_sigma=np.ones((2,2)))
    motion=dict(frames=np.arange(2),source_kind=np.full((2,2),2),valid=np.ones((2,2),bool),
        slope=np.full((2,2,2),5.),velocity_sigma=np.ones((2,2)))
    result=recover_derivatives(data,motion)
    assert not result['material_velocity_recovered'][0].any()
    assert result['material_velocity_recovered'][1].all()
    np.testing.assert_array_equal(result['velocity'][0,0],[0,0])
    with pytest.raises(ValueError,match='same source frames'):
        recover_derivatives(data,dict(motion,frames=np.array([1,2])))

def test_undamped_residual_cannot_be_hidden_by_small_step():
    metrics, residual = convergence_metrics(np.array([0., 1.]), np.array([0., 21.]),
        np.ones(2, bool), .25)
    assert not metrics['converged']
    assert residual[1] == 20.

def test_unavailable_guide_is_still_checked_for_future_roi_stability():
    metrics, _ = convergence_metrics(np.array([0., 0.]), np.array([0., 10.]),
        np.array([True, False]), .25)
    assert metrics['max_available_px'] == 0.
    assert not metrics['converged']

def test_true_fixed_point_and_nonfinite_rejection():
    metrics, _ = convergence_metrics(np.array([0., -4.]), np.array([0., -4.01]),
        np.ones(2, bool), .25)
    assert metrics['converged']
    with pytest.raises(ValueError):
        convergence_metrics(np.array([0.]), np.array([np.nan]), np.ones(1, bool), .25)

def test_under_relaxation_resolves_roi_oscillation_without_accepting_three_pass_result(monkeypatch):
    import src.analysis.dual_contact as module
    solver = object.__new__(module.ConvergedContactSolver)
    solver.data = {'frames':np.arange(2)}
    solver._old_record = lambda i,s:s
    monkeypatch.setattr(module,'pack_records',lambda records:np.array(records))
    def fusion(guide,method):
        proposed = np.array([0.,2.-guide[1]])
        arrays = dict(transport_offset_available=np.ones(2,bool),side_flux_valid=np.ones((2,2),bool),
            absolute_valid=np.ones(2,bool),offset_um=proposed.copy())
        return arrays,{'search_reference_s':proposed},{}
    solver._fuse = fusion
    unstable, failed = solver.mixed_model_converged(np.zeros(2),
        module.ConvergencePolicy(maximum_iterations=3,damping=1.))
    assert not failed['converged']
    assert unstable['spatial_fixed_point_residual_px'][1] == 2.
    stable, passed = solver.mixed_model_converged(np.zeros(2))
    assert passed['converged']
    np.testing.assert_allclose(stable['offset_um'],[0.,1.])
    np.testing.assert_allclose(stable['spatial_evaluation_guide_s_px'],[0.,1.])

def test_both_methods_on_three_saved_real_samples():
    folder = Path(__file__).resolve().parents[1]/'results/20261015_declared_fracture_sliding/1005L60C2up'
    if not (folder/'input.npz').exists():
        pytest.skip('Local video-derived preparation is not distributed with code')
    d, g, p, m = (load(folder/file) for file in ('input.npz', 'geometry.npz', 'origin.npz', 'temporal_motion.npz'))
    n = len(d['frames'])
    keep = np.arange(n) < 3
    paths = slice_ragged(d, 'path__', keep, ('path__xy', 'path__arc'))
    for key in tuple(d):
        if key.startswith(('query__', 'initial__', 'path__')):
            continue
        if np.asarray(d[key]).shape[:1] == (n,):
            d[key] = d[key][keep]
    for key in ('path__gap', 'path__projection'):
        d[key] = d[key][keep]
    d.update(paths)
    ragged = {}
    for prefix in ('path_', 'closed_'):
        fields = [key for key in g if key.startswith(prefix) and key != prefix+'offsets']
        ragged.update(slice_ragged(g, prefix, keep, fields))
    for a in (g, p, m):
        for key in tuple(a):
            if a is g and key in ragged:
                continue
            if np.asarray(a[key]).shape[:1] == (n,):
                a[key] = a[key][keep]
    g.update(ragged)
    result, meta = RepairedDualContactSolver(flattened_input(d), g, p, m).calculate()
    for key in ('mmt', 'obt'):
        assert result[key]['offset_um'][0] == 0.
        assert len(result[key]['transport_state']) == 3
        assert result[key]['method_version'].item() == 'v2'
        assert meta[key]['parameters']['legacy_absolute_qualification_compatibility'] is False
