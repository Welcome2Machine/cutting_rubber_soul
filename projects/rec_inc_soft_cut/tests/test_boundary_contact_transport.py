"""Analytic curved-surface transport, with fixed reference material IDs."""
from dataclasses import replace

import numpy as np
import pytest

from src.analysis.boundary_contact_transport import (
    BoundaryContactTransportObserver, BoundaryContactTransportConfig,
)
from src.analysis.boundary_material_contact import BoundaryMaterialContactConfig


def scene(curved=True, true_s=-12.5, stretch=1.15, normal_scale=1.):
    s, d = np.meshgrid(np.arange(-110., 111., 5.), np.arange(0., 31., 3.))
    seed = np.column_stack((s.ravel(), d.ravel()))
    stations = np.arange(-150., 151., 2.)
    reference = dict(boundary_xy=np.column_stack((stations, np.zeros(len(stations)))),
        surface_s_px=stations, tangent=np.array([1., 0.]),
        inward_normal=np.array([0., 1.]), contact_s_px=0.)
    observer = BoundaryContactTransportObserver(seed, np.arange(len(seed)), reference, 1.,
        boundary_config=replace(BoundaryMaterialContactConfig(), maximum_depths_px=(36.,)))
    s, d = seed.T
    r = stretch*(s-true_s)+.4*d+.003*d*d
    arc = np.arange(-300., 300.01, .5)
    radius = 280.
    if curved:
        current = np.column_stack(((radius-normal_scale*d)*np.sin(r/radius),
                                   radius-(radius-normal_scale*d)*np.cos(r/radius)))
        xy = np.column_stack((radius*np.sin(arc/radius), radius*(1.-np.cos(arc/radius))))
        tau = np.column_stack((np.cos(r/radius), np.sin(r/radius)))
    else:
        current = np.column_stack((r, normal_scale*d))
        xy = np.column_stack((arc, np.zeros(len(arc))))
        tau = np.broadcast_to([1., 0.], current.shape)
    return observer, seed, current, tau, dict(xy=xy, arc=arc, tip_gap=0.)


def observe(observer, current, velocity, blade=(0., 0.), tip=(0., 0.), path=None, **kwargs):
    valid = np.ones(len(current), bool)
    return observer.estimate_frame(current, valid, tip, blade, velocity, valid,
        path=path, return_models=True, **kwargs)


def test_curved_common_rigid_translation_has_exactly_zero_flux():
    observer, seed, current, tau, path = scene()
    translation = np.array([3., 7.])
    result = observe(observer, current, np.broadcast_to(translation, current.shape), translation, path=path)
    assert result['combined_flux_valid']
    assert abs(result['combined_flux_px_s']) < 1.e-12
    assert result['absolute_valid']
    # Both linear-depth and quadratic-depth models are retained; the former
    # has a small address bias. The alternative-model sensitivity must cover it.
    assert abs(result['absolute_s_px']+12.5) < .2
    assert abs(result['absolute_s_px']+12.5) < result['absolute_sigma_px']
    assert not result['original_surface_identity_proven']
    assert result['original_surface_continuity_assumed']
    assert all(np.isnan(p['normal_compression']) for p in result['sides'])


@pytest.mark.parametrize('flux', [-4., 4.])
def test_known_curved_surface_transport_and_depth_shear(flux):
    observer, seed, current, tau, path = scene()
    # Relative tangential speed varies with initial depth, but its d=0 value
    # is exactly -lambda*q. Neither current nearest point nor depth donor is s_c.
    s, d = seed.T
    vt = -1.15*flux+.08*d+.0007*s*d
    velocity = vt[:, None]*tau
    result = observe(observer, current, velocity, path=path)
    assert result['combined_flux_valid']
    assert abs(result['combined_flux_px_s']-flux) < .02
    assert result['absolute_valid']
    assert abs(result['absolute_s_px']+12.5) < .2
    assert abs(result['absolute_s_px']+12.5) < result['absolute_sigma_px']
    assert not result['compression_imposes_zero_slip']


def test_high_compression_does_not_impose_zero_transport():
    observer, seed, current, tau, path = scene(normal_scale=.08)
    result = observe(observer, current, -4.6*tau, path=path)
    assert result['combined_flux_valid']
    assert abs(result['combined_flux_px_s']-4.) < .02
    assert not result['compression_imposes_zero_slip']


def test_static_material_and_normal_penetration_is_not_contact_transport():
    observer, seed, current, tau, path = scene(curved=False)
    result = observe(observer, current, np.zeros_like(current), blade=(0., 8.), path=path)
    assert np.isfinite(result['combined_flux_px_s'])
    assert not result['combined_flux_valid']
    assert not result['absolute_valid']
    assert all('normal_velocity_incompatible' in p['reasons'] for p in result['sides'])
    # A fake caller-supplied zero gap cannot conceal the actual contour gap.
    result = observe(observer, current, np.zeros_like(current), tip=(0., 20.), path=path)
    assert result['boundary_tip_gap_px'] >= 20.
    assert not result['combined_flux_valid']
    assert not result['absolute_valid']


def test_velocity_missing_ids_are_excluded_without_changing_material_identity():
    observer, seed, current, tau, path = scene()
    velocity = -4.6*tau
    valid = np.ones(len(current), bool)
    valid[::3] = False
    velocity[~valid] = 1.e9
    result = observer.estimate_frame(current, np.ones(len(current), bool), [0., 0.], [0., 0.],
        velocity, valid, path=path, return_models=True)
    assert result['combined_flux_valid']
    assert abs(result['combined_flux_px_s']-4.) < .02
    for models in result['models']:
        for profile in models:
            assert valid[profile['support_ids']].all()


def test_actual_path_end_clamps_never_enter_velocity_models():
    observer, seed, current, tau, path = scene(curved=False, true_s=0.)
    keep = abs(path['arc']) <= 90.
    path = dict(xy=path['xy'][keep], arc=path['arc'][keep], tip_gap=0.)
    result = observe(observer, current, -4.6*tau, path=path)
    assert result['combined_flux_valid']
    ids = np.unique(np.concatenate([p['support_ids'] for ps in result['models'] for p in ps]))
    assert (abs(current[ids, 0]) <= 90.).all()


def test_multivalued_current_projection_does_not_supply_velocity_observations():
    observer, seed, current, tau, path = scene(curved=False)
    r = np.arange(-150., 151., 1.)
    xy = np.vstack((np.column_stack((r, np.zeros(len(r)))), [[150., 5.]],
                    np.column_stack((r[::-1], np.full(len(r), 10.)))))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]-150.
    current = np.column_stack((seed[:, 0], np.full(len(seed), 5.)))
    result = observe(observer, current, np.zeros_like(current), path=dict(xy=xy, arc=arc, tip_gap=0.))
    assert not result['combined_flux_valid']
    assert not np.isfinite(result['combined_flux_px_s'])


def test_tip_clamped_at_boundary_endpoint_cannot_be_called_contact():
    observer, seed, current, tau, path = scene(curved=False)
    keep = path['arc'] >= 0.
    path = dict(xy=path['xy'][keep], arc=path['arc'][keep], tip_gap=0.)
    result = observe(observer, current, -4.6*tau, tip=(-2., 0.), path=path)
    assert not result['combined_flux_valid']
    assert not result['absolute_valid']


def test_model_ensemble_does_not_reduce_correlated_uncertainty_by_model_count():
    profile = dict(side=1, flux_px_s=4., flux_sigma_px_s=2., flux_valid=True,
        absolute_s_px=3., absolute_sigma_px=5., absolute_valid=True,
        position_rms_px=1., position_holdout_px=1.)
    result = BoundaryContactTransportObserver._combine(1, [profile]*10)
    assert result['flux_sigma_px_s'] == 2.
    assert result['absolute_sigma_px'] == 5.


def test_requires_actual_current_boundary_geometry():
    observer, seed, current, tau, path = scene()
    with pytest.raises(ValueError, match='boundary path'):
        observe(observer, current, np.zeros_like(current))


def test_missing_velocity_cannot_leave_position_only_absolute_qualified():
    observer, seed, current, tau, path = scene()
    result = observer.estimate_frame(current, np.ones(len(current), bool), [0., 0.], [0., 0.],
        np.zeros_like(current), np.zeros(len(current), bool), path=path, return_models=True)
    models = [p for side in result['models'] for p in side]
    assert models and any(p['position_absolute_compatible'] for p in models)
    assert not result['absolute_valid']
    assert all(not p['absolute_valid'] and not p['joint_absolute_valid'] for p in models)
    assert all(not p['velocity_fit_valid'] for p in models)
    assert all('insufficient_continuous_fixed_id_velocity_support' in p['reasons'] for p in models)
    # Full position support survives diagnostics even when velocity support is empty.
    assert all(len(p['position_profile']['support_ids']) > 0 and len(p['support_ids']) == 0 for p in models)


def test_rank_deficient_velocity_cannot_leave_position_only_absolute_qualified():
    observer, seed, current, tau, path = scene()
    velocity_valid = seed[:, 1] == 0.  # Enough IDs but no depth variation for the velocity basis.
    result = observer.estimate_frame(current, np.ones(len(current), bool), [0., 0.], [0., 0.],
        np.zeros_like(current), velocity_valid, path=path, return_models=True)
    models = [p for side in result['models'] for p in side]
    assert any(p['position_absolute_compatible'] for p in models)
    assert any('ill_conditioned_velocity_model' in p['reasons'] for p in models)
    assert not result['absolute_valid']
    assert all(not p['absolute_valid'] and not p['joint_absolute_valid'] for p in models)


def test_repeated_model_names_at_different_depths_have_unique_position_profiles():
    observer, seed, current, tau, path = scene()
    observer.boundary.config = replace(observer.boundary.config, maximum_depths_px=(18., 36.))
    result = observe(observer, current, -4.6*tau, path=path)
    models = [p for side in result['models'] for p in side]
    assert len({p['position_profile_index'] for p in models}) == len(models)
    assert len({p['position_profile_id'] for p in models}) == len(models)
    groups = {}
    for p in models:
        position = p['position_profile']
        groups.setdefault((position['side'], position['model']), set()).add(position['depth_limit_px'])
        assert position['reference_s_px'] == p['flux_eval_s_px']
        assert set(p['support_ids']) <= set(position['support_ids'])
        assert p['absolute_valid'] == (p['position_absolute_compatible'] and p['velocity_fit_valid'] and p['normal_compatible'])
    assert any(len(depths) > 1 for depths in groups.values())


def test_boundary_provenance_survives_side_packing_including_empty_side():
    from src.analysis.contact_transport_algorithms import pack_records
    observer, seed, current, tau, path = scene()
    result = observe(observer, current, -4.6*tau, path=path)
    assert all(side['observation_model'] == 'boundary' for side in result['sides'])
    assert BoundaryContactTransportObserver._combine(1, [])['observation_model'] == 'boundary'
    packed = pack_records([result])
    assert packed['side_observation_model'].tolist() == [['boundary', 'boundary']]


@pytest.mark.parametrize('omega', [-.3, .1, .3])
def test_curved_rigid_rotation_about_contact_is_not_slip(omega):
    observer, seed, current, tau, path = scene(true_s=0., stretch=1.)
    velocity = omega*np.column_stack((-current[:, 1], current[:, 0]))
    result = observe(observer, current, velocity, path=path)
    assert result['combined_flux_valid']
    assert abs(result['combined_flux_px_s']) < 1.e-10
    assert all(p['velocity_model'] == 'objective_rotation' for ps in result['models'] for p in ps)
    legacy = BoundaryContactTransportObserver(observer.boundary.seed, observer.ids,
        dict(boundary_xy=observer.boundary.surface, surface_s_px=observer.boundary.addresses,
             tangent=observer.boundary.tangent, inward_normal=observer.boundary.normal, contact_s_px=0.), 1.,
        config=replace(observer.config, velocity_model='legacy_projected'), boundary_config=observer.boundary.config)
    old = observe(legacy, current, velocity, path=path)
    assert abs(old['combined_flux_px_s']) > .1


@pytest.mark.parametrize('omega', [-.2, .1, .3])
def test_rotation_superposed_on_nonzero_surface_shear_keeps_profile_flux(omega):
    observer, seed, current, tau, path = scene()
    s, d = seed.T
    velocity = (-4.6+.08*d+.0007*s*d)[:, None]*tau
    base = observe(observer, current, velocity, path=path)
    shifted = observe(observer, current, velocity+omega*np.column_stack((-current[:, 1], current[:, 0])), path=path)
    for before, after in zip([p for ps in base['models'] for p in ps],
                             [p for ps in shifted['models'] for p in ps]):
        assert abs(before['flux_px_s']-after['flux_px_s']) < 1.e-9
        assert abs(before['velocity_rms_px_s']-after['velocity_rms_px_s']) < 1.e-9
        assert abs(before['velocity_holdout_px_s']-after['velocity_holdout_px_s']) < 1.e-9
        assert abs(before['normal_relative_velocity_px_s']-after['normal_relative_velocity_px_s']) < 1.e-9
    # Different geometric sensitivity to angular motion can reweight correlated
    # profile alternatives; individual contact evaluations are exact invariants.
    assert abs(base['combined_flux_px_s']-shifted['combined_flux_px_s']) < .001
    assert abs(shifted['combined_flux_px_s']-4.) < .02


def test_rigid_coordinate_rotation_and_translation_preserve_result():
    observer, seed, current, tau, path = scene()
    s, d = seed.T
    velocity = (-4.6+.08*d+.0007*s*d)[:, None]*tau
    base = observe(observer, current, velocity, path=path)
    angle=.7;Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    shift=np.array([200.,80.]);b=np.array([2.,3.])
    # The initial reference coordinates remain physical (s,d); their camera
    # coordinates, current geometry, vectors and tip all transform together.
    reference=dict(boundary_xy=observer.boundary.surface@Q.T+shift,
        surface_s_px=observer.boundary.addresses,tangent=observer.boundary.tangent@Q.T,
        inward_normal=observer.boundary.normal@Q.T,contact_s_px=0.)
    transformed=BoundaryContactTransportObserver(seed@Q.T+shift,observer.ids,reference,1.,
        config=observer.config,boundary_config=observer.boundary.config)
    result=observe(transformed,current@Q.T+shift,velocity@Q.T+b,blade=b,tip=shift,
        path=dict(xy=path['xy']@Q.T+shift,arc=path['arc'],tip_gap=0.))
    # Reference-cell balancing has exact floor boundaries; round-off in the
    # transformed (s,d) can change those discrete groups. Its retained legacy
    # perturbation is tested separately from the strict velocity-fit invariant.
    assert abs(result['combined_flux_px_s']-base['combined_flux_px_s']) < 1.e-3
    assert abs(result['combined_flux_sigma_px_s']-base['combined_flux_sigma_px_s']) < 1.e-2


def test_noisy_compressed_surface_transport_remains_resolved_without_gate_relaxation():
    observer, seed, current, tau, path=scene(normal_scale=.08)
    velocity=-4.6*tau
    results=[]
    for seed_number in range(12):
        noisy=velocity+np.random.default_rng(seed_number).normal(0.,.15,velocity.shape)
        result=observe(observer,current,noisy,path=path,velocity_sigma_px_s=np.full(len(current),.15))
        assert result['combined_flux_valid']
        assert abs(result['combined_flux_px_s']-4.) <= 2.*result['combined_flux_sigma_px_s']
        results.append(result['combined_flux_px_s'])
    assert abs(np.mean(results)-4.) < .08
    assert np.std(results) < .08


def test_current_affine_shear_and_compression_have_bounded_error():
    observer, seed, current, tau, path=scene(true_s=0.,stretch=1.)
    for gradient in (np.array([[0.,.1],[0.,0.]]),np.array([[0.,0.],[0.,-.1]])):
        result=observe(observer,current,current@gradient.T,path=path)
        assert result['combined_flux_valid']
        # Only rigid rotation is an exact model invariant. General current
        # strain is an approximation checked against known zero contact flow.
        assert abs(result['combined_flux_px_s']) < .05
        assert abs(result['combined_flux_px_s']) < result['combined_flux_sigma_px_s']


def test_objective_fit_rejects_unidentifiable_contact_evaluation():
    from src.analysis.objective_boundary_velocity import fit_objective_velocity
    X=np.ones((20,1));position=np.tile([4.,6.],(20,1))
    tangent=np.tile([1.,0.],(20,1));normal=np.tile([0.,1.],(20,1))
    fit=fit_objective_velocity(X,position,tangent,normal,np.zeros((20,2)),np.ones(20),
        np.ones(1),np.zeros(2),[1.,0.],[0.,1.],floor=.25,maximum_condition=1.e7)
    assert fit is None


def test_unknown_velocity_model_is_rejected():
    observer, seed, current, tau, path=scene()
    with pytest.raises(ValueError,match='velocity model'):
        BoundaryContactTransportObserver(seed,observer.ids,
            dict(boundary_xy=observer.boundary.surface,surface_s_px=observer.boundary.addresses,
                 tangent=observer.boundary.tangent,inward_normal=observer.boundary.normal),1.,
            config=replace(observer.config,velocity_model='magic'))


def test_noisy_arbitrary_motion_plus_rotation_preserves_same_profile_evaluation():
    observer, seed, current, tau, path=scene()
    s,d=seed.T
    normal=np.column_stack((-tau[:,1],tau[:,0]))
    velocity=(-3.1+.08*d+.0007*s*d)[:,None]*tau+(.3+.02*d)[:,None]*normal
    velocity+=np.random.default_rng(61).normal(0.,.2,velocity.shape)
    a=observe(observer,current,velocity,path=path)
    b=observe(observer,current,velocity+.25*np.column_stack((-current[:,1],current[:,0])),path=path)
    for old,new in zip([p for ps in a['models'] for p in ps],[p for ps in b['models'] for p in ps]):
        for key in ('flux_px_s','normal_relative_velocity_px_s','velocity_rms_px_s','velocity_holdout_px_s','leverage'):
            assert abs(old[key]-new[key]) < 1.e-9


def test_noncoincident_geometric_root_has_analytic_rotation_contribution():
    observer, seed, current, tau, path=scene()
    tip=np.array([0.,-2.]);omega=.2
    relative=current-tip
    velocity=omega*np.column_stack((-relative[:,1],relative[:,0]))
    result=observe(observer,current,velocity,tip=tip,path=path)
    for p in [p for ps in result['models'] for p in ps]:
        c=p['surface_xy']-tip
        delta=omega*np.array([-c[1],c[0]])
        expected=-float(delta@p['tangent_xy'])/p['surface_stretch']
        assert abs(p['flux_px_s']-expected) < 1.e-9


def test_objective_velocity_helper_is_strictly_coordinate_rotation_invariant():
    from src.analysis.objective_boundary_velocity import fit_objective_velocity
    from src.analysis.boundary_material_contact import _basis
    observer, seed, current, tau, path=scene()
    s,d=seed.T;X=_basis(s,d,0.,True);C=_basis(np.array([-12.5]),np.zeros(1),0.,True)[0]
    normal=np.column_stack((-tau[:,1],tau[:,0]))
    velocity=(-4.6+.08*d+.0007*s*d)[:,None]*tau
    weights=np.ones(len(seed))
    def fit(x,t,n,v,ct,cn):
        return fit_objective_velocity(X,x,t,n,v,weights,C,np.zeros(2),ct,cn,
            floor=.25,maximum_condition=1.e7)
    a=fit(current,tau,normal,velocity,[1.,0.],[0.,1.])
    angle=.7;Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    b=fit(current@Q.T,tau@Q.T,normal@Q.T,velocity@Q.T,np.array([1.,0.])@Q.T,np.array([0.,1.])@Q.T)
    assert np.max(abs(a['evaluation']-b['evaluation'])) < 1.e-10
    assert abs(a['leverage']-b['leverage']) < 1.e-10
    assert abs(a['rms']-b['rms']) < 1.e-10
