import numpy as np

from src.analysis.contact_transport_observations import (
    ContactTransportConfig, ContactTransportObserver)


def scene(model='affine'):
    s = np.arange(-72., 72.1, 6.)
    boundary = np.column_stack((100. + s, np.full(len(s), 60.)))
    depths = np.array([1.5, 7.5, 19.5, 31.5, 49.5, 73.5, 109.5])
    seed = np.stack([boundary + [0., d] for d in depths], axis=1).reshape(-1, 2)
    reference = dict(boundary_xy=boundary, surface_s_px=s, tangent=np.array([1., 0.]),
                     inward_normal=np.array([0., 1.]), contact_s_px=0.)
    config = ContactTransportConfig(reference_half_windows_px=(72.,),
        maximum_depths_px=(120.,), models=(model,), donor_radius_px=200.)
    observer = ContactTransportObserver(seed, np.arange(len(seed)), reference, 1., config)
    return observer, seed


def observe(observer, current, velocity, tip, blade_velocity, **kwargs):
    n = len(current)
    return observer.estimate_frame(current, np.ones(n, bool), tip, blade_velocity,
        velocity, np.ones(n, bool), branch_labels=np.ones(n, int), **kwargs)


def test_rigid_translation_of_material_and_blade_has_no_address_transport():
    observer, seed = scene()
    velocity = np.broadcast_to([3., 7.], seed.shape)
    result = observe(observer, seed + [10., 20.], velocity, [110., 80.], [3., 7.])
    assert result['combined_flux_valid']
    assert abs(result['combined_flux_px_s']) < 1.e-10
    assert result['absolute_valid']
    assert abs(result['absolute_s_px']) < 1.e-10


def test_tangential_sliding_divides_by_surface_stretch():
    observer, seed = scene()
    current = (seed - [100., 60.]) * [2., .8] + [100., 60.]
    velocity = np.broadcast_to([-6., 0.], seed.shape)
    result = observe(observer, current, velocity, [100., 60.], [0., 0.])
    assert result['combined_flux_valid']
    assert abs(result['combined_flux_px_s'] - 3.) < 1.e-10
    assert abs(result['sides'][0]['surface_stretch'] - 2.) < 1.e-10


def test_high_normal_compression_does_not_force_sticking():
    observer, seed = scene()
    current = (seed - [100., 60.]) * [1.5, .1] + [100., 60.]
    velocity = np.broadcast_to([-6., 0.], seed.shape)
    result = observe(observer, current, velocity, [100., 60.], [0., 0.])
    assert result['sides'][0]['high_compression']
    assert result['combined_flux_valid']
    assert abs(result['combined_flux_px_s'] - 4.) < 1.e-10
    assert not result['compression_imposes_zero_slip']


def test_static_material_and_penetrating_blade_reject_normal_incompatibility():
    observer, seed = scene()
    result = observe(observer, seed, np.zeros_like(seed), [100., 60.], [0., 8.])
    assert not result['combined_flux_valid']
    assert 'normal_velocity_incompatible' in result['sides'][0]['reasons']
    result = observe(observer, seed, np.zeros_like(seed), [100., 95.], [0., 8.])
    assert not result['absolute_valid']
    assert not result['sides'][0]['geometry_compatible']


def test_depth_shear_is_extrapolated_to_surface_before_velocity_evaluation():
    observer, seed = scene('bilinear')
    s, d = seed[:, 0] - 100., seed[:, 1] - 60.
    current = np.column_stack((100. + 1.4*s + .3*d + .002*s*d, 60. + .5*d))
    velocity = np.column_stack((-4.2 + .2*d + .003*s*d, np.zeros(len(seed))))
    result = observe(observer, current, velocity, [100., 60.], [0., 0.])
    assert result['combined_flux_valid']
    assert abs(result['combined_flux_px_s'] - 3.) < 1.e-9
    assert abs(result['sides'][0]['normal_shear'] - .3) < 1.e-9


def test_missing_material_ids_are_not_observations_or_zero_motion():
    observer, seed = scene()
    support = np.ones(len(seed), bool)
    support[::3] = False
    altered = seed.copy()
    altered[~support] = [9999., -9999.]
    velocity = np.broadcast_to([-4., 0.], seed.shape).copy()
    velocity[~support] = [9999., -9999.]
    result = observer.estimate_frame(altered, support, [100., 60.], [0., 0.],
        velocity, support, branch_labels=np.ones(len(seed), int))
    assert abs(result['combined_flux_px_s'] - 4.) < 1.e-9
    assert not np.any(np.isin(result['sides'][0]['support_ids'], np.flatnonzero(~support)))
    empty = observer.estimate_frame(seed, np.zeros(len(seed), bool), [100., 60.], [0., 0.],
        velocity, np.zeros(len(seed), bool))
    assert not empty['combined_flux_valid']
    assert np.isnan(empty['combined_flux_px_s'])


def test_absolute_address_is_unforced_and_interior():
    observer, seed = scene()
    result = observe(observer, seed, np.zeros_like(seed), [113., 60.], [0., 0.], contact_s_px=0.)
    assert result['absolute_valid']
    assert abs(result['absolute_s_px'] - 13.) < 1.e-10
    # Normal penetration cannot become false surface migration.
    impossible = observe(observer, seed, np.zeros_like(seed), [113., 80.], [0., 0.])
    assert not impossible['absolute_valid']
    edge = observe(observer, seed, np.zeros_like(seed), [172., 60.], [0., 0.])
    assert not edge['absolute_valid']


def test_below_tip_points_are_not_counted_on_both_blade_sides():
    observer, seed = scene()
    result = observer.estimate_frame(seed, np.ones(len(seed), bool), [100., 60.], [0., 0.],
        np.zeros_like(seed), np.ones(len(seed), bool), branch_labels=np.full(len(seed), 3))
    a, b = result['sides']
    assert not np.intersect1d(a['support_ids'], b['support_ids']).size
    assert a['model_count'] and b['model_count']


def test_deforming_surface_transport_and_normal_check_at_absolute_root():
    observer, seed = scene()
    reference = seed - [100., 60.]
    F = np.array([[1.3, .2], [.4, .8]])
    Fdot = np.array([[.03, .02], [.01, -.02]])
    current = reference @ F.T + [100., 60.]
    velocity = reference @ Fdot.T + [1., 2.]
    s = 17.
    tip = F @ [s, 0.] + [100., 60.]
    tip_velocity = Fdot @ [s, 0.] + [1., 2.] + F[:, 0] * 4.
    result = observe(observer, current, velocity, tip, tip_velocity, contact_s_px=s)
    assert result['combined_flux_valid']
    assert abs(result['combined_flux_px_s'] - 4.) < 1.e-9
    assert result['absolute_valid']
    assert abs(result['absolute_s_px'] - s) < 1.e-9
    # The provisional address is wrong, but the unconstrained forward root
    # has its own normal-velocity check and can correct that provisional state.
    root = observe(observer, current, velocity, tip, tip_velocity, contact_s_px=0.)
    assert root['absolute_valid']
    assert abs(root['absolute_s_px'] - s) < 1.e-9
    tau = F[:, 0] / np.linalg.norm(F[:, 0])
    incompatible_velocity = tip_velocity + 10. * np.array([-tau[1], tau[0]])
    bad = observe(observer, current, velocity, tip, incompatible_velocity, contact_s_px=0.)
    assert not bad['absolute_valid']


def test_zero_uncertain_velocity_is_not_observed_sticking():
    observer, seed = scene()
    result = observe(observer, seed, np.zeros_like(seed), [100., 60.], [0., 0.],
                     velocity_sigma_px_s=np.full(len(seed), 15.))
    assert abs(result['combined_flux_px_s']) < 1.e-9
    assert not result['sides'][0]['near_stick_observed']
