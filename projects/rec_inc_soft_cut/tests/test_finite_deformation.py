"""Ground-truth and topology tests for the optional mesh candidate."""
import cv2
import numpy as np
import pytest

from src.tracking.finite_deformation import (
    FiniteDeformationConfig, _Objective, _EquilibriumGap, _neo_hookean, _features, _material_gray,
    deformation_diagnostics, fit_material_mesh,
)


def scene():
    rng = np.random.default_rng(7401)
    image = cv2.GaussianBlur(rng.random((160, 220)).astype(np.float32), (0, 0), 1.)
    image = np.clip((image-.5)*3+.5, .03, .97)
    x, y = np.meshgrid(np.linspace(40, 155, 7), np.linspace(40, 115, 5))
    return image, np.column_stack((x.ravel(), y.ravel()))


def warp(image, F, t):
    return cv2.warpAffine(image, np.column_stack((F, t)), (image.shape[1], image.shape[0]))


def test_large_affine_and_shadow_improve_known_material_positions():
    image, p = scene()
    F = np.array([[1.22, .12], [-.08, .82]])
    t = np.array([-16., 19.])
    truth = p@F.T+t
    initial = truth+np.column_stack((.8*np.sin(p[:, 1]/13)+.5, .6*np.cos(p[:, 0]/15)-.4))
    target = warp(image, F, t)
    target *= np.linspace(.5, 1., image.shape[1])[None]
    target += .05
    result = fit_material_mesh(image, target, p, initial, np.ones(len(p), bool))
    before = np.linalg.norm(initial-truth, axis=1).mean()
    after = np.linalg.norm(result['positions']-truth, axis=1).mean()
    assert result['accepted'], result['optimizer_steps']
    assert after < .5*before, (before, after)
    assert after < .35
    assert result['supported'].mean() > .8
    assert result['holdout']['after_normalized_rms'] < result['holdout']['before_normalized_rms']
    assert np.all(result['jacobian'][result['eligible_triangles']] > 0)


def test_rigid_rotation_is_strain_free_and_strain_is_objective():
    p = np.array([[0., 0.], [2., 0.], [0., 3.], [2., 3.]])
    tri = np.array([[0, 1, 2], [1, 3, 2]])
    angle = 1.1
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pure = deformation_diagnostics(p, p@R.T+[20, -30], tri)
    np.testing.assert_allclose(pure['green_lagrange_strain'], 0, atol=1.e-14)
    F = np.array([[1.8, .35], [.0, .65]])
    first = deformation_diagnostics(p, p@F.T, tri)
    rotated = deformation_diagnostics(p, (p@F.T)@R.T+[10, 4], tri)
    np.testing.assert_allclose(first['green_lagrange_strain'], rotated['green_lagrange_strain'], atol=1.e-14)
    np.testing.assert_allclose(first['jacobian'], rotated['jacobian'], atol=1.e-14)


def test_blade_and_cut_edges_are_not_coupled():
    image, p = scene()
    blade = np.zeros(image.shape, bool)
    blade[:, 95:101] = True
    result = fit_material_mesh(image, image, p, p, np.ones(len(p), bool), blade_mask=(blade, blade),
                               config=FiniteDeformationConfig(max_nfev=4, smoothing_levels=(.55,)))
    selected = p[result['triangles'][result['eligible_triangles']]]
    assert len(selected) > 0
    assert not np.any((selected[:, :, 0].min(1) < 95) & (selected[:, :, 0].max(1) >= 101))
    assert (~result['eligible_triangles']).any()
    edge = result['triangles'][result['eligible_triangles']][0, :2]
    cut = fit_material_mesh(image, image, p, p, np.ones(len(p), bool), excluded_edges=[edge],
                            config=FiniteDeformationConfig(max_nfev=3, smoothing_levels=(.55,)))
    for tri in cut['triangles'][cut['eligible_triangles']]:
        assert not set(edge).issubset(tri)


def test_no_observation_or_texture_does_not_invent_measurements():
    image, p = scene()
    initial = p+[2., 1.]
    absent = fit_material_mesh(image, image, p, initial, np.zeros(len(p), bool))
    assert not absent['supported'].any()
    assert absent['inferred'].all()
    assert not absent['optimizer_steps']
    np.testing.assert_array_equal(absent['positions'], initial.astype(np.float32))
    blank = np.full_like(image, .5)
    result = fit_material_mesh(blank, blank, p, initial, np.ones(len(p), bool))
    assert not result['supported'].any()
    assert not result['accepted']
    np.testing.assert_array_equal(result['positions'], initial.astype(np.float32))


def test_model_completion_retains_inferred_flag_and_input_ids():
    image, p = scene()
    target = warp(image, np.eye(2), [1.5, -.6])
    observed = np.ones(len(p), bool)
    observed[::4] = False
    result = fit_material_mesh(image, target, p, p+[1.1, -.2], observed)
    assert not result['supported'][~observed].any()
    assert result['inferred'][~observed].all()
    assert len(result['positions']) == len(p)
    np.testing.assert_array_equal(result['input_observed'], observed)


def test_hidden_region_is_not_an_observation():
    image, p = scene()
    visible = np.ones(image.shape, bool)
    visible[:, 105:] = False
    target = image.copy()
    target[~visible] = .1
    result = fit_material_mesh(image, target, p, p, np.ones(len(p), bool),
                               material_mask=(np.ones_like(visible), visible))
    assert not result['supported'][p[:, 0] >= 105].any()
    assert result['supported'][p[:, 0] < 75].all()


def test_folded_initial_elements_are_excluded_and_invalid_input_rejected():
    image, p = scene()
    initial = p.copy()
    initial[:, 0] = 190-p[:, 0]
    result = fit_material_mesh(image, image, p, initial, np.ones(len(p), bool))
    assert not result['eligible_triangles'].any()
    assert not result['supported'].any()
    assert not result['accepted']
    with pytest.raises(ValueError, match='unique'):
        fit_material_mesh(image, image, np.array([[1, 1], [1, 1], [2, 2]]),
                          np.ones((3, 2)), np.ones(3))
    with pytest.raises(ValueError, match='nondegenerate'):
        fit_material_mesh(image, image, np.array([[1, 1], [2, 2], [3, 3]]),
                          np.ones((3, 2)), np.ones(3))


def test_sparse_jacobian_matches_image_and_finite_strain_differences():
    rng = np.random.default_rng(109)
    image = rng.random((50, 50))
    p = np.array([[12., 12.], [27., 12.], [12., 28.], [27., 28.]])
    tri = np.array([[0, 1, 2], [1, 3, 2]])
    # Compressed map activates the Jacobian barrier, while the second element
    # has a distinct strain so that all regularization derivatives are tested.
    q = p@np.array([[.6, .05], [0, .11]]).T+[5.34, 19.43]
    q[3] += [.13, .02]
    objective = _Objective(image, image, p, q, tri,
                           np.array([[.21, .34, .45], [.43, .26, .31]]),
                           FiniteDeformationConfig())
    x = q[objective.active].ravel()
    analytic = objective.evaluate(x, True).toarray()
    numeric = np.empty_like(analytic)
    for j in range(len(x)):
        delta = np.zeros_like(x)
        delta[j] = 1.e-6
        numeric[:, j] = (objective.evaluate(x+delta)-objective.evaluate(x-delta))/(2.e-6)
    np.testing.assert_allclose(analytic, numeric, atol=2.e-8, rtol=2.e-6)


def test_local_moving_highlight_does_not_drag_the_whole_mesh():
    image, p = scene()
    truth = p+[1.7, -.8]
    initial = truth+[.6, -.5]
    target = warp(image, np.eye(2), [1.7, -.8])
    target[70:83, 91:106] = .99
    result = fit_material_mesh(image, target, p, initial, np.ones(len(p), bool))
    away = (p[:, 0] < 75) | (p[:, 0] > 125) | (p[:, 1] < 55) | (p[:, 1] > 100)
    errors = np.linalg.norm(result['positions']-truth, axis=1)
    assert np.median(errors[away]) < .2
    assert np.mean(errors[away]) < .35


def test_masked_statistics_ignore_bright_tool_in_all_preprocessing_stages():
    image, _ = scene()
    valid = np.indices(image.shape)[1] < 100
    dark, bright = image.copy(), image.copy()
    dark[~valid], bright[~valid] = .0, 1.
    config = FiniteDeformationConfig(feature_mode='material-masked')
    dgray, dv = _material_gray(dark, valid, config)
    bgray, bv = _material_gray(bright, valid, config)
    np.testing.assert_array_equal(dgray, bgray)
    np.testing.assert_array_equal(dv, bv)
    masked_dark, supported = _features(dgray, config, 1.2, dv, return_valid=True)
    masked_bright = _features(bgray, config, 1.2, bv)
    np.testing.assert_array_equal(masked_dark, masked_bright)
    near_edge = supported & (np.indices(image.shape)[1] >= 92)
    assert near_edge.sum() > 100
    # The counterfactual really affects the old unmasked statistics.
    legacy = FiniteDeformationConfig()
    old_change = np.abs(_features(dark, legacy, 1.2)-_features(bright, legacy, 1.2))
    assert np.mean(old_change[near_edge]) > .2


def test_all_valid_mask_keeps_legacy_features_and_fit():
    from src.tracking.material_patch_tracker import _gray
    image, p = scene()
    valid = np.ones(image.shape, bool)
    config = FiniteDeformationConfig(feature_mode='material-masked', max_nfev=8,
                                     smoothing_levels=(.55,))
    converted, support = _material_gray(image, valid, config)
    np.testing.assert_array_equal(converted, _gray(image))
    assert support.all()
    np.testing.assert_array_equal(_features(converted, config, .55, valid),
                                  _features(_gray(image), FiniteDeformationConfig(), .55))
    target = warp(image, np.eye(2), [1.4, -.6])
    masked = fit_material_mesh(image, target, p, p+[1., -.3], np.ones(len(p), bool),
                               material_mask=(valid, valid), config=config)
    legacy = fit_material_mesh(image, target, p, p+[1., -.3], np.ones(len(p), bool),
                               config=FiniteDeformationConfig(max_nfev=8, smoothing_levels=(.55,)))
    np.testing.assert_array_equal(masked['positions'], legacy['positions'])
    np.testing.assert_array_equal(masked['supported'], legacy['supported'])


def test_masked_fit_is_invariant_to_tool_brightness_and_records_mode():
    image, p = scene()
    truth = p+[1.2, -.6]
    target = warp(image, np.eye(2), [1.2, -.6])
    blade = np.indices(image.shape)[1] >= 130
    material = np.ones(image.shape, bool)
    material[:8] = False
    config = FiniteDeformationConfig(feature_mode='material-masked', max_nfev=14,
                                     smoothing_levels=(.55,))
    results = []
    for outside in (0., 1.):
        ref, cur = image.copy(), target.copy()
        ref[blade | ~material] = outside
        cur[blade | ~material] = 1-outside
        results.append(fit_material_mesh(ref, cur, p, truth+[.5, -.3], np.ones(len(p), bool),
                       material_mask=(material, material), blade_mask=(blade, blade), config=config))
    first, second = results
    np.testing.assert_array_equal(first['positions'], second['positions'])
    np.testing.assert_array_equal(first['supported'], second['supported'])
    assert first['accepted']
    assert first['supported'].any()
    assert np.linalg.norm(first['positions'][first['supported']]-truth[first['supported']], axis=1).mean() < .2
    assert not first['supported'][p[:, 0] >= 130].any()
    assert first['feature_normalization']['mode'] == 'material-masked'
    assert first['feature_normalization']['current_mask_provided']


def test_mask_boundary_does_not_create_texture_or_support_from_missing_data():
    image = np.full((80, 100), .5, np.float32)
    valid = np.indices(image.shape)[1] < 50
    image[~valid] = np.nan
    config = FiniteDeformationConfig(feature_mode='material-masked')
    gray, supported = _material_gray(image, valid, config)
    features, support = _features(gray, config, .55, supported, return_valid=True)
    np.testing.assert_allclose(features[support], 0, atol=1.e-4)
    assert support.any() and not support[~valid].any()
    tiny = np.zeros(image.shape, bool)
    tiny[40, 40] = True
    isolated, visible = _material_gray(image, tiny, config)
    _, isolated_support = _features(isolated, config, .55, visible, return_valid=True)
    assert not isolated_support.any()
    empty = np.zeros(image.shape, bool)
    p = np.array([[20., 20.], [40., 20.], [20., 50.], [40., 50.]])
    result = fit_material_mesh(image, image, p, p, np.ones(len(p), bool),
                               material_mask=(empty, empty), config=config)
    assert not result['supported'].any()
    assert not result['image_eligible_triangles'].any()
    assert not result['accepted']
    assert not result['optimizer_steps']
    np.testing.assert_array_equal(result['positions'], p.astype(np.float32))


@pytest.mark.parametrize('kwargs', [dict(feature_mode='unrecognized'),
                                  dict(min_feature_support=0), dict(min_feature_support=1.1)])
def test_invalid_feature_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError, match='configuration'):
        FiniteDeformationConfig(**kwargs)


def mechanics_scene():
    from scipy.spatial import Delaunay
    x, y = np.meshgrid(np.arange(5.)*6+10, np.arange(5.)*6+10)
    p = np.column_stack((x.ravel(), y.ravel()))
    c = FiniteDeformationConfig(equilibrium_weight=.05,
                                equilibrium_reference='stress-free-assumed')
    return p, Delaunay(p).simplices, c


def test_equilibrium_affine_compression_is_balanced_without_zero_boundary_traction():
    p, tri, c = mechanics_scene()
    eq = _EquilibriumGap(p, tri, c)
    assert eq.interior.sum() == 9
    F = np.array([[1.4, .25], [.05, .61]])
    q = p@F.T+[3, -2]
    # J is deliberately not one; a uniformly loaded compression must be allowed.
    assert abs(np.linalg.det(F)-1) > .1
    P, _ = _neo_hookean(F[None], c.equilibrium_bulk_shear_ratio, c.min_jacobian)
    assert np.linalg.norm(P) > 1
    np.testing.assert_allclose(eq.evaluate(q), 0, atol=1.e-12)
    # A single element has only unknown-force boundary nodes, not free tractions.
    boundary_only = _EquilibriumGap(p, tri[:1], c)
    assert not boundary_only.interior.any()
    np.testing.assert_array_equal(boundary_only.evaluate(q), 0)
    np.testing.assert_array_equal(boundary_only.evaluate(q, True), 0)


def test_equilibrium_nonuniform_force_rotates_objectively():
    p, tri, c = mechanics_scene()
    eq = _EquilibriumGap(p, tri, c)
    q = p.copy()
    q[12] += [1.1, -.7]
    first = eq.evaluate(q).reshape(-1, 2)
    assert np.linalg.norm(first) > .1
    angle = .87
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = eq.evaluate(q@R.T+[50, -23]).reshape(-1, 2)
    np.testing.assert_allclose(rotated, first@R.T, atol=1.e-12)
    np.testing.assert_allclose(np.sum(rotated**2), np.sum(first**2), atol=1.e-11)
    np.testing.assert_allclose(eq.evaluate(p@R.T+[50, -23]), 0, atol=1.e-12)


def test_cut_hole_boundary_is_never_treated_as_unloaded_interior():
    p, tri, c = mechanics_scene()
    baseline = _EquilibriumGap(p, tri, c)
    cut = np.flatnonzero(np.all(np.isin(tri, [6, 7, 11, 12]), axis=1))[0]
    removed_nodes = tri[cut]
    assert np.isin(removed_nodes, baseline.active[baseline.interior]).all()
    hole = _EquilibriumGap(p, np.delete(tri, cut, axis=0), c)
    assert not np.isin(removed_nodes, hole.active[hole.interior]).any()
    q = p@np.array([[1.2, .17], [-.03, .8]]).T
    np.testing.assert_allclose(hole.evaluate(q), 0, atol=1.e-12)


@pytest.mark.parametrize('compression', [1., .02, -.02])
def test_neo_hookean_tangent_matches_derivative_including_safe_trial_continuation(compression):
    F = np.array([[[1.2, .13], [.07, compression]]])
    _, analytic = _neo_hookean(F, 10., .08)
    numeric = np.empty_like(analytic)
    for k in range(2):
        for l in range(2):
            delta = np.zeros_like(F)
            delta[0, k, l] = 1.e-6
            plus, _ = _neo_hookean(F+delta, 10., .08)
            minus, _ = _neo_hookean(F-delta, 10., .08)
            numeric[:, :, :, k, l] = (plus-minus)/2.e-6
    np.testing.assert_allclose(analytic, numeric, atol=2.e-7, rtol=2.e-7)


def test_consistent_mass_equilibrium_jacobian_and_energy_match_direct_assembly():
    p, tri, c = mechanics_scene()
    eq = _EquilibriumGap(p, tri, c)
    q = p.copy()
    q[12] += [.8, -.4]
    analytic = eq.evaluate(q, True)
    numeric = np.empty_like(analytic)
    for j, node in enumerate(eq.active):
        for component in range(2):
            delta = np.zeros_like(q)
            delta[node, component] = 1.e-6
            numeric[:, j*2+component] = (eq.evaluate(q+delta)-eq.evaluate(q-delta))/2.e-6
    np.testing.assert_allclose(analytic, numeric, atol=2.e-8, rtol=2.e-6)
    # The whitened objective is exactly R.T M^-1 R, not a mesh-dependent sum of forces.
    r = eq.evaluate(q).reshape(-1, 2)
    force = eq.cholesky@r
    assert np.linalg.norm(force[eq.interior]) > .1
    np.testing.assert_allclose(force[~eq.interior], 0, atol=1.e-12)
    mass = eq.cholesky@eq.cholesky.T
    np.testing.assert_allclose(np.sum(r*r), np.sum(force*np.linalg.solve(mass, force)), atol=1.e-12)


def test_combined_equilibrium_objective_sparse_jacobian_and_quadratic_loss():
    p, tri, c = mechanics_scene()
    rng = np.random.default_rng(101)
    image = rng.random((50, 50))
    q = p+[.31, .27]
    q[12] += [.2, -.1]
    # Images exclude one triangle while mechanics has the whole eligible topology.
    objective = _Objective(image, image, p, q, tri[:-1],
                           np.array([[.21, .34, .45]]), c, mechanical_triangles=tri)
    x = q[objective.active].ravel()
    analytic = objective.evaluate(x, True).toarray()
    numeric = np.empty_like(analytic)
    for j in range(len(x)):
        delta = np.zeros_like(x)
        delta[j] = 1.e-6
        numeric[:, j] = (objective.evaluate(x+delta)-objective.evaluate(x-delta))/2.e-6
    np.testing.assert_allclose(analytic, numeric, atol=3.e-8, rtol=2.e-6)
    z = (objective.evaluate(x)/c.robust_scale)**2
    rho = objective.loss(z)
    size = 2*len(objective.equilibrium.active)
    np.testing.assert_array_equal(rho[0, -size:], z[-size:])
    np.testing.assert_array_equal(rho[1, -size:], 1)
    np.testing.assert_array_equal(rho[2, -size:], 0)
    # Image/kinematic/barrier rows retain exactly SciPy soft_l1; only the last
    # mechanics rows use squared norm, including the optimizer f_scale factor.
    root = np.sqrt(1+z[:-size])
    np.testing.assert_array_equal(rho[:, :-size],
                                  np.array([2*(root-1), 1/root, -.5/root**3]))
    residual = objective.evaluate(x)
    direct_cost = (np.sum(c.robust_scale**2*(root-1))
                   + .5*np.sum(residual[-size:]**2))
    np.testing.assert_allclose(objective.cost(x), direct_cost, atol=1.e-13)


def test_optional_mechanics_runs_with_explicit_assumption_and_preserves_evidence():
    image, p = scene()
    F = np.array([[1.08, .05], [-.01, .94]])
    truth = p@F.T+[-3., 3.]
    target = warp(image, F, [-3., 3.])
    observed = np.ones(len(p), bool)
    observed[12] = False
    config = FiniteDeformationConfig(equilibrium_weight=.02,
        equilibrium_reference='stress-free-assumed', max_nfev=15, smoothing_levels=(.55,))
    result = fit_material_mesh(image, target, p, truth+[.4, -.25], observed, config=config)
    eq = result['equilibrium_gap']
    assert result['accepted']
    assert eq['enabled'] and eq['interior_node_count'] > 0 and eq['boundary_node_count'] > 0
    assert eq['reference_assumption'] == 'stress-free-assumed'
    assert np.isfinite(eq['after_normalized_rms'])
    assert len(result['equilibrium_interior_nodes']) == eq['interior_node_count']
    assert not result['supported'][12]
    assert result['inferred'][12]
    assert np.linalg.norm(result['positions']-truth, axis=1).mean() < .2


@pytest.mark.parametrize('kwargs', [dict(equilibrium_weight=.01),
    dict(equilibrium_weight=-1), dict(equilibrium_weight=float('nan')),
    dict(equilibrium_bulk_shear_ratio=.5), dict(equilibrium_bulk_shear_ratio=float('inf')),
    dict(equilibrium_reference='frame-is-stress-free-without-check')])
def test_invalid_or_unacknowledged_equilibrium_assumptions_are_rejected(kwargs):
    with pytest.raises(ValueError, match='configuration'):
        FiniteDeformationConfig(**kwargs)


def _propose_translation(monkeypatch, translation):
    """Inject a controlled unconstrained optimum; evaluate its real image cost."""
    from types import SimpleNamespace
    import src.tracking.finite_deformation as module

    def solve(fun, x0, **kwargs):
        x = (x0.reshape(-1, 2)+translation).ravel()
        assert np.all(x >= kwargs['bounds'][0]) and np.all(x <= kwargs['bounds'][1])
        return SimpleNamespace(x=x, cost=fun.__self__.cost(x), success=True,
                               status=1, nfev=1, optimality=0.)
    monkeypatch.setattr(module, 'least_squares', solve)


def test_feasible_backtracking_retains_boundary_failures_and_uses_unchanged_gates(monkeypatch):
    image, p = scene()
    p[:, 0] *= 148/155
    mask = np.indices(image.shape)[1] < 150
    target = warp(image, np.eye(2), [3., 0.])
    _propose_translation(monkeypatch, [3., 0.])
    arguments = (image, target, p, p, np.ones(len(p), bool))
    options = dict(feature_mode='material-masked', smoothing_levels=(.55,))
    disabled = fit_material_mesh(*arguments, material_mask=(mask, mask),
                                 config=FiniteDeformationConfig(**options))
    result = fit_material_mesh(*arguments, material_mask=(mask, mask),
        config=FiniteDeformationConfig(**options, feasible_backtracking=True))
    assert not disabled['accepted']
    assert len(disabled['optimizer_steps'][0]['line_search']) == 1
    np.testing.assert_array_equal(disabled['positions'], p.astype(np.float32))
    assert result['accepted']
    step = result['optimizer_steps'][0]
    assert 0 < step['step_scale'] < 1
    assert not step['accepted_full_optimizer_solution']
    assert step['candidate_failures']['material'] > 0
    assert step['candidate_failures']['feature'] > 0
    assert not any(step['initial_failures'].values())
    assert not any(step['line_search'][-1]['failures'].values())
    assert step['accepted_cost'] <= step['initial_cost']
    np.testing.assert_allclose(result['optimizer_candidate_positions'][0], p+[3, 0], atol=1.e-5)
    np.testing.assert_allclose(result['positions'], p+step['step_scale']*np.array([3, 0]), atol=1.e-5)
    assert np.linalg.norm(result['positions']-(p+[3, 0]), axis=1).mean() < 3
    # Raw unsafe candidates remain inspectable rather than overwritten by the accepted fraction.
    np.testing.assert_array_equal(result['optimizer_candidate_positions'],
                                  disabled['optimizer_candidate_positions'])


def test_backtracking_never_accepts_a_safe_but_worse_image_objective(monkeypatch):
    image, p = scene()
    _propose_translation(monkeypatch, [2., 0.])
    result = fit_material_mesh(image, image, p, p, np.ones(len(p), bool),
        config=FiniteDeformationConfig(feasible_backtracking=True, smoothing_levels=(.55,)))
    assert not result['accepted']
    step = result['optimizer_steps'][0]
    assert step['step_scale'] == 0
    assert step['accepted_cost'] is None
    assert len(step['line_search']) == 13
    assert all(trial['safe'] and not trial['objective_not_increased'] for trial in step['line_search'])
    np.testing.assert_array_equal(result['positions'], p.astype(np.float32))


def test_safe_full_step_is_identical_with_or_without_opt_in_backtracking():
    image, p = scene()
    target = warp(image, np.eye(2), [1.3, -.7])
    args = image, target, p, p+[1., -.4], np.ones(len(p), bool)
    normal = fit_material_mesh(*args)
    searched = fit_material_mesh(*args, config=FiniteDeformationConfig(feasible_backtracking=True))
    assert normal['accepted'] and searched['accepted']
    assert all(step['step_scale'] == 1 for step in searched['optimizer_steps'])
    np.testing.assert_array_equal(normal['positions'], searched['positions'])
    np.testing.assert_array_equal(normal['supported'], searched['supported'])
