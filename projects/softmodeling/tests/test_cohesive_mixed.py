"""Mixed-mode potential and objective interface regression tests; no FE loading."""
import unittest
from dataclasses import replace

import numpy as np

from prototype.cohesive_mixed import MixedModeLaw, interface_response


class MixedModeLawTests(unittest.TestCase):
    def setUp(self):
        self.law = MixedModeLaw(shear_strength_mpa=.3)

    def test_pure_modes_and_proportional_paths_reach_shared_toughness(self):
        law = self.law
        q = np.linspace(0., law.final_opening_mm, 20001)
        alpha = law.shear_stiffness_mpa_per_mm/law.normal_stiffness_mpa_per_mm
        for angle in (0., .4, 1., np.pi/2, -.7):
            with self.subTest(angle=angle):
                direction = np.array([np.cos(angle), np.sin(angle)/np.sqrt(alpha)])
                jumps = q[:, None]*direction
                response = law.response(jumps[:, 0], jumps[:, 1], 0.)
                work = np.trapezoid(response['traction']@direction, q)
                self.assertAlmostEqual(work, law.toughness_n_per_mm, places=9)
                self.assertAlmostEqual(response['dissipated'][-1], law.toughness_n_per_mm)
                self.assertEqual(response['damage'][-1], 1.)
        normal = law.response(law.peak_opening_mm, 0., 0.)
        shear_peak = law.shear_strength_mpa/law.shear_stiffness_mpa_per_mm
        shear = law.response(0., shear_peak, 0.)
        self.assertAlmostEqual(normal['traction'][0], law.normal_strength_mpa)
        self.assertAlmostEqual(shear['traction'][1], law.shear_strength_mpa)

    def test_elliptical_onset_and_positive_initial_shear_stiffness(self):
        law = self.law
        response = law.response(.001, .001, 0.)
        criterion = np.sum((response['traction']/np.array([
            law.normal_strength_mpa, law.shear_strength_mpa]))**2)
        self.assertAlmostEqual(criterion, response['initiation_ratio']**2)
        self.assertEqual(response['damage'], 0.)
        shear = law.response(0., 0., 0.)
        self.assertEqual(shear['tangent'][1, 1], law.shear_stiffness_mpa_per_mm)
        self.assertGreater(shear['tangent'][1, 1], 0.)

    def test_compression_does_not_drive_damage_but_shear_can(self):
        law = self.law
        normal = law.response(-.1, 0., 0.)
        self.assertEqual(normal['damage'], 0.)
        np.testing.assert_array_equal(normal['traction'], [0., 0.])
        sheared = law.response(-.1, .01, 0.)
        self.assertGreater(sheared['damage'], 0.)
        self.assertEqual(sheared['traction'][0], 0.)
        self.assertGreater(sheared['traction'][1], 0.)

    def test_unloading_reloading_and_failure_are_irreversible(self):
        law = self.law
        first = law.response(.02, .01, 0.)
        old = first['history']
        unload = law.response(.001, -.002, old)
        reload = law.response(.02, .01, unload['history'])
        for response in (unload, reload):
            self.assertEqual(response['history'], old)
            self.assertEqual(response['damage'], first['damage'])
            self.assertEqual(response['dissipated'], first['dissipated'])
            self.assertEqual(response['dissipation_increment'], 0.)
        np.testing.assert_allclose(reload['traction'], first['traction'])
        failed = law.response(-.01, .003, 2*law.final_opening_mm)
        self.assertEqual(failed['damage'], 1.)
        np.testing.assert_array_equal(failed['traction'], [0., 0.])
        np.testing.assert_array_equal(failed['tangent'], np.zeros((2, 2)))

    def test_nonproportional_loading_has_nonnegative_dissipation_and_work_balance(self):
        law = self.law
        vertices = np.array([[0., 0.], [.018, 0.], [.006, .025],
                             [-.004, .009], [.016, -.025], [0., 0.]])
        path = np.concatenate([np.linspace(a, b, 1001)[1:]
                               for a, b in zip(vertices[:-1], vertices[1:])])
        old = 0.
        last_u = vertices[0]
        last_t = np.zeros(2)
        work = 0.
        old_dissipation = 0.
        for jump in path:
            response = law.response(*jump, old)
            self.assertGreaterEqual(response['history'], old)
            self.assertGreaterEqual(response['dissipated'], old_dissipation)
            self.assertGreaterEqual(response['dissipation_increment'], 0.)
            work += .5*(response['traction']+last_t)@(jump-last_u)
            old, old_dissipation = response['history'], response['dissipated']
            last_u, last_t = jump, response['traction']
        self.assertAlmostEqual(work, response['stored']+response['dissipated'], delta=3e-8)

    def test_local_tangent_and_incremental_potential_derivatives(self):
        law = self.law
        for u, old in (([.012, .007], 0.), ([.002, -.003], .025),
                       ([-.01, .015], 0.), ([.001, .0002], 0.)):
            with self.subTest(jump=u, history=old):
                u = np.asarray(u)
                response = law.response(*u, old)
                finite = np.empty((2, 2))
                gradient = np.empty(2)
                h = 1e-7
                for j in range(2):
                    shift = np.eye(2)[j]*h
                    plus, minus = law.response(*(u+shift), old), law.response(*(u-shift), old)
                    finite[:, j] = (plus['traction']-minus['traction'])/(2*h)
                    gradient[j] = (plus['incremental_potential']-minus['incremental_potential'])/(2*h)
                np.testing.assert_allclose(response['tangent'], finite, rtol=2e-8, atol=2e-8)
                np.testing.assert_allclose(response['traction'], gradient, rtol=2e-8, atol=2e-10)

    def test_broadcasting_history_is_not_mutated(self):
        old = np.array([[.01, .02], [.03, .04]])
        frozen = old.copy()
        response = self.law.response(.03, np.array([0., .02]), old)
        self.assertEqual(response['traction'].shape, (2, 2, 2))
        np.testing.assert_array_equal(old, frozen)
        response['history'][:] = 0.
        np.testing.assert_array_equal(old, frozen)

    def test_disabled_damage_is_anisotropic_elastic(self):
        law = MixedModeLaw(shear_strength_mpa=.3, damage_enabled=False)
        response = law.response(.1, -.2, .3)
        np.testing.assert_allclose(response['traction'], [5., -22.5])
        self.assertEqual(response['damage'], 0.)
        self.assertEqual(response['dissipated'], 0.)

    def test_history_recovery_and_invalid_parameters(self):
        h = np.array([[0., .02], [.03, .08]])
        response = self.law.history_response(h)
        np.testing.assert_array_equal(response['history'], h)
        self.assertEqual(response['damage'][-1, -1], 1.)
        self.assertEqual(response['dissipated'][-1, -1], self.law.toughness_n_per_mm)
        for kwargs in ({'normal_strength_mpa': 0.}, {'shear_strength_mpa': -1.},
                       {'normal_stiffness_mpa_per_mm': np.nan},
                       {'toughness_n_per_mm': 1e-9}, {'damage_enabled': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MixedModeLaw(**kwargs)
        with self.assertRaises(ValueError):
            self.law.response(0., 0., -.01)

    def test_optional_compression_is_reversible_and_does_not_drive_damage(self):
        law = replace(self.law, compression_stiffness_mpa_per_mm=80.)
        opening = np.r_[np.linspace(0., -.01, 1001), np.linspace(-.01, 0., 1001)[1:]]
        q = law.response(opening, 0., 0.)
        np.testing.assert_array_equal(q['damage'], 0.)
        np.testing.assert_array_equal(q['history'], 0.)
        np.testing.assert_array_equal(q['dissipated'], 0.)
        np.testing.assert_array_equal(q['damageable_stored'], 0.)
        np.testing.assert_allclose(q['compression_traction'], 80.*opening)
        np.testing.assert_allclose(q['compression_stored'], .5*80.*opening**2)
        np.testing.assert_allclose(q['stored'], q['compression_stored'])
        work = np.trapezoid(q['traction'][:, 0], opening)
        self.assertAlmostEqual(work, 0., places=14)
        self.assertEqual(law.response(0., 0., 0.)['tangent'][0, 0], 80.)

    def test_compression_is_not_degraded_after_complete_cohesive_failure(self):
        law = replace(self.law, compression_stiffness_mpa_per_mm=80.)
        old = 2*law.final_opening_mm
        q = law.response(-.002, .005, old)
        np.testing.assert_array_equal(q['damageable_traction'], 0.)
        self.assertEqual(q['damageable_stored'], 0.)
        self.assertEqual(q['damage'], 1.)
        self.assertEqual(q['history'], old)
        self.assertEqual(q['dissipation_increment'], 0.)
        self.assertAlmostEqual(q['dissipated'], law.toughness_n_per_mm)
        np.testing.assert_allclose(q['traction'], [-.16, 0.])
        np.testing.assert_allclose(q['tangent'], [[80., 0.], [0., 0.]])
        self.assertAlmostEqual(q['stored'], .5*80.*.002**2)

    def test_compression_and_shear_damage_have_consistent_potential_derivatives(self):
        law = replace(self.law, compression_stiffness_mpa_per_mm=80.)
        for u, old in (([-.002, .007], 0.), ([-.003, -.002], .025),
                       ([.001, .0002], 0.), ([-.002, .005], .08)):
            with self.subTest(jump=u, history=old):
                u = np.asarray(u)
                q = law.response(*u, old)
                baseline = self.law.response(*u, old)
                for name in ('history', 'damage', 'dissipated', 'initiation_ratio'):
                    self.assertEqual(q[name], baseline[name])
                finite = np.empty((2, 2))
                gradient = np.empty(2)
                h = 1e-7
                for j in range(2):
                    shift = np.eye(2)[j]*h
                    plus, minus = law.response(*(u+shift), old), law.response(*(u-shift), old)
                    finite[:, j] = (plus['traction']-minus['traction'])/(2*h)
                    gradient[j] = (plus['incremental_potential']-minus['incremental_potential'])/(2*h)
                np.testing.assert_allclose(q['tangent'], finite, rtol=2e-8, atol=2e-8)
                np.testing.assert_allclose(q['traction'], gradient, rtol=2e-8, atol=2e-10)

    def test_nonproportional_compression_and_damage_work_is_energy_consistent(self):
        law = replace(self.law, compression_stiffness_mpa_per_mm=80.)
        vertices = np.array([[0., 0.], [-.003, 0.], [-.003, .015],
                             [-.001, -.02], [.012, -.01], [-.004, 0.], [0., 0.]])
        path = np.concatenate([np.linspace(a, b, 1501)[1:]
                               for a, b in zip(vertices[:-1], vertices[1:])])
        old = 0.
        last_u = vertices[0]
        last_t = np.zeros(2)
        work = 0.
        for jump in path:
            q = law.response(*jump, old)
            self.assertGreaterEqual(q['history'], old)
            self.assertGreaterEqual(q['dissipation_increment'], 0.)
            work += .5*(q['traction']+last_t)@(jump-last_u)
            old, last_u, last_t = q['history'], jump, q['traction']
        self.assertAlmostEqual(work, q['stored']+q['dissipated'], delta=3e-8)

    def test_compression_parameter_is_explicit_validated_and_defaults_to_zero(self):
        self.assertEqual(self.law.compression_stiffness_mpa_per_mm, 0.)
        self.assertEqual(self.law.to_dict()['compression_stiffness_mpa_per_mm'], 0.)
        for value in (-1., np.nan, np.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.law, compression_stiffness_mpa_per_mm=value)

    def test_bounded_compression_validation_and_linear_default_compatibility(self):
        linear = replace(self.law, compression_stiffness_mpa_per_mm=80.)
        explicit = replace(linear, compression_regularization_mm=None)
        for key, value in linear.response(-.002, .001, .01).items():
            np.testing.assert_array_equal(value, explicit.response(-.002, .001, .01)[key])
        self.assertEqual(linear.to_dict()['compression_model'], 'linear')
        self.assertIsNone(linear.to_dict()['compression_traction_cap_mpa'])
        for value in (0., -1., np.nan, np.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(linear, compression_regularization_mm=value)

    def test_bounded_compression_is_globally_capped_and_stable_near_zero(self):
        law = replace(self.law, compression_stiffness_mpa_per_mm=80., compression_regularization_mm=.001)
        opening = np.r_[-np.logspace(-100, 100, 401), 0., .001]
        q = law.response(opening, 0., 0.)
        self.assertTrue(np.isfinite(q['compression_stored']).all())
        self.assertTrue(np.isfinite(q['compression_traction']).all())
        self.assertTrue(np.all(np.abs(q['compression_traction']) <= .08))
        self.assertTrue(np.all(q['compression_stored'] <= .08*np.abs(np.minimum(opening, 0.))))
        self.assertEqual(law.response(0., 0., 0.)['tangent'][0, 0], 80.)
        tiny = law.response(-1e-12, 0., 0.)
        self.assertGreater(tiny['compression_stored'], 0.)
        np.testing.assert_allclose(tiny['compression_stored'], .5*80.*1e-24, rtol=1e-12, atol=0.)
        np.testing.assert_allclose(tiny['compression_traction'], -80.*1e-12, rtol=1e-12, atol=0.)
        self.assertEqual(law.to_dict()['compression_traction_cap_mpa'], .08)

    def test_bounded_compression_consistent_derivatives_and_unchanged_damage(self):
        law = replace(self.law, compression_stiffness_mpa_per_mm=80., compression_regularization_mm=.001)
        for u, old in (([-.002, .007], 0.), ([-.003, -.002], .025), ([-.002, .005], .08)):
            with self.subTest(jump=u, old=old):
                u = np.asarray(u)
                q, baseline = law.response(*u, old), self.law.response(*u, old)
                for key in ('damage', 'history', 'dissipated', 'dissipation_increment', 'initiation_ratio'):
                    self.assertEqual(q[key], baseline[key])
                finite, gradient = np.empty((2, 2)), np.empty(2)
                h = 1e-7
                for j in range(2):
                    shift = np.eye(2)[j]*h
                    plus, minus = law.response(*(u+shift), old), law.response(*(u-shift), old)
                    finite[:, j] = (plus['traction']-minus['traction'])/(2*h)
                    gradient[j] = (plus['incremental_potential']-minus['incremental_potential'])/(2*h)
                np.testing.assert_allclose(q['tangent'], finite, rtol=2e-8, atol=2e-8)
                np.testing.assert_allclose(q['traction'], gradient, rtol=2e-8, atol=2e-10)

    def test_bounded_compression_reversible_work_and_small_length_limit(self):
        law = replace(self.law, compression_stiffness_mpa_per_mm=80., compression_regularization_mm=.001)
        opening = np.r_[np.linspace(0., -.01, 10001), np.linspace(-.01, 0., 10001)[1:]]
        q = law.response(opening, 0., 0.)
        self.assertAlmostEqual(np.trapezoid(q['compression_traction'], opening), 0., places=15)
        np.testing.assert_array_equal(q['dissipated'], 0.)
        np.testing.assert_array_equal(q['history'], 0.)
        load_work = np.trapezoid(q['compression_traction'][:10001], opening[:10001])
        self.assertAlmostEqual(load_work, q['compression_stored'][10000], delta=1e-11)
        responses = [replace(law, compression_regularization_mm=ell).response(-.2, 0., .08)
                     for ell in (.001, .0001, .00001)]
        for before, after in zip(responses[:-1], responses[1:]):
            self.assertLess(abs(after['compression_traction']), .101*abs(before['compression_traction']))
            self.assertLess(after['compression_stored'], .101*before['compression_stored'])
            self.assertEqual(after['damage'], 1.)
            self.assertEqual(after['dissipation_increment'], 0.)

    def test_bounded_compression_nonproportional_work_including_damage(self):
        law = replace(self.law, compression_stiffness_mpa_per_mm=80., compression_regularization_mm=.001)
        vertices = np.array([[0., 0.], [-.003, 0.], [-.003, .015],
                             [-.001, -.02], [.012, -.01], [-.004, 0.], [0., 0.]])
        path = np.concatenate([np.linspace(a, b, 1501)[1:] for a, b in zip(vertices[:-1], vertices[1:])])
        old, work, previous = 0., 0., vertices[0]
        traction = np.zeros(2)
        for u in path:
            q = law.response(*u, old)
            self.assertGreaterEqual(q['dissipation_increment'], 0.)
            work += .5*(q['traction']+traction)@(u-previous)
            old, previous, traction = q['history'], u, q['traction']
        self.assertAlmostEqual(work, q['stored']+q['dissipated'], delta=3e-8)


class MixedInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.law = MixedModeLaw(shear_strength_mpa=.3)
        self.segments = np.array([[0, 1, 2, 3]])
        self.gaps = np.array([.006])
        self.areas = np.array([2.])
        self.old = np.zeros((1, 2))
        self.reference = np.array([[-.003, 0.], [.003, 0.], [-.003, .1], [.003, .1]])
        self.deformed = np.array([[-.003, 0.], [.015, .002], [-.002, .1], [.02, .105]])

    def response(self, x, old=None, **kwargs):
        return interface_response(x, self.segments, self.gaps, self.areas,
                                  self.old if old is None else old, self.law, **kwargs)

    def test_reference_gap_is_stress_free_with_positive_shear_stiffness(self):
        q = self.response(self.reference)
        np.testing.assert_allclose(q['gradient'], 0., atol=1e-13)
        self.assertEqual(q['stored_energy_n_mm'], 0.)
        self.assertEqual(q['fracture_dissipation_n_mm'], 0.)
        np.testing.assert_allclose(q['normal_direction'], [[[1., 0.], [1., 0.]]])
        shear = self.reference.copy()
        shear[[1, 3], 1] += 1e-5
        q = self.response(shear)
        np.testing.assert_allclose(q['traction'][..., 1],
                                   self.law.shear_stiffness_mpa_per_mm*1e-5, atol=1e-13)

    def test_rigid_motion_objectivity_of_energy_force_and_tangent(self):
        q = self.response(self.deformed)
        for angle in (.73, 2.9):
            with self.subTest(angle=angle):
                rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])
                rotated = self.response(self.deformed@rotation.T+np.array([1., -2.]))
                for key in ('stored_energy_n_mm', 'fracture_dissipation_n_mm',
                            'incremental_potential_n_mm'):
                    self.assertAlmostEqual(q[key], rotated[key], places=12)
                np.testing.assert_allclose(rotated['gradient'].reshape(-1, 2),
                                           q['gradient'].reshape(-1, 2)@rotation.T, atol=1e-12)
                transform = np.kron(np.eye(4), rotation)
                np.testing.assert_allclose(rotated['hessian'].toarray(),
                    transform@q['hessian'].toarray()@transform.T, atol=1e-10)

    def test_exact_geometry_force_and_consistent_tangent(self):
        for old in (self.old, np.full((1, 2), .04)):
            with self.subTest(history=float(old.max())):
                q = self.response(self.deformed, old)
                finite = np.empty((8, 8))
                gradient = np.empty(8)
                h = 1e-7
                for j in range(8):
                    shift = (np.eye(8)[j]*h).reshape(4, 2)
                    plus = self.response(self.deformed+shift, old)
                    minus = self.response(self.deformed-shift, old)
                    finite[:, j] = (plus['gradient']-minus['gradient'])/(2*h)
                    gradient[j] = (plus['incremental_potential_n_mm']-minus['incremental_potential_n_mm'])/(2*h)
                np.testing.assert_allclose(q['hessian'].toarray(), finite, rtol=2e-7, atol=2e-8)
                np.testing.assert_allclose(q['gradient'], gradient, rtol=2e-8, atol=2e-10)
                np.testing.assert_allclose(q['hessian'].toarray(), q['hessian'].toarray().T)
                self.assertGreaterEqual(np.linalg.eigvalsh(q['metric'].toarray()).min(), -1e-12)

    def test_self_equilibrated_force_moment_and_reference_area_scaling(self):
        q = self.response(self.deformed)
        force = q['gradient'].reshape(-1, 2)
        np.testing.assert_allclose(force.sum(axis=0), 0., atol=1e-13)
        moment = np.sum(self.deformed[:, 0]*force[:, 1]-self.deformed[:, 1]*force[:, 0])
        self.assertAlmostEqual(moment, 0., places=13)
        twice = interface_response(self.deformed, self.segments, self.gaps,
                                    2*self.areas, self.old, self.law)
        np.testing.assert_allclose(twice['gradient'], 2*q['gradient'])
        self.assertAlmostEqual(twice['stored_energy_n_mm'], 2*q['stored_energy_n_mm'])
        self.assertAlmostEqual(twice['fracture_dissipation_n_mm'], 2*q['fracture_dissipation_n_mm'])

    def test_shared_node_assembly_and_padding(self):
        x = np.vstack((self.reference, [[-.003, .2], [.003, .2]]))
        segments = np.array([[0, 1, 2, 3], [2, 3, 4, 5]])
        x[[1, 3, 5], 0] += .012
        both = interface_response(x, segments, [.006, .006], [2., 2.],
                                  np.zeros((2, 2)), self.law, matrix_size=14)
        separate = [interface_response(x, segment[None], self.gaps, self.areas,
                                       self.old, self.law, matrix_size=14) for segment in segments]
        np.testing.assert_allclose(both['gradient'], sum(q['gradient'] for q in separate), atol=1e-13)
        np.testing.assert_allclose(both['hessian'].toarray(), sum(q['hessian'].toarray() for q in separate), atol=1e-12)
        np.testing.assert_array_equal(both['gradient'][-2:], [0., 0.])
        self.assertEqual(both['history'].shape, (2, 2))

    def test_shape_validation_and_trial_history_isolation(self):
        old = np.full((1, 2), .005)
        before = old.copy()
        self.response(self.deformed, old, tangent=False)
        np.testing.assert_array_equal(old, before)
        with self.assertRaises(ValueError):
            self.response(self.deformed, np.zeros(2))
        collapsed = self.reference.copy()
        collapsed[2:] = collapsed[:2]
        with self.assertRaises(ValueError):
            self.response(collapsed)
        with self.assertRaises(ValueError):
            self.response(self.reference, matrix_size=3)

    def test_compression_geometric_tangent_and_separate_energy_force_accounting(self):
        self.law = replace(self.law, compression_stiffness_mpa_per_mm=80.)
        x = np.array([[-.003, 0.], [.001, .004], [-.002, .1], [.0015, .106]])
        q = self.response(x)
        self.assertTrue(np.all(q['opening'] < 0.))
        self.assertGreater(q['compression_stored_energy_n_mm'], 0.)
        self.assertGreater(q['fracture_dissipation_n_mm'], 0.)
        self.assertAlmostEqual(q['stored_energy_n_mm'],
            q['compression_stored_energy_n_mm']+q['damageable_stored_energy_n_mm'])
        np.testing.assert_allclose(q['gradient'], q['compression_gradient']+q['damageable_gradient'])
        h = 1e-7
        finite, gradient = np.empty((8, 8)), np.empty(8)
        for j in range(8):
            shift = (np.eye(8)[j]*h).reshape(4, 2)
            plus, minus = self.response(x+shift), self.response(x-shift)
            finite[:, j] = (plus['gradient']-minus['gradient'])/(2*h)
            gradient[j] = (plus['incremental_potential_n_mm']-minus['incremental_potential_n_mm'])/(2*h)
        np.testing.assert_allclose(q['hessian'].toarray(), finite, rtol=2e-7, atol=2e-8)
        np.testing.assert_allclose(q['gradient'], gradient, rtol=2e-8, atol=2e-10)
        rotation = np.array([[np.cos(.7), -np.sin(.7)], [np.sin(.7), np.cos(.7)]])
        rotated = self.response(x@rotation.T+[1., -2.])
        self.assertAlmostEqual(rotated['compression_stored_energy_n_mm'], q['compression_stored_energy_n_mm'], places=12)
        np.testing.assert_allclose(rotated['compression_gradient'].reshape(-1, 2),
                                   q['compression_gradient'].reshape(-1, 2)@rotation.T, atol=2e-12)

    def test_bounded_compression_geometry_tangent_and_far_failed_pair_pressure_cap(self):
        self.law = replace(self.law, compression_stiffness_mpa_per_mm=80., compression_regularization_mm=.0005)
        x = np.array([[-.003, 0.], [.001, .004], [-.002, .1], [.0015, .106]])
        q = self.response(x)
        finite, gradient, h = np.empty((8, 8)), np.empty(8), 1e-7
        for j in range(8):
            shift = (np.eye(8)[j]*h).reshape(4, 2)
            plus, minus = self.response(x+shift), self.response(x-shift)
            finite[:, j] = (plus['gradient']-minus['gradient'])/(2*h)
            gradient[j] = (plus['incremental_potential_n_mm']-minus['incremental_potential_n_mm'])/(2*h)
        np.testing.assert_allclose(q['hessian'].toarray(), finite, rtol=2e-7, atol=2e-8)
        np.testing.assert_allclose(q['gradient'], gradient, rtol=2e-8, atol=2e-10)
        # Fully failed faces have slid past their endpoints. Their real nearest
        # distance exceeds 1 mm even though their paired normal projection < 0.
        far = np.array([[0., 0.], [-.2, 2.], [0., 1.], [-.2, 3.]])
        ell = 6e-7
        law = MixedModeLaw(normal_stiffness_mpa_per_mm=5000., compression_stiffness_mpa_per_mm=5000.,
                           compression_regularization_mm=ell)
        failed = np.full((1, 2), law.final_opening_mm)
        q = interface_response(far, self.segments, np.array([ell]), np.array([1.]), failed, law)
        self.assertGreater(np.hypot(.2, 1.), 1.)
        np.testing.assert_array_equal(q['damage'], 1.)
        self.assertEqual(q['damageable_stored_energy_n_mm'], 0.)
        self.assertEqual(q['fracture_dissipation_increment_n_mm'], 0.)
        self.assertTrue(np.all(np.abs(q['compression_traction']) <= .003))
        self.assertLess(q['compression_stored_energy_n_mm'], .000601)
        linear = interface_response(far, self.segments, np.array([ell]), np.array([1.]), failed,
                                    replace(law, compression_regularization_mm=None))
        self.assertGreater(linear['compression_stored_energy_n_mm'], 100.)


if __name__ == '__main__':
    unittest.main()
