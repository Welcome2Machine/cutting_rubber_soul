"""Mixed-mode potential and objective interface regression tests; no FE loading."""
import unittest

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


if __name__ == '__main__':
    unittest.main()
