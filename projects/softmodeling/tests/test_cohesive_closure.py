"""Mechanical regressions for finite-face closure; no cutting-force fitting."""
from dataclasses import replace
import unittest
import numpy as np

from prototype.cohesive_closure import finite_face_geometry, point_segment_distance_jet
from prototype.cohesive_mixed import MixedModeLaw, interface_response


class FiniteFaceClosureTests(unittest.TestCase):
    def setUp(self):
        self.x0 = np.array([[0., 0.], [.006, 0.], [0., .1], [.006, .1]])
        self.segments = np.array([[0, 1, 2, 3]])
        self.reference = finite_face_geometry(self.x0[None])[0]
        self.law = MixedModeLaw(normal_stiffness_mpa_per_mm=80.,
            compression_stiffness_mpa_per_mm=80., compression_regularization_mm=.006,
            compression_kinematics='finite_faces')

    def response(self, x, *, old=None, **kwargs):
        return interface_response(x, self.segments, [.006], [2.],
            np.zeros((1, 2)) if old is None else old, self.law,
            reference_closure_distances=self.reference, **kwargs)

    def test_reference_is_stress_free_and_parallel_closure_matches_original(self):
        q = self.response(self.x0)
        np.testing.assert_allclose(q['gradient'], 0., atol=1e-14)
        self.assertEqual(q['stored_energy_n_mm'], 0.)
        x = self.x0.copy(); x[[1, 3], 0] -= .002
        q = self.response(x)
        legacy = interface_response(x, self.segments, [.006], [2.], np.zeros((1, 2)),
            replace(self.law, compression_kinematics='projected_gap'))
        self.assertAlmostEqual(q['stored_energy_n_mm'], legacy['stored_energy_n_mm'], places=14)
        np.testing.assert_allclose(q['gradient'], legacy['gradient'], atol=1e-14)

    def test_remote_slid_faces_have_zero_force_without_erasing_failure_energy(self):
        x = self.x0.copy(); x[[1, 3]] += [-.106, .2]
        old = np.full((1, 2), 2*self.law.final_opening_mm)
        q = self.response(x, old=old)
        np.testing.assert_array_equal(q['compression_gradient'], 0.)
        np.testing.assert_array_equal(q['gradient'], 0.)
        np.testing.assert_array_equal(q['hessian'].toarray(), 0.)
        self.assertEqual(q['stored_energy_n_mm'], 0.)
        self.assertEqual(q['fracture_dissipation_n_mm'], 2*self.law.toughness_n_per_mm)
        self.assertEqual(q['fracture_dissipation_increment_n_mm'], 0.)
        np.testing.assert_array_equal(q['damage'], 1.)
        legacy = interface_response(x, self.segments, [.006], [2.], old,
            replace(self.law, compression_kinematics='projected_gap'))
        self.assertGreater(np.linalg.norm(legacy['compression_gradient']), .1)

    def test_actual_closure_remains_after_failure_and_has_no_new_damage_loss(self):
        x = self.x0.copy(); x[[1, 3], 0] -= .002
        old = np.full((1, 2), 2*self.law.final_opening_mm)
        q = self.response(x, old=old)
        self.assertGreater(q['compression_stored_energy_n_mm'], 0.)
        self.assertGreater(np.linalg.norm(q['compression_gradient']), 0.)
        self.assertEqual(q['fracture_dissipation_increment_n_mm'], 0.)
        self.assertEqual(q['damageable_stored_energy_n_mm'], 0.)
        np.testing.assert_array_equal(old, 2*self.law.final_opening_mm)

    def test_consistent_gradient_hessian_for_rotating_sliding_and_endpoint_faces(self):
        cases = [np.array([[0., 0.], [.003, .018], [0., .1], [.004, .118]]),
                 np.array([[0., 0.], [.003, .025], [.001, .1], [.005, .125]])]
        for x in cases:
            for old in (np.zeros((1, 2)), np.full((1, 2), .2)):
                with self.subTest(x=x.tolist(), history=old[0, 0]):
                    q = self.response(x, old=old)
                    h = 2e-7; finite = np.zeros((8, 8)); gradient = np.zeros(8)
                    for i in range(8):
                        step = h*np.eye(8)[i].reshape(4, 2)
                        plus, minus = self.response(x+step, old=old), self.response(x-step, old=old)
                        finite[:, i] = (plus['gradient']-minus['gradient'])/(2*h)
                        gradient[i] = (plus['incremental_potential_n_mm']-minus['incremental_potential_n_mm'])/(2*h)
                    np.testing.assert_allclose(q['gradient'], gradient, rtol=2e-6, atol=2e-8)
                    np.testing.assert_allclose(q['hessian'].toarray(), finite, rtol=2e-5, atol=2e-6)

    def test_rigid_motion_force_moment_and_tangent_objectivity(self):
        x = np.array([[0., 0.], [.003, .018], [0., .1], [.004, .118]])
        q = self.response(x)
        angle = .7; r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        qr = self.response(x@r.T+[1., -2.])
        transform = np.kron(np.eye(4), r)
        self.assertAlmostEqual(q['stored_energy_n_mm'], qr['stored_energy_n_mm'], places=12)
        np.testing.assert_allclose(qr['gradient'], transform@q['gradient'], atol=2e-12)
        np.testing.assert_allclose(qr['hessian'].toarray(), transform@q['hessian'].toarray()@transform.T, atol=2e-10)
        force = q['gradient'].reshape(-1, 2)
        np.testing.assert_allclose(force.sum(axis=0), 0., atol=1e-14)
        self.assertAlmostEqual(np.sum(x[:, 0]*force[:, 1]-x[:, 1]*force[:, 0]), 0., places=13)

    def test_point_edge_and_endpoint_distance_derivatives_and_feature_continuity(self):
        for point in ([-.05, .3], [.4, .3], [1.05, .3]):
            z = np.array([point, [0., 0.], [1., 0.]])
            d, j, h = point_segment_distance_jet(z[None, 0], z[None, 1], z[None, 2])
            for i in range(6):
                shift = 1e-6*np.eye(6)[i].reshape(3, 2)
                plus, minus = z+shift, z-shift
                dp, jp, _ = point_segment_distance_jet(plus[None, 0], plus[None, 1], plus[None, 2])
                dm, jm, _ = point_segment_distance_jet(minus[None, 0], minus[None, 1], minus[None, 2])
                np.testing.assert_allclose(j[:, i], (dp-dm)/2e-6, atol=1e-9)
                np.testing.assert_allclose(h[:, :, i], (jp-jm)/2e-6, atol=1e-9)
        a, b = np.array([[0., 0.]]), np.array([[1., 0.]])
        _, left, _ = point_segment_distance_jet(np.array([[-1e-9, .3]]), a, b)
        _, right, _ = point_segment_distance_jet(np.array([[1e-9, .3]]), a, b)
        np.testing.assert_allclose(left, right, atol=1e-8)

    def test_reversible_slide_out_work_converges_to_potential_change(self):
        from prototype.cohesive_closure import finite_face_closure
        # Fully failed interface, prescribed finite sliding: no FE equilibrium claim.
        start = self.x0.copy(); start[[1, 3], 0] -= .002
        end = start.copy(); end[[1, 3], 1] += .14
        errors = []
        for n in (128, 256, 512):
            x = np.linspace(start, end, n+1)
            q = finite_face_closure(x, np.broadcast_to(self.reference, (n+1, 2, 2)), self.law)
            work = np.sum(.5*(q['gradient'][1:]+q['gradient'][:-1])*np.diff(x.reshape(n+1, 8), axis=0))
            energy = q['stored'].mean(axis=(1, 2))
            errors.append(abs(work-(energy[-1]-energy[0])))
        self.assertLess(errors[-1], 2e-7)
        self.assertLess(errors[-1], errors[0])

    def test_explicit_reference_and_mode_validation(self):
        with self.assertRaises(ValueError):
            interface_response(self.x0, self.segments, [.006], [2.], np.zeros((1, 2)), self.law)
        for kwargs in ({'compression_kinematics': 'unknown'}, {'compression_regularization_mm': None}):
            with self.assertRaises(ValueError):
                replace(self.law, **kwargs)


if __name__ == '__main__':
    unittest.main()
