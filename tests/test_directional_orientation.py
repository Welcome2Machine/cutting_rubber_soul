"""Continuous QR search checks, independent of a cutting mesh or local fixture."""
import unittest

import numpy as np

from prototype.directional_fracture import best_orientation, relaxed_response
from prototype.material import response
from prototype.model_config import FEMConfig


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


class ContinuousOrientationTests(unittest.TestCase):
    def setUp(self):
        self.config = FEMConfig()

    def search(self, F, *, normals=None, refine=True, samples=16):
        F = np.asarray(F, float)
        if normals is None:
            normals = np.tile([np.cos(.31), np.sin(.31)], (len(F), 1))
        seed = relaxed_response(F, normals, self.config, tangent=False)['energy']
        cauchy = response(F, self.config, tangent=False)['cauchy']
        return best_orientation(F, cauchy, normals, seed, self.config,
                                orientation_samples=samples, refine=refine)

    def test_dense_angle_sweep_open_closed_shear_and_rotated_loading(self):
        # Include an opening optimum, a closed compressed optimum, singular-
        # stretch endpoints, large shear and nearly isotropic F. The dense
        # sweep evaluates normals directly, without using the analytic target.
        states = np.array([[[1.4, 0.], [0., .8]],
                           [[.5, .45], [0., .8]],
                           [[1.3, 0.], [0., 1.7]],
                           [[1., 2.5], [0., 1.]],
                           [[.08, 0.], [0., 3.]],
                           [[.8, 0.], [0., .8]],
                           [[1.001, 0.], [0., 1.002]]])
        states = states @ rotation(.137).T
        best, selected = self.search(states)
        angles = np.arange(16384)*np.pi/16384
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        for i, F in enumerate(states):
            with self.subTest(state=i):
                dense = relaxed_response(np.broadcast_to(F, (len(angles), 2, 2)),
                                         normals, self.config, tangent=False)['energy'].min()
                self.assertLessEqual(best[i], dense+1e-10)
                self.assertLess(dense-best[i], 2e-6)
        actual = relaxed_response(states, selected, self.config, tangent=False)['energy']
        np.testing.assert_allclose(actual, best, rtol=1e-12, atol=1e-12)

    def test_random_states_never_worsen_seed_or_discrete_search(self):
        rng = np.random.default_rng(721)
        states = []
        for _ in range(80):
            stretches = np.exp(rng.uniform(-2., 1.6, 2))
            states.append(rotation(rng.uniform(-np.pi, np.pi)) @ np.diag(stretches)
                          @ rotation(rng.uniform(-np.pi, np.pi)))
        F = np.asarray(states)
        angles = rng.uniform(-np.pi, np.pi, len(F))
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        F_before, normal_before = F.copy(), normals.copy()
        fine, chosen = self.search(F, normals=normals)
        seed = relaxed_response(F, normals, self.config, tangent=False)['energy']
        self.assertTrue(np.all(fine <= seed))
        for count in (4, 16, 64):
            coarse, _ = self.search(F, normals=normals, refine=False, samples=count)
            self.assertTrue(np.all(fine <= coarse+2e-11))
        np.testing.assert_array_equal(F, F_before)
        np.testing.assert_array_equal(normals, normal_before)
        np.testing.assert_allclose(np.linalg.norm(chosen, axis=1), 1., atol=1e-14)

    def test_spatial_and_reference_rotation_covariance_modulo_normal_sign(self):
        F = np.array([[[1.3, .25], [.1, .9]],
                      [[.7, .1], [0., 1.1]],
                      [[.8, .3], [.05, .9]]])
        normals = np.tile([np.cos(.31), np.sin(.31)], (len(F), 1))
        energy, selected = self.search(F, normals=normals)
        Q = rotation(.613)
        spatial_energy, spatial_normal = self.search(Q @ F, normals=normals)
        np.testing.assert_allclose(spatial_energy, energy, atol=2e-12)
        np.testing.assert_allclose(abs(np.einsum('ei,ei->e', spatial_normal, selected)), 1., atol=2e-12)
        reference_energy, reference_normal = self.search(F @ Q.T, normals=normals @ Q.T)
        np.testing.assert_allclose(reference_energy, energy, atol=2e-12)
        expected = selected @ Q.T
        np.testing.assert_allclose(abs(np.einsum('ei,ei->e', reference_normal, expected)), 1., atol=2e-12)

    def test_normal_sign_and_isotropic_compression_do_not_create_release(self):
        F = np.array([[[1.3, .25], [.1, .9]], .8*np.eye(2), np.eye(2)])
        normals = np.tile([np.cos(.31), np.sin(.31)], (len(F), 1))
        positive, Np = self.search(F, normals=normals)
        negative, Nm = self.search(F, normals=-normals)
        np.testing.assert_allclose(negative, positive, atol=1e-12)
        np.testing.assert_allclose(abs(np.einsum('ei,ei->e', Np, Nm)), 1., atol=1e-12)
        intact = response(F, self.config, tangent=False)['energy']
        np.testing.assert_allclose(positive[1:], intact[1:], atol=1e-12)

    def test_continuous_mode_removes_reference_grid_bias_and_is_opt_in(self):
        F = np.array([[[2., .8], [0., .7]]])
        normals = np.array([[1., 0.]])
        seed = relaxed_response(F, normals, self.config, tangent=False)['energy']
        C = response(F, self.config, tangent=False)['cauchy']
        default = best_orientation(F, C, normals, seed, self.config)
        discrete = best_orientation(F, C, normals, seed, self.config, refine=False)
        np.testing.assert_array_equal(default[0], discrete[0])
        np.testing.assert_array_equal(default[1], discrete[1])
        refined, _ = self.search(F, normals=normals)
        self.assertGreater(default[0][0]-refined[0], 1e-5)
        self.assertAlmostEqual(refined[0], 0., places=12)
        for angle in np.arange(8)*np.pi/29:
            Q = rotation(angle)
            rotated, _ = self.search(F @ Q.T, normals=normals @ Q.T)
            np.testing.assert_allclose(rotated, refined, atol=2e-12)

    def test_rejects_degenerate_inverted_nonfinite_or_invalid_seed(self):
        normals = np.array([[1., 0.]])
        C = np.zeros((1, 3, 3))
        for F in (np.zeros((1, 2, 2)), np.array([np.diag([-1., 1.])]),
                  np.array([np.diag([1., 1e-18])])):
            with self.subTest(F=F.tolist()), self.assertRaises(FloatingPointError):
                best_orientation(F, C, normals, np.zeros(1), self.config, refine=True)
        for F in (np.full((1, 2, 2), np.nan), np.full((1, 2, 2), np.inf)):
            with self.assertRaises(ValueError):
                best_orientation(F, C, normals, np.zeros(1), self.config, refine=True)
        with self.assertRaises(ValueError):
            best_orientation(np.eye(2)[None], C, 2*normals, np.zeros(1), self.config, refine=True)
        with self.assertRaises(ValueError):
            best_orientation(np.eye(2)[None], C, normals, np.array([np.nan]), self.config, refine=True)


if __name__ == '__main__':
    unittest.main()
