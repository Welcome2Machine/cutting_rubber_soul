import unittest
import numpy as np
from mechanics.finite_strain import Material, response
from mechanics.finite_strain_3d import response_3d


def material():
    g, k = .14, 23.34333333333331
    return Material((g,), (2.,), (1.24,), (2.,), poisson=(3*k-2*g)/(2*(3*k+g)),
                    relaxation_s=1.7, viscous_flow='isochoric_maxwell')


class ThreeDimensionalMaterialTests(unittest.TestCase):
    def test_plane_strain_equivalence_and_ps_condensation(self):
        F = np.array([[[1., .4], [.1, .75]], [[.7, 0.], [0., 1.1]], np.eye(2)])
        A = np.tile(np.eye(3), (3, 1, 1))
        old = response(F, A, .05, material(), kinematics='plane_strain')['metric']
        for kin in ('plane_strain', 'plane_stress'):
            q = response(F, old, .05, material(), kinematics=kin)
            F3 = np.tile(np.eye(3), (3, 1, 1)); F3[:, :2, :2] = F; F3[:, 2, 2] = q['lambda_out']
            t = response_3d(F3, old, .05, material())
            np.testing.assert_allclose(t['P'][:, :2, :2], q['P'], atol=2e-9)
            np.testing.assert_allclose(t['metric'], q['metric'], atol=2e-10)
            idx = [0, 1, 3, 4]; C = t['tangent']; reduced = C[:, idx][:, :, idx]
            if kin == 'plane_stress':
                reduced -= C[:, idx, 8, None]*C[:, None, 8, idx]/C[:, 8, 8, None, None]
            np.testing.assert_allclose(reduced, q['tangent'], atol=3e-8)

    def test_full_tangent_potential_objectivity_and_history(self):
        F = np.array([[[1.1, .2, -.05], [.1, .8, .2], [.12, .03, .95]], np.eye(3), np.eye(3)*.8])
        A = np.tile(np.eye(3), (3, 1, 1)); m = material()
        q = response_3d(F, A, .05, m)
        A = q['metric']; q = response_3d(F, A, .05, m)
        D = np.random.default_rng(42).normal(size=F.shape)
        eps = 1e-6
        plus = response_3d(F+eps*D, A, .05, m, tangent=False)
        minus = response_3d(F-eps*D, A, .05, m, tangent=False)
        pred = np.einsum('nij,nj->ni', q['tangent'], D.reshape(-1, 9))
        np.testing.assert_allclose((plus['P']-minus['P']).reshape(-1, 9)/(2*eps), pred, rtol=2e-7, atol=1e-7)
        np.testing.assert_allclose((plus['incremental_potential']-minus['incremental_potential'])/(2*eps),
                                  np.einsum('nij,nij->n', q['P'], D), rtol=2e-7, atol=1e-8)
        np.testing.assert_allclose(q['tangent'], q['tangent'].transpose(0, 2, 1), atol=1e-9)
        R, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
        rotated = response_3d(R@F, A, .05, m)
        np.testing.assert_allclose(rotated['P'], R@q['P'], atol=2e-10)
        np.testing.assert_allclose(rotated['metric'], q['metric'], atol=2e-10)
        np.testing.assert_allclose(np.linalg.det(q['metric']), 1., atol=1e-11)
        self.assertGreater(np.linalg.eigvalsh(q['metric']).min(), 0.)
        self.assertGreaterEqual(q['viscous_rate'].min(), -1e-12)

    def test_small_strain_maxwell_limit(self):
        F = np.eye(3)[None]; F[0, 2, 0] = 1e-6
        A = np.eye(3)[None]
        for i in range(1, 8):
            q = response_3d(F, A, .05, material(), tangent=False); A = q['metric']
            expected = 1e-6*(.14+1.24/(1+.05/1.7)**i)
            self.assertAlmostEqual(q['P'][0, 2, 0], expected, delta=2e-14)


if __name__ == '__main__':
    unittest.main()
