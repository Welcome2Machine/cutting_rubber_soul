import unittest
import numpy as np
from prototype.phase_field import FixedDisplacementAT1


class PhaseFieldTests(unittest.TestCase):
    def problem(self, width=20., ell=.1):
        X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        return FixedDisplacementAT1(X, [[0, 1, 2], [1, 3, 2]], width, 2.15, ell)

    def test_homogeneous_threshold_and_irreversibility(self):
        p = self.problem(); critical = 3*p.gc/(16*p.ell*(1-p.residual))
        for ratio in (.5, 1., 2., 10.):
            d, info = p.solve(np.full(2, critical*ratio))
            np.testing.assert_allclose(d, max(0., 1-1/ratio), atol=1e-10)
            self.assertLessEqual(info['kkt_residual_mpa'], 1e-6)
            self.assertLessEqual(info['energy_change_n_mm'], 1e-10)
        unloaded, _ = p.solve(np.zeros(2), old=d)
        np.testing.assert_array_equal(unloaded, d)

    def test_energy_gradient_and_hessian_are_consistent(self):
        p = self.problem(); psi = np.array([2., 20.]); A, b = p.system(psi)
        d = np.array([.1, .3, .5, .2]); v = np.array([.2, -.1, .3, -.4]); eps=1e-6
        def E(x):
            release, fracture = p.energy(x, psi)
            return fracture-release
        self.assertAlmostEqual((E(d+eps*v)-E(d-eps*v))/(2*eps), float((A@d-b)@v), places=7)
        self.assertAlmostEqual(E(d), float(.5*d@(A@d)-b@d), places=11)
        np.testing.assert_allclose(A.toarray(), A.toarray().T, atol=1e-12)

    def test_width_scales_energy_but_not_damage(self):
        psi = np.array([2., 25.]); d1, r1 = self.problem(1.).solve(psi)
        d2, r2 = self.problem(20.).solve(psi)
        np.testing.assert_allclose(d1, d2, atol=1e-10)
        self.assertAlmostEqual(r2['energy_change_n_mm'], 20*r1['energy_change_n_mm'], places=10)

    def test_length_changes_unvalidated_at1_threshold(self):
        d1, _ = self.problem(ell=.05).solve(np.full(2, 5.))
        d2, _ = self.problem(ell=.2).solve(np.full(2, 5.))
        self.assertEqual(d1.max(), 0.)
        self.assertGreater(d2.min(), .5)

    def test_invalid_damage_and_energy_rejected(self):
        p = self.problem()
        for psi in ([-1., 2.], [float('nan'), 0.]):
            with self.assertRaises(ValueError): p.solve(psi)
        with self.assertRaises(ValueError): p.solve([2., 2.], old=np.ones(4)*1.1)


if __name__ == '__main__': unittest.main()
