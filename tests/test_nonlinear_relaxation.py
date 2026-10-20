"""Physical qualification of an isolated nonlinear flow; no experimental fit."""
import unittest
import numpy as np
from mechanics.finite_strain import Material, response as linear_response
from mechanics.nonlinear_relaxation import response


def material():
    G, K = .14, 23.34333333333331
    nu = (3*K-2*G)/(2*(3*K+G))
    return Material((G,), (2.,), (1.24,), (2.,), nu, 1.7,
                    viscous_flow='isochoric_maxwell')


class NonlinearRelaxationTests(unittest.TestCase):
    def setUp(self):
        self.m = material()
        self.F = np.array([np.eye(2), [[1.3, .6], [-.1, .85]], [[.45, .2], [.1, 1.2]]])
        self.A = np.tile(np.eye(3), (3, 1, 1))

    def test_infinite_scale_recovers_existing_kernel_including_tangent(self):
        A = linear_response(self.F, self.A, .2, self.m, kinematics='plane_strain')['metric']
        a = response(self.F, A, .05, self.m, np.inf)
        b = linear_response(self.F, A, .05, self.m, kinematics='plane_strain')
        for key in ('P', 'metric', 'energy', 'viscous_rate', 'incremental_potential', 'tangent'):
            np.testing.assert_allclose(a[key], b[key], atol=2e-10, rtol=1e-9)

    def test_variational_potential_and_consistent_symmetric_tangent(self):
        A = response(self.F, self.A, .2, self.m, .5)['metric']
        for scale in (.25, 1., 2.):
            q = response(self.F, A, .07, self.m, scale)
            for i in range(4):
                d = np.eye(4)[i].reshape(2, 2)*1e-6
                a = response(self.F+d, A, .07, self.m, scale, tangent=False)
                b = response(self.F-d, A, .07, self.m, scale, tangent=False)
                np.testing.assert_allclose((a['incremental_potential']-b['incremental_potential'])/2e-6,
                    q['P'].reshape(-1, 4)[:, i], atol=3e-7, rtol=2e-6)
                np.testing.assert_allclose(((a['P']-b['P'])/2e-6).reshape(-1, 4),
                    q['tangent'][:, :, i], atol=3e-7, rtol=2e-6)
            np.testing.assert_allclose(q['tangent'], q['tangent'].transpose(0, 2, 1), atol=2e-10)

    def test_objectivity_isochoric_flow_and_no_history_mutation(self):
        q = response(self.F, self.A, .08, self.m, .5)
        angle = .8
        Q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = response(Q@self.F, self.A, .08, self.m, .5)
        np.testing.assert_allclose(rotated['P'], Q@q['P'], atol=2e-10)
        np.testing.assert_allclose(rotated['metric'], q['metric'], atol=2e-11)
        np.testing.assert_allclose(np.linalg.det(q['metric']), 1., atol=2e-11)
        np.testing.assert_array_equal(self.A, np.tile(np.eye(3), (3, 1, 1)))
        self.assertTrue(np.all(q['viscous_rate'] >= 0))
        # Reference-coordinate rotation tests material isotropy, too.
        rotated = response(self.F@Q, self.A, .08, self.m, .5)
        np.testing.assert_allclose(rotated['P'], q['P']@Q, atol=2e-10)

    def test_hold_decreases_energy_and_small_strain_time_is_unchanged(self):
        for scale in (.25, 2.):
            A = self.A.copy(); last = np.full(3, np.inf)
            for _ in range(40):
                q = response(self.F, A, .1, self.m, scale, tangent=False)
                self.assertTrue(np.all(q['energy'] <= last+1e-11))
                self.assertTrue(np.all(q['viscous_rate'] >= 0))
                last = q['energy']; A = q['metric']
        F = np.array([[[1., 1e-6], [0., 1.]]]); A = np.eye(3)[None]
        a = response(F, A, .05, self.m, .25, tangent=False)
        b = linear_response(F, A, .05, self.m, tangent=False, kinematics='plane_strain')
        np.testing.assert_allclose(a['P'], b['P'], atol=1e-12, rtol=1e-8)

    def test_time_refinement_and_rate_effect(self):
        def ramp(n, total):
            A = np.eye(3)[None]
            for i in range(1, n+1):
                F = np.array([[[1., .7*i/n], [0., 1.]]])
                q = response(F, A, total/n, self.m, .5, tangent=False); A = q['metric']
            return q['P'][0, 0, 1]
        a, b, c = [ramp(n, 1.) for n in (20, 40, 80)]
        self.assertLess(abs(c-b), abs(b-a))
        self.assertGreater(ramp(80, .5), ramp(80, 2.))

    def test_invalid_metric_and_time_rejected(self):
        for dt in (0., -1., np.nan):
            with self.assertRaises(ValueError):response(self.F, self.A, dt, self.m, 1.)
        for scale in (0., -1., np.nan):
            with self.assertRaises(ValueError):response(self.F, self.A, .1, self.m, scale)
        bad = self.A.copy(); bad[0, 0, 0] = -1
        with self.assertRaises(ValueError):response(self.F, bad, .1, self.m, 1.)


if __name__ == '__main__':
    unittest.main()
