"""History sensitivities are different from fixed-previous-history probes."""
from dataclasses import replace
import unittest
import numpy as np
from mechanics.finite_strain import Material, response
from prototype.history_sensitivity import metric_direction, metric_perturb, common_material, previous_friction_action, PARAMETERS


def metric_fixture():
    Q, _ = np.linalg.qr(np.random.default_rng(732).normal(size=(3, 3)))
    return Q @ np.diag([.3, 2., 1/.6]) @ Q.T


class HistorySensitivityTests(unittest.TestCase):
    def test_common_coordinates_preserve_bulk_and_control_only_named_factor(self):
        ge,gv,k=.14,1.24,23.34333333333331
        m=Material((ge,),(2.,),(gv,),(2.,),(3*k-2*ge)/(2*(3*k+ge)),1.7,viscous_flow='isochoric_maxwell')
        for key in PARAMETERS:
            p=common_material(m,key,.1); g,b=p.moduli(); v,_=p.moduli(True)
            self.assertAlmostEqual(b,k,places=10)
            np.testing.assert_allclose([np.log((g+v)/(ge+gv)),np.log(g/v)-np.log(ge/gv),np.log(p.relaxation_s/1.7)],
                                       [.1 if key==j else 0. for j in PARAMETERS],atol=2e-14)

    def test_analytic_previous_friction_action_smoothing_and_sliding(self):
        from tests.test_ipc_contact_formulation import ContactFormulationTests
        ContactFormulationTests.setUpClass(); fixture=ContactFormulationTests()
        p=fixture.rigid_contact('improved-max')
        for slide in (2e-6, 2e-5):
            x=p.rest.copy(); x[4:,0]+=slide; lag=p.lag(x); delta=x-p.previous
            h=p.friction.hessian(lag,p.cmesh,delta,p.ipc.PSDProjectionMethod.NONE)
            d=np.random.default_rng(913).normal(size=p.rest.size); d/=np.linalg.norm(d)
            analytic=previous_friction_action(p,h,d)
            for eps in (1e-9,3e-10):
                plus=p.friction.gradient(lag,p.cmesh,delta-eps*d.reshape(-1,2))
                minus=p.friction.gradient(lag,p.cmesh,delta+eps*d.reshape(-1,2))
                np.testing.assert_allclose((plus-minus)/(2*eps),analytic,rtol=3e-5,atol=1e-5)
            translation=np.tile([.3,-.2],len(x))
            np.testing.assert_allclose(previous_friction_action(p,h,translation),0.,atol=1e-8)
            batch=previous_friction_action(p,h,np.column_stack((d,translation)))
            np.testing.assert_allclose(batch[:,0],analytic,atol=1e-8)

    def test_rotated_metric_retraction_is_spd_and_preserves_determinant(self):
        A = np.stack([metric_fixture(), 1.2*metric_fixture()])
        D = np.random.default_rng(183).normal(size=A.shape)
        for eps in (-.4, -.01, .01, .4):
            q = metric_perturb(A, D, eps)
            np.testing.assert_allclose(q, q.swapaxes(-2, -1), atol=1e-14)
            self.assertGreater(np.linalg.eigvalsh(q).min(), 0.)
            np.testing.assert_allclose(np.linalg.det(q), np.linalg.det(A), rtol=3e-14)

    def test_zero_direction_returns_exact_independent_copy(self):
        A = metric_fixture()
        for q in (metric_perturb(A, np.zeros_like(A), .1),
                  metric_perturb(A, np.ones_like(A), 0.)):
            np.testing.assert_array_equal(q, A)
            self.assertFalse(np.shares_memory(A, q))
        batch = np.stack([A, A]); directions = np.stack([np.zeros_like(A), np.eye(3)])
        np.testing.assert_array_equal(metric_perturb(batch, directions, .01)[0], A)

    def test_derivative_matches_projected_direction_and_is_coordinate_covariant(self):
        A = metric_fixture(); D = np.random.default_rng(42).normal(size=A.shape)
        D = .5*(D+D.T); inverse = np.linalg.inv(A)
        expected = D-np.trace(inverse@D)*A/3
        for eps in (1e-4, 5e-5):
            p, m = metric_perturb(A, D, eps), metric_perturb(A, D, -eps)
            derivative = metric_direction(A, p, m, eps)
            np.testing.assert_allclose(derivative, expected, rtol=3e-7, atol=3e-8)
            self.assertLess(abs(np.trace(inverse@derivative)), 2e-14)
        Q, _ = np.linalg.qr(np.random.default_rng(52).normal(size=(3, 3)))
        np.testing.assert_allclose(metric_perturb(Q@A@Q.T, Q@D@Q.T, .05),
                                  Q@metric_perturb(A, D, .05)@Q.T, atol=2e-14)

    def test_invalid_history_and_steps_are_rejected(self):
        A = metric_fixture(); D = np.zeros_like(A)
        for bad in (np.zeros((3, 3)), np.diag([1., 1., -1.]), np.eye(2),
                    np.full((3, 3), np.nan), A+np.triu(np.ones((3, 3)), 1)):
            with self.assertRaises(ValueError):metric_perturb(bad, D, .01)
        with self.assertRaises(ValueError):metric_direction(A, A, A, 0.)
        with self.assertRaises(ValueError):metric_perturb(A, D, np.nan)
        with self.assertRaises(ValueError):metric_perturb(A, np.zeros((2, 2)), .01)

    def test_propagated_maxwell_derivative_matches_independent_full_path(self):
        ge, gv, k, tau = .14, 1.24, 23.34333333333331, 1.7
        base = Material((ge,), (2.,), (gv,), (2.,), (3*k-2*ge)/(2*(3*k+ge)),
                        tau, viscous_flow='isochoric_maxwell')
        dt, steps = .05, 50
        for eps in (1e-4, 5e-5):
            plus = replace(base, relaxation_s=tau*np.exp(eps))
            minus = replace(base, relaxation_s=tau*np.exp(-eps))
            A = np.eye(3)[None]; D = np.zeros_like(A)
            full_plus, full_minus = A.copy(), A.copy()
            for i in range(1, steps+1):
                F = np.array([[[1., .2*i*dt], [0., 1.]]])
                q = response(F, A, dt, base, kinematics='plane_strain', tangent=False)
                p = response(F, metric_perturb(A, D, eps), dt, plus,
                             kinematics='plane_strain', tangent=False)
                m = response(F, metric_perturb(A, D, -eps), dt, minus,
                             kinematics='plane_strain', tangent=False)
                fp = response(F, full_plus, dt, plus, kinematics='plane_strain', tangent=False)
                fm = response(F, full_minus, dt, minus, kinematics='plane_strain', tangent=False)
                if i == steps:
                    cp = response(F, A, dt, plus, kinematics='plane_strain', tangent=False)
                    cm = response(F, A, dt, minus, kinematics='plane_strain', tangent=False)
                D = metric_direction(q['metric'], p['metric'], m['metric'], eps)
                A, full_plus, full_minus = q['metric'], fp['metric'], fm['metric']
            propagated = (p['P']-m['P'])/(2*eps)
            independently_integrated = (fp['P']-fm['P'])/(2*eps)
            np.testing.assert_allclose(propagated, independently_integrated, rtol=2e-6, atol=2e-8)
            full_D = metric_direction(A, full_plus, full_minus, eps)
            np.testing.assert_allclose(D, full_D, rtol=2e-6, atol=2e-8)
            # This is a prescribed homogeneous shear path, not FE force data.
            frozen = (cp['P']-cm['P'])/(2*eps)
            self.assertLess(abs(frozen[0, 0, 1]), .06*abs(independently_integrated[0, 0, 1]))


if __name__ == '__main__':
    unittest.main()
