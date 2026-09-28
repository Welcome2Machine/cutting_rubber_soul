import contextlib
import io
import time
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.ipc_fracture import CoupledAT1Problem
from prototype.ipc_reference import IPCProblem, toolkit
from prototype.phase_field import FixedDisplacementAT1


class CouplingHistoryTests(unittest.TestCase):
    def test_uncoupled_mixed_pressure_cannot_be_silently_used_with_damage(self):
        with self.assertRaisesRegex(ValueError, 'mixed damage/pressure'):
            CoupledAT1Problem(volumetric_discretization='macro_mixed')

    def problem(self):
        p = object.__new__(CoupledAT1Problem)
        p.rest = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        p.n = 4
        p.phase = FixedDisplacementAT1(p.rest, [[0, 1, 2], [1, 3, 2]], 20., 2.15, .1)
        p.previous = p.rest.copy(); p.damage = np.zeros(4)
        p.elastic_density = lambda x: np.full(2, 10.+x[0, 0])
        p.acceptance = lambda *a: dict(residual_n=0., reaction_fz_n=1., energy_n_mm=2.)
        return p

    def test_history_fixed_and_kkt_checked_after_mechanics(self):
        p = self.problem(); previous = p.rest+.01; old = np.full(4, .05)
        calls = []
        def mechanics(x, depth, deadline):
            np.testing.assert_array_equal(p.previous, previous)
            calls.append(p.damage.copy())
            x = x.copy(); x[0, 0] = 2. if len(calls) == 1 else 1.
            return x, 0
        p.solve_friction = mechanics
        with contextlib.redirect_stdout(io.StringIO()):
            x, d, r = p.recouple(p.rest, 1., previous, old, deadline=time.monotonic()+10)
        self.assertEqual(len(calls), 3)
        # Damage can decrease within one staggered solve, but never below the
        # previous PHYSICAL load's damage. Locking every iteration is incorrect.
        self.assertLess(d.max(), calls[1].max())
        self.assertTrue(np.all(d >= old))
        self.assertLessEqual(r['phase_kkt_residual_mpa'], 1e-6)
        np.testing.assert_array_equal(p.previous, previous)

    def test_failed_outer_iteration_does_not_commit_damage_or_slip(self):
        p = self.problem(); p.damage[:] = .1
        saved_damage = p.damage.copy(); saved_previous = p.previous.copy()
        p.solve_friction = lambda x, *a: (x+1., 0)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'iteration budget'):
            p.recouple(p.rest, 1., p.rest+.2, np.zeros(4), deadline=time.monotonic()+10, max_outer=1)
        np.testing.assert_array_equal(p.damage, saved_damage)
        np.testing.assert_array_equal(p.previous, saved_previous)
        self.assertTrue(hasattr(p, 'failed_damage'))


class NativeCouplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: toolkit()
        except ModuleNotFoundError: raise unittest.SkipTest('Local native IPC toolkit unavailable')

    def test_zero_damage_is_intact_and_coupled_energy_derivatives(self):
        with threadpool_limits(limits=2):
            p = CoupledAT1Problem(case='patch', mu=0., local_spacing=.2, surface_spacing=.1)
            p.ipc.set_num_threads(2)
            x = p.rest.copy()
            x[:p.n, 1] += .04*p.mesh.reference_xy[:, 1]
            # Blade moved away for an independent bulk derivative test.
            x[p.n:, 1] += 1.
            a = IPCProblem.bulk_terms(p, x, True); b = p.bulk_terms(x, True)
            for index in (0, 1, 2): np.testing.assert_array_equal(a[index], b[index])
            self.assertEqual(a[3:], b[3:])
            rng = np.random.default_rng(23)
            p.damage = rng.uniform(.1, .5, p.n)
            v = np.zeros_like(x); v[:p.n] = rng.normal(size=(p.n, 2))*.01
            eps = 1e-5
            E, g, H, _ = p.evaluate(x, tangent=True, projected=False)
            Ep, gp, _, _ = p.evaluate(x+eps*v, tangent=False)
            Em, gm, _, _ = p.evaluate(x-eps*v, tangent=False)
            self.assertAlmostEqual((Ep-Em)/(2*eps), g@v.ravel(), places=7)
            np.testing.assert_allclose((gp-gm)/(2*eps), H@v.ravel(), rtol=2e-6, atol=2e-7)
            direction = rng.normal(size=p.n)*.1
            original = p.damage.copy(); psi = p.elastic_density(x)
            def total(d):
                p.damage = d
                return p.evaluate(x, tangent=False)[0]+p.phase.energy(d, psi)[1]
            derivative = (total(original+eps*direction)-total(original-eps*direction))/(2*eps)
            A, b = p.phase.system(psi)
            self.assertAlmostEqual(derivative, (A@original-b)@direction, places=7)
            p.damage = original
            self.assertLess(E, IPCProblem.bulk_terms(p, x, False)[3])

    def test_patch_joint_acceptance(self):
        with threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
            p = CoupledAT1Problem(case='patch', mu=.35, local_spacing=.2, surface_spacing=.1)
            p.ipc.set_num_threads(2)
            x, intact = p.step(p.rest.copy(), 0., .05, time.monotonic()+30)
            previous = p.previous.copy()
            final, d, balance = p.recouple(x, .05, previous, np.zeros(p.n), deadline=time.monotonic()+30)
            self.assertLessEqual(balance['residual_n'], 1e-4)
            self.assertLessEqual(balance['phase_kkt_residual_mpa'], 1e-6)
            np.testing.assert_array_equal(d, 0.)
            np.testing.assert_array_equal(final, x)
            np.testing.assert_array_equal(p.previous, previous)
            # Native parallel reductions may differ by a few floating-point ulps
            # even with identical positions/damage; this is far below force tolerance.
            self.assertAlmostEqual(balance['reaction_fz_n'], intact['reaction_fz_n'], places=12)


if __name__ == '__main__': unittest.main()
