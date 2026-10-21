"""Physical patch checks for explicitly separate contact diagnostic factors."""
import unittest
import numpy as np
from prototype.contact_controls import ContactControlMixin, projected_surface_slip
from tests.test_ipc_contact_formulation import ContactFormulationTests


class ContactControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ContactFormulationTests.setUpClass()
        cls.fixture = ContactFormulationTests()

    def controlled(self, p, *, ev=.0002, exclude=False, n=None):
        p.__class__ = type("FixtureControl", (ContactControlMixin, type(p)), {})
        p.n = len(p.rest) if n is None else n
        p.element = "p1"; p.enrichment = None
        p._connected_boundary_pairs = {tuple(sorted(map(int, e))) for e in p.cmesh.edges}
        p.start = -.01; p.loading_rate = .2; p.material_dt = .05
        p.enable_contact_controls(velocity_epsilon=ev, exclude_adjacent=exclude)
        return p

    def test_velocity_regularization_keeps_fixed_velocity_force(self):
        p = self.controlled(self.fixture.rigid_contact("improved-max"), n=4)
        force = []
        for dt in (.05, .025, .0125):
            p.material_dt = dt
            x = p.rest.copy(); x[4:, 0] += .0001*dt
            # Freeze geometry and normal load to isolate the smoothing/time law.
            lag = p.lag(p.rest)
            g = p.friction.gradient(lag, p.cmesh, x-p.previous)
            force.append(g)
            self.assertAlmostEqual(p.friction.eps_v, .0002*dt)
        np.testing.assert_allclose(force[1:], np.broadcast_to(force[0], (2, len(force[0]))),
                                   rtol=1e-12, atol=1e-12)
        p.material_dt = 0
        with self.assertRaises(ValueError): p.lag(p.rest)

    def test_initial_preparation_remains_common_and_control_is_opt_in(self):
        p = self.controlled(self.fixture.rigid_contact("improved-max"), n=4)
        p.material_dt = .0003; p._contact_initial_preparation = True
        p.lag(p.rest); self.assertEqual(p.friction.eps_v, 1e-5)
        p._contact_initial_preparation = False
        p.lag(p.rest); self.assertAlmostEqual(p.friction.eps_v, 6e-8)
        q = self.fixture.rigid_contact("improved-max")
        self.assertEqual(q.friction.eps_v, 1e-5)

    def test_adjacent_compression_zero_force_but_full_ccd_retained(self):
        rest = np.array([[0.,0.],[.005,0.],[.01,0.],[.015,0.]])
        p = self.controlled(self.fixture.geometry(rest, [[0,1],[1,2],[2,3]],
                                                  "improved-max"), exclude=True)
        for shift in (.002, .003, .004):
            x = rest.copy(); x[2:,0] -= shift
            self.assertTrue(p.collisions(x).empty())
            e, g, h = self.fixture.barrier_response(p, x)
            self.assertEqual(e, 0.); self.assertEqual(np.linalg.norm(g), 0.)
            self.assertEqual(np.linalg.norm(h), 0.)
        folded = np.array([[0.,0.],[.005,0.],[.001,.001],[.006,.001]])
        c = p.collisions(folded)
        self.assertTrue(c.ev_collisions or c.vv_collisions)
        self.assertGreater(self.fixture.barrier_response(p, folded)[0], 0.)
        crossed = folded.copy(); crossed[2:,1] = -.001
        self.assertLess(p.ipc.compute_collision_free_stepsize(p.cmesh, folded, crossed), 1.)
        # Filtering forces never changes the mesh or native CCD candidate policy.
        np.testing.assert_array_equal(p.cmesh.edges, [[0,1],[1,2],[2,3]])

    def test_blade_terms_untouched_and_fold_derivatives(self):
        p = self.fixture.rigid_contact("improved-max")
        before = self.fixture.barrier_response(p, p.rest)
        self.controlled(p, exclude=True, n=4)
        after = self.fixture.barrier_response(p, p.rest)
        for a, b in zip(before, after): np.testing.assert_array_equal(a, b)
        p = self.controlled(self.fixture.geometry(
            [[0,0],[.005,0],[.01,0],[.015,0]], [[0,1],[1,2],[2,3]],
            "improved-max"), exclude=True)
        x = np.array([[0.,0.],[.005,0.],[.001,.001],[.006,.001]])
        e,g,h = self.fixture.barrier_response(p,x)
        d = np.random.default_rng(45).normal(size=x.shape); d /= np.linalg.norm(d)
        eps = 1e-8
        plus = self.fixture.barrier_response(p,x+eps*d)
        minus = self.fixture.barrier_response(p,x-eps*d)
        self.assertAlmostEqual((plus[0]-minus[0])/(2*eps), g@d.ravel(), delta=2e-6)
        np.testing.assert_allclose((plus[1]-minus[1])/(2*eps), h@d.ravel(), rtol=1e-6, atol=1e-3)

    def test_filtered_friction_fresh_jacobian(self):
        p = self.controlled(self.fixture.rigid_contact("improved-max"), n=4, exclude=True)
        x = p.rest.copy(); x[4:,0] += 3e-6
        f = p.lag(x); delta = x-p.previous
        h = p.friction.hessian(f,p.cmesh,delta).toarray()
        h -= p.friction.force_jacobian(f,p.cmesh,p.rest,x-p.rest,delta,p.barrier,
            p.ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS).toarray()
        d = np.random.default_rng(98).normal(size=x.shape); d /= np.linalg.norm(d)
        eps = 1e-9
        def gradient(y): return p.friction.gradient(p.lag(y),p.cmesh,y-p.previous)
        fd = (gradient(x+eps*d)-gradient(x-eps*d))/(2*eps)
        np.testing.assert_allclose(fd,h@d.ravel(),rtol=2e-6,atol=2e-3)

    def test_adjacent_deletion_is_disqualified_at_feature_switch(self):
        # Connected nodes 1--2 are also an EV/VV correction at the corner.
        # Removing VV alone breaks the native inclusion/exclusion cancellation.
        jumps = {}
        for excluded in (False, True):
            p = self.controlled(self.fixture.geometry(
                [[0,0],[.005,0],[.01,0],[.015,0]],[[0,1],[1,2],[2,3]],
                "improved-max"),exclude=excluded)
            jumps[excluded] = []
            for eps in (1e-6,1e-8,1e-10,1e-12):
                g = []
                for sign in (-1,1):
                    x=np.array([[0.,0.],[.005,0.],[.005+sign*eps,.001],[.01,.001]])
                    g.append(self.fixture.barrier_response(p,x)[1])
                jumps[excluded].append(float(np.linalg.norm(g[1]-g[0])))
        self.assertLess(jumps[False][-1],1e-8)
        self.assertGreater(jumps[True][-1],5.)
        self.assertAlmostEqual(jumps[True][-1],jumps[True][-2],delta=1e-6)
        from tools.run_contact_controls import GROUPS
        self.assertTrue(all(not p["exclude_adjacent"] for p in GROUPS.values()))

    def test_initial_flag_uses_builtin_bool_for_exact_history_json(self):
        import json
        class Host:
            def step(self,*args): return self._contact_initial_preparation
        class Fake(ContactControlMixin,Host): pass
        p = Fake()
        result = p.step(None,np.float64(-.01),0.,0.)
        self.assertIs(result,True)
        self.assertEqual(json.dumps({"initial":result}),'{"initial": true}')

    def test_local_objective_slip_rotation_normal_motion_and_edge_dilation(self):
        old = np.array([[.003,.0015],[0.,0.],[.01,0.]])
        for theta in (.01,.1,1.):
            Q = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            current = old@Q.T+[.02,-.03]
            s, g = projected_surface_slip(current,old)
            self.assertAlmostEqual(s,0.,delta=2e-17)
            np.testing.assert_allclose(g.sum(axis=0),0.,atol=1e-14)
            # No work on a current rigid rotation (force/moment consistency).
            rotation = np.column_stack((-current[:,1], current[:,0]))
            self.assertAlmostEqual(float(np.sum(g*rotation)),0.,delta=2e-17)
        current = old.copy(); current[0,1] += .002
        self.assertAlmostEqual(projected_surface_slip(current,old)[0],0.,delta=1e-17)
        current = old.copy(); current[:,0] *= 1.7
        self.assertAlmostEqual(projected_surface_slip(current,old)[0],0.,delta=1e-17)

    def test_local_objective_sliding_covariance_and_gradient(self):
        old = np.array([[.003,.0015],[0.,0.],[.01,0.]])
        x = old.copy(); x[0,0] += .0007
        s,g = projected_surface_slip(x,old)
        self.assertAlmostEqual(s,.0007)
        Q = np.array([[.6,-.8],[.8,.6]])
        ss, gg = projected_surface_slip(x@Q.T+[.1,.2],old)
        self.assertAlmostEqual(ss,s,delta=1e-16)
        np.testing.assert_allclose(gg,g@Q.T,atol=1e-14)
        d = np.random.default_rng(81).normal(size=(3,2))
        eps = 1e-8
        fd = (projected_surface_slip(x+eps*d,old)[0]-
              projected_surface_slip(x-eps*d,old)[0])/(2*eps)
        self.assertAlmostEqual(fd,float(np.sum(g*d)),delta=1e-8)


if __name__ == "__main__":
    unittest.main()
