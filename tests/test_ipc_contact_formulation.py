"""Contact geometry checks; no equilibrium solve or experimental validation.

OGC is an unqualified diagnostic geometry construction with CCD safeguards.
These derivatives are checked inside fixed feature/feasibility regions, not at
nonsmooth changes in contact topology. A separate synthetic crossing below
records why fixed-region derivative checks do not qualify OGC for our PTC host.
"""
import unittest
from types import SimpleNamespace
import numpy as np
from prototype.ipc_reference import IPCProblem, toolkit


class ContactFormulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ipc = toolkit()
        except ModuleNotFoundError:
            raise unittest.SkipTest('Optional ipctk==1.6.0 not installed')
        cls.ipc.set_num_threads(1)

    def geometry(self, rest, edges, formulation, n_dynamic=None):
        p = object.__new__(IPCProblem)
        p.ipc = self.ipc
        p.contact_formulation = formulation
        p.rest = np.asarray(rest, dtype=float)
        p.previous = p.rest.copy()
        p.surface = np.arange(len(rest))
        p.surface_dofs = np.arange(p.rest.size)
        p.dhat = .0025
        p.cmesh = self.ipc.CollisionMesh(p.rest, np.asarray(edges, dtype=int))
        if n_dynamic is not None:
            p.cmesh.can_collide = self.ipc.make_static_obstacle_filter(n_dynamic)
        p.barrier = self.ipc.BarrierPotential(p.dhat, 132., True)
        p.friction = self.ipc.FrictionPotential(1e-5)
        p.config = SimpleNamespace(friction_coefficient=.35)
        return p

    def rigid_contact(self, formulation='offset-geometric', reverse_edges=False):
        # Disconnected closed surfaces with stable interior point-edge contact.
        # The upper rectangle is a rigid blade; static-static pairs are disabled.
        rest = np.array([[0., 0.], [.01, 0.], [.01, -.01], [0., -.01],
                         [-.003, .0015], [.013, .0015], [.013, .01], [-.003, .01]])
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4]])
        if reverse_edges:
            edges = edges[::-1, ::-1].copy()
        return self.geometry(rest, edges, formulation, 4)

    def barrier_response(self, p, x):
        c = p.collisions(x)
        return (float(p.barrier(c, p.cmesh, x)),
                p.barrier.gradient(c, p.cmesh, x),
                p.barrier.hessian(c, p.cmesh, x).toarray())

    def test_legacy_default_remains_explicit_and_invalid_type_rejected(self):
        import inspect
        signature = inspect.signature(IPCProblem)
        self.assertEqual(signature.parameters['contact_formulation'].default, 'improved-max')
        with self.assertRaisesRegex(ValueError, '[Cc]ontact formulation'):
            IPCProblem('patch', contact_formulation='drop_all_self_contact')

    def test_stress_free_reference_check_accepts_separated_samples(self):
        top = np.column_stack((np.linspace(0., .015, 4), np.zeros(4)))
        rest = np.vstack((top, [[.015, -.02], [0., -.02]]))
        edges = [[i, (i+1)%len(rest)] for i in range(len(rest))]
        p = self.geometry(rest, edges, 'improved-max', len(rest))
        p.n = len(rest)
        check = p.check_reference_contact()
        self.assertEqual(check['status'], 'passed')
        self.assertEqual(check['body_contact_force_norm_n'], 0.)
        self.assertEqual(check['normal_barrier_energy_n_mm'], 0.)

    def test_stress_free_reference_check_rejects_same_polygon_dense_sampling(self):
        # Same rectangle, only its top-edge sampling changes. There is no
        # geometric overlap, yet adjacent samples inside dhat create forces.
        top = np.column_stack((np.linspace(0., .015, 13), np.zeros(13)))
        rest = np.vstack((top, [[.015, -.02], [0., -.02]]))
        edges = [[i, (i+1)%len(rest)] for i in range(len(rest))]
        p = self.geometry(rest, edges, 'improved-max', len(rest))
        p.n = len(rest)
        self.assertFalse(self.ipc.has_intersections(p.cmesh, rest))
        with self.assertRaisesRegex(ValueError, 'contact-induced prestress'):
            p.check_reference_contact()

    def test_straight_connected_edge_compression_has_no_ogc_contact_force(self):
        rest = np.array([[0., 0.], [.005, 0.], [.01, 0.], [.015, 0.]])
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        compressed = rest.copy()
        compressed[2:, 0] -= .003
        old = self.geometry(rest, edges, 'improved-max')
        new = self.geometry(rest, edges, 'offset-geometric')
        original = old.collisions(compressed)
        self.assertTrue(any({v.vertex0_id, v.vertex1_id} == {1, 2}
                            for v in original.vv_collisions))
        self.assertGreater(self.barrier_response(old, compressed)[0], 0.)
        corrected = new.collisions(compressed)
        self.assertTrue(corrected.empty())
        energy, gradient, hessian = self.barrier_response(new, compressed)
        self.assertEqual(energy, 0.)
        np.testing.assert_array_equal(gradient, np.zeros(rest.size))
        np.testing.assert_array_equal(hessian, np.zeros((rest.size, rest.size)))

    def test_feature_crossing_exposes_ogc_jump_but_reference_is_continuous(self):
        # Hand-constructed closed surfaces; no experimental/numerical fixture.
        # Move the middle lower vertex through the upper corner's projection.
        rest = np.array([[-.01,-.0015], [0.,-.0015], [.01,-.0015],
                         [.01,-.02], [-.01,-.02], [-.01,.002], [0.,0.],
                         [.01,0.], [.01,.02], [-.01,.02]])
        edges = [[i, (i+1)%5] for i in range(5)]
        edges += [[i, 5+(i-4)%5] for i in range(5,10)]
        jumps = {}
        for formulation in ('improved-max', 'offset-geometric'):
            p = self.geometry(rest, edges, formulation, 5)
            jumps[formulation] = []
            for eps in (1e-8, 1e-12):
                responses = []
                for sign in (-1, 1):
                    x = rest.copy(); x[1,0] = sign*eps
                    self.assertFalse(p.ipc.has_intersections(p.cmesh, x))
                    responses.append(self.barrier_response(p, x))
                jumps[formulation].append(np.linalg.norm(responses[1][1]-responses[0][1]))
        self.assertLess(jumps['improved-max'][1], 2e-8)
        self.assertLess(jumps['improved-max'][1], .001*jumps['improved-max'][0])
        self.assertGreater(jumps['offset-geometric'][1], 1.)
        self.assertAlmostEqual(jumps['offset-geometric'][0], jumps['offset-geometric'][1], delta=1e-3)

    def test_nonadjacent_fold_selfcontact_is_retained(self):
        rest = np.array([[0., 0.], [.005, 0.], [.01, 0.], [.015, 0.]])
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        folded = np.array([[0., 0.], [.005, 0.], [.001, .001], [.006, .001]])
        p = self.geometry(rest, edges, 'offset-geometric')
        c = p.collisions(folded)
        pairs = {(e.edge_id, e.vertex_id) for e in c.ev_collisions}
        self.assertTrue({(0, 2), (2, 1)} <= pairs)
        self.assertGreater(self.barrier_response(p, folded)[0], 0.)
        # Collision-free step computation continues to see the actual fold.
        crossed = folded.copy()
        crossed[2:, 1] = -.001
        alpha = self.ipc.compute_collision_free_stepsize(p.cmesh, folded, crossed)
        self.assertGreater(alpha, 0.)
        self.assertLess(alpha, 1.)

    def test_closed_rigid_blade_contact_and_orientation_invariance(self):
        a, b = self.rigid_contact(), self.rigid_contact(reverse_edges=True)
        x = a.rest.copy()
        self.assertFalse(self.ipc.has_intersections(a.cmesh, x))
        ca = a.collisions(x)
        self.assertTrue(ca.ev_collisions)
        ea, ga, ha = self.barrier_response(a, x)
        eb, gb, hb = self.barrier_response(b, x)
        self.assertGreater(ea, 0.)
        self.assertLess(ga.reshape(-1, 2)[4:, 1].sum(), 0.)
        self.assertAlmostEqual(ga.reshape(-1, 2)[4:, 0].sum(), 0., delta=1e-12)
        self.assertAlmostEqual(ea, eb, delta=1e-15)
        np.testing.assert_allclose(ga, gb, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(ha, hb, rtol=1e-13, atol=1e-10)
        crossed = x.copy()
        crossed[4:, 1] -= .003
        self.assertLess(self.ipc.compute_collision_free_stepsize(a.cmesh, x, crossed), 1.)

    def test_contact_energy_gradient_and_hessian_inside_regular_region(self):
        p = self.rigid_contact()
        x = p.rest.copy()
        _, gradient, hessian = self.barrier_response(p, x)
        direction = np.random.default_rng(773).normal(size=x.shape)
        direction /= np.linalg.norm(direction)
        eps = 2e-8
        plus = self.barrier_response(p, x+eps*direction)
        minus = self.barrier_response(p, x-eps*direction)
        self.assertAlmostEqual((plus[0]-minus[0])/(2*eps), gradient@direction.ravel(),
                               delta=1e-7)
        np.testing.assert_allclose((plus[1]-minus[1])/(2*eps), hessian@direction.ravel(),
                                   rtol=2e-7, atol=2e-5)

    def test_lagged_friction_and_fresh_geometry_jacobian(self):
        p = self.rigid_contact()
        x = p.rest.copy()
        x[4:, 0] += 5e-5
        lag = p.lag(x)
        delta = x-p.previous
        gradient = p.friction.gradient(lag, p.cmesh, delta)
        hessian = p.friction.hessian(lag, p.cmesh, delta).toarray()
        self.assertGreater(np.linalg.norm(gradient), 0.)
        self.assertGreaterEqual(float(gradient@delta.ravel()), 0.)
        direction = np.random.default_rng(882).normal(size=x.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-8
        plus, minus = delta+eps*direction, delta-eps*direction
        finite_energy = (p.friction(lag, p.cmesh, plus)-p.friction(lag, p.cmesh, minus))/(2*eps)
        self.assertAlmostEqual(finite_energy, gradient@direction.ravel(), delta=1e-7)
        finite_frozen = (p.friction.gradient(lag, p.cmesh, plus)
                        - p.friction.gradient(lag, p.cmesh, minus))/(2*eps)
        np.testing.assert_allclose(finite_frozen, hessian@direction.ravel(),
                                   rtol=3e-7, atol=2e-5)
        geometry_jacobian = -p.friction.force_jacobian(lag, p.cmesh, p.rest,
            x-p.rest, delta, p.barrier,
            self.ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS).toarray()
        def fresh_gradient(y):
            return p.friction.gradient(p.lag(y), p.cmesh, y-p.previous)
        finite_fresh = (fresh_gradient(x+eps*direction)-fresh_gradient(x-eps*direction))/(2*eps)
        np.testing.assert_allclose(finite_fresh, (hessian+geometry_jacobian)@direction.ravel(),
                                   rtol=5e-7, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
