"""Macro mixed diagnostic: finite-strain derivatives, stability and IPC wiring.

These checks qualify implementation properties, not the loaded cutting path.
"""
import contextlib
import io
import time
import unittest
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits
from prototype.ipc_reference import IPCProblem, parse_args
from prototype.macro_element import refine_macro_mesh
from prototype.material import bulk, NodalVolume, response
from prototype.model_config import FEMMesh, FEMConfig


def patch_mesh(n):
    xy = np.array([(i/n, j/n) for j in range(n+1) for i in range(n+1)])
    # Distortion vanishes on the boundary; identical recipe on both refinements.
    bump = np.sin(np.pi*xy[:, 0])*np.sin(np.pi*xy[:, 1])
    xy = xy + bump[:, None]*[.06, -.04]
    tri = []
    for j in range(n):
        for i in range(n):
            a = j*(n+1)+i
            tri.extend([[a, a+1, a+n+2], [a, a+n+2, a+n+1]])
    base = FEMMesh(xy, xy, np.array(tri), np.empty((0, 2), int),
                   np.empty(0), np.empty(0, int), np.empty(0, int), np.zeros(len(xy), int))
    return refine_macro_mesh(base, np.empty((0, 2), int))[0]


def sparse_local(k, dofs, size):
    return coo_matrix((k.ravel(), (np.broadcast_to(dofs[:, :, None], k.shape).ravel(),
        np.broadcast_to(dofs[:, None, :], k.shape).ravel())), shape=(size, size)).tocsc()


class IPCMacroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = threadpool_limits(2)

    @classmethod
    def tearDownClass(cls):
        cls.pool.restore_original_limits()

    def problem(self, element='p1-macro-mixed', stabilization=0.):
        p = IPCProblem('patch', mu=0., dhat=.0025, element=element,
            macro_stabilization_shear=stabilization,
            local_spacing=.2, surface_spacing=.05, fine_halfwidth=.3,
            max_iterations=100, hessian_mode='regularized')
        p.ipc.set_num_threads(2)
        return p

    def test_finite_deformation_energy_gradient_tangent_and_rigid_rotation(self):
        mesh = patch_mesh(4)
        cfg = FEMConfig(width_mm=1., shear_modulus_n_mm2=.47, poisson_ratio=.49)
        nv = NodalVolume(mesh, cfg)
        xy = mesh.reference_xy
        u = (xy@np.array([[.35, .2], [-.15, -.25]])
             + .025*np.column_stack((np.sin(3*xy[:, 0])*xy[:, 1], xy[:, 0]**2))).ravel()
        def evaluate(v):
            r, k, dofs, energy, mj = bulk(mesh, v, cfg)
            r, k, extra, energy = nv.correct(v, r, k, energy)
            g = np.zeros(mesh.dof_count)
            np.add.at(g, dofs.ravel(), r.ravel())
            return energy, g, sparse_local(k, dofs, mesh.dof_count)+extra, mj
        E, g, H, mj = evaluate(u)
        self.assertGreater(mj, .5)
        v = np.random.default_rng(415).normal(size=len(u)); v /= np.linalg.norm(v)
        h = 2e-6
        A, B = evaluate(u+h*v), evaluate(u-h*v)
        self.assertAlmostEqual((A[0]-B[0])/(2*h), g@v, delta=1e-8)
        np.testing.assert_allclose((A[1]-B[1])/(2*h), H@v, atol=2e-7, rtol=2e-6)
        theta = .8
        Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated = ((xy+u.reshape(-1, 2))@Q.T-xy).ravel()
        Er, gr, Hr, _ = evaluate(rotated)
        self.assertAlmostEqual(E, Er, delta=1e-11)
        np.testing.assert_allclose(gr.reshape(-1, 2), g.reshape(-1, 2)@Q.T, atol=1e-11)

    def test_distorted_pressure_space_at_finite_compression_and_shear(self):
        minima = []
        F = np.array([[1.25, .5], [.08, .7]])
        cof = np.linalg.det(F)*np.linalg.inv(F).T
        for n in (3, 6):
            mesh = patch_mesh(n)
            area, grad, dofs = mesh.bulk_data
            nv = NodalVolume(mesh, FEMConfig(width_mm=1.))
            dj = np.einsum('iJ,eaJ->eai', cof, grad).reshape(-1, 6)
            D = coo_matrix((dj.ravel(), (np.repeat(np.arange(len(area)), 6), dofs.ravel())),
                          shape=(len(area), mesh.dof_count)).tocsr()
            B = nv.incidence@D
            local = np.einsum('e,eai,ebi,jk->eajbk', area, grad, grad, np.eye(2)).reshape(-1, 6, 6)
            A = sparse_local(local, dofs, mesh.dof_count)
            xy = mesh.reference_xy
            inner = np.flatnonzero(np.all((xy > 1e-10)&(xy < 1-1e-10), axis=1))
            free = (2*inner[:, None]+np.arange(2)).ravel()
            C = (diags(1/np.sqrt(nv.nodal_volumes))@B)[:, free].toarray()
            ev = np.linalg.eigvalsh(C@splu(A[free][:, free]).solve(C.T))
            self.assertEqual(np.count_nonzero(ev < 1e-10), 1)
            minima.append(ev[1])
        self.assertGreater(min(minima), .01)
        self.assertGreater(minima[1]/minima[0], .75)

    def test_ipc_metric_is_psd_and_paired_exact_tangent_is_unchanged(self):
        p = self.problem()
        x = p.rest.copy()
        xy = p.mesh.reference_xy
        x[:p.n] += np.column_stack((.04*xy[:, 0]*xy[:, 1], -.15*xy[:, 1]))
        E, g, H, mj, M = p.evaluate(x, projected=False, with_metric=True)
        q = p.evaluate(x, projected=False)
        self.assertEqual(E, q[0]); np.testing.assert_array_equal(g, q[1])
        self.assertLess(np.max(abs((H-q[2]).data), initial=0.), 1e-12)
        ev = np.linalg.eigvalsh(M[:2*p.n, :2*p.n].toarray())
        self.assertGreater(ev.min(), -1e-9*ev.max())
        v = np.zeros(p.total); v[:2*p.n] = np.random.default_rng(17).normal(size=2*p.n)
        v /= np.linalg.norm(v); v = v.reshape(-1, 2)
        h = 1e-6
        a = p.evaluate(x+h*v, tangent=False); b = p.evaluate(x-h*v, tangent=False)
        np.testing.assert_allclose((a[1]-b[1])/(2*h), H@v.ravel(), rtol=2e-5, atol=3e-6)

    def test_full_control_mesh_contact_and_homogeneous_law_are_identical(self):
        a, b = self.problem(), self.problem('p1-macro-full')
        np.testing.assert_array_equal(a.rest, b.rest)
        np.testing.assert_array_equal(a.mesh.triangles, b.mesh.triangles)
        np.testing.assert_array_equal(a.surface, b.surface)
        self.assertEqual(a.reference_contact_check['body_contact_force_norm_n'], 0.)
        x = a.rest.copy()
        x[:a.n] += a.mesh.reference_xy@np.array([[.1, .2], [.02, -.1]])
        ea, ga, _, _ = a.evaluate(x, tangent=False)
        eb, gb, _, _ = b.evaluate(x, tangent=False)
        self.assertAlmostEqual(ea, eb, delta=1e-10)
        np.testing.assert_allclose(ga, gb, atol=1e-9)
        self.assertIn('macro_parents', a.checkpoint_data())

    def test_pressure_incidence_matches_exact_red_child_integrals(self):
        # Independent analytic weights, rather than the implementation's
        # barycentric coordinate calculation, on a distorted parent mesh.
        mesh = patch_mesh(3)
        nv = NodalVolume(mesh, FEMConfig(width_mm=20.))
        shape = np.array([[4, 1, 1], [1, 4, 1], [1, 1, 4], [2, 2, 2]])/6
        for parent, nodes in enumerate(mesh.macro_triangles):
            children = np.flatnonzero(mesh.macro_parents == parent)
            weights = nv.incidence[nodes][:, children].toarray()
            np.testing.assert_allclose(weights, (nv.volumes[children, None]*shape).T,
                                       atol=1e-13, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(nv.incidence.sum(axis=0)).ravel(),
                                   nv.volumes, atol=1e-13, rtol=1e-12)

    def test_coarse_pressure_does_not_certify_subtriangle_volume(self):
        # This is an algebraic volume-field mode, not a compatible displacement
        # or a zero-total-energy mode. It guards the interpretation of Jbar.
        mesh = patch_mesh(3)
        nv = NodalVolume(mesh, FEMConfig(width_mm=1.))
        children = np.flatnonzero(mesh.macro_parents == 4)
        J = np.ones(len(mesh.triangles))
        J[children] += .08*np.array([1., 1., 1., -3.])
        np.testing.assert_allclose(nv.S@J, 1., atol=1e-13)
        np.testing.assert_allclose(nv.pressure(J), 0., atol=1e-11)
        self.assertAlmostEqual(J.min(), .76)
        full_volume_energy = .5*nv.K*(nv.volumes@((J-1)**2))
        self.assertGreater(full_volume_energy, 0.)

    def test_pressure_block_directions_equal_condensed_for_each_ptc_shift(self):
        from scipy.sparse.linalg import spsolve
        from prototype.macro_element import solve_pressure_blocks
        p = self.problem()
        x = p.rest.copy()
        xy = p.mesh.reference_xy
        x[:p.n] += np.column_stack((.03*xy[:, 0]*xy[:, 1], -.12*xy[:, 1]))
        E, g, H, mj, M = p.evaluate(x, projected=False, with_metric=True)
        e, q, L, j, ML, (B, mass) = p.evaluate(x, projected=False,
            with_metric=True, with_pressure_blocks=True)
        self.assertEqual(E, e); np.testing.assert_array_equal(g, q)
        C = B.T@diags(1/mass)@B
        self.assertLess(np.max(abs((H-L-C).data), initial=0.), 1e-10)
        self.assertLess(np.max(abs((M-ML-C).data), initial=0.), 1e-10)
        free = p.free
        for shift in (0., .1, 10.):
            du, info = solve_pressure_blocks(L[free][:, free], ML[free][:, free],
                (B[:, free], mass), shift, -g[free])
            expected = spsolve((H+shift*M)[free][:, free], -g[free])
            np.testing.assert_allclose(du, expected, rtol=1e-9, atol=1e-11)
            self.assertLessEqual(info['condensed_residual_n'], info['residual_limit_n'])

    def test_stabilized_energy_tangent_and_auxiliary_pressure_scaling(self):
        from scipy.sparse.linalg import spsolve
        from prototype.macro_element import solve_pressure_blocks
        cfg = FEMConfig()
        for stabilization in (1., 4., cfg.bulk_modulus_n_mm2/cfg.shear_modulus_n_mm2):
            with self.subTest(stabilization=stabilization):
                p = self.problem(stabilization=stabilization)
                x = p.rest.copy(); xy = p.mesh.reference_xy
                x[:p.n] += np.column_stack((.03*xy[:, 0]*xy[:, 1], -.12*xy[:, 1]))
                E, g, H, _, M = p.evaluate(x, projected=False, with_metric=True)
                e, q, L, _, ML, (B, mass) = p.evaluate(x, projected=False,
                    with_metric=True, with_pressure_blocks=True)
                self.assertEqual(E, e); np.testing.assert_array_equal(g, q)
                C = B.T@diags(1/mass)@B
                self.assertLess(np.max(abs((H-L-C).data), initial=0.), 1e-10)
                self.assertLess(np.max(abs((M-ML-C).data), initial=0.), 1e-10)
                v = np.zeros(p.total)
                v[p.free] = np.random.default_rng(172).normal(size=len(p.free))
                v /= np.linalg.norm(v); v = v.reshape(-1, 2)
                h = 2e-6
                a = p.evaluate(x+h*v, tangent=False); b = p.evaluate(x-h*v, tangent=False)
                self.assertAlmostEqual((a[0]-b[0])/(2*h), g@v.ravel(), delta=2e-7)
                np.testing.assert_allclose((a[1]-b[1])/(2*h), H@v.ravel(), rtol=3e-5, atol=3e-6)
                for shift in (0., .1, 10.):
                    free=p.free
                    du, info = solve_pressure_blocks(L[free][:,free],ML[free][:,free],
                        (B[:,free],mass),shift,-g[free])
                    expected = spsolve((H+shift*M)[free][:,free],-g[free])
                    np.testing.assert_allclose(du,expected,rtol=1e-9,atol=1e-11)

    def test_stabilization_preserves_homogeneous_law_and_full_integration_limit(self):
        a, full = self.problem(stabilization=4.), self.problem('p1-macro-full')
        x = a.rest.copy()
        x[:a.n] += a.mesh.reference_xy@np.array([[.1,.2],[.02,-.1]])
        ea,ga,_,_ = a.evaluate(x,tangent=False)
        ef,gf,_,_ = full.evaluate(x,tangent=False)
        self.assertAlmostEqual(ea,ef,delta=1e-10)
        np.testing.assert_allclose(ga,gf,atol=1e-9)
        limit = a.config.bulk_modulus_n_mm2/a.config.shear_modulus_n_mm2
        a.nodal_volume = NodalVolume(a.mesh,a.config,stabilization_shear=limit)
        x[:a.n,0] += .03*a.mesh.reference_xy[:,1]**2
        ea,ga,Ha,_ = a.evaluate(x,projected=False)
        ef,gf,Hf,_ = full.evaluate(x,projected=False)
        self.assertAlmostEqual(ea,ef,delta=1e-10)
        np.testing.assert_allclose(ga,gf,atol=1e-9)
        self.assertLess(np.max(abs((Ha-Hf).data),initial=0),1e-9)
        for bad in (-1., float('nan'), limit+1.):
            with self.assertRaises(ValueError): self.problem(stabilization=bad)
        with self.assertRaises(ValueError): self.problem('p1',stabilization=1.)

    def test_confined_finite_compression_ipc_reaction(self):
        p = self.problem()
        x, old = p.rest.copy(), p.start
        with contextlib.redirect_stdout(io.StringIO()):
            for depth in (.05, .1):
                x, d = p.step(x, old, depth, time.monotonic()+30)
                old = depth
                p.acceptance(x, depth, 0, 0)
                # Barrier leaves a small positive gap; compare the homogeneous
                # solution at the actual mean compression, not blade depth.
                J = np.linalg.det(np.eye(2)+np.einsum('eai,eaJ->eiJ',
                    (x[:p.n]-p.mesh.reference_xy)[p.mesh.triangles], p.grad))
                mean_j = np.average(J, weights=p.mesh.bulk_data[0])
                expected = -response(np.array([[[1., 0.], [0., mean_j]]]),
                                     p.config, tangent=False)['P'][0, 1, 1]*p.config.width_mm
                # Discrete barrier endpoint weights make the local field
                # nonuniform; the analytic comparison concerns total reaction.
                self.assertGreater(J.min(), 0.)
                self.assertAlmostEqual(d['reaction_fz_n'], expected, delta=2e-3)

    def test_macro_cli_is_explicit_and_visualization_is_guarded(self):
        args = parse_args(['--current', '--output', 'unused', '--element', 'p1-macro-mixed', '--no-visualize'])
        self.assertEqual(args.element, 'p1-macro-mixed')
        self.assertEqual(parse_args(['--current', '--output', 'unused']).element, 'p1')
        with self.assertRaisesRegex(ValueError, 'reused-lu is not implemented'):
            IPCProblem('patch', element='p1-macro-mixed', linear_solver='reused-lu')


if __name__ == '__main__':
    unittest.main()
