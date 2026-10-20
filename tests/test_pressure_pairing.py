"""Nonlinear, mixed-block and projection tests for the bounded pairing study."""
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.ipc_p2 import InteriorP2
from prototype.pressure_pairing import P2P1Volume
from prototype.model_config import FEMConfig
from tests.test_ipc_p2 import rectangle, assembled


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.pool = threadpool_limits(2)
        self.el = InteriorP2(rectangle(.25), FEMConfig(width_mm=1., shear_modulus_n_mm2=.47))

    def tearDown(self):
        self.pool.restore_original_limits()

    def evaluate(self, v, mixed, tangent=True):
        r, k, d, e, j = self.el.bulk(v, with_tangent=tangent)
        r, k, B, e = mixed.correct(v, r, k, e)
        g, A, _, _ = assembled((r, k, d, e, j), len(v))
        H = None if B is None else A.toarray()+B.T@mixed._mass_factor.solve(B.toarray())
        return e, g, H, A, B

    def test_nonlinear_derivatives_objectivity_and_block_elimination(self):
        xy = self.el.mesh.reference_xy
        rng = np.random.default_rng(845)
        u = (xy@np.array([[.15, .10], [-.05, -.20]])).ravel()+rng.normal(size=xy.size)*.0003
        v = rng.normal(size=len(u)); v /= np.linalg.norm(v)
        fixed = (2*np.flatnonzero(xy[:, 0] == 0)[:, None]+np.arange(2)).ravel()
        free = np.setdiff1d(np.arange(len(u)), fixed)
        for lumped in (False, True):
            nv = P2P1Volume(self.el, lumped=lumped)
            e, g, H, A, B = self.evaluate(u, nv)
            h = 1e-6
            ep, gp, *_ = self.evaluate(u+h*v, nv, False)
            em, gm, *_ = self.evaluate(u-h*v, nv, False)
            self.assertAlmostEqual((ep-em)/(2*h), g@v, delta=1e-8)
            np.testing.assert_allclose((gp-gm)/(2*h), H@v, atol=3e-7, rtol=3e-6)
            np.testing.assert_allclose(H, H.T, atol=2e-12)
            Q = np.array([[np.cos(.7), -np.sin(.7)], [np.sin(.7), np.cos(.7)]])
            rotated = ((xy+u.reshape(-1, 2))@Q.T-xy).ravel()
            er, gr, *_ = self.evaluate(rotated, nv, False)
            self.assertAlmostEqual(er, e, delta=1e-11)
            np.testing.assert_allclose(gr.reshape(-1, 2), g.reshape(-1, 2)@Q.T, atol=1e-11)
            # Positive shift here is only a linear algebra test, not an equilibrium.
            from scipy.sparse import eye
            shift = 100.
            du = nv.solve(A+shift*eye(len(u)), B, free, v[free])
            reference = np.linalg.solve(H[np.ix_(free, free)]+shift*np.eye(len(free)), v[free])
            np.testing.assert_allclose(du, reference, atol=1e-12)

    def test_homogeneous_law_and_consistent_linear_pressure_reproduction(self):
        xy = self.el.mesh.reference_xy
        F = np.array([[1.2, .4], [.1, .8]])
        u = (xy@(F-np.eye(2)).T).ravel()
        full = self.el.bulk(u)
        for lumped in (False, True):
            nv = P2P1Volume(self.el, lumped=lumped)
            r, k, B, e = nv.correct(u, full[0], full[1], full[3])
            self.assertAlmostEqual(e, full[3], places=11)
            np.testing.assert_allclose(r, full[0], atol=1e-12)
        nv = P2P1Volume(self.el)
        qxy = np.einsum('qa,eai->eqi', self.el.points, xy[self.el.elements[:, :3]])
        values = (.1+.2*qxy[:, :, 0]-.1*qxy[:, :, 1]).ravel()
        p = nv._mass_factor.solve(nv.incidence@values)
        exact = nv.K*(.1+.2*xy[:self.el.base_nodes, 0]-.1*xy[:self.el.base_nodes, 1])
        np.testing.assert_allclose(p, exact, atol=4e-13)

    def test_patch_retains_immutable_seed_for_displacement_change(self):
        from dataclasses import asdict
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from prototype.macro_element import refine_macro_mesh
        from tools.check_tip_pairing import solve
        mesh, _ = refine_macro_mesh(rectangle(.5), np.empty((0, 2), int))
        xy = mesh.reference_xy
        seed = .002*np.column_stack((np.sin(3*xy[:, 0])*np.sin(3*xy[:, 1]),
                                      np.sin(2*xy[:, 0])*np.sin(2*xy[:, 1])))
        with TemporaryDirectory() as directory:
            source = Path(directory)/'state.npz'
            np.savez(source, body_nodes=mesh.node_count, reference_xy=xy,
                     positions=xy+seed, triangles=mesh.triangles,
                     macro_triangles=mesh.macro_triangles)
            row, field = solve(source, asdict(self.el.config), 10., 'p1-full')
        from scipy.spatial import cKDTree
        _, order = cKDTree(xy).query(field['reference_xy'])
        change = np.max(np.linalg.norm(field['displacement'].reshape(-1, 2)-seed[order], axis=1))
        self.assertGreater(change, 1e-5)
        self.assertAlmostEqual(row['max_interior_change_mm'], change, places=13)


if __name__ == '__main__':
    unittest.main()
