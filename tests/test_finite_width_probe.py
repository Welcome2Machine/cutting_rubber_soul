import unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import coo_matrix
from mechanics.finite_strain_3d import response_3d
from prototype.finite_width_probe import embed, transverse_blocks, integrated_bulk, deformation
from tests.test_finite_strain_3d import material


class WidthProbeTests(unittest.TestCase):
    def test_assembled_cross_blocks_against_fresh_3d_residual(self):
        xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        tri = np.array([[0, 1, 2], [0, 2, 3]])
        grad = np.array([[[-1., 0.], [1., -1.], [0., 1.]], [[0., -1.], [1., 0.], [-1., 1.]]])
        dofs = (2*tri[:, :, None]+np.arange(2)).reshape(2, 6)
        p = SimpleNamespace(n=4, total=8, grad=grad, material=material(), material_dt=.05,
            config=SimpleNamespace(width_mm=2.), previous_material_metric=np.tile(np.eye(3), (2, 1, 1)),
            mesh=SimpleNamespace(reference_xy=xy, triangles=tri, bottom_nodes=np.array([0, 1]),
                bulk_data=(np.ones(2)*.5, grad, dofs)))
        x = xy @ np.array([[1.1, .1], [.2, .8]]).T
        q = response_3d(embed(deformation(p, x)), p.previous_material_metric, .05, p.material)
        blocks = transverse_blocks(p, q, 3)
        rng = np.random.default_rng(3); du = rng.normal(size=(4, 2)); dw = rng.normal(size=(4, 4)); dw[0] = 0.
        eps = 1e-6
        a, b, _ = integrated_bulk(p, x+eps*du, eps*dw, blocks['z'])
        c, d, _ = integrated_bulk(p, x-eps*du, -eps*dw, blocks['z'])
        C = q['tangent'][:, [0, 1, 3, 4]][:, :, [0, 1, 3, 4]].reshape(-1, 2, 2, 2, 2)
        k = np.einsum('eaJ,eiJkL,ebL->eaibk', grad, C, grad).reshape(2, 6, 6)
        K = coo_matrix((k.ravel(), (np.repeat(dofs, 6, axis=1).ravel(), np.tile(dofs, (1, 6)).ravel())), shape=(8, 8)).tocsr()
        np.testing.assert_allclose((a-c)/(2*eps), K@du.ravel()+blocks['B']@dw.ravel(), rtol=2e-7, atol=1e-7)
        # Center-plane reactions need not match: center displacement is fixed to zero.
        expected = blocks['B'].T@du.ravel()+blocks['K']@dw.ravel()
        np.testing.assert_allclose(((b-d)/(2*eps))[4:], expected[4:], rtol=2e-7, atol=1e-7)
        _, r0, _ = integrated_bulk(p, x, np.zeros_like(dw), blocks['z'])
        np.testing.assert_allclose(r0[4:], blocks['r'][4:], atol=1e-11)


if __name__ == '__main__': unittest.main()
