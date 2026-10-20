"""Equivalence to the original P2 contraction, including a block boundary."""
import unittest
import numpy as np
from prototype.ipc_p2 import InteriorP2
from prototype.material import response
from prototype.model_config import FEMConfig
from tests.test_ipc_p2 import rectangle


def original_bulk(p, u, tangent=True):
    """Frozen algebra before the performance change; no new contraction helper."""
    minimum = p.minimum_j(u)
    F = p.deformation(u)
    q = response(F.reshape(-1,2,2),p.config,tangent=tangent)
    volumes = p.area*p.config.width_mm
    nq = len(p.weights)
    r = np.einsum('e,q,eqiJ,eqaJ->eai',volumes,p.weights,
                  q['P'].reshape(-1,nq,2,2),p.grad,optimize=True).reshape(-1,12)
    k = None
    if tangent:
        k = np.einsum('e,q,eqaJ,eqiJkL,eqbL->eaibk',volumes,p.weights,p.grad,
                      q['tangent'].reshape(-1,nq,2,2,2,2),p.grad,
                      optimize=True).reshape(-1,12,12)
    energy = float(np.einsum('e,q,eq->',volumes,p.weights,q['energy'].reshape(-1,nq)))
    return r,k,p.dofs,energy,minimum


class P2AssemblyEquivalenceTests(unittest.TestCase):
    def check_nonaffine(self, h, order=4, tangent=True):
        p = InteriorP2(rectangle(h,length=1.),FEMConfig(width_mm=20.),order)
        x,y = p.mesh.reference_xy.T
        # Smooth nonaffine displacement and independent midpoint motion make
        # F vary over quadrature points, including cells with constrained edges.
        rng = np.random.default_rng(925)
        u = np.column_stack((.07*x+.012*np.sin(2*x+y),
                             -.06*y+.018*np.cos(x-2*y)))
        u[p.base_nodes:] += rng.normal(size=(len(u)-p.base_nodes,2))*h*.01
        F = p.deformation(u)
        self.assertGreater(float(np.max(np.ptp(F,axis=1))),.01)
        expected = original_bulk(p,u,tangent)
        actual = p.bulk(u,with_tangent=tangent)
        np.testing.assert_allclose(actual[0],expected[0],rtol=2e-13,atol=2e-12)
        if tangent:
            np.testing.assert_allclose(actual[1],expected[1],rtol=2e-12,atol=2e-10)
        else:
            self.assertIsNone(actual[1])
        np.testing.assert_array_equal(actual[2],expected[2])
        self.assertAlmostEqual(actual[3],expected[3],delta=1e-12)
        self.assertEqual(actual[4],expected[4])
        self.assertGreater(actual[4],0)
        return len(p.area)

    def test_nonaffine_tangent_across_4096_element_boundary(self):
        self.assertGreater(self.check_nonaffine(1/48),4096)

    def test_higher_quadrature_uses_identical_weights_and_tangent(self):
        self.check_nonaffine(.25,order=6)

    def test_force_only_path_is_unchanged(self):
        self.check_nonaffine(.25,tangent=False)


if __name__=='__main__':
    unittest.main()
