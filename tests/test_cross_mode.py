"""Independent physical controls for the isolated coupling benchmark."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from mechanics.finite_strain import response
from tools.check_cross_mode import Block, material, analytic_checks, derivative_checks


class CrossModeTests(unittest.TestCase):
    def test_affine_patch_exact_material_energy_and_interior_balance(self):
        F = np.array([[1.06, .13], [.02, .93]])
        dt = .17
        for kin in ('plane_strain', 'plane_stress'):
            q = response(F[None], np.eye(3)[None], dt, material(), kinematics=kin)
            expected = 4*2*20*q['incremental_potential'][0]
            for kind in ('p1', 'p2'):
                b = Block(4, kind, kin)
                xy = b.mesh.reference_xy
                u = (xy@(F-np.eye(2)).T).ravel()
                A = np.tile(np.eye(3), (len(b.elements)*b.nq, 1, 1))
                E, g, _, _ = b.evaluate(u, A, dt)
                self.assertAlmostEqual(E, expected, delta=2e-10)
                interior = (xy[:, 0]>0)&(xy[:, 0]<4)&(xy[:, 1]>0)&(xy[:, 1]<2)
                self.assertLess(np.linalg.norm(g.reshape(-1, 2)[interior]), 2e-10)

    def test_analytic_small_strain_and_physical_assembly_derivatives(self):
        with threadpool_limits(2):
            analytic_checks()
            self.assertEqual(len(derivative_checks()), 4)

    def test_enrichment_preserves_exact_exterior_trace(self):
        a = Block(4, 'p1', 'plane_strain')
        b = Block(4, 'p2', 'plane_strain')
        np.testing.assert_array_equal(a.base.reference_xy, b.base.reference_xy)
        np.testing.assert_array_equal(a.fixed, b.fixed)
        extra = b.mesh.reference_xy[a.mesh.node_count:]
        self.assertTrue(np.all((extra[:, 0]>0)&(extra[:, 0]<4)&(extra[:, 1]>0)&(extra[:, 1]<2)))

    def test_surface_gate_detects_motion_between_sparse_probes(self):
        from tools.analyze_cross_mode import comparison
        with tempfile.TemporaryDirectory() as directory:
            x = np.array([0., .5, 1., 1.25, 1.5, 2., 2.5, 3., 3.5, 4.])
            xy = np.column_stack((x, np.full_like(x, 2.)))
            row = dict(depth_mm=.2, force_n=1., min_j=.9, true_inplane_min_j=.9)
            paths = []
            for i in (0, 1):
                u = np.zeros_like(xy); u[3, 0] = i*.01
                p = Path(directory)/f'{i}.npz'
                np.savez(p, reference_xy=xy, displacement=u.ravel()[None])
                paths.append(dict(status='completed', rows=[row], checkpoint=str(p), id=str(i)))
            self.assertAlmostEqual(comparison(*paths)['max_surface_vector_change_mm'], .01)


if __name__ == '__main__':
    unittest.main()
