"""The diagnostic must isolate interior refinement from contact and material."""
import unittest
import copy
import numpy as np
from tools.check_native_mesh import (make_problem, geometry_audit, COMMON, baseline_problem,
                                     validate_resume, signature)


class NativeInteriorMeshTests(unittest.TestCase):
    def test_actual_angle_contact_trace_and_histories_remain_exact(self):
        for angle in (33., 57.):
            base = baseline_problem(angle=angle, **COMMON)
            p = make_problem(angle, 1, 1.)
            checked = geometry_audit(p, base, level=1, radius=1.)
            self.assertTrue(checked['passed'])
            self.assertGreater(p.n, base.n)
            self.assertEqual(p.material_metric.shape[0], len(p.mesh.triangles))
            self.assertIsNone(p.material_dt)
            self.assertEqual(p.material_dissipation, 0.)

    def test_affine_material_energy_is_invariant_under_refinement(self):
        base = make_problem(33., 0, 1.)
        fine = make_problem(33., 1, 1.)
        F = np.array([[1.15, .24], [-.06, .87]])
        energies = []
        for p in (base, fine):
            p.material_dt = .05
            u = (p.mesh.reference_xy@(F-np.eye(2)).T).ravel()
            result = p.material_bulk(u, with_tangent=False)
            energies.append(result[3])
            self.assertAlmostEqual(result[4], np.linalg.det(F), places=10)
            np.testing.assert_array_equal(p.material_metric,
                np.tile(np.eye(3), (len(p.mesh.triangles), 1, 1)))
        self.assertAlmostEqual(*energies, delta=1e-8)

    def test_invalid_refinement_fails_before_model_construction(self):
        for angle, level, radius in ((30, 1, 1), (33, 3, 1), (57, 1, 0), (57, 1, float('nan'))):
            with self.assertRaises(ValueError):
                make_problem(angle, level, radius)

    def test_cross_mesh_history_resume_is_rejected(self):
        base = make_problem(33., 0, 1.)
        fine = make_problem(33., 1, 1.)
        original = dict(signature=signature(base), source_sha256={'tools/check_native_mesh.py': 'same'},
                        mesh_diagnostic=dict(level=0, radius_mm=1.))
        current = copy.deepcopy(original)
        current['mesh_diagnostic']['level'] = 1
        with self.assertRaisesRegex(ValueError, 'exact interior-refinement'):
            validate_resume(original, current)
        current['mesh_diagnostic'] = original['mesh_diagnostic'].copy()
        current['signature'] = signature(fine)
        with self.assertRaisesRegex(ValueError, 'identical material, angle, mesh'):
            validate_resume(original, current)


if __name__ == '__main__':
    unittest.main()
