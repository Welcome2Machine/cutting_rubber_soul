"""Assembly-only material output preserves the complete constitutive response."""
import unittest
from types import SimpleNamespace
import numpy as np
from mechanics.finite_strain import Material, elastic_response
from prototype.material import bulk, response
from prototype.model_config import FEMConfig


class MaterialFastPathTests(unittest.TestCase):
    def setUp(self):
        self.material = Material(mu_eq=(.66,), alpha_eq=(2.,), poisson=.49)
        angles = np.array([0., .7, -1.9])
        self.rotations = np.stack([np.array([[np.cos(a), -np.sin(a)],
                                             [np.sin(a), np.cos(a)]]) for a in angles])
        # Distinct quadrature-point values represent an inhomogeneous field;
        # include compression, shear, large stretch and near-identity response.
        self.F = np.concatenate((self.rotations, np.array([
            [[1.00001, .00003], [-.00002, .99996]],
            [[.68, .31], [-.17, 1.42]],
            [[1.85, -.4], [.11, .71]],
            [[.72, 0.], [0., .81]],
        ])))

    def test_complete_and_assembly_only_results_are_identical(self):
        for tangent in (False, True):
            full = elastic_response(self.F, self.material, tangent=tangent)
            fast = elastic_response(self.F, self.material, tangent=tangent, diagnostics=False)
            keys = {'P', 'energy', 'J'} | ({'tangent'} if tangent else set())
            self.assertEqual(set(fast), keys)
            for key in keys:
                np.testing.assert_array_equal(fast[key], full[key])
            self.assertTrue({'cauchy', 'metric', 'lambda_out', 'viscous_rate',
                             'plane_stress_scaled_residual'} <= full.keys())

    def test_adapter_preserves_fast_output(self):
        # Exercise the revised catalogue default and an explicit legacy input.
        for config in (FEMConfig(), FEMConfig(shear_modulus_n_mm2=.66)):
            with self.subTest(shear_modulus=config.shear_modulus_n_mm2):
                material = Material(mu_eq=(config.shear_modulus_n_mm2,),
                                    alpha_eq=(2.,), poisson=config.poisson_ratio)
                actual = response(self.F, config, diagnostics=False)
                expected = elastic_response(self.F, material, diagnostics=False)
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])

    def test_energy_force_and_tangent_directional_derivatives(self):
        direction = np.random.default_rng(217).normal(size=self.F.shape)
        direction /= np.linalg.norm(direction)
        eps = 2e-6
        q = elastic_response(self.F, self.material, diagnostics=False)
        plus = elastic_response(self.F+eps*direction, self.material,
                                tangent=False, diagnostics=False)
        minus = elastic_response(self.F-eps*direction, self.material,
                                 tangent=False, diagnostics=False)
        np.testing.assert_allclose((plus['energy']-minus['energy'])/(2*eps),
            np.einsum('eiJ,eiJ->e', q['P'], direction), rtol=1e-7, atol=3e-9)
        np.testing.assert_allclose((plus['P']-minus['P'])/(2*eps),
            np.einsum('eij,ej->ei', q['tangent'], direction.reshape(-1, 4)).reshape(-1,2,2),
            rtol=2e-8, atol=3e-8)

    def test_rigid_rotation_and_objectivity(self):
        rigid = elastic_response(self.rotations, self.material, diagnostics=False)
        np.testing.assert_allclose(rigid['energy'], 0., atol=2e-15)
        np.testing.assert_allclose(rigid['P'], 0., atol=3e-14)
        q = elastic_response(self.F, self.material, diagnostics=False)
        rotation = self.rotations[1]
        rotated = elastic_response(rotation@self.F, self.material, diagnostics=False)
        np.testing.assert_allclose(rotated['energy'], q['energy'], rtol=1e-12, atol=3e-15)
        np.testing.assert_allclose(rotated['P'], rotation@q['P'], rtol=1e-9, atol=5e-14)

    def test_general_law_and_plane_stress_keep_existing_fallback(self):
        cases = ((Material(mu_eq=(.66,), alpha_eq=(1.3,)), 'plane_strain'),
                 (self.material, 'plane_stress'))
        for material, kinematics in cases:
            full = elastic_response(self.F, material, kinematics=kinematics)
            fast = elastic_response(self.F, material, kinematics=kinematics, diagnostics=False)
            self.assertEqual(full.keys(), fast.keys())
            for key in full:
                np.testing.assert_array_equal(fast[key], full[key])


def full_diagnostics_bulk(mesh, u, config, *, with_tangent=True):
    """Frozen pre-optimization P1 assembly, with complete material diagnostics."""
    areas, gradients, dofs = mesh.bulk_data
    F = np.eye(2)+np.einsum('eai,eaJ->eiJ', u.reshape(-1, 2)[mesh.triangles], gradients)
    q = response(F, config, tangent=with_tangent, diagnostics=True)
    volumes = areas*config.width_mm
    residual = np.einsum('e,eiJ,eaJ->eai', volumes, q['P'], gradients).reshape(-1, 6)
    stiffness = None
    if with_tangent:
        C = q['tangent'].reshape(-1, 2, 2, 2, 2)
        stiffness = np.einsum('e,eaJ,eiJkL,ebL->eaibk', volumes, gradients, C,
                              gradients, optimize=True).reshape(-1, 6, 6)
    return residual, stiffness, dofs, float(volumes@q['energy']), float(q['J'].min())


class P1BulkFastPathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7201)
        count = 48
        points = np.array([[0., 0.], [1., 0.], [.2, .8]])
        reference = points[None]*rng.uniform(.2, 2., (count, 1, 1))
        dm = np.stack((reference[:, 1]-reference[:, 0], reference[:, 2]-reference[:, 0]), axis=2)
        areas = np.linalg.det(dm)/2
        gradients = np.array([[-1., -1.], [1., 0.], [0., 1.]])@np.linalg.inv(dm)
        triangles = np.arange(3*count).reshape(count, 3)
        dofs = (2*triangles[:, :, None]+np.arange(2)).reshape(-1, 6)
        self.mesh = SimpleNamespace(triangles=triangles, bulk_data=(areas, gradients, dofs))
        angles = rng.uniform(-np.pi, np.pi, count)
        R = np.zeros((count, 2, 2))
        R[:, 0, 0] = R[:, 1, 1] = np.cos(angles)
        R[:, 1, 0] = np.sin(angles); R[:, 0, 1] = -np.sin(angles)
        D = np.zeros_like(R)
        D[:, 0, 0], D[:, 1, 1] = np.exp(rng.uniform(np.log(.3), np.log(4.), (2, count)))
        D[:, 0, 1] = rng.uniform(-.8, .8, count)
        F = R@D
        self.u = np.einsum('eij,eaj->eai', F-np.eye(2), reference).ravel()
        self.config = FEMConfig()

    def test_p1_bulk_matches_full_diagnostics_at_random_finite_deformations(self):
        for tangent in (False, True):
            with self.subTest(tangent=tangent):
                full = full_diagnostics_bulk(self.mesh, self.u, self.config, with_tangent=tangent)
                fast = bulk(self.mesh, self.u, self.config, with_tangent=tangent)
                for actual, expected in zip(fast, full):
                    if expected is None:
                        self.assertIsNone(actual)
                    else:
                        np.testing.assert_array_equal(actual, expected)
                self.assertGreater(fast[4], 0.)

    def test_p1_bulk_energy_and_residual_finite_differences(self):
        rng = np.random.default_rng(7202)
        direction = rng.normal(size=self.u.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        residual, stiffness, dofs, _, _ = bulk(self.mesh, self.u, self.config)
        plus = bulk(self.mesh, self.u+eps*direction, self.config, with_tangent=False)
        minus = bulk(self.mesh, self.u-eps*direction, self.config, with_tangent=False)
        derivative = float(np.sum(residual*direction[dofs]))
        self.assertAlmostEqual((plus[3]-minus[3])/(2*eps), derivative,
                               delta=2e-6*max(1., abs(derivative)))
        expected = np.einsum('eij,ej->ei', stiffness, direction[dofs])
        np.testing.assert_allclose((plus[0]-minus[0])/(2*eps), expected,
                                   rtol=2e-5, atol=3e-5)


if __name__ == '__main__':
    unittest.main()
