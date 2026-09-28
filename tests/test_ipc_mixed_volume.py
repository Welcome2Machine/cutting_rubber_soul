"""Independent mechanics and provenance checks for the opt-in intact mixed host."""
import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.initiation import InitiationOptions, intact_mesh
from prototype.ipc_reference import IPCProblem, parse_args, restore_accepted_prefix, run, toolkit
from prototype.macro_element import refine_macro_mesh
from prototype.material import response
from prototype.model_config import FEMConfig


class MixedConfigurationTests(unittest.TestCase):
    def test_opt_in_and_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(parse_args(['--output', folder]).volumetric_discretization, 'element')
            self.assertEqual(parse_args(['--output', folder, '--volumetric-discretization',
                                        'macro_mixed']).volumetric_discretization, 'macro_mixed')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--output', folder, '--baseline', '--volumetric-discretization', 'macro_mixed'])

    def test_damage_adapters_cannot_silently_use_intact_pressure(self):
        class DamageAdapter(IPCProblem):
            def bulk_terms(self, x, tangent):
                return super().bulk_terms(x, tangent)
        with self.assertRaisesRegex(ValueError, 'damage/pressure coupling'):
            DamageAdapter(volumetric_discretization='macro_mixed')
        with self.assertRaisesRegex(ValueError, 'element or macro_mixed'):
            IPCProblem(volumetric_discretization='nodal_volume')

    def test_macro_preserves_requested_fine_region(self):
        config = FEMConfig(mesh_size_mm=1.)
        options = InitiationOptions(refinement_mm=.25, surface_spacing_mm=.08,
                                    mesh_grading='graded', contact_discretization='integral',
                                    volumetric_discretization='macro_mixed')
        actual, boundary = intact_mesh(config, options, fine_halfwidth_mm=2.5)
        coarse = replace(options, volumetric_discretization='element',
                         refinement_mm=.5, surface_spacing_mm=.16)
        expected, expected_boundary = refine_macro_mesh(
            *intact_mesh(replace(config, mesh_size_mm=2.), coarse, fine_halfwidth_mm=2.5))
        np.testing.assert_array_equal(actual.reference_xy, expected.reference_xy)
        np.testing.assert_array_equal(actual.triangles, expected.triangles)
        np.testing.assert_array_equal(boundary, expected_boundary)
        old, _ = intact_mesh(config, options, fine_halfwidth_mm=.5)
        self.assertGreater(actual.node_count, old.node_count)


class NativeMixedVolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            toolkit().set_num_threads(2)
        except ModuleNotFoundError:
            raise unittest.SkipTest('Optional ipctk==1.6.0 not installed')

    def problem(self):
        return IPCProblem('patch', local_spacing=.25, surface_spacing=.125,
                          tip_radius=.02, fine_halfwidth=.4,
                          volumetric_discretization='macro_mixed')

    def test_full_nonlocal_energy_gradient_tangent_and_consistent_stress(self):
        p = self.problem()
        x = p.rest.copy()
        X = p.mesh.reference_xy
        x[:p.n] = X@np.array([[.93, .07], [-.025, 1.035]])
        x[:p.n, 0] += .008*np.sin(2.3*X[:, 0])*X[:, 1]**2
        x[:p.n, 1] += .012*np.sin(1.7*X[:, 1])*X[:, 0]**2
        x[p.n:, 1] += 2.
        v = np.zeros_like(x)
        v[:p.n] = np.random.default_rng(765).normal(size=(p.n, 2))
        v /= np.linalg.norm(v)
        E, g, H, J, metric = p.evaluate(x, projected=False, with_metric=True)
        h = 1e-6
        a = p.evaluate(x+h*v, tangent=False)
        b = p.evaluate(x-h*v, tangent=False)
        self.assertAlmostEqual((a[0]-b[0])/(2*h), g@v.ravel(), delta=1e-7)
        np.testing.assert_allclose((a[1]-b[1])/(2*h), H@v.ravel(), rtol=2e-6, atol=2e-6)
        self.assertGreater(J, .8)
        self.assertLess(np.max(abs((H-H.T).data), initial=0.), 1e-10)
        self.assertGreater(np.linalg.eigvalsh(metric.toarray()).min(), -1e-8)
        np.testing.assert_allclose(metric.toarray(), p.evaluate(x)[2].toarray(), atol=1e-10)
        q = p.bulk_fields(x)
        V = p.mesh.bulk_data[0]*p.config.width_mm
        self.assertAlmostEqual(float(V@q['energy']), E, delta=1e-10)
        stress_force = np.einsum('e,eiJ,eaJ->eai', V, q['P'], p.grad).reshape(-1, 6)
        assembled = np.zeros(p.total)
        np.add.at(assembled, p.dofs.ravel(), stress_force.ravel())
        np.testing.assert_allclose(assembled, g, atol=1e-10)
        np.testing.assert_allclose(q['cauchy'][:, :2, :2],
                                  q['P']@q['F'].transpose(0, 2, 1)/q['J'][:, None, None], atol=1e-11)
        self.assertGreater(np.max(abs(q['volumetric_pressure_mpa']
                                    -p.config.bulk_modulus_n_mm2*(q['J']-1))), 1e-3)
        self.assertNotIn('tangent', q)

    def test_homogeneous_confined_compression_preserves_material(self):
        p = self.problem()
        F = np.array([[1., 0.], [0., .7]])
        x = p.rest.copy()
        x[:p.n] = p.mesh.reference_xy@F.T
        q = p.bulk_fields(x)
        expected = response(F[None], p.config, tangent=False)
        for key in ('P', 'cauchy', 'energy', 'J'):
            np.testing.assert_allclose(q[key], np.broadcast_to(expected[key], q[key].shape),
                                       rtol=1e-12, atol=1e-11)
        _, _, _, E, _, extra = p.bulk_assembly(x, True)
        self.assertIsNotNone(extra)
        self.assertAlmostEqual(E, float(expected['energy'][0]*p.config.width_mm), places=11)
        self.assertLess(p.mesh.macro_node_count, p.n)

    def test_native_contact_compression_matches_force_and_gap(self):
        p = self.problem()
        x = p.rest.copy()
        old = p.start
        with contextlib.redirect_stdout(io.StringIO()), threadpool_limits(limits=2):
            for depth in (0., .025, .05):
                x, info = p.step(x, old, depth, time.monotonic()+10)
                old = depth
        top = np.flatnonzero(np.isclose(p.mesh.reference_xy[:, 1], 1.))
        stretch = x[top, 1].mean()
        analytic = -response(np.array([[[1., 0.], [0., stretch]]]), p.config,
                             tangent=False)['P'][0, 1, 1]*p.config.width_mm
        self.assertLess(abs(info['reaction_fz_n']/analytic-1), .003)
        self.assertLess(info['residual_n'], 1e-4)
        self.assertGreater(info['minimum_distance_mm'], 0.)
        self.assertFalse(p.ipc.has_intersections(p.cmesh, x[p.surface]))

    def test_runner_checkpoint_and_resume_reject_wrong_pressure_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source_case'
            args = parse_args(['--output', str(source), '--case', 'patch', '--depth', '.025',
                               '--seconds', '15', '--threads', '2', '--local-spacing', '.25',
                               '--surface-spacing', '.125', '--tip-radius', '.02',
                               '--fine-halfwidth', '.4', '--volumetric-discretization', 'macro_mixed'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(run(args))
            report = json.loads((source/'result.json').read_text())
            self.assertEqual(report['volumetric_discretization'], 'macro_mixed')
            state_path = source/report['last_checkpoint']
            with np.load(state_path) as state:
                data = {k: state[k].copy() for k in state.files}
            for name in ('macro_triangles', 'macro_node_count', 'macro_boundary_edges',
                         'macro_parents', 'cauchy', 'J', 'pressure_mpa'):
                self.assertIn(name, data)
            p = self.problem()
            out = Path(folder)/'resumed'
            (out/'states').mkdir(parents=True)
            x, depth = restore_accepted_prefix(source, out, report.copy(), p, np.array([0., .025, .05]))
            self.assertEqual(depth, .025)
            np.testing.assert_array_equal(x, data['positions'])
            data['macro_parents'][0] += 1
            np.savez_compressed(state_path, **data)
            with self.assertRaises(AssertionError):
                restore_accepted_prefix(source, out, report.copy(), p, np.array([0., .025, .05]))
            data['macro_parents'][0] -= 1
            data['pressure_mpa'][0] += .1
            np.savez_compressed(state_path, **data)
            with self.assertRaises(AssertionError):
                restore_accepted_prefix(source, out, report.copy(), p, np.array([0., .025, .05]))
            wrong = report.copy(); wrong['volumetric_discretization'] = 'element'
            with self.assertRaisesRegex(ValueError, 'volumetric_discretization'):
                restore_accepted_prefix(source, out, wrong, p, np.array([0., .025, .05]))


if __name__ == '__main__':
    unittest.main()
