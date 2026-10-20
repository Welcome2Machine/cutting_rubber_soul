"""Explicit uncalibrated material candidates must not change contact or geometry."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.ipc_reference import BASELINE, IPCProblem, parse_args, run, toolkit
from prototype.material import bulk


class ParameterCommandTests(unittest.TestCase):
    def test_interior_coarsening_is_explicit_and_mini_only(self):
        with tempfile.TemporaryDirectory() as folder:
            common=['--current','--output',folder,'--no-visualize']
            self.assertIsNone(parse_args(common).interior_spacing)
            args=parse_args(common+['--element','mini','--interior-spacing','.2'])
            self.assertEqual(args.interior_spacing,.2)
            for extra in (['--interior-spacing','.2'],['--element','mini','--interior-spacing','.04'],
                          ['--element','mini','--interior-spacing','nan']):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    parse_args(common+extra)

    def test_current_and_frozen_catalogue_moduli_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(parse_args(['--output', folder]).shear_modulus, .47)
            self.assertEqual(parse_args(['--output', folder]).process_policy, 'compute')
            for options in (['--baseline'], ['--baseline', '--shear-modulus', '.66']):
                args = parse_args(['--output', folder] + options)
                self.assertEqual(args.shear_modulus, .66)
                self.assertEqual(args.process_policy, 'system')
            args = parse_args(['--output', folder, '--shear-modulus', '.4'])
            self.assertEqual(args.shear_modulus, .4)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--output', folder, '--baseline', '--shear-modulus', '.4'])

    def test_current_preset_and_explicit_diagnostic_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            args = parse_args(['--current', '--output', folder])
            self.assertEqual((args.depth, args.shear_modulus, args.ogden_alpha), (3., .47, 2.))
            self.assertEqual((args.tip_radius, args.friction, args.nose_facets), (.02, .35, 64))
            self.assertEqual((args.local_spacing, args.surface_spacing, args.fine_halfwidth), (.05, .005, 3.))
            self.assertEqual(args.predictor, 'secant')
            self.assertTrue(args.track_tip_contact and args.adaptive_load)
            args = parse_args(['--current', '--output', folder, '--depth', '.1', '--seconds', '30',
                               '--ogden-alpha', '1', '--predictor', 'affine', '--no-adaptive-load',
                               '--no-track-tip-contact'])
            self.assertEqual((args.depth, args.seconds, args.ogden_alpha), (.1, 30., 1.))
            self.assertEqual(args.predictor, 'affine')
            self.assertFalse(args.adaptive_load or args.track_tip_contact)
            for extra in (['--baseline', '--current'], ['--baseline', '--ogden-alpha', '1'],
                          ['--ogden-alpha', '0'], ['--ogden-alpha', 'nan'],
                          ['--baseline', '--process-policy', 'compute'],
                          ['--current', '--contact-formulation', 'offset-geometric']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(['--output', folder] + extra)
            diagnostic = parse_args(['--output', folder, '--contact-formulation', 'offset-geometric'])
            self.assertEqual(diagnostic.contact_formulation, 'offset-geometric')

    def test_nonfinite_and_nonpositive_moduli_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for value in ('nan', 'inf', '-inf', '0', '-.1'):
                with self.subTest(value=value):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parse_args(['--output', folder, '--shear-modulus=' + value])
                    with self.assertRaisesRegex(ValueError, 'finite and positive'):
                        IPCProblem('patch', shear_modulus=float(value))


class NativeParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ipc = toolkit()
        except ModuleNotFoundError:
            raise unittest.SkipTest('Optional ipctk==1.6.0 not installed')
        cls.ipc.set_num_threads(1)

    def setUp(self):
        limiter = threadpool_limits(limits=1)
        limiter.__enter__()
        self.addCleanup(limiter.__exit__, None, None, None)

    @staticmethod
    def active_state(p):
        x = p.rest.copy()
        x[:p.n, 1] *= .99
        x[p.n:, 1] -= .01 + 2*p.dhat - .006
        p.previous = x.copy()
        x[p.n:, 0] += .002
        return x

    def test_default_equals_explicit_current_in_both_discretizations(self):
        for element in ('p1', 'p2-interior'):
            with self.subTest(element=element):
                options = dict(case='patch', mu=.35, element=element,
                               local_spacing=.2, surface_spacing=.2)
                original = IPCProblem(**options)
                explicit = IPCProblem(**options, shear_modulus=.47)
                self.assertEqual(original.config, explicit.config)
                np.testing.assert_array_equal(original.rest, explicit.rest)
                x = self.active_state(original)
                explicit.previous = original.previous.copy()
                a = original.evaluate(x, original.lag(x), projected=False)
                b = explicit.evaluate(x, explicit.lag(x), projected=False)
                self.assertEqual(a[0], b[0])
                self.assertEqual(a[3], b[3])
                np.testing.assert_array_equal(a[1], b[1])
                np.testing.assert_array_equal(a[2].toarray(), b[2].toarray())

    def test_changed_modulus_changes_material_only_at_fixed_state(self):
        a = IPCProblem('patch', mu=.35, shear_modulus=.66)
        b = IPCProblem('patch', mu=.35, shear_modulus=1.32)
        np.testing.assert_array_equal(a.rest, b.rest)
        np.testing.assert_array_equal(a.surface_dofs, b.surface_dofs)
        np.testing.assert_array_equal(a.free, b.free)
        np.testing.assert_array_equal(a.mesh.triangles, b.mesh.triangles)
        self.assertEqual(a.options, b.options)
        ca, cb = a.config.to_dict(), b.config.to_dict()
        ca.pop('shear_modulus_n_mm2'); cb.pop('shear_modulus_n_mm2')
        self.assertEqual(ca, cb)
        self.assertEqual(a.config.poisson_ratio, .49)
        self.assertEqual(b.config.bulk_modulus_n_mm2, 2*a.config.bulk_modulus_n_mm2)
        x = self.active_state(a)
        b.previous = a.previous.copy()
        u = (x[:a.n] - a.mesh.reference_xy).ravel()
        ba, bb = bulk(a.mesh, u, a.config), bulk(b.mesh, u, b.config)
        np.testing.assert_allclose(bb[0], 2*ba[0], rtol=1e-14, atol=1e-12)
        np.testing.assert_allclose(bb[1], 2*ba[1], rtol=1e-14, atol=1e-12)
        self.assertAlmostEqual(bb[3], 2*ba[3], places=12)
        # Contact is active and sliding, so this checks a nonzero normal and
        # tangential response, not just equality of two empty collision sets.
        ga = a.evaluate(x, a.lag(x), tangent=False)[1]
        gb = b.evaluate(x, b.lag(x), tangent=False)[1]
        contact = []
        for p, material, full in ((a, ba, ga), (b, bb, gb)):
            internal = np.zeros(p.total)
            np.add.at(internal, material[2].ravel(), material[0].ravel())
            contact.append(full-internal)
        self.assertGreater(np.linalg.norm(contact[0][2*a.n:]), 1e-3)
        np.testing.assert_allclose(contact[0], contact[1], rtol=1e-12, atol=1e-12)

    def test_run_records_candidate_and_rejects_changed_modulus_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            common = ['--case', 'patch', '--depth', '.01', '--increment', '.01',
                      '--seconds', '30', '--threads', '1', '--no-visualize']
            args = parse_args(common + ['--output', str(root/'candidate'),
                                        '--shear-modulus', '.4'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(run(args))
            report = json.loads((root/'candidate/result.json').read_text(encoding='utf-8'))
            self.assertEqual(report['config']['shear_modulus_n_mm2'], .4)
            self.assertEqual(report['config']['poisson_ratio'], .49)
            self.assertEqual(report['barrier_stiffness_mpa'], BASELINE['barrier_stiffness_mpa'])
            status = report['material_parameter_status']
            self.assertFalse(status['experimentally_calibrated'])
            self.assertEqual(status['shear_modulus_role'], 'diagnostic_or_calibration_candidate')
            self.assertTrue(status['poisson_ratio_fixed'])
            self.assertFalse(status['barrier_stiffness_scales_with_shear_modulus'])
            self.assertFalse(report['simplifications']['fracture'])
            self.assertEqual(report['runtime_process_policy']['requested'], 'compute')
            self.assertIn('restored', report['runtime_process_policy'])
            self.assertEqual(report['config']['ogden_alpha'], 2.)
            resumed = parse_args(common + ['--output', str(root/'changed'), '--depth', '.02',
                                          '--resume', str(root/'candidate'), '--shear-modulus', '.5'])
            with self.assertRaisesRegex(ValueError, 'Resume mismatch: config'):
                run(resumed)
            changed_alpha = parse_args(common + ['--output', str(root/'changed_alpha'), '--depth', '.02',
                '--resume', str(root/'candidate'), '--shear-modulus', '.4', '--ogden-alpha', '1'])
            with self.assertRaisesRegex(ValueError, 'Resume mismatch: config'):
                run(changed_alpha)

    def test_ogden_shape_keeps_small_strain_moduli_and_all_contact_inputs(self):
        from prototype.material import from_config, response
        from dataclasses import replace
        a = IPCProblem('patch', mu=.35, ogden_alpha=2.)
        b = IPCProblem('patch', mu=.35, ogden_alpha=1.)
        self.assertEqual(a.config, replace(b.config, ogden_alpha=2.))
        np.testing.assert_array_equal(a.rest, b.rest)
        np.testing.assert_array_equal(a.mesh.triangles, b.mesh.triangles)
        self.assertEqual(from_config(a.config).moduli(), from_config(b.config).moduli())
        for config in (a.config, b.config):
            F = np.array([[[1., 1e-6], [0., 1.]]])
            q = response(F, config)
            self.assertAlmostEqual(q['P'][0,0,1]/1e-6, .47, places=9)
        F = np.array([[[1., 2.], [0., 1.]]])
        self.assertLess(response(F,b.config)['P'][0,0,1], response(F,a.config)['P'][0,0,1])


if __name__ == '__main__':
    unittest.main()
