"""Independent, small free-side elastic equilibria and qualification gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from tools.verify_network_compliance import (
    CompliancePatch, DHAT_MM, FORCE_TOLERANCE_N, build_matrix, run, run_case,
)


class NetworkComplianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = build_matrix()
        with threadpool_limits(limits=1):
            cls.intact, cls.intact_fields = run_case(cls.specs[5], steps=4, seconds=15)
            cls.network, cls.network_fields = run_case(cls.specs[7], steps=4, seconds=15)
            cls.zero, cls.zero_fields = run_case(cls.specs[-1], steps=4, seconds=15)

    def test_each_axis_has_three_levels_without_confounding(self):
        varied = {'mesh': 'cells', 'gap': 'gap_mm', 'stiffness': 'stiffness_mpa_per_mm'}
        for group, key in varied.items():
            cases = [s for s in self.specs if group in s['groups'] and s['mode'] == 'tension']
            self.assertEqual(len({c[key] for c in cases}), 3)
            for other in {'cells', 'gap_mm', 'stiffness_mpa_per_mm'}-{key}:
                self.assertEqual(len({c[other] for c in cases}), 1)
            self.assertTrue(all(c['compression_stiffness_mpa_per_mm'] == c['stiffness_mpa_per_mm'] for c in cases))
        self.assertTrue(all(s['gap_mm'] > 2*DHAT_MM for s in self.specs))
        joint = sorted((s for s in self.specs if 'joint_limit' in s['groups'] and s['mode'] == 'tension'), key=lambda s: s['cells'])
        self.assertTrue(np.all(np.diff([s['cells']*s['gap_mm'] for s in joint]) < 0.))
        self.assertTrue(np.all(np.diff([s['cells']/s['stiffness_mpa_per_mm'] for s in joint]) < 0.))

    def test_intact_control_is_homogeneous_with_truly_free_sides(self):
        problem = CompliancePatch(3, network=False)
        self.assertEqual(len(problem.fixed[problem.fixed % 2 == 0]), 1)
        self.assertEqual(len(problem.fixed[problem.fixed % 2 == 1]), len(problem.bottom)+len(problem.top))
        displacement = self.intact_fields['positions'][-1]-problem.reference
        gradients = problem.mesh.bulk_data[1]
        deformation = np.eye(2)+np.einsum('eai,eaJ->eiJ', displacement[problem.mesh.triangles], gradients)
        np.testing.assert_allclose(deformation, np.broadcast_to(deformation[0], deformation.shape), atol=1e-10)
        self.assertGreater(deformation[0, 0, 0], 1.)
        self.assertAlmostEqual(self.intact['apparent_modulus_mpa'], 10.6666666667, delta=.005)

    def test_compression_branch_carries_load_but_network_remains_more_compliant(self):
        for result in (self.intact, self.network, self.zero):
            self.assertTrue(result['qualified'], result.get('failure'))
            self.assertLessEqual(result['audit']['maximum_residual_n'], FORCE_TOLERANCE_N)
            self.assertGreater(result['audit']['min_j'], 0.)
            self.assertGreaterEqual(result['audit']['minimum_saved_transition_safe_fraction'], 1.-1e-12)
            self.assertTrue(result['contact']['full_native_EV_VV'])
            self.assertTrue(result['contact']['full_CCD'])
            self.assertLessEqual(result['contact']['ccd_tolerance_mm'], min(result['gap_mm'], DHAT_MM)/10.)
            self.assertEqual(result['contact']['ccd_conservative_rescaling'], .8)
            self.assertTrue(all(r['maximum_damage'] == 0. and r['fracture_dissipation_n_mm'] == 0. for r in result['rows']))
        self.assertLess(self.network['apparent_modulus_mpa'], .95*self.intact['apparent_modulus_mpa'])
        self.assertGreater(self.network['apparent_modulus_mpa'], .8*self.intact['apparent_modulus_mpa'])
        self.assertGreater(self.network['rows'][-1]['compression_stored_n_mm'], 0.)
        self.assertLess(abs(self.zero['rows'][-1]['reaction_n']), self.zero['reaction_resolution_n'])
        self.assertFalse(self.zero['reaction_resolved'])
        self.assertIsNone(self.zero['apparent_modulus_mpa'])
        self.assertIsNone(self.zero['energy']['maximum_relative_remainder'])
        self.assertEqual(self.zero['energy']['qualification_basis'], 'absolute_remainder_only_unresolved_work')
        self.assertEqual(self.intact['duration_s'], self.network['duration_s'])
        self.assertEqual(self.intact['duration_s'], self.zero['duration_s'])

    def test_rejected_equilibrium_is_never_promoted(self):
        def reject(problem, accepted, previous, displacement, deadline):
            problem.last_trial = accepted+.001
            raise RuntimeError('Injected rejection')
        with patch.object(CompliancePatch, 'solve', reject):
            result, fields = run_case(self.specs[7], steps=4, seconds=15)
        self.assertFalse(result['completed'])
        self.assertFalse(result['qualified'])
        self.assertFalse(result['energy']['qualified'])
        self.assertEqual(result['accepted_states'], 1)
        self.assertIsNone(result['apparent_modulus_mpa'])
        np.testing.assert_allclose(fields['failed_trial'], fields['reference']+.001)

    def test_bounded_mode_uses_gap_without_an_extra_fitting_parameter(self):
        spec = build_matrix('gap_bounded')[7]
        with threadpool_limits(limits=1):
            result, _ = run_case(spec, steps=4, seconds=15)
        self.assertTrue(result['qualified'], result['failure'])
        self.assertEqual(result['compression_regularization'], 'gap_bounded')
        self.assertEqual(result['compression_regularization_mm'], result['gap_mm'])
        self.assertEqual(result['compression_traction_cap_mpa'],
                         result['compression_stiffness_mpa_per_mm']*result['gap_mm'])
        self.assertEqual(result['constitutive_identity']['compression_model'], 'bounded_reversible_pseudo_huber')
        self.assertLess(result['apparent_modulus_mpa'], self.network['apparent_modulus_mpa'])
        self.assertEqual(self.network['compression_regularization'], 'linear')
        self.assertIsNone(self.network['compression_regularization_mm'])
        self.assertIsNone(self.network['compression_traction_cap_mpa'])

    def test_report_is_bounded_reproducible_and_never_full_cutting_certification(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'report'
            self.assertEqual(run(output, seconds=15, steps=4, specs=[self.specs[5], self.specs[7], self.specs[-1]]), 0)
            report = json.loads((output/'verification.json').read_text(encoding='utf-8'))
            self.assertTrue(report['passed'])
            self.assertEqual(report['source_sha256'], report['source_sha256_at_end'])
            self.assertEqual(report['experimental_inputs_used'], [])
            self.assertEqual(report['compression_regularization'], ['linear'])
            self.assertIsNone(report['physical_onset_prediction'])
            self.assertFalse(report['interpretation']['complete_cutting_qualified'])
            self.assertFalse(report['contact_scope']['closed_contact_or_friction_qualified'])
            self.assertTrue((output/'summary.csv').is_file())
            self.assertTrue((output/'compliance.png').is_file())
            self.assertIsNone(report['cases'][-1]['modulus_ratio_to_intact'])
            self.assertTrue(all(c['execution']['hard_wall_budget_s'] == 20. for c in report['cases']))
            with self.assertRaises(FileExistsError):
                run(output, seconds=15)

    def test_invalid_inputs_fail_before_loading(self):
        for options in (dict(cells=0), dict(cells=1.5), dict(gap_mm=DHAT_MM), dict(stiffness_mpa_per_mm=-1),
                        dict(compression_regularization='unknown')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CompliancePatch(**options)
        for options in (dict(strain=0), dict(strain=float('nan')), dict(seconds=0), dict(steps=1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_case(self.specs[0], **options)


if __name__ == '__main__':
    unittest.main()
