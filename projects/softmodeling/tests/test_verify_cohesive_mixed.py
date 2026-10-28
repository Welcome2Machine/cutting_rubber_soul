"""Reusable verification gates and artifact failure semantics; no cutting run."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from prototype.cohesive_mixed import MixedModeLaw
from tools.verify_cohesive_mixed import EXAMPLE_PARAMETERS, integrate_path, run, verify


class MixedVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence, cls.traces, cls.series = verify()

    def test_all_independent_gates_pass_and_three_scales_are_recorded(self):
        self.assertTrue(self.evidence['passed'])
        self.assertGreaterEqual(len(self.evidence['gates']), 20)
        self.assertTrue(all(gate['passed'] for gate in self.evidence['gates']))
        for rows in self.evidence['paths'].values():
            self.assertEqual([row['subdivisions_per_leg'] for row in rows], [128, 256, 512])
            self.assertLess(rows[-1]['max_absolute_energy_residual_n_per_mm'],
                            rows[0]['max_absolute_energy_residual_n_per_mm']*.4)
        for row in self.evidence['stable_series']:
            self.assertGreater(row['minimum_total_energy_second_derivative_n_per_mm'], 0.)
            self.assertLess(row['maximum_force_residual_n'], 1e-10)
            self.assertLess(row['maximum_analytic_opening_error_mm'], 1e-10)

    def test_work_is_independent_of_reported_dissipation(self):
        """A 1% traction error must fail even when the law still reports exact Gc."""
        class CorruptTraction(MixedModeLaw):
            def response(self, *arguments, **keywords):
                response = super().response(*arguments, **keywords)
                response['traction'] = response['traction']*1.01
                return response

        evidence, _, _ = verify(CorruptTraction(**EXAMPLE_PARAMETERS))
        self.assertFalse(evidence['passed'])
        gates = {row['name']: row for row in evidence['gates']}
        self.assertFalse(gates['mode_I_work_to_common_toughness']['passed'])
        self.assertFalse(gates['constitutive_derivatives_growing_damage']['passed'])
        self.assertAlmostEqual(evidence['paths']['mode_I'][-1]['final_dissipation_n_per_mm'],
                               EXAMPLE_PARAMETERS['toughness_n_per_mm'])

    def test_unloading_turning_and_closure_have_distinct_semantics(self):
        law = MixedModeLaw(**EXAMPLE_PARAMETERS)
        path = integrate_path(law, [[0., 0.], [.03, 0.], [.01, 0.], [-.02, 0.]])
        self.assertGreater(path['damage'][1], 0.)
        np.testing.assert_array_equal(path['history'][1:], np.full(3, .03))
        np.testing.assert_array_equal(path['dissipated'][1:], np.full(3, path['dissipated'][1]))
        self.assertEqual(path['stored'][-1], 0.)
        np.testing.assert_array_equal(path['traction'][-1], [0., 0.])

    def test_artifact_contains_provenance_and_no_onset_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'new'
            self.assertEqual(run(output, 30), 0)
            report = json.loads((output/'verification.json').read_text(encoding='utf-8'))
            self.assertTrue(report['passed'])
            self.assertEqual(report['parameter_status'], 'theory_verification_only')
            self.assertTrue(report['no_inverse_fitting'])
            self.assertEqual(report['experimental_inputs_used'], [])
            self.assertIsNone(report['physical_onset_depth_prediction'])
            self.assertEqual(set(report['source_sha256']),
                             {'prototype/cohesive_mixed.py', 'tools/verify_cohesive_mixed.py'})
            self.assertTrue(all(len(value) == 64 for value in report['source_sha256'].values()))
            self.assertTrue(report['interpreter'])
            self.assertTrue((output/'verification.svg').is_file())
            self.assertIn('不构成切割起裂深度预测', (output/'index.html').read_text(encoding='utf-8'))
            before = (output/'verification.json').read_bytes()
            with self.assertRaises(FileExistsError):
                run(output, 30)
            self.assertEqual(before, (output/'verification.json').read_bytes())

    def test_failed_gate_has_nonzero_exit_and_never_reports_passed(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence = dict(passed=False, gates=[dict(name='injected_failure', passed=False)])
            with patch('tools.verify_cohesive_mixed.verify', return_value=(evidence, None, None)):
                self.assertEqual(run(Path(temp)/'failed', 30), 1)
            report = json.loads((Path(temp)/'failed'/'verification.json').read_text(encoding='utf-8'))
            self.assertFalse(report['passed'])
            self.assertEqual(report['status'], 'failed_gates')

    def test_timeout_is_an_unqualified_report(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('tools.verify_cohesive_mixed._tick', side_effect=TimeoutError('injected budget limit')):
                self.assertEqual(run(Path(temp)/'timeout', 30), 1)
            report = json.loads((Path(temp)/'timeout'/'verification.json').read_text(encoding='utf-8'))
            self.assertFalse(report['passed'])
            self.assertEqual(report['status'], 'budget_exceeded')
            self.assertNotIn('verification', report)

    def test_invalid_budgets_and_refinement_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for seconds in (0., .5, 301., float('nan'), float('inf')):
                with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                    run(Path(temp)/'unused', seconds)
            self.assertFalse((Path(temp)/'unused').exists())
        for levels in ((8, 16), (8, 16, 16), (8, 16, 12), (4, 8, 16), (8, 16, 32.5)):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                verify(subdivisions=levels)
        with self.assertRaises(ValueError):
            integrate_path(MixedModeLaw(), [[0., 0.]])


if __name__ == '__main__':
    unittest.main()
