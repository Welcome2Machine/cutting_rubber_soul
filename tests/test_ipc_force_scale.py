"""Synthetic checks only: no experiment-derived readings are distributed."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.fit_ipc_force_scale import diagnose, fit_scale, load_points, validate_history, validate_points


def synthetic_report():
    return dict(completed=True, status='completed', target_mm=1., last_depth_mm=1.,
                force_tolerance_n=1e-4, simplifications={'fracture': False},
                config={'shear_modulus_n_mm2': .66}, source_sha256={'synthetic.py': 'a'*64},
                steps=[dict(depth_mm=d, reaction_fz_n=f, residual_n=1e-6, min_j=.95,
                            minimum_distance_mm=.002, friction_work_n_mm=0.)
                       for d, f in ((0., 0.), (.25, 2.), (.5, 4.), (1., 8.))])


def readings(values=(1., 2., 4.), halfwidth=.1):
    return [dict(depth_mm=d, force_n=f, lower_n=f-halfwidth, upper_n=f+halfwidth)
            for d, f in zip((.25, .5, 1.), values)]


class ForceScaleTests(unittest.TestCase):
    def test_exact_scale_and_original_residual(self):
        history, audit = validate_history(synthetic_report())
        fit = fit_scale(history, readings())
        self.assertEqual(fit['scale'], .5)
        self.assertEqual(fit['scaled_rmse_n'], 0.)
        self.assertAlmostEqual(fit['original_rmse_n'], np.sqrt(7.))
        self.assertTrue(fit['uniform_scale_feasible'])
        self.assertEqual(fit['allowed_scale_intersection'], [.4875, .5125])
        self.assertEqual(audit['accepted_states'], 4)
        self.assertEqual([r['accepted_step_index'] for r in fit['points']], [1, 2, 3])

    def test_no_single_scale_can_repair_shape(self):
        history, _ = validate_history(synthetic_report())
        fit = fit_scale(history, readings((2., 2., 2.)))
        self.assertAlmostEqual(fit['scale'], 1/3)
        self.assertFalse(fit['uniform_scale_feasible'])
        self.assertIsNone(fit['allowed_scale_intersection'])
        self.assertGreater(fit['interval_intersection_bounds']['max_lower'],
                           fit['interval_intersection_bounds']['min_upper'])
        row = fit['points'][0]
        self.assertAlmostEqual(row['scaled_error_lower_n'], row['scaled_force_n']-2.1)
        self.assertAlmostEqual(row['scaled_error_upper_n'], row['scaled_force_n']-1.9)

    def test_invalid_readings_and_unmatched_depth_are_rejected(self):
        invalid = []
        for key, value in (('force_n', 0), ('force_n', np.nan), ('lower_n', -1),
                           ('lower_n', 3), ('upper_n', .9), ('depth_mm', np.inf),
                           ('depth_mm', 1.5)):
            bad = readings(); bad[0][key] = value; invalid.append(bad)
        invalid.extend([readings()[::-1], [readings()[0]]*2, []])
        for points in invalid:
            with self.subTest(points=points), self.assertRaises(ValueError):
                validate_points(points)
        history, _ = validate_history(synthetic_report())
        bad = readings(); bad[0]['depth_mm'] = .251
        with self.assertRaisesRegex(ValueError, 'interpolation is forbidden'):
            fit_scale(history, bad)

    def test_all_history_checked_even_beyond_fit_points(self):
        report = synthetic_report()
        mutations = [('completed', False), ('status', 'failed'), ('target_mm', 3.),
                     ('last_depth_mm', .5), ('force_tolerance_n', float('nan')),
                     ('force_tolerance_n', 1.),
                     ('source_sha256', {})]
        for key, value in mutations:
            bad = copy.deepcopy(report); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_history(bad)
        bad = copy.deepcopy(report); bad['simplifications']['fracture'] = True
        with self.assertRaisesRegex(ValueError, 'no-fracture'):
            validate_history(bad)
        for key, value in (('residual_n', 2e-4), ('residual_n', -1), ('min_j', 0),
                           ('minimum_distance_mm', 0), ('friction_work_n_mm', -1e-4),
                           ('reaction_fz_n', float('inf')), ('depth_mm', .5)):
            bad = copy.deepcopy(report); bad['steps'][-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_history(bad)

    def test_cli_artifacts_keep_provenance_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run = root/'run'; run.mkdir()
            (run/'result.json').write_text(json.dumps(synthetic_report()), encoding='utf-8')
            source = root/'manual.csv'
            with source.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['depth_mm', 'force_n', 'lower_n', 'upper_n'])
                writer.writeheader(); writer.writerows(readings())
            self.assertEqual(len(load_points(source)), 3)
            output = root/'diagnostic'
            result = diagnose(run, source, output, plot=True)
            saved = json.loads((output/'fit_points.json').read_text())
            self.assertEqual(saved['scale'], result['scale'])
            self.assertFalse(saved['material_calibrated'])
            self.assertFalse(saved['new_fe_solve'])
            self.assertEqual(len(saved['provenance']['manual_points_sha256']), 64)
            self.assertEqual(saved['provenance']['run_source_sha256'], {'synthetic.py': 'a'*64})
            self.assertEqual(len(saved['provenance']['tool_sha256']), 64)
            self.assertGreater((output/'force_scale.png').stat().st_size, 1000)
            self.assertFalse(saved['figure']['manual_observations_connected'])
            self.assertTrue((output/'fit_points.csv').is_file())
            with self.assertRaisesRegex(ValueError, 'must be new'):
                diagnose(run, source, output)


if __name__ == '__main__':
    unittest.main()
