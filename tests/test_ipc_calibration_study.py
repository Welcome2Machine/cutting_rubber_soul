"""Synthetic provenance/qualification tests; do not launch finite-element cases."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

from run_support import write_json
from tools.run_ipc_calibration_study import prepare, summarize
from tools.run_ipc_queue import mechanical_completion, run_jobs


class CalibrationStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)/'study'
        self.state = prepare(self.output, depth=.02)

    def report(self, job, count=3, status='completed'):
        folder = Path(job['output'])
        (folder/'states').mkdir(parents=True, exist_ok=True)
        config = dict(self.state['base_config'], shear_modulus_n_mm2=job['shear_modulus_mpa'],
                      friction_coefficient=job['friction'])
        steps, prior = [], None
        for index in range(count):
            depth = .01*index
            coordinates = np.array([[0., 0.], [.5, -np.sqrt(3)/2], [np.sqrt(3)/2, .5], [1., 1.]])
            positions = coordinates.copy()
            positions[3, 1] -= depth + .005/np.cos(np.pi/3)
            np.savez(folder/'states'/f'{index:04d}.npz', reference_xy=coordinates,
                     positions=positions, previous_positions=coordinates if prior is None else prior,
                     triangles=np.array([[0, 1, 2]], dtype=np.int32), body_nodes=3, depth_mm=depth)
            prior = positions
            steps.append(dict(depth_mm=depth, reaction_fz_n=depth*(2*job['shear_modulus_mpa']+3*job['friction']),
                              residual_n=1e-5, min_j=.9, minimum_distance_mm=.001, friction_work_n_mm=0.,
                              tip_contact=dict(status='unique', depth_mm=depth, gap_mm=.001,
                                               reference_offset_mm=depth*(4*job['shear_modulus_mpa']+5*job['friction']))))
        report = dict(status=status, completed=status == 'completed', case='oblique',
                      target_mm=.02, loading_increment_mm=.01, last_depth_mm=steps[-1]['depth_mm'],
                      force_tolerance_n=1e-4, config=config, steps=steps,
                      dhat_mm=.0025, barrier_stiffness_mpa=6.6, max_inner_iterations=500,
                      hessian_mode='regularized', linear_solver=self.state.get('linear_solver', 'reused-lu'),
                      baseline_config_sha256=self.state['baseline_config_sha256'],
                      mesh_settings=dict(bulk_spacing_mm=1., local_spacing_mm=.05,
                                         surface_spacing_mm=.005, fine_halfwidth_mm=3.),
                      simplifications=dict(displacement_element='P1, fixed mesh', tip_radius_mm=.02,
                                           nose_facets=64, fracture=False,
                                           material='existing finite-strain plane-strain elastic law',
                                           friction_smoothing_mm=1e-5),
                      source_sha256=self.state['model_source_sha256'], ipc_binary_sha256='binary', ipc_version='1.6.0',
                      friction_solver='coupled_pseudo_transient',
                      friction_solver_settings=dict(initial_metric_shift=.1, model_error_fraction=.25,
                                                    pseudo_time_growth=1.2, max_tip_radius_fraction=.5,
                                                    max_model_retries=18, rejection_metric_shift_floor=.1))
        write_json(folder/'result.json', report)
        return report

    def change(self, index, update):
        job = self.state['jobs'][index]
        path = Path(job['output'])/'result.json'
        report = json.loads(path.read_text())
        update(report)
        # json.dumps deliberately permits malformed NaN input for rejection tests.
        path.write_text(json.dumps(report), encoding='utf-8')

    def all_reports(self):
        for job in self.state['jobs']:
            self.report(job)

    def test_plan_preserves_physics_and_freezes_source(self):
        state = self.state
        self.assertEqual([j['name'] for j in state['jobs']], ['base', 'g_minus', 'g_plus', 'mu_minus', 'mu_plus'])
        self.assertEqual(state['main_target_mm'], 3.)
        self.assertEqual(state['solve_budget_s'], 900.)
        self.assertEqual(state['linear_solver'], 'direct')
        self.assertEqual(state['base_config']['poisson_ratio'], .49)
        self.assertEqual(state['base_config']['ogden_alpha'], 2.)
        self.assertEqual(state['contact_formulation'], 'improved-max')
        for job in state['jobs']:
            command = job['command']
            for option, value in {'--inclination': '60', '--element': 'p1', '--tip-radius': '.02',
                                  '--local-spacing': '.05', '--surface-spacing': '.005', '--fine-halfwidth': '3',
                                  '--nose-facets': '64', '--dhat': '.0025', '--increment': '.01',
                                  '--linear-solver': 'direct', '--max-iterations': '500',
                                  '--ogden-alpha': '2', '--contact-formulation': 'improved-max'}.items():
                self.assertEqual(command[command.index(option)+1], value)
            self.assertNotIn('--resume', command)
            self.assertIn('--track-tip-contact', command)
            self.assertEqual(command[command.index('--shear-modulus')+1], str(job['shear_modulus_mpa']))
        for relative in state['source_sha256']:
            self.assertTrue((self.output/'source'/relative).is_file())

    def test_explicit_reused_lu_and_legacy_queue_without_solver_key(self):
        self.output = self.output.parent/'reuse'
        self.state = prepare(self.output, depth=.02, linear_solver='reused-lu')
        self.assertEqual(self.state['linear_solver'], 'reused-lu')
        for job in self.state['jobs']:
            command = job['command']
            self.assertEqual(command[command.index('--linear-solver')+1], 'reused-lu')
        self.all_reports()
        self.assertEqual(summarize(self.output)['qualification'], 'passed')
        # Original frozen queues used reused-lu and predate this explicit key.
        self.state.pop('linear_solver')
        write_json(self.output/'queue.json', self.state)
        self.assertEqual(summarize(self.output)['qualification'], 'passed')
        self.change(1, lambda r: r.update(linear_solver='direct'))
        self.assertEqual(summarize(self.output)['qualification'], 'failed')

    def test_new_default_queue_rejects_unrecorded_solver_change(self):
        self.all_reports()
        self.change(1, lambda r: r.update(linear_solver='reused-lu'))
        self.assertEqual(summarize(self.output)['qualification'], 'failed')

    def test_invalid_plan_rejected_before_writing_and_existing_not_overwritten(self):
        new = self.output.parent/'invalid'
        for kwargs in ({'seconds': 0}, {'depth': 3}, {'g0': float('nan')}, {'g_fraction': 1},
                       {'mu_step': .5}, {'threads': 0}, {'linear_solver': 'unknown'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare(new, **kwargs)
            self.assertFalse(new.exists())
        with self.assertRaises(ValueError):
            prepare(self.output)

    def test_pending_and_partial_produce_no_derivatives(self):
        self.assertEqual(summarize(self.output)['qualification'], 'pending')
        self.report(self.state['jobs'][0], count=2, status='running')
        result = summarize(self.output)
        self.assertEqual(result['qualification'], 'pending')
        self.assertEqual(result['cases'][0]['accepted_states'], 2)
        self.assertEqual(result['sensitivities'], [])
        self.assertEqual(len((self.output/'accepted_prefixes.csv').read_text().splitlines()), 3)

    def test_complete_five_cases_centered_derivatives(self):
        self.all_reports()
        result = summarize(self.output)
        self.assertEqual(result['qualification'], 'passed')
        self.assertFalse(result['calibration_ready'])
        self.assertFalse(result['experiment_fitted'])
        for row in result['sensitivities']:
            for key, slope in [('dFz_dG', 2), ('dFz_dmu', 3), ('dTipOffset_dG', 4), ('dTipOffset_dmu', 5)]:
                self.assertAlmostEqual(row[key], slope*row['depth_mm'])

    def test_false_completion_and_failed_prefix_are_not_inverse_samples(self):
        self.all_reports()
        self.report(self.state['jobs'][1], count=2, status='completed')
        result = summarize(self.output)
        self.assertEqual(result['qualification'], 'failed')
        self.assertEqual(result['cases'][1]['accepted_states'], 2)
        self.assertEqual(result['sensitivities'], [])
        self.report(self.state['jobs'][1], count=2, status='failed')
        self.assertEqual(summarize(self.output)['cases'][1]['qualification'], 'failed')

    def test_nan_force_or_jacobian_or_unique_tip_rejected(self):
        for key in ('reaction_fz_n', 'min_j', 'tip'):
            self.all_reports()
            def corrupt(report):
                if key == 'tip':
                    report['steps'][1]['tip_contact']['reference_offset_mm'] = float('nan')
                else:
                    report['steps'][1][key] = float('nan')
            self.change(1, corrupt)
            self.assertEqual(summarize(self.output)['qualification'], 'failed')

    def test_source_physics_runtime_mismatch_rejected(self):
        changes = [lambda r: r['source_sha256'].update({'prototype/material.py': 'wrong'}),
                   lambda r: r['config'].update({'poisson_ratio': .48}),
                   lambda r: r['config'].update({'ogden_alpha': 1.}),
                   lambda r: r.update({'contact_formulation': 'offset-geometric'}),
                   lambda r: r.update({'ipc_binary_sha256': 'different'}),
                   lambda r: r['friction_solver_settings'].update({'model_error_fraction': .5})]
        for change in changes:
            self.all_reports()
            self.change(1, change)
            self.assertEqual(summarize(self.output)['qualification'], 'failed')

    def test_legacy_reports_without_explicit_law_fields_remain_readable(self):
        self.all_reports()
        self.state['base_config'].pop('ogden_alpha')
        self.state.pop('contact_formulation')
        write_json(self.output/'queue.json', self.state)
        for index in range(len(self.state['jobs'])):
            self.change(index, lambda r: r['config'].pop('ogden_alpha'))
        self.assertEqual(summarize(self.output)['qualification'], 'passed')

    def test_geometry_mismatch_rejected(self):
        self.all_reports()
        path = Path(self.state['jobs'][1]['output'])/'states/0001.npz'
        with np.load(path) as saved:
            arrays = {k: saved[k] for k in saved.files}
        arrays['reference_xy'] = arrays['reference_xy']+.1
        np.savez(path, **arrays)
        self.assertEqual(summarize(self.output)['qualification'], 'failed')

    def test_missing_or_ambiguous_tip_stays_empty(self):
        self.all_reports()
        self.change(1, lambda r: r['steps'][1].pop('tip_contact'))
        self.change(3, lambda r: r['steps'][1]['tip_contact'].update(status='ambiguous', reference_offset_mm=999))
        result = summarize(self.output)
        self.assertEqual(result['qualification'], 'passed')
        row = result['sensitivities'][1]
        self.assertIsNone(row['dTipOffset_dG'])
        self.assertIsNone(row['dTipOffset_dmu'])
        self.assertIsNotNone(row['dFz_dG'])

    def test_invalid_contact_gap_or_missing_nominal_contact_has_no_tip_derivative(self):
        for gap in (None, -.001, .0025, float('nan')):
            self.all_reports()
            self.change(1, lambda r: r['steps'][1]['tip_contact'].update(gap_mm=gap))
            result = summarize(self.output)
            self.assertEqual(result['qualification'], 'passed')
            self.assertIsNone(result['sensitivities'][1]['dTipOffset_dG'])
        self.all_reports()
        self.change(0, lambda r: r['steps'][1].pop('tip_contact'))
        row = summarize(self.output)['sensitivities'][1]
        self.assertIsNone(row['dTipOffset_dG'])
        self.assertIsNone(row['dTipOffset_dmu'])

    def test_broken_history_fixed_boundary_or_blade_are_rejected(self):
        for key, point in [('previous_positions', 2), ('positions', 0), ('positions', 3)]:
            self.all_reports()
            path = Path(self.state['jobs'][1]['output'])/'states/0001.npz'
            with np.load(path) as saved:
                arrays = {k: saved[k] for k in saved.files}
            arrays[key][point, 0] += .01
            np.savez(path, **arrays)
            self.assertEqual(summarize(self.output)['qualification'], 'failed')

    def test_missing_depth_or_checkpoint_not_interpolated(self):
        self.all_reports()
        self.change(1, lambda r: r['steps'].pop(1))
        self.assertEqual(summarize(self.output)['qualification'], 'failed')
        self.all_reports()
        (Path(self.state['jobs'][1]['output'])/'states/0001.npz').unlink()
        self.assertEqual(summarize(self.output)['qualification'], 'failed')

    def test_terminal_process_failure_overrides_completed_report(self):
        self.all_reports()
        self.state['jobs'][1]['status'] = 'timed_out'
        write_json(self.output/'queue.json', self.state)
        result = summarize(self.output)
        self.assertEqual(result['qualification'], 'failed')
        self.assertEqual(result['cases'][1]['termination_category'], 'process_timeout')
        self.assertEqual(result['failure_counts'], {'process_timeout': 1})

    def test_solver_budget_is_distinct_from_numerical_failure_and_has_no_loss(self):
        self.with_readings()
        self.all_reports()
        self.report(self.state['jobs'][1], count=2, status='failed')
        self.change(1, lambda r: r.update(error='TimeoutError: IPC friction pseudo-transient wall budget'))
        self.state['jobs'][1].update(status='failed', exit_code=1)
        self.report(self.state['jobs'][2], count=2, status='failed')
        self.change(2, lambda r: r.update(error='RuntimeError: IPC friction pseudo-transient model mismatch'))
        self.state['jobs'][2].update(status='failed', exit_code=1)
        write_json(self.output/'queue.json', self.state)
        result = summarize(self.output)
        self.assertEqual(result['qualification'], 'failed')
        self.assertFalse(result['calibration_ready'])
        self.assertEqual(result['sensitivities'], [])
        self.assertEqual(result['cases'][1]['termination_category'], 'solver_wall_budget')
        self.assertEqual(result['cases'][2]['termination_category'], 'numerical_failure')
        self.assertEqual(result['failure_counts'], {'solver_wall_budget': 1, 'numerical_failure': 1})
        for index in (1, 2):
            self.assertIsNone(result['cases'][index]['force_loss'])
            self.assertFalse(result['cases'][index]['numerical_complete'])
            self.assertEqual(result['cases'][index]['accepted_states'], 2)
            self.assertEqual(result['cases'][index]['exit_code'], 1)

    def test_process_timeout_takes_precedence_over_solver_timeout(self):
        self.all_reports()
        self.report(self.state['jobs'][1], count=2, status='failed')
        self.change(1, lambda r: r.update(error='TimeoutError: IPC wall budget'))
        self.state['jobs'][1]['status'] = 'timed_out'
        write_json(self.output/'queue.json', self.state)
        result = summarize(self.output)
        self.assertEqual(result['cases'][1]['termination_category'], 'process_timeout')
        self.assertEqual(result['failure_counts'], {'process_timeout': 1})

    def test_terminal_missing_or_invalid_evidence_is_not_numerical_failure(self):
        self.all_reports()
        self.state['jobs'][1]['status'] = 'timed_out'
        write_json(self.output/'queue.json', self.state)
        path = Path(self.state['jobs'][1]['output'])/'result.json'
        path.unlink()
        result = summarize(self.output)
        self.assertEqual(result['cases'][1]['termination_category'], 'invalid_evidence')
        self.assertEqual(result['cases'][1]['job_status'], 'timed_out')
        self.assertEqual(result['failure_counts'], {'invalid_evidence': 1})
        self.report(self.state['jobs'][1])
        (path.parent/'states/0001.npz').unlink()
        self.assertEqual(summarize(self.output)['cases'][1]['termination_category'], 'invalid_evidence')

    def test_pending_running_completed_and_process_failure_categories(self):
        self.state['jobs'][1]['status'] = 'running'
        write_json(self.output/'queue.json', self.state)
        result = summarize(self.output)
        self.assertEqual(result['cases'][0]['termination_category'], 'pending')
        self.assertEqual(result['cases'][1]['termination_category'], 'running')
        self.assertEqual(result['failure_counts'], {})
        self.all_reports()
        result = summarize(self.output)
        self.assertTrue(all(case['termination_category'] == 'completed' for case in result['cases']))
        self.report(self.state['jobs'][1], count=2, status='running')
        self.state['jobs'][1].update(status='failed', exit_code=139, error='Native process exited')
        write_json(self.output/'queue.json', self.state)
        result = summarize(self.output)
        self.assertEqual(result['cases'][1]['termination_category'], 'process_failure')
        self.assertEqual(result['cases'][1]['process_error'], 'Native process exited')

    def test_frozen_source_tamper_is_rejected(self):
        (self.output/'source/prototype/material.py').write_text('# changed')
        with self.assertRaises(ValueError):
            summarize(self.output)

    def test_default_queue_completion_still_requires_three_mm(self):
        job = self.state['jobs'][0]
        report = self.report(job)
        self.assertFalse(mechanical_completion(report, job['output']))
        self.assertTrue(mechanical_completion(report, job['output'], expected_target=.02))
        self.assertFalse(mechanical_completion(report, job['output'], expected_target=float('nan')))

    def test_queue_forwards_explicit_target_and_continues_after_failure(self):
        for job in self.state['jobs'][:2]:
            self.report(job)
        state = dict(self.state, jobs=copy.deepcopy(self.state['jobs'][:2]))
        state.pop('comparison_command')
        fake_process = mock.MagicMock()
        fake_process.pid = 123
        fake_process.wait.side_effect = [1, 0]
        with mock.patch('tools.run_ipc_queue.subprocess.Popen', return_value=fake_process):
            run_jobs(self.output, state, {})
        self.assertEqual(state['jobs'][0]['status'], 'failed')
        self.assertEqual(state['jobs'][1]['status'], 'completed')
        self.assertTrue(state['jobs'][1]['mechanics_completed'])

    def with_readings(self, text='0,0,-.001,.001\n.01,.02,.019,.021\n.02,.04,.039,.041\n'):
        readings = Path(self.temporary.name)/'readings.csv'
        readings.write_text('depth_mm,force_n,lower_n,upper_n\n'+text, encoding='utf-8')
        self.output = self.output.parent/'with_readings'
        self.state = prepare(self.output, depth=.02, force_points_path=readings)
        return readings

    def test_force_readings_are_copied_hashed_and_original_can_change(self):
        original = self.with_readings()
        original.write_text('changed original')
        self.assertEqual(summarize(self.output)['qualification'], 'pending')
        self.assertEqual(self.state['force_points']['source_path'], str(original.resolve()))
        (self.output/'force_points.csv').write_text('changed frozen readings')
        with self.assertRaises(ValueError):
            summarize(self.output)

    def test_unscaled_force_loss_only_for_completed_candidates(self):
        self.with_readings()
        self.all_reports()
        self.report(self.state['jobs'][1], count=2, status='failed')
        result = summarize(self.output)
        loss = result['cases'][0]['force_loss']
        self.assertAlmostEqual(loss['rmse_N'], np.sqrt((.0037**2+.0074**2)/3))
        self.assertEqual(loss['outside_range_count'], 2)
        self.assertAlmostEqual(loss['outside_range_rmse_N'], np.sqrt((.0027**2+.0064**2)/3))
        self.assertFalse(loss['scaling_applied'])
        self.assertIsNone(result['cases'][1]['force_loss'])
        self.assertEqual(result['sensitivities'], [])

    def test_force_readings_validate_values_and_never_interpolate_depth(self):
        path = Path(self.temporary.name)/'invalid.csv'
        for text in ('0,nan,0,1\n', '0,1,0,2\n0,1,0,2\n', '0,3,0,2\n'):
            path.write_text('depth_mm,force_n,lower_n,upper_n\n'+text)
            with self.assertRaises(ValueError):
                prepare(self.output.parent/'invalid', force_points_path=path)
            self.assertFalse((self.output.parent/'invalid').exists())
        self.with_readings('.015,.03,.029,.031\n')
        self.all_reports()
        result = summarize(self.output)
        self.assertEqual(result['qualification'], 'passed')
        self.assertTrue(all(case['force_loss'] is None for case in result['cases']))


if __name__ == '__main__':
    unittest.main()
