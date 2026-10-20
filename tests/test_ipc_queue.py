"""Queue lifecycle checks without launching finite-element solves."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from tools.run_ipc_queue import (CASES, prepare, run_jobs, process_identity,
                                 observe_existing, mechanical_completion, main, write_outcome)


SYNTHETIC_COMPLETION_SCRIPT = """
def completed_report(out):
    (out/'states').mkdir()
    steps = []
    for index in range(301):
        (out/'states'/f'{index:04d}.npz').touch()
        steps.append(dict(depth_mm=index*.01, residual_n=1e-5, min_j=.8, reaction_fz_n=float(index)))
    return dict(status='completed', completed=True, target_mm=3., loading_increment_mm=.01,
                last_depth_mm=3., force_tolerance_n=1e-4, steps=steps)
"""


class IPCQueueTests(unittest.TestCase):
    @staticmethod
    def accepted_report(root, pid):
        (root/'states').mkdir(parents=True)
        steps = []
        for index in range(301):
            (root/'states'/f'{index:04d}.npz').touch()
            steps.append(dict(depth_mm=index*.01, residual_n=1e-5, min_j=.8,
                              reaction_fz_n=float(index)))
        report = dict(pid=pid, status='completed', completed=True, target_mm=3.,
                      loading_increment_mm=.01, last_depth_mm=3., force_tolerance_n=1e-4,
                      steps=steps)
        (root/'result.json').write_text(json.dumps(report), encoding='utf-8')
        return report

    def test_current_process_identity_is_stable_without_signals(self):
        if os.name != 'nt' and not sys.platform.startswith('linux'):
            self.skipTest('Process birth identity is only supported on Windows/Linux')
        first = process_identity(os.getpid())
        second = process_identity(os.getpid())
        self.assertEqual(first['status'], 'alive')
        self.assertEqual(first, second)
        self.assertTrue(first['birth'])

    def test_custom_cases_freeze_tracking_and_iteration_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'queue'
            cases = ['60:0', '60:.05', '60:.1', '60:.2', '30:.35', '0:.35']
            state = prepare(root, 2, 0, cases=cases, track_tip_contact=True, max_iterations=500)
            self.assertEqual(len(state['jobs']), 6)
            self.assertEqual(state['jobs'][-1]['name'], '06_angle0_mu0.35')
            for job in state['jobs']:
                self.assertIn('--track-tip-contact', job['command'])
                self.assertEqual(job['command'][job['command'].index('--max-iterations')+1], '500')
                self.assertNotIn('--no-visualize', job['command'])
            self.assertEqual(state['shear_modulus_mpa'], .47)
            self.assertEqual(state['ogden_alpha'], 2.)
            self.assertEqual(state['contact_formulation'], 'improved-max')
            command = state['jobs'][0]['command']
            self.assertEqual(command[command.index('--shear-modulus')+1], '0.47')
            self.assertEqual(command[command.index('--ogden-alpha')+1], '2')
            self.assertEqual(command[command.index('--contact-formulation')+1], 'improved-max')
            self.assertEqual(command[command.index('--process-policy')+1], 'compute')
            self.assertEqual(command[command.index('--friction-globalization')+1], 'model-error')
            self.assertEqual(state['process_policy'], 'compute')
            self.assertEqual(state['friction_globalization'], 'model-error')

    def test_runtime_and_globalization_are_frozen_and_invalid_values_leave_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for options in (dict(process_policy='invalid'), dict(friction_globalization='invalid')):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    prepare(root/'invalid', 2, 21600, cases=['60:.35'], **options)
                self.assertFalse((root/'invalid').exists())
            state = prepare(root/'queue', 2, 21600, cases=['60:.35'],
                            process_policy='system', friction_globalization='residual-merit')
            command = state['jobs'][0]['command']
            for option, value in (('process-policy', 'system'), ('friction-globalization', 'residual-merit')):
                self.assertEqual(command[command.index('--'+option)+1], value)
                self.assertEqual(state[option.replace('-', '_')], value)

    def test_single_current_case_resolves_to_same_arguments_as_current_preset(self):
        from prototype.ipc_reference import parse_args
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = prepare(root/'queue', 2, 21600, cases=['60:.35'], shear_modulus=.47,
                            predictor='secant', adaptive_load=True, track_tip_contact=True,
                            max_iterations=500, process_policy='compute',
                            friction_globalization='model-error')
            queued = vars(parse_args(state['jobs'][0]['command'][4:]))
            current = vars(parse_args(['--current', '--seconds', '21600', '--output', str(root/'current')]))
            for args in (queued, current):
                args.pop('current'); args.pop('output')
            self.assertEqual(queued, current)

    def test_outcome_classifies_failures_without_claiming_physical_impossibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = dict(target_mm=3., source_root=str(root/'source'),
                         source_sha256={'prototype/ipc_reference.py': 'frozen-sha'},
                         solve_budget_s=21600., process_timeout_s=23400.,
                         process_policy='compute', friction_globalization='model-error')
            cases = [
                ('completed', None, 'completed', True, 'completed'),
                ('timed_out', None, 'running', False, 'wall_budget_exhausted'),
                ('failed', 'TimeoutError: IPC friction pseudo-transient wall budget',
                 'failed', False, 'wall_budget_exhausted'),
                ('failed', 'LoadStepFailure: IPC adaptive load exhausted: from=.1 to=.11 mm',
                 'failed', False, 'nonlinear_nonconvergence'),
                ('failed', 'RuntimeError: IPC Newton directions/energy line search failed',
                 'failed', False, 'nonlinear_nonconvergence'),
                ('failed', 'RuntimeError: Invalid geometry at acceptance',
                 'failed', False, 'execution_error'),
            ]
            for index, (job_status, error, report_status, complete, expected) in enumerate(cases):
                with self.subTest(expected=expected, error=error):
                    folder = root/str(index)
                    report = self.accepted_report(folder, 123)
                    report.update(status=report_status, completed=complete, error=error)
                    if not complete:
                        report['steps'] = report['steps'][:2]
                        report['last_depth_mm'] = .01
                    job = dict(output=str(folder), status=job_status, mechanics_completed=complete,
                               elapsed_s=15., exit_code=0 if complete else 1, error=error)
                    outcome = write_outcome(job, state, report)
                    self.assertEqual(outcome['status'], expected)
                    self.assertEqual(outcome['last_accepted_depth_mm'], 3. if complete else .01)
                    self.assertEqual(outcome['source_sha256'], state['source_sha256'])
                    self.assertEqual(outcome['original_error'], error)
                    self.assertEqual(json.loads((folder/'outcome.json').read_text(encoding='utf-8')), outcome)
                    text = (folder/'outcome.md').read_text(encoding='utf-8')
                    self.assertGreater(len(text.splitlines()), 15)
                    self.assertIn('数值失败不证明物理过程不可实现', text)
                    self.assertIn('未全程重新装配', text)

    def test_outcome_rejects_tampered_completion_and_preserves_missing_contact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.accepted_report(root, 123)
            report['steps'][-1]['min_j'] = -1.
            report['steps'][-1]['tip_contact'] = {'status': 'unique'}
            report['attempted_depth_mm'] = 3.
            report['elapsed_s'] = .8
            report['rejected_load_attempts'] = [{'attempted_depth_mm': 3., 'error': 'nonconvergence'}]
            report['tip_contact_tracking'] = {'errors': [{'depth_mm': .03, 'error': 'observation failed'}]}
            for step, status in zip(report['steps'], ('unique', 'no_tip_contact', 'ambiguous', 'error')):
                step['tip_contact'] = {'status': status}
            job = dict(output=str(root), status='failed', mechanics_completed=mechanical_completion(report, root),
                       exit_code=0, elapsed_s=1.)
            outcome = write_outcome(job, dict(track_tip_contact=True), report)
            self.assertEqual(outcome['status'], 'completion_audit_failed')
            self.assertFalse(outcome['mechanics_completed'])
            self.assertEqual(outcome['last_accepted_depth_mm'], 2.99)
            self.assertEqual(outcome['reported_last_depth_mm'], 3.)
            self.assertTrue(outcome['tip_observations_available'])
            self.assertEqual(outcome['tip_observation_count'], 3)
            self.assertEqual(outcome['tip_status_counts']['error'], 1)
            self.assertEqual(outcome['tip_status_counts']['not_recorded'], 296)
            self.assertEqual(outcome['tip_status_counts']['unique'], 1)
            self.assertEqual(outcome['attempted_depth_mm'], 3.)
            self.assertEqual(outcome['solver_elapsed_s'], .8)
            self.assertEqual(outcome['rejected_load_attempt_count'], 1)
            self.assertEqual(outcome['last_rejected_load_attempt'], report['rejected_load_attempts'][-1])
            self.assertEqual(outcome['tip_tracking_errors'], report['tip_contact_tracking']['errors'])
            self.assertEqual(report['steps'][1]['tip_contact'], {'status': 'no_tip_contact'})
            report['steps'][-1]['min_j'] = .8
            report['steps'][-1]['depth_mm'] = 4.
            outcome = write_outcome(job, dict(track_tip_contact=True), report)
            self.assertEqual(outcome['last_accepted_depth_mm'], 2.99)

    def test_outcome_for_missing_report_creates_folder_and_keeps_depth_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'not_created_by_solver'
            job = dict(output=str(root), status='failed', mechanics_completed=False, exit_code=7)
            outcome = write_outcome(job, {}, {})
            self.assertEqual(outcome['status'], 'execution_error')
            self.assertIsNone(outcome['last_accepted_depth_mm'])
            self.assertFalse(outcome['tip_observations_available'])
            self.assertTrue((root/'outcome.md').is_file())

    def test_outcome_write_failure_cannot_stop_following_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = [dict(name=str(index), output=str(root/str(index)), status='pending',
                         command=[sys.executable, '-c', 'print("fake job")']) for index in range(2)]
            state = dict(jobs=jobs, source_root=str(root), process_timeout_s=10)
            with mock.patch('tools.run_ipc_queue.write_outcome', side_effect=OSError('report disk error')) as writer, \
                    contextlib.redirect_stdout(io.StringIO()):
                run_jobs(root, state, os.environ.copy())
            self.assertEqual(writer.call_count, 2)
            self.assertTrue(all(job['exit_code'] == 0 for job in jobs))
            self.assertTrue(all(job['outcome_error'] == 'OSError: report disk error' for job in jobs))
            self.assertEqual(state['status'], 'finished')

    def test_legacy_modulus_must_be_requested_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = prepare(Path(temporary)/'queue', 1, 0, cases=['60:.35'], shear_modulus=.66)
            self.assertEqual(state['shear_modulus_mpa'], .66)
            command = state['jobs'][0]['command']
            self.assertEqual(command[command.index('--shear-modulus')+1], '0.66')
            for value in (0., -.1, float('nan'), float('inf')):
                with self.assertRaisesRegex(ValueError, 'Shear modulus'):
                    prepare(Path(temporary)/'invalid', 1, 0, shear_modulus=value)

    def test_accuracy_queue_rejects_changed_material_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference = root/'reference'
            report = self.accepted_report(reference, os.getpid())
            report['config'] = {'shear_modulus_n_mm2': .66}
            (reference/'result.json').write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'material/contact differs'):
                prepare(root/'bad', 1, 0, cases=['60:.35'], accuracy_reference=reference)
            self.assertFalse((root/'bad').exists())
            state = prepare(root/'legacy', 1, 0, cases=['60:.35'], accuracy_reference=reference,
                            shear_modulus=.66)
            self.assertEqual(state['shear_modulus_mpa'], .66)

    def test_adaptive_accuracy_entry_freezes_one_job_and_comparer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference = root/'reference'
            self.accepted_report(reference, os.getpid())
            output = root/'queue'
            state = prepare(output, 2, 0, cases=['60:.35'], track_tip_contact=True,
                            predictor='secant', adaptive_load=True, minimum_increment=.000625,
                            accuracy_reference=reference)
            self.assertEqual(len(state['jobs']), 1)
            command = state['jobs'][0]['command']
            for flag, value in (('--predictor', 'secant'), ('--minimum-increment', '0.000625'),
                                ('--local-spacing', '.05'), ('--surface-spacing', '.005'),
                                ('--fine-halfwidth', '3'), ('--depth', '3'), ('--increment', '.01')):
                self.assertEqual(command[command.index(flag)+1], value)
            self.assertIn('--adaptive-load', command)
            comparison = state['comparison_command']
            self.assertEqual(comparison[comparison.index('--candidate')+1], state['jobs'][0]['output'])
            self.assertEqual(comparison[comparison.index('--reference')+1], str(reference.resolve()))
            self.assertEqual(comparison[comparison.index('--output')+1], str(output.resolve()/'comparison'))
            self.assertIn('--allow-pending', comparison)
            frozen = output/'source/tools/compare_ipc_accuracy.py'
            self.assertEqual(frozen.read_bytes(), (Path(__file__).resolve().parents[1]/'tools/compare_ipc_accuracy.py').read_bytes())
            self.assertEqual(state['source_sha256']['tools/compare_ipc_accuracy.py'], hashlib.sha256(frozen.read_bytes()).hexdigest())
            self.assertEqual(state['accuracy_reference_result_sha256'], hashlib.sha256((reference/'result.json').read_bytes()).hexdigest())
            self.assertEqual(state['predictor'], 'secant')
            self.assertTrue(state['adaptive_load'])
            self.assertEqual(state['minimum_increment_mm'], .000625)

    def test_new_options_reject_conflicts_before_output_and_preserve_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root/'queue'
            for options in (dict(predictor='unknown'),
                            dict(accuracy_reference=root/'reference', comparison_plan=root/'plan.json'),
                            dict(accuracy_reference=root/'reference'),
                            *(dict(adaptive_load=True, minimum_increment=value)
                              for value in (0., -.001, .02, float('nan'), float('inf')))):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    prepare(output, 2, 0, **options)
                self.assertFalse(output.exists())
            state = prepare(output, 2, 0, cases=['60:.35'], minimum_increment=float('nan'))
            command = state['jobs'][0]['command']
            self.assertEqual(command[command.index('--predictor')+1], 'affine')
            self.assertNotIn('--adaptive-load', command)
            self.assertNotIn('--minimum-increment', command)
            self.assertNotIn('comparison_command', state)
            self.assertIsNone(state['minimum_increment_mm'])
            self.assertFalse(state['adaptive_load'])

    def test_accuracy_reference_requires_completed_three_mm_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference = root/'reference'; output = root/'queue'
            baseline = self.accepted_report(reference, os.getpid())
            invalid = [dict(status='failed'), dict(completed=False), dict(target_mm=.03),
                       dict(last_depth_mm=float('nan')), dict(steps=baseline['steps'][:-1]),
                       dict(steps=[]), dict(steps=[baseline['steps'][0], baseline['steps'][0], baseline['steps'][-1]])]
            for changed in invalid:
                with self.subTest(changed=list(changed)):
                    (reference/'result.json').write_text(json.dumps(dict(baseline, **changed)), encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'completed accepted history'):
                        prepare(output, 2, 0, cases=['60:.35'], accuracy_reference=reference)
                    self.assertFalse(output.exists())

    def test_cli_passes_adaptive_and_accuracy_options_without_launching_fe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference = root/'reference'; output = root/'queue'
            self.accepted_report(reference, os.getpid())
            argv = ['run_ipc_queue.py', '--output', str(output), '--cases', '60:.35', '--seconds', '0',
                    '--predictor', 'secant', '--adaptive-load', '--track-tip-contact',
                    '--accuracy-reference', str(reference), '--process-policy', 'system',
                    '--friction-globalization', 'residual-merit']
            with mock.patch.object(sys, 'argv', argv), mock.patch('tools.run_ipc_queue.run_jobs') as jobs:
                main()
            state = jobs.call_args.args[1]
            self.assertEqual(state['process_policy'], 'system')
            self.assertEqual(state['friction_globalization'], 'residual-merit')
            jobs.assert_called_once()
            state = jobs.call_args.args[1]
            self.assertEqual(state['minimum_increment_mm'], .0003125)
            self.assertEqual(state['predictor'], 'secant')
            self.assertIn('--allow-pending', state['comparison_command'])
            other = root/'invalid'
            with (mock.patch.object(sys, 'argv', ['queue', '--output', str(other), '--cases', '60:.35',
                                                '--comparison-plan', 'x.json', '--accuracy-reference', str(reference)]),
                  contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                main()
            self.assertFalse(other.exists())

    def test_plan_and_existing_run_are_frozen_without_becoming_a_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); repository = root/'repository'
            for name in ('prototype/ipc_baseline.json', 'tools/visualize_ipc.py',
                         'tools/run_ipc_queue.py', 'requirements.txt', 'tools/compare_ipc_study.py'):
                target = repository/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('# frozen test input\n')
            plan = root/'plan.json'; plan.write_text('{"schema_version": 1}\n')
            external = root/'external'; external.mkdir()
            (external/'result.json').write_text(json.dumps(dict(pid=os.getpid(), status='running')))
            with mock.patch('tools.run_ipc_queue.ROOT', repository):
                state = prepare(root/'queue', 2, 0, cases=['60:0'], comparison_plan=plan,
                                observed_runs=[dict(output=external, pid=os.getpid())])
            self.assertEqual(len(state['jobs']), 1)
            self.assertEqual(state['observed_runs'][0]['pid'], os.getpid())
            self.assertEqual(state['observed_runs'][0]['ownership'], 'external_read_only')
            self.assertEqual((root/'queue/study_plan.json').read_bytes(), plan.read_bytes())
            self.assertTrue((root/'queue/source/tools/compare_ipc_study.py').is_file())
            self.assertIn('tools/compare_ipc_study.py', state['source_sha256'])
            self.assertEqual(state['comparison_command'][-1], str(root/'queue/comparison'))

    def test_invalid_or_conflicting_cases_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'queue'
            for cases in ([], ['60:0', '60:0.0'], ['30:nan'], ['60:-.1'], ['90:.1'], ['oops']):
                with self.subTest(cases=cases), self.assertRaises(ValueError):
                    prepare(root, 2, 0, cases=cases)
                self.assertFalse(root.exists())
            with self.assertRaises(ValueError):
                prepare(root, 2, 0, cases=['60:0'], friction_only=True)
            with self.assertRaises(ValueError):
                prepare(root, 2, 0, cases=['60:0'], resume_first=Path(temporary))

    def test_accepted_report_requires_full_path_and_residuals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.accepted_report(root, os.getpid())
            self.assertTrue(mechanical_completion(report, root))
            report['steps'][30]['residual_n'] = .01
            self.assertFalse(mechanical_completion(report, root))
            report['steps'][30]['residual_n'] = 1e-5
            report['steps'].pop()
            self.assertFalse(mechanical_completion(report, root))

    def test_adaptive_completion_keeps_all_substeps_and_output_anchors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.accepted_report(root, os.getpid())
            for step in report['steps']:
                step.update(minimum_distance_mm=.001, friction_work_n_mm=0.)
            extra = dict(report['steps'][53], depth_mm=.535)
            report['steps'].insert(54, extra)
            (root/'states/0301.npz').touch()
            # Extra states cannot silently change the old fixed schedule.
            self.assertFalse(mechanical_completion(report, root))
            report['load_control'] = dict(mode='adaptive_bisection', minimum_increment_mm=.00125,
                maximum_increment_mm=.01, output_grid_increment_mm=.01, retry_predictor='affine',
                stagnation_window=40, stagnation_improvement_fraction=.1)
            self.assertTrue(mechanical_completion(report, root))
            for key, value in (('residual_n', .001), ('min_j', 0.), ('minimum_distance_mm', 0.),
                               ('friction_work_n_mm', -1e-5), ('depth_mm', .53)):
                with self.subTest(key=key):
                    bad = copy.deepcopy(report); bad['steps'][54][key] = value
                    self.assertFalse(mechanical_completion(bad, root))
            bad = copy.deepcopy(report)
            # All increments remain <= .01, but .54 is missing.
            bad['steps'][55]['depth_mm'] = .545
            self.assertFalse(mechanical_completion(bad, root))
            (root/'states/0054.npz').unlink()
            self.assertFalse(mechanical_completion(report, root))

    def test_adaptive_metadata_and_terminal_state_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.accepted_report(root, os.getpid())
            for step in report['steps']:
                step.update(minimum_distance_mm=.001, friction_work_n_mm=0.)
            report['load_control'] = dict(mode='adaptive_bisection', minimum_increment_mm=.00125,
                maximum_increment_mm=.01, output_grid_increment_mm=.01)
            for key, value in (('minimum_increment_mm', float('nan')),
                               ('minimum_increment_mm', .02), ('maximum_increment_mm', .02),
                               ('output_grid_increment_mm', .02), ('mode', 'unknown')):
                with self.subTest(key=key, value=value):
                    bad = copy.deepcopy(report); bad['load_control'][key] = value
                    self.assertFalse(mechanical_completion(bad, root))
            bad = copy.deepcopy(report); bad['steps'][10].pop('friction_work_n_mm')
            self.assertFalse(mechanical_completion(bad, root))
            bad = copy.deepcopy(report); bad['steps'].pop(); bad['last_depth_mm'] = 2.99
            self.assertFalse(mechanical_completion(bad, root))
            bad = copy.deepcopy(report); bad['last_depth_mm'] = float('nan')
            self.assertFalse(mechanical_completion(bad, root))

    def test_external_process_disappearance_and_pid_reuse_finish_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'result.json').write_text(json.dumps(dict(pid=123, status='running')))
            for identity, expected in [(dict(status='missing', pid=123), 'external_process_missing'),
                                       (dict(status='alive', pid=123, birth='new', platform='test'), 'external_pid_reused'),
                                       (dict(status='unavailable', pid=123), 'external_identity_unavailable')]:
                run = dict(output=str(root), pid=123, status='pending',
                           process_identity=dict(status='alive', pid=123, birth='old', platform='test'))
                with mock.patch('tools.run_ipc_queue.process_identity', return_value=identity):
                    observe_existing(run)
                self.assertEqual(run['status'], expected)
                self.assertFalse(run['mechanics_completed'])

    def test_external_wait_finishes_without_signalling_or_editing_external_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); queue = root/'queue'; queue.mkdir()
            external = root/'external'; external.mkdir()
            (external/'result.json').write_text(json.dumps(dict(pid=123, status='running')))
            identity = dict(status='alive', pid=123, birth='old', platform='test')
            run = dict(output=str(external), pid=123, status='pending', process_identity=identity)
            state = dict(jobs=[], observed_runs=[run], source_root=str(root), process_timeout_s=None)
            def finish(_):
                self.accepted_report(external, 123)
            with (mock.patch('tools.run_ipc_queue.process_identity', return_value=identity),
                  mock.patch('tools.run_ipc_queue.time.sleep', side_effect=finish) as sleep,
                  contextlib.redirect_stdout(io.StringIO())):
                run_jobs(queue, state, os.environ.copy())
            sleep.assert_called_once_with(60)
            self.assertEqual(run['status'], 'completed')
            self.assertTrue(state['all_mechanics_completed'])
            self.assertEqual(json.loads((external/'result.json').read_text())['pid'], 123)

    def test_failing_comparison_does_not_stop_case_or_final_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = dict(name='case', output=str(root/'case'), status='pending',
                       command=[sys.executable, '-c', 'print("mechanics invoked")'])
            state = dict(jobs=[job], source_root=str(root), process_timeout_s=10,
                         comparison_command=[sys.executable, '-c', 'import sys; print("plot failure"); sys.exit(4)'])
            with contextlib.redirect_stdout(io.StringIO()):
                run_jobs(root, state, os.environ.copy())
            self.assertEqual(state['status'], 'finished')
            self.assertEqual(job['exit_code'], 0)
            self.assertFalse(state['all_mechanics_completed'])
            self.assertEqual(len(state['comparison_attempts']), 3)
            self.assertTrue(all(v['status'] == 'failed' for v in state['comparison_attempts']))
            self.assertEqual((root/'comparison.log').read_text().count('plot failure'), 3)

    def test_claimed_completion_without_accepted_history_is_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = "import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'result.json').write_text(json.dumps(dict(status='completed',completed=True,last_depth_mm=3.)))"
            job = dict(name='claimed', output=str(root/'claimed'), status='pending',
                       command=[sys.executable, '-c', script, str(root/'claimed')])
            state = dict(jobs=[job], source_root=str(root), process_timeout_s=10)
            with contextlib.redirect_stdout(io.StringIO()):
                run_jobs(root, state, os.environ.copy())
            self.assertEqual(job['exit_code'], 0)
            self.assertEqual(job['status'], 'failed')
            self.assertFalse(job['mechanics_completed'])
            self.assertEqual(state['successful_cases'], 0)
            self.assertFalse(state['all_mechanics_completed'])

    def test_unlimited_continuation_has_no_process_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'queue'; source = Path(temporary)/'prefix'
            state = prepare(root, threads=2, seconds=0, friction_only=True, resume_first=source)
            self.assertIsNone(state['process_timeout_s'])
            self.assertEqual(state['solve_budget_s'],0)
            self.assertIn('--resume',state['jobs'][0]['command'])
            self.assertTrue(all('--resume' not in j['command'] for j in state['jobs'][1:]))
            for job in state['jobs']:
                job['command']=[sys.executable,'-c','print("no deadline")']
            with contextlib.redirect_stdout(io.StringIO()):
                run_jobs(root,state,os.environ.copy())
            self.assertTrue(all('exit_code' in j for j in state['jobs']))

    def test_timeout_is_recorded_and_next_case_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = SYNTHETIC_COMPLETION_SCRIPT+"\nimport pathlib,json,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'result.json').write_text(json.dumps(completed_report(p)))"
            jobs = [dict(name='hang',output=str(root/'hang'),status='pending',
                         command=[sys.executable,'-c','import time; time.sleep(30)']),
                    dict(name='next',output=str(root/'next'),status='pending',
                         command=[sys.executable,'-c',script,str(root/'next')])]
            state = dict(jobs=jobs,source_root=str(root),process_timeout_s=.5)
            with contextlib.redirect_stdout(io.StringIO()):
                run_jobs(root,state,os.environ.copy())
            self.assertEqual([j['status'] for j in jobs],['timed_out','completed'])
            for job, status in zip(jobs, ('wall_budget_exhausted', 'completed')):
                self.assertEqual(job['outcome_status'], status)
                self.assertEqual(json.loads((Path(job['output'])/'outcome.json').read_text(encoding='utf-8'))['status'], status)

    def test_failure_and_missing_or_incomplete_reports_do_not_stop_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = SYNTHETIC_COMPLETION_SCRIPT+"""
import json, pathlib, sys
root, name, code, completed, present = sys.argv[1:]
root = pathlib.Path(root)
with (root/'order.log').open('a') as log: log.write(name+' start\\n')
out = root/name
out.mkdir()
if present == '1':
    report = completed_report(out) if completed == '1' else dict(status='failed', completed=False, last_depth_mm=.1)
    (out/'result.json').write_text(json.dumps(report))
with (root/'order.log').open('a') as log: log.write(name+' end\\n')
sys.exit(int(code))
"""
            jobs = []
            for index, (code, completed, present) in enumerate(((1,0,1),(0,1,1),(0,0,1),(7,0,0),(0,1,1))):
                name = str(index)
                jobs.append(dict(name=name, output=str(root/name), status='pending',
                                 command=[sys.executable, '-c', script, str(root), name,
                                          str(code), str(completed), str(present)]))
            state = dict(jobs=jobs, source_root=str(root), process_timeout_s=10)
            with contextlib.redirect_stdout(io.StringIO()):
                run_jobs(root, state, os.environ.copy())
            self.assertEqual(state['status'], 'finished')
            self.assertEqual(state['successful_cases'], 2)
            self.assertEqual([j['status'] for j in jobs], ['failed','completed','failed','failed','completed'])
            self.assertEqual((root/'order.log').read_text().splitlines(),
                             [f'{i} {stage}' for i in range(5) for stage in ('start','end')])
            self.assertEqual(json.loads((root/'queue.json').read_text())['status'], 'finished')
            self.assertTrue(all((Path(job['output'])/'outcome.md').is_file() for job in jobs))
            self.assertEqual(jobs[3]['outcome_status'], 'execution_error')

    def test_plan_freezes_exact_five_cases_without_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'queue'
            state = prepare(root, threads=2, seconds=21600)
            self.assertEqual([(j['angle'],j['friction']) for j in state['jobs']], list(CASES))
            self.assertEqual(state['process_timeout_s'],23400)
            self.assertTrue((root/'source/prototype/ipc_reference.py').exists())
            self.assertTrue((root/'source/tools/visualize_ipc.py').exists())
            for job in state['jobs']:
                self.assertNotIn('--resume',job['command'])
                self.assertNotIn('--detach',job['command'])
                self.assertNotIn('--no-visualize',job['command'])
            with self.assertRaises(ValueError):
                prepare(root, threads=2, seconds=21600)

    def test_friction_restart_selects_three_cases_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'queue'
            state = prepare(root, threads=2, seconds=21600, friction_only=True)
            self.assertEqual([(j['angle'], j['friction']) for j in state['jobs']],
                             [(60, .35), (30, .35), (0, .35)])
            self.assertTrue(all('--resume' not in j['command'] for j in state['jobs']))
            state['jobs'] = state['jobs'][:1]
            state['jobs'][0]['command'] = [sys.executable, '-c', "print('queue log check')"]
            with contextlib.redirect_stdout(io.StringIO()):
                run_jobs(root, state, os.environ.copy())
            self.assertEqual((root/'01_angle60_mu0.35.stdout.log').read_text().strip(),
                             'queue log check')
