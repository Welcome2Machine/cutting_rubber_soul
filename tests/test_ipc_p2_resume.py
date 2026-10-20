"""Native small-patch checks for preserving accepted P2 continuation history."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.ipc_reference import IPCProblem, parse_args, restore_accepted_prefix, run, toolkit


class P2ResumeTests(unittest.TestCase):
    common = [
        '--case', 'patch', '--element', 'p2-interior', '--no-visualize',
        '--friction', '.35', '--local-spacing', '.2', '--surface-spacing', '.2',
        '--increment', '.01', '--seconds', '30', '--threads', '1',
        '--hessian-mode', 'regularized', '--linear-solver', 'direct',
    ]

    @classmethod
    def setUpClass(cls):
        try:
            toolkit().set_num_threads(1)
        except ModuleNotFoundError:
            raise unittest.SkipTest('Optional ipctk==1.6.0 not installed')
        cls.folder = tempfile.TemporaryDirectory(prefix='ipc-p2-resume-')
        cls.addClassCleanup(cls.folder.cleanup)
        cls.root = Path(cls.folder.name)
        cls.source = cls.root / 'prefix'
        args = parse_args(cls.common + ['--output', str(cls.source), '--depth', '.01'])
        with contextlib.redirect_stdout(io.StringIO()):
            completed = run(args)
        if not completed:
            raise AssertionError((cls.source / 'result.json').read_text(encoding='utf-8'))

    def setUp(self):
        self.threads = threadpool_limits(limits=1)
        self.threads.__enter__()
        self.addCleanup(self.threads.__exit__, None, None, None)
        self.work = tempfile.TemporaryDirectory(prefix='check-', dir=self.root)
        self.addCleanup(self.work.cleanup)
        self.work_path = Path(self.work.name)

    def problem(self, element='p2-interior', linear_solver='direct'):
        return IPCProblem('patch', mu=.35, local_spacing=.2, surface_spacing=.2,
                          hessian_mode='regularized', element=element,
                          linear_solver=linear_solver)

    def restore(self, source=None, report=None, problem=None, name='restored'):
        source = self.source if source is None else source
        report = (json.loads((source / 'result.json').read_text(encoding='utf-8'))
                  if report is None else report)
        problem = self.problem() if problem is None else problem
        report.update(steps=[], completed=False, status='running', target_mm=.02)
        out = self.work_path / name
        (out / 'states').mkdir(parents=True)
        x, depth = restore_accepted_prefix(source, out, report, problem,
                                            np.array([0., .01, .02]))
        return x, depth, report, problem, out

    def copied_prefix(self, name):
        destination = self.work_path / name
        (destination / 'states').mkdir(parents=True)
        shutil.copyfile(self.source / 'result.json', destination / 'result.json')
        for state in (self.source / 'states').glob('*.npz'):
            shutil.copyfile(state, destination / 'states' / state.name)
        return destination

    def test_restore_reassembles_with_saved_previous_and_copies_exact_states(self):
        p = self.problem()
        with np.load(self.source / 'states/0001.npz', allow_pickle=False) as state:
            positions = state['positions'].copy()
            previous = state['previous_positions'].copy()
        observed_previous = []
        evaluate = p.evaluate

        def checked_evaluate(*args, **kwargs):
            observed_previous.append(p.previous.copy())
            return evaluate(*args, **kwargs)

        with patch.object(p, 'evaluate', side_effect=checked_evaluate):
            x, depth, report, _, out = self.restore(problem=p)
        self.assertEqual(len(observed_previous), 1)
        np.testing.assert_array_equal(observed_previous[0], previous)
        np.testing.assert_array_equal(p.previous, previous)
        np.testing.assert_array_equal(x, positions)
        self.assertGreater(np.linalg.norm(x - previous), 0.)
        self.assertEqual(depth, .01)
        self.assertEqual(report['resume']['accepted_states'], 2)
        self.assertEqual(report['last_checkpoint'], 'states/0001.npz')
        for relative, fingerprint in report['resume']['checkpoints_sha256'].items():
            self.assertEqual((out / relative).read_bytes(), (self.source / relative).read_bytes())
            self.assertEqual(fingerprint, hashlib.sha256((out / relative).read_bytes()).hexdigest())
        accepted = p.acceptance(x, depth, 0, 0)
        self.assertLessEqual(accepted['residual_n'], report['force_tolerance_n'])
        self.assertGreater(accepted['min_j'], 0.)
        self.assertAlmostEqual(accepted['reaction_fz_n'], report['steps'][-1]['reaction_fz_n'],
                               delta=1e-10)

    def test_resumed_p2_loading_matches_continuous_native_run(self):
        resumed, continuous = self.work_path / 'resumed', self.work_path / 'continuous'
        for out, extra in ((resumed, ['--resume', str(self.source)]), (continuous, [])):
            args = parse_args(self.common + ['--output', str(out), '--depth', '.02'] + extra)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(run(args))
        for index in range(3):
            relative = Path('states') / f'{index:04d}.npz'
            with np.load(resumed / relative, allow_pickle=False) as a, np.load(continuous / relative, allow_pickle=False) as b:
                for key in ('positions', 'previous_positions', 'p2_elements', 'p2_boundary_slots',
                            'p2_base_nodes', 'p2_quadrature_order'):
                    np.testing.assert_array_equal(a[key], b[key])

    def test_direct_prefix_can_resume_with_reused_lu_and_keeps_physical_result(self):
        report = json.loads((self.source / 'result.json').read_text(encoding='utf-8'))
        report['linear_solver'] = 'reused-lu'
        p = self.problem(linear_solver='reused-lu')
        _, _, restored, _, _ = self.restore(report=report, problem=p)
        self.assertEqual(p.linear_solver, 'reused-lu')
        self.assertEqual(restored['resume']['previous_linear_solver'], 'direct')
        paths = {}
        for solver in ('direct', 'reused-lu'):
            out = self.work_path / solver
            args = parse_args(self.common + ['--linear-solver', solver, '--output', str(out),
                                             '--depth', '.02', '--resume', str(self.source)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(run(args))
            paths[solver] = out
        with np.load(paths['direct'] / 'states/0002.npz', allow_pickle=False) as a, np.load(paths['reused-lu'] / 'states/0002.npz', allow_pickle=False) as b:
            np.testing.assert_array_equal(a['previous_positions'], b['previous_positions'])
            np.testing.assert_allclose(a['positions'], b['positions'], rtol=0, atol=1e-9)
        direct = json.loads((paths['direct'] / 'result.json').read_text(encoding='utf-8'))
        reused = json.loads((paths['reused-lu'] / 'result.json').read_text(encoding='utf-8'))
        self.assertEqual(reused['resume']['previous_linear_solver'], 'direct')
        self.assertLessEqual(reused['steps'][-1]['residual_n'], reused['force_tolerance_n'])
        self.assertAlmostEqual(direct['steps'][-1]['reaction_fz_n'],
                               reused['steps'][-1]['reaction_fz_n'], delta=1e-7)

    def test_element_and_quadrature_report_mismatch_are_rejected(self):
        for name, changed in (('p1', None), ('quadrature', {'quadrature_order': 5})):
            with self.subTest(name=name):
                report = json.loads((self.source / 'result.json').read_text(encoding='utf-8'))
                if changed is None:
                    report.pop('discretization')
                else:
                    report['discretization'].update(changed)
                with self.assertRaisesRegex(ValueError, 'Resume mismatch: displacement discretization'):
                    self.restore(report=report, name=name)
        # A real P1 invocation must also fail before attempting to use P2 states.
        args = parse_args(self.common + ['--element', 'p1', '--output', str(self.work_path / 'p1-run'),
                                         '--depth', '.02', '--resume', str(self.source)])
        with self.assertRaisesRegex(ValueError, 'Resume mismatch: simplifications'):
            run(args)

    def test_each_p2_checkpoint_metadata_change_is_rejected(self):
        keys = ('p2_elements', 'p2_boundary_slots', 'p2_base_nodes', 'p2_quadrature_order')
        for key in keys:
            with self.subTest(key=key):
                source = self.copied_prefix(key)
                path = source / 'states/0001.npz'
                with np.load(path, allow_pickle=False) as state:
                    data = {name: state[name].copy() for name in state.files}
                if data[key].dtype == np.bool_:
                    data[key].flat[0] = not data[key].flat[0]
                else:
                    data[key].flat[0] += 1
                np.savez_compressed(path, **data)
                with self.assertRaises(AssertionError):
                    self.restore(source=source, name=key + '-rejected')

    def test_changed_previous_history_is_rejected(self):
        source = self.copied_prefix('history')
        path = source / 'states/0001.npz'
        with np.load(path, allow_pickle=False) as state:
            data = {name: state[name].copy() for name in state.files}
        data['previous_positions'][0, 1] += 1e-4
        np.savez_compressed(path, **data)
        with self.assertRaises(AssertionError):
            self.restore(source=source)

    def test_reported_acceptance_does_not_hide_an_unbalanced_p2_state(self):
        source = self.copied_prefix('unbalanced')
        path = source / 'states/0001.npz'
        with np.load(path, allow_pickle=False) as state:
            data = {name: state[name].copy() for name in state.files}
        # Alter an independent interior midpoint, leaving the recorded residual,
        # reference mesh, quadrature and preceding accepted state untouched.
        data['positions'][int(data['p2_base_nodes']), 1] += 1e-4
        np.savez_compressed(path, **data)
        with self.assertRaisesRegex(ValueError, 'failed independent equilibrium/geometry check'):
            self.restore(source=source)

    def test_fixed_body_and_prescribed_blade_positions_cannot_be_changed(self):
        p = self.problem()
        for name, node, component in (('bottom', int(p.mesh.bottom_nodes[0]), 1),
                                      ('blade-horizontal', p.n, 0),
                                      ('blade-depth', p.n, 1)):
            with self.subTest(name=name):
                source = self.copied_prefix(name)
                path = source / 'states/0001.npz'
                with np.load(path, allow_pickle=False) as state:
                    data = {key: state[key].copy() for key in state.files}
                data['positions'][node, component] += 1e-4
                np.savez_compressed(path, **data)
                with self.assertRaises(AssertionError):
                    self.restore(source=source, name=name + '-rejected')

    def test_nonfinite_and_invalid_acceptance_metadata_are_rejected(self):
        fields = ('depth_mm', 'residual_n', 'min_j', 'minimum_distance_mm',
                  'reaction_fz_n', 'friction_work_n_mm')
        invalid = [(key, float('nan')) for key in fields]
        invalid += [('residual_n', -1e-8), ('min_j', 0.),
                    ('minimum_distance_mm', 0.), ('friction_work_n_mm', -1e-6)]
        for index, (key, value) in enumerate(invalid):
            with self.subTest(key=key, value=value):
                source = self.copied_prefix(f'metadata-{index}')
                path = source / 'result.json'
                report = json.loads(path.read_text(encoding='utf-8'))
                report['steps'][-1][key] = value
                path.write_text(json.dumps(report), encoding='utf-8')
                # Invalid depth can be rejected by the preceding depth-schedule
                # check; other metadata must fail acceptance verification.
                expected = (AssertionError, ValueError) if key == 'depth_mm' else ValueError
                with self.assertRaises(expected):
                    self.restore(source=source, name=f'metadata-{index}-rejected')


if __name__ == '__main__':
    unittest.main()
