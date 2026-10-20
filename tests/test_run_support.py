"""Output integrity shared by the cutting and paper runners."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from run_support import ROOT,write_json,save_npz,source_hashes,compute_process_policy
from run_support import background_child_options,classify_process_exit


class ChildProcessTests(unittest.TestCase):
    def test_control_exit_is_not_a_solver_failure_or_timeout(self):
        for code in (3221225786,-1073741510):
            self.assertEqual(classify_process_exit(code),'interrupted_control_event')
            self.assertEqual(classify_process_exit(code,supervisor_timeout=True),'supervisor_timeout')
        self.assertEqual(classify_process_exit(0),'exited_normally')
        self.assertEqual(classify_process_exit(None),'running')
        for code in (1,139,0xc0000005):self.assertEqual(classify_process_exit(code),'process_failed')

    def test_real_background_child_preserves_output_without_console(self):
        import subprocess,sys
        code="import sys; print('compute-child'); print('console='+str(sys.platform))"
        if sys.platform=='win32':
            code="import ctypes; k=ctypes.WinDLL('kernel32'); k.GetConsoleWindow.restype=ctypes.c_void_p; print('compute-child'); print('console='+str(bool(k.GetConsoleWindow())))"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=15,
                              **background_child_options())
        self.assertEqual(result.returncode,0)
        self.assertIn('compute-child',result.stdout)
        if sys.platform=='win32':self.assertIn('console=False',result.stdout)

    def test_non_windows_uses_new_session(self):
        with patch('run_support.sys.platform','linux'):
            self.assertEqual(background_child_options(),{'start_new_session':True})


class RunSupportTests(unittest.TestCase):
    def test_invalid_json_does_not_replace_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'status.json'
            write_json(path,{'state':'计算中'})
            before=path.read_bytes()
            with self.assertRaises(ValueError):write_json(path,{'residual':float('nan')})
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),{'state':'计算中'})
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_checkpoint_replacement_is_readable_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.npz'
            save_npz(path,u=np.zeros(3));save_npz(path,u=np.arange(4))
            with np.load(path) as state:np.testing.assert_array_equal(state['u'],np.arange(4))
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_source_snapshot_preserves_layout_and_shared_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            hashes=source_hashes(('mechanics',),snapshot=directory)
            self.assertEqual(hashes,source_hashes(('mechanics',)))
            self.assertIn('run_support.py',hashes)
            self.assertIn('mechanics/finite_strain.py',hashes)
            for name in hashes:self.assertEqual((Path(directory)/name).read_bytes(),(ROOT/name).read_bytes())


class FakeProcessPolicy:
    def __init__(self, control=6, state=4):
        self.state = dict(version=1, control_mask=control, state_mask=state)
        self.original = self.state.copy()
        self.writes = []
        self.read_count = 0
        self.fail_read = set()
        self.fail_write = set()

    def read(self):
        self.read_count += 1
        if self.read_count in self.fail_read:
            raise OSError('read unavailable')
        return self.state.copy()

    def write(self, value):
        self.writes.append(value.copy())
        if len(self.writes) in self.fail_write:
            raise OSError('write unavailable')
        self.state = value.copy()


class ComputeProcessPolicyTests(unittest.TestCase):
    def test_compute_preserves_other_flags_and_restores_exact_original(self):
        api = FakeProcessPolicy(control=6, state=5)
        with patch('run_support.sys.platform', 'win32'), patch(
                'run_support._WindowsProcessPowerPolicy', return_value=api):
            with compute_process_policy() as metadata:
                self.assertTrue(metadata['applied'])
                self.assertIsNone(metadata['restored'])
                self.assertEqual(metadata['previous'], api.original)
                self.assertEqual(metadata['active'], dict(version=1, control_mask=7, state_mask=4))
                self.assertEqual(api.state, metadata['active'])
            self.assertTrue(metadata['restored'])
            self.assertEqual(api.state, api.original)
            self.assertEqual(api.writes, [metadata['active'], api.original])
            self.assertIsNone(metadata['error'])

    def test_exception_from_computation_is_preserved_after_restoration(self):
        api = FakeProcessPolicy()
        failure = RuntimeError('solver failed')
        with patch('run_support.sys.platform', 'win32'), patch(
                'run_support._WindowsProcessPowerPolicy', return_value=api):
            with self.assertRaises(RuntimeError) as caught:
                with compute_process_policy() as metadata:
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertTrue(metadata['restored'])
        self.assertEqual(api.state, api.original)

    def test_restore_failure_does_not_mask_solver_exception(self):
        api = FakeProcessPolicy()
        api.fail_write = {2}
        failure = ValueError('numerical failure')
        with patch('run_support.sys.platform', 'win32'), patch(
                'run_support._WindowsProcessPowerPolicy', return_value=api):
            with self.assertRaises(ValueError) as caught:
                with compute_process_policy() as metadata:
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertFalse(metadata['restored'])
        self.assertIn('restore: OSError', metadata['error'])

    def test_system_and_non_windows_never_touch_api(self):
        for platform, mode in [('win32', 'system'), ('linux', 'compute'), ('darwin', 'system')]:
            with self.subTest(platform=platform, mode=mode), patch(
                    'run_support.sys.platform', platform), patch(
                    'run_support._WindowsProcessPowerPolicy') as constructor:
                with compute_process_policy(mode) as metadata:
                    self.assertFalse(metadata['applied'])
                constructor.assert_not_called()
                self.assertIsNone(metadata['restored'])
                self.assertIsNone(metadata['error'])

    def test_missing_api_is_reported_without_stopping_computation(self):
        with patch('run_support.sys.platform', 'win32'), patch(
                'run_support._WindowsProcessPowerPolicy', side_effect=AttributeError('not available')):
            with compute_process_policy() as metadata:
                self.assertFalse(metadata['applied'])
                self.assertIn('apply: AttributeError', metadata['error'])
        self.assertIsNone(metadata['previous'])
        self.assertIsNone(metadata['restored'])

    def test_failed_initial_read_never_attempts_write(self):
        api = FakeProcessPolicy()
        api.fail_read = {1}
        with patch('run_support.sys.platform', 'win32'), patch(
                'run_support._WindowsProcessPowerPolicy', return_value=api):
            with compute_process_policy() as metadata:
                self.assertFalse(metadata['applied'])
        self.assertEqual(api.writes, [])
        self.assertIn('read unavailable', metadata['error'])

    def test_failed_set_rolls_back_before_computation(self):
        api = FakeProcessPolicy()
        api.fail_write = {1}
        with patch('run_support.sys.platform', 'win32'), patch(
                'run_support._WindowsProcessPowerPolicy', return_value=api):
            with compute_process_policy() as metadata:
                self.assertFalse(metadata['applied'])
                self.assertTrue(metadata['restored'])
                self.assertEqual(api.state, api.original)
        self.assertEqual(len(api.writes), 2)
        self.assertIn('write unavailable', metadata['error'])

    def test_failed_apply_readback_rolls_back_before_computation(self):
        api = FakeProcessPolicy()
        api.fail_read = {2}
        with patch('run_support.sys.platform', 'win32'), patch(
                'run_support._WindowsProcessPowerPolicy', return_value=api):
            with compute_process_policy() as metadata:
                self.assertFalse(metadata['applied'])
                self.assertTrue(metadata['restored'])
                self.assertEqual(metadata['active'], api.original)
                self.assertEqual(api.state, api.original)

    def test_setup_interrupt_restores_policy_and_propagates_unchanged(self):
        for exception_type in (KeyboardInterrupt, SystemExit):
            for stage in ('write', 'readback'):
                with self.subTest(exception=exception_type, stage=stage):
                    api = FakeProcessPolicy()
                    failure = exception_type('interrupted during setup')
                    if stage == 'write':
                        original_write = api.write

                        def interrupted_write(value):
                            original_write(value)
                            if len(api.writes) == 1:
                                raise failure

                        api.write = interrupted_write
                    else:
                        original_read = api.read

                        def interrupted_read():
                            value = original_read()
                            if api.read_count == 2:
                                raise failure
                            return value

                        api.read = interrupted_read
                    with patch('run_support.sys.platform', 'win32'), patch(
                            'run_support._WindowsProcessPowerPolicy', return_value=api):
                        with self.assertRaises(exception_type) as caught:
                            with compute_process_policy():
                                self.fail('interrupted setup must not enter computation')
                    self.assertIs(caught.exception, failure)
                    self.assertEqual(api.state, api.original)
                    self.assertEqual(api.writes[-1], api.original)

    def test_bad_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            with compute_process_policy('invalid'):
                self.fail('invalid policy must not enter the computation')
