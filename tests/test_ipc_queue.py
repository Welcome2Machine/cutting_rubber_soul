"""Queue lifecycle checks without launching finite-element solves."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from tools.run_ipc_queue import CASES, prepare, run_jobs


class IPCQueueTests(unittest.TestCase):
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
            script = "import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'result.json').write_text(json.dumps(dict(status='completed',completed=True)))"
            jobs = [dict(name='hang',output=str(root/'hang'),status='pending',
                         command=[sys.executable,'-c','import time; time.sleep(30)']),
                    dict(name='next',output=str(root/'next'),status='pending',
                         command=[sys.executable,'-c',script,str(root/'next')])]
            state = dict(jobs=jobs,source_root=str(root),process_timeout_s=.5)
            with contextlib.redirect_stdout(io.StringIO()):
                run_jobs(root,state,os.environ.copy())
            self.assertEqual([j['status'] for j in jobs],['timed_out','completed'])

    def test_failure_and_missing_or_incomplete_reports_do_not_stop_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = """
import json, pathlib, sys
root, name, code, completed, present = sys.argv[1:]
root = pathlib.Path(root)
with (root/'order.log').open('a') as log: log.write(name+' start\\n')
out = root/name
out.mkdir()
if present == '1':
    (out/'result.json').write_text(json.dumps(dict(status='completed' if completed=='1' else 'failed', completed=completed=='1', last_depth_mm=3.0 if completed=='1' else .1)))
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
