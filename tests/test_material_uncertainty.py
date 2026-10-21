"""Diagnostic provenance and identical physical iterations with instrumentation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import eye
from prototype.ipc_reference import IPCProblem, LoadStepFailure
from tools.run_material_uncertainty import direction_trace, jobs, GROUPS, window_metrics


class MaterialUncertaintyTests(unittest.TestCase):
    def test_common_parameters_preserve_pairing_and_exclude_unqualified57(self):
        tasks=jobs()
        self.assertEqual(len(tasks),4)
        self.assertEqual({t['angle'] for t in tasks},{0,33})
        for group in GROUPS:
            pair=[t for t in tasks if t['group']==group]
            self.assertEqual(pair[0]['parameters'],pair[1]['parameters'])
            self.assertAlmostEqual(sum(pair[0]['parameters'][:2]),1.38*.7)
            self.assertEqual(pair[0]['parameters'][2],1.7)

    def test_raw_force_offset_is_preserved_and_short_paths_not_extrapolated(self):
        d=np.arange(51)/100;f=10*d+.51
        self.assertIsNone(window_metrics(d[:-1],f[:-1],d,f))
        r=window_metrics(d,f,d,10*d)
        self.assertAlmostEqual(r['rmse_n'],.51)
        np.testing.assert_allclose(r['fe_secants_n_mm'],[10.,10.])

    @staticmethod
    def fixture(safe):
        p=object.__new__(IPCProblem);p.max_iterations=4;p.stagnation_window=0
        p.linear_solver='direct';p.enrichment=None;p.nodal_volume=None
        p.free=np.arange(2);p.total=2;p.options=SimpleNamespace(tip_radius_mm=1000.)
        p.friction_globalization='model-error';p.lag=lambda x:None;p.safe_step=lambda x,d:safe
        p.coupled_friction_tangent=lambda x,lag,H:eye(2,format='csc')
        p.previous=np.zeros((1,2));p.previous_material_metric=np.eye(3)[None]
        p.material_dt=.05
        def evaluate(x,lag=None,tangent=True,**kw):
            r=(0.,np.array([1.,0.])+x.ravel(),eye(2,format='csc'),1.)
            return r+(eye(2,format='csc'),) if kw.get('with_metric') else r
        p.evaluate=evaluate;return p

    def test_trace_keeps_actual_iterations_identical_and_records_used_next_shift(self):
        for safe in (1.,1e-4):
            before=sys.gettrace();plain=self.fixture(safe);traced=self.fixture(safe)
            with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
                def run(p):
                    try:return p.solve_friction(np.zeros((1,2)),.01,time.monotonic()+10)[0]
                    except LoadStepFailure:return p.trial.copy()
                expected=run(plain)
                with direction_trace(Path(temp),{'name':'case'}):actual=run(traced)
                np.testing.assert_array_equal(actual,expected)
                self.assertIs(sys.gettrace(),before)
                rows=[json.loads(s) for s in (Path(temp)/'case.ptc.jsonl').read_text().splitlines()]
                self.assertGreater(len(rows),0)
                self.assertEqual(len(rows),len({(r['depth_mm'],r['iteration'],r['retry'],r['accepted']) for r in rows}))
                self.assertAlmostEqual(rows[0]['shift_used'],.1)
                for left,right in zip(rows,rows[1:]):self.assertAlmostEqual(left['shift_next'],right['shift_used'])
                for r in rows:
                    self.assertAlmostEqual(r['predicted_norm_n'],r['trial_norm_n'],places=12)
                with np.load(Path(temp)/'case.last_direction.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['previous_positions'],np.zeros((1,2)))


if __name__=='__main__':unittest.main()
