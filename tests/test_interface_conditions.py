"""Scenario semantics and diagnostic integration; no calibrated oil data invented."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from prototype.interface_conditions import interface_condition
from prototype.ipc_reference import parse_args, run, toolkit
from tools.run_ipc_queue import prepare


class InterfaceConditionTests(unittest.TestCase):
    def test_oil_changes_only_effective_friction_and_preserves_missing_data(self):
        dry = interface_condition('dry'); oil = interface_condition('silicone_oil')
        self.assertEqual(dry['friction_coefficient'], .35)
        self.assertEqual(oil['friction_coefficient'], .10)
        self.assertEqual(dry['normal_contact'], oil['normal_contact'])
        self.assertFalse(oil['calibrated'])
        self.assertFalse(oil['film_pressure'])
        self.assertIsNone(oil['oil_kinematic_viscosity_cst'])
        self.assertFalse(oil['material_parameters_changed'])

    def test_temperature_and_viscosity_are_not_silent_friction_multipliers(self):
        cases = [interface_condition('silicone_oil', temperature_c=t, oil_viscosity_cst=v)
                 for t in (15., 25., 35.) for v in (10., 1000.)]
        self.assertTrue(all(q['friction_coefficient'] == .10 for q in cases))
        self.assertTrue(all(q['thermal_coupling'] is False for q in cases))
        self.assertEqual([cases[0]['ambient_temperature_c'], cases[-1]['ambient_temperature_c']], [15., 35.])

    def test_invalid_or_misattributed_inputs_rejected(self):
        for kwargs in (dict(kind='water'), dict(friction=float('nan')), dict(friction=-1),
                       dict(temperature_c=-300), dict(kind='silicone_oil', oil_viscosity_cst=0),
                       dict(kind='silicone_oil', oil_wait_minutes=-1), dict(kind='dry', oil_grade='oil')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                interface_condition(**kwargs)

    def test_cli_keeps_legacy_baseline_and_requires_explicit_onset_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            base = ['--output', str(Path(temp)/'run')]
            self.assertEqual(parse_args(base+['--baseline']).friction, 0.)
            self.assertEqual(parse_args(base+['--interface','dry']).friction, .35)
            a = parse_args(base+['--interface','silicone_oil','--initiation'])
            self.assertEqual(a.friction, .10)
            self.assertTrue(a.initiation)
            for options in (['--baseline','--interface','silicone_oil'],
                            ['--initiation','--volumetric-discretization','macro_mixed'],
                            ['--initiation','--resume',temp]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(base+options)

    def test_three_case_queue_frozen_commands_match_requested_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            state = prepare(Path(temp)/'queue', 2, 0, study='lubrication',
                            oil_grade='declared grade', oil_viscosity_cst=100)
            self.assertEqual([(j['angle'],j['interface'],j['friction']) for j in state['jobs']],
                             [(60,'dry',.35),(60,'silicone_oil',.10),(30,'silicone_oil',.10)])
            self.assertIsNone(state['process_timeout_s'])
            self.assertTrue(Path(state['source_root'],'prototype/interface_conditions.py').is_file())
            for j in state['jobs']:
                cmd=j['command']
                self.assertEqual(cmd[cmd.index('--depth')+1], '3')
                self.assertEqual(cmd[cmd.index('--tip-radius')+1], '.02')
                self.assertIn('--initiation',cmd)
                self.assertIn('--predictor',cmd)
                self.assertEqual('--oil-grade' in cmd, j['interface']=='silicone_oil')
                self.assertNotIn('prototype.ipc_cutting',cmd)


class OnlineInitiationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: toolkit()
        except (ImportError, RuntimeError) as exc:raise unittest.SkipTest(str(exc))

    def test_diagnostic_trigger_cannot_stop_or_change_intact_mechanics(self):
        """A forced diagnostic event affects reporting, never the solved path."""
        from prototype.ipc_initiation import IntactInitiationMonitor
        class ForcedEventMonitor(IntactInitiationMonitor):
            def observe(self, depth_mm, positions, body_nodes, *, checkpoint=None):
                row = super().observe(depth_mm, positions, body_nodes, checkpoint=checkpoint)
                self.report['primary_event'] = dict(depth_bracket_mm=[0.,0.], test_event=True)
                return row
        with tempfile.TemporaryDirectory() as temp:
            common=['--case','patch','--depth','.02','--increment','.01',
                    '--interface','silicone_oil','--no-visualize','--seconds','30','--threads','1',
                    '--predictor','secant','--friction-linear-solver','retry-gmres']
            a=Path(temp)/'plain';b=Path(temp)/'diagnostic'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(run(parse_args(['--output',str(a)]+common)))
                with patch('prototype.ipc_initiation.IntactInitiationMonitor',ForcedEventMonitor):
                    self.assertTrue(run(parse_args(['--output',str(b)]+common+['--initiation'])))
            before=json.loads((a/'result.json').read_text())
            after=json.loads((b/'result.json').read_text())
            self.assertEqual(after['last_depth_mm'], .02)
            self.assertFalse(after['simplifications']['fracture'])
            self.assertEqual(after['initiation']['accepted_states_analyzed'], 3)
            self.assertTrue(after['initiation']['primary_event']['test_event'])
            for left,right in zip(before['steps'],after['steps']):
                for key in ('depth_mm','reaction_fx_n','reaction_fz_n','residual_n','min_j'):
                    self.assertAlmostEqual(left[key],right[key],delta=1e-10)
            with np.load(a/'states/0002.npz') as aa, np.load(b/'states/0002.npz') as bb:
                np.testing.assert_array_equal(aa['positions'],bb['positions'])

    def test_diagnostic_failure_is_reported_but_mechanics_keeps_running(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'case'
            args=parse_args(['--output',str(folder),'--case','patch','--depth','.02',
                            '--increment','.01','--initiation','--interface','silicone_oil',
                            '--no-visualize','--seconds','30','--threads','1'])
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'prototype.ipc_initiation.IntactInitiationMonitor.observe',side_effect=RuntimeError('diagnostic only')):
                self.assertTrue(run(args))
            result=json.loads((folder/'result.json').read_text())
            self.assertEqual(result['last_depth_mm'],.02)
            self.assertTrue(result['completed'])
            self.assertEqual(result['initiation']['status'],'failed')
            self.assertFalse(json.loads((folder/'initiation/result.json').read_text())['completed'])

    def test_diagnostic_initialization_and_finalization_failures_are_isolated(self):
        for method in ('__init__', 'finish'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                folder=Path(temp)/'case'
                args=parse_args(['--output',str(folder),'--case','patch','--depth','.01',
                                '--increment','.01','--initiation','--interface','silicone_oil',
                                '--no-visualize','--seconds','30','--threads','1'])
                with contextlib.redirect_stdout(io.StringIO()), patch(
                        'prototype.ipc_initiation.IntactInitiationMonitor.'+method,
                        side_effect=RuntimeError('diagnostic lifecycle failure')):
                    self.assertTrue(run(args))
                result=json.loads((folder/'result.json').read_text())
                self.assertTrue(result['completed'])
                self.assertEqual(result['last_depth_mm'],.01)
                self.assertEqual(result['initiation']['status'],'failed')
                self.assertIn('diagnostic lifecycle failure',result['initiation']['source_error'])


if __name__=='__main__': unittest.main()
