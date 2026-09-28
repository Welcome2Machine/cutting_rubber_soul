"""Analytical/objectivity and saved-history contracts for initiation screening."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from prototype.fracture_criteria import principal_tensile_measures
from prototype.ipc_initiation import InitiationScreen, DirectionalOnset, IntactInitiationMonitor, first_event, scan
from prototype.material import response
from prototype.model_config import FEMConfig
from run_support import source_hashes, save_npz, write_json


class InitiationTests(unittest.TestCase):
    def test_online_onset_locks_event_continues_and_never_mutates_mechanics(self):
        X = np.array([[0., 0.], [1., 0.], [0., 1.], [4., 0.], [5., 0.], [4., 1.]])
        tri = np.array([[0, 1, 2], [3, 4, 5]])
        config = FEMConfig(shear_modulus_n_mm2=66., inclination_deg=0., speed_mm_s=.25)
        with tempfile.TemporaryDirectory() as tmp:
            monitor = IntactInitiationMonitor(X, tri, config, Path(tmp)/'initiation')
            snapshots = []
            for i, stretch in enumerate([1., 1.2]):
                body = X.copy()
                # Load only the remote element; no blade-centred screening mask.
                body[3:] = X[3:]@np.diag([stretch, 1/stretch])
                x = np.vstack((body, [[0., 1.], [-1., 2.], [1., 2.]]))
                before = x.copy()
                row = monitor.observe(.01*i, x, 6, checkpoint=f'states/{i:04d}.npz')
                np.testing.assert_array_equal(x, before)
                snapshots.append(row)
            key = 'directional_at1_local_stability'
            event = monitor.summary()['primary_event']
            self.assertEqual(event['depth_bracket_mm'], [0., .01])
            self.assertEqual(event['model_time_bracket_s'], [0., .04])
            self.assertIn(event['evidence']['peak_node'], [3, 4, 5])
            self.assertTrue((monitor.output/event['fields_file']).exists())
            self.assertFalse(snapshots[0]['directional_orientation_search_performed'])
            self.assertTrue(snapshots[1]['directional_orientation_search_performed'])
            # After onset, a different element can become the stress peak. The
            # first event must remain fixed and orientation work must cease.
            body[:3] = X[:3]@np.diag([1.5, 1/1.5])
            x = np.vstack((body, [[0., 1.], [-1., 2.], [1., 2.]])); before = x.copy()
            with patch.object(monitor.onset, 'evaluate', side_effect=AssertionError('redundant direction work')):
                last = monitor.observe(.02, x, 6, checkpoint='states/0002.npz')
            np.testing.assert_array_equal(x, before)
            np.testing.assert_array_equal(monitor.screen.reference, X)
            self.assertEqual(last['criteria']['nominal_local']['element'], 0)
            self.assertIsNone(last['directional_onset'])
            self.assertFalse(last['directional_evaluation_performed'])
            self.assertTrue(last['directional_search_retired'])
            self.assertEqual(last['branch'], 'intact_counterfactual_after_candidate')
            self.assertEqual(monitor.summary()['primary_event'], event)
            self.assertEqual(monitor.report['events'][key], event)
            # Returned summaries are copies; callers cannot move a saved event.
            detached = monitor.summary(); detached['primary_event']['depth_bracket_mm'][1] = 99.
            last['criteria']['nominal_local']['element'] = 999
            summary = monitor.finish('completed')
            report = json.loads((monitor.output/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['accepted_states_analyzed'], 3)
            self.assertEqual(summary['primary_event'], event)
            self.assertEqual(report['steps'][-1]['criteria']['nominal_local']['element'], 0)
            self.assertTrue(report['completed'])
            self.assertFalse(report['physical_nucleation_validated'])
            self.assertFalse(report['mechanics_feedback'])
            self.assertFalse(report['fracture_evolution'])

    def test_online_monitor_rejects_unsupported_fields_and_nonmonotone_history(self):
        X = np.array([[0., 0.], [1., 0.], [0., 1.]])
        x = np.vstack((X, [[0., 1.], [-1., 2.], [1., 2.]]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'mixed pressure'):
                IntactInitiationMonitor(X, [[0, 1, 2]], FEMConfig(), Path(tmp)/'bad',
                                        volumetric_discretization='macro_mixed')
            monitor = IntactInitiationMonitor(X, [[0, 1, 2]], FEMConfig(), Path(tmp)/'ok',
                                            directional_onset=False)
            monitor.observe(.1, x, 3)
            for depth in (.1, .05, -1., np.nan):
                with self.assertRaisesRegex(ValueError, 'strictly increasing'):
                    monitor.observe(depth, x, 3)
            self.assertEqual(len(monitor.report['steps']), 1)
            self.assertIsNone(monitor.summary()['primary_event'])
            monitor.finish('failed', 'mechanics failed before next accepted state')
            self.assertFalse(monitor.report['completed'])
            with self.assertRaisesRegex(ValueError, 'finished'):
                monitor.observe(.2, x, 3)

    def test_directional_onset_uses_full_domain_p1_variation(self):
        X = np.array([[0., 0.], [1., 0.], [0., 1.], [4., 0.], [5., 0.], [4., 1.]])
        tri = np.array([[0, 1, 2], [3, 4, 5]])
        screen = InitiationScreen(X, tri, FEMConfig(inclination_deg=0), radii=())
        onset = DirectionalOnset(screen)
        x = X.copy(); x[3:] = X[3:]@np.diag([8., 1/8.])
        positions = np.vstack((x, [[0., 1.], [-1., 2.], [1., 2.]]))
        evidence, Y, ratio = onset.evaluate(screen.fields(x), positions, 6)
        exact = onset.phase.terms(np.zeros(6), Y, tangent=False)[1]/onset.phase.lumped
        np.testing.assert_allclose(exact, 3*onset.phase.gc/(8*onset.phase.ell)*(1-ratio), atol=1e-10)
        self.assertTrue(evidence['admissible_descent'])
        self.assertFalse(evidence['ratio_is_upper_bound'])
        self.assertIn(evidence['peak_node'], [3, 4, 5])
        self.assertGreater(evidence['offset_from_tip_xz_mm'][0], 20.)

    def test_directional_safe_bound_cannot_trigger_an_event(self):
        X = np.array([[0., 0.], [1., 0.], [0., 1.]])
        screen = InitiationScreen(X, [[0, 1, 2]], FEMConfig(), radii=())
        positions = np.vstack((X, [[0., 1.], [-1., 2.], [1., 2.]]))
        evidence, _, ratio = DirectionalOnset(screen).evaluate(screen.fields(X), positions, 3)
        self.assertTrue(evidence['ratio_is_upper_bound'])
        self.assertFalse(evidence['admissible_descent'])
        self.assertIsNone(evidence['reference_normal_xz'])
        np.testing.assert_allclose(ratio, 0., atol=1e-12)

    def test_nominal_traction_and_normals_are_objective(self):
        F = np.diag([4.2, 4.2**-.5, 4.2**-.5])
        sigma = np.diag([4.4*4.2, 0., 0.])
        a = .64
        Q = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
        f = principal_tensile_measures(F, sigma)
        g = principal_tensile_measures(Q@F, Q@sigma@Q.T)
        self.assertAlmostEqual(float(f['nominal_tensile_mpa']), 4.4)
        self.assertAlmostEqual(float(g['nominal_tensile_mpa']), 4.4)
        np.testing.assert_allclose(np.outer(g['current_normal'], g['current_normal']),
                                   Q@np.outer(f['current_normal'], f['current_normal'])@Q.T, atol=1e-12)
        np.testing.assert_allclose(np.outer(g['reference_normal'], g['reference_normal']),
                                   np.outer(f['reference_normal'], f['reference_normal']), atol=1e-12)
        self.assertTrue(f['direction_unique'])

    def test_hydrostatic_ties_and_compression_do_not_invent_direction(self):
        f = principal_tensile_measures(np.tile(np.eye(3), (2, 1, 1)),
                                      np.array([np.eye(3)*3, -np.eye(3)*3]))
        np.testing.assert_allclose(f['nominal_tensile_mpa'], [3, 0])
        self.assertFalse(f['direction_unique'].any())
        with self.assertRaises(ValueError):
            principal_tensile_measures(np.diag([-1., 1., 1.]), np.eye(3))

    def test_whole_domain_peak_can_be_away_from_blade_and_out_of_plane(self):
        X = np.array([[0., 0.], [1., 0.], [0., 1.], [4., 0.], [5., 0.], [4., 1.]])
        tri = np.array([[0, 1, 2], [3, 4, 5]])
        screen = InitiationScreen(X, tri, FEMConfig(inclination_deg=0), radii=(.1, 10.))
        for A in screen.averages.values():
            np.testing.assert_allclose(A@np.ones(2), 1.)
        x = X.copy(); x[4, 0] = 7.
        fields = screen.fields(x)
        positions = np.vstack((x, [[0., 1.], [-1., 2.], [1., 2.]]))
        e = screen.summarize(fields, np.array([.1, 1.1]), positions, 6)
        self.assertEqual(e['element'], 1)
        self.assertGreater(e['offset_from_tip_xz_mm'][0], 4.)
        fields['current_normal'][1] = [0, 0, 1]
        fields['direction_unique'][1] = True
        e = screen.summarize(fields, np.array([.1, 1.1]), positions, 6)
        self.assertFalse(e['direction']['representable_in_2d'])
        self.assertIsNone(e['direction']['current_tangent_xz'])

    def test_event_is_bracketed_without_interpolated_certainty(self):
        self.assertIsNone(first_event(.1, .2, {'peak_ratio': .99}))
        self.assertEqual(first_event(.1, .2, {'peak_ratio': 1.1})['depth_bracket_mm'], [.1, .2])
        self.assertTrue(first_event(None, 0., {'peak_ratio': 2.})['already_exceeded_at_first_frame'])

    def make_source(self, folder):
        folder.mkdir(); (folder/'states').mkdir()
        # Synthetic homogeneous strain fields test the postprocessor, not IPC equilibrium.
        X = np.array([[0., 0.], [1., 0.], [0., 1.], [0., 2.], [1., 2.], [0., 3.]])
        tri = np.array([[0, 1, 2]])
        cfg = dict(shear_modulus_n_mm2=66., inclination_deg=0.)
        rows = []; previous = X
        for i, (depth, stretch) in enumerate([(0., 1.), (.01, 1.2), (.02, 1.3)]):
            x = X.copy(); x[:3] = X[:3]@np.diag([stretch, 1/stretch])
            save_npz(folder/'states'/f'{i:04d}.npz', reference_xy=X, positions=x,
                     previous_positions=previous, triangles=tri, body_nodes=3, depth_mm=depth)
            rows.append(dict(depth_mm=depth, min_j=1., residual_n=0., reaction_fz_n=0.))
            previous = x
        report = dict(status='completed', completed=True, config=cfg, source_sha256=source_hashes(),
                      target_mm=.02, loading_increment_mm=.01, force_tolerance_n=1e-4,
                      simplifications=dict(fracture=False), steps=rows)
        write_json(folder/'result.json', report)

    def test_streaming_scan_exports_first_event_and_rejects_broken_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'input'; self.make_source(source)
            self.assertTrue(scan(source, root/'good', radii=(.1,)))
            r = json.loads((root/'good/result.json').read_text())
            self.assertFalse(r['physical_nucleation_validated'])
            self.assertFalse(r['fracture_evolution'])
            self.assertEqual(r['events']['nominal_local']['depth_bracket_mm'], [0., .01])
            self.assertTrue((root/'good/nominal_local_candidate.npz').exists())
            with np.load(source/'states/0002.npz') as z:
                data = {k: z[k].copy() for k in z.files}
            data['previous_positions'][0, 0] += .1
            save_npz(source/'states/0002.npz', **data)
            self.assertFalse(scan(source, root/'bad', radii=(.1,)))
            self.assertEqual(json.loads((root/'bad/result.json').read_text())['status'], 'failed')

    def test_directional_history_stops_at_first_variation_with_bracket(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'input'; self.make_source(source)
            self.assertTrue(scan(source, root/'onset', radii=(), directional_onset=True, stop_at_onset=True))
            r = json.loads((root/'onset/result.json').read_text())
            event = r['events']['directional_at1_local_stability']
            self.assertEqual(event['depth_bracket_mm'], [0., .01])
            self.assertTrue(event['evidence']['admissible_descent'])
            self.assertFalse(r['analyzed_entire_source'])
            self.assertFalse(r['physical_nucleation_validated'])
            self.assertTrue((root/'onset/directional_onset_candidate.npz').exists())

    def test_mixed_history_rejects_local_screen_and_plots_saved_pressure_stress(self):
        from tools.visualize_ipc import load_frames
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'input'; self.make_source(source)
            r = json.loads((source/'result.json').read_text())
            r['volumetric_discretization'] = 'macro_mixed'
            write_json(source/'result.json', r)
            self.assertFalse(scan(source, root/'unsupported', radii=()))
            self.assertIn('mixed pressure', json.loads((root/'unsupported/result.json').read_text())['error'])
            for i in range(3):
                file = source/'states'/f'{i:04d}.npz'
                with np.load(file) as z:
                    data = {key: z[key].copy() for key in z.files}
                # A deliberately distinct stress verifies that the reader does
                # not silently replace projected pressure with the local law.
                data.update(volumetric_discretization='macro_mixed',
                            cauchy=np.array([np.diag([1., 2., 3.])]), J=np.ones(1))
                save_npz(file, **data)
            frames = load_frames(source)[5]
            np.testing.assert_array_equal([frame['peak_mpa'] for frame in frames], [3., 3., 3.])
            data['J'] = np.array([.8]); save_npz(file, **data)
            with self.assertRaises(AssertionError): load_frames(source)

    def test_failed_prefix_is_explicit_and_never_accepts_active_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source=root/'input'; self.make_source(source)
            r = json.loads((source/'result.json').read_text())
            r.update(status='failed', completed=False, error='synthetic failure', steps=r['steps'][:2])
            write_json(source/'result.json', r)
            self.assertFalse(scan(source, root/'default', radii=()))
            self.assertTrue(scan(source, root/'prefix', radii=(), allow_failed_prefix=True,
                                 energy_probe_depths=(.01,), energy_lengths=(.1,)))
            result = json.loads((root/'prefix/result.json').read_text())
            self.assertFalse(result['source_completed'])
            self.assertEqual(result['last_depth_mm'], .01)
            self.assertEqual(result['source_target_mm'], .02)
            self.assertEqual(len(result['checkpoints_sha256']), 2)
            self.assertTrue(result['energy_probes'][0]['converged'])
            self.assertFalse(result['energy_probes'][0]['mechanics_recoupled'])
            r['status']='running'; write_json(source/'result.json', r)
            self.assertFalse(scan(source, root/'active', radii=(), allow_failed_prefix=True))

    def test_side_comparison_keeps_both_sides_without_strength_conversion(self):
        X=np.array([[-2.,0.],[-1.,0.],[-2.,1.],[1.,0.],[2.,0.],[1.,1.]])
        p=InitiationScreen(X, [[0,1,2],[3,4,5]], FEMConfig(inclination_deg=0), radii=())
        f=p.fields(X); f['nominal_tensile_mpa'][:]=[5.,2.]
        f['cauchy_principal_mpa'][:,-1]=[2.,8.]
        positions=np.vstack((X,[[0.,0.],[0.,1.]]))
        s=p.side_comparison(f,positions,6,radius=3)['regions']
        self.assertEqual(s['near_tip_left']['metrics']['nominal_mpa']['peak'],5.)
        self.assertEqual(s['near_tip_right']['metrics']['cauchy_tension_mpa']['peak'],8.)


if __name__ == '__main__':
    unittest.main()
