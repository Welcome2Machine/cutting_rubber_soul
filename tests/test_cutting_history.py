"""Synthetic archive audits and an independent uniform-stress support check."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prototype.material import response
from prototype.model_config import FEMConfig
from tools.export_cutting_history import digest, export_history, load_history


ROOT = Path(__file__).resolve().parents[1]


class CuttingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def archive(self, *, directional=False, status='completed', native_fx=False, saved_stress=False):
        source = self.root/'case'; source.mkdir(); (source/'states').mkdir()
        c = FEMConfig(length_mm=1., height_mm=1., width_mm=2., inclination_deg=60.)
        theta = np.deg2rad(c.inclination_deg)
        es, en = np.array([np.cos(theta), -np.sin(theta)]), np.array([np.sin(theta), np.cos(theta)])
        transform = np.array([es, en])
        local = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        X = local@transform
        knife = np.array([[.4, 1.1], [.6, 1.1], [.6, 1.3], [.4, 1.3]])@transform
        rest = np.vstack((X, knife)); tri = np.array([[0, 1, 2], [0, 2, 3]])
        depths = [0., .04, .1] if directional else [0., .05, .1]
        report = dict(status=status, completed=status == 'completed', target_mm=.1 if status == 'completed' else .2,
                      config=c.to_dict(), force_tolerance_n=1e-4, last_depth_mm=depths[-1],
                      last_checkpoint='states/0002.npz', source_sha256={}, steps=[], dhat_mm=.0025)
        if directional:
            report.update(framework='directional-rational-at1-v1', parameters=dict(increment=.1, case='oblique'),
                          first_damage_interval_mm=[0., .04], first_damage_note='Numerical damage only')
        else:
            report.update(case='oblique', loading_increment_mm=.05, simplifications=dict(fracture=False))
        for name in ('prototype/material.py', 'mechanics/finite_strain.py'):
            target = source/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes()); report['source_sha256'][name] = digest(target)
        previous = rest.copy(); old_damage = np.zeros(4)
        self.expected_force = []
        for i, depth in enumerate(depths):
            x = rest.copy(); x[:4, 1] -= depth*local[:, 1]; x[4:, 1] -= depth
            F = np.eye(2)-depth*np.outer([0., 1.], en)
            q = response(np.repeat(F[None], 2, axis=0), c, tangent=False)
            d = np.full(4, .01*i if directional else 0.)
            degradation = (1-d[0])**2
            stress = q['cauchy']*degradation
            force = -c.width_mm*c.length_mm*(degradation*q['P'][0]@en)
            self.expected_force.append(force)
            step = dict(depth_mm=depth, reaction_fz_n=float(force[1]), residual_n=1e-10, min_j=float(q['J'].min()))
            if native_fx:
                step['reaction_fx_n'] = float(force[0])
            if directional:
                step['max_damage'] = float(d.max())
            report['steps'].append(step)
            data = dict(reference_xy=rest, triangles=tri, positions=x, previous_positions=previous,
                        body_nodes=4, depth_mm=depth)
            if saved_stress or directional:
                data.update(cauchy=stress, J=q['J'])
            if directional:
                data.update(damage=d, old_damage=old_damage)
            np.savez_compressed(source/'states'/f'{i:04d}.npz', **data)
            previous = x.copy(); old_damage = d.copy()
        (source/'result.json').write_text(json.dumps(report), encoding='utf-8')
        return source

    def mutate_report(self, source, update):
        path = source/'result.json'; report = json.loads(path.read_text())
        update(report); path.write_text(json.dumps(report))

    def mutate_state(self, source, index, update):
        path = source/'states'/f'{index:04d}.npz'
        with np.load(path) as state:
            data = {key:state[key].copy() for key in state.files}
        update(data); np.savez_compressed(path, **data)

    def test_original_surface_material_ids_trajectory_and_geometric_depth(self):
        source = self.archive(); out = self.root/'export'
        report = export_history(source, out)
        self.assertEqual(report['accepted_frames'], 3)
        self.assertEqual(report['fx_available_frames'], 0)
        self.assertFalse(report['physical_crack_event_inferred'])
        self.assertFalse((out/'body_history.npz').exists())
        self.assertEqual(report['all_body_positions']['field'], 'positions[:body_nodes]')
        with np.load(out/'surface_history.npz') as data:
            np.testing.assert_array_equal(data['surface_node_ids'], [3, 2])
            np.testing.assert_allclose(data['surface_displacements'][-1], [[0., -.1], [0., -.1]], atol=1e-15)
            np.testing.assert_allclose(data['geometric_depths_mm'], [-.2, -.15, -.1], atol=1e-14)
            self.assertTrue(np.isnan(data['reaction_fx_n']).all())
            self.assertFalse(data['fx_available'].any())
        with (out/'forces.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(row['reaction_fx_n'] == '' for row in rows))
        with (out/'top_surface.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual([int(row['material_node_id']) for row in rows], [3, 2]*3)

    def test_diffuse_adaptive_prefix_needs_opt_in_and_preserves_damage(self):
        source = self.archive(directional=True, status='stopped')
        with self.assertRaisesRegex(ValueError, 'allow-incomplete'):
            load_history(source)
        report = export_history(source, self.root/'out', allow_incomplete_prefix=True, include_body=True)
        self.assertFalse(report['source_completed'])
        self.assertEqual(report['source_first_damage_interval_mm'], [0., .04])
        self.assertEqual(report['model'], 'continuous directional diffuse damage')
        with np.load(self.root/'out/body_history.npz') as data:
            np.testing.assert_allclose(data['diffuse_damage'][-1], .02)
            np.testing.assert_allclose(data['previous_diffuse_damage'][-1], .01)

    def test_running_is_never_an_accepted_prefix(self):
        source = self.archive(status='running')
        with self.assertRaisesRegex(ValueError, 'allow-incomplete'):
            load_history(source, allow_incomplete_prefix=True)

    def test_failed_prefix_alias_and_no_rejected_state_export(self):
        source = self.archive(status='failed')
        np.savez_compressed(source/'failed_trial.npz', positions=np.full((8, 2), np.nan))
        report = export_history(source, self.root/'out', allow_failed_prefix=True)
        self.assertEqual(report['accepted_frames'], 3)
        self.assertNotIn('failed_trial.npz', report['checkpoints_sha256'])

    def test_visualization_failed_prefix_requires_opt_in_and_rejects_running(self):
        from tools.visualize_ipc import load_frames
        source = self.archive(status='failed', native_fx=True)
        np.savez_compressed(source/'failed_trial.npz', positions=np.full((8, 2), np.nan))
        with self.assertRaisesRegex(ValueError, 'allow-failed-prefix'):
            load_frames(source)
        report, _, _, _, _, frames, hashes = load_frames(source, allow_failed_prefix=True)
        self.assertFalse(report['completed'])
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[-1]['depth'], .1)
        self.assertNotIn('failed_trial.npz', hashes)
        np.testing.assert_allclose([f['force_x'] for f in frames], np.array(self.expected_force)[:, 0])
        self.mutate_report(source, lambda r:r.update(status='running'))
        with self.assertRaisesRegex(ValueError, 'running'):
            load_frames(source, allow_failed_prefix=True)

    def test_visualization_validates_chain_residual_and_loading_prefix(self):
        from tools.visualize_ipc import load_frames
        source = self.archive(status='failed')
        state = source/'states/0001.npz'; original = state.read_bytes()
        self.mutate_state(source, 1, lambda d:d['previous_positions'].__setitem__((2, 0), 999.))
        with self.assertRaisesRegex(ValueError, 'accepted-state chain'):
            load_frames(source, allow_failed_prefix=True)
        state.write_bytes(original)
        self.mutate_report(source, lambda r:r['steps'][1].update(residual_n=.1))
        with self.assertRaisesRegex(ValueError, 'force residual'):
            load_frames(source, allow_failed_prefix=True)
        self.mutate_report(source, lambda r:(r['steps'][1].update(residual_n=0.),
                                             r['steps'][2].update(depth_mm=.15)))
        with self.assertRaisesRegex(ValueError, 'continuous loading prefix'):
            load_frames(source, allow_failed_prefix=True)

    def test_visualization_onset_is_tied_to_accepted_state_and_force_plot_marks_prefix(self):
        from tools.visualize_ipc import load_frames, load_online_onset, render_force_summary
        source = self.archive(status='failed', native_fx=True)
        report, _, reference, _, count, frames, _ = load_frames(source, allow_failed_prefix=True)
        event = dict(depth_bracket_mm=[0., .05], checkpoint='states/0001.npz',
                     evidence=dict(peak_node=2, reference_xz_mm=reference[2].tolist(),
                                   current_xz_mm=frames[1]['xy'][2].tolist()))
        key = 'directional_at1_local_stability'
        audit = dict(mechanics_feedback=False, fracture_evolution=False, primary_criterion=key,
                     primary_event=event, events={key:event})
        path = source/'initiation/result.json'; path.parent.mkdir()
        path.write_text(json.dumps(audit), encoding='utf-8')
        verified, fingerprint = load_online_onset(source, report, frames, reference, count)
        self.assertEqual(verified, event)
        self.assertEqual(fingerprint, digest(path))
        metadata = render_force_summary(report, frames, self.root/'figures', verified)
        self.assertFalse(metadata['source_completed'])
        self.assertEqual(metadata['source_status'], 'failed')
        self.assertEqual(metadata['onset_depth_bracket_mm'], [0., .05])
        self.assertEqual(metadata['last_accepted_depth_mm'], .1)
        self.assertEqual(metadata['post_onset_style'], 'dashed intact counterfactual')
        self.assertEqual(metadata['sha256'], digest(self.root/'figures/force_depth.png'))
        event['evidence']['current_xz_mm'][0] += .1
        path.write_text(json.dumps(audit), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'material point differs'):
            load_online_onset(source, report, frames, reference, count)

    def test_missing_or_extra_accepted_checkpoints_rejected(self):
        source = self.archive()
        path = source/'states/0001.npz'; original = path.read_bytes(); path.unlink()
        with self.assertRaisesRegex(ValueError, 'state files'):
            load_history(source)
        path.write_bytes(original)
        (source/'states/0003.npz').write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'state files'):
            load_history(source)

    def test_chain_topology_and_depth_changes_are_rejected(self):
        source = self.archive()
        path = source/'states/0001.npz'; original = path.read_bytes()
        changes = [lambda d:d['previous_positions'].__setitem__((2, 0), 999.),
                   lambda d:d['triangles'].__setitem__((0, 1), 3),
                   lambda d:d.__setitem__('depth_mm', np.array(.06))]
        for update in changes:
            with self.subTest(update=update):
                self.mutate_state(source, 1, update)
                with self.assertRaises(ValueError):
                    load_history(source)
                path.write_bytes(original)

    def test_acceptance_and_completion_are_not_assumed(self):
        source = self.archive()
        self.mutate_report(source, lambda r:r['steps'][1].update(residual_n=.1))
        with self.assertRaisesRegex(ValueError, 'residual'):
            load_history(source)
        self.mutate_report(source, lambda r:(r['steps'][1].update(residual_n=0.), r.update(target_mm=.15)))
        with self.assertRaisesRegex(ValueError, 'does not reach'):
            load_history(source)

    def test_frozen_hash_corruption_is_rejected_without_requiring_current_code_match(self):
        source = self.archive(); path = source/'source/prototype/material.py'
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Frozen source hash'):
            load_history(source)

    def test_saved_stress_recovers_uniform_support_force_and_keeps_native_separate(self):
        source = self.archive(directional=True, native_fx=True)
        report = export_history(source, self.root/'out', recover_fx=True)
        with np.load(self.root/'out/surface_history.npz') as data:
            np.testing.assert_allclose(data['recovered_fx_n'], np.array(self.expected_force)[:, 0], atol=1e-12)
            np.testing.assert_allclose(data['recovered_fz_n'], np.array(self.expected_force)[:, 1], atol=1e-12)
            np.testing.assert_allclose(data['reaction_fx_n'], data['recovered_fx_n'], atol=1e-12)
        self.assertIn('archived', report['recovered_force_provenance']['stress_source'])
        self.assertFalse(report['recovered_force_provenance']['recovered_Fx_is_native'])

    def test_intact_recovery_requires_matching_material_and_fz_agreement(self):
        source = self.archive()
        result = export_history(source, self.root/'out', recover_fx=True)
        self.assertEqual(result['fx_available_frames'], 0)
        with np.load(self.root/'out/surface_history.npz') as data:
            self.assertTrue(np.isnan(data['reaction_fx_n']).all())
            np.testing.assert_allclose(data['recovered_fx_n'], np.array(self.expected_force)[:, 0], atol=1e-12)
        self.mutate_report(source, lambda r:r['steps'][1].update(reaction_fz_n=999.))
        with self.assertRaisesRegex(ValueError, 'disagrees with native Fz'):
            export_history(source, self.root/'bad', recover_fx=True)
        self.assertFalse((self.root/'bad').exists())

    def test_force_recovery_refuses_bottom_contact_and_moving_support(self):
        source = self.archive(directional=True)
        data = load_history(source)
        from tools.export_cutting_history import recover_forces
        data['positions'][1, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'stationary bottom'):
            recover_forces(data)
        data = load_history(source); data['knives'][:] = 0.
        with self.assertRaisesRegex(ValueError, 'reach bottom'):
            recover_forces(data)

    def test_damage_healing_and_unknown_framework_rejected(self):
        source = self.archive(directional=True)
        self.mutate_state(source, 1, lambda d:d['old_damage'].__setitem__(0, .5))
        with self.assertRaisesRegex(ValueError, 'damage history'):
            load_history(source)
        self.mutate_report(source, lambda r:r.update(framework='closed-crack-component'))
        with self.assertRaisesRegex(ValueError, 'Unsupported history'):
            load_history(source)


if __name__ == '__main__':
    unittest.main()
