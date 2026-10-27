"""Local constitutive, objectivity, topology and history checks; no long solve."""
import unittest
from pathlib import Path
import json
import tempfile
import numpy as np
from unittest.mock import patch
from prototype.ipc_reference import LoadStepFailure
from prototype.cohesive import CohesiveLaw, bonds_response
from prototype.cohesive_cut import make_cohesive_problem


class CohesiveTests(unittest.TestCase):
    def setUp(self):
        self.law=CohesiveLaw()

    def test_strength_energy_and_unloading(self):
        d=np.linspace(0, self.law.final_opening_mm, 20001)
        q=self.law.response(d, np.zeros_like(d))
        self.assertAlmostEqual(np.trapz(q['traction'], d), self.law.toughness_n_per_mm, places=8)
        self.assertAlmostEqual(self.law.response([self.law.peak_opening_mm], [0.])['traction'][0], self.law.strength_mpa)
        old=.7*self.law.final_opening_mm
        unload=self.law.response(np.array([old/3, 0., -.1]), np.full(3, old))
        np.testing.assert_array_equal(unload['history'], np.full(3, old))
        self.assertTrue(np.all(unload['dissipated'] >= 0))
        self.assertEqual(unload['traction'][-1], 0)
        self.assertEqual(self.law.response([self.law.final_opening_mm], [0.])['damage'][0], 1.)

    def test_reference_gap_objectivity_and_tangent(self):
        pairs=np.array([[0, 1]]);length=np.array([.006]);area=np.array([2.])
        x=np.array([[0., 0.], [.019, .003]])
        def response(xx):
            return bonds_response(xx, pairs, length, area, [0.], self.law)
        q=response(x)
        rotation=np.array([[np.cos(.7), -np.sin(.7)], [np.sin(.7), np.cos(.7)]])
        rotated=response(x@rotation.T+np.array([4., -3.]))
        self.assertAlmostEqual(q['stored_energy_n_mm'], rotated['stored_energy_n_mm'], places=12)
        np.testing.assert_allclose(rotated['gradient'].reshape(-1, 2), q['gradient'].reshape(-1, 2)@rotation.T, atol=1e-11)
        finite=np.zeros((4, 4));h=1e-7
        for j in range(4):
            a=x.copy().ravel();b=x.copy().ravel();a[j]+=h;b[j]-=h
            finite[:, j]=(response(a.reshape(2,2))['gradient']-response(b.reshape(2,2))['gradient'])/(2*h)
        np.testing.assert_allclose(q['hessian'].toarray(), finite, rtol=1e-7, atol=1e-7)
        zero=bonds_response(np.array([[0.,0.],[.006,0.]]), pairs, length, area, [0.], self.law)
        self.assertEqual(zero['stored_energy_n_mm'], 0.)
        self.assertEqual(zero['fracture_dissipation_n_mm'], 0.)

    def test_disabled_damage_is_elastic(self):
        q=CohesiveLaw(damage_enabled=False).response([.1], [0.])
        self.assertEqual(q['damage'][0], 0.)
        self.assertEqual(q['traction'][0], 5.)
        self.assertEqual(q['dissipated'][0], 0.)

    def test_real_seam_has_full_ipc_and_no_preload(self):
        root=Path(__file__).resolve().parents[1]
        config=json.loads((root/'configs/fracture_demo.json').read_text())
        p=make_cohesive_problem(config)
        self.assertEqual(p.config.length_mm, 30.)
        self.assertEqual(p.config.height_mm, 5.)
        self.assertEqual(p.config.width_mm, 20.)
        self.assertAlmostEqual(p.bond_areas.sum(), 100.)
        self.assertTrue(np.isin(p.mesh.seam_pairs, p.surface).all())
        self.assertFalse(p.ipc.has_intersections(p.cmesh, p.rest[p.surface]))
        self.assertLess(p.reference_contact_check['body_contact_force_norm_n'], 1e-4)
        np.testing.assert_allclose(p.cohesive_response(p.rest)['gradient'], 0., atol=1e-12)
        x=p.rest.copy();x[p.mesh.seam_pairs[-3:,1], 0]+=.03
        old=p.cohesive_maximum.copy()
        q=p.cohesive_response(x)
        self.assertGreater(q['damage'].max(), 0.)
        np.testing.assert_array_equal(p.cohesive_maximum, old)
        # CCD limits a path that would close/cross the actual two mesh boundaries.
        direction=np.zeros_like(p.rest)
        direction[np.flatnonzero(p.mesh.side_of_node > 0), 0]=-.012
        self.assertLess(p.safe_step(p.rest, direction), 1.)

    def test_failed_attempt_does_not_commit_either_history(self):
        root=Path(__file__).resolve().parents[1]
        config=json.loads((root/'configs/fracture_demo.json').read_text())
        p=make_cohesive_problem(config)
        before_metric=p.material_metric.copy();before_damage=p.cohesive_maximum.copy()
        def fail(x, depth, deadline):
            y=x.copy();y[p.mesh.seam_pairs[-3:,1], 0]+=.08
            self.assertGreater(p.cohesive_response(y)['damage'].max(), .99)
            raise LoadStepFailure('Injected rejected trial')
        with patch.object(p, 'solve_friction', side_effect=fail):
            with self.assertRaises(LoadStepFailure):
                p.step(p.rest.copy(), p.start, 0., float('inf'))
        np.testing.assert_array_equal(p.material_metric, before_metric)
        np.testing.assert_array_equal(p.cohesive_maximum, before_damage)

    def test_audit_failure_masks_events_and_replaces_positive_audit(self):
        from tools.run_cohesive_cut import run
        root=Path(__file__).resolve().parents[1]
        config=json.loads((root/'configs/fracture_demo.json').read_text())
        config['target_mm']=.001
        def broken_audit(p, out, report, **kwargs):
            Path(kwargs['destination']).write_text('{"passed":true}', encoding='utf-8')
            raise RuntimeError('Injected history chain failure after provisional audit')
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp);source=parent/'config.json';out=parent/'run'
            source.write_text(json.dumps(config), encoding='utf-8')
            with patch('tools.run_cohesive_cut.audit', side_effect=broken_audit), patch('tools.run_cohesive_cut.render'):
                self.assertEqual(run(source, out, 10), 1)
            report=json.loads((out/'result.json').read_text())
            audit=json.loads((out/'audit.json').read_text())
            self.assertFalse(audit['passed'])
            self.assertFalse(audit['cohesive_history_chain_passed'])
            self.assertEqual(audit['qualified_prefix_states'], 0)
            self.assertFalse((out/'audit.pending.json').exists())
            self.assertEqual(report['event_qualification'], 'unqualified')
            self.assertTrue(all(e['status']=='unavailable_unqualified_history' for e in report['events'].values()))
            self.assertEqual(report['execution_budget_s'], report['solve_budget_s']+report['reserved_audit_budget_s'])
            import csv
            with (out/'curves.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(row['reaction_fz_n']=='' and row['raw_reaction_fz_n']!='' for row in rows))


if __name__=='__main__':
    unittest.main()
