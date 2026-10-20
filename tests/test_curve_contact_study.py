"""History, provenance and scheduling gates for the independent curve study."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools import run_curve_contact_study as study


class StudyControlTests(unittest.TestCase):
    def test_requested_angle_queue_is_fresh_common_physics_and_bounded(self):
        jobs=study.make_angle_jobs(Path('out'),[0,33,57],2.)
        self.assertEqual([j['angle'] for j in jobs],[0,33,57])
        self.assertTrue(all(not j['resume_pilot'] and j['target_mm']==2. for j in jobs))
        self.assertTrue(all(study.refinement_allowed(j,jobs) for j in jobs))
        self.assertEqual({j['mu'] for j in jobs},{.35})
        self.assertEqual({(j['contact_spacing'],j['contact_order'],j['increment']) for j in jobs},{(.0025,2,.01)})
        for angles,target in (([0,0],2.),([30,60],2.),([0],3.),([],2.)):
            with self.assertRaises(ValueError):study.make_angle_jobs(Path('out'),angles,target)

    def test_angle_queue_advances_after_solver_and_postprocessing_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);jobs=study.make_angle_jobs(out,[0,33,57])
            (out/'manifest.json').write_text(json.dumps(dict(kind='angle_queue',jobs=jobs)))
            called=[]
            def run(out,m,j,i,s):
                called.append(j['angle'])
                if j['angle']==0:raise RuntimeError('solver fixture')
                if j['angle']==33:return dict(status='process_timeout',mechanics_completed=False)
                return dict(status='completed',mechanics_completed=True,fresh_audit_passed=True)
            with patch.object(study,'run_job',side_effect=run),patch.object(study,'summarize',side_effect=RuntimeError('plot fixture')),contextlib.redirect_stdout(io.StringIO()):
                study.worker(out)
            self.assertEqual(called,[0,33,57])
            state=study.read(out/'queue_status.json')
            self.assertEqual(state['status'],'finished');self.assertFalse(state['all_cases_completed'])
            self.assertEqual(state['jobs'][1]['status'],'process_timeout')

    def test_angle_experiment_mapping_keeps_unknown_zero_separate_and_full_range(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);csv=out/'experiment57.csv';image=out/'experiment33.json'
            csv.write_text('Depth (\u03bcm),Fz (N),depth_unit,force_unit,curve_type,component,speed_um_s\n'
                           '0,.51,\u03bcm,N,experiment,Fz,200\n1000,17.69,\u03bcm,N,experiment,Fz,200\n'
                           '2000,29.86,\u03bcm,N,experiment,Fz,200\n',encoding='utf-8')
            image.write_text(json.dumps([dict(depth_mm=2.,estimate_n=31.5,lower_n=30.5,upper_n=32.5)]))
            m=dict(experiments={'0':dict(kind='unavailable',interpretation='no numeric data'),
                '57':dict(kind='csv',path=str(csv),interpretation='nominal60 now57'),
                '33':dict(kind='image',path=str(image),interpretation='33 screenshot')})
            rows=[dict(depth_mm=d,reaction_fz_n=15*d) for d in (0.,1.,2.)]
            self.assertFalse(study.angle_experiment(rows,0,m)['available'])
            e=study.angle_experiment(rows,57,m)
            self.assertEqual(e['angle_deg'],57);self.assertEqual(e['anchors'][-1]['depth_mm'],2.)
            self.assertEqual(e['anchors'][0]['experiment_n'],.51)
            e=study.angle_experiment(rows,33,m)
            self.assertEqual(e['anchors'][0]['estimate_n'],31.5)
            self.assertFalse(e['independent_validation'])

    def test_failed_fresh_angle_preflight_cannot_enable_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);m=dict(kind='angle_queue',jobs=study.make_angle_jobs(out,[0,33,57]),target_mm=2.)
            (out/'manifest.json').write_text(json.dumps(m));before=(out/'manifest.json').read_bytes()
            def failed(folder):
                (folder/'queue_status.json').write_text(json.dumps(dict(all_cases_completed=False)))
            with patch.object(study,'verify_manifest'),patch.object(study,'worker',side_effect=failed):
                with self.assertRaises(RuntimeError):study.preflight_angles(out)
            self.assertEqual(before,(out/'manifest.json').read_bytes())

    def test_manual_anchor_reader_handles_bom_and_rejects_wrong_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'anchors.csv'
            header='depth_mm,experiment_image_estimate_n,image_reading_lower_n,image_reading_upper_n\n'
            path.write_text(header+'0,0,-.3,.3\n.25,4,3,5\n',encoding='utf-8-sig')
            before=path.read_bytes();rows=study.load_image_estimates(path)
            self.assertEqual(rows[1]['estimate_n'],4);self.assertEqual(path.read_bytes(),before)
            path.write_text(header+'0,4,5,6\n',encoding='utf-8-sig')
            with self.assertRaises(ValueError):study.load_image_estimates(path)

    def test_refinements_separate_history_and_quadrature(self):
        jobs=study.make_jobs(Path('out'))
        self.assertEqual(len(jobs),6)
        for angle in (30,60):
            base,q,half=[j for j in jobs if j['angle']==angle]
            self.assertTrue(base['resume_pilot']);self.assertFalse(q['resume_pilot']);self.assertFalse(half['resume_pilot'])
            self.assertEqual(base['increment'],q['increment'])
            self.assertEqual(base['increment'],2*half['increment'])
            self.assertEqual((base['contact_spacing'],base['contact_order']),(half['contact_spacing'],half['contact_order']))
            self.assertNotEqual(q['contact_order'],base['contact_order'])
            self.assertEqual({j['mu'] for j in (base,q,half)},{.35})
            self.assertFalse(study.refinement_allowed(q,jobs))
            base.update(mechanics_completed=True,fresh_audit_passed=False)
            self.assertFalse(study.refinement_allowed(q,jobs))
            base['fresh_audit_passed']=True;self.assertTrue(study.refinement_allowed(q,jobs))

    def test_worker_failure_skips_its_refinements_but_continues_other_angle(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);jobs=study.make_jobs(out)
            (out/'manifest.json').write_text(json.dumps(dict(jobs=jobs)))
            called=[]
            def run(out,m,j,i,s):
                called.append((j['variant'],j['angle']))
                if j['angle']==30:raise RuntimeError('Numerical failure fixture')
                return dict(status='completed',mechanics_completed=True,fresh_audit_passed=True)
            with patch.object(study,'run_job',side_effect=run),patch.object(study,'summarize',side_effect=RuntimeError('plot fixture')),contextlib.redirect_stdout(io.StringIO()):
                study.worker(out)
            self.assertEqual(called,[('base',30),('base',60),('quadrature',60),('halfstep',60)])
            report=study.read(out/'queue_status.json');self.assertEqual(report['status'],'finished')
            self.assertFalse(report['all_cases_completed']);self.assertEqual(report['jobs'][2]['status'],'skipped_base_not_completed')
            with self.assertRaises(FileExistsError):study.worker(out)

    def test_wrong_contact_order_or_time_grid_cannot_be_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);job=dict(expected_signature={'rigid_contact':{'order':2}},increment=.01)
            good=dict(signature=job['expected_signature'],settings=dict(increment=.01))
            with patch.object(study,'summarize_case',return_value=dict(mechanics_completed=True,status='completed')):
                for field,value in (('signature',{'rigid_contact':{'order':3}}),('settings',{'increment':.005})):
                    bad=dict(good,**{field:value});(out/'result.json').write_text(json.dumps(bad))
                    result=study.summarize_curve_case(out,job,0)
                    self.assertFalse(result['mechanics_completed']);self.assertEqual(result['status'],'configuration_mismatch')

    def test_source_or_input_change_is_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'kernel.py').write_text('original');(out/'history').write_text('original')
            m=dict(source=str(out),source_sha256={'kernel.py':study.sha(out/'kernel.py')},input_sha256={str(out/'history'):study.sha(out/'history')})
            study.verify_manifest(m)
            (out/'history').write_text('changed')
            with self.assertRaises(RuntimeError):study.verify_manifest(m)
            (out/'history').write_text('original');(out/'kernel.py').write_text('changed')
            with self.assertRaises(RuntimeError):study.verify_manifest(m)

    def test_duplicate_launch_cannot_truncate_existing_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'launch.json').write_text('{}');(out/'worker.stdout.log').write_text('preserve')
            manifest=dict(python='python',runtime_path='runtime',source=str(out))
            with patch.object(study.subprocess,'Popen') as spawn:
                with self.assertRaises(FileExistsError):study.launch(out,manifest)
                spawn.assert_not_called()
            self.assertEqual((out/'worker.stdout.log').read_text(),'preserve')

    def test_failed_preflight_cannot_promote_history_or_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);manifest=dict(jobs=study.make_jobs(out),policy={},target_mm=.5)
            (out/'manifest.json').write_text(json.dumps(manifest));before=(out/'manifest.json').read_bytes()
            def failed(folder):
                (folder/'queue_status.json').write_text(json.dumps(dict(all_cases_completed=False)))
            with patch.object(study,'verify_manifest'),patch.object(study,'worker',side_effect=failed):
                with self.assertRaises(RuntimeError):study.preflight(out)
            self.assertEqual((out/'manifest.json').read_bytes(),before)
            with patch.object(study.subprocess,'Popen') as spawn:
                with self.assertRaises(RuntimeError):study.launch(out,manifest)
                spawn.assert_not_called()

    def test_field_matching_does_not_fill_missing_depths_or_interval_holes(self):
        rows=[dict(depth_mm=d) for d in (0,.01,.015,.02)]
        matches=study.matched_records(rows,[dict(depth_mm=d) for d in (0,.01,.02)])
        self.assertEqual([m[1]['depth_mm'] for m in matches],[0,.01,.02])
        self.assertIsNone(study.hausdorff([],[[0,1]]))
        self.assertAlmostEqual(study.hausdorff([[0,10]],[[0,2],[8,10]]),3.)


class ActualPilotHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]/'outputs/curve_contact_20261015'
        if not (cls.root/'pilot.json').exists():raise unittest.SkipTest('Local qualified analytic pilots unavailable')
        cls.report=study.read(cls.root/'pilot.json')

    def fixture(self,out,angle):
        row=next(r for r in self.report['rows'] if r['angle']==angle)
        job=next(j for j in study.make_jobs(out) if j['variant']=='base' and j['angle']==angle)
        p=study.make_problem(job)
        states=[dict(path=str(self.root/f'pilot{angle}'/f'{i:04d}.npz'),sha256=row['checkpoint_sha256'][f'{i:04d}.npz']) for i in range(len(row['steps']))]
        manifest=dict(source=str(self.root/'source'),physical_source_sha256=self.report['source_sha256'],
            pilots={str(angle):dict(signature=study.curve_signature(p),steps=row['steps'],states=states,elapsed_s=row['elapsed_s'])})
        (out/'states').mkdir();report=dict(checkpoint_sha256={})
        return p,manifest,job,report

    def test_same_angle_resume_restores_both_histories_and_exact_dissipation(self):
        for angle in (30,60):
            with self.subTest(angle=angle),tempfile.TemporaryDirectory() as directory:
                out=Path(directory);p,m,j,r=self.fixture(out,angle)
                with contextlib.redirect_stdout(io.StringIO()):x,depth=study.restore_pilot(p,m,j,out,r)
                saved=study.load(self.root/f'pilot{angle}/0005.npz')
                self.assertEqual(depth,.05);np.testing.assert_array_equal(x,saved['positions'])
                np.testing.assert_array_equal(p.previous,x);np.testing.assert_array_equal(p.material_metric,saved['material_metric'])
                self.assertEqual(p.material_dissipation,float(saved['cumulative_viscous_dissipation_n_mm']))
                self.assertEqual(len(p._accepted_predictor_history),2)
                self.assertEqual(study.sha(out/'states/0005.npz'),m['pilots'][str(angle)]['states'][-1]['sha256'])

    def test_wrong_angle_or_changed_quadrature_cannot_restore_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);p,m,j,r=self.fixture(out,30)
            bad=study.make_problem(dict(j,angle=60))
            with self.assertRaises(ValueError):study.restore_pilot(bad,m,j,out,r)
            bad=study.make_problem(dict(j,contact_order=3))
            with self.assertRaises(ValueError):study.restore_pilot(bad,m,j,out,r)
            with self.assertRaises(ValueError):study.restore_pilot(p,m,dict(j,increment=.005),out,r)

    def test_broken_history_link_is_rejected_even_when_record_claims_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);p,m,j,r=self.fixture(out,30);real=study.load
            def corrupt(path):
                value=real(path)
                if str(path).endswith('0001.npz'):value['previous_material_metric'][0,0,0]+=.01
                return value
            with patch.object(study,'load',side_effect=corrupt):
                with self.assertRaises(AssertionError):study.restore_pilot(p,m,j,out,r)

    def test_observer_keeps_exact_pole_and_does_not_fill_missing_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);p,_,_,_=self.fixture(out,60);s=study.load(self.root/'pilot60/0005.npz')
            observer=study.CurveObserver(p);result=observer.observe(s['positions'],.05)
            expected=s['positions'][p.n+2+p.nose_facets//2]
            np.testing.assert_allclose(result['knife_tip_mm'],expected,rtol=0,atol=1e-12)
            self.assertEqual(result['status'],'no_tip_contact');self.assertIsNone(result['reference_offset_mm'])
            self.assertTrue(result['nose_offset_intervals_mm']);self.assertLess(observer.metadata['circle_sag_bound_mm'],2e-8)


if __name__=='__main__':unittest.main()
