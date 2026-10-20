"""Guard independent histories, complete-path gates and full-surface comparisons."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tools import run_cutting_discretization as study
from tools import run_material_queue as queue


class CuttingDiscretizationTests(unittest.TestCase):
    def test_command_and_job_scope_preserve_explicit_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for a in (0,33,57):
                for tag in study.TAGS:
                    j=study.case_job(root,a,tag)
                    cmd=queue.make_command(dict(python='python',solve_seconds_per_case=900),j)
                    self.assertIn('tools.check_cutting_discretization',cmd)
                    for key in ('kind','kinematics','radius','order','increment'):
                        self.assertEqual(cmd[cmd.index('--'+key)+1],str(j[key]))
                    self.assertEqual('resume' in j,tag in ('pe_base','pe_depth'))
                    if tag=='pe_depth':self.assertEqual(j['resume'],str(root/f'pe_base_{a}'))
            j['target_mm']=1.
            with self.assertRaises(ValueError):queue.make_command(dict(python='python'),j)

    def test_zero_pilot_is_fresh_and_zero_geometry_preserves_physics(self):
        from tools.check_cutting_discretization import make_problem,geometry_audit
        from tools.check_native_mesh import baseline_problem,COMMON
        j=study.case_job(Path('unused'),0,'pilot')
        self.assertNotIn('resume',j);self.assertEqual(j['target_mm'],.01)
        self.assertEqual(j['name'],'pilot_p2_pe_0')
        base=baseline_problem(angle=0,**COMMON)
        for kin in ('plane_strain','plane_stress'):
            p=make_problem(0,kin=kin)
            checked=geometry_audit(p,base)
            self.assertTrue(checked['passed'])
            self.assertEqual(p.config.inclination_deg,0)
            self.assertEqual(p.material,base.material)

    def test_complete_mechanics_flag_does_not_override_wrong_discretization(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'states').mkdir()
            job=study.case_job(folder,33,'pe_base');job['target_mm']=.01
            rows=[dict(depth_mm=d,reaction_fz_n=1.,residual_n=1e-6,min_j=.9,minimum_distance_mm=.001)
                  for d in (0.,.01)]
            for i in range(2):(folder/'states'/f'{i:04d}.npz').touch()
            r=dict(status='completed',completed=True,target_mm=.01,steps=rows,force_tolerance_n=1e-4,
                signature={'config':dict(inclination_deg=33,friction_coefficient=.35)},
                settings={k:job[k] for k in ('kind','kinematics','radius','order','increment')})
            (folder/'audit.json').write_text(json.dumps(dict(passed=True,states=2,rows=[{'passed':True}]*2)))
            def check():
                (folder/'result.json').write_text(json.dumps(r));return queue.summarize_case(folder,job,0)
            self.assertTrue(check()['mechanics_completed'])
            r['settings']['order']=4
            self.assertFalse(check()['mechanics_completed'])
            r['settings']['order']=3;r['surface_export_error']='failed'
            self.assertFalse(check()['mechanics_completed'])

    def test_far_surface_motion_and_missing_contact_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);ref=np.array([[0.,0.],[1.,0.],[2.,0.]])
            geometry=out/'geometry.npz'
            np.savez(geometry,top_nodes=np.arange(3),reference_xy=ref,top_s_from_contact_mm=[0.,.5,2.])
            signature={k:{} for k in ('config','material','loading_rate','contact_formulation','dhat','physical_force_tolerance_n')}
            for name,delta in (('a',0.),('b',.003)):
                folder=out/name;(folder/'states').mkdir(parents=True);rows=[]
                for i,d in enumerate((0.,.01,.02)):
                    x=ref.copy();x[-1,1]+=delta
                    np.savez(folder/'states'/f'{i:04d}.npz',reference_xy=ref,positions=x)
                    rows.append(dict(depth_mm=d,reaction_fz_n=1.,min_j=.8,material={},
                        tip_contact=dict(status='no_tip_contact',reference_offset_mm=None)))
                (folder/'result.json').write_text(json.dumps(dict(steps=rows,signature=signature)))
            result=study.compare_paths(out/'a',out/'b',geometry,.02)
            self.assertFalse(result['force_and_surface_gate_passed'])
            self.assertAlmostEqual(result['maximum_surface_difference_mm'],.003)
            self.assertEqual(result['rows'][-1]['near1mm_surface_max_difference_mm'],0.)
            self.assertEqual(result['unique_tip_comparison_count'],0)
            self.assertIsNone(result['rows'][-1]['reference_tip_difference_mm'])
            r=json.loads((out/'b/result.json').read_text());r['steps'].pop(1)
            (out/'b/result.json').write_text(json.dumps(r))
            self.assertFalse(study.compare_paths(out/'a',out/'b',geometry,.02)['all_nominal_depths_matched'])

    def test_joint_gate_requires_both_angles_and_every_control(self):
        jobs=[study.case_job(Path('unused'),a,t) for t in study.TAGS for a in study.ANGLES]
        m={'geometry':{'33':'g33','57':'g57'}}
        for j in jobs:j.update(mechanics_completed=True,evidence_verified=True)
        with patch.object(study,'compare_paths',return_value={'force_and_surface_gate_passed':True}):
            self.assertTrue(study.depth_gate(m,jobs)['passed'])
            next(j for j in jobs if j['name']=='pe_radius1_57')['evidence_verified']=False
            self.assertFalse(study.depth_gate(m,jobs)['passed'])

    def test_zero_gate_needs_its_own_controls(self):
        jobs=[study.case_job(Path('unused'),0,t) for t in ('pilot',)+study.TAGS]
        m={'angles':[0],'geometry':{'0':'g0'}}
        for j in jobs:j.update(mechanics_completed=True,evidence_verified=True)
        with patch.object(study,'compare_paths',return_value={'force_and_surface_gate_passed':True}) as compare:
            self.assertTrue(study.depth_gate(m,jobs)['passed']);self.assertEqual(compare.call_count,3)
            next(j for j in jobs if j['tag']=='pe_halfstep')['mechanics_completed']=False
            self.assertFalse(study.depth_gate(m,jobs)['passed'])

    def test_predecessor_must_be_terminal_and_timeout_never_launches(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);prior=out/'prior';prior.mkdir()
            state={'jobs':[]};m=dict(input_sha256={},predecessor=dict(output=str(prior),wait_seconds=0.,poll_seconds=.001))
            file=prior/'queue_status.json'
            for status,jobs,expected in (
                ('running',[{'status':'completed'}],False),
                ('finished',[{'status':'running'}],False),
                ('finished',[{'status':'failed_or_unverified'}],True)):
                file.write_text(json.dumps(dict(status=status,jobs=jobs)))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(study.wait_for_predecessor(out,m,state),expected)

    def test_failed_zero_pilot_blocks_own_continuation_but_independent_checks_advance(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);jobs=[study.case_job(out,0,t) for t in ('pilot',)+study.TAGS]
            m=dict(angles=[0],jobs=jobs,geometry={'0':'g0'},input_sha256={})
            (out/'manifest.json').write_text(json.dumps(m));called=[]
            def run(out,m,j,state):
                called.append(j['tag'])
                if j['tag']=='pilot':raise RuntimeError('Synthetic pilot failure')
                return dict(status='completed',mechanics_completed=True)
            with patch.object(queue,'run_case',side_effect=run),patch.object(study,'verify_case',return_value={}),\
                 patch.object(study,'report'),patch.object(study,'compare_paths',return_value={'force_and_surface_gate_passed':True}),\
                 contextlib.redirect_stderr(io.StringIO()):study.worker(out)
            result=json.loads((out/'queue_status.json').read_text())
            self.assertEqual(len(called),6)
            self.assertEqual(result['jobs'][1]['status'],'skipped_pilot_gate')
            self.assertEqual(result['jobs'][-1]['status'],'skipped_shallow_gate')

    def test_worker_advances_after_failure_and_skips_both_depth_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);jobs=[study.case_job(out,a,t) for t in study.TAGS for a in study.ANGLES]
            m=dict(jobs=jobs,geometry={'33':'g33','57':'g57'},input_sha256={})
            (out/'manifest.json').write_text(json.dumps(m));called=[]
            def run(out,m,j,state):
                called.append(j['name'])
                if j['name']=='pe_base_33':raise RuntimeError('Synthetic child failure')
                return dict(status='completed',mechanics_completed=True)
            with patch.object(queue,'run_case',side_effect=run),patch.object(study,'verify_case',return_value={}),\
                 patch.object(study,'report'),patch.object(study,'compare_paths',return_value={'force_and_surface_gate_passed':True}),\
                 contextlib.redirect_stderr(io.StringIO()):study.worker(out)
            s=json.loads((out/'queue_status.json').read_text())
            self.assertEqual(len(called),12)
            self.assertEqual(s['status'],'finished')
            self.assertEqual([j['status'] for j in s['jobs'][-2:]],['skipped_shallow_gate']*2)


if __name__=='__main__':unittest.main()
