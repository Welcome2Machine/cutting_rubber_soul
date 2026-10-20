"""Physical path, frame, power and exact-history controls for oblique loading."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits
from tools import run_angle_path_study as study
from tools.run_material_candidate import resume_compatibility
from prototype.ipc_reference import IPCProblem


class InclinedLoadingTests(unittest.TestCase):
    def setUp(self):
        context=threadpool_limits(2);context.__enter__();self.addCleanup(context.__exit__,None,None,None)

    def test_normal_history_and_shared_approach(self):
        for angle in (0,33,57):
            b=study.make_problem(angle,'normal',coarse=True)
            c=study.make_problem(angle,'oblique',coarse=True)
            np.testing.assert_array_equal(b.rest,c.rest)
            for q in (b.start,0.):
                np.testing.assert_array_equal(b.prescribed_blade_positions(q),c.prescribed_blade_positions(q))
            q=.04
            for p in (b,c):
                delta=p.prescribed_blade_positions(q)[0]-p.blade_at_zero[0]
                self.assertAlmostEqual(-delta@p.en,q,places=13)
                self.assertAlmostEqual(delta@p.es,q*p.tangent_ratio,places=13)
                x=p.affine_loading_candidate(p.rest,p.start,q)
                np.testing.assert_array_equal(x[p.mesh.bottom_nodes],p.rest[p.mesh.bottom_nodes])
            delta=c.prescribed_blade_positions(q)[0]-c.blade_at_zero[0]
            np.testing.assert_allclose(delta,[0.,-q/np.cos(np.deg2rad(angle))],atol=2e-14)

    def test_default_vertical_hook_preserves_arithmetic(self):
        p=IPCProblem('patch',mu=.35,local_spacing=.2,surface_spacing=.2)
        x=p.rest.copy();old=p.start;depth=.02
        expected=x.copy();expected[:p.n,1]-=(depth-old)*p.mesh.local_sn[:,1]/p.config.height_mm
        expected[p.n:,1]-=depth-old
        np.testing.assert_array_equal(p.affine_loading_candidate(x,old,depth),expected)
        np.testing.assert_array_equal(p.prescribed_blade_positions(depth),expected[p.n:])

    def test_rotated_material_history_contact_and_force_covariance(self):
        for angle,mode in ((33,'normal'),(57,'oblique')):
            p=study.make_problem(angle,mode,coarse=True)
            r=study.make_problem(angle,mode,coarse=True,frame_rotation=angle)
            Q=r.frame_matrix;Q3=np.eye(3);Q3[:2,:2]=Q
            np.testing.assert_allclose(p.rest@Q.T,r.rest,rtol=0,atol=3e-14)
            np.testing.assert_array_equal(p.mesh.triangles,r.mesh.triangles)
            np.testing.assert_array_equal(p.point_elements,r.point_elements)
            before=p.affine_loading_candidate(p.rest,p.start,.002)
            p.material_dt=.07;r.material_dt=.07
            history=p.material_response(before)['metric']
            p.previous_material_metric=history
            r.previous_material_metric=Q3@history@Q3.T
            p.previous=before;r.previous=before@Q.T
            x=p.affine_loading_candidate(before,.002,.007);y=x@Q.T
            ea,ga,_,ja=p.evaluate(x,p.lag(x),tangent=False)
            eb,gb,_,jb=r.evaluate(y,r.lag(y),tangent=False)
            self.assertAlmostEqual(ea,eb,delta=1e-10)
            self.assertAlmostEqual(ja,jb,delta=2e-11)
            np.testing.assert_allclose(ga.reshape(-1,2)@Q.T,gb.reshape(-1,2),rtol=2e-7,atol=1e-9)
            np.testing.assert_allclose(Q3@p.material_response(x)['metric']@Q3.T,
                                       r.material_response(y)['metric'],rtol=1e-10,atol=1e-10)

    def test_real_path_audit_power_and_restart_history(self):
        for mode in ('normal','oblique'):
            p=study.make_problem(33,mode,coarse=True);x=p.rest.copy();old=p.start;history=[]
            with contextlib.redirect_stdout(io.StringIO()):
                for q in (0.,.005):
                    prior=p.material_metric.copy()
                    x,row=p.step(x,old,q,time.monotonic()+45);old=q
                    np.testing.assert_array_equal(p.previous_material_metric,prior)
                    state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,
                        previous_positions=p.previous,depth_mm=q,**p.checkpoint_data())
                    self.assertTrue(p.audit_state(state,row)['passed'])
                    history.append((q,x.copy()))
                p.seed_predictor_history(history)
            # Frozen lag potential virtual work against blade translation;
            # force projection must agree with a finite-difference derivative.
            lag=p.lag(x);delta=np.zeros_like(x);delta[p.n:]=p.loading_direction
            h=1e-6
            ep=p.evaluate(x+h*delta,lag,tangent=False)[0]
            em=p.evaluate(x-h*delta,lag,tangent=False)[0]
            self.assertAlmostEqual((ep-em)/(2*h),row['motion']['generalized_resistance_n'],delta=1e-4)
            bad=dict(state,path_parameters=state['path_parameters'].copy());bad['path_parameters'][3]+=.1
            with self.assertRaisesRegex(RuntimeError,'loading path'):p.audit_state(bad,row)
            bad=dict(state,positions=state['positions'].copy());bad['positions'][p.n:,0]+=.001
            with self.assertRaisesRegex(RuntimeError,'boundaries'):p.audit_state(bad,row)

    def test_signatures_forbid_cross_path_and_time_transfer(self):
        b=study.make_problem(33,'normal',coarse=True);c=study.make_problem(33,'oblique',coarse=True)
        r=dict(signature=study.signature(b),source_sha256={})
        with self.assertRaises(ValueError):resume_compatibility(r,dict(r,signature=study.signature(c)))
        s=copy.deepcopy(r);s['signature']['loading_path']['normal_speed_mm_s']=.2
        with self.assertRaises(ValueError):resume_compatibility(r,s)

    def test_queue_scope_independent_controls_and_original_history_only(self):
        out=Path('unused');jobs=[study.case_job(out,*case,tag) for tag in study.CONTROLS for case in study.CASES]
        self.assertEqual(len(jobs),20)
        for j in jobs:
            cmd=study.command(dict(python='python'),j)
            self.assertIn('tools.run_angle_path_study',cmd)
            self.assertEqual(j['target_mm'],.1)
            self.assertEqual('resume' in j,j['tag']=='base')
            if 'resume' in j:self.assertEqual(j['resume'],str(out/f'{j["case"]}_pilot'))
        bad=dict(jobs[0],target_mm=.25)
        with self.assertRaises(ValueError):study.command(dict(python='python'),bad)

    def test_incomplete_numerical_controls_never_pass(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);jobs=[study.case_job(out,*case,tag) for tag in study.CONTROLS for case in study.CASES]
            r=study.report(out,{},dict(status='prepared',jobs=jobs))
            self.assertFalse(r['joint_gate_passed']);self.assertEqual(r['paired_contrasts'],[])

    def test_force_vector_and_far_surface_errors_cannot_hide_in_fz(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);theta=np.deg2rad(33)
            es=np.array([np.cos(theta),-np.sin(theta)]);en=np.array([np.sin(theta),np.cos(theta)])
            reference=np.array([0.,1.,10.])[:,None]*es+5*en
            sig={k:{} for k in ('config','material','loading_rate','loading_path','contact_formulation','dhat')}
            sig['config']={'height_mm':5.}
            def write(name,normal_error=0.,motion_error=0.):
                folder=root/name;(folder/'states').mkdir(parents=True,exist_ok=True);rows=[]
                for i,q in enumerate(np.linspace(0,.1,11)):
                    x=reference.copy();x[-1,0]+=motion_error
                    np.savez(folder/'states'/f'{i:04d}.npz',base_body_nodes=3,reference_xy=reference,positions=x)
                    rows.append(dict(depth_mm=float(q),reaction_fz_n=1.,min_j=.9,motion=dict(
                        normal_reaction_n=1.+normal_error,tangent_resistance_n=.5,generalized_resistance_n=1.)))
                (folder/'result.json').write_text(json.dumps(dict(steps=rows,signature=sig,settings={'angle':33})))
                return folder
            a=write('a');b=write('b')
            self.assertTrue(study.compare_control(a,b)['passed'])
            write('b',normal_error=.03)
            self.assertFalse(study.compare_control(a,b)['passed'])
            write('b',motion_error=.003)
            self.assertFalse(study.compare_control(a,b)['passed'])


if __name__=='__main__':unittest.main()
