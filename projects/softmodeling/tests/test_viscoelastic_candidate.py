"""Qualify the actual material law, not fitted force-curve surrogates."""
import unittest
from unittest.mock import patch
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import numpy as np
from mechanics.finite_strain import Material,response,elastic_response
from prototype.ipc_viscoelastic import ViscoelasticProblem
from prototype.ipc_reference import IPCProblem,LoadStepFailure


def material(tau=.4):
    return Material((.47,),(2.,),(.94,),(2.,),.49,tau,
                    viscous_flow='isochoric_maxwell')


class QueueContinuationTests(unittest.TestCase):
    def test_ptc_pause_does_not_trigger_retry_or_commit(self):
        from tools.material_queue_control import QueuePaused,checkpoint
        from prototype.ipc_reference import adaptive_load_steps
        from unittest.mock import Mock
        import os,time
        p=ViscoelasticProblem(maxwell_shear=.94,relaxation_s=.4,loading_rate=.2,case='patch',mu=.35,
            local_spacing=.2,surface_spacing=.1)
        x=p.rest.copy()
        before=p.material_metric.copy()
        reject=Mock()
        with tempfile.TemporaryDirectory() as directory:
            flag=Path(directory)/'pause_requested.json';flag.touch()
            with patch.dict(os.environ,SOFTMODELING_QUEUE_PAUSE=str(flag)):
                p.queue_checkpoint=checkpoint
                with self.assertRaises(QueuePaused):
                    list(adaptive_load_steps(p,x,p.start,p.start+.001,time.monotonic()+20,.00025,reject))
        reject.assert_not_called()
        np.testing.assert_array_equal(p.material_metric,before)

    def test_resume_uses_case_identity_in_same_angle_batch(self):
        from tools import material_queue_control as control
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            jobs=[]
            for name in ('A33','E33'):
                folder=root/name;folder.mkdir()
                jobs.append(dict(name=name,case_id=name,angle=33,output=str(folder),target_mm=2.,solve_seconds=20))
            (root/'manifest.json').write_text(json.dumps(dict(jobs=jobs,audit_allowance_s=10)))
            (root/'queue_status.json').write_text(json.dumps(dict(status='paused',jobs=[
                dict(jobs[0],status='completed',mechanics_completed=True),
                dict(jobs[1],status='paused',active_wall_s=5.,active_solve_s=3.)])))
            (root/'E33/result.json').write_text(json.dumps(dict(steps=[dict(depth_mm=.005)])))
            launch=Mock()
            control.restart(root,launch)
            attempt,manifest=launch.call_args.args
            self.assertTrue(manifest['jobs'][0]['reused'])
            self.assertEqual(manifest['jobs'][1]['resume_source'],str(root/'E33'))
            self.assertEqual(manifest['jobs'][1]['solve_seconds'],17.)
            self.assertEqual(manifest['jobs'][1]['remaining_wall_s'],25.)
            self.assertEqual(attempt,control.active(root))
            with self.assertRaises(RuntimeError):control.restart(root,launch)
            self.assertEqual(launch.call_count,1)

    def test_exploratory_resume_preserves_budget_and_paused_checkpoint(self):
        from tools import material_queue_control as control
        from run_support import write_json
        for changed in (False,True):
            with tempfile.TemporaryDirectory() as work:
                root=Path(work)
                folder=root/'case';(folder/'states').mkdir(parents=True)
                job=dict(name='case',case_id='case',output=str(folder),solve_seconds=100.,audit_seconds=30.,
                         export_seconds=20.,remaining_wall_s=160.,target_mm=2.)
                write_json(root/'manifest.json',dict(jobs=[job],input_sha256={},audit_allowance_s=30.,run_purpose='exploratory_long'))
                saved=dict(case_id='case',steps=[dict(depth_mm=0.)],solve_status='failed',target_mm=2.)
                write_json(folder/'result.json',saved)
                state=folder/'states/0000.npz';state.write_bytes(b'accepted')
                receipt=control.prefix_identity(folder,saved)
                old=dict(job,status='paused',active_solve_s=12.,active_audit_s=3.,active_export_s=2.,
                         active_wall_s=19.,paused_prefix=receipt,solve_status='failed')
                write_json(root/'queue_status.json',dict(status='paused',jobs=[old]))
                if changed:
                    state.write_bytes(b'different')
                    with self.assertRaisesRegex(ValueError,'checkpoint identity changed'):
                        control.restart(root,lambda *args: None)
                else:
                    control.restart(root,lambda *args: None)
                    new=control.read(control.active(root)/'manifest.json')['jobs'][0]
                    self.assertEqual((new['solve_seconds'],new['audit_seconds'],new['export_seconds'],new['remaining_wall_s']),(88.,27.,18.,141.))
                    self.assertEqual(new['paused_parent_binding'],receipt)
                    self.assertEqual(new['finish_prefix'],'failed')


class MaxwellCandidateTests(unittest.TestCase):
    def test_repartition_preserves_bulk_geometry_and_positive_relaxation(self):
        kw=dict(maxwell_shear=.94,relaxation_s=.4,loading_rate=.2,
                case='patch',mu=.35,local_spacing=.2,surface_spacing=.1)
        p=ViscoelasticProblem(**kw)
        k=p.config.bulk_modulus_n_mm2
        same=ViscoelasticProblem(**kw,bulk_modulus=k)
        q=ViscoelasticProblem(**dict(kw,maxwell_shear=1.24,relaxation_s=1.7),
                             shear_modulus=.14,bulk_modulus=k)
        self.assertAlmostEqual(q.material.moduli()[1],k,places=10)
        self.assertAlmostEqual(q.material.moduli()[0],.14)
        np.testing.assert_array_equal(p.rest,q.rest)
        np.testing.assert_array_equal(p.mesh.triangles,q.mesh.triangles)
        F=np.array([[[1.1,.3],[.05,.9]],[[.8,.0],[.0,.8]]]);A=np.tile(np.eye(3),(2,1,1))
        a=response(F,A,.05,p.material,kinematics='plane_strain')
        b=response(F,A,.05,same.material,kinematics='plane_strain')
        for key in ('P','tangent','incremental_potential','metric'):
            np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)
        initial=response(F,A,.05,q.material,kinematics='plane_strain')
        end=response(F,initial['metric'],20.,q.material,kinematics='plane_strain')
        self.assertTrue(np.all(end['energy']<=initial['energy']))
        self.assertTrue(np.all(end['viscous_rate']>=0))
        np.testing.assert_allclose(np.linalg.det(end['metric']),1.,atol=1e-12)
        for material_ in (p.material,q.material):
            # Actual volumetric energy has the same K, irrespective of G split.
            self.assertAlmostEqual(material_.moduli()[1],k,places=10)
        # New G/K split and longer tau: differentiate the real updated law.
        for i in range(4):
            perturb=np.eye(4)[i].reshape(2,2)*1e-6
            a=response(F+perturb,A,.05,q.material,kinematics='plane_strain',tangent=False)
            b=response(F-perturb,A,.05,q.material,kinematics='plane_strain',tangent=False)
            np.testing.assert_allclose((a['incremental_potential']-b['incremental_potential'])/2e-6,
                                       initial['P'].reshape(-1,4)[:,i],rtol=1e-6,atol=2e-7)
            np.testing.assert_allclose(((a['P']-b['P'])/2e-6).reshape(-1,4),
                                       initial['tangent'][:,:,i],rtol=2e-6,atol=2e-7)
        for value in (-1.,0.,float('nan')):
            with self.assertRaises(ValueError):ViscoelasticProblem(**kw,bulk_modulus=value)

    def test_assembled_candidate_force_tangent_and_incremental_potential(self):
        p=ViscoelasticProblem(maxwell_shear=.94,relaxation_s=.4,loading_rate=.2,
                             case='patch',mu=.35,local_spacing=.2,surface_spacing=.1)
        p.material_dt=.05
        xy=p.mesh.reference_xy
        u=np.column_stack((.2*xy[:,0]+.1*xy[:,1],-.1*xy[:,1])).ravel()
        r,k,dofs,_,_=p.material_bulk(u)
        v=np.random.default_rng(742).normal(size=u.size)*.1;h=1e-6
        a=p.material_bulk(u+h*v,with_tangent=False)
        b=p.material_bulk(u-h*v,with_tangent=False)
        self.assertAlmostEqual((a[3]-b[3])/(2*h),float(np.sum(r*v[dofs])),delta=2e-7)
        actual=np.zeros(u.size);expected=np.zeros(u.size)
        np.add.at(actual,dofs.ravel(),((a[0]-b[0])/(2*h)).ravel())
        np.add.at(expected,dofs.ravel(),np.einsum('eij,ej->ei',k,v[dofs]).ravel())
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)

    def test_potential_stress_tangent_objectivity_and_trial_purity(self):
        m=material();A=np.tile(np.eye(3),(3,1,1))
        F=np.array([np.eye(2),[[1.3,.5],[-.1,.85]],[[.6,.2],[.1,1.2]]])
        A=response(F,A,.2,m,kinematics='plane_strain')['metric'];saved=A.copy()
        F=F+.07*np.array([[.2,1.],[-.3,.1]])
        q=response(F,A,.13,m,kinematics='plane_strain')
        for i in range(4):
            d=np.eye(4)[i].reshape(2,2)*1e-6
            plus=response(F+d,A,.13,m,kinematics='plane_strain',tangent=False)
            minus=response(F-d,A,.13,m,kinematics='plane_strain',tangent=False)
            np.testing.assert_allclose((plus['incremental_potential']-minus['incremental_potential'])/2e-6,
                                       q['P'].reshape(-1,4)[:,i],atol=2e-7,rtol=1e-6)
            np.testing.assert_allclose(((plus['P']-minus['P'])/2e-6).reshape(-1,4),
                                       q['tangent'][:,:,i],atol=3e-7,rtol=1e-6)
        np.testing.assert_array_equal(A,saved)
        np.testing.assert_allclose(q['tangent'],q['tangent'].transpose(0,2,1),atol=1e-10)
        theta=.7;Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        rotated=response(Q@F,A,.13,m,kinematics='plane_strain')
        np.testing.assert_allclose(rotated['P'],Q@q['P'],atol=1e-10)
        np.testing.assert_allclose(rotated['metric'],q['metric'],atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(q['metric']),1.,atol=1e-11)
        self.assertTrue(np.all(q['viscous_rate']>=-1e-12))


    def test_independent_elastic_patch_and_nonuniform_nearly_incompressible_solution(self):
        from tools.verify_intact_numerics import affine_patch, manufactured_case, GE
        checks=affine_patch()
        self.assertLess(max(x['force_relative'] for x in checks),1e-8)
        self.assertLess(max(x['max_F_error'] for x in checks),1e-8)
        coarse,fine=(manufactured_case(n,GE) for n in (16,32))
        self.assertLess(fine['displacement_L2_relative'],.02)
        self.assertLess(fine['reaction_relative'],.02)
        self.assertLess(fine['pressure_face_jump_rms_MPa'],coarse['pressure_face_jump_rms_MPa'])

    def test_actual_small_shear_time_constant(self):
        m=material();dt=.02;F=np.array([[[1.,1e-5],[0.,1.]]]);A=np.eye(3)[None]
        equilibrium=elastic_response(F,m)['P'][0,0,1]
        stresses=[]
        for _ in range(30):
            q=response(F,A,dt,m,kinematics='plane_strain',tangent=False)
            A=q['metric'];stresses.append(q['P'][0,0,1]-equilibrium)
        np.testing.assert_allclose(np.array(stresses[1:])/stresses[:-1],1/(1+dt/.4),rtol=3e-7)

    def test_volume_has_no_added_branch_and_relaxation_lowers_stored_energy(self):
        m=material();self.assertEqual(m.moduli(True)[1],0.)
        # Pure 3D dilation can be tested directly in principal coordinates.
        from mechanics.finite_strain import principal
        w,t,h=principal(np.full((1,3),.2),m,True)
        np.testing.assert_allclose(w,0.,atol=1e-14)
        np.testing.assert_allclose(t,0.,atol=1e-14)
        F=np.array([[[1.5,.3],[0.,1/1.5]]]);A=np.eye(3)[None]
        energies=[]
        for _ in range(100):
            q=response(F,A,.08,m,kinematics='plane_strain',tangent=False)
            A=q['metric'];energies.append(q['energy'][0])
            self.assertGreaterEqual(q['viscous_rate'][0],-1e-12)
        self.assertTrue(np.all(np.diff(energies)<=1e-11))
        eq=elastic_response(F,m)['energy'][0]
        self.assertLess(abs(energies[-1]-eq),1e-6)

    def test_time_refinement_converges_and_faster_loading_is_stiffer(self):
        def ramp(dt,total):
            m=material();A=np.eye(3)[None];n=round(total/dt)
            for i in range(1,n+1):
                F=np.array([[[1.,.5*i/n],[0.,1.]]])
                q=response(F,A,total/n,m,kinematics='plane_strain',tangent=False);A=q['metric']
            return q['P'][0,0,1]
        a,b,c=[ramp(dt,1.) for dt in (.1,.05,.025)]
        self.assertLess(abs(c-b),abs(b-a))
        self.assertGreater(ramp(.005,.5),ramp(.005,2.))

    def test_fixed_velocity_rebuilds_friction_at_retry_and_during_audit(self):
        p=ViscoelasticProblem(maxwell_shear=1.24,relaxation_s=1.7,loading_rate=.2,
            case='patch',mu=.35,local_spacing=.2,surface_spacing=.1,friction_velocity_mm_s=.0004)
        metric=p.material_metric.copy()
        def reject(*args):
            self.assertEqual(p.friction_smoothing_mm,.0004*p.material_dt)
            raise LoadStepFailure('rejected')
        with patch.object(IPCProblem,'step',side_effect=reject):
            for depth in (.005,.0025):
                with self.assertRaises(LoadStepFailure):p.step(p.rest,0.,depth,0.)
                np.testing.assert_array_equal(p.material_metric,metric)
        before=p.friction
        snapshot=dict(previous_positions=p.rest,previous_material_metric=metric,material_dt_s=.025,
            epsilon_u_mm=.0004*.025,epsilon_v_mm_s=.0004,reference_xy=p.rest,
            triangles=p.mesh.triangles,positions=p.rest,depth_mm=p.start,material_metric=metric)
        record=dict(depth_mm=p.start,reaction_x_n=1.,reaction_fz_n=0.,epsilon_u_mm=.0004*.025)
        def acceptance(*args):
            self.assertEqual(p.friction_smoothing_mm,.0004*.025)
            return dict(reaction_x_n=0.,reaction_fz_n=0.)
        with patch.object(p,'acceptance',side_effect=acceptance),patch.object(p,'material_response',return_value={'metric':metric}):
            with self.assertRaisesRegex(RuntimeError,'horizontal force'):p.audit_state(snapshot,record)
        self.assertIs(p.friction,before)
        self.assertAlmostEqual(p.friction_smoothing_mm,.0004*(.0025/.2))
        with self.assertRaisesRegex(RuntimeError,'regularization'):
            p.audit_state(dict(snapshot,epsilon_u_mm=1.),record)

    def test_failed_step_never_commits_metric_or_dissipation(self):
        p=ViscoelasticProblem.__new__(ViscoelasticProblem)
        p.loading_rate=.2;p.material_metric=np.eye(3)[None];p.material_dissipation=0.
        p.friction_velocity_mm_s=None  # This synthetic host bypasses __init__.
        def fail(*args):raise LoadStepFailure('rejected physical equilibrium')
        with patch.object(IPCProblem,'step',side_effect=fail):
            with self.assertRaises(LoadStepFailure):p.step(np.zeros((1,2)),.1,.12,0.)
            np.testing.assert_array_equal(p.material_metric,np.eye(3)[None])
            self.assertEqual(p.material_dissipation,0.)
            with self.assertRaises(LoadStepFailure):p.step(np.zeros((1,2)),.1,.11,0.)
            self.assertAlmostEqual(p.material_dt,.05)
            np.testing.assert_array_equal(p.previous_material_metric,np.eye(3)[None])


    def test_full_blade_replay_avoids_accumulated_roundoff_without_accepting_tampering(self):
        p=ViscoelasticProblem(maxwell_shear=1.24,relaxation_s=1.7,loading_rate=.2,
            case='patch',mu=.35,local_spacing=.2,surface_spacing=.1)
        # Large coordinates and 601 actual depth updates reproduce the old
        # absolute-reconstruction false rejection, without a research solve.
        p.rest[p.n:,1]+=80.-float(np.max(p.rest[p.n:,1]))
        replay=p.rest[p.n:].copy();prior=p.start
        for depth in np.r_[0.,np.arange(.005,3.,.005),3.]:
            replay[:,1]-=float(depth)-float(prior);prior=depth
        positions=p.rest.copy();positions[p.n:]=replay
        state=dict(reference_xy=p.rest.copy(),triangles=p.mesh.triangles.copy(),
            positions=positions,previous_positions=p.rest.copy(),depth_mm=3.,
            previous_material_metric=p.material_metric.copy(),material_dt_s=.025)
        self.assertGreater(np.max(abs(replay-p.prescribed_blade_positions(3.))),
                           p.blade_boundary_roundoff_tolerance(3.))
        with self.assertRaisesRegex(RuntimeError,'prescribed boundaries'):
            p.audit_state(state,{})
        # Reaching force assembly proves only the geometric gate here; the
        # actual full numerical audit is separate and remains mandatory.
        with patch.object(p,'acceptance',side_effect=RuntimeError('force assembly reached')):
            with self.assertRaisesRegex(RuntimeError,'force assembly reached'):
                p.audit_state(state,{},incremental_blade_positions=replay)
            for axis in (0,1):
                bad=positions.copy();bad[p.n,axis]=np.nextafter(bad[p.n,axis],np.inf)
                with self.assertRaisesRegex(RuntimeError,'prescribed boundaries'):
                    p.audit_state(dict(state,positions=bad),{},incremental_blade_positions=replay)
            bad=positions.copy();bad[p.mesh.bottom_nodes[0],0]+=1e-12
            with self.assertRaisesRegex(RuntimeError,'prescribed boundaries'):
                p.audit_state(dict(state,positions=bad),{},incremental_blade_positions=replay)


class MaterialRunControlTests(unittest.TestCase):
    def _prepared_queue_worker(self,directory,*,completed=True,exit_code=0):
        """A real, bounded subprocess exercises launch/wait without an FE solve."""
        import sys,textwrap
        root=Path(directory);out=root/'queue';out.mkdir()
        source=root/'source';(source/'tools').mkdir(parents=True)
        (source/'tools/__init__.py').write_text('')
        (source/'tools/run_material_queue.py').write_text(textwrap.dedent('''\
            import json,os,sys,time
            from pathlib import Path
            out=Path(sys.argv[sys.argv.index('--output')+1])
            manifest=json.loads((out/'manifest.json').read_text())
            def state(status,completed=False):
                temporary=out/'state.tmp'
                temporary.write_text(json.dumps(dict(status=status,jobs=[],all_cases_completed=completed)))
                temporary.replace(out/'queue_status.json')
            state('running')
            (out/'worker_ready.json').write_text(json.dumps(dict(session=os.getsid(0) if hasattr(os,'getsid') else None)))
            deadline=time.monotonic()+15
            while not (out/'release_worker').exists():
                if (out/'pause_requested.json').exists():
                    state('paused');sys.exit(0)
                if time.monotonic()>deadline:sys.exit(9)
                time.sleep(.01)
            if manifest['test_exit']:sys.exit(manifest['test_exit'])
            state('finished',manifest['test_completed'])
        '''))
        manifest=dict(python=sys.executable,source=str(source),source_sha256={},
            runtime_path=str(source),pardiso_runtime=str(source),target_mm=.25,jobs=[],
            test_completed=completed,test_exit=exit_code)
        (out/'manifest.json').write_text(json.dumps(manifest))
        return out

    def test_foreground_launch_waits_and_propagates_outcomes(self):
        import os,time
        from concurrent.futures import ThreadPoolExecutor
        from tools import run_material_queue as queue
        for completed,exit_code,expected in ((True,0,0),(False,0,1),(False,7,7)):
            with self.subTest(completed=completed,exit_code=exit_code),tempfile.TemporaryDirectory() as directory:
                out=self._prepared_queue_worker(directory,completed=completed,exit_code=exit_code)
                argv=['queue','--output',str(out),'--launch-prepared','--foreground']
                with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()),ThreadPoolExecutor(1) as pool:
                    future=pool.submit(queue.main)
                    try:
                        deadline=time.monotonic()+5
                        while not (out/'worker_ready.json').exists() and time.monotonic()<deadline:
                            time.sleep(.01)
                        self.assertTrue((out/'worker_ready.json').exists())
                        self.assertFalse(future.done())
                        if hasattr(os,'getsid'):
                            self.assertEqual(json.loads((out/'worker_ready.json').read_text())['session'],os.getsid(0))
                    finally:(out/'release_worker').touch()
                    self.assertEqual(future.result(timeout=5),expected)
                self.assertTrue(json.loads((out/'launch.json').read_text())['foreground'])

    def test_foreground_resume_releases_control_lock_and_can_pause(self):
        import time
        from concurrent.futures import ThreadPoolExecutor
        from tools import run_material_queue as queue
        with tempfile.TemporaryDirectory() as directory:
            out=self._prepared_queue_worker(directory)
            (out/'queue_status.json').write_text(json.dumps(dict(status='paused',jobs=[])))
            argv=['queue','--output',str(out),'--resume-queue','--foreground']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()),ThreadPoolExecutor(1) as pool:
                future=pool.submit(queue.main)
                active=None
                try:
                    deadline=time.monotonic()+5
                    while time.monotonic()<deadline:
                        active=queue.control.active(out)
                        if (active/'worker_ready.json').exists():break
                        time.sleep(.01)
                    self.assertNotEqual(active,out)
                    self.assertTrue((active/'worker_ready.json').exists())
                    self.assertFalse(future.done())
                    with queue.control.control_lock(out):pass
                    self.assertEqual(queue.control.pause(out)['status'],'paused')
                    self.assertEqual(future.result(timeout=5),2)
                finally:
                    if active is not None:(active/'release_worker').touch()

    def test_foreground_preserves_launch_gates_and_slurm_rejects_detach(self):
        import os
        from tools import run_material_queue as queue
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(queue.subprocess,'Popen') as popen:
                with self.assertRaisesRegex(ValueError,'short evidence'):
                    queue.launch(out,dict(case_config='cases.json',target_mm=2.),foreground=True)
                with patch.dict(os.environ,SLURM_JOB_ID='123'):
                    with self.assertRaisesRegex(RuntimeError,'--foreground'):
                        queue.launch(out,{})
                popen.assert_not_called()

    def test_actual_angle_experiments_do_not_leak_to_other_cases(self):
        from tools.run_material_queue import make_command
        m=dict(python='python',solve_seconds_per_case=21600,experiment='57.csv',experiment_angle=57,
               image_estimates='33.csv',image_angle=33)
        for angle in (0,33,57):
            cmd=make_command(m,dict(angle=angle,target_mm=2,output='new'))
            self.assertEqual('--experiment' in cmd,angle==57)
            self.assertEqual('--image-estimates' in cmd,angle==33)
            self.assertNotIn('--resume',cmd)
        m['experiments']={'0':'0.csv','33':'33.csv','57':'57_new.csv'}
        for angle in (0,33,57):
            cmd=make_command(m,dict(angle=angle,target_mm=.5,output='new'))
            self.assertEqual(cmd[cmd.index('--experiment')+1],m['experiments'][str(angle)])

    def test_queue_forwards_common_performance_policy_without_resume(self):
        from tools.run_material_queue import make_command
        from prototype.material_performance import performance_policy
        m=dict(python='python',solve_seconds_per_case=100,threads=2,
               diagnostics=True,
               performance_policy=performance_policy(material_batch_size=2048,linear_backend='pardiso'))
        for angle in (0,33,57):
            command=make_command(m,dict(angle=angle,target_mm=2,output='unused',resume='ignored'))
            self.assertNotIn('--resume',command)
            self.assertIn('--diagnostics',command)
            for flag,value in (('--material-batch-size','2048'),('--linear-backend','pardiso'),
                               ('--linear-threads','4'),('--threads','2')):
                self.assertEqual(command[command.index(flag)+1],value)
        m['performance_policy']=performance_policy()
        command=make_command(m,dict(angle=57,target_mm=2,output='unused'))
        self.assertGreater(int(command[command.index('--linear-threads')+1]),0)
        command=make_command(m,dict(angle=57,target_mm=.5,output='new',resume='foreign_legacy',resume_source='qualified_own_prefix'))
        self.assertEqual(command[command.index('--resume')+1],'qualified_own_prefix')
        self.assertNotIn('foreign_legacy',command)

    def test_comparison_inputs_reject_mutation(self):
        import hashlib
        from tools.run_material_queue import verify_inputs
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'state';p.write_bytes(b'accepted checkpoint')
            m=dict(input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()})
            verify_inputs(m);p.write_bytes(b'changed checkpoint')
            with self.assertRaises(RuntimeError):verify_inputs(m)


    def test_force_csv_units_and_full_range_comparison(self):
        from tools.run_material_candidate import load_force_csv,compare
        import csv
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);p=out/'experiment.csv'
            with p.open('w',encoding='utf-8',newline='') as stream:
                w=csv.writer(stream)
                w.writerow(['Depth (\u03bcm)','Fz (N)','depth_unit','force_unit','curve_type','component','speed_um_s'])
                for d in (0,500,1000,1500,2000):w.writerow([d,.5+d/100,'\u03bcm','N','experiment','Fz',200])
            d,f,v=load_force_csv(p)
            self.assertEqual(v,.2);self.assertEqual(f[-1],20.5);self.assertEqual(d[-1],2.)
            r=dict(status='completed',signature=dict(config={'inclination_deg':60.},loading_rate=.2),
                   steps=[dict(depth_mm=x,reaction_fz_n=10*x) for x in d])
            s=compare(out,r,experiment=p)['experiment']
            self.assertEqual(s['common_end_mm'],2.);self.assertAlmostEqual(s['rmse_n'],.5)
            self.assertAlmostEqual(s['force_increment_rmse_n'],0.)
            self.assertEqual(len(s['interval_stiffness']),4)
            self.assertFalse(s['experimental_validation'])
            r['signature']['config']['inclination_deg']=57.
            with self.assertRaises(ValueError):compare(out,r,experiment=p)
            explicit=compare(out,r,experiment=p,experiment_angle=57.)['experiment']
            self.assertAlmostEqual(explicit['rmse_n'],.5)
            p.write_text(p.read_text(encoding='utf-8').replace(',N,',',V,'),encoding='utf-8')
            with self.assertRaises(ValueError):load_force_csv(p)

    def test_single_case_command_preserves_explicit_material_and_experiment(self):
        from tools.run_material_queue import make_command
        m=dict(python='python',solve_seconds_per_case=21600,experiment='local.csv',
               material=dict(equilibrium_shear=.14,bulk_modulus=23.3433333333333,
                             maxwell_shear=1.24,relaxation_s=1.7,loading_rate=.2))
        cmd=make_command(m,dict(angle=60,target_mm=2,output='new',resume='own_pilot'))
        for flag,value in (('--equilibrium-shear','.14'),('--maxwell-shear','1.24'),('--relaxation-s','1.7')):
            self.assertAlmostEqual(float(cmd[cmd.index(flag)+1]),float(value))
        self.assertIn('--bulk-modulus',cmd);self.assertIn('--experiment',cmd)

    def test_common_friction_is_forwarded_to_each_fresh_angle(self):
        from tools.run_material_queue import make_command
        m=dict(python='python',solve_seconds_per_case=7200,mu=.30)
        for angle in (60,30):
            cmd=make_command(m,dict(angle=angle,target_mm=.5,output='new',resume=None))
            self.assertEqual(float(cmd[cmd.index('--mu')+1]),.30)
            self.assertNotIn('--resume',cmd)

    def test_queue_records_independent_jobs_without_copying_sources(self):
        from tools import run_material_queue as queue
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'queue'
            argv=['queue','--output',str(out),'--fresh','--angles','60','30',
                  '--mu','.30','--depth','.5','--seconds','7200']
            with patch('sys.argv',argv),patch.object(queue.subprocess,'Popen') as popen,\
                 patch.dict(queue.os.environ,SLURM_JOB_ID=''),\
                 patch.object(queue.control,'identity',return_value={'pid':123}),\
                 contextlib.redirect_stdout(io.StringIO()):
                popen.return_value.pid=123
                queue.main()
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['mu'],.30)
            self.assertEqual([j['angle'] for j in manifest['jobs']],[60,30])
            self.assertTrue(all(j['resume'] is None and j['mu']==.30 for j in manifest['jobs']))
            self.assertFalse((out/'source').exists())
            self.assertEqual(manifest['material']['equilibrium_shear'],.14)
            self.assertEqual(manifest['material']['maxwell_shear'],1.24)
            self.assertEqual(manifest['material']['relaxation_s'],1.7)

    def test_changed_mu_cannot_be_reported_as_requested_completed_case(self):
        from tools.run_material_queue import summarize_case
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'states').mkdir()
            rows=[dict(depth_mm=d,reaction_fz_n=1.,residual_n=1e-5,min_j=.9,
                       minimum_distance_mm=.001) for d in (0.,.01)]
            report=dict(status='completed',completed=True,target_mm=.01,force_tolerance_n=1e-4,
                signature=dict(config=dict(inclination_deg=60,friction_coefficient=.35)),
                settings=dict(increment=.01),steps=rows)
            (out/'result.json').write_text(json.dumps(report))
            (out/'audit.json').write_text(json.dumps(dict(passed=True,states=2,
                rows=[dict(passed=True),dict(passed=True)])))
            for i in range(2):(out/f'states/{i:04d}.npz').touch()
            result=summarize_case(out,dict(angle=60,mu=.30,target_mm=.01),0)
            self.assertFalse(result['mechanics_completed'])
            self.assertFalse(result['requested_mu_matches'])
    def test_resume_allows_only_runner_revision_and_identical_physics(self):
        from tools.run_material_candidate import resume_compatibility
        old=dict(signature={'angle':60.,'G':.47,'mesh':'same'},source_sha256={
            'tools/run_material_candidate.py':'old','mechanics/finite_strain.py':'physical'})
        new=copy.deepcopy(old);new['source_sha256']['tools/run_material_candidate.py']='new'
        self.assertTrue(resume_compatibility(old,new)['physical_sources_identical'])
        for candidate in (
            dict(new,signature=dict(new['signature'],angle=30.)),
            dict(new,solver_policy=dict(version=1,predictor='affine',affine_stagnation_window=0)),
            dict(new,source_sha256=dict(new['source_sha256'],**{'mechanics/finite_strain.py':'changed'})),
            dict(new,source_sha256={'tools/run_material_candidate.py':'new'}),
        ):
            with self.assertRaises(ValueError):resume_compatibility(old,candidate)

    def test_angle_command_does_not_reuse_60_degree_history_or_force_data(self):
        from tools.run_material_queue import make_command
        m=dict(python='python',solve_seconds_per_case=21600,reference='ref60',image_estimates='image60')
        job=dict(angle=60,target_mm=2,output='out60',resume='history60')
        cmd=make_command(m,job)
        self.assertNotIn('--resume',cmd);self.assertIn('--reference',cmd)
        for angle in (30,0):
            cmd=make_command(m,dict(angle=angle,target_mm=2,output=f'out{angle}'))
            self.assertNotIn('--resume',cmd);self.assertNotIn('--reference',cmd)
            self.assertNotIn('--image-estimates',cmd)
            self.assertEqual(cmd[cmd.index('--angle')+1],str(angle))

    def test_exited_running_result_or_failed_audit_is_not_completion(self):
        from tools.run_material_queue import summarize_case
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'states').mkdir()
            rows=[dict(depth_mm=d,reaction_fz_n=10.,residual_n=1e-5,min_j=.4,minimum_distance_mm=.001) for d in (0.,1.,2.)]
            row=rows[-1]
            r=dict(status='completed',completed=True,target_mm=2.,force_tolerance_n=1e-4,
                signature={'config':{'inclination_deg':60}},settings=dict(increment=1.),steps=rows)
            audit=dict(passed=True,states=3,rows=[dict(passed=True) for _ in rows])
            for i in range(3):(folder/f'states/{i:04d}.npz').write_bytes(b'fixture checkpoint presence; no physical audit here')
            (folder/'audit.json').write_text(json.dumps(audit))
            def check():
                (folder/'result.json').write_text(json.dumps(r))
                return summarize_case(folder,dict(angle=60,target_mm=2),0)
            self.assertTrue(check()['mechanics_completed'])
            r['status']='running';self.assertFalse(check()['mechanics_completed'])
            r['status']='completed';row['residual_n']=.01
            self.assertFalse(check()['mechanics_completed'])
            row['residual_n']=1e-5;audit['passed']=False
            (folder/'audit.json').write_text(json.dumps(audit))
            self.assertFalse(check()['mechanics_completed'])
            audit['passed']=True;(folder/'audit.json').write_text(json.dumps(audit))
            rows[1]['depth_mm']=.9
            self.assertFalse(check()['mechanics_completed'])

    def test_queue_advances_after_case_exception_and_postprocessing_failure(self):
        from tools import run_material_queue as queue
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            manifest=dict(target_mm=2.,jobs=[dict(name=str(a),angle=a,output=str(out/str(a)),status='pending') for a in (60,30,0)])
            (out/'manifest.json').write_text(json.dumps(manifest))
            called=[]
            def run_case(out,manifest,job,state):
                called.append(job['angle'])
                if job['angle']==60:raise RuntimeError('simulated child failure')
                return dict(status='completed',mechanics_completed=True,last_depth_mm=2.)
            with patch.object(queue,'run_case',side_effect=run_case),\
                 patch.object(queue,'export_summary',side_effect=RuntimeError('plot failed')),\
                 contextlib.redirect_stdout(io.StringIO()):queue.worker(out)
            self.assertEqual(called,[60,30,0])
            state=json.loads((out/'queue_status.json').read_text())
            self.assertEqual(state['status'],'finished');self.assertFalse(state['all_cases_completed'])
            self.assertEqual(state['jobs'][0]['status'],'worker_case_failed')
            self.assertEqual(state['jobs'][-1]['status'],'completed')

    def test_comparison_uses_whole_two_mm_and_rejects_wrong_angle(self):
        from tools.run_material_candidate import compare
        rows=[dict(depth_mm=d,reaction_fz_n=d*d+.1) for d in (0.,.5,1.,1.5,2.)]
        report=dict(status='completed',steps=rows,signature={'config':{'inclination_deg':30.}})
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            result=compare(out,report)
            self.assertEqual(result['candidate']['horizon_mm'],2.)
            self.assertAlmostEqual(result['candidate']['half_depth_ratio'],.25)
            ref=out/'reference.json'
            ref.write_text(json.dumps(dict(config={'inclination_deg':60.},steps=rows)))
            with self.assertRaises(ValueError):compare(out,report,ref)


if __name__=='__main__':unittest.main()

class FrictionlessEntryTests(unittest.TestCase):
    def test_cli_rejects_invalid_mu_and_zero_reaches_construction(self):
        from tools.run_material_candidate import main
        import sys
        base = ['run_material_candidate', '--maxwell-shear', '1.24', '--relaxation-s', '1.7', '--loading-rate', '.2']
        with tempfile.TemporaryDirectory() as folder:
            for i, mu in enumerate(('-1', 'nan', 'inf', '0')):
                args = base+['--output', str(Path(folder)/str(i)), '--mu', mu]
                with patch.object(sys, 'argv', args), contextlib.redirect_stderr(io.StringIO()):
                    if mu != '0':
                        with self.assertRaises(SystemExit) as error:
                            main()
                        self.assertEqual(error.exception.code, 2)
                    else:
                        with patch('tools.run_material_candidate.make_problem', side_effect=RuntimeError('construction reached')) as make:
                            main()
                        self.assertEqual(make.call_args.kwargs['mu'], 0.)

    def test_zero_mu_uses_existing_frictionless_solve_and_audits_history(self):
        import time
        from prototype.material_case import audit, signature
        p = ViscoelasticProblem(maxwell_shear=1.24, relaxation_s=1.7, loading_rate=.2,
            case='patch', mu=0., local_spacing=.2, surface_spacing=.1, enforce_physical_chord=True)
        self.assertIsNone(p.lag(p.rest))
        with contextlib.redirect_stdout(io.StringIO()):
            x, record = p.step(p.rest.copy(), p.start, 0., time.monotonic()+30.)
        self.assertEqual(record['friction_work_n_mm'], 0.)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder);(out/'states').mkdir()
            np.savez(out/'states'/'0000.npz', reference_xy=p.rest, triangles=p.mesh.triangles,
                body_nodes=p.n, positions=x, previous_positions=p.previous, depth_mm=0., **p.checkpoint_data())
            report = dict(signature=signature(p), source_sha256={}, settings={}, steps=[record])
            result = audit(p, out, report)
        self.assertTrue(result['passed'])
        self.assertEqual(result['rows'][0]['blade_boundary_check'],
                         'exact full accepted-increment replay')
        self.assertEqual(result['rows'][0]['blade_replay_max_error_mm'],0.)
        for mu in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ViscoelasticProblem(maxwell_shear=1.24, relaxation_s=1.7, loading_rate=.2, mu=mu)


def test_remote_pause_binds_to_attempt_without_inspecting_local_pid(tmp_path, monkeypatch):
    import pytest
    import os
    from tools import material_queue_control as control
    from run_support import write_json
    owner=dict(pid=os.getpid(),created_epoch=1.,hostname='remote-compute-node',
               slurm_job_id='456',python='/env/bin/python')
    write_json(tmp_path/'launch.json',dict(identity=owner,foreground=True,manifest=str(tmp_path/'manifest.json')))
    write_json(tmp_path/'queue_status.json',dict(status='running',jobs=[]))
    monkeypatch.setattr(control,'identity',lambda pid: pytest.fail('remote PID inspected'))
    assert control.status(tmp_path)['status']=='remote_reported_running'
    assert control.pause(tmp_path)['pause_requested']
    assert control.read(tmp_path/'pause_requested.json')['owner']['slurm_job_id']=='456'
    write_json(tmp_path/'queue_status.json',dict(status='paused',jobs=[]))
    with pytest.raises(RuntimeError,match='reaped'):
        control.restart(tmp_path,None)
    write_json(tmp_path/'worker_exit.json',dict(identity=owner,directory=str(tmp_path),returncode=0))
    assert control.status(tmp_path)['worker_reaped']


def test_mesh_prerequisite_holds_finer_cases_when_h1_review_fails(tmp_path):
    from unittest.mock import patch
    from tools import run_material_queue as queue
    from run_support import write_json
    jobs=[dict(name=name,angle=33,status='queued',output=str(tmp_path/name)) for name in ('H1','H0','H2')]
    for job in jobs[1:]:job['requires_case']='H1'
    write_json(tmp_path/'manifest.json',dict(target_mm=.25,mu=.35,jobs=jobs))
    with patch.object(queue,'run_case',return_value=dict(status='export_failed',mechanics_completed=True)) as run, \
            patch.object(queue,'export_summary'),patch.object(queue,'write_snapshot'):
        queue.worker(tmp_path)
    assert run.call_count==1
    assert [j['status'] for j in queue.read(tmp_path/'queue_status.json')['jobs']]==[
        'export_failed','prerequisite_failed','prerequisite_failed']


def test_numpy_window_balance_is_serializable_at_the_gate():
    import json
    import numpy as np
    from tools.run_material_queue import passes_window_balance
    row=dict(max_free_L2_N=np.float64(1.e-4), max_force_fraction=.001, max_moment_fraction=.001,
             max_window_energy_fraction=.01, end_cumulative_energy_fraction=np.float64(.01))
    assert json.dumps(passes_window_balance(row))=='true'
    row['end_cumulative_energy_fraction']=np.nextafter(.01,1.)
    assert passes_window_balance(row) is False


def test_crashed_foreground_worker_persists_terminal_status(tmp_path):
    from unittest.mock import Mock
    from tools import run_material_queue as queue
    from run_support import write_json
    write_json(tmp_path/'launch.json',dict(identity={'pid':123}))
    write_json(tmp_path/'queue_status.json',dict(status='running',jobs=[{'name':'H1'}]))
    process=Mock()
    process.wait.return_value=1
    assert queue.wait_for_worker(tmp_path,process)==1
    state=queue.read(tmp_path/'queue_status.json')
    assert state['status']=='system_error' and state['jobs']==[{'name':'H1'}]


def test_material_interval_distance_includes_interior_gap_and_missing():
    import pytest
    from tools.verify_intact_numerics import interval_distance_mm
    assert interval_distance_mm([[0.,1.]],[[0.,.2],[.6,1.]])==pytest.approx(.2)
    assert interval_distance_mm([],[[0.,1.]]) is None
