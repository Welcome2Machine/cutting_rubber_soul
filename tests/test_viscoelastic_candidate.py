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

    def test_failed_step_never_commits_metric_or_dissipation(self):
        p=ViscoelasticProblem.__new__(ViscoelasticProblem)
        p.loading_rate=.2;p.material_metric=np.eye(3)[None];p.material_dissipation=0.
        def fail(*args):raise LoadStepFailure('rejected physical equilibrium')
        with patch.object(IPCProblem,'step',side_effect=fail):
            with self.assertRaises(LoadStepFailure):p.step(np.zeros((1,2)),.1,.12,0.)
            np.testing.assert_array_equal(p.material_metric,np.eye(3)[None])
            self.assertEqual(p.material_dissipation,0.)
            with self.assertRaises(LoadStepFailure):p.step(np.zeros((1,2)),.1,.11,0.)
            self.assertAlmostEqual(p.material_dt,.05)
            np.testing.assert_array_equal(p.previous_material_metric,np.eye(3)[None])


class MaterialRunControlTests(unittest.TestCase):
    def test_actual_angle_experiments_do_not_leak_to_other_cases(self):
        from tools.run_material_queue import make_command
        m=dict(python='python',solve_seconds_per_case=21600,experiment='57.csv',experiment_angle=57,
               image_estimates='33.csv',image_angle=33)
        for angle in (0,33,57):
            cmd=make_command(m,dict(angle=angle,target_mm=2,output='new'))
            self.assertEqual('--experiment' in cmd,angle==57)
            self.assertEqual('--image-estimates' in cmd,angle==33)
            self.assertNotIn('--resume',cmd)

    def test_queue_forwards_common_performance_policy_without_resume(self):
        from tools.run_material_queue import make_command
        from prototype.material_performance import performance_policy
        m=dict(python='python',solve_seconds_per_case=100,threads=2,
               performance_policy=performance_policy(material_batch_size=2048,linear_backend='pardiso'))
        for angle in (0,33,57):
            command=make_command(m,dict(angle=angle,target_mm=2,output='unused',resume='ignored'))
            self.assertNotIn('--resume',command)
            for flag,value in (('--material-batch-size','2048'),('--linear-backend','pardiso'),
                               ('--linear-threads','4'),('--threads','2')):
                self.assertEqual(command[command.index(flag)+1],value)
        m['performance_policy']=performance_policy()
        command=make_command(m,dict(angle=57,target_mm=2,output='unused'))
        self.assertGreater(int(command[command.index('--linear-threads')+1]),0)

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
