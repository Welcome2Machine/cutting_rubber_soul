import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.model_config import FEMConfig
from prototype.material import response, from_config
from mechanics.finite_strain import principal
from prototype.directional_fracture import (relaxed_response, strength_calibration, RationalAT1,
    best_orientation, free_uniaxial_release, free_normal_stretch, diagonal_energy)
from prototype.ipc_cutting import (DirectionalFractureProblem, check_step_budget,
                                   EnergyAuditError, check_model_boundary, ModelBoundaryReached, supervise,
                                   bisected_target, run_case)
from prototype.ipc_reference import IPCProblem, toolkit


class RelaxationTests(unittest.TestCase):
    def test_uncoupled_mixed_pressure_cannot_be_silently_used_with_damage(self):
        with self.assertRaisesRegex(ValueError, 'mixed damage/pressure'):
            DirectionalFractureProblem(volumetric_discretization='macro_mixed')

    def setUp(self): self.c = FEMConfig()

    def test_open_closed_and_parallel_traction(self):
        F = np.array([np.diag([1.4,1.]),np.diag([.8,1.]),np.diag([1.,1.4])])
        N = np.tile([1.,0.],(3,1)); q = relaxed_response(F,N,self.c)
        np.testing.assert_allclose(q['P'][0],0.,atol=1e-10)
        np.testing.assert_allclose(q['P'][1,0,0],response(F,self.c)['P'][1,0,0],atol=1e-10)
        self.assertLess(q['P'][1,0,0],0.)
        self.assertGreater(q['P'][2,1,1],0.)
        self.assertAlmostEqual(q['P'][2,0,0],0.,places=9)

    def test_shear_released_without_relaxing_compression(self):
        F=np.array([[[.8,0.],[.4,1.]]]); q=relaxed_response(F,np.array([[1.,0.]]),self.c)
        self.assertLess(q['P'][0,0,0],0.)
        self.assertAlmostEqual(q['P'][0,1,0],0.,places=9)
        self.assertLess(q['energy'][0],response(F,self.c)['energy'][0])

    def test_isotropic_inplane_compression_has_no_release(self):
        angles=np.arange(12)*np.pi/12
        N=np.column_stack((np.cos(angles),np.sin(angles)))
        F=np.tile(.8*np.eye(2),(len(N),1,1))
        np.testing.assert_allclose(relaxed_response(F,N,self.c)['energy'],response(F,self.c)['energy'],atol=1e-11)

    def test_frame_indifference_and_crack_normal_sign(self):
        F=np.array([[[1.3,.25],[.1,.9]],[[.7,.1],[0.,1.1]]]);N=np.tile([1.,0.],(2,1))
        Q=np.array([[np.cos(.6),-np.sin(.6)],[np.sin(.6),np.cos(.6)]])
        q=relaxed_response(F,N,self.c);r=relaxed_response(Q@F,N,self.c)
        np.testing.assert_allclose(q['energy'],r['energy'],atol=1e-11)
        np.testing.assert_allclose(Q@q['P'],r['P'],atol=1e-10)
        np.testing.assert_allclose(q['P'],relaxed_response(F,-N,self.c)['P'],atol=1e-10)
        np.testing.assert_allclose(q['cauchy'],q['cauchy'].transpose(0,2,1),atol=1e-10)

    def test_analytic_energy_stress_and_tangent(self):
        F=np.array([[[1.3,.25],[.1,.9]],[[.7,.1],[0.,1.1]]]);N=np.tile([1.,0.],(2,1))
        v=np.array([[[.13,-.2],[.3,.16]],[[-.1,.2],[.12,-.15]]]);eps=1e-5
        q=relaxed_response(F,N,self.c);p=relaxed_response(F+eps*v,N,self.c);m=relaxed_response(F-eps*v,N,self.c)
        np.testing.assert_allclose((p['energy']-m['energy'])/(2*eps),np.einsum('eij,eij->e',q['P'],v),atol=1e-8)
        np.testing.assert_allclose((p['P']-m['P']).reshape(-1,4)/(2*eps),np.einsum('eij,ej->ei',q['tangent'],v.reshape(-1,4)),atol=1e-7)
        np.testing.assert_allclose(q['tangent'],q['tangent'].transpose(0,2,1),atol=1e-12)

    def test_nominal_strength_calibration_not_cauchy_or_plane_strain(self):
        q=strength_calibration(self.c)
        stretches=np.array([q['calibration_axial_stretch'],q['calibration_transverse_stretch'],q['calibration_transverse_stretch']])
        _,tau,_=principal(np.log(stretches)[None],from_config(self.c))
        np.testing.assert_allclose(tau[0]/stretches,[4.4,0.,0.],atol=1e-7)
        self.assertNotAlmostEqual(stretches[0],4.2,places=1)
        self.assertGreater(q['critical_release_mpa'],0.)

    def test_shared_orientation_search_spatial_objectivity_and_seed(self):
        F=np.array([[[1.3,.25],[.1,.9]],[[.7,.1],[0.,1.1]]]);N=np.tile([1.,0.],(2,1))
        intact=response(F,self.c);seed=relaxed_response(F,N,self.c)['energy']
        energy,normal=best_orientation(F,intact['cauchy'],N,seed,self.c)
        Q=np.array([[np.cos(.6),-np.sin(.6)],[np.sin(.6),np.cos(.6)]])
        rotated=response(Q@F,self.c)
        re,rn=best_orientation(Q@F,rotated['cauchy'],N,seed,self.c)
        np.testing.assert_allclose(re,energy,atol=1e-11)
        np.testing.assert_allclose(rn,normal,atol=1e-11)
        self.assertTrue(np.all(energy<=seed+1e-12))
        np.testing.assert_array_equal(N,np.tile([1.,0.],(2,1)))

    def test_continuous_free_uniaxial_maximum_matches_independent_angle_sweep(self):
        G,K=from_config(self.c).moduli()
        for stretch in (1.001,1.5,3.,4.2,7.8625,12.):
            state=free_uniaxial_release(self.c,[stretch]);t=state['transverse_stretch'][0]
            angle=np.linspace(0.,np.pi/2,10001)
            a=np.sqrt(stretch**2*np.sin(angle)**2+t*t*np.cos(angle)**2);b=stretch*t/a
            W=diagonal_energy(stretch,t,t,G,K)[0]
            grid=W-diagonal_energy(a,np.minimum(b,free_normal_stretch(a,t,G,K)),t,G,K)[0]
            exact=state['maximum_release_mpa'][0]
            self.assertGreaterEqual(exact+1e-11,grid.max())
            self.assertLess(exact-grid.max(),2e-6)
            self.assertLess(abs(angle[grid.argmax()]-state['maximizing_normal_angle_rad'][0]),np.pi/20000)

    def test_uniaxial_path_derivatives_and_first_onset_not_fixed_axial_normal(self):
        q=strength_calibration(self.c);target=q['calibration_axial_stretch'];Yc=q['critical_release_mpa']
        path=free_uniaxial_release(self.c,np.exp(np.linspace(0.,np.log(target),513)))
        self.assertTrue(np.all(np.diff(path['maximum_release_mpa'])>0.))
        self.assertTrue(np.all(np.diff(path['axial_normal_release_mpa'])>0.))
        self.assertTrue(np.all(path['maximum_release_mpa'][:-1]<Yc))
        self.assertAlmostEqual(path['maximum_release_mpa'][-1],Yc,places=10)
        self.assertAlmostEqual(q['legacy_first_onset_nominal_mpa'],3.4990857541,places=7)
        legacy=free_uniaxial_release(self.c,[q['legacy_first_onset_stretch']])
        self.assertAlmostEqual(legacy['maximum_release_mpa'][0],q['legacy_axial_normal_threshold_mpa'],places=8)
        self.assertAlmostEqual(q['catalogue_breaking_nominal_mpa'],2.5634110414,places=8)
        self.assertAlmostEqual(Yc,16.9923272986,places=7)
        for stretch in (1.2,4.2,target):
            eps=1e-5;state=free_uniaxial_release(self.c,[stretch*np.exp(-eps),stretch,stretch*np.exp(eps)])
            for value,derivative in [('maximum_release_mpa','release_log_stretch_derivative_mpa'),
                                      ('nominal_stress_mpa','nominal_log_stretch_derivative_mpa')]:
                self.assertAlmostEqual((state[value][2]-state[value][0])/(2*eps),state[derivative][1],places=7)

    def test_continuous_calibration_is_coordinate_invariant_and_sampling_error_explicit(self):
        q=strength_calibration(self.c);stretch=q['calibration_axial_stretch'];Yc=q['critical_release_mpa']
        aligned=free_uniaxial_release(self.c,[stretch],orientation_samples=16)
        rotated=free_uniaxial_release(self.c,[stretch],orientation_samples=16,loading_angle=np.pi/32)
        fine=free_uniaxial_release(self.c,[stretch],orientation_samples=128,loading_angle=np.pi/32)
        self.assertEqual(aligned['maximum_release_mpa'][0],rotated['maximum_release_mpa'][0])
        self.assertLess(1-aligned['sampled_release_mpa'][0]/Yc,4e-5)
        self.assertGreater(1-rotated['sampled_release_mpa'][0]/Yc,.029)
        self.assertLess(1-fine['sampled_release_mpa'][0]/Yc,4e-5)


class RationalPhaseTests(unittest.TestCase):
    def make(self,ell=.04):
        return RationalAT1([[0.,0.],[1.,0.],[0.,1.]],[[0,1,2]],20.,2.15,ell,critical_release_mpa=9.5)

    def test_independent_onset_first_variation(self):
        for ell in (.02,.04,.1):
            p=self.make(ell);d=np.zeros(3)
            for factor in (.99,1.,1.01):
                g=p.terms(d,np.array([9.5*factor]))[1]/p.lumped
                np.testing.assert_allclose(g,3*p.gc/(8*ell)*(1-factor),atol=1e-10)

    def test_intact_nodal_stability_uses_exact_weighted_release_and_admissible_gradient(self):
        p=RationalAT1([[0.,0.],[1.,0.],[1.,1.],[0.,1.]],[[0,1,2],[0,2,3]],20.,2.15,.04,
                      critical_release_mpa=9.5)
        Y=np.array([19.,0.]);state=p.intact_stability(Y)
        np.testing.assert_allclose(state['nodal_release_mpa'],[9.5,19.,9.5,0.],atol=1e-12)
        np.testing.assert_allclose(state['ratio'],[1.,2.,1.,0.],atol=1e-12)
        np.testing.assert_allclose(state['normalized_gradient_mpa'],p.terms(np.zeros(4),Y)[1]/p.lumped)
        self.assertAlmostEqual(state['kkt_residual_mpa'],p.kkt_residual(np.zeros(4),Y,np.zeros(4)))
        eps=1e-7;direction=np.array([0.,1.,0.,0.])
        self.assertLess(p.terms(eps*direction,Y)[0],0.)
        with self.assertRaises(ValueError):p.intact_stability(np.array([float('nan'),0.]))

    def test_element_spike_does_not_imply_discrete_damage_onset(self):
        p=RationalAT1([[0.,0.],[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]],
                      [[0,1,2],[0,2,3],[0,3,4],[0,4,1]],20.,2.15,.04,
                      critical_release_mpa=9.5)
        Y=np.array([1.5*9.5,0.,0.,0.]);state=p.intact_stability(Y)
        self.assertGreater(Y.max(),p.critical_release)
        self.assertTrue(np.all(state['ratio']<1.))
        self.assertEqual(state['kkt_residual_mpa'],0.)
        for node in range(p.count):
            perturbation=np.zeros(p.count);perturbation[node]=1e-7
            self.assertGreater(p.terms(perturbation,Y)[0],0.)

    def test_damage_derivatives_and_quadrature(self):
        p=self.make();d=np.array([.2,.3,.5]);v=np.array([.1,-.2,.3]);Y=np.array([12.]);eps=1e-5
        E,g,H,_=p.terms(d,Y);ep=p.terms(d+eps*v,Y);em=p.terms(d-eps*v,Y)
        self.assertAlmostEqual((ep[0]-em[0])/(2*eps),g@v,places=7)
        np.testing.assert_allclose((ep[1]-em[1])/(2*eps),H@v,atol=1e-7)
        released,fracture=p.energy(d,Y)
        self.assertAlmostEqual(fracture-released,E,places=10)
        self.assertAlmostEqual(released,float(p.volumes@((1-p.degradation(d))*Y)),places=10)

    def test_subcritical_damage_and_irreversibility(self):
        p=self.make();d,info=p.solve(np.array([9.4]));np.testing.assert_array_equal(d,0.)
        d,info=p.solve(np.array([12.]));self.assertGreater(d.min(),0.)
        lower=d.copy();d,_=p.solve(np.array([1.]),old=lower)
        np.testing.assert_array_equal(d,lower)
        self.assertLessEqual(p.kkt_residual(d,np.array([1.]),lower),1e-6)

    def test_endpoint_degradation(self):
        p=self.make();g,gp,gpp=p.law(np.array([0.,1.]))
        np.testing.assert_allclose(g,[1.,p.residual],atol=1e-15)
        self.assertAlmostEqual(gp[1],0.)
        self.assertTrue(np.isfinite(gpp).all())


class AcceptanceTests(unittest.TestCase):
    def test_energy_budget_excludes_solver_pseudopotential(self):
        previous=dict(reaction_fz_n=2.,stored_plus_fracture_n_mm=1.)
        current=dict(reaction_fz_n=4.,stored_plus_fracture_n_mm=1.2,friction_work_n_mm=.1,energy_n_mm=900.)
        current.update(conservative_endpoint_work_n_mm=.3,free_residual_work_n_mm=0.)
        q=check_step_budget(previous,current,.1)
        self.assertAlmostEqual(q['energy_balance_defect_n_mm'],0.)
        with self.assertRaises(EnergyAuditError):check_step_budget(previous,current,.2)

    def test_spring_with_changing_friction_load_is_not_energy_conserving_quadrature(self):
        # U=u^2, prescribed u: 1 -> 1.1, Coulomb resistance: .9 -> .3.
        # True smooth-path work=.27, friction=.06, delta U=.21.
        # Endpoint friction work=.03 cannot be mixed with trapezoidal work
        # and then required to yield equality, even with exact equilibrium.
        old=dict(reaction_fz_n=2.9,stored_plus_fracture_n_mm=1.)
        new=dict(reaction_fz_n=2.5,stored_plus_fracture_n_mm=1.21,
                 friction_work_n_mm=.03,conservative_endpoint_work_n_mm=.22,
                 free_residual_work_n_mm=0.)
        audit=check_step_budget(old,new,.1)
        self.assertAlmostEqual(audit['energy_balance_defect_n_mm'],.03)
        self.assertAlmostEqual(audit['endpoint_energy_surplus_n_mm'],.01)
        self.assertAlmostEqual(audit['endpoint_virtual_work_defect_n_mm'],0.)
        self.assertFalse(audit['energy_balance_resolved'])

    def test_energy_creation_bad_work_and_nonfinite_inputs_are_rejected(self):
        old=dict(reaction_fz_n=2.,stored_plus_fracture_n_mm=1.)
        new=dict(reaction_fz_n=2.,stored_plus_fracture_n_mm=1.3,
                 friction_work_n_mm=.1,conservative_endpoint_work_n_mm=.1,
                 free_residual_work_n_mm=0.)
        with self.assertRaisesRegex(EnergyAuditError,'energy creation'):
            check_step_budget(old,new,.1)
        for key in new:
            with self.subTest(key=key),self.assertRaises(EnergyAuditError):
                check_step_budget(old,{**new,key:float('nan')},.1)
        with self.assertRaisesRegex(EnergyAuditError,'virtual work'):
            check_step_budget(old,{**new,'conservative_endpoint_work_n_mm':0.},.1)
        with self.assertRaisesRegex(EnergyAuditError,'Negative friction'):
            check_step_budget(old,{**new,'friction_work_n_mm':-.1},.1)

    def test_minimum_increment_is_reached_without_roundoff_early_exit(self):
        old=.2725000000000001;target=.27375000000000005;minimum=.000625
        nxt=bisected_target(old,target,minimum)
        self.assertIsNotNone(nxt)
        self.assertAlmostEqual(nxt-old,minimum,places=15)
        self.assertIsNone(bisected_target(old,nxt,minimum))

    def test_model_boundary_is_not_silent_continuation(self):
        check_model_boundary(np.array([0.,.5]))
        with self.assertRaises(ModelBoundaryReached):check_model_boundary(np.array([.951]))

    def test_supervisor_records_completion_boundary_and_native_crash(self):
        for status,completed,code,expected in [('completed',True,0,'completed'),('model_boundary',False,1,'model_boundary'),('running',False,3221225477,'failed')]:
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory);(out/'case').mkdir()
                (out/'launch.json').write_text(json.dumps(dict(command=['mock'],source_root=directory,target_mm=3.)),encoding='utf-8')
                (out/'case/result.json').write_text(json.dumps(dict(status=status,completed=completed,last_depth_mm=.1)),encoding='utf-8')
                process=Mock(pid=12);process.wait.return_value=code
                with patch('prototype.ipc_cutting.subprocess.Popen',return_value=process):
                    state=supervise(out)
                self.assertEqual(state['status'],expected)
                self.assertEqual(state['exit_code'],code)
                if expected=='failed':
                    self.assertEqual(json.loads((out/'case/result.json').read_text())['status'],'failed')
                with self.assertRaisesRegex(ValueError,'already started'):supervise(out)


class NativeDirectionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:toolkit()
        except ModuleNotFoundError:raise unittest.SkipTest('Local IPC toolkit unavailable')

    def problem(self):
        p=DirectionalFractureProblem(case='patch',mu=.35,local_spacing=.2,surface_spacing=.1)
        p.ipc.set_num_threads(2);return p

    def test_intact_and_damaged_assembled_derivatives(self):
        with threadpool_limits(limits=2):
            p=self.problem();x=p.rest.copy();x[p.n:,1]+=1.
            x[:p.n]=x[:p.n]@np.array([[1.2,.1],[.1,1.1]]).T
            a=p.bulk_terms(x,True);b=IPCProblem.bulk_terms(p,x,True)
            for i in (0,1,2):np.testing.assert_array_equal(a[i],b[i])
            self.assertEqual(a[3:],b[3:])
            rng=np.random.default_rng(3);p.damage=rng.uniform(.1,.5,p.n)
            p.normals[:]=[np.cos(.3),np.sin(.3)]
            v=np.zeros_like(x);v[:p.n]=rng.normal(size=(p.n,2))*.01;eps=1e-5
            E,g,H,_=p.evaluate(x,tangent=True,projected=False)
            ep=p.evaluate(x+eps*v,tangent=False);em=p.evaluate(x-eps*v,tangent=False)
            self.assertAlmostEqual((ep[0]-em[0])/(2*eps),g@v.ravel(),places=7)
            np.testing.assert_allclose((ep[1]-em[1])/(2*eps),H@v.ravel(),rtol=2e-6,atol=2e-7)
            direction=rng.normal(size=p.n)*.1;original=p.damage.copy();Y=p.release(x)
            def total(d):
                p.damage=d
                return p.evaluate(x,tangent=False)[0]+p.phase.energy(d,Y)[1]
            derivative=(total(original+eps*direction)-total(original-eps*direction))/(2*eps)
            self.assertAlmostEqual(derivative,p.phase.terms(original,Y)[1]@direction,places=6)

    def test_continuous_patch_history_and_rejected_step_rollback(self):
        with threadpool_limits(limits=2),contextlib.redirect_stdout(io.StringIO()):
            p=self.problem();x=p.rest.copy();old=p.start
            for depth in (0.,.01,.02):
                previous=x.copy();lower=p.damage.copy()
                x,balance=p.advance(x,old,depth,time.monotonic()+30)
                np.testing.assert_array_equal(p.previous,previous)
                self.assertTrue(np.all(p.damage>=lower))
                self.assertLessEqual(balance['residual_n'],1e-4)
                self.assertLessEqual(balance['phase_kkt_residual_mpa'],1e-6)
                old=depth
            saved=p.snapshot_history()
            with patch.object(IPCProblem,'step',side_effect=RuntimeError('failure')):
                with self.assertRaisesRegex(RuntimeError,'failure'):p.advance(x,old,.03,time.monotonic()+10)
            for a,b in zip(saved,p.snapshot_history()):np.testing.assert_array_equal(a,b)

    def test_runner_keeps_accepted_predictor_history_through_energy_retry(self):
        calls=[]; rejected=False
        real_advance=DirectionalFractureProblem.advance
        def advance(p,x,old,depth,deadline,**kwargs):
            history=kwargs.get('previous_accepted')
            calls.append(dict(old=old,depth=depth,x=x.copy(),history=None if history is None
                              else (history[0],history[1].copy())))
            return real_advance(p,x,old,depth,deadline,**kwargs)
        def audit(previous,current,delta):
            nonlocal rejected
            if not rejected and abs(current['depth_mm']-.02)<1e-12:
                rejected=True
                raise EnergyAuditError('Injected post-solve rejection')
            return check_step_budget(previous,current,delta)
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            out=Path(folder)/'case'
            args=SimpleNamespace(output=out,case='patch',friction=.35,inclination=0.,
                local_spacing=.2,max_iterations=50,ell=.04,threads=1,seconds=30.,depth=.025,
                increment=.01,min_increment=.0025,no_visualize=True,predictor='secant',
                friction_linear_solver='retry-gmres')
            with patch.object(DirectionalFractureProblem,'advance',advance), \
                    patch('prototype.ipc_cutting.check_step_budget',audit):
                self.assertTrue(run_case(args))
            report=json.loads((out/'result.json').read_text())
            self.assertEqual(len(report['retries']),1)
            before=next(q for q in calls if abs(q['depth']-.02)<1e-12)
            retry=next(q for q in calls if abs(q['depth']-.015)<1e-12)
            self.assertEqual(before['history'][0],retry['history'][0])
            np.testing.assert_array_equal(before['history'][1],retry['history'][1])
            np.testing.assert_array_equal(before['x'],retry['x'])
            self.assertAlmostEqual(retry['history'][0],0.)
            with np.load(out/'states/0000.npz') as z:
                np.testing.assert_array_equal(retry['history'][1],z['positions'])
            rows=report['steps']+report['retries']
            for key in ('mechanical_solves','mechanical_iterations','factorizations'):
                self.assertEqual(report['work_totals'][key],sum(row['work'][key] for row in rows))
            self.assertGreater(report['retries'][0]['work']['mechanical_solves'],0)
            self.assertTrue(all(row['work']['wall_seconds']>0 for row in rows))

    def test_paired_matrices_reuse_bulk_without_caching_damage_or_geometry(self):
        with threadpool_limits(limits=2):
            p=self.problem();x=p.rest.copy()
            for damage,stretch in ((0.,1.01),(.3,1.1),(.6,1.2)):
                p.damage[:]=damage
                x[:p.n]=p.rest[:p.n]@np.diag([stretch,1.])
                lag=p.lag(x)
                exact=p.evaluate(x,lag,projected=False)
                metric=p.evaluate(x,lag,projected=True)
                with patch.object(p,'bulk_terms',wraps=p.bulk_terms) as bulk_call:
                    paired=p.evaluate(x,lag,projected=False,with_metric=True)
                    self.assertEqual(bulk_call.call_count,1)
                self.assertAlmostEqual(exact[0],paired[0],places=12)
                self.assertAlmostEqual(exact[3],paired[3],places=12)
                np.testing.assert_allclose(exact[1],paired[1],rtol=1e-12,atol=1e-12)
                for original,combined in ((exact[2],paired[2]),(metric[2],paired[4])):
                    np.testing.assert_allclose(original.toarray(),combined.toarray(),rtol=1e-12,atol=1e-12)
            with self.assertRaises(ValueError):p.evaluate(x,with_metric=True)
            with self.assertRaises(ValueError):p.evaluate(x,tangent=False,projected=False,with_metric=True)

    def test_positive_damage_coupling_and_fixed_reference_orientation(self):
        with threadpool_limits(limits=2),contextlib.redirect_stdout(io.StringIO()):
            p=self.problem()
            # Fully prescribed homogeneous deformation isolates the fracture
            # constitutive coupling from a separate free-equilibrium problem.
            p.free=np.array([],dtype=int)
            x=p.rest.copy();x[:p.n,0]*=8.;x[p.n:,1]+=100.
            final,balance=p.advance(x,0.,0.,time.monotonic()+30)
            self.assertGreater(p.damage.max(),.1)
            self.assertLessEqual(balance['phase_kkt_residual_mpa'],1e-6)
            normals=p.normals.copy();lower=p.damage.copy()
            final,balance=p.advance(final,0.,0.,time.monotonic()+30)
            self.assertTrue(np.all(p.damage>=lower))
            np.testing.assert_array_equal(p.normals,normals)
            np.testing.assert_array_equal(p.previous,final)

    def test_damage_direction_and_slip_roll_back_after_phase_failure(self):
        with threadpool_limits(limits=2),contextlib.redirect_stdout(io.StringIO()):
            p=self.problem();p.free=np.array([],dtype=int)
            x=p.rest.copy();x[:p.n,0]*=8.;x[p.n:,1]+=100.
            saved=p.snapshot_history()
            def fail(*args,**kwargs):
                p.damage[:]=.3;p.normals[:]=[0.,1.];p.previous+=.1
                raise RuntimeError('phase failure')
            with patch.object(p.phase,'solve',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'phase failure'):p.advance(x,0.,0.,time.monotonic()+10)
            for before,after in zip(saved,p.snapshot_history()):np.testing.assert_array_equal(before,after)
            np.testing.assert_array_equal(p.failed_damage,.3)

    def test_actual_rejected_energy_step_with_original_friction_history(self):
        source=Path(__file__).resolve().parents[1]/'outputs/directional60_3mm_20260913/case'
        if not (source/'last_rejected_trial.npz').exists():
            self.skipTest('Local 60 degree rejected-state fixture unavailable')
        with threadpool_limits(limits=2):
            p=DirectionalFractureProblem(mu=.35,inclination=60.,tip_radius=.02,dhat=.0025,
                local_spacing=.02,surface_spacing=.005,fine_halfwidth=3.,nose_facets=64,
                hessian_mode='regularized',max_iterations=500,ell_mm=.04)
            p.ipc.set_num_threads(2)
            report=json.loads((source/'result.json').read_text())
            # This is a legacy energy-audit regression, not a rerun with the
            # corrected strength law. Damage checkpoints retain their original
            # degradation curve through the calibration saved with the source.
            p.calibration=report['calibration'].copy()
            p.phase=RationalAT1(p.mesh.reference_xy,p.mesh.triangles,p.config.width_mm,
                p.config.fracture_energy_n_mm,report['parameters']['ell'],
                critical_release_mpa=p.calibration['critical_release_mpa'])
            self.assertAlmostEqual(p.phase.a,report['degradation_a'],places=12)
            with np.load(source/'last_rejected_trial.npz') as z:
                np.testing.assert_array_equal(z['reference_xy'],p.rest)
                np.testing.assert_array_equal(z['triangles'],p.mesh.triangles)
                x=z['positions'];p.previous=z['previous_positions'];p.damage=z['damage']
                p.normals=z['reference_normals'];depth=float(z['depth_mm'])
            with np.load(source/report['last_checkpoint']) as z:
                np.testing.assert_array_equal(p.previous,z['positions'])
            current=p.acceptance(x,depth,0,0)
            current.update(p.energy_state(x));current.update(p.incremental_work(x))
            audit=check_step_budget(report['steps'][-1],current,depth-report['last_depth_mm'])
            self.assertLess(current['residual_n'],1e-4)
            self.assertAlmostEqual(audit['energy_balance_defect_n_mm'],.00141855728845,places=10)
            self.assertLess(abs(audit['endpoint_virtual_work_defect_n_mm']),1e-10)
            self.assertFalse(audit['energy_balance_resolved'])
            self.assertGreater(audit['endpoint_energy_surplus_n_mm'],.0014)


class OrientationSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:toolkit()
        except ModuleNotFoundError:raise unittest.SkipTest('Local IPC toolkit unavailable')

    def problem(self):
        p=DirectionalFractureProblem(case='patch',mu=.35,local_spacing=.2,surface_spacing=.1)
        p.ipc.set_num_threads(2)
        x=p.rest.copy();x[:p.n]=x[:p.n]@np.array([[1.4,.2],[.0,.9]]).T
        p.normals[:]=[0.,1.]
        return p,x

    @staticmethod
    def legacy_search(p,x):
        # Independent full-domain reference from the pre-optimization algorithm.
        F=p.deformation(x);intact=response(F,p.config,tangent=False)
        effective=relaxed_response(F,p.normals,p.config,tangent=False)['energy']
        selected=p.normals.copy();free=~p.locked
        if np.any(free):
            f=F[free];best=effective[free].copy();normals=selected[free].copy()
            for angle in np.arange(p.orientation_samples)*np.pi/p.orientation_samples:
                N=np.tile([np.cos(angle),np.sin(angle)],(len(f),1))
                value=relaxed_response(f,N,p.config,tangent=False)['energy']
                better=value<best-1e-12
                best[better]=value[better];normals[better]=N[better]
            _,vectors=np.linalg.eigh(intact['cauchy'][free,:2,:2])
            N=np.einsum('eji,ej->ei',f,vectors[:,:,-1])
            N/=np.linalg.norm(N,axis=1)[:,None]
            value=relaxed_response(f,N,p.config,tangent=False)['energy']
            better=value<best-1e-12
            best[better]=value[better];normals[better]=N[better]
            selected[free]=normals;effective[free]=best
        return np.maximum(intact['energy']-effective,0.),selected

    def test_global_driving_search_still_includes_intact_body_away_from_tip(self):
        with threadpool_limits(limits=2):
            p,x=self.problem();before=p.normals.copy()
            expected,normals=self.legacy_search(p,x)
            tip=p.rest[p.n:][np.argmin(p.rest[p.n:,1])]
            far=int(np.argmax(np.linalg.norm(p.mesh.reference_xy[p.mesh.triangles].mean(axis=1)-tip,axis=1)))
            self.assertFalse(np.any(p.damage))
            actual=p.release(x,choose=True)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(p.normals,normals)
            self.assertGreater(np.linalg.norm(p.normals[far]-before[far]),.1)
            self.assertEqual(p.work_counters['orientation_search_elements'],len(p.mesh.triangles))

    def test_exact_certification_for_intact_locked_and_fresh_off_tip_damage(self):
        with threadpool_limits(limits=2):
            for case in ('intact','locked','fresh_off_tip','mixed'):
                with self.subTest(case=case):
                    p,x=self.problem()
                    if case=='locked':p.damage[:]=.3;p.locked[:]=True
                    if case=='fresh_off_tip':
                        tip=p.rest[p.n:][np.argmin(p.rest[p.n:,1])]
                        far=int(np.argmax(np.linalg.norm(p.mesh.reference_xy-tip,axis=1)))
                        p.damage[far]=.3
                    if case=='mixed':
                        p.damage[:]=np.random.default_rng(17).uniform(.1,.5,p.n)
                        p.locked[::2]=True
                    Y=p.release(x);best,_=self.legacy_search(p,x)
                    weights=p.phase.degradation(p.damage)
                    expected=float(np.max((1-weights)*np.abs(best-Y)))
                    history=p.snapshot_history()
                    with patch.object(p,'_best_orientation',wraps=p._best_orientation) as search:
                        actual=p.orientation_error(x,Y)
                    self.assertAlmostEqual(actual,expected,places=13)
                    for a,b in zip(history,p.snapshot_history()):np.testing.assert_array_equal(a,b)
                    count=int(np.count_nonzero((~p.locked)&(weights<1.)))
                    self.assertEqual(p.work_counters['orientation_certification_elements'],count)
                    if case in ('intact','locked'):
                        self.assertEqual(actual,0.);search.assert_not_called()
                    else:
                        self.assertGreater(actual,1e-5)
                        self.assertEqual(len(search.call_args.args[0]),count)
                    if case=='fresh_off_tip':self.assertLess(count,len(p.mesh.triangles))

    def test_certification_preserves_nonnegative_release_clamp(self):
        with threadpool_limits(limits=2):
            p,x=self.problem();x=p.rest.copy();p.damage[:]=.1
            # The existing release check allows tiny negative roundoff then
            # clamps it to zero; certification must keep this exact convention.
            def relaxed(F,*args,**kwargs):return dict(energy=np.full(len(F),1e-8))
            with patch('prototype.ipc_cutting.relaxed_response',side_effect=relaxed):
                Y=p.release(x)
                with patch.object(p,'_best_orientation',side_effect=lambda F,C,N,E:(E-5e-9,N.copy())):
                    self.assertEqual(p.orientation_error(x,Y),0.)
            np.testing.assert_array_equal(Y,0.)


if __name__=='__main__':unittest.main()
