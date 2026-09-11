"""Physical contact checks on a small loading/reversal path; no full case."""
import contextlib
import io
import json
import unittest
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path
import numpy as np
from prototype.model_config import FEMConfig,FEMMesh
from prototype.initiation import InitiationOptions,IntactProblem,ContactSolveError,ContactResolutionError


class ContactRegressionTests(unittest.TestCase):
    def test_curved_path_switch_gradient_and_both_outgoing_jacobians(self):
        self.check_path_event('friction')

    def test_nearly_open_path_event_and_reclosure_update_both_contact_rows(self):
        self.check_path_event('normal')

    def check_path_event(self,kind):
        from prototype.quadratic import QuadraticProblem
        from prototype.contact_partitioned import path_step
        p=QuadraticProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,
            surface_spacing_mm=.02,mesh_grading='graded',contact_discretization='quadratic_integral',volumetric_discretization='quadratic_full'))
        previous=np.zeros(p.mesh.dof_count);history=p.initial_contact_state()
        initial=history.copy();initial[:,0]=1.
        q=p.surface_contact.evaluate(previous,previous,.0173,0.,initial,with_tangent=False,
            formulation='alart_curnier',roundoff_guard=False)
        i=int(np.argmax(q['pressure']));bound=p.config.friction_coefficient*q['pressure'][i]
        for direction in (1,-1):
            if kind=='normal':
                initial[i,0]=-p.config.contact_surface_penalty_n_mm3*q['gap'][i]-direction*1e-9
                initial[i,1]=.1-p.options.tangential_stiffness_mpa_per_mm*q['slip'][i]
            else:initial[i,1]=bound*(.5 if direction==1 else 1.5)-p.options.tangential_stiffness_mpa_per_mm*q['slip'][i]
            def inspect(evaluate,start,**kwargs):
                column=len(p.free)+2*i+int(kind=='friction')
                for power in range(-40,25):
                    predicted=start.copy();predicted[column]=direction*2.**power
                    hit=kwargs['event'](start,predicted)
                    if hit is not None:break
                else:self.fail('Curved-contact switch was not located')
                at=start+hit['fraction']*(predicted-start)
                value,gradient=hit['equation'](at)
                self.assertLess(abs(value),1e-7)
                self.assertEqual(hit['direction'],direction)
                self.assertTrue(hit['label'].startswith(kind))
                v=np.zeros_like(at);v[column]=direction/abs(gradient[column])
                eps=1e-4
                measured=(hit['equation'](at+eps*v)[0]-hit['equation'](at-eps*v)[0])/(2*eps)
                self.assertAlmostEqual(measured,gradient@v,places=6)
                base,j,b=evaluate(at,True)
                outgoing=hit['jacobian'](at,j)
                measured=(evaluate(at+eps*v,False)[0]-base)/eps
                np.testing.assert_allclose(measured,outgoing@v[:-1],rtol=2e-5,atol=2e-5)
                hit['on_accept'](at)
                self.assertIsNone(kwargs['event'](at,at+eps*v))
                self.assertIsNotNone(kwargs['event'](at-eps*v,at+eps*v))
                return at,None
            with contextlib.redirect_stdout(io.StringIO()),patch('prototype.continuation.follow_path',side_effect=inspect):
                path_step(p,previous,0.,.0173,.1,history,initial_multipliers=initial)
        np.testing.assert_array_equal(previous,0.)
        np.testing.assert_array_equal(history,0.)

    def test_interface_anderson_solves_affine_feedback_without_nan(self):
        from prototype.contact_partitioned import anderson_candidate
        x0=np.array([0.,0.]);g0=np.array([1.,2.]);x1=g0;g1=.5*x1+g0
        answer=anderson_candidate([x0,x1],[g0,g1],np.arange(2),np.ones(2))
        np.testing.assert_allclose(answer,[2.,4.],atol=1e-12)
        duplicate=anderson_candidate([x0,x0],[g0,g0],np.arange(2),np.ones(2))
        self.assertTrue(np.all(np.isfinite(duplicate)))
        # A strongly relaxed history has large coefficients, but its affine
        # fixed point remains well defined and must reach the outer safeguard.
        tiny=1e-6*g0
        answer=anderson_candidate([x0,tiny],[g0,.5*tiny+g0],np.arange(2),np.ones(2))
        np.testing.assert_allclose(answer,[2.,4.],rtol=1e-9)

    def test_path_resume_rejects_changed_history_and_metric(self):
        import tempfile
        from types import SimpleNamespace
        from prototype.contact_partitioned import load_path_resume
        p=SimpleNamespace(config=FEMConfig(),options=InitiationOptions(),contact_nodes=np.array([0,1]),
            mesh=SimpleNamespace(reference_xy=np.array([[0.,0.],[1.,0.],[0.,1.]]),triangles=np.array([[0,1,2]])))
        previous=np.zeros(6);history=np.zeros((2,2))
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);trial=folder/'original.npz'
            record=dict(config=p.config.to_dict(),options=vars(p.options),method='path',path_metric_ratio=.1,
                trial=str(trial),old_depth_mm=.498,depth_mm=.5,dt_s=.01)
            (folder/'result.json').write_text(json.dumps(record),encoding='utf-8')
            np.savez(folder/'trial.npz',previous=previous,previous_multipliers=history,
                reference_xy=p.mesh.reference_xy,triangles=p.mesh.triangles,contact_nodes=p.contact_nodes,
                auxiliary_displacement=previous,auxiliary_multipliers=history,auxiliary_parameter=-.2,
                auxiliary_tangent=np.r_[np.ones(10),-1.])
            loaded=load_path_resume(folder,p,previous,history,.498,.5,.01,trial,.1)
            self.assertEqual(loaded['parameter'],-.2)
            self.assertLess(loaded['tangent'][-1],0.)
            with self.assertRaisesRegex(ValueError,'metric differs'):
                load_path_resume(folder,p,previous,history,.498,.5,.01,trial,.2)
            with self.assertRaisesRegex(ValueError,'previous differs'):
                load_path_resume(folder,p,previous+.01,history,.498,.5,.01,trial,.1)

    def test_auxiliary_convergence_is_not_physical_acceptance(self):
        from prototype.quadratic import QuadraticProblem
        p=QuadraticProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,
            surface_spacing_mm=.1,mesh_grading='graded',contact_discretization='quadratic_integral',volumetric_discretization='quadratic_full'))
        previous=np.zeros(p.mesh.dof_count);history=p.initial_contact_state()
        # Freeze a fully open geometry at a remote blade position, then ask
        # for a penetrated physical blade position. Inner success must not pass.
        model=dict(contact=p.surface_contact.freeze(previous,-1.,history),anchor=previous.copy(),stiffness=np.zeros_like(previous))
        with contextlib.redirect_stdout(io.StringIO()):
            u,lam,info=p._mixed_step(previous,0.,.01,.05,history,subproblem=model,iteration_limit=1)
            self.assertIn('inner_iterations',info)
            self.assertNotIn('contact_solver',info)
            with self.assertRaises((ContactSolveError,ContactResolutionError)):
                p._mixed_step(previous,0.,.01,.05,history,initial=u,initial_multipliers=lam,verify_only=True)
        np.testing.assert_array_equal(previous,0.)
        np.testing.assert_array_equal(history,0.)

    def integral_failure(self,name='contact_integral_0321.npz'):
        if not (Path(__file__).parent/'fixtures'/name).is_file():
            self.skipTest('Local-only contact state is not distributed; this regression was not run')
        with np.load(Path(__file__).parent/'fixtures'/name) as z:
            p=IntactProblem.__new__(IntactProblem)
            p.config=FEMConfig(**json.loads(str(z['config_json'])))
            p.options=InitiationOptions(**json.loads(str(z['options_json'])))
            if p.options.volumetric_discretization in ('quadratic_full','taylor_hood'):
                from prototype.quadratic import QuadraticProblem
                p.__class__=QuadraticProblem
            p.strength_surface=None
            xy,sn,tri=z['reference_xy'],z['local_sn'],z['triangles']
            p.mesh=FEMMesh(xy,sn,tri,np.empty((0,2),int),np.empty(0),
                np.flatnonzero(np.isclose(sn[:,1],0)),np.empty(0,int),np.zeros(len(xy),int))
            from prototype.macro_element import restore
            restore(p.mesh,z)
            p._rebuild_boundary(z['boundary_edges'])
            return (p,z['accepted_displacement'],z['accepted_contact_multipliers'],
                    z['trial_displacement'],z['trial_contact_multipliers'],
                    float(z['last_accepted_depth_mm']),float(z['depth']),float(z['dt']))

    def test_p2_projected_fallback_cannot_commit_unbalanced_state(self):
        p,previous,history,trial,lam,old,depth,dt=self.integral_failure('p2_cone_fallback.npz')
        p.config=replace(p.config,max_contact_iterations=1)
        saved_previous=previous.copy();saved_history=history.copy()
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ContactSolveError) as caught:
            p._mixed_step(previous,old,depth,dt,history,initial=trial,initial_multipliers=lam)
        failed=caught.exception
        self.assertGreaterEqual(failed.multipliers[:,0].min(),0.)
        self.assertTrue(np.all(abs(failed.multipliers[:,1])<=p.config.friction_coefficient*failed.multipliers[:,0]))
        r,_,q,_=p.assemble(failed.u,previous,depth,old,dt,failed.multipliers[:,0],failed.multipliers[:,1],with_tangent=False,roundoff_guard=False)
        # The new candidate repairs projection, but its force still fails.
        # It must remain an exception/trial, never an accepted physical step.
        defect=np.column_stack((q['pressure'],q['shear']))-failed.multipliers
        self.assertLess(np.max(abs(defect)),p.config.contact_multiplier_tolerance_n_mm2)
        self.assertGreater(np.linalg.norm(r[p.free]),p.config.newton_tolerance_n)
        np.testing.assert_array_equal(previous,saved_previous)
        np.testing.assert_array_equal(history,saved_history)

    def test_damped_failed_state_must_pass_original_equations(self):
        p,previous,history,trial,lam,old,depth,dt=self.integral_failure('contact_integral_damping.npz')
        saved_previous=previous.copy();saved_history=history.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            u,state,d=p._mixed_step(previous,old,depth,dt,history,initial=trial,initial_multipliers=lam)
        self.assertGreater(d['contact_regularized_steps'],0)
        r,_,q,_=p.assemble(u,previous,depth,old,dt,state[:,0],state[:,1],with_tangent=False,roundoff_guard=False)
        self.assertLessEqual(np.linalg.norm(r[p.free]),p.config.newton_tolerance_n)
        self.assertLessEqual(np.max(np.abs(np.column_stack((q['pressure'],q['shear']))-state)),p.config.contact_multiplier_tolerance_n_mm2)
        self.assertLessEqual(p.edge_penetration(u,depth),p.config.contact_penetration_tolerance_mm)
        self.assertGreaterEqual(state[:,0].min(),0.)
        self.assertTrue(np.all(np.abs(state[:,1])<=p.config.friction_coefficient*state[:,0]))
        self.assertGreaterEqual(d['friction_dissipation_n_mm'],-d['friction_dissipation_tolerance_n_mm'])
        np.testing.assert_array_equal(previous,saved_previous);np.testing.assert_array_equal(history,saved_history)

    def test_integral_off_cone_trial_can_pass_original_equations_after_correction(self):
        p,previous,history,trial,lam,old,depth,dt=self.integral_failure()
        saved_previous=previous.copy();saved_history=history.copy()
        r,_,q,_=p.assemble(trial,previous,depth,old,dt,lam[:,0],lam[:,1],mixed=True,roundoff_guard=False)
        self.assertLess(np.linalg.norm(r[p.free]),p.config.newton_tolerance_n)
        self.assertGreater(np.max(np.abs(np.column_stack((q['pressure'],q['shear']))-lam)),p.config.contact_multiplier_tolerance_n_mm2)
        with contextlib.redirect_stdout(io.StringIO()):
            u,state,d=p._mixed_step(previous,old,depth,dt,history,initial=trial,initial_multipliers=lam)
        r,_,q,_=p.assemble(u,previous,depth,old,dt,state[:,0],state[:,1],with_tangent=False,roundoff_guard=False)
        self.assertLess(np.linalg.norm(r[p.free]),p.config.newton_tolerance_n)
        self.assertLess(np.max(np.abs(np.column_stack((q['pressure'],q['shear']))-state)),p.config.contact_multiplier_tolerance_n_mm2)
        self.assertLess(p.edge_penetration(u,depth),p.config.contact_penetration_tolerance_mm)
        self.assertGreaterEqual(state[:,0].min(),0.)
        self.assertTrue(np.all(np.abs(state[:,1])<=p.config.friction_coefficient*state[:,0]))
        self.assertGreaterEqual(d['friction_dissipation_n_mm'],0.)
        np.testing.assert_array_equal(previous,saved_previous);np.testing.assert_array_equal(history,saved_history)

    def test_cone_correction_cannot_bypass_force_tolerance(self):
        p,previous,history,trial,lam,old,depth,dt=self.integral_failure()
        p.config=replace(p.config,newton_tolerance_n=4e-5,max_contact_iterations=1)
        corrected=lam.copy();corrected[:,0]=np.maximum(lam[:,0],0.)
        corrected[:,1]=np.clip(lam[:,1],-.35*corrected[:,0],.35*corrected[:,0])
        raw=p.assemble(trial,previous,depth,old,dt,lam[:,0],lam[:,1],mixed=True,with_tangent=False)[0]
        actual=p.assemble(trial,previous,depth,old,dt,corrected[:,0],corrected[:,1],with_tangent=False)[0]
        self.assertLess(np.linalg.norm(raw[p.free]),p.config.newton_tolerance_n)
        self.assertGreater(np.linalg.norm(actual[p.free]),p.config.newton_tolerance_n)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ContactSolveError):
            p._mixed_step(previous,old,depth,dt,history,initial=trial,initial_multipliers=lam)

    def test_nearly_converged_negative_warm_start_is_polished_and_rechecked(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5,max_contact_iterations=2),
            InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        u=np.zeros(p.mesh.dof_count);history=p.initial_contact_state()
        # Off-cone numerical warm start at an open, unconstrained surface node.
        i=next(i for i,node in enumerate(p.contact_nodes) if 2*node in p.free)
        history[i,0]=-1e-6
        saved=history.copy()
        with contextlib.redirect_stdout(io.StringIO()),patch('prototype.initiation.splu',side_effect=AssertionError('A feasible converged candidate needs no Newton solve')):
            new,lam,d=p._mixed_step(u,0.,0.,.01,history)
        self.assertGreaterEqual(lam[:,0].min(),0.)
        r,_,contact,_=p.assemble(new,u,0.,0.,.01,lam[:,0],lam[:,1])
        self.assertLessEqual(np.linalg.norm(r[p.free]),p.config.newton_tolerance_n)
        self.assertLessEqual(np.max(np.abs(np.column_stack((contact['pressure'],contact['shear']))-lam)),p.config.contact_multiplier_tolerance_n_mm2)
        np.testing.assert_array_equal(history,saved)

    def test_saved_contact_switch_is_solved_without_changing_physical_history(self):
        fixture=Path(__file__).parent/'fixtures/contact_v2_transition.npz'
        if not fixture.is_file():
            self.skipTest('Local-only contact state is not distributed; this regression was not run')
        with np.load(fixture) as z:
            p=IntactProblem(FEMConfig(**json.loads(str(z['config_json']))),InitiationOptions())
            p.mesh=FEMMesh(z['xy'],z['sn'],z['triangles'],np.empty((0,2),int),np.empty(0),
                np.flatnonzero(np.isclose(z['sn'][:,1],0)),np.empty(0,int),np.zeros(len(z['xy']),int))
            p._rebuild_boundary(z['edges'])
            u,state,old,depth=z['u'],z['state'],float(z['old']),float(z['depth'])
        previous=u.copy();history=state.copy();dt=(depth-old)/p.config.speed_mm_s
        with contextlib.redirect_stdout(io.StringIO()):
            new,lam,d=p.step(u,old,depth,dt,state)
        r,_,contact,_=p.assemble(new,u,depth,old,dt,lam[:,0],lam[:,1])
        self.assertLessEqual(np.linalg.norm(r[p.free]),p.config.newton_tolerance_n)
        self.assertLessEqual(np.max(np.abs(np.column_stack((contact['pressure'],contact['shear']))-lam)),p.config.contact_multiplier_tolerance_n_mm2)
        self.assertLessEqual(p.edge_penetration(new,depth),p.config.contact_penetration_tolerance_mm)
        self.assertGreaterEqual(lam[:,0].min(),0)
        self.assertGreaterEqual(d['friction_dissipation_n_mm'],-d['friction_dissipation_tolerance_n_mm'])
        np.testing.assert_array_equal(u,previous)
        np.testing.assert_array_equal(state,history)

    def test_loading_hold_reversal_and_original_contact_equations(self):
        p=IntactProblem(FEMConfig.from_json(Path(__file__).resolve().parents[1]/'prototype/initiation_config.json'),InitiationOptions())
        u=np.zeros(p.mesh.dof_count)
        state=p.initial_contact_state()
        old=0.
        with contextlib.redirect_stdout(io.StringIO()):
            for depth in (.00625,.0125,.0125,.00625,0.):
                previous=u.copy(); history=state.copy()
                saved_previous=previous.copy(); saved_history=history.copy()
                dt=max(abs(depth-old)/p.config.speed_mm_s,.01)
                u,state,d=p.step(previous,old,depth,dt,history)
                r,k,contact,actual=p.assemble(u,previous,depth,old,dt,state[:,0],state[:,1])
                target=np.column_stack((contact['pressure'],contact['shear']))
                self.assertLessEqual(np.linalg.norm(r[p.free]),p.config.newton_tolerance_n)
                self.assertLessEqual(np.max(np.abs(target-state)),p.config.contact_multiplier_tolerance_n_mm2)
                self.assertLessEqual(p.edge_penetration(u,depth),p.config.contact_penetration_tolerance_mm)
                self.assertTrue(np.all(state[:,0]>=0))
                self.assertTrue(np.all(np.abs(contact['shear'])<=p.config.friction_coefficient*contact['pressure']+1e-12))
                self.assertGreaterEqual(d['friction_dissipation_n_mm'],-d['friction_dissipation_tolerance_n_mm'])
                self.assertGreater(actual['min_j'],0)
                np.testing.assert_array_equal(previous,saved_previous)
                np.testing.assert_array_equal(history,saved_history)
                # The same displacement increment is rate independent.
                other=p.assemble(u,previous,depth,old,dt*100,state[:,0],state[:,1])
                np.testing.assert_array_equal(r,other[0])
                old=depth
