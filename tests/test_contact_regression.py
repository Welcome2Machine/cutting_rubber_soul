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
from prototype.initiation import InitiationOptions,IntactProblem,ContactSolveError


class ContactRegressionTests(unittest.TestCase):
    def integral_failure(self,name='contact_integral_0321.npz'):
        if not (Path(__file__).parent/'fixtures'/name).is_file():
            self.skipTest('Local-only contact state is not distributed; this regression was not run')
        with np.load(Path(__file__).parent/'fixtures'/name) as z:
            p=IntactProblem.__new__(IntactProblem)
            p.config=FEMConfig(**json.loads(str(z['config_json'])))
            p.options=InitiationOptions(**json.loads(str(z['options_json'])))
            p.strength_surface=None
            xy,sn,tri=z['reference_xy'],z['local_sn'],z['triangles']
            p.mesh=FEMMesh(xy,sn,tri,np.empty((0,2),int),np.empty(0),
                np.flatnonzero(np.isclose(sn[:,1],0)),np.empty(0,int),np.zeros(len(xy),int))
            p._rebuild_boundary(z['boundary_edges'])
            return (p,z['accepted_displacement'],z['accepted_contact_multipliers'],
                    z['trial_displacement'],z['trial_contact_multipliers'],
                    float(z['last_accepted_depth_mm']),float(z['depth']),float(z['dt']))

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
