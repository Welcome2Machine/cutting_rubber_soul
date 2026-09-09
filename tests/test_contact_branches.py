"""Contact-law branches and the saved oblique-cut stick/slip failure."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from prototype.contact_branches import admissible_branches,branch_predictors
from prototype.model_config import FEMConfig,FEMMesh
from prototype.initiation import IntactProblem,InitiationOptions,ContactSolveError


class BranchTests(unittest.TestCase):
    def test_failed_auxiliary_branches_preserve_primary_failure_evidence(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        u=np.zeros(p.mesh.dof_count);state=p.initial_contact_state()
        primary=ContactSolveError('primary residual',u,state)
        errors=[primary,ContactSolveError('geometric seed',u,state),ContactSolveError('bad branch',u+1,state+1)]
        with patch.object(p,'_mixed_step',side_effect=errors),patch('prototype.initiation.branch_predictors',return_value=iter([(u.copy(),state.copy(),1)])),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ContactSolveError) as caught:p.step(u,0.,0.,.1,state)
        self.assertIs(caught.exception,primary)
        np.testing.assert_array_equal(caught.exception.u,u)

    def test_inverted_auxiliary_predictor_is_rejected_without_losing_physical_state(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        previous=np.zeros(p.mesh.dof_count);history=p.initial_contact_state()
        bad=-p.mesh.reference_xy.ravel();bad[p.fixed]=0.
        original=p._mixed_step;calls=0
        def solve(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls<=2:raise ContactSolveError('force fallback in this contract test',previous,history)
            return original(*args,**kwargs)
        predictors=[(bad,history.copy(),1),(previous.copy(),history.copy(),2)]
        with patch.object(p,'_mixed_step',side_effect=solve),patch('prototype.initiation.branch_predictors',return_value=iter(predictors)),contextlib.redirect_stdout(io.StringIO()):
            new,state,d=p.step(previous,0.,0.,.1,history)
        self.assertEqual(calls,4)
        self.assertEqual(d['contact_branch_predictors'],2)
        self.assertLessEqual(d['residual_n'],p.config.newton_tolerance_n)
        np.testing.assert_array_equal(new,previous);np.testing.assert_array_equal(state,history)

    def test_exact_affine_open_stick_and_both_sliding_directions(self):
        # Independent elastic normal/tangential compliances, in the code's
        # positive-penetration / positive-resisting-shear convention.
        mobility=-np.eye(2)
        for offset,expected in (((-1.,3.),(0.,0.)),((1.,.2),(1.,.2)),
                                ((1.,2.),(1.,.35)),((1.,-2.),(1.,-.35))):
            candidates=list(admissible_branches(mobility,np.array(offset),.35,[(0,1,-1,2)]))
            self.assertTrue(candidates)
            for traction,_,_ in candidates:np.testing.assert_allclose(traction[0],expected,atol=1e-12)
        self.assertEqual(list(admissible_branches(mobility,np.array([-1.,3.]),.35,[(0,1,-1,2)],budget=3)),[])

    def test_failed_cutting_state_recovered_and_verified_with_tighter_tolerances(self):
        if not (Path(__file__).parent/'fixtures/contact_branch_transition.npz').is_file():
            self.skipTest('Local-only contact state is not distributed; this regression was not run')
        with np.load(Path(__file__).parent/'fixtures/contact_branch_transition.npz') as z:
            p=IntactProblem.__new__(IntactProblem)
            p.config=FEMConfig(**json.loads(str(z['config_json'])))
            p.options=InitiationOptions(**json.loads(str(z['options_json'])));p.strength_surface=None
            xy,sn,tri=z['reference_xy'],z['local_sn'],z['triangles']
            p.mesh=FEMMesh(xy,sn,tri,np.empty((0,2),int),np.empty(0),
                np.flatnonzero(np.isclose(sn[:,1],0)),np.empty(0,int),np.zeros(len(xy),int))
            p._rebuild_boundary(z['boundary_edges'])
            previous=z['accepted_displacement'];history=z['accepted_contact_multipliers']
            trial=z['trial_displacement'];lam=z['trial_contact_multipliers']
            old=float(z['last_accepted_depth_mm']);depth=float(z['target_depth_mm'])
        saved_u=previous.copy();saved_lam=history.copy();dt=(depth-old)/p.config.speed_mm_s
        with contextlib.redirect_stdout(io.StringIO()):
            result=None
            for seed,warm,_ in branch_predictors(p,previous,old,depth,dt,trial,lam):
                try:
                    result=p._mixed_step(previous,old,depth,dt,history,initial=seed,initial_multipliers=warm)
                    break
                except ContactSolveError:pass
            self.assertIsNotNone(result,'No branch recovered the true failed state')
            p.config=replace(p.config,newton_tolerance_n=1e-6,contact_multiplier_tolerance_n_mm2=1e-6)
            new,state,d=p._mixed_step(previous,old,depth,dt,history,initial=result[0],initial_multipliers=result[1])
        r,_,q,_=p.assemble(new,previous,depth,old,dt,state[:,0],state[:,1],mixed=True,roundoff_guard=False,with_tangent=False)
        self.assertLessEqual(np.linalg.norm(r[p.free]),1e-6)
        self.assertLessEqual(np.max(np.abs(np.column_stack((q['pressure'],q['shear']))-state)),1e-6)
        self.assertGreaterEqual(state[:,0].min(),0.)
        self.assertTrue(np.all(np.abs(state[:,1])<=p.config.friction_coefficient*state[:,0]))
        self.assertGreater(d['friction_dissipation_n_mm'],0.)
        self.assertLessEqual(p.edge_penetration(new,depth),p.config.contact_penetration_tolerance_mm)
        self.assertGreater(d['min_j'],0.)
        np.testing.assert_array_equal(previous,saved_u);np.testing.assert_array_equal(history,saved_lam)


if __name__=='__main__':unittest.main()
