import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from prototype.contact_study import bounded_path
from prototype.model_config import FEMConfig
from prototype.initiation import IntactProblem,InitiationOptions,ContactSolveError


class ContactStudyTests(unittest.TestCase):
    def test_geometry_rejection_saves_trial_and_unchanged_physical_history(self):
        from prototype.initiation import ContactResolutionError
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        u=np.zeros(p.mesh.dof_count);lam=p.initial_contact_state()
        def step(problem,previous,old,depth,dt,history):
            raise ContactResolutionError('geometry rejection',[0],u=previous+.001,multipliers=history,depth=depth)
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            out=Path(folder)
            report=bounded_path(p,u,lam,0.,[.01],out,adaptive=True,refine_mesh=False,record_history=True,step_solver=step)
            self.assertEqual(report['error_type'],'ContactResolutionError')
            self.assertEqual(report['last_depth_mm'],0.)
            with np.load(out/'failed_trial.npz') as z:
                np.testing.assert_array_equal(z['displacement'],u+.001)
                np.testing.assert_array_equal(z['previous_displacement'],u)
                np.testing.assert_array_equal(z['previous_multipliers'],lam)
                np.testing.assert_array_equal(z['geometry_bad_edges'],[0])
            with np.load(out/'last_state.npz') as z:np.testing.assert_array_equal(z['displacement'],u)

    def test_normal_baseline_starts_from_zero_without_changing_experiment(self):
        from prototype.contact_study import normal_baseline
        from prototype.quadratic import QuadraticProblem
        p=QuadraticProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,
            surface_spacing_mm=.2,contact_discretization='quadratic_integral',volumetric_discretization='quadratic_full'))
        original=p.config;old_u=np.ones(p.mesh.dof_count);old_lam=p.initial_contact_state()+1
        p._accepted_velocity=old_u.copy();p._velocity_depth=.5;p._velocity_loading=True
        with tempfile.TemporaryDirectory() as folder,patch('prototype.contact_study.load_failed_problem',return_value=(p,old_u,old_lam,.5)),patch('prototype.contact_study.bounded_path',return_value={}) as run:
            normal_baseline(Path('unused'),Path(folder),.013,.00625,10.)
            call=run.call_args
            np.testing.assert_array_equal(call.args[1],0.)
            np.testing.assert_array_equal(call.args[2],0.)
            self.assertEqual(call.args[3],0.)
            self.assertAlmostEqual(call.args[4][-1],.013)
            self.assertEqual(p.config.friction_coefficient,0.)
            self.assertEqual(original.friction_coefficient,.35)
            self.assertFalse(hasattr(p,'_accepted_velocity'))
            self.assertTrue(call.kwargs['record_history']);self.assertFalse(call.kwargs['refine_mesh'])
            self.assertEqual({k for k,v in original.to_dict().items() if p.config.to_dict()[k]!=v},{'friction_coefficient'})

    def test_normal_resume_preserves_history_and_rejects_frictional_source(self):
        from prototype.contact_study import normal_baseline
        from prototype.quadratic import QuadraticProblem
        from dataclasses import replace
        p=QuadraticProblem(FEMConfig(mesh_size_mm=1.5,friction_coefficient=0.),InitiationOptions(refinement_mm=.7,
            surface_spacing_mm=.2,contact_discretization='quadratic_integral',volumetric_discretization='quadratic_full'))
        u=np.full(p.mesh.dof_count,.01);lam=p.initial_contact_state()+.1
        with tempfile.TemporaryDirectory() as folder,patch('prototype.contact_study.load_failed_problem',return_value=(p,u,lam,.075)), \
                patch('prototype.contact_study.bounded_path',return_value={}) as run:
            source=Path(folder);out=source/'tail';out.mkdir()
            (source/'provenance.json').write_text(json.dumps(dict(case='normal_baseline')),encoding='utf-8')
            normal_baseline(source,out,.1,.00625,10.,resume=True)
            args=run.call_args.args
            np.testing.assert_array_equal(args[1],u);np.testing.assert_array_equal(args[2],lam)
            self.assertEqual(args[3],.075)
            np.testing.assert_allclose(args[4],[.08125,.0875,.09375,.1])
            p.config=replace(p.config,friction_coefficient=.35)
            with self.assertRaisesRegex(ValueError,'frictionless baseline'):
                normal_baseline(source,out,.1,.00625,10.,resume=True)

    def test_inner_budget_preserves_failed_history_without_subdividing(self):
        from prototype.initiation import ContactBudgetError
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5),InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        u=np.zeros(p.mesh.dof_count);lam=p.initial_contact_state();calls=[]
        def step(problem,previous,old,depth,dt,history):
            calls.append(depth)
            error=ContactBudgetError('inner budget',previous+.01,history)
            error.inner_state={'inner_primal':np.array([1.])}
            raise error
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            out=Path(folder)
            report=bounded_path(p,u,lam,0.,[.01],out,adaptive=True,refine_mesh=False,record_history=True,step_solver=step)
            self.assertEqual(report['status'],'budget_exhausted');self.assertEqual(calls,[.01])
            self.assertEqual(report['last_depth_mm'],0.)
            with np.load(out/'failed_trial.npz') as z:
                np.testing.assert_array_equal(z['previous_displacement'],u)
                np.testing.assert_array_equal(z['previous_multipliers'],lam)
                np.testing.assert_array_equal(z['inner_primal'],[1.])
                self.assertAlmostEqual(float(z['dt_s']),.01/p.config.speed_mm_s)

    def test_switch_certificate_detects_nonroot_cusp_and_preserves_matrix(self):
        from scipy.sparse import csr_matrix,diags
        from prototype.audit_contact import row_switch_certificate
        # H=(abs(x)+1,y) at the cusp: both branch Newton steps cross into the
        # other branch and INCREASE the true norm. Both matrices are invertible.
        left=diags([-1.,1.],format='csc');saved=left.toarray().copy()
        report=row_switch_certificate(left,np.array([1.,0.]),0,csr_matrix([[1.,0.]]),csr_matrix([[1.,0.]]),np.ones(2))
        self.assertTrue(report['both_steps_inconsistent']);self.assertTrue(report['both_steps_ascent'])
        self.assertAlmostEqual(report['determinant_ratio'],-1.)
        self.assertAlmostEqual(report['singular_mix_fraction'],.5)
        self.assertLess(report['null_relative_residual'],1e-12)
        np.testing.assert_array_equal(left.toarray(),saved)
        # H=(x+1,y) has a feasible Newton step on the x<0 side.
        ordinary=row_switch_certificate(diags([1.,1.],format='csc'),np.array([1.,0.]),0,
            csr_matrix([[1.,0.]]),csr_matrix([[1.,0.]]),np.ones(2))
        self.assertFalse(ordinary['both_steps_inconsistent']);self.assertFalse(ordinary['both_steps_ascent'])
        # The same opposite orientations at an actual root H=(abs(x),y)
        # must NOT be reported as two ascent steps or as failure/nonexistence.
        root=row_switch_certificate(left,np.zeros(2),0,csr_matrix([[1.,0.]]),csr_matrix([[1.,0.]]),np.ones(2))
        self.assertLess(root['determinant_ratio'],0.)
        self.assertFalse(root['both_steps_ascent']);self.assertFalse(root['both_steps_inconsistent'])

    def test_topology_change_preserves_domain_and_positive_elements(self):
        from prototype.contact_study import flip_local_edges
        from prototype.model_config import FEMMesh,local_to_lab
        c=FEMConfig();sn=np.array([[6.9,4.8],[7.1,4.8],[7.1,5.],[6.9,5.]])
        base=FEMMesh(local_to_lab(sn,c),sn,np.array([[0,1,2],[0,2,3]]),
            np.empty((0,2),int),np.empty(0),np.empty(0,int),np.empty(0,int),np.zeros(4,int))
        boundary=np.array([[0,1],[1,2],[2,3],[3,0]])
        changed,flips=flip_local_edges(base,boundary,c)
        self.assertEqual(len(flips),1)
        np.testing.assert_array_equal(changed.reference_xy,base.reference_xy)
        self.assertEqual({tuple(sorted(t)) for t in changed.triangles},{(0,1,3),(1,2,3)})
        self.assertAlmostEqual(changed.bulk_data[0].sum(),base.bulk_data[0].sum())
        self.assertTrue(np.all(changed.bulk_data[0]>0))
        np.testing.assert_array_equal(base.triangles,[[0,1,2],[0,2,3]])

    def test_recorded_increments_keep_actual_history_through_subdivision(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5,min_adaptive_increment_mm=.001),
            InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        u=np.zeros(p.mesh.dof_count);state=p.initial_contact_state()
        def solve(previous,old,depth,dt,history):
            if depth-old>.006:raise ContactSolveError('subdivide',previous,history)
            return previous+depth-old,history+depth-old,dict(reaction_fz_n=0.,residual_n=0.)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(p,'step',side_effect=solve),contextlib.redirect_stdout(io.StringIO()):
                result=bounded_path(p,u,state,0.,[.01],out,adaptive=True,refine_mesh=False,record_history=True)
            self.assertTrue(result['completed'])
            with np.load(out/'increments/00000.npz') as first,np.load(out/'increments/00001.npz') as second:
                np.testing.assert_array_equal(first['previous_displacement'],u)
                np.testing.assert_array_equal(second['previous_displacement'],first['displacement'])
                np.testing.assert_array_equal(second['previous_multipliers'],first['contact_multipliers'])
                self.assertAlmostEqual(float(second['old_depth_mm']),.005)
                self.assertAlmostEqual(float(second['dt_s']),.005/p.config.speed_mm_s)
            self.assertEqual(len(list((out/'increments').glob('*.npz'))),2)

    def test_mode_probe_preserves_saved_time_and_does_not_commit_failure(self):
        from prototype.audit_contact import perturb_modes
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5),
            InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        u=np.zeros(p.mesh.dof_count);state=p.initial_contact_state()
        saved_u=u.copy();saved_state=state.copy()
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);archive=directory/'modes.npz';out=directory/'probe'
            # Saved time need not equal displacement divided by configured speed.
            np.savez_compressed(archive,previous_displacement=u,previous_multipliers=state,
                base_displacement=u,base_multipliers=state,displacement_modes=u[None],
                traction_modes=state[None],old_depth_mm=0.,depth_mm=.01,dt_s=.125,
                config_json=json.dumps(p.config.to_dict()),options_json=json.dumps(vars(p.options)))
            with patch('prototype.audit_contact.load_failed_problem',return_value=(p,u,state,0.)), \
                    patch.object(p,'_mixed_step',side_effect=RuntimeError('bounded failure')) as solve, \
                    contextlib.redirect_stdout(io.StringIO()):
                result=perturb_modes(directory,archive,out,[.0005],iterations=1)
            self.assertEqual(result['status'],'exhausted')
            self.assertEqual(solve.call_count,2)
            self.assertTrue(all(call.args[3]==.125 for call in solve.call_args_list))
            self.assertFalse((out/'accepted_state.npz').exists())
            self.assertTrue(all(not trial['accepted'] for trial in result['trials']))
            np.testing.assert_array_equal(u,saved_u);np.testing.assert_array_equal(state,saved_state)

    def test_coupled_singular_modes_match_dense_nonsymmetric_reference(self):
        from scipy.sparse import csc_matrix
        from prototype.audit_contact import coupled_modes
        a=np.array([[3.,12.,0.,1.],[-.1,2.,5.,0.],[0.,.3,4.,2.],[1.,0.,.2,6.]])
        result,v=coupled_modes(csc_matrix(a),2)
        row=1/abs(a).max(axis=1);scaled=row[:,None]*a
        column=1/abs(scaled).max(axis=0);scaled*=column
        expected=np.linalg.svd(scaled,compute_uv=False)[-2:][::-1]
        np.testing.assert_allclose(result['singular_values'],expected,rtol=1e-10)
        self.assertLess(max(max(r.values()) for r in result['triplet_residuals']),1e-10)
        # Returned vectors are in the original unknown coordinates.
        np.testing.assert_allclose(np.linalg.norm(v/column[:,None],axis=0),1.,atol=1e-10)

    def test_contact_constraint_removes_a_negative_bulk_mode(self):
        from scipy.sparse import csc_matrix
        from prototype.audit_contact import constrained_eigenmodes
        # u0 has negative bulk curvature but is exactly constrained. Only u1
        # has a finite eigenvalue; assigning mass to lambda would invent modes.
        matrix=csc_matrix([[-2.,0.,1.],[0.,3.,0.],[1.,0.,0.]])
        values,vectors=constrained_eigenmodes(matrix,2,1)
        self.assertAlmostEqual(values[0]['real'],1.)
        self.assertAlmostEqual(values[0]['imaginary'],0.)
        self.assertLess(abs(vectors[0,0]),1e-12)
        self.assertLess(values[0]['relative_residual'],1e-12)

    def test_constrained_nonsymmetric_spectrum_matches_generalized_dense_problem(self):
        from scipy.linalg import eigvals
        from scipy.sparse import csc_matrix
        from prototype.audit_contact import constrained_eigenmodes
        # Sliding-like traction coupling: neither C=B.T nor D symmetric.
        a=np.array([[2.,1.,1.,0.],[0.,-3.,0.,1.],
                    [1.,0.,0.,0.],[0.,1.,-.35,1.]])
        metric=np.diag([2.,3.,0.,0.])
        reference=eigvals(a,metric);reference=reference[np.isfinite(reference)]
        near,_=constrained_eigenmodes(csc_matrix(a),2,1)
        negative,_=constrained_eigenmodes(csc_matrix(a),2,1,selection='negative')
        self.assertEqual(len(reference),1)
        self.assertLess(reference[0].real,0.)
        for value in (near[0],negative[0]):
            self.assertAlmostEqual(value['real'],reference[0].real)
            self.assertLess(value['relative_residual'],1e-12)

    def test_iteration_limit_reports_saved_trial_residual(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5,max_contact_iterations=2),
            InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        u=np.zeros(p.mesh.dof_count);state=p.initial_contact_state()
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ContactSolveError) as caught:
            p._mixed_step(u,0.,.01,.05,state)
        exc=caught.exception
        r,_,q,_=p.assemble(exc.u,u,.01,0.,.05,exc.multipliers[:,0],exc.multipliers[:,1],mixed=True)
        norm=np.linalg.norm(r[p.free])
        defect=np.max(np.abs(np.column_stack((q['pressure'],q['shear']))-exc.multipliers))
        self.assertIn(f'residual={norm:.6g} N, projection={defect:.6g} MPa',str(exc))

    def test_saved_failure_merges_with_scheduled_float_without_tiny_step(self):
        from prototype.loading import loading_depths,recovery_targets
        c=FEMConfig(approach_steps=481)
        scheduled=[d for d,_ in loading_depths(c,.00625,stop_mm=.4) if d>.375]
        merged=recovery_targets(c,.00625,.375,.4,.38125)
        self.assertEqual(merged,scheduled)
        self.assertGreater(min(np.diff(np.r_[.375,merged])),.006)
        inserted=recovery_targets(c,.00625,.375,.4,.378)
        self.assertEqual(inserted[0],.378)
    def test_zero_time_or_backward_loading_targets_are_rejected_before_output(self):
        for targets in ([],[.1],[np.nextafter(.1,1.)],[.2,.15],[float('nan')]):
            with self.assertRaisesRegex(ValueError,'strictly exceed'):
                bounded_path(None,None,None,.1,targets,Path('unused'),adaptive=True)

    def test_failed_snapshot_labels_actual_recursive_increment(self):
        p=IntactProblem(FEMConfig(mesh_size_mm=1.5,min_adaptive_increment_mm=.001),
            InitiationOptions(refinement_mm=.7,surface_spacing_mm=.2))
        u=np.zeros(p.mesh.dof_count);state=p.initial_contact_state()
        def solve(previous,old,depth,dt,history):
            if depth>.00625:
                raise ContactSolveError(f'synthetic failure depth={depth}',previous,history)
            return previous.copy(),history.copy(),dict(reaction_fz_n=0.,residual_n=0.)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(p,'step',side_effect=solve),contextlib.redirect_stdout(io.StringIO()):
                report=bounded_path(p,u,state,0.,[.01],out,adaptive=True)
            self.assertFalse(report['completed'])
            self.assertAlmostEqual(report['last_depth_mm'],.00625)
            self.assertAlmostEqual(report['failed_depth_mm'],.0075)
            with np.load(out/'failed_trial.npz') as z:
                self.assertAlmostEqual(float(z['indentation_mm']),.0075)


if __name__=='__main__':unittest.main()
