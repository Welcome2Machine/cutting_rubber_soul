"""Physical equivalence and rejected-history safety for optional AL solvers."""
import contextlib
import io
import time
import unittest
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.al_globalization import (GlobalizedAugmentedBladeProblem,WarmStartedAugmentedBladeProblem,
                                       WarmGlobalizedAugmentedBladeProblem)
from prototype.augmented_blade import AugmentedBladeProblem
from prototype.ipc_reference import LoadStepFailure
from prototype.separated_self_contact import SeparatedContactDomainError
from tools.run_augmented_blade import make_problem


class GlobalizationTests(unittest.TestCase):
    def setUp(self):
        c=threadpool_limits(1);c.__enter__();self.addCleanup(c.__exit__,None,None,None)

    def candidate(self,kind):
        p=make_problem(57,coarse=True,spacing=.000625);p.__class__=kind
        p.configure_compute(material_workers=2,linear_threads=4)
        self.addCleanup(p.close_compute);p.ipc.set_num_threads(2)
        return p

    def test_independent_loading_preserves_force_surface_and_original_audit(self):
        reference=make_problem(57,coarse=True,spacing=.000625);reference.ipc.set_num_threads(2)
        variants=[self.candidate(k) for k in (GlobalizedAugmentedBladeProblem,WarmGlobalizedAugmentedBladeProblem)]
        adaptive=self.candidate(WarmGlobalizedAugmentedBladeProblem)
        adaptive.direction_policy='adaptive';adaptive.nested_contact_model=True
        adaptive.nonmonotone_window=3;variants.append(adaptive)
        problems=[reference]+variants;positions=[p.rest.copy() for p in problems];old=reference.start
        with contextlib.redirect_stdout(io.StringIO()):
            for depth in (0.,.005,.01):
                records=[]
                for i,p in enumerate(problems):
                    positions[i],record=p.step(positions[i],old,depth,time.monotonic()+90);records.append(record)
                    state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=positions[i],
                        previous_positions=p.previous,depth_mm=depth,**p.checkpoint_data())
                    self.assertTrue(reference.audit_state(state,record)['passed'])
                for i in range(1,len(problems)):
                    self.assertLess(abs(records[i]['reaction_fz_n']-records[0]['reaction_fz_n']),2e-5)
                    np.testing.assert_allclose(positions[i],positions[0],rtol=0,atol=1e-6)
                    # Same absolute metric gate as the earlier exact-increment
                    # qualification; near-zero off-diagonal entries do not
                    # provide a meaningful componentwise relative scale.
                    self.assertLess(float(np.max(abs(problems[i].material_metric-reference.material_metric))),1e-6)
                old=depth

    def test_failed_warm_step_restores_edge_labels_and_multipliers(self):
        p=self.candidate(WarmStartedAugmentedBladeProblem)
        with contextlib.redirect_stdout(io.StringIO()):
            x,_=p.step(p.rest.copy(),p.start,0.,time.monotonic()+90)
        saved=[v.copy() for v in (p.qedges,p.qfraction,p.qweights,p.multipliers,p.material_metric)]
        previous=p.previous_material_metric.copy()
        def fail(x,deadline):
            p.multipliers[:]=17.
            raise LoadStepFailure('deliberate rejected nonlinear iterate')
        with patch.object(p,'_equilibrate',side_effect=fail):
            with self.assertRaises(LoadStepFailure):p.step(x,0.,.005,time.monotonic()+90)
        for actual,want in zip((p.qedges,p.qfraction,p.qweights,p.multipliers,p.material_metric),saved):
            np.testing.assert_array_equal(actual,want)
        self.assertTrue(np.all(p.rejected_al_state['al_multipliers']==17.))
        self.assertEqual(p.rejected_al_state['attempted_depth_mm'],.005)
        # Physical previous metric must be the last accepted CURRENT metric,
        # never a trial return map. Step initialization legitimately updates it.
        np.testing.assert_array_equal(p.previous_material_metric,saved[-1])

    def test_newton_does_not_stop_before_final_cone_and_normal_checks(self):
        from prototype.augmented_blade import contact_constraints_satisfied
        for kind in (AugmentedBladeProblem,GlobalizedAugmentedBladeProblem):
            for invalid in ('cone','negative_normal'):
                p=make_problem(57,coarse=True,spacing=.000625) if kind is AugmentedBladeProblem else self.candidate(kind)
                p.ipc.set_num_threads(2)
                with contextlib.redirect_stdout(io.StringIO()):
                    x,_=p.step(p.rest.copy(),p.start,0.,time.monotonic()+90)
                    # Tiny inadmissible traction on an open patch has small
                    # integral residual but must NOT trigger Newton success.
                    node=int(np.argmin(p.multipliers[:,0]))
                    p.multipliers[node]=[0.,2e-7] if invalid=='cone' else [-2e-8,0.]
                    checks=p.constraint_audit(x)[1]
                    self.assertLess(checks['integrated_projection_defect_n'],5e-5)
                    self.assertFalse(contact_constraints_satisfied(checks,p.multipliers))
                    history=p.material_metric.copy()
                    y,it=p._equilibrate(x,time.monotonic()+90)
                    self.assertGreater(it,0)
                    self.assertTrue(contact_constraints_satisfied(p.constraint_audit(y)[1],p.multipliers))
                    p.acceptance(y,0.,it,1)
                    np.testing.assert_array_equal(p.material_metric,history)

    def test_trial_domain_rejection_backtracks_and_preserves_original_acceptance(self):
        for kind in (AugmentedBladeProblem,GlobalizedAugmentedBladeProblem):
            p=make_problem(57,coarse=True,spacing=.000625) if kind is AugmentedBladeProblem else self.candidate(kind)
            p.ipc.set_num_threads(2)
            with contextlib.redirect_stdout(io.StringIO()):
                x,_=p.step(p.rest.copy(),p.start,0.,time.monotonic()+90)
            original=p.evaluate;rejected=[]
            def evaluate(xx,*args,**kwargs):
                if kwargs.get('tangent') is False and not rejected:
                    rejected.append(True)
                    raise SeparatedContactDomainError('trial outside valid domain')
                return original(xx,*args,**kwargs)
            log=io.StringIO()
            with contextlib.redirect_stdout(log),patch.object(p,'evaluate',side_effect=evaluate):
                y,record=p.step(x,0.,.005,time.monotonic()+90)
            self.assertEqual(len(rejected),1)
            self.assertIn('AL_DOMAIN_REJECT',log.getvalue())
            state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=y,
                previous_positions=p.previous,depth_mm=.005,**p.checkpoint_data())
            auditor=make_problem(57,coarse=True,spacing=.000625)
            self.assertTrue(auditor.audit_state(state,record)['passed'])

    def test_current_domain_error_and_unrelated_trial_error_are_not_swallowed(self):
        p=self.candidate(GlobalizedAugmentedBladeProblem)
        x=p.affine_loading_candidate(p.rest,p.start,.005)
        p.previous=p.rest.copy();p.previous_material_metric=p.material_metric.copy()
        p.material_dt=(.005-p.start)/p.loading_rate;p.prepare_quadrature(x)
        original=p.evaluate;before=p.multipliers.copy()
        with patch.object(p,'evaluate',side_effect=SeparatedContactDomainError('current iterate')):
            with self.assertRaisesRegex(SeparatedContactDomainError,'current iterate'):
                p._equilibrate(x,time.monotonic()+90)
        def evaluate(xx,*args,**kwargs):
            if kwargs.get('tangent') is False:raise FloatingPointError('unrelated material error')
            return original(xx,*args,**kwargs)
        with contextlib.redirect_stdout(io.StringIO()),patch.object(p,'evaluate',side_effect=evaluate):
            with self.assertRaisesRegex(FloatingPointError,'unrelated material'):
                p._equilibrate(x,time.monotonic()+90)
        np.testing.assert_array_equal(p.multipliers,before)

    def test_bad_linear_direction_is_rejected_without_acceptance(self):
        p=self.candidate(GlobalizedAugmentedBladeProblem)
        x=p.affine_loading_candidate(p.rest,p.start,.005)
        p.previous=p.rest.copy();p.previous_material_metric=p.material_metric.copy()
        p.material_dt=(.005-p.start)/p.loading_rate;p.prepare_quadrature(x)
        multipliers=p.multipliers.copy();history=p.material_metric.copy()
        with contextlib.redirect_stdout(io.StringIO()),patch.object(p,'linear_direction',side_effect=lambda A,b:np.zeros_like(b)):
            with self.assertRaisesRegex(LoadStepFailure,'no reliable'):
                p._equilibrate(x,time.monotonic()+90)
        np.testing.assert_array_equal(p.multipliers,multipliers)
        np.testing.assert_array_equal(p.material_metric,history)

    def test_timeout_after_update_captures_synchronized_failed_state(self):
        p=self.candidate(WarmGlobalizedAugmentedBladeProblem)
        with contextlib.redirect_stdout(io.StringIO()):
            x,_=p.step(p.rest.copy(),p.start,0.,time.monotonic()+90)
        material=p.material_metric.copy();last_trial={};expired=[False]
        evaluate=p.evaluate
        def observe(xx,*a,**kw):
            value=evaluate(xx,*a,**kw)
            if kw.get('tangent') is False:
                last_trial.update(positions=xx.copy(),multipliers=p.multipliers.copy())
            return value
        def printed(*args,**kwargs):
            if str(args[0]).startswith('AL_GLOBAL_UPDATE'):expired[0]=True
        # Expire only AFTER an update, not during evaluation or a rejected trial.
        # The snapshot must pair that displacement with the same multipliers.
        with patch.object(p,'evaluate',side_effect=observe),patch('builtins.print',side_effect=printed),patch(
                'prototype.al_globalization.time.monotonic',side_effect=lambda:2. if expired[0] else 0.):
            with self.assertRaises(TimeoutError):p.step(x,0.,.005,1.)
        self.assertTrue(expired[0])
        np.testing.assert_array_equal(p.rejected_al_state['positions'],last_trial['positions'])
        np.testing.assert_array_equal(p.rejected_al_state['al_multipliers'],last_trial['multipliers'])
        np.testing.assert_array_equal(p.material_metric,material)

    def test_conditional_prefix_audit_checks_inherited_and_new_history(self):
        from tools import run_augmented_blade as base
        from tools.replay_al_globalization import sequence,audit_conditional
        from run_support import save_npz
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            initial=base.run(argparse.Namespace(output=root/'initial',angle=57.,coarse=True,
                spacing=.000625,depth=.005,increment=.005,seconds=90.,resume=None,phase='run',blas_threads=1))
            self.assertTrue(initial['completed'])
            result=sequence(root/'initial',root/'continued',start=.005,target=.01,increment=.005,seconds=90)
            self.assertTrue(result['completed']);self.assertEqual(len(result['steps']),2)
            original=make_problem(57,coarse=True,spacing=.000625)
            self.assertTrue(audit_conditional(original,root/'continued',result)['passed'])
            file=root/'continued/states/0001.npz'
            with np.load(file) as z:state={k:z[k].copy() for k in z.files}
            state['previous_positions'][3,0]+=.001
            save_npz(file,**state)
            with self.assertRaises(AssertionError):audit_conditional(original,root/'continued',result)


if __name__=='__main__':unittest.main()
