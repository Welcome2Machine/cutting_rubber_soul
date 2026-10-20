"""Equivalence of contact blocks and fresh-physics acceptance of inner solves."""
import contextlib
import io
import time
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.contact_refresh import ContactRefreshMixin, contact_terms
from prototype.compact_p2_assembly import CompactP2AssemblyMixin, compact_matrices
from prototype.ipc_reference import BASELINE, LoadStepFailure
from tools.run_angle_path_study import make_problem, signature


class ContactRefreshTests(unittest.TestCase):
    def setUp(self):
        ctx=threadpool_limits(2);ctx.__enter__();self.addCleanup(ctx.__exit__,None,None,None)

    def candidate(self,p):
        p.__class__=type('TestContactRefresh',(ContactRefreshMixin,CompactP2AssemblyMixin,type(p)),{})
        return p

    def test_decomposed_force_tangent_and_metric_match_original_assembly(self):
        p=make_problem(57,'oblique',coarse=True)
        p.material_dt=.05
        p.previous=p.affine_loading_candidate(p.rest,p.start,.002)
        x=p.affine_loading_candidate(p.previous,.002,.007)
        r,k,dofs,_,_=p.material_bulk((x[:p.n]-p.mesh.reference_xy).ravel())
        gb=np.zeros(p.total);np.add.at(gb,dofs.ravel(),r.ravel())
        hb,mb=compact_matrices(k,dofs,p.total,p.linear_elements,p.active_elements,with_metric=True)
        gc,jc,mc=contact_terms(p,x,tangent=True)
        _,g,H,_,M=p.evaluate(x,p.lag(x),projected=False,with_metric=True)
        J=p.coupled_friction_tangent(x,p.lag(x),H)
        np.testing.assert_allclose(g,gb+gc,atol=1e-12,rtol=0)
        for a,b in ((J,hb+jc),(M,mb+mc)):
            self.assertLess(np.max(np.abs((a-b).data),initial=0),1e-8)

    def test_same_accepted_history_replay_passes_full_physics_and_motion(self):
        p=make_problem(57,'oblique',coarse=True);q=self.candidate(make_problem(57,'oblique',coarse=True))
        self.assertEqual(signature(p),signature(q));x=p.rest.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            x,_=p.step(x,p.start,0.,time.monotonic()+40)
            metric=p.material_metric.copy()
            for z in (p,q):
                z.material_metric=metric.copy();z.material_dissipation=p.material_dissipation
                z.predictor='affine'
            a,ra=p.step(x,0.,.005,time.monotonic()+40)
            b,rb=q.step(x,0.,.005,time.monotonic()+40)
        np.testing.assert_array_equal(q.previous,x)
        np.testing.assert_array_equal(q.previous_material_metric,metric)
        state=dict(reference_xy=q.rest,triangles=q.mesh.triangles,positions=b,
                   previous_positions=q.previous,depth_mm=.005,**q.checkpoint_data())
        self.assertTrue(q.audit_state(state,rb)['passed'])
        self.assertLess(rb['residual_n'],BASELINE['force_tolerance_n'])
        self.assertLess(abs(ra['reaction_fz_n']-rb['reaction_fz_n']),.002)
        self.assertLess(np.max(np.linalg.norm(a-b,axis=1)),.002)

    def test_failed_inner_solve_never_commits_material_history_or_loose_acceptance(self):
        p=self.candidate(make_problem(57,'oblique',coarse=True));p.max_iterations=0
        before=p.material_metric.copy();diss=p.material_dissipation
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(LoadStepFailure):
            p.step(p.rest,p.start,.005,time.monotonic()+10)
        np.testing.assert_array_equal(p.material_metric,before)
        self.assertEqual(p.material_dissipation,diss)
        original=make_problem(57,'oblique',coarse=True)
        self.assertNotIsInstance(original,ContactRefreshMixin)
        self.assertEqual(BASELINE['force_tolerance_n'],1e-4)


if __name__=='__main__':unittest.main()
