"""Positive integration, original-history audit and geometry rejection controls."""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.ipc_p2 import composite_triangle_rule,triangle_rule
from prototype.ipc_reference import LoadStepFailure
from prototype.ipc_viscoelastic_p2 import QuadratureViscoelasticProblem
from tests.test_viscoelastic_p2 import COMMON,assembled
from tools.diagnose_al_geometry import argminimum
from tools.run_al_integration import factory,run_case


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        c=threadpool_limits(1);c.__enter__();self.addCleanup(c.__exit__,None,None,None)

    def test_positive_composite_rule_and_unchanged_default(self):
        for order in (3,5):
            for level in (0,1,2):
                p,w=composite_triangle_rule(order,level)
                self.assertTrue(np.all(w>0));self.assertTrue(np.all(p>0));np.testing.assert_allclose(p.sum(axis=1),1,atol=2e-15)
                if level==0:
                    a,b=triangle_rule(order);np.testing.assert_array_equal(a,p);np.testing.assert_array_equal(b,w)
                for i in range(5):
                    for j in range(5-i):
                        self.assertAlmostEqual(w@(p[:,1]**i*p[:,2]**j),2*math.factorial(i)*math.factorial(j)/math.factorial(i+j+2),places=13)

    def test_localized_minimum_in_edge_and_interior(self):
        l=np.vstack((np.eye(3),[[.5,.5,0],[0,.5,.5],[.5,0,.5]]));x=l[:,1];y=l[:,2]
        for v,want in (((x-.2)**2+y**2+.01,[.8,.2,0]),((x-.2)**2+(y-.3)**2+.01,[.5,.2,.3])):
            minimum,p=argminimum(v);self.assertAlmostEqual(minimum,.01,places=13);np.testing.assert_allclose(p,want,atol=1e-14)

    def test_composite_viscous_energy_tangent_and_exact_affine_history(self):
        a=QuadratureViscoelasticProblem(**COMMON,enrichment_radius=.35)
        b=QuadratureViscoelasticProblem(**COMMON,enrichment_radius=.35,quadrature_refinement=1)
        F=np.array([[1.04,.08],[.01,.94]])
        for p in (a,b):
            p.material_dt=.12;x=p.rest.copy();x[:p.n]=p.mesh.reference_xy@F.T
            p.previous_material_metric=p.material_response(x)['metric'].copy()
        ua=a.mesh.reference_xy@(F-np.eye(2)).T
        ga,Ha,Ea,Ja=assembled(a,ua.ravel());gb,Hb,Eb,Jb=assembled(b,ua.ravel())
        np.testing.assert_allclose(ga,gb,atol=1e-10);self.assertAlmostEqual(Ea,Eb,places=10)
        rng=np.random.default_rng(31);u=rng.normal(size=2*b.n)*.001;d=rng.normal(size=len(u));d/=np.linalg.norm(d);eps=1e-6
        g,H,E,_=assembled(b,u);gp,_,Ep,_=assembled(b,u+eps*d);gm,_,Em,_=assembled(b,u-eps*d)
        np.testing.assert_allclose((gp-gm)/(2*eps),H@d,rtol=2e-6,atol=2e-6)
        self.assertAlmostEqual((Ep-Em)/(2*eps),g@d,delta=1e-7)

    def test_represented_invalid_endpoint_rejects_without_material_trial(self):
        p=factory(57,coarse=True,order=3);self.addCleanup(p.close_compute);p.ipc.set_num_threads(2)
        x=p.affine_loading_candidate(p.rest,p.start,.005);p.previous=p.rest.copy();p.previous_material_metric=p.material_metric.copy()
        p.material_dt=.1;p.prepare_quadrature(x);real=p.enrichment.minimum_j;calls=[]
        def endpoint(u):
            calls.append(True)
            return real(u) if len(calls)==1 else -1e-15
        with contextlib.redirect_stdout(io.StringIO()),patch.object(p,'safe_step',return_value=1.),patch.object(p.enrichment,'minimum_j',side_effect=endpoint):
            with self.assertRaisesRegex(LoadStepFailure,'no reliable'):p._equilibrate(x,time.monotonic()+90)
        self.assertGreater(len(calls),1)

    def test_new_rule_path_restart_and_wrong_rule_rejection(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            def args(out,depth,resume=None,order=5):
                return argparse.Namespace(output=root/out,angle=57.,coarse=True,spacing=.000625,order=order,refinement=0,
                    depth=depth,increment=.005,seconds=120.,resume=resume,phase='case',blas_threads=1)
            first=run_case(args('first',.005));self.assertTrue(first['completed']);self.assertTrue(first['original_kernel_audit_passed'])
            continued=run_case(args('continued',.01,root/'first'));self.assertTrue(continued['completed'])
            independent=run_case(args('independent',.01));self.assertTrue(independent['completed'])
            self.assertAlmostEqual(continued['steps'][-1]['reaction_fz_n'],independent['steps'][-1]['reaction_fz_n'],delta=1e-7)
            with self.assertRaisesRegex(ValueError,'independent history'):run_case(args('bad',.01,root/'first',order=8))


if __name__=='__main__':unittest.main()
