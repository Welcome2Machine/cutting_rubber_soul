"""History, trace, derivative and native-contact controls for opt-in enrichment."""
import contextlib
import io
import time
import unittest
import numpy as np
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits
from prototype.ipc_viscoelastic import ViscoelasticProblem
from prototype.ipc_viscoelastic_p2 import QuadratureViscoelasticProblem


COMMON = dict(maxwell_shear=1.24, relaxation_s=1.7, loading_rate=.2,
    bulk_modulus=23.34333333333331, shear_modulus=.14, mu=.35,
    case='patch', local_spacing=.2, surface_spacing=.2, max_iterations=500,
    hessian_mode='regularized')


def assembled(p, u):
    r,k,d,E,J=p.material_bulk(u)
    g=np.zeros(p.n*2);np.add.at(g,d.ravel(),r.ravel())
    H=coo_matrix((k.ravel(),(np.broadcast_to(d[:,:,None],k.shape).ravel(),
        np.broadcast_to(d[:,None,:],k.shape).ravel())),shape=(len(g),len(g))).tocsr()
    return g,H,E,J


class QuadratureViscoelasticTests(unittest.TestCase):
    def test_p1_host_remains_identical_with_explicit_material_layout(self):
        a=ViscoelasticProblem(**COMMON);b=QuadratureViscoelasticProblem(**COMMON)
        np.testing.assert_array_equal(a.rest,b.rest)
        F=np.array([[1.04,.08],[.01,.93]])
        for p in (a,b):
            p.material_dt=.13
            x=p.rest.copy();x[:p.n]=p.mesh.reference_xy@F.T
            p.previous_material_metric=p.material_response(x)['metric'].copy()
        u=(a.mesh.reference_xy@(F-np.eye(2)).T).ravel()
        ga,Ha,Ea,Ja=assembled(a,u);gb,Hb,Eb,Jb=assembled(b,u)
        np.testing.assert_array_equal(ga,gb)
        np.testing.assert_array_equal(Ha.toarray(),Hb.toarray())
        self.assertEqual(Ea,Eb);self.assertEqual(Ja,Jb)

    def test_enriched_p1_subspace_preserves_history_and_force_pullback(self):
        a=QuadratureViscoelasticProblem(**COMMON)
        b=QuadratureViscoelasticProblem(**COMMON,enrichment_radius=.35)
        self.assertGreater(len(b.active_elements),0);self.assertGreater(len(b.linear_elements),0)
        np.testing.assert_array_equal(a.rest[a.surface],b.rest[b.surface])
        np.testing.assert_array_equal(a.cmesh.edges,b.cmesh.edges)
        rng=np.random.default_rng(881)
        u=rng.normal(size=(a.n,2))*.005
        ub=np.vstack((u,u[b.enrichment.interior_edges].mean(axis=1)))
        for p,v in ((a,u),(b,ub)):
            p.material_dt=.12;x=p.rest.copy();x[:p.n]+=.5*v
            p.previous_material_metric=p.material_response(x)['metric'].copy()
        np.testing.assert_allclose(b.previous_material_metric,
            a.previous_material_metric[b.point_elements],atol=2e-14)
        ga,Ha,Ea,_=assembled(a,u.ravel());gb,Hb,Eb,_=assembled(b,ub.ravel())
        gb=gb.reshape(-1,2);pull=gb[:a.n].copy()
        for side in (0,1):np.add.at(pull,b.enrichment.interior_edges[:,side],.5*gb[a.n:])
        np.testing.assert_allclose(pull.ravel(),ga,atol=2e-11)
        self.assertAlmostEqual(Ea,Eb,delta=1e-11)

    def test_actual_quadrature_potential_and_tangent_in_both_constraints(self):
        for kin in ('plane_strain','plane_stress'):
            p=QuadratureViscoelasticProblem(**COMMON,enrichment_radius=.35,kinematics=kin)
            rng=np.random.default_rng(34);u=rng.normal(size=2*p.n)*.002
            p.material_dt=.11;x=p.rest.copy();x[:p.n]+=.5*u.reshape(-1,2)
            p.previous_material_metric=p.material_response(x)['metric'].copy()
            d=rng.normal(size=len(u));d/=np.linalg.norm(d);eps=1e-6
            g,H,E,_=assembled(p,u)
            gp,_,Ep,_=assembled(p,u+eps*d);gm,_,Em,_=assembled(p,u-eps*d)
            np.testing.assert_allclose((gp-gm)/(2*eps),H@d,rtol=2e-6,atol=2e-6)
            self.assertAlmostEqual((Ep-Em)/(2*eps),g@d,delta=1e-7)

    def test_native_path_commits_and_audits_every_quadrature_history(self):
        for kin in ('plane_strain','plane_stress'):
            p=QuadratureViscoelasticProblem(**COMMON,enrichment_radius=.35,kinematics=kin)
            x=p.rest.copy();old=p.start
            with threadpool_limits(2),contextlib.redirect_stdout(io.StringIO()):
                for depth in (0.,.01,.02):
                    prior=p.material_metric.copy()
                    x,row=p.step(x,old,depth,time.monotonic()+30);old=depth
                    np.testing.assert_array_equal(p.previous_material_metric,prior)
                    state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,
                        previous_positions=p.previous,depth_mm=depth,**p.checkpoint_data())
                    self.assertTrue(p.audit_state(state,row)['passed'])
                    bad=dict(state,kinematics=np.array('incorrect'))
                    with self.assertRaisesRegex(RuntimeError,'discretization'):
                        p.audit_state(bad,row)


if __name__=='__main__':unittest.main()
