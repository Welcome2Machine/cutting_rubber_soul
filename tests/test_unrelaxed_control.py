import unittest
from dataclasses import replace
import numpy as np
from mechanics.finite_strain import Material, principal, elastic_response
from prototype.unrelaxed_control import UnrelaxedProblem
from tools.check_contact_transport import closest_blade, weighted_quantiles


class UnrelaxedControlTests(unittest.TestCase):
    def test_bounded_schedule_preserves_zero_and_final_target(self):
        from tools.run_unrelaxed_study import targets
        d=targets(.5,.01)
        self.assertEqual(len(d),51)
        self.assertEqual(d[0],0.)
        self.assertEqual(d[-1],.5)
        self.assertTrue(np.all(np.diff(d)>0))
        for depth in (0.,1.,2.):
            with self.assertRaises(ValueError):targets(depth,.01)

    def test_two_frozen_branches_equal_merged_principal_energy_stress_tangent(self):
        K=23.34333333333331
        def nu(g): return (3*K-2*g)/(2*(3*K+g))
        m=Material(mu_eq=(.14,),alpha_eq=(2.,),mu_ov=(1.24,),alpha_ov=(2.,),
                   poisson=nu(.14),relaxation_s=1.7,viscous_flow='isochoric_maxwell')
        merged=Material(mu_eq=(1.38,),alpha_eq=(2.,),poisson=nu(1.38))
        e=np.array([[0.,0.,0.],[.7,-.3,0.],[-1.1,.2,0.],[.2,.2,.2]])
        a=principal(e,m);b=principal(e,m,True);c=principal(e,merged)
        for x,y,z in zip(a,b,c):np.testing.assert_allclose(x+y,z,rtol=2e-13,atol=2e-13)
        self.assertAlmostEqual(merged.moduli()[1],K,places=11)

    def test_assembly_is_same_as_audited_material_and_independent_of_time(self):
        p=UnrelaxedProblem(case='patch',mu=.35,dhat=.0025,local_spacing=.1,surface_spacing=.02,nose_facets=8)
        p.material_dt=.05
        rng=np.random.default_rng(917)
        u=rng.normal(size=2*p.n)*.0001
        fast=p.material_bulk(u)
        from prototype.ipc_viscoelastic import ViscoelasticProblem
        full=ViscoelasticProblem.material_bulk(p,u)
        for a,b in zip(fast,full):np.testing.assert_allclose(a,b,rtol=1e-11,atol=2e-10)
        F=np.array([[[1.1,.3],[-.1,.75]],[[.5,.1],[.2,1.4]]])
        q=p.constitutive_response(F,tangent=True)
        p.material_dt=100.
        other=p.constitutive_response(F,tangent=True)
        for key in q:np.testing.assert_array_equal(q[key],other[key])
        np.testing.assert_array_equal(q['metric'],np.tile(np.eye(3),(2,1,1)))
        self.assertTrue(np.all(q['viscous_rate']==0))
        self.assertEqual(p.material.mu_ov,())

    def test_invalid_inherited_material_history_is_rejected(self):
        p=object.__new__(UnrelaxedProblem)
        state={'material_metric':np.eye(3)[None].copy(),'previous_material_metric':np.eye(3)[None].copy()}
        state['previous_material_metric'][0,0,0]=1.01
        with self.assertRaisesRegex(ValueError,'cannot inherit'):p.audit_state(state,{})

    def test_closest_polygon_and_zero_force_missing_values(self):
        blade=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        d,k=closest_blade(np.array([[.5,-.2],[-.3,.5],[2.,2.]]),blade)
        np.testing.assert_allclose(d,[.2,.3,2**.5])
        self.assertEqual(weighted_quantiles(np.arange(3),np.zeros(3)),[None,None,None])
        self.assertEqual(weighted_quantiles(np.array([2.,0.,1.]),np.array([0.,1.,0.])),[0.,0.,0.])


if __name__=='__main__':unittest.main()
