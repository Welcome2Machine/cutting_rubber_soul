"""Independent local and assembled checks for the explicit transverse control."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from mechanics.finite_strain import Material,response
from prototype.transverse_control import PlaneStressControl
from tools.probe_angle_factors import parameter_state
from prototype.material_case import signature
from tools.run_transverse_study import jobs,features


class TransverseControlTests(unittest.TestCase):
    def setUp(self):
        ge=.14;k=23.34333333333331
        self.m=Material((ge,),(2.,),(1.24,),(2.,),poisson=(3*k-2*ge)/(2*(3*k+ge)),
                        relaxation_s=1.7,viscous_flow='isochoric_maxwell')

    def test_factor_parameters_change_one_coordinate_and_keep_k(self):
        cfg=__import__('prototype.model_config',fromlist=['FEMConfig']).FEMConfig()
        for name in ('ge','gv','tau','bulk','friction'):
            m,c=parameter_state(self.m,cfg,name,.01)
            original=np.array([self.m.moduli()[0],self.m.moduli(True)[0],1.7,self.m.moduli()[1],cfg.friction_coefficient])
            actual=np.array([m.moduli()[0],m.moduli(True)[0],m.relaxation_s,m.moduli()[1],c.friction_coefficient])
            expected=original.copy();expected[('ge','gv','tau','bulk','friction').index(name)]*=np.exp(.01)
            np.testing.assert_allclose(actual,expected,rtol=1e-11)

    def test_total_stress_elimination_and_history_consistent_tangent(self):
        A=np.tile(np.eye(3),(3,1,1))
        F0=np.array([[[1.1,.25],[0.,.8]],[[.85,-.3],[.1,1.05]],[[.75,.1],[0.,.7]]])
        old=response(F0,A,.15,self.m,tangent=False,kinematics='plane_stress')['metric']
        p=PlaneStressControl();p.material=self.m;p.previous_material_metric=old;p.material_dt=.05
        F=F0+np.array([[.02,.01],[-.005,-.015]])
        q=p.constitutive_response(F,tangent=True)
        self.assertLess(q['plane_stress_scaled_residual'].max(),1e-8)
        self.assertGreater(q['lambda_out'].min(),0.)
        self.assertGreater(q['J'].min(),0.)
        self.assertGreaterEqual(q['viscous_rate'].min(),-1e-10)
        np.testing.assert_allclose(np.linalg.det(q['metric']),1.,atol=1e-9)
        self.assertGreater(np.max(abs(q['lambda_out']-1)),.01)
        D=np.array([[.2,-.13],[.07,-.11]])
        h=1e-6
        plus=p.constitutive_response(F+h*D,tangent=False)
        minus=p.constitutive_response(F-h*D,tangent=False)
        actual=(plus['P']-minus['P'])/(2*h)
        analytic=(q['tangent']@D.ravel()).reshape(-1,2,2)
        np.testing.assert_allclose(actual,analytic,rtol=3e-5,atol=2e-7)
        de=(plus['incremental_potential']-minus['incremental_potential'])/(2*h)
        np.testing.assert_allclose(de,np.einsum('eij,ij->e',q['P'],D),rtol=3e-5,atol=2e-7)
        # A material call must not commit the previous viscous history.
        np.testing.assert_array_equal(p.previous_material_metric,old)

    def test_small_strain_independent_plane_stress_matrix(self):
        dt=.05;G=.14+1.24/(1+dt/1.7);K=self.m.moduli()[1];lam=K-2*G/3
        strain=np.array([[.2,.13],[.13,-.11]])*1e-7
        q=response((np.eye(2)+strain)[None],np.eye(3)[None],dt,self.m,kinematics='plane_stress')
        expected=2*G*strain+2*G*lam/(lam+2*G)*np.trace(strain)*np.eye(2)
        np.testing.assert_allclose(q['P'][0],expected,rtol=5e-6,atol=2e-12)

    def test_signature_separates_constraint_without_changing_old_identity(self):
        p=SimpleNamespace(config=SimpleNamespace(to_dict=lambda:dict(width_mm=20)),material=self.m,
            loading_rate=.2,rest=np.zeros((3,2)),mesh=SimpleNamespace(triangles=np.array([[0,1,2]])),
            contact_formulation='native',dhat=.0025)
        old=signature(p);self.assertNotIn('material_kinematics',old)
        p.material_kinematics='plane_stress';new=signature(p)
        self.assertEqual(new.pop('material_kinematics'),'plane_stress');self.assertEqual(old,new)

    def test_queue_has_common_material_bounded_budget_and_no_parameter_fit(self):
        tasks=jobs();self.assertEqual([j['angle'] for j in tasks],[0,33,57])
        for j in tasks:
            self.assertEqual(j['parameters'],[.14,1.24,1.7]);self.assertEqual(j['seconds'],3600)
            self.assertEqual(j['target_mm'],.5);self.assertNotIn('resume',j)
        d=np.arange(51)/100;f=10*d+.51
        self.assertIsNone(features(d[:-1],f[:-1],d,f))
        self.assertAlmostEqual(features(d,f,d,10*d)['rmse_n'],.51)


if __name__=='__main__':unittest.main()
