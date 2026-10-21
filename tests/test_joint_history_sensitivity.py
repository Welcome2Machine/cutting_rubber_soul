"""Whole material histories for common stiffness and branch partition."""
import unittest
import numpy as np
from mechanics.finite_strain import Material, response, elastic_response
from prototype.history_sensitivity import common_material,metric_perturb,metric_direction


class JointHistoryTests(unittest.TestCase):
    def test_two_increment_improvement_cannot_certify_slope_evolution(self):
        from tools.report_joint_history import common_step
        c=dict(baseline_fz_n=[4.,5.],experimental_fz_n=[2.,3.],derivatives_n=[[-1.,0,0],[-1.,0,0]],
            slope_points_mm=[.1,.25,.5],baseline_slopes_n_mm=[4.,3.],experimental_slopes_n_mm=[2.,3.],
            slope_derivatives=[[-1.,0,0],[-2.,0,0]])
        self.assertEqual(common_step({'0':c},[1.,0,0])['status'],'completed')
        self.assertEqual(common_step({'0':c},[1.,0,0],include_slope_change=True)['status'],'no_strict_common_step')

    def test_fixed_bulk_explicit_partials_match_analytic_branch_partition(self):
        ge,gv,k=.14,1.24,23.34333333333331;g0=ge+gv;q=ge/g0
        mat=Material((ge,),(2.,),(gv,),(2.,),(3*k-2*ge)/(2*(3*k+ge)),1.7,viscous_flow='isochoric_maxwell')
        F=np.array([[[1.1,.3],[.03,.92]],[[.85,1.2],[0.,1.15]]])
        history=response(.5*(F+np.eye(2)),np.tile(np.eye(3),(2,1,1)),.1,mat,tangent=False,kinematics='plane_strain')['metric']
        full=response(F,history,.05,mat,tangent=False,kinematics='plane_strain')
        equilibrium=elastic_response(F,mat,tangent=False,kinematics='plane_strain')['P']
        J=np.linalg.det(F)
        bulk=(k*J*(J-1))[:,None,None]*np.linalg.inv(F).transpose(0,2,1)
        expected={'log_g0':full['P']-bulk,
                  'logit_q':g0*q*(1-q)*((equilibrium-bulk)/ge-(full['P']-equilibrium)/gv)}
        # With fixed F/history/tau and alpha=2, Gv cancels from the local flow
        # equation. Changing branch weights changes stress, not this return map.
        for key in expected:
            eps=1e-4
            plus=response(F,history,.05,common_material(mat,key,eps),tangent=False,kinematics='plane_strain')
            minus=response(F,history,.05,common_material(mat,key,-eps),tangent=False,kinematics='plane_strain')
            np.testing.assert_allclose((plus['P']-minus['P'])/(2*eps),expected[key],rtol=2e-7,atol=1e-8)
            np.testing.assert_allclose(plus['metric'],minus['metric'],rtol=1e-12,atol=1e-13)

    def test_common_tangent_step_stops_before_any_individual_error_rises(self):
        from tools.report_joint_history import common_step
        curves={}
        for name,excess in (('0',2.),('33',.1),('57',.5)):
            curves[name]=dict(baseline_fz_n=[1+excess,2+excess],experimental_fz_n=[1.,2.],
                derivatives_n=[[1.,0,0],[1.,0,0]],slope_points_mm=[.1,.25,.5],
                baseline_slopes_n_mm=[1+excess,1+excess],experimental_slopes_n_mm=[1.,1.],
                slope_derivatives=[[1.,0,0],[1.,0,0]])
        r=common_step(curves,[-1.,0,0])
        self.assertAlmostEqual(r['fraction'],.1)
        self.assertTrue(all(v<1 for v in r['objective_ratios']))
        # The unconstrained unit ray improves total error but would harm 33.
        self.assertGreater((.1-1)**2,.1**2)

    def test_common_tangent_step_rejects_one_conflicting_angle(self):
        from tools.report_joint_history import common_step
        c=dict(baseline_fz_n=[2.,3.],experimental_fz_n=[1.,2.],
               derivatives_n=[[-1.,0,0],[-1.,0,0]],slope_points_mm=[.1,.25,.5],
               baseline_slopes_n_mm=[2.,2.],experimental_slopes_n_mm=[1.,1.],
               slope_derivatives=[[-1.,0,0],[-1.,0,0]])
        self.assertEqual(common_step({'57':c},[-1.,0,0])['status'],'no_strict_common_step')

    def test_common_descent_does_not_hide_conflicting_angle_objectives(self):
        from tools.report_joint_history import descent
        r=descent([[1.,0,0],[-1.,0,0],[0,1.,0]])
        self.assertFalse(r['strict_common_descent'])
        self.assertAlmostEqual(r['normalized_margin'],0.,places=10)
        self.assertTrue(r['linearized_conflict_certificate']['verified'])
        self.assertAlmostEqual(sum(r['linearized_conflict_certificate']['weights']),1.)

    def test_common_descent_reduces_every_nonconflicting_objective(self):
        from tools.report_joint_history import descent
        g=np.array([[1.,2.,.1],[2.,1.,.2],[.5,1.,1.]])
        r=descent(g)
        self.assertTrue(r['strict_common_descent'])
        self.assertTrue(np.all(g@r['coordinate_direction']<0))

    def test_stiffness_and_partition_history_match_independent_reintegration(self):
        ge,gv,k=.14,1.24,23.34333333333331
        mat=Material((ge,),(2.,),(gv,),(2.,),(3*k-2*ge)/(2*(3*k+ge)),1.7,viscous_flow='isochoric_maxwell')
        for key in ('log_g0','logit_q'):
            for eps in (1e-4,3e-5):
                A=np.eye(3)[None];D=np.zeros_like(A);ap=A.copy();am=A.copy()
                mp=common_material(mat,key,eps);mm=common_material(mat,key,-eps)
                for i in range(1,31):
                    F=np.array([[[1+.002*i,.01*i],[0.,1/(1+.002*i)]]])
                    q=response(F,A,.05,mat,tangent=False)
                    p=response(F,metric_perturb(A,D,eps),.05,mp,tangent=False)
                    m=response(F,metric_perturb(A,D,-eps),.05,mm,tangent=False)
                    fp=response(F,ap,.05,mp,tangent=False);fm=response(F,am,.05,mm,tangent=False)
                    D=metric_direction(q['metric'],p['metric'],m['metric'],eps)
                    A,ap,am=q['metric'],fp['metric'],fm['metric']
                np.testing.assert_allclose((p['P']-m['P'])/(2*eps),(fp['P']-fm['P'])/(2*eps),rtol=1e-5,atol=1e-8)
                np.testing.assert_allclose(D,metric_direction(A,ap,am,eps),rtol=1e-5,atol=1e-8)


if __name__=='__main__': unittest.main()
