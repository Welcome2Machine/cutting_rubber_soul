"""Physical consistency of the shared finite-strain shape candidate."""
import unittest
import numpy as np
from mechanics.finite_strain import Material, response, elastic_response
from tools.check_material_shape import shape_material, decide, ALPHAS, ANCHORS


def base():
    g=.14; k=23.34333333333331
    return Material((g,), (2.,), (1.24,), (2.,), (3*k-2*g)/(2*(3*k+g)),
                    1.7, viscous_flow='isochoric_maxwell')


class MaterialShapeTests(unittest.TestCase):
    def test_shape_keeps_small_moduli_bulk_and_control(self):
        m=base()
        self.assertEqual(shape_material(m, 2.), m)
        for alpha in ALPHAS:
            q=shape_material(m, alpha)
            np.testing.assert_allclose(q.moduli(), m.moduli(), rtol=1e-14)
            np.testing.assert_allclose(q.moduli(True), m.moduli(True), rtol=1e-14)
            self.assertEqual(q.relaxation_s, m.relaxation_s)
        for a in (0., .5, 5., float('nan')):
            with self.assertRaises(ValueError):shape_material(m, a)

    def test_updated_tangent_potential_objectivity_and_dissipation(self):
        F=np.array([np.eye(2), [[.7,.0],[0,1/.7]], [[1.,1.1],[0,1.]], [[1.3,.4],[.1,.8]]])
        theta=.6; Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        for alpha in ALPHAS:
            m=shape_material(base(), alpha); A=np.tile(np.eye(3),(len(F),1,1))
            A=response(F,A,.05,m,kinematics='plane_strain')['metric']; old=A.copy()
            trial=F+.03*np.array([[.2,.5],[-.1,.3]])
            q=response(trial,A,.05,m,kinematics='plane_strain')
            for i in range(4):
                D=1e-6*np.eye(4)[i].reshape(2,2)
                plus=response(trial+D,A,.05,m,kinematics='plane_strain',tangent=False)
                minus=response(trial-D,A,.05,m,kinematics='plane_strain',tangent=False)
                np.testing.assert_allclose((plus['incremental_potential']-minus['incremental_potential'])/2e-6,
                    q['P'].reshape(-1,4)[:,i],rtol=4e-6,atol=3e-7)
                np.testing.assert_allclose(((plus['P']-minus['P'])/2e-6).reshape(-1,4),
                    q['tangent'][:,:,i],rtol=4e-6,atol=4e-7)
            rotated=response(Q@trial,A,.05,m,kinematics='plane_strain')
            np.testing.assert_allclose(rotated['P'], Q@q['P'],atol=1e-10)
            np.testing.assert_allclose(rotated['metric'],q['metric'],atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(q['metric']),1.,atol=1e-11)
            np.testing.assert_array_equal(A,old)
            self.assertGreaterEqual(q['viscous_rate'].min(),-1e-12)

    def test_small_shear_relaxation_time_is_preserved(self):
        F=np.array([[[1.,1e-5],[0.,1.]]])
        for alpha in ALPHAS:
            m=shape_material(base(),alpha); A=np.eye(3)[None]; stresses=[]
            eq=elastic_response(F,m,kinematics='plane_strain')['P'][0,0,1]
            for _ in range(8):
                q=response(F,A,.05,m,kinematics='plane_strain',tangent=False)
                A=q['metric']; stresses.append(q['P'][0,0,1]-eq)
            np.testing.assert_allclose(np.array(stresses[1:])/stresses[:-1],1/(1+.05/1.7),rtol=2e-7)

    def test_common_softening_that_worsens_57_is_not_promoted(self):
        rows=[]
        for angle in (33,57):
            for d in ANCHORS:
                f=(12. if angle==33 else 8.5)*d/.5
                rows.append(dict(angle=angle,depth_mm=d,candidates={str(a):dict(work_proxy_n=
                    f if a==2. else .7*f) for a in ALPHAS}))
        decision=decide(rows,{33:[4.,8.5],57:[4.6,8.5]})
        self.assertFalse(any(v['promising_direction'] for v in decision['candidates'].values()))

    def test_runner_uses_shared_shape_and_records_opt_in_predictor(self):
        from tools.run_material_candidate import make_problem, signature, resume_compatibility
        from tools.check_material_shape import COMMON
        from tools.replay_affine_policy import ActualAffinePolicy
        from tools.run_material_queue import make_command
        p=make_problem(angle=33,**COMMON,equilibrium_alpha=1.5,maxwell_alpha=1.5,actual_affine_policy=True)
        self.assertIsInstance(p,ActualAffinePolicy)
        self.assertEqual(p.material,shape_material(base(),1.5))
        self.assertAlmostEqual(p.config.bulk_modulus_n_mm2,COMMON['bulk_modulus'],places=10)
        sig=signature(p)
        self.assertEqual(sig['predictor_policy'],'actual-affine-stagnation-rule')
        incompatible=dict(sig);incompatible.pop('predictor_policy')
        with self.assertRaises(ValueError):resume_compatibility(dict(signature=incompatible),dict(signature=sig))
        m=dict(python='python',solve_seconds_per_case=100,material=dict(COMMON,equilibrium_alpha=1.5,maxwell_alpha=1.5),actual_affine_policy=True,shear_limit_mpa=1.)
        for angle in (33,57):
            cmd=make_command(m,dict(angle=angle,target_mm=.5,output='new',solve_seconds=50))
            self.assertEqual(cmd[cmd.index('--equilibrium-alpha')+1],'1.5')
            self.assertEqual(cmd[cmd.index('--maxwell-alpha')+1],'1.5')
            self.assertIn('--actual-affine-policy',cmd)
            self.assertEqual(cmd[cmd.index('--seconds')+1],'50')
            self.assertEqual(cmd[cmd.index('--shear-limit-mpa')+1],'1.0')


if __name__=='__main__': unittest.main()
