"""Response restrictions and an explicitly separated volumetric-law control."""
from dataclasses import replace
import unittest
import numpy as np
from mechanics.finite_strain import Material, response, elastic_response, principal


def make_material(eq_alpha=2.,over_alpha=2.,volumetric='quadratic_j'):
    ge,gv,K=.2898,.6762,23.34333333333331
    nu=(3*K-2*ge)/(2*(3*K+ge))
    return Material((2*ge/eq_alpha,),(eq_alpha,),(2*gv/over_alpha,),(over_alpha,),
                    nu,.4,volumetric=volumetric,viscous_flow='isochoric_maxwell')


def deviatoric_cauchy(F,q):
    s=np.zeros((len(F),3,3));s[:,:2,:2]=q['P']@F.transpose(0,2,1)
    s[:,2,2]=q['plane_stress_residual'];s/=q['J'][:,None,None]
    return s-np.trace(s,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3


class MaterialStructureTests(unittest.TestCase):
    def test_same_shape_endpoint_ratio_across_compression_shear_and_stretch(self):
        F=np.array([[[1.,.5],[0.,1.]],[[1.6,0],[0,1/1.6]],[[.8,.2],[0,.7]]])
        m=make_material();old=np.tile(np.eye(3),(len(F),1,1))
        instant=response(F,old,1e-10,m,kinematics='plane_strain',tangent=False)
        current=old.copy()
        for _ in range(150):
            held=response(F,current,.08,m,kinematics='plane_strain',tangent=False)
            current=held['metric']
        np.testing.assert_allclose(deviatoric_cauchy(F,held),.3*deviatoric_cauchy(F,instant),
                                   rtol=2e-7,atol=2e-8)
        self.assertGreater(abs(current[-1,2,2]-1),.01)

    def test_distinct_shapes_change_endpoint_ratio_without_changing_small_G_or_K(self):
        F=np.array([[[v,0],[0,1/v]] for v in (1.1,1.5,2.,3.)]);old=np.tile(np.eye(3),(4,1,1))
        ratios=[]
        for m in (make_material(),make_material(1.,2.)):
            instant=response(F,old,1e-10,m,kinematics='plane_strain',tangent=False)
            eq=elastic_response(F,m,kinematics='plane_strain')
            s=deviatoric_cauchy(F,instant)
            ratios.append(np.linalg.norm(eq['cauchy'],axis=(1,2))/np.linalg.norm(s,axis=(1,2)))
        np.testing.assert_allclose(ratios[0],.3,rtol=1e-8)
        self.assertGreater(np.ptp(ratios[1]),.03)
        self.assertEqual(make_material().moduli(),make_material(1.,2.).moduli())

    def test_log_volume_alias_does_not_switch_to_paper_viscous_flow(self):
        F=np.array([[[.8,.3],[0.,.75]],[[1.2,.2],[0.,1.1]]]);A=np.tile(np.eye(3),(2,1,1))
        m=make_material(volumetric='log_j')
        q=response(F,A,.05,m,kinematics='plane_strain')
        old=response(F,A,.05,replace(m,volumetric='paper2023'),kinematics='plane_strain')
        for key in q:np.testing.assert_array_equal(q[key],old[key])
        self.assertEqual(m.viscous_flow,'isochoric_maxwell')
        self.assertEqual(m.moduli(True)[1],0.)
        for i in range(4):
            d=np.eye(4)[i].reshape(2,2)*1e-6
            a=response(F+d,A,.05,m,kinematics='plane_strain',tangent=False)
            b=response(F-d,A,.05,m,kinematics='plane_strain',tangent=False)
            np.testing.assert_allclose(((a['P']-b['P'])/2e-6).reshape(-1,4),q['tangent'][:,:,i],rtol=3e-6,atol=1e-6)

    def test_volume_formulas_and_plane_strain_total_energy_limit(self):
        J=np.array([.001,.1,.5,1.]);e=np.log(J)[:,None]*np.ones((1,3))/3
        K=make_material().moduli()[1]
        for law in ('quadratic_j','log_j'):
            m=make_material(volumetric=law);w,t,_=principal(e,m)
            pressure=-t.mean(axis=1)/J
            expected=K*(1-J) if law=='quadratic_j' else K/2*(1/J-J)
            np.testing.assert_allclose(pressure,expected,atol=1e-10)
        m=make_material();F=np.array([np.eye(2)*np.sqrt(j) for j in (1e-2,1e-4,1e-6)])
        energy=elastic_response(F,m)['energy']
        self.assertTrue(np.all(np.diff(energy)>0));self.assertGreater(energy[-1],100*energy[0])

    def test_fe_host_volume_selection_is_explicit_and_signed_in_checkpoint_identity(self):
        from prototype.ipc_viscoelastic import ViscoelasticProblem
        from prototype.material_case import signature
        kw=dict(maxwell_shear=.6762,relaxation_s=1.7,loading_rate=.2,bulk_modulus=23.34333333333331,
                shear_modulus=.2898,case='patch',mu=.35,local_spacing=.2,surface_spacing=.1)
        a=ViscoelasticProblem(**kw);b=ViscoelasticProblem(**kw,volumetric='log_j')
        self.assertEqual(a.material.volumetric,'quadratic_j')
        self.assertNotEqual(signature(a),signature(b))
        np.testing.assert_array_equal(a.rest,b.rest)
        self.assertEqual(a.material.moduli(),b.material.moduli())
        self.assertEqual(a.material.viscous_flow,b.material.viscous_flow)
        with self.assertRaisesRegex(ValueError,'volume law'):
            ViscoelasticProblem(**kw,volumetric='unqualified')


if __name__=='__main__':unittest.main()
