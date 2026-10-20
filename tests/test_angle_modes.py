"""Physical checks for the bounded diagnostic, not a new constitutive model."""
import unittest
import numpy as np
from prototype.curve_contact import CurveBladeProblem
from prototype.model_config import _rotation_vectors
from tools.check_angle_modes import basis, plane_stress_methods, tensor_parts


class AngleModeTests(unittest.TestCase):
    def problem(self, angle):
        p=CurveBladeProblem(maxwell_shear=1.24,relaxation_s=1.7,loading_rate=.2,
            bulk_modulus=23.34333333333331,shear_modulus=.14,local_spacing=.2,
            surface_spacing=.02,mu=.35,tip_radius=.02,dhat=.0025,inclination=angle)
        p.material_dt=.05
        return p

    def test_rotated_slider_allows_tangent_but_not_normal_or_anchor(self):
        for angle in (30,60):
            p=self.problem(angle);B,anchor=basis(p,True)
            rng=np.random.default_rng(824);v=(B@rng.standard_normal(B.shape[1])).reshape(-1,2)
            es,en=_rotation_vectors(angle);bottom=p.mesh.bottom_nodes
            np.testing.assert_allclose(v[bottom]@en,0,atol=1e-14)
            np.testing.assert_array_equal(v[anchor],0)
            np.testing.assert_array_equal(v[p.n:],0)
            self.assertGreater(np.linalg.norm(v[bottom]@es),1)
            np.testing.assert_allclose((B.T@B).diagonal(),1,atol=1e-14)
            self.assertEqual(B.shape[1],len(p.free)+len(bottom)-1)

    def test_plane_stress_assembled_tangent_and_energy_work(self):
        p=self.problem(30);plane_stress_methods(p)
        u=(p.rest[:p.n]-p.rest[:p.n].mean(axis=0))@np.array([[.03,.02],[-.01,-.04]])
        r,k,dofs,_,_=p.material_bulk(u.ravel())
        rng=np.random.default_rng(22);v=rng.standard_normal(u.shape)*.02
        h=1e-6
        rp,_,_,ep,_=p.material_bulk((u+h*v).ravel(),with_tangent=False)
        rm,_,_,em,_=p.material_bulk((u-h*v).ravel(),with_tangent=False)
        analytic=np.einsum('eij,ej->ei',k,v.ravel()[dofs])
        np.testing.assert_allclose((rp-rm)/(2*h),analytic,rtol=3e-5,atol=3e-7)
        self.assertAlmostEqual((ep-em)/(2*h),float(np.sum(r*v.ravel()[dofs])),places=6)
        x=p.rest.copy();x[:p.n]+=u;q=p.material_response(x)
        self.assertLess(np.max(q['plane_stress_scaled_residual']),1e-8)

    def test_hydrostatic_part_is_isotropic_cauchy(self):
        p=self.problem(60);x=p.rest.copy()
        x[:p.n]=(p.rest[:p.n]@np.array([[1.05,.08],[.02,.94]]))
        from tools.check_angle_modes import deformation
        F=deformation(p,x);q=p.material_response(x)
        stresses,energies=tensor_parts(F,q,p.material)
        sigma=stresses['volume']@F.transpose(0,2,1)/q['J'][:,None,None]
        expected=p.material.moduli()[1]*(q['J']-1)[:,None,None]*np.eye(2)
        np.testing.assert_allclose(sigma,expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(sum(stresses.values()),q['P'],atol=1e-12)
        self.assertTrue(all(np.min(v)>-1e-12 for v in energies.values()))


if __name__=='__main__':unittest.main()
