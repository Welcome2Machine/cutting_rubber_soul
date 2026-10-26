import unittest
import numpy as np
from mechanics.finite_strain import Material,response,elastic_response
from prototype.material import bulk,from_config
from prototype.model_config import FEMConfig
from types import SimpleNamespace
from prototype.geometry import intact_mesh,GeometryOptions


class SharedMaterialTests(unittest.TestCase):
    def test_closed_form_neo_hookean_matches_general_spectral_kernel(self):
        from mechanics.finite_strain import _spectral_elastic_response
        rng=np.random.default_rng(17)
        F=np.eye(2)+rng.normal(size=(1000,2,2))*.4
        F=F[np.linalg.det(F)>.1]
        material=Material((.66,),(2.,))
        fast=elastic_response(F,material);general=_spectral_elastic_response(F,material)
        for key in fast:
            np.testing.assert_allclose(fast[key],general[key],rtol=1e-10,atol=1e-10)

    def test_assembled_bulk_force_is_energy_gradient(self):
        c=FEMConfig();p=SimpleNamespace(mesh=intact_mesh(c,GeometryOptions(refinement_mm=.5,surface_spacing_mm=.1))[0])
        u=np.column_stack((.15*p.mesh.reference_xy[:,0]+.07*p.mesh.reference_xy[:,1],-.04*p.mesh.reference_xy[:,1])).ravel()
        residual,_,dofs,_,_=bulk(p.mesh,u,c)
        direction=np.random.default_rng(12).normal(size=u.shape);eps=1e-6
        plus=bulk(p.mesh,u+eps*direction,c,with_tangent=False)[3]
        minus=bulk(p.mesh,u-eps*direction,c,with_tangent=False)[3]
        self.assertAlmostEqual(float(np.sum(residual*direction[dofs])),(plus-minus)/(2*eps),delta=1e-5)
        q=elastic_response(np.array([[[1.2,.1],[0,.93]]]),from_config(c))
        self.assertEqual(q['lambda_out'][0],1.)
        self.assertGreater(abs(q['cauchy'][0,2,2]),.01)

    def test_plane_strain_viscoelastic_tangent_and_no_trial_history_commit(self):
        m=Material((.66,),(2.,),(.2,),(2.,),.49,1.5)
        F=np.array([np.eye(2),[[1.3,.3],[-.1,.85]]]);A=np.tile(np.eye(3),(2,1,1));saved=A.copy()
        q=response(F,A,.4,m,kinematics='plane_strain')
        np.testing.assert_array_equal(q['lambda_out'],1.)
        for i in range(4):
            delta=np.eye(4)[i].reshape(2,2)*1e-6
            a=response(F+delta,A,.4,m,kinematics='plane_strain',tangent=False)
            b=response(F-delta,A,.4,m,kinematics='plane_strain',tangent=False)
            np.testing.assert_allclose(((a['P']-b['P'])/2e-6).reshape(2,4),q['tangent'][:,:,i],rtol=1e-6,atol=2e-7)
        np.testing.assert_array_equal(A,saved)

    def test_elastic_energy_derivative_and_both_volumetric_forms(self):
        F=np.array([[[1.3,.12],[.03,.8]],np.eye(2)])
        for volume in ('quadratic_j','paper2023'):
            m=Material((.66,),(2.,),volumetric=volume)
            for mode in ('plane_strain','plane_stress'):
                q=elastic_response(F,m,kinematics=mode)
                for i in range(4):
                    delta=np.eye(4)[i].reshape(2,2)*1e-6
                    a=elastic_response(F+delta,m,kinematics=mode);b=elastic_response(F-delta,m,kinematics=mode)
                    np.testing.assert_allclose((a['energy']-b['energy'])/2e-6,q['P'].reshape(-1,4)[:,i],atol=1e-8)
                    np.testing.assert_allclose(((a['P']-b['P'])/2e-6).reshape(-1,4),q['tangent'][:,:,i],rtol=1e-6,atol=1e-7)

    def test_uncalibrated_viscosity_is_not_silently_enabled(self):
        self.assertFalse(from_config(FEMConfig()).mu_ov)
        with self.assertRaises(ValueError):Material((.66,),(2.,),(.2,),(2.,))

    def test_normal_indentation_geometry_is_supported(self):
        c=FEMConfig(inclination_deg=0.)
        p=SimpleNamespace(mesh=intact_mesh(c,GeometryOptions(refinement_mm=.5,surface_spacing_mm=.1))[0])
        np.testing.assert_allclose(p.mesh.reference_xy,p.mesh.local_sn,atol=1e-12)


if __name__=='__main__':unittest.main()
