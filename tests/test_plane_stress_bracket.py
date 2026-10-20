"""Synthetic strong-compression regressions; no experimental fixtures."""
import unittest
import numpy as np
from scipy.optimize import brentq
from mechanics.finite_strain import Material,elastic_response,principal


class PlaneStressBracketTests(unittest.TestCase):
    def setUp(self):
        self.material=Material((.47,),(2.,),poisson=.49)
        # Simple in-plane compression has negative thickness curvature at z=1.
        # Include a shear/rotation, strong compression and large tension.
        self.F=np.array([[[.55,0],[0,.6]],[[.55,.7],[0,.6]],
                         [[.1,0],[0,.1]],[[2.5,.4],[0,2.]],np.eye(2)])

    def test_compression_negative_initial_curvature_has_stable_root(self):
        e2=np.log([.55,.6])
        _,tau,h=principal(np.r_[e2,0.][None],self.material)
        self.assertLess(tau[0,2],0.)
        self.assertLess(h[0,2,2],0.)
        q=elastic_response(self.F,self.material,kinematics='plane_stress')
        self.assertTrue(np.all(q['lambda_out']>0))
        self.assertLess(np.max(q['plane_stress_scaled_residual']),1e-9)
        pe=elastic_response(self.F,self.material)
        self.assertTrue(np.all(q['energy']<=pe['energy']+1e-10))
        np.testing.assert_allclose(q['J'],np.linalg.det(self.F)*q['lambda_out'],rtol=1e-13)

    def test_matches_independent_brent_scalar_equation(self):
        G,K=self.material.moduli()
        for F in self.F:
            d=np.linalg.det(F);a=np.sum(F*F)
            # Direct Kirchhoff stress formula; independent of principal().
            def tau(q):
                z=np.exp(q);J=d*z
                return G/3*J**(-2/3)*(2*z*z-a)+K*J*(J-1)
            root=brentq(tau,-15.,15.,xtol=5e-14)
            actual=elastic_response(F[None],self.material,kinematics='plane_stress')
            self.assertAlmostEqual(np.log(actual['lambda_out'][0]),root,delta=2e-9)

    def test_condensed_force_tangent_and_frame_invariance(self):
        q=elastic_response(self.F,self.material,kinematics='plane_stress')
        eps=1e-6
        for i in range(4):
            D=np.eye(4)[i].reshape(2,2)
            a=elastic_response(self.F+eps*D,self.material,kinematics='plane_stress',tangent=False)
            b=elastic_response(self.F-eps*D,self.material,kinematics='plane_stress',tangent=False)
            np.testing.assert_allclose((a['energy']-b['energy'])/(2*eps),
                q['P'].reshape(-1,4)[:,i],rtol=3e-6,atol=2e-6)
            np.testing.assert_allclose(((a['P']-b['P'])/(2*eps)).reshape(-1,4),
                q['tangent'][:,:,i],rtol=3e-6,atol=1e-4)
        R=np.array([[.6,-.8],[.8,.6]])
        rotated=elastic_response(R@self.F,self.material,kinematics='plane_stress')
        np.testing.assert_allclose(q['energy'],rotated['energy'],rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(R@q['P'],rotated['P'],rtol=1e-10,atol=1e-9)


if __name__=='__main__':unittest.main()
