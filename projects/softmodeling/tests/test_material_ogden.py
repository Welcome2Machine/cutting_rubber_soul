"""Independent checks of the optional exact alpha=1 plane-strain kernel."""
import unittest
from unittest.mock import patch
import numpy as np

from mechanics.finite_strain import Material, elastic_response, _spectral_elastic_response


def rotations(angles):
    result=np.zeros((len(angles),2,2))
    result[:,0,0]=result[:,1,1]=np.cos(angles)
    result[:,1,0]=np.sin(angles);result[:,0,1]=-np.sin(angles)
    return result


class OgdenOneFastPathTests(unittest.TestCase):
    def setUp(self):
        self.G=.47
        # The stored Ogden coefficient is 2*G/alpha, not G for alpha=1.
        self.material=Material((2*self.G,),(1.,),poisson=.49)
        rng=np.random.default_rng(7311)
        angles=rng.uniform(-np.pi,np.pi,(2,128))
        stretches=np.zeros((128,2,2))
        stretches[:,0,0],stretches[:,1,1]=np.exp(rng.uniform(-1.8,1.8,(2,128)))
        random=rotations(angles[0])@stretches@rotations(angles[1])
        # Exactly and nearly repeated stretches, rotations, shear, and large
        # isochoric stretch cover spectral-limit and finite-strain behavior.
        special=np.array([np.eye(2),.7*np.eye(2),1.3*np.eye(2),
            [[1+1e-10,0],[0,1-1e-10]],[[1.,6.],[0.,1.]],[[6.,0.],[0.,1/6]]])
        self.F=np.concatenate((random,special,rotations(np.array([.4,-1.7]))))

    def test_full_response_matches_independent_spectral_kernel(self):
        for tangent in (False,True):
            actual=elastic_response(self.F,self.material,tangent=tangent)
            expected=_spectral_elastic_response(self.F,self.material,tangent=tangent)
            self.assertEqual(actual.keys(),expected.keys())
            for key in actual:
                with self.subTest(tangent=tangent,field=key):
                    np.testing.assert_allclose(actual[key],expected[key],rtol=2e-10,atol=2e-10)

    def test_assembly_only_is_identical_and_avoids_spectral_work(self):
        for tangent in (False,True):
            full=elastic_response(self.F,self.material,tangent=tangent)
            with patch('numpy.linalg.eigh',side_effect=AssertionError('unexpected spectral work')), \
                 patch('numpy.linalg.eigvalsh',side_effect=AssertionError('unexpected spectral work')):
                fast=elastic_response(self.F,self.material,tangent=tangent,diagnostics=False)
            expected={'energy','P','J'}|({'tangent'} if tangent else set())
            self.assertEqual(set(fast),expected)
            for key in expected:np.testing.assert_array_equal(fast[key],full[key])

    def test_energy_gradient_and_force_tangent_are_consistent(self):
        q=elastic_response(self.F,self.material,diagnostics=False)
        epsilon=2e-6
        for i in range(4):
            direction=np.eye(4)[i].reshape(2,2)
            plus=elastic_response(self.F+epsilon*direction,self.material,tangent=False,diagnostics=False)
            minus=elastic_response(self.F-epsilon*direction,self.material,tangent=False,diagnostics=False)
            with self.subTest(component=i):
                np.testing.assert_allclose((plus['energy']-minus['energy'])/(2*epsilon),
                    q['P'].reshape(-1,4)[:,i],rtol=3e-7,atol=2e-6)
                np.testing.assert_allclose(((plus['P']-minus['P'])/(2*epsilon)).reshape(-1,4),
                    q['tangent'][:,:,i],rtol=3e-7,atol=2e-6)
        np.testing.assert_allclose(q['tangent'],q['tangent'].transpose(0,2,1),rtol=2e-13,atol=2e-12)

    def test_objectivity_including_tangent(self):
        R=rotations(np.array([.73]))[0]
        q=elastic_response(self.F,self.material)
        rotated=elastic_response(R@self.F,self.material)
        np.testing.assert_allclose(rotated['energy'],q['energy'],rtol=2e-12,atol=1e-11)
        np.testing.assert_allclose(rotated['P'],R@q['P'],rtol=2e-11,atol=2e-10)
        expected=np.einsum('ia,eaJbL,kb->eiJkL',R,q['tangent'].reshape(-1,2,2,2,2),R)
        np.testing.assert_allclose(rotated['tangent'],expected.reshape(-1,4,4),rtol=3e-11,atol=3e-10)
        rigid=elastic_response(rotations(np.linspace(-3,3,11)),self.material)
        np.testing.assert_allclose(rigid['energy'],0.,atol=3e-15)
        np.testing.assert_allclose(rigid['P'],0.,atol=2e-14)

    def test_analytical_simple_shear_and_small_strain_modulus(self):
        gamma=np.array([0.,1e-7,.1,.5,2.,6.])
        F=np.tile(np.eye(2),(len(gamma),1,1));F[:,0,1]=gamma
        q=elastic_response(F,self.material)
        np.testing.assert_allclose(q['P'][:,0,1],2*self.G*gamma/np.sqrt(4+gamma**2),rtol=2e-12,atol=1e-14)
        np.testing.assert_allclose(q['energy'],2*self.G*(np.sqrt(4+gamma**2)-2),rtol=2e-11,atol=1e-15)
        self.assertAlmostEqual(q['P'][1,0,1]/gamma[1],self.G,places=12)
        self.assertAlmostEqual(q['tangent'][0,1,1],self.G,places=13)

    def test_analytical_isochoric_plane_strain_stretch(self):
        stretches=np.array([.2,.5,1.,1.5,3.,6.])
        F=np.zeros((len(stretches),2,2))
        F[:,0,0]=stretches;F[:,1,1]=1/stretches
        q=elastic_response(F,self.material)
        stress_difference=q['cauchy'][:,0,0]-q['cauchy'][:,1,1]
        np.testing.assert_allclose(stress_difference,2*self.G*(stretches-1/stretches),rtol=2e-13,atol=1e-13)

    def test_other_laws_keep_spectral_reference(self):
        for material,kinematics in ((self.material,'plane_stress'),
            (Material((2*self.G,),(1.,),volumetric='paper2023'),'plane_strain'),
            (Material((2*self.G/1.5,),(1.5,)),'plane_strain')):
            # Moderate deformation avoids unrelated extreme plane-stress
            # convergence limits while testing dispatch exactly.
            F=np.array([np.eye(2),[[1.2,.1],[.05,.9]]])
            actual=elastic_response(F,material,kinematics=kinematics)
            expected=_spectral_elastic_response(F,material,kinematics=kinematics)
            for key in actual:np.testing.assert_array_equal(actual[key],expected[key])

    def test_invalid_deformations_are_rejected(self):
        for F in (np.eye(2),np.full((1,2,2),np.nan)):
            with self.assertRaises(ValueError):elastic_response(F,self.material)
        for F in (np.zeros((1,2,2)),np.array([[[1.,0.],[0.,-1.]]])):
            with self.assertRaises(FloatingPointError):elastic_response(F,self.material)


if __name__=='__main__':unittest.main()
