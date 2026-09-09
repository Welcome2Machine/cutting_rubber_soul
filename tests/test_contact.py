import unittest
import numpy as np
from prototype.contact import coulomb_return


class CoulombTests(unittest.TestCase):
    def test_fixed_points_enforce_coulomb_independently_of_augmentation(self):
        for augmentation in (50.,250.,10000.):
            for slip,shear in ((0.,.2),(0.,-.2),(.001,.7),(-.001,-.7)):
                projected=coulomb_return(np.array([slip]),np.array([shear]),np.array([2.]),.35,augmentation)[0]
                np.testing.assert_allclose(projected,shear)
                self.assertGreaterEqual(shear*slip,0.)
    def test_stick_slip_reversal_and_opening(self):
        k,mu,p = 1000.,.35,np.array([2.])
        old = np.zeros(1)
        for increment,expected in [(0.0001,.1),(.001,.7),(-.0001,.6),(-.002,-.7)]:
            shear,ds,dp,diss,stick = coulomb_return(np.array([increment]),old,p,mu,k)
            self.assertAlmostEqual(float(shear[0]),expected)
            self.assertGreaterEqual(float(diss[0]),-1e-14)
            self.assertLessEqual(abs(float(shear[0])),mu*p[0])
            old = shear
        opened = coulomb_return(np.array([0.]),old,np.array([0.]),mu,k)
        np.testing.assert_array_equal(opened[0],0)

    def test_elastic_stick_preserves_static_traction(self):
        shear,ds,dp,diss,stick = coulomb_return(np.array([0.]),np.array([.2]),np.array([2.]),.35,1000.)
        np.testing.assert_allclose(shear,.2)
        np.testing.assert_allclose(diss,0)
        self.assertTrue(stick[0])

    def test_sliding_dissipation_and_consistent_derivatives(self):
        for slip in [1e-5,1e-2,-1e-2]:
            args=(np.array([slip]),np.array([.05]),np.array([2.]),.35,1000.)
            tau,ds,dp,diss,_ = coulomb_return(*args)
            eps=1e-8
            ds_fd=(coulomb_return(args[0]+eps,*args[1:])[0]-coulomb_return(args[0]-eps,*args[1:])[0])/(2*eps)
            dp_fd=(coulomb_return(args[0],args[1],args[2]+eps,*args[3:])[0]-coulomb_return(args[0],args[1],args[2]-eps,*args[3:])[0])/(2*eps)
            np.testing.assert_allclose(ds,ds_fd,atol=1e-6)
            np.testing.assert_allclose(dp,dp_fd,atol=1e-6)
            self.assertGreaterEqual(diss[0],-1e-12)

    def test_frictionless_and_invalid_stiffness(self):
        np.testing.assert_array_equal(coulomb_return(np.ones(2),np.zeros(2),np.ones(2),0.,1000.)[0],0)
        with self.assertRaises(ValueError):
            coulomb_return(np.zeros(1),np.zeros(1),np.ones(1),.35,0.)
