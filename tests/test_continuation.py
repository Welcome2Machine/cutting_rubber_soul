"""Independent smooth and nonsmooth turning paths; no cutting validation."""
import unittest
import numpy as np
from prototype.continuation import follow_path,PathFailure


class ContinuationTests(unittest.TestCase):
    def test_cubic_crosses_two_folds_and_corrects_original_target(self):
        def equation(z,tangent):
            x,a=z
            return np.array([x**3-x-a]),np.array([[3*x*x-1]]),np.array([-1.])
        points=[]
        initial=np.array([-1.,0.])
        root,_=follow_path(equation,initial,step=.12,tolerance=1e-10,observer=points.append)
        self.assertAlmostEqual(root[1],1.,places=12)
        self.assertAlmostEqual(root[0]**3-root[0],1.,places=9)
        slopes=[p['tangent_parameter'] for p in points if p['event']=='point']
        self.assertTrue(any(s<0 for s in slopes))
        self.assertGreater(slopes[-1],0.)
        np.testing.assert_array_equal(initial,[-1.,0.])

    def test_piecewise_linear_turning_path(self):
        def equation(z,tangent):
            x,a=z
            value,derivative=(x+1,1) if x<=0 else ((1-x,-1) if x<2 else (x-3,1))
            return np.array([.4*value-a]),np.array([[.4*derivative]]),np.array([-1.])
        points=[]
        root,_=follow_path(equation,[-1.,0.],step=.12,tolerance=1e-10,observer=points.append)
        np.testing.assert_allclose(root,[5.5,1.],atol=1e-9)
        self.assertTrue(any(p.get('tangent_parameter',0)<0 for p in points))

    def test_budget_and_auxiliary_root_cannot_override_external_acceptance(self):
        def equation(z,tangent):
            return np.array([z[0]-z[1]]),np.array([[1.]]),np.array([-1.])
        with self.assertRaises(PathFailure) as caught:
            follow_path(equation,[0.,0.],max_steps=1)
        self.assertTrue(caught.exception.budget)
        with self.assertRaisesRegex(PathFailure,'original acceptance failed'):
            follow_path(equation,[0.,0.],accept=lambda point:None)

    def test_invalid_start_is_rejected(self):
        def equation(z,tangent):return np.array([1.]),np.array([[1.]]),np.array([-1.])
        with self.assertRaisesRegex(ValueError,'start does not satisfy'):
            follow_path(equation,[0.,0.])

    def test_resume_preserves_orientation_on_returning_branch(self):
        def equation(z,tangent):
            x,a=z
            return np.array([x**3-x-a]),np.array([[3*x*x-1]]),np.array([-1.])
        with self.assertRaises(PathFailure) as stopped:
            follow_path(equation,[-1.,0.],step=.12,tolerance=1e-10,max_steps=8)
        saved=stopped.exception
        self.assertLess(saved.tangent[-1],0.)
        root,_=follow_path(equation,saved.point,initial_tangent=saved.tangent,tolerance=1e-10)
        self.assertAlmostEqual(root[1],1.,places=12)
        self.assertAlmostEqual(root[0]**3-root[0],1.,places=9)

    def test_sharp_corners_use_outgoing_branch_instead_of_tangent_alignment(self):
        def equation(z,tangent):
            x,a=z
            value,slope=(x,1.) if x<=0 else ((-100*x,-100.) if x<.01 else (x-1.01,1.))
            return np.array([value-a]),np.array([[slope]]),np.array([-1.])
        def event(a,b):
            for boundary,slope in ((0.,-100.),(.01,1.)):
                if a[0]<boundary-1e-12 and b[0]>=boundary:
                    return dict(fraction=(boundary-a[0])/(b[0]-a[0]),direction=1,label=str(boundary),
                        equation=lambda z,cut=boundary:(z[0]-cut,np.array([1.,0.])),
                        jacobian=lambda z,j,derivative=slope:np.array([[derivative]]))
        trace=[]
        root,_=follow_path(equation,[-.01,-.01],target=.1,step=.05,tolerance=1e-10,event=event,observer=trace.append)
        np.testing.assert_allclose(root,[1.11,.1],atol=1e-9)
        corners=[p for p in trace if p['event']=='switch']
        self.assertEqual(len(corners),2)
        self.assertTrue(all(p['incoming_alignment']<0 for p in corners))
