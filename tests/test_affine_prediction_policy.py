"""Actual predictor identity controls only accelerated stagnation handling."""
import unittest
from types import SimpleNamespace
import numpy as np
from tools.replay_affine_policy import ActualAffinePolicy


class FakeBase:
    def _loading_predictor(self,x,old,depth):
        value=x.copy();value[:self.n,1]-=(depth-old)*self.mesh.local_sn[:,1]/self.config.height_mm
        value[self.n:,1]-=depth-old
        if self.use_secant:value[0,0]+=1e-12
        return value
    def solve_friction(self,x,depth,deadline):
        self.observed=(self.stagnation_window,deadline,self.max_iterations)
        if self.fail:raise RuntimeError("original failure")
        return x,7


class Host(ActualAffinePolicy,FakeBase):
    def __init__(self):
        self.n=2;self.mesh=SimpleNamespace(local_sn=np.array([[0.,1.],[1.,.5]]))
        self.config=SimpleNamespace(height_mm=2.);self.predictor="secant"
        self.stagnation_window=40;self.max_iterations=500;self.use_secant=False;self.fail=False


class TestPolicy(unittest.TestCase):
    def test_exact_affine_uses_full_original_budget(self):
        p=Host();x=np.zeros((3,2));value=p._loading_predictor(x,.2,.21)
        self.assertTrue(p._actual_affine_selected)
        p.solve_friction(value,.21,123.)
        self.assertEqual(p.observed,(0,123.,500));self.assertEqual(p.stagnation_window,40)
        np.testing.assert_array_equal(x,np.zeros((3,2)))
    def test_small_but_nonzero_secant_retains_guard(self):
        p=Host();p.use_secant=True;v=p._loading_predictor(np.zeros((3,2)),.2,.21)
        self.assertFalse(p._actual_affine_selected);p.solve_friction(v,.21,123.)
        self.assertEqual(p.observed,(40,123.,500))
    def test_guard_restored_after_original_failure(self):
        p=Host();p.fail=True;v=p._loading_predictor(np.zeros((3,2)),.2,.21)
        with self.assertRaisesRegex(RuntimeError,"original failure"):p.solve_friction(v,.21,123.)
        self.assertEqual(p.stagnation_window,40)
    def test_stale_flag_cleared_on_next_choice(self):
        p=Host();p._loading_predictor(np.zeros((3,2)),.2,.21);p.use_secant=True
        p._loading_predictor(np.zeros((3,2)),.21,.22)
        self.assertFalse(p._actual_affine_selected)
    def test_explicit_affine_policy_unchanged(self):
        p=Host();p.predictor="affine";p.stagnation_window=0
        v=p._loading_predictor(np.zeros((3,2)),.2,.21);p.solve_friction(v,.21,123.)
        self.assertEqual(p.observed,(0,123.,500));self.assertFalse(p._actual_affine_selected)


if __name__=="__main__":unittest.main()
