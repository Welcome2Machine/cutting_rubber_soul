import unittest
import numpy as np
from scipy.sparse import csc_matrix
from prototype.residual_trust import least_squares_direction, reduction_test


class ResidualTrustTests(unittest.TestCase):
    def test_nonnormal_direction_is_descent_and_matches_dense_regularization(self):
        A = csc_matrix([[1., 30.], [0., .1]])
        r = np.array([1., -2.]); damp = .03
        du, _ = least_squares_direction(A, r, damp)
        scale = 1/np.linalg.norm(A.toarray(), axis=0)
        B = A.toarray()*scale
        expected = scale*np.linalg.solve(B.T@B+damp**2*np.eye(2), -B.T@r)
        np.testing.assert_allclose(du, expected, rtol=1e-7)
        self.assertLess(r@(A@du), 0)
        self.assertTrue(reduction_test(r, r+A@du, A@du, 1)[0])

    def test_accurate_linear_model_cannot_accept_residual_increase(self):
        r = np.array([1., 0.]); action = np.array([.01, 0.])
        self.assertFalse(reduction_test(r, r+action, action, 1)[0])

    def test_small_model_error_does_not_override_absent_progress(self):
        r = np.array([1., 0.]); action = np.array([-.01, 0.])
        self.assertFalse(reduction_test(r, r, action, 1e-4)[0])

    def test_inconsistent_least_squares_is_not_zero_force(self):
        A = csc_matrix([[1., 0.], [0., 0.]])
        r = np.array([0., 1.])
        du, info = least_squares_direction(A, r, .01)
        self.assertEqual(info["scaled_gradient_norm"], 0)
        self.assertEqual(info["linear_model_residual_n"], 1)
        self.assertFalse(reduction_test(r, r+A@du, A@du, 1)[0])
