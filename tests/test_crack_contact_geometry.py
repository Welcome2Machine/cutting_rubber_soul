"""Independent differential and invariance checks for oriented crack contact."""

import unittest

import numpy as np

from prototype.crack_contact_geometry import nearest_segment, signed_gap_derivatives


class SignedGapGeometryTests(unittest.TestCase):
    @staticmethod
    def evaluate(coordinates):
        return signed_gap_derivatives(*np.asarray(coordinates).reshape(3, 2))

    @staticmethod
    def direct_gap(coordinates):
        # Independent scalar expression: signed triangle area / edge length.
        point, a, b = np.asarray(coordinates).reshape(3, 2)
        e, r = b - a, point - a
        return (e[0] * r[1] - e[1] * r[0]) / np.linalg.norm(e)

    def test_gap_gradient_and_hessian_against_finite_differences(self):
        rng = np.random.default_rng(49016)
        for scale in (0.01, 1.0, 100.0):
            for _ in range(8):
                x = scale * rng.normal(size=(3, 2))
                if np.linalg.norm(x[2] - x[1]) < 0.2 * scale:
                    x[2, 0] += scale
                x = x.ravel()
                gap, gradient, hessian = self.evaluate(x)
                step = 1e-5 * scale
                directions = step * np.eye(6)
                gradient_fd = np.array([
                    (self.direct_gap(x + dx) - self.direct_gap(x - dx))
                    / (2.0 * step) for dx in directions
                ])
                hessian_fd = np.column_stack([
                    (self.evaluate(x + dx)[1] - self.evaluate(x - dx)[1])
                    / (2.0 * step) for dx in directions
                ])
                with self.subTest(scale=scale, gap=gap):
                    self.assertAlmostEqual(gap / scale, self.direct_gap(x) / scale)
                    np.testing.assert_allclose(gradient, gradient_fd, rtol=2e-7, atol=2e-8)
                    np.testing.assert_allclose(scale * hessian, scale * hessian_fd,
                                               rtol=2e-7, atol=2e-8)
                    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=1e-12 / scale)

    def test_zero_gap_including_coincident_endpoints_is_regular(self):
        for parameter in (0.0, 0.37, 1.0):
            x = np.array([[parameter, 0.0], [0.0, 0.0], [1.0, 0.0]])
            gap, gradient, hessian = self.evaluate(x)
            self.assertEqual(gap, 0.0)
            np.testing.assert_array_equal(gradient.reshape(3, 2)[:, 0], np.zeros(3))
            np.testing.assert_allclose(gradient.reshape(3, 2)[:, 1],
                                       [1.0, -(1.0 - parameter), -parameter])
            self.assertTrue(np.all(np.isfinite(hessian)))
            # The zero gap is smooth, rather than abs(g) or a gap barrier.
            for offset in (-1e-8, 1e-8):
                shifted = x.copy()
                shifted[0, 1] += offset
                self.assertEqual(self.evaluate(shifted)[0], offset)

    def test_reversing_edge_reverses_gap_and_derivatives(self):
        x = np.array([[0.4, -0.8], [-0.3, 0.2], [1.2, 0.7]])
        gap, gradient, hessian = self.evaluate(x)
        reverse = np.array([0, 1, 4, 5, 2, 3])
        reversed_gap, reversed_gradient, reversed_hessian = self.evaluate(x[[0, 2, 1]])
        self.assertAlmostEqual(reversed_gap, -gap)
        np.testing.assert_allclose(reversed_gradient, -gradient[reverse], atol=1e-14)
        np.testing.assert_allclose(reversed_hessian, -hessian[np.ix_(reverse, reverse)], atol=1e-14)

    def test_rigid_motion_objectivity_and_force_torque_balance(self):
        x = np.array([[0.4, -0.8], [-0.3, 0.2], [1.2, 0.7]])
        gap, gradient, hessian = self.evaluate(x)
        forces = 3.7 * gradient.reshape(3, 2)
        np.testing.assert_allclose(forces.sum(axis=0), np.zeros(2), atol=1e-14)
        torque = np.sum(x[:, 0] * forces[:, 1] - x[:, 1] * forces[:, 0])
        self.assertAlmostEqual(torque, 0.0, places=13)
        translations = np.tile(np.eye(2), (3, 1))
        np.testing.assert_allclose(hessian @ translations, 0.0, atol=1e-14)
        for angle in (0.3, np.pi / 2, 2.7):
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            transform = np.kron(np.eye(3), rotation)
            moved = x @ rotation.T + [2.3, -4.1]
            moved_gap, moved_gradient, moved_hessian = self.evaluate(moved)
            self.assertAlmostEqual(moved_gap, gap, places=13)
            np.testing.assert_allclose(moved_gradient, transform @ gradient, atol=1e-14)
            np.testing.assert_allclose(moved_hessian, transform @ hessian @ transform.T, atol=1e-14)

    def test_rotational_derivative_identity(self):
        x = np.array([[0.4, -0.8], [-0.3, 0.2], [1.2, 0.7]])
        _, gradient, hessian = self.evaluate(x)
        generator = np.kron(np.eye(3), [[0.0, -1.0], [1.0, 0.0]])
        # Unlike translation, infinitesimal rotation is not generally a null
        # vector of the Hessian away from equilibrium.
        np.testing.assert_allclose(hessian @ (generator @ x.ravel()),
                                   generator @ gradient, atol=1e-14)

    def test_outside_projection_is_supporting_line_gap(self):
        gap, gradient, _ = signed_gap_derivatives([2.0, 0.5], [0.0, 0.0], [1.0, 0.0])
        self.assertEqual(gap, 0.5)
        np.testing.assert_array_equal(gradient[:2], [0.0, 1.0])

    def test_degenerate_and_invalid_geometry_is_rejected(self):
        for point, a, b in (([0, 0], [1, 1], [1, 1]),
                            ([np.nan, 0], [0, 0], [1, 0]),
                            ([0, 0, 0], [0, 0], [1, 0])):
            with self.assertRaises(ValueError):
                signed_gap_derivatives(point, a, b)


class NearestSegmentTests(unittest.TestCase):
    def test_selects_nearest_finite_segment_not_infinite_line(self):
        vertices = [[10, 0], [11, 0], [0, 0.2], [1, 0.2]]
        self.assertEqual(nearest_segment([0.5, 0], vertices, [[0, 1], [2, 3]]), (1, 0.5))

    def test_clamps_both_endpoints(self):
        for point, expected in (([-2, 0.1], 0.0), ([2, -0.1], 1.0), ([0.4, 0.2], 0.4)):
            self.assertEqual(nearest_segment(point, [[0, 0], [1, 0]], [[0, 1]]), (0, expected))

    def test_negative_signed_gap_does_not_bias_geometric_nearest(self):
        vertices = [[0, -0.1], [1, -0.1], [0, 10], [1, 10]]
        self.assertEqual(nearest_segment([0.4, 0], vertices, [[0, 1], [2, 3]]), (0, 0.4))

    def test_ties_choose_first_candidate_and_orientation_changes_parameter(self):
        vertices = [[0, 0], [1, 0]]
        self.assertEqual(nearest_segment([0.25, 0.4], vertices, [[1, 0], [0, 1]]), (0, 0.75))

    def test_nearest_is_objective(self):
        vertices = np.array([[0, 0], [1, 0], [2, 1], [3, 1]])
        point = np.array([2.4, 1.1])
        edges = [[0, 1], [2, 3]]
        rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
        original = nearest_segment(point, vertices, edges)
        moved = nearest_segment(point @ rotation.T + [4, 7],
                                vertices @ rotation.T + [4, 7], edges)
        self.assertEqual(moved[0], original[0])
        self.assertAlmostEqual(moved[1], original[1], places=14)

    def test_invalid_candidates_are_rejected(self):
        for vertices, edges in (([[0, 0], [0, 0]], [[0, 1]]),
                                ([[0, 0], [1, 0]], []),
                                ([[0, 0], [1, 0]], [[0.0, 1.0]]),
                                ([[0, 0], [1, 0]], [[0, 2]]),
                                ([[0, 0], [1, 0]], [[-1, 1]]),
                                ([[np.inf, 0], [1, 0]], [[0, 1]])):
            with self.assertRaises(ValueError):
                nearest_segment([0.5, 0], vertices, edges)


if __name__ == "__main__":
    unittest.main()
