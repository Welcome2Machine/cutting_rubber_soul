"""Checks of the optional native IPC path, including independent mechanics."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.ipc_reference import (
    IPCProblem,
    determinant_step,
    toolkit,
    parse_args,
    run,
    BASELINE_PATH,
)
from prototype.material import response


class DeterminantPathTests(unittest.TestCase):
    def test_positive_endpoints_do_not_hide_inversion(self):
        F = np.eye(2)[None]
        D = np.diag([-4.0, -2.0])[None]
        self.assertGreater(np.linalg.det(F + D)[0], 0)
        self.assertAlmostEqual(determinant_step(F, D), 0.9 / 4)

    def test_linear_double_root_and_no_root(self):
        F = np.eye(2)[None]
        for D, expected in ((np.diag([-2.0, 0.0]), 0.45), (-2 * np.eye(2), 0.45), (np.eye(2), 1.0)):
            self.assertAlmostEqual(determinant_step(F, D[None]), expected)


class BaselineCommandTests(unittest.TestCase):
    def test_frozen_baseline_rejects_changed_physics_and_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            arguments = ["--baseline", "--output", folder]
            args = parse_args(arguments)
            self.assertEqual(
                (args.depth, args.increment, args.friction, args.max_iterations),
                (3.0, 0.025, 0.0, 200),
            )
            for override in (
                ["--friction", ".35"],
                ["--increment", ".00625"],
                ["--case", "patch"],
                ["--depth", ".4"],
            ):
                with self.subTest(override=override), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(arguments + override)
            # The same CLI still permits explicitly labelled diagnostic variants.
            diagnostic = parse_args(["--output", folder, "--depth", ".4", "--friction", ".35"])
            self.assertFalse(diagnostic.baseline)
            self.assertEqual(diagnostic.friction, 0.35)


class NativeIPCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ipc = toolkit()
        except ModuleNotFoundError:
            raise unittest.SkipTest("Optional ipctk==1.6.0 not installed")
        cls.ipc.set_num_threads(4)

    def test_compression_force_geometry_and_material(self):
        p = IPCProblem("patch")
        x = p.rest.copy()
        old = p.start
        with contextlib.redirect_stdout(io.StringIO()), threadpool_limits(limits=4):
            for depth in (0.0, 0.025, 0.05, 0.075, 0.1):
                x, info = p.step(x, old, depth, time.monotonic() + 15)
                old = depth
        _, g, _, minj = p.evaluate(x, tangent=False)
        forces = g.reshape(-1, 2)
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-9)
        self.assertAlmostEqual(
            forces[p.mesh.bottom_nodes, 1].sum(), info["reaction_fz_n"], delta=1e-3
        )
        top = np.flatnonzero(np.isclose(p.mesh.reference_xy[:, 1], 1.0))
        stretch = x[top, 1].mean()
        analytic = (
            -response(np.array([[[1.0, 0.0], [0.0, stretch]]]), p.config, tangent=False)["P"][
                0, 1, 1
            ]
            * p.config.width_mm
        )
        self.assertLess(abs(info["reaction_fz_n"] / analytic - 1), 0.002)
        self.assertAlmostEqual(info["minimum_distance_mm"], 0.9 - x[top, 1].max(), places=10)
        self.assertGreater(minj, 0)
        self.assertFalse(p.ipc.has_intersections(p.cmesh, x[p.surface]))

    def test_contact_and_friction_energy_gradient_and_thickness(self):
        p = IPCProblem("patch", mu=0.35)
        x = p.rest.copy()
        x[p.n :, 1] -= 2 * p.dhat - 0.006
        p.previous = x.copy()
        x[p.n :, 0] += 0.002
        lag = p.lag(x)
        rng = np.random.default_rng(918)
        v = rng.normal(size=x.shape)
        v /= np.linalg.norm(v)
        eps = 1e-7
        E, g, _, _ = p.evaluate(x, lag, tangent=False)
        fd = (
            p.evaluate(x + eps * v, lag, tangent=False)[0]
            - p.evaluate(x - eps * v, lag, tangent=False)[0]
        ) / (2 * eps)
        self.assertAlmostEqual(fd, float(g @ v.ravel()), delta=1e-5 * max(1.0, abs(fd)))
        from dataclasses import replace

        p.config = replace(p.config, width_mm=40.0)
        p.barrier = p.ipc.BarrierPotential(p.dhat, 6.6 * p.config.width_mm, True)
        E2, g2, _, _ = p.evaluate(x, p.lag(x), tangent=False)
        self.assertAlmostEqual(E2, 2 * E, places=9)
        np.testing.assert_allclose(g2, 2 * g, rtol=1e-10, atol=1e-9)

    def test_ccd_limits_crossing_trajectory(self):
        p = IPCProblem("patch")
        x = p.rest.copy()
        v = np.zeros_like(x)
        v[p.n :, 1] = -0.1
        alpha = p.safe_step(x, v)
        self.assertGreater(alpha, 0)
        self.assertLess(alpha, 0.21)
        self.assertFalse(p.ipc.has_intersections(p.cmesh, (x + alpha * v)[p.surface]))

    def test_runner_records_acceptance_and_failure_without_false_success(self):
        for iterations, expected in ((200, True), (1, False)):
            with self.subTest(iterations=iterations), tempfile.TemporaryDirectory() as folder:
                args = parse_args(
                    [
                        "--case",
                        "patch",
                        "--output",
                        folder,
                        "--depth",
                        ".025",
                        "--seconds",
                        "15",
                        "--max-iterations",
                        str(iterations),
                    ]
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run(args), expected)
                out = Path(folder)
                result = json.loads((out / "result.json").read_text())
                self.assertEqual(result["completed"], expected)
                self.assertIsNone(result["baseline_id"])
                self.assertEqual(
                    (out / "baseline_config.json").read_bytes(), BASELINE_PATH.read_bytes()
                )
                if expected:
                    self.assertEqual(result["last_depth_mm"], 0.025)
                    self.assertTrue((out / result["last_checkpoint"]).is_file())
                else:
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["steps"], [])
                    self.assertTrue((out / "failed_trial.npz").is_file())
