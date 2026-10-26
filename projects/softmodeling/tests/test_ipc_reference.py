"""Checks of the optional native IPC path, including independent mechanics."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.ipc_reference import (
    IPCProblem,
    determinant_step,
    toolkit,
    parse_args,
    run,
    BASELINE_PATH,
    export_visualization,
    regularize_curvature,
    restore_accepted_prefix,
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
    def test_unlimited_wall_budget_is_explicit_and_negative_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(parse_args(['--output',folder,'--seconds','0']).seconds, 0)
            args = parse_args(['--output',folder,'--seconds','0','--friction','.35',
                               '--resume',folder])
            self.assertIsNotNone(args.resume)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--output',folder,'--seconds','-1'])

    def test_elastic_reference_rejects_changed_physics_and_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            arguments = ["--baseline", "--output", folder]
            args = parse_args(arguments)
            self.assertEqual(
                (args.depth, args.increment, args.friction, args.max_iterations),
                (2.0, 0.025, 0.0, 200),
            )
            for override in (
                ["--friction", ".35"],
                ["--increment", ".00625"],
                ["--case", "patch"],
                ["--depth", ".4"],
                ["--local-spacing", ".1"],
                ["--surface-spacing", ".05"],
                ["--nose-facets", "64"],
                ["--inclination", "30"],
                ["--tip-radius", ".02"],
                ["--fine-halfwidth", ".5"],
                ["--hessian-mode", "hybrid"],
            ):
                with self.subTest(override=override), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(arguments + override)
            # The same CLI still permits explicitly labelled diagnostic variants.
            diagnostic = parse_args(["--output", folder, "--depth", ".4", "--friction", ".35"])
            self.assertFalse(diagnostic.baseline)
            self.assertEqual(diagnostic.friction, 0.35)

    def test_invalid_mesh_and_tip_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for extra in (["--local-spacing", "nan"], ["--surface-spacing", "0"],
                          ["--nose-facets", "63"], ["--inclination", "90"],
                          ["--tip-radius", "0"], ["--tip-radius", ".2"],
                          ["--fine-halfwidth", "-1"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(["--output", folder] + extra)

    def test_export_failure_does_not_relabel_completed_solve(self):
        with tempfile.TemporaryDirectory() as folder:
            report = dict(completed=True, status="completed")
            with patch("prototype.ipc_reference.subprocess.run", side_effect=OSError("no encoder")):
                export_visualization(Path(folder), report)
            saved = json.loads((Path(folder) / "result.json").read_text())
            self.assertTrue(saved["completed"])
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["visualization"]["status"], "failed")


class NativeIPCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ipc = toolkit()
        except ModuleNotFoundError:
            raise unittest.SkipTest("Optional ipctk==1.6.0 not installed")
        cls.ipc.set_num_threads(4)

    def test_roundoff_refinement_keeps_force_and_energy_safeguards(self):
        p = IPCProblem('patch', hessian_mode='regularized')
        x = p.rest.copy()
        trial = x.copy()
        trial.ravel()[p.free[0]] += 1e-9
        g = np.zeros(x.size)
        g[p.free[0]] = -1.5e-4
        gt = np.zeros(x.size)
        midpoint = (g+gt)/2
        with patch.object(p, 'evaluate', return_value=(1., midpoint, None, 1.)), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(p.roundoff_refinement(x, trial, 1., 1.+1e-14, g, gt))
            # Significant energy growth, unbalanced force, a no-op, and a large
            # step must never be excused as floating-point cancellation.
            self.assertFalse(p.roundoff_refinement(x, trial, 1., 1.+1e-5, g, gt))
            self.assertFalse(p.roundoff_refinement(x, trial, 1., 1., g, g))
            self.assertFalse(p.roundoff_refinement(x, x, 1., 1., g, gt))
            self.assertFalse(p.roundoff_refinement(x, x+100*(trial-x), 1., 1., g, gt))
        with patch.object(p, 'evaluate', return_value=(1., -g, None, 1.)):
            self.assertFalse(p.roundoff_refinement(x, trial, 1., 1., g, gt))


    def test_global_curvature_shift_preserves_well_conditioned_directions(self):
        from scipy.sparse import diags
        from scipy.linalg import eigvalsh
        metric = diags([2., 3., 4.]).tocsc()
        A = diags([-2., 6., 12.]).tocsc()
        corrected, smallest, shift = regularize_curvature(A, metric)
        self.assertAlmostEqual(smallest, -1., places=10)
        self.assertAlmostEqual(shift, 1.05, places=10)
        self.assertAlmostEqual(eigvalsh(corrected.toarray(), metric.toarray()).min(), .05, places=10)
        unchanged, _, shift = regularize_curvature(metric, metric)
        self.assertEqual(shift, 0.)
        np.testing.assert_allclose(unchanged.toarray(), metric.toarray(), atol=1e-12)

    def test_unprojected_hessian_matches_force_derivative(self):
        p = IPCProblem('patch', hessian_mode='hybrid')
        x = p.rest.copy()
        x[:p.n, 1] *= .99
        x[p.n:, 1] -= .01 + 2*p.dhat - .006
        rng = np.random.default_rng(28)
        direction = rng.normal(size=x.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-7
        E, g, H, _ = p.evaluate(x, projected=False)
        projected = p.evaluate(x, projected=True)
        self.assertAlmostEqual(E, projected[0], places=12)
        np.testing.assert_allclose(g, projected[1], rtol=1e-12, atol=1e-12)
        fd = (p.evaluate(x+eps*direction, tangent=False)[1] -
              p.evaluate(x-eps*direction, tangent=False)[1])/(2*eps)
        np.testing.assert_allclose(H @ direction.ravel(), fd, rtol=2e-5, atol=1e-4)

    def test_resume_preserves_accepted_history_and_matches_continuous_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            common = ['--case', 'patch', '--seconds', '30', '--hessian-mode', 'regularized']
            for name, depth, resume in (('prefix', '.025', []),
                                        ('resumed', '.05', ['--resume', str(root/'prefix')]),
                                        ('continuous', '.05', [])):
                args = parse_args(common + ['--output', str(root/name), '--depth', depth] + resume)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(run(args))
            for i in range(3):
                with np.load(root/'resumed/states'/f'{i:04d}.npz') as a, np.load(root/'continuous/states'/f'{i:04d}.npz') as b:
                    np.testing.assert_array_equal(a['positions'], b['positions'])
            args = parse_args(common + ['--output', str(root/'invalid'), '--depth', '.05',
                                         '--resume', str(root/'prefix'), '--dhat', '.02'])
            with self.assertRaisesRegex(ValueError, 'Resume mismatch: dhat_mm'):
                run(args)



    def test_refinement_preserves_geometry_and_passes_short_path(self):
        coarse = IPCProblem()
        fine = IPCProblem(local_spacing=.1, surface_spacing=.05, nose_facets=64)
        self.assertGreater(fine.n, coarse.n)
        self.assertEqual(fine.config, coarse.config)
        self.assertEqual(fine.options.tip_radius_mm, coarse.options.tip_radius_mm)
        # Doubling circular segments retains every original blade vertex exactly.
        np.testing.assert_allclose(fine.rest[fine.n:][2:-2:2], coarse.rest[coarse.n:][2:-2], atol=1e-14)
        np.testing.assert_allclose(fine.rest[fine.n:][[0, 1, -2, -1]],
                                   coarse.rest[coarse.n:][[0, 1, -2, -1]], atol=1e-14)
        x, old = fine.rest.copy(), fine.start
        with contextlib.redirect_stdout(io.StringIO()), threadpool_limits(limits=4):
            deadline = time.monotonic() + 30
            for depth in (0., .025, .05):
                x, info = fine.step(x, old, depth, deadline)
                old = depth
        self.assertLessEqual(info["residual_n"], 1e-4)
        self.assertGreater(info["min_j"], 0)
        self.assertFalse(fine.ipc.has_intersections(fine.cmesh, x[fine.surface]))

    def test_angle_variants_and_configurable_radius(self):
        # Synthetic radius for geometry/solver tests, not a measured blade radius.
        for angle in (60., 30., 0.):
            with self.subTest(angle=angle):
                p = IPCProblem(inclination=angle, tip_radius=.02, nose_facets=64,
                               surface_spacing=.01, fine_halfwidth=.1)
                self.assertEqual(p.config.inclination_deg, angle)
                self.assertEqual(p.options.tip_radius_mm, .02)
                arc = p.rest[p.n:][2:-2]
                center_x = arc[len(arc)//2, 0]
                center_y = arc[len(arc)//2, 1] + .02
                np.testing.assert_allclose(np.linalg.norm(arc-[center_x, center_y], axis=1),
                                           .02, atol=1e-14)
                # Body rotation is clockwise; blade remains vertical in lab coordinates.
                top_edge = p.mesh.reference_xy[np.isclose(p.mesh.local_sn[:, 1], 5.)]
                normal = np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
                np.testing.assert_allclose(top_edge @ normal, 5., atol=1e-12)
                x, old = p.rest.copy(), p.start
                with contextlib.redirect_stdout(io.StringIO()), threadpool_limits(limits=4):
                    deadline = time.monotonic() + 30
                    for depth in (0., .01):
                        x, info = p.step(x, old, depth, deadline)
                        old = depth
                np.testing.assert_allclose(x[p.n:, 0], p.rest[p.n:, 0], atol=1e-14)
                self.assertLessEqual(info['residual_n'], 1e-4)
                self.assertGreater(info['min_j'], 0.)

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

    def test_friction_initial_contact_breaks_outer_two_cycle(self):
        # The production 0-degree geometry previously exhausted 20 outer
        # iterations at depth zero despite converged frozen-friction solves.
        p = IPCProblem(mu=.35, dhat=.0025, inclination=0, tip_radius=.02,
                       local_spacing=.05, surface_spacing=.005,
                       fine_halfwidth=3, nose_facets=64, hessian_mode="regularized")
        log = io.StringIO()
        previous = p.rest.copy()
        with contextlib.redirect_stdout(log), threadpool_limits(limits=2):
            x, info = p.step(previous, p.start, 0., time.monotonic()+180)
        np.testing.assert_array_equal(p.previous, previous)
        # Independent refreshed assembly, not the converged lagged subproblem.
        _, g, _, j = p.evaluate(x, p.lag(x), tangent=False)
        self.assertLessEqual(np.linalg.norm(g[p.free]), 1e-4)
        self.assertGreater(j, 0)
        self.assertFalse(p.ipc.has_intersections(p.cmesh, x[p.surface]))
        self.assertLess(info["friction_outer_iterations"], 20)

    def test_coupled_friction_jacobian_matches_fresh_force_difference(self):
        p = IPCProblem('patch', mu=.35)
        x = p.rest.copy()
        x[p.n:, 1] -= 2*p.dhat-.006
        p.previous = x.copy()
        x[p.n:, 0] += .002
        lag = p.lag(x)
        H = p.evaluate(x, lag, projected=False)[2]
        A = p.coupled_friction_tangent(x, lag, H)
        v = np.random.default_rng(35).normal(size=x.shape)
        v /= np.linalg.norm(v)
        eps = 1e-7
        gp = p.evaluate(x+eps*v, p.lag(x+eps*v), tangent=False)[1]
        gm = p.evaluate(x-eps*v, p.lag(x-eps*v), tangent=False)[1]
        fd = (gp-gm)/(2*eps)
        np.testing.assert_allclose(A@v.ravel(), fd, rtol=2e-5, atol=2e-5)
        # Lagged derivative omits geometry coupling; the check must detect that.
        self.assertGreater(np.linalg.norm(H@v.ravel()-fd), .01*np.linalg.norm(fd))

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


class ReferenceContactGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ipc = toolkit()
        except ModuleNotFoundError:
            raise unittest.SkipTest('Optional ipctk==1.6.0 not installed')
        cls.ipc.set_num_threads(1)

    def geometry(self, rest, edges, formulation, n_dynamic=None):
        p = object.__new__(IPCProblem)
        p.ipc = self.ipc
        p.contact_formulation = formulation
        p.rest = np.asarray(rest, dtype=float)
        p.previous = p.rest.copy()
        p.surface = np.arange(len(rest))
        p.surface_dofs = np.arange(p.rest.size)
        p.dhat = .0025
        p.cmesh = self.ipc.CollisionMesh(p.rest, np.asarray(edges, dtype=int))
        if n_dynamic is not None:
            p.cmesh.can_collide = self.ipc.make_static_obstacle_filter(n_dynamic)
        p.barrier = self.ipc.BarrierPotential(p.dhat, 132., True)
        p.friction = self.ipc.FrictionPotential(1e-5)
        p.config = SimpleNamespace(friction_coefficient=.35)
        return p

    def test_stress_free_reference_check_accepts_separated_samples(self):
        top = np.column_stack((np.linspace(0., .015, 4), np.zeros(4)))
        rest = np.vstack((top, [[.015, -.02], [0., -.02]]))
        edges = [[i, (i+1)%len(rest)] for i in range(len(rest))]
        p = self.geometry(rest, edges, 'improved-max', len(rest))
        p.n = len(rest)
        check = p.check_reference_contact()
        self.assertEqual(check['status'], 'passed')
        self.assertEqual(check['body_contact_force_norm_n'], 0.)
        self.assertEqual(check['normal_barrier_energy_n_mm'], 0.)

    def test_stress_free_reference_check_rejects_same_polygon_dense_sampling(self):
        # Same rectangle, only its top-edge sampling changes. There is no
        # geometric overlap, yet adjacent samples inside dhat create forces.
        top = np.column_stack((np.linspace(0., .015, 13), np.zeros(13)))
        rest = np.vstack((top, [[.015, -.02], [0., -.02]]))
        edges = [[i, (i+1)%len(rest)] for i in range(len(rest))]
        p = self.geometry(rest, edges, 'improved-max', len(rest))
        p.n = len(rest)
        self.assertFalse(self.ipc.has_intersections(p.cmesh, rest))
        with self.assertRaisesRegex(ValueError, 'contact-induced prestress'):
            p.check_reference_contact()
