"""Actual-blade comparison plumbing and geometry; never launch a loading matrix."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from prototype.cohesive_cut import CohesiveCutProblem
from prototype.ipc_viscoelastic import ViscoelasticProblem
from prototype.ipc_reference import toolkit
from run_support import write_json
from tools import verify_cutting_compliance as tool


def specs(**kwargs):
    options = dict(gap_mm=6e-7, normal_stiffness_mpa_per_mm=5000.,
                   compression_stiffness_mpa_per_mm=5000.)
    options.update(kwargs)
    return tool.build_matrix(**options)


def record(depth, dt, force=1.):
    return dict(depth_mm=depth, reaction_fz_n=force, residual_n=1e-7, min_j=1.,
        physical_time_including_preparation_s=(depth+.02)/.2,
        material=dict(dt_s=dt, stored_energy_n_mm=0., equilibrium_energy_n_mm=0.,
            overstress_energy_n_mm=0., viscous_dissipation_increment_n_mm=0.,
            cumulative_viscous_dissipation_n_mm=0.), friction_work_n_mm=0.,
        compliance_observations=dict(ipc_ev_contacts=0, ipc_vv_contacts=0,
                                    ipc_barrier_stored_energy_n_mm=0.))


def comparable_pair():
    result = []
    for spec in specs(surfaces=(.02,), increment_mm=.005):
        result.append(dict(**spec, completed=True, status='completed',
            qualified_for_comparison=True, artifacts_authenticated=True,
            source_sha256={'core': 'same'},
            geometry=dict(initial_blade_sha256='blade', zero_depth_blade_sha256='zero',
                initial_depth_mm=-.02, loading_rate_mm_s=.2, base_mesh_sha256='mesh'),
            steps=[record(0., .1, .01), record(.005, .025, .5), record(.01, .025, 1.)]))
    for step in result[1]['steps']:
        step['reaction_fz_n'] *= .9
    return result


def test_front_diagnostic_is_explicit_bounded_and_keeps_damage_disabled():
    import pytest
    with pytest.raises(ValueError):
        specs(target_mm=.5)
    cases = specs(target_mm=.5, front_diagnostic=True)
    assert all(c['front_diagnostic'] and c['configuration']['cohesive']['damage_enabled'] is False for c in cases)
    with pytest.raises(ValueError):
        specs(target_mm=.51, front_diagnostic=True)
    with pytest.raises(ValueError):
        specs(front_diagnostic=1)


def test_saved_depth_schedule_does_not_transplant_material_history():
    import pytest
    config = specs()[0]['configuration']
    config['diagnostic_depths_mm'] = [0., .002, .006, .01]
    np.testing.assert_array_equal(tool.targets(config), config['diagnostic_depths_mm'])
    for bad in ([.002, .01], [0., .006, .005, .01], [0., .009], [0., float('nan'), .01]):
        config['diagnostic_depths_mm'] = bad
        with pytest.raises(ValueError):
            tool.targets(config)


class NativeSavedPathFixture:
    """One real triangle and an isolated moving point, for geometric CCD only."""
    safe_step = CohesiveCutProblem.safe_step

    def __init__(self):
        self.ipc = toolkit()
        self.n = 3
        self.reference = np.array([[-.1, -.1], [.1, -.1], [.1, .1], [0., np.sqrt(2)*.02]])
        self.mesh = SimpleNamespace(reference_xy=self.reference[:3], triangles=np.array([[0, 1, 2]]))
        inverse = np.linalg.inv(np.column_stack((self.reference[1]-self.reference[0],
                                                 self.reference[2]-self.reference[0])))
        self.grad = np.array([np.vstack((-inverse.sum(axis=0), inverse))])
        self.surface = np.arange(4)
        self.cmesh = self.ipc.CollisionMesh(self.reference, np.array([[0, 1], [1, 2], [2, 0]]))
        self.cmesh.can_collide = self.ipc.make_static_obstacle_filter(3)
        self.network_narrow_phase_ccd = self.ipc.TightInclusionCCD(tolerance=1e-9,
            max_iterations=self.ipc.TightInclusionCCD.DEFAULT_MAX_ITERATIONS, conservative_rescaling=.8)

    def _record_phase(self, *args):
        pass


class ActualBladeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specifications = specs(surfaces=(.2,))
        # Construction is the only real numerical work in these tests.
        with patch.object(ViscoelasticProblem, 'step', side_effect=AssertionError('No loading in geometry tests')):
            cls.intact = tool.make_problem(cls.specifications[0])
            cls.network = tool.make_problem(cls.specifications[1])

    def test_common_base_blade_and_real_collision_reference(self):
        first, second = map(tool.describe_problem, (self.intact, self.network))
        for key in ('base_mesh_sha256', 'initial_blade_sha256', 'zero_depth_blade_sha256',
                    'initial_depth_mm', 'loading_rate_mm_s', 'base_triangles'):
            self.assertEqual(first[key], second[key], key)
        self.assertEqual(first['initial_depth_mm'], -.02)
        self.assertLess(second['area_ratio_to_base'], 1.)
        self.assertIs(tool.PreparedIntactProblem.safe_step, CohesiveCutProblem.safe_step)
        for problem in (self.intact, self.network):
            np.testing.assert_array_equal(problem.previous, problem.rest)
            np.testing.assert_array_equal(np.asarray(problem.cmesh.rest_positions), problem.rest[problem.surface])
            np.testing.assert_array_equal(problem.material_metric,
                np.tile(np.eye(3), (len(problem.mesh.triangles), 1, 1)))
            self.assertIsNone(problem.material_dt)
            self.assertEqual(problem.material_dissipation, 0.)
            self.assertEqual(problem.network_narrow_phase_ccd.tolerance, 1e-9)
            self.assertEqual(problem.network_narrow_phase_ccd.conservative_rescaling, .8)
            self.assertEqual(problem.ptc_shift_policy, 'residual_ratio')
            self.assertEqual(problem.ptc_model_backtracks, 12)
            self.assertFalse(problem.ptc_residual_guard)
            self.assertFalse(problem.ipc.has_intersections(problem.cmesh, problem.rest[problem.surface]))

    def test_native_ccd_blocks_blade_crossing_for_both_models(self):
        for problem in (self.intact, self.network):
            with self.subTest(type=type(problem).__name__):
                direction = np.zeros_like(problem.rest)
                self.assertEqual(problem.safe_step(problem.rest, direction), 1.)
                direction[problem.n:, 1] = -.03
                fraction = problem.safe_step(problem.rest, direction)
                self.assertGreater(fraction, 0.)
                self.assertLess(fraction, 1.)

    def test_dhat_does_not_silently_change_preparation_or_time(self):
        config = copy.deepcopy(self.specifications[0])
        config['configuration']['dhat_mm'] /= 2
        other = tool.make_problem(config)
        np.testing.assert_allclose(other.rest[other.n:], self.intact.rest[self.intact.n:], rtol=0, atol=2e-14)
        np.testing.assert_allclose(other.prescribed_blade_positions(0.),
                                   self.intact.prescribed_blade_positions(0.), rtol=0, atol=2e-14)
        self.assertEqual(other.start, self.intact.start)
        self.assertAlmostEqual((0.-other.start)/other.loading_rate, .1)

    def test_intact_extra_audit_rejects_corrupt_energy(self):
        saved = record(0., .1)
        fresh = copy.deepcopy(saved)
        self.intact._audit_extra_acceptance(fresh, saved)
        fresh['material']['stored_energy_n_mm'] = 1.
        with self.assertRaises(AssertionError):
            self.intact._audit_extra_acceptance(fresh, saved)

    def test_optional_backtrack_aware_policy_is_identical_in_both_hosts(self):
        for spec in specs(surfaces=(.2,), ptc_shift_policy='backtrack_aware'):
            problem = tool.make_problem(spec)
            self.assertEqual(problem.ptc_shift_policy, 'backtrack_aware')
            self.assertEqual(problem.ptc_model_backtracks, 12)
            self.assertFalse(problem.ptc_residual_guard)

    def test_safe_near_contact_has_complete_coverage_without_relaxing_native_ccd(self):
        problem = NativeSavedPathFixture()
        start, end = problem.reference.copy(), problem.reference.copy()
        end[3, 1] = np.sqrt(2)*2.5e-5
        original = problem.safe_step(start, end-start)
        self.assertGreater(original, .99)
        self.assertLess(original, 1.)
        result = tool.certify_saved_transition(problem, start, end)
        self.assertTrue(result['passed'], result)
        self.assertTrue(tool.coverage_evidence_valid(result))
        self.assertEqual(result['initial_query_fraction'], original)
        self.assertGreater(len(result['pieces']), 1)
        self.assertEqual(result['pieces'][0]['alpha0'], 0.)
        self.assertEqual(result['pieces'][-1]['alpha1'], 1.)
        self.assertFalse(result['physical_substeps_added'])
        self.assertFalse(result['histories_modified'])
        self.assertEqual(problem.network_narrow_phase_ccd.conservative_rescaling, .8)
        self.assertEqual(problem.network_narrow_phase_ccd.tolerance, 1e-9)
        short = tool.certify_saved_transition(problem, start, end, maximum_pieces=1)
        self.assertFalse(short['passed'])
        self.assertLess(short['covered_fraction'], 1.)

    def test_real_crossing_cannot_obtain_complete_coverage(self):
        problem = NativeSavedPathFixture()
        start, end = problem.reference.copy(), problem.reference.copy()
        end[3, 1] = -np.sqrt(2)*.02
        result = tool.certify_saved_transition(problem, start, end)
        self.assertFalse(result['passed'], result)
        self.assertFalse(tool.coverage_evidence_valid(result))
        self.assertLessEqual(result['covered_fraction'], .5)
        self.assertTrue('progress' in result['reason'] or 'advance' in result['reason']
                        or 'budget' in result['reason'] or 'prefix' in result['reason'])

    def test_coverage_cannot_hide_a_hole_invalid_j_or_failed_piece(self):
        problem = NativeSavedPathFixture()
        start, end = problem.reference.copy(), problem.reference.copy()
        end[3, 1] = np.sqrt(2)*2.5e-5
        result = tool.certify_saved_transition(problem, start, end)
        self.assertTrue(tool.coverage_evidence_valid(result))
        for change in (lambda x: x['pieces'][1].update(alpha0=.1),
                       lambda x: x['pieces'][-1].update(alpha1=.99),
                       lambda x: x['pieces'][0].update(subinterval_recheck_fraction=.99),
                       lambda x: x['pieces'][0].update(min_j_start=-1.),
                       lambda x: x['pieces'][0].update(end_intersection=True),
                       lambda x: x.update(physical_substeps_added=True)):
            bad = copy.deepcopy(result)
            change(bad)
            self.assertFalse(tool.coverage_evidence_valid(bad))


class ActualBladeQualificationTests(unittest.TestCase):
    def test_matrix_is_paired_and_only_mesh_axis_changes(self):
        matrix = specs()
        self.assertEqual(len(matrix), 6)
        self.assertEqual({x['configuration']['mesh']['surface_spacing_mm'] for x in matrix}, {.02, .01, .005})
        original = copy.deepcopy(matrix[0]['configuration'])
        original['mesh'].pop('surface_spacing_mm')
        for first, second in zip(matrix[::2], matrix[1::2]):
            self.assertEqual(first['configuration'], second['configuration'])
            config = copy.deepcopy(first['configuration'])
            self.assertFalse(config['cohesive']['damage_enabled'])
            self.assertEqual(config['preparation_clearance_mm'], .02)
            self.assertEqual(config['mesh']['fine_halfwidth_mm'], .2)
            self.assertEqual(config['mesh']['local_spacing_mm'], .75)
            self.assertEqual(config['mesh']['coarse_spacing_mm'], 1.5)
            self.assertEqual(config['mesh']['nose_facets'], 24)
            self.assertFalse(config['ptc_residual_guard'])
            self.assertEqual(config['ptc_shift_policy'], 'residual_ratio')
            self.assertEqual(config['cohesive']['compression_regularization_mm'], config['reference_gap_mm'])
            config['mesh'].pop('surface_spacing_mm')
            self.assertEqual(config, original)
        np.testing.assert_allclose(tool.targets(matrix[0]['configuration']), np.linspace(0., .01, 9))

    def test_invalid_scope_and_ccd_are_rejected_before_construction(self):
        invalid = [dict(gap_mm=4e-7), dict(ccd_tolerance_mm=1e-6), dict(target_mm=.1),
            dict(surfaces=()), dict(surfaces=(.02, .02)), dict(normal_stiffness_mpa_per_mm=-1),
            dict(minimum_increment_mm=.002), dict(preparation_clearance_mm=float('nan')),
            dict(ptc_model_backtracks=0, ptc_residual_guard=True), dict(nose_facets=3),
            dict(ptc_shift_policy='unknown'), dict(ptc_shift_policy='backtrack_aware', ptc_model_backtracks=0)]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                specs(**options)

    def test_complete_pair_is_compared_without_claiming_convergence(self):
        intact, network = comparable_pair()
        result = tool.compare_cases(network, intact)
        self.assertTrue(result['comparison_available'])
        self.assertAlmostEqual(result['terminal_force_ratio'], .9)
        self.assertAlmostEqual(result['relative_l2_force_difference'], .1)
        self.assertFalse(result['complete_cutting_qualified'])
        self.assertFalse(result['mesh_convergence_proven'])

    def test_incomplete_unaudited_mismatched_or_unresolved_pair_is_not_compared(self):
        mutations = [lambda x: x.update(qualified_for_comparison=False),
            lambda x: x.update(artifacts_authenticated=False),
            lambda x: x['source_sha256'].update(core='changed'),
            lambda x: x['geometry'].update(base_mesh_sha256='different'),
            lambda x: x['geometry'].update(initial_blade_sha256='different'),
            lambda x: x['configuration'].update(preparation_clearance_mm=.01),
            lambda x: x['steps'].pop(1),
            lambda x: x['steps'].pop(),
            lambda x: x.update(completed=False),
            lambda x: x.update(status='budget_exhausted'),
            lambda x: x['steps'][1]['material'].update(dt_s=.02)]
        for mutation in mutations:
            intact, network = comparable_pair()
            mutation(network)
            with self.subTest(mutation=mutation):
                result = tool.compare_cases(network, intact)
                self.assertFalse(result['comparison_available'])
                self.assertIsNone(result['terminal_force_ratio'])
                self.assertIsNone(result['maximum_absolute_force_difference_n'])
        intact, network = comparable_pair()
        intact['steps'][-1]['reaction_fz_n'] = 1e-5
        self.assertFalse(tool.compare_cases(network, intact)['comparison_available'])

    def test_mesh_comparison_allows_only_surface_spacing_and_base_mesh_to_change(self):
        reference, candidate = comparable_pair()
        candidate['model'] = 'intact'
        candidate['configuration']['mesh']['surface_spacing_mm'] = .01
        candidate['geometry']['base_mesh_sha256'] = 'different'
        self.assertTrue(tool.compare_cases(candidate, reference, kind='surface_mesh')['comparison_available'])
        candidate['configuration']['cohesive']['normal_stiffness_mpa_per_mm'] *= 2
        self.assertFalse(tool.compare_cases(candidate, reference, kind='surface_mesh')['comparison_available'])

    def test_checkpoint_record_and_configuration_tampering_fail_authentication(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'states').mkdir()
            report = comparable_pair()[0]
            report['source_sha256_at_end'] = report['source_sha256'].copy()
            write_json(directory/'config.json', report['configuration'])
            report['input_sha256'] = tool.digest((directory/'config.json').read_bytes())
            rows = []
            for i, step in enumerate(report['steps']):
                checkpoint = directory/'states'/f'{i:04d}.npz'
                checkpoint.write_bytes(b'authenticated-test-checkpoint-'+str(i).encode())
                rows.append(dict(passed=True, checkpoint_sha256=tool.digest(checkpoint.read_bytes()),
                    accepted_record_sha256=tool.record_digest(step), residual_n=1e-7,
                    min_j=1., transition_safe_fraction=1.))
            write_json(directory/'audit.json', dict(passed=True, rows=rows,
                full_saved_transition_ccd_j=True, damage_disabled_verified=True))
            self.assertTrue(tool.authenticate_case(directory, report))
            changed = copy.deepcopy(report)
            changed['steps'][0]['reaction_fz_n'] = 10.
            self.assertFalse(tool.authenticate_case(directory, changed))
            changed = copy.deepcopy(report)
            changed['configuration']['preparation_clearance_mm'] = .01
            self.assertFalse(tool.authenticate_case(directory, changed))
            (directory/'states'/'0001.npz').write_bytes(b'changed')
            self.assertFalse(tool.authenticate_case(directory, report))

    def test_energy_ledger_counts_compression_once(self):
        first, second = record(0., .1, 1000.), record(.01, .05, 1000.)
        second['material']['stored_energy_n_mm'] = 3.
        first['fracture'] = dict(cohesive_stored_energy_n_mm=0., compression_stored_energy_n_mm=0.)
        second['fracture'] = dict(cohesive_stored_energy_n_mm=5., compression_stored_energy_n_mm=4.)
        result = tool.energy_ledger([first, second])
        self.assertEqual(result['rows'][0]['unaccounted_energy_n_mm'], 2.)
        self.assertFalse(result['qualified'])

    def test_saved_transition_audit_rejects_ccd_before_qualification(self):
        class Problem:
            start, loading_rate = -.02, .2
            def safe_step(self, x, direction):
                return .5
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'states').mkdir()
            np.savez(directory/'states'/'0000.npz', positions=np.zeros((2, 2)), previous_positions=np.zeros((2, 2)))
            with (patch.object(tool, 'audit', return_value=dict(rows=[{}], passed=True)),
                  patch.object(tool, 'certify_saved_transition', return_value=dict(passed=False, reason='unsafe path'))):
                with self.assertRaisesRegex(RuntimeError, 'complete CCD/J'):
                    tool.path_audit(Problem(), directory, {'steps': [record(0., .1)]}, time.monotonic()+5)

    def test_controller_failure_is_not_an_accepted_prefix_or_comparison(self):
        specification = specs(surfaces=(.2,))[0]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'case'
            with (patch.object(tool, 'adaptive_load_steps', side_effect=RuntimeError('Injected rejected trial')),
                  patch.object(ViscoelasticProblem, 'step', side_effect=AssertionError('No FE solve'))):
                result = tool.run_case(specification, output, seconds=10.)
            self.assertEqual(result['steps'], [])
            self.assertEqual(result['qualified_prefix_states'], 0)
            self.assertFalse(result['completed'])
            self.assertFalse(result['qualified_for_comparison'])
            self.assertIn('Injected rejected trial', result['error'])
            self.assertAlmostEqual(result['physical_time']['preparation_s'], .1)
            self.assertAlmostEqual(result['physical_time']['total_requested_s'], .15)
            self.assertEqual(result['source_sha256'], result['source_sha256_at_end'])

    def test_hard_budget_retains_inconclusive_report_and_never_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'report'
            with patch.object(tool.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 11.)) as child:
                self.assertEqual(tool.run(output, specs(surfaces=(.2,)), seconds=1.), 1)
            self.assertEqual(child.call_count, 2)
            for call in child.call_args_list:
                self.assertIsInstance(call.args[0], list)
                self.assertEqual(call.kwargs['timeout'], 11.)
                self.assertNotIn('shell', call.kwargs)
            report = json.loads((output/'verification.json').read_text(encoding='utf-8'))
            self.assertFalse(report['all_comparisons_available'])
            self.assertTrue(all(not c['qualified_for_comparison'] for c in report['cases']))
            self.assertFalse(report['comparisons'][0]['comparison_available'])
            self.assertEqual(report['experimental_inputs_used'], [])
            self.assertTrue((output/'index.html').is_file())
            self.assertTrue((output/'curves.csv').is_file())
            self.assertFalse(report['complete_cutting_qualified'])

    def test_prepare_only_cannot_launch_a_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'manifest'
            with patch.object(tool.subprocess, 'run', side_effect=AssertionError('No subprocess')):
                status = tool.main(['--output', str(output), '--gap-mm', '6e-7', '--normal-stiffness', '5000',
                                   '--compression-stiffness', '5000', '--prepare-only'])
            self.assertEqual(status, 0)
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(manifest['loading_started'])
            self.assertEqual(len(manifest['cases']), 6)


if __name__ == '__main__':
    unittest.main()
