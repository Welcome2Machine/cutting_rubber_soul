"""Workbench physics/configuration contracts, without launching loading jobs."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from workbench.config import catalog, describe, normalize_config, solver_command


class WorkbenchConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = catalog()['defaults']

    def changed(self, section, key, value):
        candidate = copy.deepcopy(self.config)
        (candidate if section is None else candidate[section])[key] = value
        return candidate

    def test_default_is_common_maxwell_not_elastic_reference(self):
        c = normalize_config(self.config)
        self.assertEqual(c['material']['law'], 'isochoric_maxwell')
        self.assertEqual(tuple(c['material'][k] for k in ('Ginf_MPa', 'Gv_MPa', 'tau_s')),
                         (.14, 1.24, 1.7))
        self.assertAlmostEqual(c['material']['K_MPa'], 23.34333333333331)
        self.assertEqual(c['blade']['id'], '99755')
        self.assertEqual(c['blade']['thickness_mm'], .245)
        self.assertEqual(c['blade']['included_angle_deg'], 23)
        self.assertEqual(c['geometry']['tip_radius_mm'], .02)
        self.assertEqual(c['contact']['mu'], .35)
        self.assertEqual(c['simulation']['speed_mm_s'], .2)
        self.assertEqual(c['simulation']['coarse_spacing_mm'], 1.)
        info = describe(c)
        self.assertEqual(info['requested_output_states'], 201)
        self.assertAlmostEqual(info['physical_dt_s'], .05)
        self.assertAlmostEqual(info['physical_duration_s'], 10.)
        self.assertEqual(info['force_tolerance_N'], 1e-4)

    def test_normalization_is_pure_and_catalogues_are_independent(self):
        before = copy.deepcopy(self.config)
        c = normalize_config(self.config)
        self.assertEqual(self.config, before)
        c['material']['Ginf_MPa'] = .9
        self.assertEqual(self.config, before)
        self.config['blade']['thickness_mm'] = 1.
        self.assertEqual(catalog()['defaults']['blade']['thickness_mm'], .245)

    def test_unknown_and_missing_keys_fail_at_every_config_level(self):
        for section in (None, 'material', 'blade', 'geometry', 'contact', 'simulation', 'resources', 'output'):
            with self.subTest(section=section, kind='unknown'), self.assertRaises(ValueError):
                normalize_config(self.changed(section, 'unsupported_control', 1))
            candidate = copy.deepcopy(self.config)
            part = candidate if section is None else candidate[section]
            del part[next(iter(part))]
            with self.subTest(section=section, kind='missing'), self.assertRaises(ValueError):
                normalize_config(candidate)
        for raw in (None, [], 'configuration'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_config(raw)

    def test_nonfinite_boolean_and_text_numbers_are_not_controls(self):
        numeric = {
            'material': ('Ginf_MPa', 'Gv_MPa', 'tau_s', 'K_MPa'),
            'blade': ('thickness_mm', 'included_angle_deg'),
            'geometry': ('angle_deg', 'tip_radius_mm'),
            'contact': ('mu',),
            'simulation': ('depth_mm', 'increment_mm', 'speed_mm_s', 'local_spacing_mm',
                           'surface_spacing_mm', 'fine_halfwidth_mm', 'coarse_spacing_mm',
                           'nose_facets', 'blade_spacing_mm'),
            'resources': ('solve_seconds', 'total_seconds', 'ipc_threads', 'linear_threads',
                          'material_batch_size'),
            'output': ('max_frames',),
        }
        for section, fields in numeric.items():
            for field in fields:
                for bad in (True, False, float('nan'), float('inf'), -float('inf'), '0.2'):
                    with self.subTest(section=section, field=field, bad=bad), self.assertRaises(ValueError):
                        normalize_config(self.changed(section, field, bad))

    def test_physical_scope_and_circle_contact_are_enforced(self):
        bad = [('simulation', 'depth_mm', 2.001), ('simulation', 'depth_mm', 0.),
               ('simulation', 'speed_mm_s', .1), ('contact', 'mu', 0.), ('contact', 'mu', -.1),
               ('material', 'law', 'elastic'), ('material', 'Gv_MPa', 0.),
               ('material', 'K_MPa', 2*.14/3), ('blade', 'shape', 'single_bevel'),
               ('geometry', 'angle_deg', 75.001), ('geometry', 'angle_deg', -.01),
               ('geometry', 'tip_radius_mm', .245/2), ('geometry', 'tip_radius_mm', 0.),
               ('contact', 'condition', 'hydrodynamic')]
        for section, field, value in bad:
            with self.subTest(section=section, field=field, value=value), self.assertRaises(ValueError):
                normalize_config(self.changed(section, field, value))
        c = self.changed('geometry', 'angle_deg', 75.)
        c['blade']['included_angle_deg'] = 30.
        with self.assertRaisesRegex(ValueError, '90'):
            normalize_config(c)
        c['geometry']['angle_deg'] = 74.9
        self.assertEqual(normalize_config(c)['geometry']['angle_deg'], 74.9)
        c = self.changed('simulation', 'depth_mm', .005)
        with self.assertRaises(ValueError):
            normalize_config(c)

    def test_mesh_preset_consistency_and_integer_controls(self):
        with self.assertRaises(ValueError):
            normalize_config(self.changed('simulation', 'local_spacing_mm', .075))
        c = self.changed('simulation', 'local_spacing_mm', .075)
        c['simulation']['mesh'] = 'custom'
        self.assertEqual(normalize_config(c)['simulation']['local_spacing_mm'], .075)
        for updates in ({'surface_spacing_mm': .08}, {'coarse_spacing_mm': .1, 'local_spacing_mm': .2},
                        {'nose_facets': 63}, {'nose_facets': 64.0}, {'fine_halfwidth_mm': 0.},
                        {'blade_spacing_mm': .1}):
            candidate = copy.deepcopy(c)
            candidate['simulation'].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                normalize_config(candidate)
        c['simulation']['blade_spacing_mm'] = None
        self.assertIsNone(normalize_config(c)['simulation']['blade_spacing_mm'])

    def test_resource_budget_and_backend_are_explicit(self):
        for field, value in (('total_seconds', 7200.), ('solve_seconds', 0.),
                             ('ipc_threads', 0), ('linear_threads', 4.0),
                             ('material_batch_size', -1), ('linear_backend', 'automatic')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize_config(self.changed('resources', field, value))
        for field, value in (('movie', 1), ('max_frames', 42), ('max_frames', 2.0)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize_config(self.changed('output', field, value))

    def test_oil_label_never_silently_changes_friction(self):
        oil = self.changed('contact', 'condition', 'oil')
        c = normalize_config(oil)
        self.assertEqual(c['contact']['mu'], .35)
        self.assertEqual(solver_command(c, 'outputs/contract') ,
                         solver_command(self.config, 'outputs/contract'))
        self.assertTrue(any('油润滑' in message for message in describe(c)['messages']))

    def test_solver_argument_vector_maps_custom_values_and_ignores_display_text(self):
        c = copy.deepcopy(self.config)
        c['name'] = 'test; $(not-a-command)'
        c['material'].update(Ginf_MPa=.2, Gv_MPa=.9, tau_s=2., K_MPa=24.)
        c['blade'].update(thickness_mm=.3, included_angle_deg=27.)
        c['geometry'].update(angle_deg=42.5, tip_radius_mm=.03)
        c['contact'].update(condition='oil', mu=.12)
        c['simulation'].update(mesh='custom', depth_mm=.5, increment_mm=.005,
            local_spacing_mm=.08, surface_spacing_mm=.008, fine_halfwidth_mm=2.,
            coarse_spacing_mm=1.5, nose_facets=48, blade_spacing_mm=None)
        c['resources'].update(solve_seconds=900., total_seconds=1200., ipc_threads=3,
                             linear_threads=2, linear_backend='pardiso', material_batch_size=1024)
        destination = Path('outputs')/'name with spaces; $literal'
        argv = solver_command(c, destination, python='chosen-python')
        self.assertIsInstance(argv, list)
        self.assertEqual(argv[:6], ['chosen-python', '-X', 'utf8', '-u', '-m', 'tools.run_material_candidate'])
        flags = dict(zip(argv[6::2], argv[7::2]))
        expected = {'--output':str(destination.resolve()), '--equilibrium-shear':'.2', '--maxwell-shear':'.9',
            '--relaxation-s':'2', '--bulk-modulus':'24', '--loading-rate':'.2', '--angle':'42.5',
            '--mu':'.12', '--blade-thickness':'.3', '--blade-angle':'27', '--tip-radius':'.03',
            '--depth':'.5', '--increment':'.005', '--local-spacing':'.08', '--surface-spacing':'.008',
            '--fine-halfwidth':'2', '--coarse-spacing':'1.5', '--nose-facets':'48', '--seconds':'900',
            '--threads':'3', '--linear-backend':'pardiso', '--linear-threads':'2', '--material-batch-size':'1024'}
        self.assertEqual(set(flags), set(expected))
        for key, value in expected.items():
            if key in ('--output', '--linear-backend'):
                self.assertEqual(flags[key], value)
            else:
                self.assertEqual(float(flags[key]), float(value))
        self.assertNotIn(c['name'], argv)
        self.assertNotIn('--blade-spacing', flags)
        default_argv = solver_command(self.config, destination)
        self.assertEqual(default_argv[default_argv.index('--blade-spacing')+1], '0.005')


class WorkbenchProblemConstructionTests(unittest.TestCase):
    def test_make_problem_preserves_old_defaults_and_passes_explicit_geometry(self):
        from prototype.material_case import make_problem
        inputs = dict(angle=57., equilibrium_shear=.14, maxwell_shear=1.24,
                      relaxation_s=1.7, loading_rate=.2, bulk_modulus=23.34333333333331)
        expected = dict(maxwell_shear=1.24, relaxation_s=1.7, loading_rate=.2,
            bulk_modulus=23.34333333333331, volumetric='quadratic_j', case='oblique', mu=.35,
            shear_modulus=.14, ogden_alpha=2., dhat=.0025, local_spacing=.05,
            surface_spacing=.005, nose_facets=64, inclination=57., tip_radius=.02,
            fine_halfwidth=3., max_iterations=500, predictor='secant', hessian_mode='regularized',
            element='p1', blade_spacing=None, blade_thickness=None, blade_angle=None, coarse_spacing=None)
        with patch('prototype.material_case.ViscoelasticProblem', return_value=SimpleNamespace()) as constructor:
            p = make_problem(**inputs)
            constructor.assert_called_once_with(**expected)
            self.assertEqual(p.stagnation_window, 40)
        overrides = dict(blade_thickness=.3, blade_angle=27., coarse_spacing=1.5,
                         tip_radius=.03, local_spacing=.08, surface_spacing=.008,
                         fine_halfwidth=2., nose_facets=48, blade_spacing=.01)
        with patch('prototype.material_case.ViscoelasticProblem', return_value=SimpleNamespace()) as constructor:
            make_problem(**inputs, **overrides)
            constructor.assert_called_once_with(**dict(expected, **overrides))

    def test_real_unloaded_geometry_uses_blade_and_far_field_controls(self):
        from prototype.ipc_reference import IPCProblem, toolkit
        from threadpoolctl import threadpool_limits
        try:
            ipc = toolkit()
        except ModuleNotFoundError:
            self.skipTest('Optional ipctk==1.6.0 not installed')
        ipc.set_num_threads(1)
        common = dict(case='oblique', mu=.35, inclination=33., tip_radius=.02,
                      dhat=.0025, local_spacing=.5, surface_spacing=.1,
                      fine_halfwidth=.5, nose_facets=8, blade_spacing=None)
        with threadpool_limits(limits=1):
            a = IPCProblem(**common, coarse_spacing=1.)
            b = IPCProblem(**common, blade_thickness=.3, blade_angle=30., coarse_spacing=2.)
        self.assertEqual(a.config.blade_thickness_mm, .245)
        self.assertEqual(a.config.blade_tip_angle_deg, 23.)
        self.assertEqual(b.config.blade_thickness_mm, .3)
        self.assertEqual(b.config.blade_tip_angle_deg, 30.)
        self.assertEqual(b.config.mesh_size_mm, 2.)
        self.assertNotEqual(a.n, b.n)
        self.assertAlmostEqual(float(np.ptp(a.rest[a.n:, 0])), .245, places=12)
        self.assertAlmostEqual(float(np.ptp(b.rest[b.n:, 0])), .3, places=12)
        self.assertFalse(np.allclose(a.rest[a.n:], b.rest[b.n:]))
        for p in (a, b):
            self.assertFalse(ipc.has_intersections(p.cmesh, p.rest[p.surface]))
            np.testing.assert_array_equal(p.previous, p.rest)
            np.testing.assert_array_equal(p.rest[:p.n], p.mesh.reference_xy)


if __name__ == '__main__':
    unittest.main()
