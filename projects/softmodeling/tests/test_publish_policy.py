"""Keep reusable source/config assets publishable and local research artifacts out."""
import unittest

from tools.check_publish import allowed


class PublicationPolicyTests(unittest.TestCase):
    def test_maintained_code_and_workbench_assets_are_allowed(self):
        for path in ('README.md', 'AGENTS.md', 'prototype/material_case.py',
                     'workbench/config.py', 'workbench/static/index.html',
                     'workbench/static/app.js', 'workbench/static/app.css',
                     'configs/materials.json', 'configs/blades.json',
                     'docs/workbench.md', 'references/README.md',
                     'reproductions/dammass2023/acceptance.json'):
            with self.subTest(path=path):
                self.assertTrue(allowed(path))

    def test_local_data_and_unknown_configs_are_rejected(self):
        for path in ('outputs/run/config.json', 'outputs/run/report.html',
                     'outputs/run/reproduce.py', '.env', '.runtime/library.dll',
                     'docs/force.csv', 'docs/frame.png', 'references/paper.pdf',
                     'configs/experiment.json', 'workbench/static/video.mp4',
                     'workbench/static/result.json', '../README.md', '/README.md', ''):
            with self.subTest(path=path):
                self.assertFalse(allowed(path))
