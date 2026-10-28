"""Reported cumulative dissipation audit regression; no FE construction."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from prototype.material_case import audit


class CumulativeReportingAuditTests(unittest.TestCase):
    def fixture(self, directory):
        (directory/'states').mkdir()
        x = np.array([[0., 0.], [1., 0.], [0., 1.]])
        metric = np.eye(3)[None]
        report = {'steps': []}
        for index, (depth, increment, cumulative) in enumerate(((0., .2, .2), (.01, .3, .5))):
            np.savez(directory/'states'/f'{index:04d}.npz', previous_positions=x,
                positions=x, previous_material_metric=metric, material_metric=metric,
                depth_mm=depth, material_dt_s=.05,
                cumulative_viscous_dissipation_n_mm=cumulative)
            report['steps'].append(dict(depth_mm=depth, material=dict(
                viscous_dissipation_increment_n_mm=increment,
                cumulative_viscous_dissipation_n_mm=cumulative)))
        host = SimpleNamespace(rest=x, material_metric=metric, start=-.01, loading_rate=.2,
                               audit_state=lambda state, record: {'passed': True})
        return host, report

    def test_matching_record_and_checkpoint_cumulative_values_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            host, report = self.fixture(directory)
            self.assertTrue(audit(host, directory, report)['passed'])

    def test_tampered_report_cumulative_value_is_rejected(self):
        for index in (0, 1):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                host, report = self.fixture(directory)
                broken = copy.deepcopy(report)
                broken['steps'][index]['material']['cumulative_viscous_dissipation_n_mm'] += 1.
                with self.assertRaises(AssertionError):
                    audit(host, directory, broken)

    def test_missing_report_cumulative_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            host, report = self.fixture(directory)
            del report['steps'][1]['material']['cumulative_viscous_dissipation_n_mm']
            with self.assertRaises(KeyError):
                audit(host, directory, report)


if __name__ == '__main__':
    unittest.main()
