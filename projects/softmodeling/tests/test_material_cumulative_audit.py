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

    def test_safe_endpoints_do_not_certify_unsafe_physical_chord(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            host,report=self.fixture(directory)
            host.audit_state=lambda state,record: dict(
                passed=record['depth_mm']==0., endpoint_passed=True,
                trajectory_passed=record['depth_mm']==0.,
                physical_chord_safe_fraction=1. if record['depth_mm']==0. else .006)
            result=audit(host,directory,report)
            self.assertFalse(result['passed'])
            self.assertTrue(result['endpoint_checks_passed'])
            self.assertTrue(result['trajectory_checked'])
            self.assertEqual(result['qualified_prefix_states'],1)
            self.assertEqual(len(result['rows']),2)

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


import json
import pytest
from unittest.mock import patch
from prototype.material_case import resume

@pytest.mark.parametrize('passed,prefix,states', [(False, 2, 2), (True, 1, 2), (True, 2, 1)])
def test_resume_rejects_failed_or_short_audit_before_copy_or_history(tmp_path, passed, prefix, states):
    source=tmp_path/'source';out=tmp_path/'out';source.mkdir();out.mkdir()
    report=dict(signature={},source_sha256={},steps=[{'depth_mm':0.}, {'depth_mm':.01}])
    (source/'result.json').write_text(json.dumps(report))
    host=SimpleNamespace(material_metric=np.eye(3),previous=np.ones((3,2)))
    metric=host.material_metric.copy();previous=host.previous.copy()
    with patch('prototype.material_case.audit',return_value=dict(passed=passed,
            qualified_prefix_states=prefix,states=states)), patch('prototype.material_case.shutil.copy2') as copy_file:
        with pytest.raises(ValueError,match='complete requested prefix'):
            resume(host,source,out,copy.deepcopy(report))
        copy_file.assert_not_called()
    np.testing.assert_array_equal(host.material_metric,metric)
    np.testing.assert_array_equal(host.previous,previous)
    assert list(out.iterdir())==[]
