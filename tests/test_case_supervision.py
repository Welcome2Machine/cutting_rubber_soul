import unittest
from unittest.mock import Mock, patch
from tools.run_common_material_study import wait_for_case


class CaseSupervisionTests(unittest.TestCase):
    def test_reporting_io_does_not_orphan_or_cancel_solver(self):
        p=Mock();p.poll.side_effect=[None,None,0,0]
        callback=Mock(side_effect=[PermissionError("reader lock"),None])
        with patch("tools.run_common_material_study.time.sleep"):
            forced,errors=wait_for_case(p,100,callback)
        self.assertIsNone(forced)
        self.assertEqual(len(errors),1)
        self.assertEqual(callback.call_count,2)
        p.terminate.assert_not_called()

    def test_programming_error_cleans_up_owned_solver(self):
        p=Mock();p.poll.return_value=None
        with self.assertRaises(ValueError):
            wait_for_case(p,100,Mock(side_effect=ValueError("bad dashboard")))
        p.terminate.assert_called_once()
        p.wait.assert_called_once()

    def test_timeout_still_enforced_without_dashboard(self):
        p=Mock();p.poll.side_effect=[None,0]
        with patch("tools.run_common_material_study.time.monotonic",side_effect=[0,101]):
            forced,errors=wait_for_case(p,100,Mock())
        self.assertEqual(forced,"supervisor_timeout")
        self.assertEqual(errors,[])
        p.terminate.assert_called_once()
