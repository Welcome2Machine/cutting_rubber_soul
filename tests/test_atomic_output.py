"""A reporting-reader race must not discard an accepted mechanical state."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from run_support import replace_with_retry, write_json


class AtomicOutputTests(unittest.TestCase):
    def test_transient_replace_denial_keeps_destination_until_success(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/"result.json";target.write_text('{"old":true}')
            original=Path.replace;calls=[]
            def locked(source,destination):
                calls.append(source)
                if len(calls)<3:
                    self.assertEqual(target.read_text(),'{"old":true}')
                    raise PermissionError("temporary reader lock")
                return original(source,destination)
            with patch.object(Path,"replace",locked):
                write_json(target,{"accepted_depth_mm":.845})
            self.assertEqual(json.loads(target.read_text()),{"accepted_depth_mm":.845})
            self.assertEqual(len(calls),3)
            self.assertFalse(target.with_suffix(".tmp").exists())

    def test_persistent_denial_propagates_without_erasing_old_result(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/"result.json";target.write_text("old")
            temporary=Path(d)/"result.tmp";temporary.write_text("new")
            with patch.object(Path,"replace",side_effect=PermissionError("ACL")):
                with self.assertRaises(PermissionError):
                    replace_with_retry(temporary,target,timeout_s=.02)
            self.assertEqual(target.read_text(),"old")
            self.assertEqual(temporary.read_text(),"new")

    def test_unrelated_error_is_not_retried(self):
        with patch.object(Path,"replace",side_effect=OSError("disk full")) as replace:
            with self.assertRaises(OSError):
                replace_with_retry("a","b")
            self.assertEqual(replace.call_count,1)
