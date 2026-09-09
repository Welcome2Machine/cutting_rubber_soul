"""Output integrity shared by the cutting and paper runners."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from run_support import ROOT,write_json,save_npz,source_hashes


class RunSupportTests(unittest.TestCase):
    def test_invalid_json_does_not_replace_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'status.json'
            write_json(path,{'state':'计算中'})
            before=path.read_bytes()
            with self.assertRaises(ValueError):write_json(path,{'residual':float('nan')})
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),{'state':'计算中'})
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_checkpoint_replacement_is_readable_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.npz'
            save_npz(path,u=np.zeros(3));save_npz(path,u=np.arange(4))
            with np.load(path) as state:np.testing.assert_array_equal(state['u'],np.arange(4))
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_source_snapshot_preserves_layout_and_shared_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            hashes=source_hashes(('mechanics',),snapshot=directory)
            self.assertEqual(hashes,source_hashes(('mechanics',)))
            self.assertIn('run_support.py',hashes)
            self.assertIn('mechanics/finite_strain.py',hashes)
            for name in hashes:self.assertEqual((Path(directory)/name).read_bytes(),(ROOT/name).read_bytes())
