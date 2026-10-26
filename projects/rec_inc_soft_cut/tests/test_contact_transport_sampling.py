import numpy as np
import pytest

from src.workflows.contact_inputs import sample_rows


def test_missing_blade_depths_are_never_sampled_as_contact_observations():
    depth = np.linspace(0.,1.1,101)
    depth[12:45] = np.nan
    for preview in (False,True):
        rows = sample_rows(depth,preview)
        assert rows[0] == 0 and rows[-1] == 100
        assert np.isfinite(depth[rows]).all()
        assert not ((rows >= 12) & (rows < 45)).any()
        assert int(np.argmin(abs(np.nan_to_num(depth,nan=100)-1.))) in rows


def test_unchanged_sampling_when_all_depths_are_available():
    depth = np.linspace(0.,1.5,101)
    expected = set(range(0,101,2))
    expected.add(100)
    expected.update(int(np.argmin(abs(depth-d))) for d in (.1,.3,.5,1.,2.))
    np.testing.assert_array_equal(sample_rows(depth),sorted(expected))


def test_unknown_depth_series_is_rejected():
    with pytest.raises(ValueError,match='finite'):
        sample_rows(np.full(10,np.nan))
