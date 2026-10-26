"""Publication guard prevents research assets leaking into source history."""
from tools.check_publish import reason
import pytest


@pytest.mark.parametrize("path", [
    "projects/softmodeling/outputs/a/result.json",
    "projects/rec_inc_soft_cut/data/movie.mp4",
    "projects/rec_inc_soft_cut/models/checkpoint.pt",
    "projects/softmodeling/docs/experiment.csv",
    "settings.local.json", ".env.secret", "provenance/video-original.bundle",
    "misc/parameters.key", "another/.runtime/module.py",
])
def test_local_assets_rejected(path):
    assert reason(path, 100)


def test_large_arbitrary_blob_rejected_but_source_and_reviewed_config_allowed():
    assert reason("misc/raw.dat", 3 * 1024 * 1024)
    assert reason("softcut_solution/comparison.py", 30000) is None
    assert reason("configs/comparison.example.json", 1000) is None
