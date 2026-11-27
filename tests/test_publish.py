"""Publication guard prevents research assets leaking into source history."""
import io
import subprocess

from tools import check_publish
from tools.check_publish import reason
import pytest


@pytest.mark.parametrize("path", [
    "projects/softmodeling/outputs/a/result.json",
    "projects/rec_inc_soft_cut/data/movie.mp4",
    "projects/rec_inc_soft_cut/models/checkpoint.pt",
    "projects/softmodeling/docs/experiment.csv",
    "settings.local.json", ".env.secret", "provenance/video-original.bundle",
    "misc/parameters.key", "another/.runtime/module.py",
    "researchtopic/bladetype.pptx", "researchtopic/microscope.ppt",
])
def test_local_assets_rejected(path):
    assert reason(path, 100)


def test_large_arbitrary_blob_rejected_but_source_and_reviewed_config_allowed():
    assert reason("misc/raw.dat", 3 * 1024 * 1024)
    assert reason("softcut_solution/comparison.py", 30000) is None
    assert reason("configs/comparison.example.json", 1000) is None


@pytest.fixture
def publication_repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for key, value in (("user.name", "Publication test"),
                       ("user.email", "publication-test@example.invalid"),
                       ("commit.gpgsign", "false")):
        subprocess.run(["git", "config", key, value], cwd=tmp_path, check=True)
    monkeypatch.setattr(check_publish, "ROOT", tmp_path)
    return tmp_path


def publication_commit(repo, files=None, delete=()):
    for name, content in (files or {}).items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for name in delete:
        (repo / name).unlink()
    subprocess.run(["git", "add", "--all"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "Publication fixture"], cwd=repo, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def publication_push(monkeypatch, local, remote):
    monkeypatch.setattr(check_publish.sys, "stdin", io.StringIO(
        f"refs/heads/research {local} refs/heads/research {remote}\n"))
    return check_publish.main(["--pre-push"])


def test_published_debt_can_be_removed_without_weakening_full_history_audit(publication_repo, monkeypatch):
    remote = publication_commit(publication_repo, {"legacy.csv": "depth,force\n2,30\n"})
    local = publication_commit(publication_repo, {"README.md": "Summary retained in text.\n"},
                               delete=("legacy.csv",))
    assert publication_push(monkeypatch, local, remote) == 0
    assert check_publish.main(["--ref", local]) == 1


def test_pre_push_still_rejects_debt_left_in_current_tree(publication_repo, monkeypatch):
    remote = publication_commit(publication_repo, {"legacy.csv": "depth,force\n2,30\n"})
    local = publication_commit(publication_repo, {"README.md": "Documentation only.\n"})
    assert publication_push(monkeypatch, local, remote) == 1


@pytest.mark.parametrize("new_path,new_contents", [
    ("new.csv", "new experimental data\n"),
    ("copy.csv", "already published\n"),
    ("legacy.csv", "changed content\n"),
])
def test_new_path_or_blob_cannot_be_hidden_by_later_deletion(
        publication_repo, monkeypatch, new_path, new_contents):
    remote = publication_commit(publication_repo, {"legacy.csv": "already published\n"})
    publication_commit(publication_repo, {"README.md": "Source only.\n"}, delete=("legacy.csv",))
    publication_commit(publication_repo, {new_path: new_contents})
    local = publication_commit(publication_repo, delete=(new_path,))
    assert publication_push(monkeypatch, local, remote) == 1


def test_new_destination_has_no_implicit_published_baseline(publication_repo, monkeypatch):
    publication_commit(publication_repo, {"legacy.csv": "published elsewhere\n"})
    local = publication_commit(publication_repo, {"README.md": "Source only.\n"},
                               delete=("legacy.csv",))
    assert publication_push(monkeypatch, local, "0" * 40) == 1


def test_new_clean_destination_passes_full_history_check(publication_repo, monkeypatch):
    local = publication_commit(publication_repo, {"README.md": "Source only.\n"})
    assert publication_push(monkeypatch, local, "0" * 40) == 0


def test_missing_remote_object_fails_closed(publication_repo, monkeypatch, capsys):
    local = publication_commit(publication_repo, {"README.md": "Source only.\n"})
    assert publication_push(monkeypatch, local, "f" * 40) == 1
    assert "fetch and review" in capsys.readouterr().out


def test_nonancestor_remote_is_not_replaced_by_merge_base(publication_repo, monkeypatch, capsys):
    local = publication_commit(publication_repo, {"README.md": "Source only.\n"})
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"],
                                   cwd=publication_repo, text=True).strip()
    remote = subprocess.check_output(["git", "commit-tree", tree, "-m", "Unrelated remote history"],
                                     cwd=publication_repo, text=True).strip()
    assert publication_push(monkeypatch, local, remote) == 1
    assert "not a verified ancestor" in capsys.readouterr().out


def test_deleting_a_ref_does_not_republish_its_history(publication_repo, monkeypatch):
    remote = publication_commit(publication_repo, {"legacy.csv": "already published\n"})
    assert publication_push(monkeypatch, "0" * 40, remote) == 0


def test_each_destination_uses_its_own_remote_baseline(publication_repo, monkeypatch):
    remote = publication_commit(publication_repo, {"legacy.csv": "already published\n"})
    local = publication_commit(publication_repo, {"README.md": "Source only.\n"},
                               delete=("legacy.csv",))
    monkeypatch.setattr(check_publish.sys, "stdin", io.StringIO(
        f"refs/heads/old {local} refs/heads/old {remote}\n"
        f"refs/heads/new {local} refs/heads/new {'0' * 40}\n"))
    assert check_publish.main(["--pre-push"]) == 1


def test_staged_mode_still_rejects_new_data(publication_repo):
    publication_commit(publication_repo, {"README.md": "Source only.\n"})
    (publication_repo / "new.csv").write_text("new data\n")
    subprocess.run(["git", "add", "new.csv"], cwd=publication_repo, check=True)
    assert check_publish.main(["--staged"]) == 1


def test_pre_push_rejects_non_object_arguments(publication_repo, monkeypatch):
    local = publication_commit(publication_repo, {"README.md": "Source only.\n"})
    assert publication_push(monkeypatch, local, "--all") == 1
