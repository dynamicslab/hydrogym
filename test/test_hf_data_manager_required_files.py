"""Unit tests: solver-profile requirements and the check before a download.

- The NEK5000_v17 profile accepts the file set of the Nek5000 wing folder NACA0012_3D_Re200000_AOA0 on the Hub
  (no ``*.ma2``, no ``*.sch``).
- A folder that lacks a required file fails from the HF file listing, before ``snapshot_download`` starts.
- When the listing is unavailable the download goes ahead as before.

``snapshot_download`` and ``HfApi`` are mocked: no network, no HF cache.
"""

from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

from hydrogym import data_manager as dm_mod  # noqa: E402
from hydrogym.data_manager import HFDataManager, MissingEnvironmentFilesError  # noqa: E402

WING = "NACA0012_3D_Re200000_AOA0"
# The files of the wing folder on the Hub (restart fields shortened to two)
WING_FILES = [
    ".NEK5000_v17",
    "environment_config.yaml",
    "forparam.i",
    "mask_naca_wing0.f00002",
    "naca_wing.map",
    "naca_wing.re2",
    "naca_wing.rea",
    "naca_wing.restart",
    "naca_wing.wall",
    "restart_files/init_1/rs8naca_wing0.f00001",
    "restart_files/init_1/rs8naca_wing0.f00002",
    "stat_pts.in",
]


def _materialize(root: Path, files):
    for f in files:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """A Hub holding the wing folder with ``state["files"]``; records snapshot_download calls."""
    state = {"files": list(WING_FILES), "listing_fails": False}
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        snapshot = tmp_path / "hf" / "snapshots" / "abc"
        _materialize(snapshot / WING, state["files"])
        return str(snapshot)

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def list_repo_files(self, repo_id, repo_type=None, revision=None):
            if state["listing_fails"]:
                raise ConnectionError("offline")
            return [f"{WING}/{f}" for f in state["files"]] + ["Other_2D/.MAIA_LB"]

    monkeypatch.setattr(dm_mod, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(dm_mod, "HfApi", FakeApi)
    monkeypatch.setattr(HFDataManager, "_detect_solver_profile", lambda self, env: "NEK5000_v17")
    return state, calls


def _dm(tmp_path, use_clean_cache):
    return HFDataManager(repo_id="t/r", cache_dir=str(tmp_path / "clean"), use_clean_cache=use_clean_cache)


def test_wing_file_set_validates(tmp_path):
    env = tmp_path / WING
    _materialize(env, WING_FILES)
    assert HFDataManager(cache_dir=str(tmp_path / "c"))._validate_environment_files(str(env), "NEK5000_v17")


def test_wing_workspace_links_without_ma2_sch(tmp_path):
    env = tmp_path / WING
    _materialize(env, WING_FILES + ["extra.ma2"])
    work = tmp_path / "run"
    HFDataManager(cache_dir=str(tmp_path / "c")).prepare_working_directory(str(env), str(work), "NEK5000_v17")
    for f in ("naca_wing.re2", "naca_wing.map", "mask_naca_wing0.f00002", "stat_pts.in", "forparam.i", "extra.ma2"):
        assert (work / f).is_symlink(), f
    assert (work / "restart_files").is_symlink()


@pytest.mark.parametrize("use_clean_cache", [True, "copy", False])
def test_wing_download_passes(fake_hub, tmp_path, use_clean_cache):
    _, calls = fake_hub
    path = _dm(tmp_path, use_clean_cache).get_environment_path(WING)
    assert len(calls) == 1
    assert (Path(path) / "naca_wing.re2").exists()


@pytest.mark.parametrize("use_clean_cache", [True, "copy", False])
@pytest.mark.parametrize("drop", ["stat_pts.in", "mask_naca_wing0.f00002", "restart_files"])
def test_missing_required_fails_before_download(fake_hub, tmp_path, use_clean_cache, drop):
    state, calls = fake_hub
    state["files"] = [f for f in WING_FILES if not f.startswith(drop)]
    with pytest.raises(MissingEnvironmentFilesError, match=drop.split("_")[0]):
        _dm(tmp_path, use_clean_cache).get_environment_path(WING)
    assert calls == []  # no download started


def test_missing_required_uses_local_fallback(fake_hub, tmp_path):
    state, calls = fake_hub
    state["files"] = [f for f in WING_FILES if f != "stat_pts.in"]
    fallback = tmp_path / "local"
    (fallback / WING).mkdir(parents=True)  # present, but incomplete: the final fallback returns it as before
    dm = HFDataManager(cache_dir=str(tmp_path / "clean"), local_fallback_dir=str(fallback))
    assert dm.get_environment_path(WING) == str(fallback / WING)
    assert calls == []


def test_listing_unavailable_downloads_as_before(fake_hub, tmp_path):
    state, calls = fake_hub
    state["listing_fails"] = True
    _dm(tmp_path, True).get_environment_path(WING)
    assert len(calls) == 1


def test_wing_workspace_never_links_sch(tmp_path):
    env = tmp_path / WING
    _materialize(env, WING_FILES + ["naca_wing.sch"])
    work = tmp_path / "run"
    HFDataManager(cache_dir=str(tmp_path / "c")).prepare_working_directory(str(env), str(work), "NEK5000_v17")
    assert not (work / "naca_wing.sch").exists()
