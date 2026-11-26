"""The cache symlink of an environment is made atomically: several processes that prepare the same environment at
once (same cache directory) must not fail with FileExistsError."""

import multiprocessing as mp
import os

import pytest

pytest.importorskip("huggingface_hub")

from hydrogym.data_manager import HFDataManager  # noqa: E402


def _make_link(args):
    target, link, n = args
    for _ in range(n):
        HFDataManager._atomic_symlink(target, link)
    return True


def test_atomic_symlink_replaces_existing_link(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    link = str(tmp_path / "env")
    HFDataManager._atomic_symlink(str(a), link)
    HFDataManager._atomic_symlink(str(b), link)
    assert os.path.realpath(link) == str(b)
    assert not [p for p in os.listdir(tmp_path) if ".tmp-" in p]


def test_atomic_symlink_concurrent(tmp_path):
    target = tmp_path / "snapshot" / "Env"
    target.mkdir(parents=True)
    link = str(tmp_path / "cache" / "Env")
    os.makedirs(os.path.dirname(link))
    with mp.get_context("spawn").Pool(4) as pool:
        assert all(pool.map(_make_link, [(str(target), link, 50)] * 4))
    assert os.path.realpath(link) == str(target)
    assert os.listdir(os.path.dirname(link)) == ["Env"]


def test_remove_link_tolerates_missing(tmp_path):
    HFDataManager._remove_link(str(tmp_path / "gone"))
