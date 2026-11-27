"""The policy loader refuses a pickle whose sha256 differs from MANIFEST.yaml (a pickle can run code)."""

import importlib.util
import shutil
from pathlib import Path

import pytest

POLICIES = Path(__file__).resolve().parents[1] / "policies"


def _loader():
    spec = importlib.util.spec_from_file_location("load_policy_under_test", POLICIES / "load_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_listed_pickle_with_other_bytes_is_refused(tmp_path):
    pytest.importorskip("yaml")
    L = _loader()
    pid = "cylinder_2d_re200/ppo/seed1"
    e = L.entry(pid)
    rel = next(f["path"] for f in e["files"] if f["path"].endswith(".pkl"))
    src = POLICIES / rel
    if not src.exists():
        pytest.skip("policy file not present")
    dst = tmp_path / rel
    dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)
    L.HERE = str(tmp_path)  # the loader resolves the MANIFEST path of a file relative to HERE
    L._unpickle(str(dst))  # unchanged copy: accepted
    dst.write_bytes(dst.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="sha256 differs"):
        L._unpickle(str(dst))


def test_unlisted_pickle_is_loaded(tmp_path):
    L = _loader()
    import pickle

    p = tmp_path / "own.pkl"
    p.write_bytes(pickle.dumps({"a": 1}))
    L.HERE = str(tmp_path)
    assert L._unpickle(str(p)) == {"a": 1}
