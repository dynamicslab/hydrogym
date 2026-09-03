"""Unit tests for audit Task 2.5: `cache_dir` forwarded to `snapshot_download`.

Before this change, none of HFDataManager's three `snapshot_download()` call
sites passed `cache_dir`, so raw HF downloads always landed in the
huggingface_hub default cache (~/.cache/huggingface) regardless of the
`cache_dir` the user configured -- a problem on quota-limited HPC home dirs.

Deliberate deviation from the audit's literal spec ("pass
`cache_dir=self.cache_dir`"): `self.cache_dir` is never None (it defaults to
~/.cache/hydrogym), so forwarding it unconditionally would move EVERY user's
HF download cache and orphan existing downloads. Instead only an
explicitly-provided `cache_dir` is forwarded; the default case preserves the
previous download location exactly. The clean-cache symlink/copy layer
(``self.cache_dir``) is untouched in both cases.

All three caching strategies (symlink / copy / direct) are exercised through
the real `get_environment_path` dispatch with `snapshot_download` mocked --
no network, no real HF cache.

pytest.importorskip guards: hydrogym.data_manager imports huggingface_hub.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

from hydrogym import data_manager as dm_mod  # noqa: E402
from hydrogym.data_manager import HFDataManager  # noqa: E402

ENV_NAME = "TestEnv"


@pytest.fixture
def fake_hf_snapshot(tmp_path, monkeypatch):
    """Fake snapshot_download: records kwargs and materializes a snapshot dir
    under the cache_dir it was given (mirroring the real function's cache
    layout), holding one environment (JAXFLUIDS profile: no required files,
    sentinel only)."""
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        root = kwargs.get("cache_dir") or (tmp_path / "hf_default_cache")
        snapshot_dir = Path(root) / "datasets--t--r" / "snapshots" / "abc123"
        env_dir = snapshot_dir / ENV_NAME
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / ".JAXFLUIDS").write_text("")
        return str(snapshot_dir)

    monkeypatch.setattr(dm_mod, "snapshot_download", fake_snapshot_download)
    # Skip _detect_solver_profile's HF file-listing (network); the profile
    # under test is pinned to JAXFLUIDS, whose validation is a no-op.
    monkeypatch.setattr(HFDataManager, "_detect_solver_profile", lambda self, env: "JAXFLUIDS")
    return calls


class TestExplicitCacheDirForwarded:
    @pytest.mark.parametrize("use_clean_cache", [True, "copy", False])
    def test_all_three_strategies_forward_cache_dir(self, fake_hf_snapshot, tmp_path, use_clean_cache):
        clean_cache = tmp_path / "clean_cache"
        dm = HFDataManager(
            repo_id="t/r",
            cache_dir=str(clean_cache),
            use_clean_cache=use_clean_cache,
            fallback_profile="JAXFLUIDS",
        )
        path = dm.get_environment_path(ENV_NAME, force_download=True)

        assert len(fake_hf_snapshot) == 1
        assert fake_hf_snapshot[0]["cache_dir"] == str(clean_cache)
        assert os.path.exists(path)  # env usable whichever strategy ran

    def test_forwarded_dir_receives_download(self, fake_hf_snapshot, tmp_path):
        """End-to-end: the snapshot path the mock 'downloaded to' sits under
        the configured cache_dir (what a user on a quota-limited home dir
        would point at scratch)."""
        scratch = tmp_path / "scratch"
        dm = HFDataManager(
            repo_id="t/r",
            cache_dir=str(scratch),
            use_clean_cache=False,
            fallback_profile="JAXFLUIDS",
        )
        path = dm.get_environment_path(ENV_NAME, force_download=True)
        assert os.path.realpath(path).startswith(str(scratch) + os.sep)


class TestDefaultBehaviorUnchanged:
    def test_no_cache_dir_passed_keeps_hf_default(self, fake_hf_snapshot, tmp_path):
        """Omitting cache_dir must preserve the pre-change behavior: no
        cache_dir kwarg forwarded, so snapshot_download uses the
        huggingface_hub default cache. (use_clean_cache=False avoids creating
        the real ~/.cache/hydrogym from __init__.)"""
        dm = HFDataManager(repo_id="t/r", use_clean_cache=False, fallback_profile="JAXFLUIDS")
        dm.get_environment_path(ENV_NAME, force_download=True)

        assert fake_hf_snapshot[0]["cache_dir"] is None
