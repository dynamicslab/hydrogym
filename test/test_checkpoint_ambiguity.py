"""Unit tests for deterministic checkpoint selection in `FlowConfig`.

Checkpoint selection is deterministic: the last file by sorted name.

Called on a bare `Cylinder` instance (`object.__new__`) with
`hydrogym.data_manager.HFDataManager` monkeypatched to point at a local
temp directory -- no firedrake mesh construction, no network access.
"""

import logging

import pytest

pytest.importorskip("firedrake")

from hydrogym.firedrake.envs.cylinder.flow import Cylinder  # noqa: E402


@pytest.fixture
def flow():
    return object.__new__(Cylinder)


def _patch_env_path(monkeypatch, path):
    import hydrogym.data_manager as data_manager_mod

    class _FakeDataManager:
        def __init__(self, *args, **kwargs):
            pass

        def get_environment_path(self, checkpoint):
            return str(path)

    monkeypatch.setattr(data_manager_mod, "HFDataManager", _FakeDataManager)


class TestCheckpointAmbiguity:
    def test_single_file_resolves_as_before(self, flow, tmp_path, monkeypatch):
        (tmp_path / "cylinder_00000100.ckpt").touch()
        _patch_env_path(monkeypatch, tmp_path)

        resolved = flow._resolve_single_checkpoint("SomeEnv", silent=True)
        assert resolved is not None
        assert resolved.endswith("cylinder_00000100.ckpt")

    def test_multiple_files_auto_inferred_picks_deterministically(self, flow, tmp_path, monkeypatch, caplog):
        for i in (100, 200, 300):
            (tmp_path / f"cylinder_{i:08d}.ckpt").touch()
        _patch_env_path(monkeypatch, tmp_path)

        with caplog.at_level(logging.INFO):
            resolved = flow._resolve_single_checkpoint("AutoInferredEnv", silent=True)
        # Auto-inferred restarts resolve to a real checkpoint when candidates exist.
        assert resolved is not None
        assert resolved.endswith("cylinder_00000300.ckpt")

    def test_multiple_files_explicit_restart_picks_deterministically(self, flow, tmp_path, monkeypatch):
        for i in (100, 300, 200):
            (tmp_path / f"cylinder_{i:08d}.ckpt").touch()
        _patch_env_path(monkeypatch, tmp_path)

        resolved = flow._resolve_single_checkpoint("ExplicitEnvName", silent=False)
        # Deterministic: last by sorted filename, regardless of glob/creation order.
        assert resolved.endswith("cylinder_00000300.ckpt")

    def test_multiple_files_explicit_restart_stable_across_repeats(self, flow, tmp_path, monkeypatch):
        for i in (500, 100, 900, 300):
            (tmp_path / f"cylinder_{i:08d}.ckpt").touch()
        _patch_env_path(monkeypatch, tmp_path)

        results = {flow._resolve_single_checkpoint("ExplicitEnvName", silent=False) for _ in range(5)}
        assert len(results) == 1
        assert next(iter(results)).endswith("cylinder_00000900.ckpt")

    def test_no_files_still_returns_none(self, flow, tmp_path, monkeypatch):
        _patch_env_path(monkeypatch, tmp_path)
        assert flow._resolve_single_checkpoint("EmptyEnv", silent=True) is None
