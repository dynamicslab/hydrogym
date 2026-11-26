"""Unit tests for `hydrogym.core_external`.

Covers the single-rank-testable surface (error paths, mixin delegation,
re-export identity, backend wiring). The multi-rank behavior of both split
strategies is validated live in the MPMD smoke script
``test/mpmd_smoke_split.py`` (run under mpirun; see its docstring).
"""

import pytest

pytest.importorskip("mpi4py")

from hydrogym import core_external  # noqa: E402
from hydrogym.core_external import ExternalProcessEnvMixin, mpi_split  # noqa: E402


class TestMpiSplitErrorPaths:
    def test_single_rank_world_raises(self):
        # Single-rank MPI world (plain pytest): the < 2 guard must fire
        # with the MPMD launch hint before anything else.
        with pytest.raises(RuntimeError, match="world size must be >= 2"):
            mpi_split(core_external.MPI.COMM_WORLD)

    def test_single_rank_world_hint_names_mpmd_launch(self):
        with pytest.raises(RuntimeError, match="mpirun -n 1 python"):
            mpi_split(core_external.MPI.COMM_WORLD)


class TestExternalProcessEnvMixin:
    def _host(self, **attrs):
        class _Host(ExternalProcessEnvMixin):
            pass

        for k, v in attrs.items():
            setattr(_Host, k, v)
        return _Host()

    def test_default_knobs(self):
        host = self._host()
        assert host.CONTROLLER_RANK == 0
        assert host.INTERCOMM_TAG == 99
        assert host.MPI_SPLIT_LOG_PREFIX == "[MPI_SPLIT] "

    def test_split_delegates_with_default_knobs(self, monkeypatch):
        calls = {}

        def fake_split(comm, nproc=None, controller_rank=0, intercomm_tag=99, log_prefix=""):
            calls.update(
                comm=comm,
                nproc=nproc,
                controller_rank=controller_rank,
                intercomm_tag=intercomm_tag,
                log_prefix=log_prefix,
            )
            return "SENTINEL"

        monkeypatch.setattr(core_external, "mpi_split", fake_split)
        host = self._host()
        assert host._split_mpmd_comm("COMM", nproc=4) == "SENTINEL"
        assert calls == {
            "comm": "COMM",
            "nproc": 4,
            "controller_rank": 0,
            "intercomm_tag": 99,
            "log_prefix": "[MPI_SPLIT] ",
        }

    def test_split_delegates_with_overridden_knobs(self, monkeypatch):
        calls = {}

        def fake_split(comm, nproc=None, controller_rank=0, intercomm_tag=99, log_prefix=""):
            calls.update(controller_rank=controller_rank, intercomm_tag=intercomm_tag, log_prefix=log_prefix)
            return "SENTINEL"

        monkeypatch.setattr(core_external, "mpi_split", fake_split)

        class _Host(ExternalProcessEnvMixin):
            CONTROLLER_RANK = 2
            INTERCOMM_TAG = 123
            MPI_SPLIT_LOG_PREFIX = "[X] "

        assert _Host()._split_mpmd_comm("COMM") == "SENTINEL"
        assert calls == {"controller_rank": 2, "intercomm_tag": 123, "log_prefix": "[X] "}


class TestBackendWiring:
    def test_nek_env_uses_mixin(self):
        mod = pytest.importorskip("hydrogym.nek.env")
        assert issubclass(mod.NekEnv, ExternalProcessEnvMixin)
        # The split goes through the mixin extension point
        assert "_split_mpmd_comm" not in vars(mod.NekEnv), "NekEnv overrides the mixin split"
        src = open(mod.__file__).read()
        assert "self.sub_comm = self._split_mpmd_comm(" in src, "NekEnv must delegate to the mixin split"
        assert "remote_leader" not in src, "split handshake should live only in core_external"

    def test_nek_mpi_split_reexport_identity(self):
        # hydrogym.nek.mpi_split resolves to the shared
        # implementation (public API + monkeypatch idiom).
        nek_init = pytest.importorskip("hydrogym.nek")
        assert nek_init.mpi_split is mpi_split

    def test_maia_init_comm_delegates(self):
        mod = pytest.importorskip("hydrogym.maia.mpmd_interface")
        src = open(mod.__file__).read()
        assert "split_comm_by_appnum(comm_world)" in src, "init_comm must delegate the split math"
        # The split math itself lives only in core_external
        assert "Allreduce" not in src, "Allreduce-based root discovery should live only in core_external"
        assert "Translate_ranks" not in src
        assert "Split(" not in src.replace("split_comm_by_appnum(", ""), "no local Split calls should remain"
        # Attribute contract (consumed by env_core / MaiaInterface)
        for attr in (
            "worldComm",
            "appComm",
            "appRank",
            "appNoRanks",
            "appGroup",
            "appnum",
            "appRootInWorld",
            "remoteRoot",
        ):
            assert f"self.{attr}" in src

    def test_maia_split_helper_is_core_external(self):
        from hydrogym.maia import mpmd_interface

        assert mpmd_interface.split_comm_by_appnum is core_external.split_comm_by_appnum


class TestIntercommSetup:
    """Inter-communicator setup: correct remote leader and remote root for any app layout
    (the MPMD smoke script runs both sides in Python)."""

    def test_mpi_split_remote_leader_not_hardcoded_one(self):
        # The remote leader is the other app's leader (1 - color), not a
        # fixed rank, so either side can be the Python process.
        src = open(core_external.__file__).read()
        assert "remote_leader=1 - color" in src
        assert "remote_leader=1," not in src

    def test_appnum_root_translation_direction(self):
        # The app root (app rank 0) is translated to its world rank.
        src = open(core_external.__file__).read()
        assert "app_group.Translate_ranks([app_root], group_world)" in src
        assert "group_world.Translate_ranks([app_root], app_group)" not in src


class TestSplitCommByAppnumSingleApp:
    """split_comm_by_appnum on a non-MPMD single-rank world: APPNUM is None
    (or 0 depending on MPI implementation), so full validation happens in
    the MPMD smoke script; here we only pin that the helper is exported and
    the module imports cleanly without solver binaries."""

    def test_helper_importable_without_solver(self):
        assert callable(core_external.split_comm_by_appnum)
