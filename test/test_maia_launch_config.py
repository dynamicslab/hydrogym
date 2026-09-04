"""Unit tests for MAIA MPMD launch-config validation.

Verification Addendum Finding B (HYDROGYM_ENGINEERING_AUDIT_v2.md): the
original audit's Finding 2.5 flagged that MAIA had zero config surface or
mismatch detection for the MPMD launch's rank count, unlike Nek's
`mpi_split` (which validates `nproc` against the actual MPI world size).
`MaiaInterface.init_comm` now accepts an optional `nproc` and validates it
against the actual number of remote (m-AIA) ranks when given.

`split_comm_by_appnum` itself needs a real multi-rank MPMD world to behave
meaningfully (see `test/mpmd_smoke_split.py` and `test_core_external.py`'s
`TestSplitCommByAppnumSingleApp`), so it is monkeypatched here to keep this
test single-rank-pytest-safe, matching the existing idiom in
`test_core_external.py::TestExternalProcessEnvMixin`.
"""

import pytest

pytest.importorskip("mpi4py")

from hydrogym.maia import mpmd_interface  # noqa: E402


class _FakeComm:
    def __init__(self, world_size):
        self._world_size = world_size

    def Get_size(self):
        return self._world_size


def _patch_split(monkeypatch, app_no_ranks):
    """Make split_comm_by_appnum return a fixed controller-side rank count,
    matching the tuple shape `MaiaInterface.init_comm` unpacks."""

    def fake_split(comm_world):
        return (0, "APP_COMM", 0, app_no_ranks, "APP_GROUP", 0, 1)

    monkeypatch.setattr(mpmd_interface, "split_comm_by_appnum", fake_split)


class TestMaiaNprocValidation:
    def test_none_skips_validation(self, monkeypatch):
        # Default behavior (no nproc given) must be unchanged: no error,
        # regardless of actual world size.
        _patch_split(monkeypatch, app_no_ranks=1)
        iface = mpmd_interface.MaiaInterface(nDim=2)
        iface.init_comm(_FakeComm(world_size=99), nproc=None)
        assert iface.remoteRoot == 1

    def test_matching_nproc_passes(self, monkeypatch):
        # 1 controller rank + 4 solver ranks = 5 world ranks.
        _patch_split(monkeypatch, app_no_ranks=1)
        iface = mpmd_interface.MaiaInterface(nDim=2)
        iface.init_comm(_FakeComm(world_size=5), nproc=4)
        assert iface.appNoRanks == 1

    def test_mismatched_nproc_raises_actionable_error(self, monkeypatch):
        # Caller expected 4 solver ranks but only 2 are actually present.
        _patch_split(monkeypatch, app_no_ranks=1)
        iface = mpmd_interface.MaiaInterface(nDim=2)
        with pytest.raises(RuntimeError, match=r"expected 4 .* got 2"):
            iface.init_comm(_FakeComm(world_size=3), nproc=4)

    def test_mismatch_error_names_the_fix(self, monkeypatch):
        _patch_split(monkeypatch, app_no_ranks=1)
        iface = mpmd_interface.MaiaInterface(nDim=2)
        with pytest.raises(RuntimeError, match=r"mpirun -n 1 python \.\.\. : -n 4 maia"):
            iface.init_comm(_FakeComm(world_size=3), nproc=4)

    def test_multi_controller_rank_arithmetic(self, monkeypatch):
        # 3 controller ranks + 7 solver ranks = 10 world ranks -- the
        # "any controller rank count" MPMD case this backend is designed
        # for (unlike Nek's rank-0-only protocol).
        _patch_split(monkeypatch, app_no_ranks=3)
        iface = mpmd_interface.MaiaInterface(nDim=2)
        iface.init_comm(_FakeComm(world_size=10), nproc=7)  # no raise
        with pytest.raises(RuntimeError, match=r"expected 8 .* got 7"):
            iface.init_comm(_FakeComm(world_size=10), nproc=8)
