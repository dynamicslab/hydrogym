"""Application number of the MAIA MPMD split: MAIA_MPMD_APPNUM (srun --multi-prog launches) takes precedence over
MPI_APPNUM (mpirun MPMD launches); invalid or missing values give clear errors."""

import pytest

pytest.importorskip("mpi4py")

from mpi4py import MPI  # noqa: E402

from hydrogym import core_external  # noqa: E402
from hydrogym.core_external import MPMD_APPNUM_ENV, mpmd_appnum  # noqa: E402


class _Comm:
    """communicator stub: Get_attr(MPI.APPNUM) returns the launcher's application number"""

    def __init__(self, appnum):
        self._appnum = appnum
        self.queried = False

    def Get_attr(self, key):
        assert key == MPI.APPNUM
        self.queried = True
        return self._appnum


@pytest.mark.parametrize("launcher", [0, 1])
def test_mpi_appnum_without_environment_variable(launcher):
    assert mpmd_appnum(_Comm(launcher), environ={}) == launcher


@pytest.mark.parametrize("value, expected", [("0", 0), ("1", 1), (" 1 ", 1)])
def test_environment_variable_takes_precedence(value, expected):
    comm = _Comm(0)  # srun --multi-prog: every rank sees the same MPI_APPNUM
    assert mpmd_appnum(comm, environ={MPMD_APPNUM_ENV: value}) == expected
    assert not comm.queried


def test_environment_variable_when_launcher_sets_none():
    assert mpmd_appnum(_Comm(None), environ={MPMD_APPNUM_ENV: "1"}) == 1


def test_empty_environment_variable_falls_back_to_mpi_appnum():
    assert mpmd_appnum(_Comm(1), environ={MPMD_APPNUM_ENV: ""}) == 1


@pytest.mark.parametrize("value", ["2", "-1", "maia", "0.5"])
def test_invalid_environment_variable(value):
    with pytest.raises(ValueError, match=MPMD_APPNUM_ENV):
        mpmd_appnum(_Comm(0), environ={MPMD_APPNUM_ENV: value})


def test_no_application_number_at_all():
    with pytest.raises(RuntimeError, match=MPMD_APPNUM_ENV):
        mpmd_appnum(_Comm(None), environ={})


def test_default_environment_is_os_environ(monkeypatch):
    monkeypatch.setenv(MPMD_APPNUM_ENV, "1")
    assert mpmd_appnum(_Comm(0)) == 1
    monkeypatch.delenv(MPMD_APPNUM_ENV)
    assert mpmd_appnum(_Comm(0)) == 0


def test_split_uses_mpmd_appnum(monkeypatch):
    calls = []

    def fake(comm, environ=None):
        calls.append(comm)
        raise RuntimeError("stop")

    monkeypatch.setattr(core_external, "mpmd_appnum", fake)
    with pytest.raises(RuntimeError, match="stop"):
        core_external.split_comm_by_appnum("WORLD")
    assert calls == ["WORLD"]
