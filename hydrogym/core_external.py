"""Shared MPMD communicator setup for external-process backends.

The Nek and MAIA backends both talk to their CFD solver as a separate MPI
application launched in the same MPMD world
(``mpirun -n 1 python ... : -n N ./solver``). This module provides the two world-split strategies:

  - ``mpi_split`` (used by Nek) -- rank-color split (rank 0 = controller,
    ranks 1+ = workers) followed by ``Create_intercomm``; validates the
    world size against the expected worker count.
  - ``split_comm_by_appnum`` (used by MAIA) -- APPNUM-based split (true
    MPMD, any controller rank count), with group translation + Allreduce
    discovery of the remote application's root rank.

``ExternalProcessEnvMixin`` is the common extension point carrying the
protocol knobs (controller rank, intercomm tag, log prefix). The
per-solver tag protocol (the actual command/data messages) lives in each
backend.

Callers:
  - NekEnv (mixin ``_split_mpmd_comm``); ``hydrogym.nek.env.mpi_split`` is
    re-exported from here, so ``hydrogym.nek.mpi_split`` remains available.
  - ``MaiaInterface.init_comm`` delegates its communicator setup to
    ``split_comm_by_appnum`` and assigns the results.
"""

import os
from typing import Mapping, Optional, Tuple

import numpy as np

# mpi4py is an optional backend dependency (pyproject: maia/nek extras), and
# importing it has the side effect of initializing MPI -- which the package's
# own __init__ deliberately avoids doing eagerly (MPMD-mode safety). Guard it
# so the module (and the developer-template external skeleton that builds on
# it) is importable in a plain Python environment; the split functions raise
# a clear error if actually called without it.
try:
    from mpi4py import MPI

    MPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without mpi4py
    MPI = None
    MPI_AVAILABLE = False


def _require_mpi():
    """Raise a clear ImportError if mpi4py is unavailable.

    Called at the top of every public entry point in this module so the
    no-mpi4py path fails loudly at call time instead of at import time
    (the module itself imports fine without mpi4py).
    """
    if not MPI_AVAILABLE:
        raise ImportError(
            "mpi4py is required for MPMD communicator splitting but is not installed. "
            "Install it with e.g. `pip install mpi4py` (see the maia/nek extras in pyproject.toml)."
        )


def mpi_split(
    comm_world: "MPI.Comm",
    nproc: Optional[int] = None,
    controller_rank: int = 0,
    intercomm_tag: int = 99,
    log_prefix: str = "[MPI_SPLIT] ",
) -> "MPI.Comm":
    """Split the MPMD world into a controller/worker inter-communicator
    (Nek protocol: rank 0 is the controller, ranks 1+ are the workers).

    Args:
      comm_world: MPI communicator (typically MPI.COMM_WORLD)
      nproc: Expected number of solver workers (for validation)
      controller_rank: World rank of the controller (default 0)
      intercomm_tag: MPI tag for the inter-communicator handshake
      log_prefix: Prefix for the split log lines

    Returns:
      Inter-communicator between controller and workers
    """
    _require_mpi()
    mpi_rank = comm_world.Get_rank()
    mpi_size = comm_world.Get_size()

    if mpi_size < 2:
        raise RuntimeError(
            "MPI world size must be >= 2 to create the Nek inter-communicator. "
            "Launch with MPMD, e.g. `mpirun -n 1 python ... : -n N ./nek5000`, "
            "so rank 0 can connect to the Nek worker ranks."
        )

    # Validate MPI size matches nproc
    if nproc is not None:
        expected_size = 1 + nproc  # 1 controller + N workers
        if mpi_size != expected_size:
            raise RuntimeError(
                f"MPI world size mismatch: expected {expected_size} "
                f"(1 controller + {nproc} workers), got {mpi_size}. "
                f"Launch with: mpirun -n 1 python ... : -n {nproc} ./nek5000"
            )

    color = 0 if mpi_rank == controller_rank else 1

    local_comm = comm_world.Split(color, mpi_rank)
    print(
        f"{log_prefix}World rank {mpi_rank}, color {color}, "
        f"local_comm size: {local_comm.Get_size()}, "
        f"local rank: {local_comm.Get_rank()}",
        flush=True,
    )

    # remote_leader = 1 - color: each side contacts the other application's
    # leader (world rank 0 or 1).
    sub_comm = local_comm.Create_intercomm(
        local_leader=0, peer_comm=MPI.COMM_WORLD, remote_leader=1 - color, tag=intercomm_tag
    )
    print(
        f"{log_prefix}Inter-comm created: local_size={sub_comm.Get_size()}, remote_size={sub_comm.Get_remote_size()}",
        flush=True,
    )
    return sub_comm


MPMD_APPNUM_ENV = "MAIA_MPMD_APPNUM"


def mpmd_appnum(comm_world: "MPI.Comm", environ: Optional[Mapping[str, str]] = None) -> int:
    """Application number of this process in the MAIA MPMD world: 0 = Python controller, 1 = m-AIA.

    ``MAIA_MPMD_APPNUM`` (environment variable) takes precedence when it is set. Launchers that start every rank as
    one application, e.g. ``srun --multi-prog`` (Slurm, ParaStation MPI), give all ranks the same (or no)
    ``MPI_APPNUM``; set ``MAIA_MPMD_APPNUM=0`` for the Python ranks and ``MAIA_MPMD_APPNUM=1`` for the m-AIA ranks
    there (m-AIA reads the same variable). Otherwise ``MPI_APPNUM`` of the MPMD launcher is used
    (``mpirun -np 1 python ... : -np N maia properties_run.toml``).

    Args:
      comm_world: MPI communicator, typically MPI.COMM_WORLD.
      environ: environment mapping (default ``os.environ``).

    Raises:
      ValueError: ``MAIA_MPMD_APPNUM`` is set but not 0 or 1.
      RuntimeError: neither ``MAIA_MPMD_APPNUM`` nor ``MPI_APPNUM`` is available (not an MPMD launch).
    """
    environ = os.environ if environ is None else environ
    value = environ.get(MPMD_APPNUM_ENV)
    if value is not None and value.strip() != "":
        try:
            appnum = int(value)
        except ValueError:
            appnum = None
        if appnum not in (0, 1):
            raise ValueError(f"{MPMD_APPNUM_ENV}={value!r}: expected 0 (Python controller) or 1 (m-AIA)")
        return appnum
    _require_mpi()
    appnum = comm_world.Get_attr(MPI.APPNUM)
    if appnum is None:
        raise RuntimeError(
            "MPI_APPNUM is not set: launch the controller and m-AIA as one MPMD job "
            "(mpirun -np 1 python ... : -np N maia properties_run.toml), or set "
            f"{MPMD_APPNUM_ENV}=0 for the Python ranks and {MPMD_APPNUM_ENV}=1 for the m-AIA ranks "
            "(e.g. with srun --multi-prog)"
        )
    return int(appnum)


def split_comm_by_appnum(
    comm_world: "MPI.Comm",
) -> Tuple[int, "MPI.Comm", int, int, "MPI.Group", int, int]:
    """Split the MPMD world by application number (MAIA protocol: works for
    any controller rank count; the remote application's root is discovered
    via group translation + Allreduce). The application number comes from
    ``mpmd_appnum`` (``MAIA_MPMD_APPNUM`` if set, else ``MPI_APPNUM``).

    Args:
      comm_world: MPI communicator, typically MPI.COMM_WORLD.

    Returns:
      Tuple of (appnum, appComm, appRank, appNoRanks, appGroup,
      appRootInWorld, remoteRoot) for MaiaInterface to assign.
    """
    _require_mpi()
    appnum = mpmd_appnum(comm_world)
    rank_world = comm_world.Get_rank()
    app_comm = comm_world.Split(appnum, rank_world)
    app_rank = app_comm.Get_rank()
    app_no_ranks = app_comm.Get_size()
    app_group = app_comm.Get_group()
    app_root = 0

    # Get root of other application.
    # Translate the app root to its world rank (app group -> world group).
    group_world = comm_world.Get_group()
    app_root_in_world = app_group.Translate_ranks([app_root], group_world)[0]

    no_app = 2
    buff_send = np.zeros(no_app, dtype="i")
    app_roots_in_world = np.empty_like(buff_send)
    buff_send.fill(-1)
    buff_send[appnum] = app_root_in_world
    comm_world.Allreduce(buff_send, app_roots_in_world, op=MPI.MAX)
    remote_root = app_roots_in_world[1 - appnum]

    return appnum, app_comm, app_rank, app_no_ranks, app_group, app_root_in_world, remote_root


class ExternalProcessEnvMixin:
    """Mixin for environments whose CFD solver runs as a separate MPI
    application in the same MPMD world.

    Class attributes (override per backend):
      CONTROLLER_RANK: world rank of the RL controller app (default 0)
      INTERCOMM_TAG: tag for the inter-communicator handshake (default 99)
      MPI_SPLIT_LOG_PREFIX: prefix for split log lines (default "[MPI_SPLIT] ")
    """

    CONTROLLER_RANK: int = 0
    INTERCOMM_TAG: int = 99
    MPI_SPLIT_LOG_PREFIX: str = "[MPI_SPLIT] "

    def _split_mpmd_comm(self, comm_world: "MPI.Comm", nproc: Optional[int] = None) -> "MPI.Comm":
        """Split the world and return the controller<->solver
        inter-communicator (rank-color strategy; see `mpi_split`)."""
        return mpi_split(
            comm_world,
            nproc=nproc,
            controller_rank=self.CONTROLLER_RANK,
            intercomm_tag=self.INTERCOMM_TAG,
            log_prefix=self.MPI_SPLIT_LOG_PREFIX,
        )
