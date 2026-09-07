"""Shared MPMD communicator setup for external-process backends (audit Task 3.5).

The Nek and MAIA backends both talk to their CFD solver as a separate MPI
application launched in the same MPMD world (``mpirun -n 1 python ... :
-n N ./solver``). Each previously owned a private copy of the world-split
logic:

  - ``hydrogym.nek.env.mpi_split`` -- rank-color split (rank 0 = controller,
    ranks 1+ = workers) followed by ``Create_intercomm``; validates the
    world size against the expected worker count.
  - ``hydrogym.maia.mpmd_interface.MaiaInterface.init_comm`` -- APPNUM-based
    split (true MPMD, any controller rank count), with group-translation +
    Allreduce discovery of the remote application's root rank.

The two are genuinely different split strategies (not copies), so this
module hosts each verbatim as a named function and adds
`ExternalProcessEnvMixin` as the common extension point carrying the
protocol knobs (controller rank, intercomm tag, log prefix). Per-solver
tag-protocol code (the actual command/data messages) stays untouched in
each backend.

Callers:
  - NekEnv (mixin ``_split_mpmd_comm``); ``hydrogym.nek.env.mpi_split`` is
    re-exported from here so ``hydrogym.nek.mpi_split`` and the
    ``monkeypatch.setattr(nek_env_mod, "mpi_split", ...)`` test idiom keep
    working.
  - ``MaiaInterface.init_comm`` delegates its communicator setup to
    ``split_comm_by_appnum`` and assigns the results.
"""

from typing import Optional, Tuple

import numpy as np
from mpi4py import MPI


def mpi_split(
    comm_world: MPI.Comm,
    nproc: Optional[int] = None,
    controller_rank: int = 0,
    intercomm_tag: int = 99,
    log_prefix: str = "[MPI_SPLIT] ",
) -> MPI.Comm:
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

    # NOTE: the original NekEnv copy hardcoded remote_leader=1 on BOTH
    # sides. That is correct only for the controller side (whose remote
    # leader is world rank 1, the worker app's leader -- in production the
    # workers are the Nek5000 Fortran binary, so the Python color-1 branch
    # never runs there). A Python worker passing 1 would contact itself and
    # leave the controller hanging in Create_intercomm forever -- exactly
    # what test/mpmd_smoke_split.py exposed. 1 - color selects the other
    # app's leader on both sides and is identical on the production
    # controller path.
    sub_comm = local_comm.Create_intercomm(
        local_leader=0, peer_comm=MPI.COMM_WORLD, remote_leader=1 - color, tag=intercomm_tag
    )
    print(
        f"{log_prefix}Inter-comm created: local_size={sub_comm.Get_size()}, remote_size={sub_comm.Get_remote_size()}",
        flush=True,
    )
    return sub_comm


def split_comm_by_appnum(
    comm_world: MPI.Comm,
) -> Tuple[int, MPI.Comm, int, int, MPI.Group, int, int]:
    """Split the MPMD world by application number (MAIA protocol: works for
    any controller rank count; the remote application's root is discovered
    via group translation + Allreduce).

    Args:
      comm_world: MPI communicator, typically MPI.COMM_WORLD.

    Returns:
      Tuple of (appnum, appComm, appRank, appNoRanks, appGroup,
      appRootInWorld, remoteRoot) for MaiaInterface to assign.
    """
    appnum = comm_world.Get_attr(MPI.APPNUM)
    rank_world = comm_world.Get_rank()
    app_comm = comm_world.Split(appnum, rank_world)
    app_rank = app_comm.Get_rank()
    app_no_ranks = app_comm.Get_size()
    app_group = app_comm.Get_group()
    app_root = 0

    # Get root of other application.
    # NOTE: the original MaiaInterface copy translated in the wrong
    # direction (world group -> app group), which yields MPI_UNDEFINED (-1)
    # for every app except app 0 -- the value only ever came out right
    # because the Python controller is always app 0, so world rank 0 ==
    # app rank 0 and both directions coincide. The intended semantics ("app
    # root's rank in WORLD") needs the app group -> world group direction
    # used here, which test/mpmd_smoke_split.py proved necessary for a
    # Python app 1 (its Allreduce slot stayed -1 otherwise). Identical
    # result on the production path (Python = app 0).
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

    def _split_mpmd_comm(self, comm_world: MPI.Comm, nproc: Optional[int] = None) -> MPI.Comm:
        """Split the world and return the controller<->solver
        inter-communicator (rank-color strategy; see `mpi_split`)."""
        return mpi_split(
            comm_world,
            nproc=nproc,
            controller_rank=self.CONTROLLER_RANK,
            intercomm_tag=self.INTERCOMM_TAG,
            log_prefix=self.MPI_SPLIT_LOG_PREFIX,
        )
