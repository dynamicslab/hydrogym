---
sidebar_label: core_external
title: hydrogym.core_external
---

Shared MPMD communicator setup for external-process backends.

The Nek and MAIA backends both talk to their CFD solver as a separate MPI
application launched in the same MPMD world
(`mpirun -n 1 python ... : -n N ./solver`). This module provides the two world-split strategies:

- `mpi_split` (used by Nek) -- rank-color split (rank 0 = controller,
ranks 1+ = workers) followed by `Create_intercomm`; validates the
world size against the expected worker count.
- `split_comm_by_appnum` (used by MAIA) -- APPNUM-based split (true
MPMD, any controller rank count), with group translation + Allreduce
discovery of the remote application's root rank.

`ExternalProcessEnvMixin` is the common extension point carrying the
protocol knobs (controller rank, intercomm tag, log prefix). The
per-solver tag protocol (the actual command/data messages) lives in each
backend.

Callers:
- NekEnv (mixin `_split_mpmd_comm`); `hydrogym.nek.env.mpi_split` is
re-exported from here, so `hydrogym.nek.mpi_split` remains available.
- `MaiaInterface.init_comm` delegates its communicator setup to
`split_comm_by_appnum` and assigns the results.

#### mpi\_split

```python
def mpi_split(comm_world: "MPI.Comm",
              nproc: Optional[int] = None,
              controller_rank: int = 0,
              intercomm_tag: int = 99,
              log_prefix: str = "[MPI_SPLIT] ") -> "MPI.Comm"
```

Split the MPMD world into a controller/worker inter-communicator
(Nek protocol: rank 0 is the controller, ranks 1+ are the workers).

**Arguments**:

- `comm_world` - MPI communicator (typically MPI.COMM_WORLD)
- `nproc` - Expected number of solver workers (for validation)
- `controller_rank` - World rank of the controller (default 0)
- `intercomm_tag` - MPI tag for the inter-communicator handshake
- `log_prefix` - Prefix for the split log lines
  

**Returns**:

  Inter-communicator between controller and workers

#### mpmd\_appnum

```python
def mpmd_appnum(comm_world: "MPI.Comm",
                environ: Optional[Mapping[str, str]] = None) -> int
```

Application number of this process in the MAIA MPMD world: 0 = Python controller, 1 = m-AIA.

`MAIA_MPMD_APPNUM` (environment variable) takes precedence when it is set. Launchers that start every rank as
one application, e.g. `srun --multi-prog` (Slurm, ParaStation MPI), give all ranks the same (or no)
`MPI_APPNUM`; set `MAIA_MPMD_APPNUM=0` for the Python ranks and `MAIA_MPMD_APPNUM=1` for the m-AIA ranks
there (m-AIA reads the same variable). Otherwise `MPI_APPNUM` of the MPMD launcher is used
(`mpirun -np 1 python ... : -np N maia properties_run.toml`).

**Arguments**:

- `comm_world` - MPI communicator, typically MPI.COMM_WORLD.
- `environ` - environment mapping (default `os.environ`).
  

**Raises**:

- `ValueError` - `MAIA_MPMD_APPNUM` is set but not 0 or 1.
- `RuntimeError` - neither `MAIA_MPMD_APPNUM` nor `MPI_APPNUM` is available (not an MPMD launch).

#### split\_comm\_by\_appnum

```python
def split_comm_by_appnum(
    comm_world: "MPI.Comm"
) -> Tuple[int, "MPI.Comm", int, int, "MPI.Group", int, int]
```

Split the MPMD world by application number (MAIA protocol: works for
any controller rank count; the remote application's root is discovered
via group translation + Allreduce). The application number comes from
`mpmd_appnum` (`MAIA_MPMD_APPNUM` if set, else `MPI_APPNUM`).

**Arguments**:

- `comm_world` - MPI communicator, typically MPI.COMM_WORLD.
  

**Returns**:

  Tuple of (appnum, appComm, appRank, appNoRanks, appGroup,
  appRootInWorld, remoteRoot) for MaiaInterface to assign.

## ExternalProcessEnvMixin Objects

```python
class ExternalProcessEnvMixin()
```

Mixin for environments whose CFD solver runs as a separate MPI
application in the same MPMD world.

Class attributes (override per backend):
CONTROLLER_RANK: world rank of the RL controller app (default 0)
INTERCOMM_TAG: tag for the inter-communicator handshake (default 99)
MPI_SPLIT_LOG_PREFIX: prefix for split log lines (default "[MPI_SPLIT] ")

