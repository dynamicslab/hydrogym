---
sidebar_label: core_external
title: hydrogym.core_external
---

Shared MPMD communicator setup for external-process backends (audit Task 3.5).

The Nek and MAIA backends both talk to their CFD solver as a separate MPI
application launched in the same MPMD world (``mpirun -n 1 python ... :
-n N ./solver``). Each previously owned a private copy of the world-split
logic:

- ``hydrogym.nek.env.mpi_split`` -- rank-color split (rank 0 = controller,
ranks 1+ = workers) followed by ``Create_intercomm``; validates the
world size against the expected worker count.
- ``hydrogym.maia.mpmd_interface.MaiaInterface.init_comm`` -- APPNUM-based
split (true MPMD, any controller rank count), with group-translation +
Allreduce discovery of the remote application&#x27;s root rank.

The two are genuinely different split strategies (not copies), so this
module hosts each verbatim as a named function and adds
`ExternalProcessEnvMixin` as the common extension point carrying the
protocol knobs (controller rank, intercomm tag, log prefix). Per-solver
tag-protocol code (the actual command/data messages) stays untouched in
each backend.

Callers:
- NekEnv (mixin ``_split_mpmd_comm``); ``hydrogym.nek.env.mpi_split`` is
re-exported from here so ``hydrogym.nek.mpi_split`` and the
``monkeypatch.setattr(nek_env_mod, &quot;mpi_split&quot;, ...)`` test idiom keep
working.
- ``MaiaInterface.init_comm`` delegates its communicator setup to
``split_comm_by_appnum`` and assigns the results.

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

#### split\_comm\_by\_appnum

```python
def split_comm_by_appnum(
    comm_world: "MPI.Comm"
) -> Tuple[int, "MPI.Comm", int, int, "MPI.Group", int, int]
```

Split the MPMD world by application number (MAIA protocol: works for
any controller rank count; the remote application&#x27;s root is discovered
via group translation + Allreduce).

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
MPI_SPLIT_LOG_PREFIX: prefix for split log lines (default &quot;[MPI_SPLIT] &quot;)

