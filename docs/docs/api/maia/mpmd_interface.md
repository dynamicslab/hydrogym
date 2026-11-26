---
sidebar_label: mpmd_interface
title: hydrogym.maia.mpmd_interface
---

MPI Interface for m-AIA CFD Solver
===================================

This module provides the MPI MPMD (Multiple Program Multiple Data) interface
for communication between Python RL controllers and the m-AIA CFD solver.

## MaiaInterface Objects

```python
class MaiaInterface()
```

MPI MPMD interface for communication with the m-AIA CFD solver.

This class provides an interface to m-AIA via the MPI multiple program
multiple data (MPMD) execution model. The m-AIA binary and Python controller
must be launched together, sharing MPI_COMM_WORLD.

Example launch command:
mpirun -np &lt;n_controller_ranks> &lt;controller> : -np &lt;n_maia_ranks> &lt;maia>

With launchers that give all ranks the same application number (e.g. `srun --multi-prog`), set the
environment variable `MAIA_MPMD_APPNUM` to 0 for the controller ranks and 1 for the m-AIA ranks; it takes
precedence over `MPI_APPNUM` (see `hydrogym.core_external.mpmd_appnum`).

**Attributes**:

- `nDim` - Number of spatial dimensions (2 or 3).
- `worldComm` - MPI world communicator.
- `appComm` - Application-specific communicator.
- `appRank` - Rank within the application communicator.
- `remoteRoot` - Root rank of the remote (m-AIA) application.

#### \_\_init\_\_

```python
def __init__(nDim: int)
```

Initialize the MaiaInterface.

**Arguments**:

- `nDim` - Number of spatial dimensions (2 or 3).

#### init\_comm

```python
def init_comm(comm_world: MPI.Comm, nproc: Optional[int] = None) -> None
```

Initialize MPI communication with m-AIA.

Sets up the communicators and determines the root ranks for both
the Python controller and the m-AIA solver. The split math itself
lives in `hydrogym.core_external.split_comm_by_appnum` (shared
module for MPMD world-split strategies); only the
attribute assignment stays here.

**Arguments**:

- `comm_world` - MPI communicator, typically MPI.COMM_WORLD.
- `nproc` - Expected number of m-AIA solver ranks. When given,
  validated against the actual remote-app rank count
  (`comm_world.Get_size() - appNoRanks`, i.e. every world
  rank not part of this app) and a clear `RuntimeError` is
  raised on mismatch, as `hydrogym.core_external.mpi_split`
  does for Nek. `None` (default) skips the check.
  

**Raises**:

- `RuntimeError` - If `nproc` is given and does not match the
  actual number of m-AIA solver ranks in the MPMD job.

#### continueRun

```python
def continueRun() -> None
```

Signal m-AIA to continue the simulation run.

This sends a continue command to allow the CFD solver to proceed
after data exchange.

#### runTimeSteps

```python
def runTimeSteps(time_steps: int = 1) -> None
```

Advance the m-AIA simulation by a specified number of time steps.

**Arguments**:

- `time_steps` - Number of time steps to advance. Use 0 to signal finish.

#### finishRun

```python
def finishRun() -> None
```

Signal m-AIA to finish the simulation run.

This sends a time step count of 0, indicating the run should terminate.

#### setControlProperties

```python
def setControlProperties(control_actions: Union[List, np.ndarray]) -> None
```

Set control properties for boundary condition actuation.

**Arguments**:

- `control_actions` - Control action values. For jets in 2D with n jets,
  format is [u0, v0, u1, v1, ..., un, vn].

#### getProbeData

```python
def getProbeData(probe_point_coords: Union[List, np.ndarray]) -> np.ndarray
```

Get flow field data at specified probe locations.

**Arguments**:

- `probe_point_coords` - Probe coordinates. For 2D with 3 probes,
  format is [x0, y0, x1, y1, x2, y2].
  

**Returns**:

  Array of probe data. For 2D: [u, v, rho, p] per probe.
  For 3D: [u, v, w, rho, p] per probe.

#### getForce

```python
def getForce(bc_segment_id: int) -> np.ndarray
```

Get the force acting on a boundary segment.

**Arguments**:

- `bc_segment_id` - Index of the boundary condition segment.
  

**Returns**:

  Force vector array of length nDim.

#### getActuationPower

```python
def getActuationPower() -> np.ndarray
```

Get the actuation power of the jets of the structured finite-volume solver (boundary condition 1008).

m-AIA forms the power per wall face of the jet footprints: the flux of pressure work and kinetic energy
`v_n (p - p_ref + rho |u|^2 / 2)` through the wall, with `p_ref` the mean wall pressure over the jets,
normalised like the force coefficients of the solver.

**Returns**:

  `[P, P_pos, P_abs, P_0, ..., P_{N-1}, Q_0, ..., Q_{N-1}]`: the signed, positive-part and absolute power
  coefficient of all jets, the signed power coefficient of each jet and its volume flux into the fluid
  (normalised by the free-stream speed).

#### reinit

```python
def reinit() -> None
```

Reinitialize the m-AIA simulation.

Triggers the m-AIA solver to recall its initialization routine,
resetting the simulation to its initial state.

