---
sidebar_label: interface
title: hydrogym.nekrs.interface
---

MPI interface for the NekRS CFD solver
======================================

Counterpart of hydrogym/maia/mpmd_interface.py for NekRS. The solver side is
hydrogym/nekrs/udf/hydrogym_nekrs.hpp, included by the case udf. Probe locations and the actuator
layout are sent from Python at run time; the case only marks the boundaries that may be actuated
([CASEDATA] hydrogymBoundaries).

Launch (MPMD, one controller rank, NekRS on the remaining ranks):
mpirun -n 1 python controller.py : -n N env NEKRS_MPMD_COLOR=1 nekrs-mpmd --setup case.par
Several independent NekRS instances (e.g. one GPU each) are colours 1..M of the same launch, each with N ranks:
mpirun -n 1 python controller.py : -n N env NEKRS_MPMD_COLOR=1 nekrs-mpmd ... : -n N env NEKRS_MPMD_COLOR=2 ...
and connect(M, N) gives one inter-communicator per instance.

## NekRSError Objects

```python
class NekRSError(RuntimeError)
```

A request was rejected by the solver (nothing was changed on the solver side).

#### mpi\_split

```python
def mpi_split(comm_world: MPI.Comm, nproc: Optional[int] = None) -> MPI.Comm
```

Inter-communicator between the controller (world rank 0) and the NekRS ranks (as hydrogym.nek.env).

#### connect

```python
def connect(n_instances: int,
            nproc: int,
            comm_world: MPI.Comm = MPI.COMM_WORLD) -> List[MPI.Comm]
```

Inter-communicators to n_instances NekRS instances of nproc ranks each, launched in this order after the
controller (world ranks 1 + k nproc ... for instance k, NEKRS_MPMD_COLOR = k + 1).

## NekRSInterface Objects

```python
class NekRSInterface()
```

MPI MPMD interface for communication with NekRS.

**Attributes**:

- `comm` - Inter-communicator to the NekRS ranks (requests go to NekRS rank 0).
- `info` - Solver information from HELLO (ranks, polynomial order, actuatable boundaries, mu, rho).

#### \_\_init\_\_

```python
def __init__(nproc: Optional[int] = None,
             comm_world: MPI.Comm = MPI.COMM_WORLD,
             comm: Optional[MPI.Comm] = None)
```

comm: an inter-communicator from connect() (several instances); by default the single-instance split.

#### request

```python
def request(
    cmd: str, ints: Sequence[int] = (), dbls: Sequence[float] = ()
) -> Tuple[np.ndarray, np.ndarray]
```

Send a request and return (ints, doubles) of the reply; raises NekRSError if it was rejected.

#### send\_request

```python
def send_request(
    cmd: str, ints: Sequence[int] = (), dbls: Sequence[float] = ()) -> None
```

First half of request(): the solver starts working, the reply is collected with recv_reply().

#### boundary\_nodes

```python
def boundary_nodes(bid: int, nodes: str = "all") -> Dict[str, np.ndarray]
```

GLL nodes of a boundary in the Nek5000 HydroGym registration order (rank, element, face, iz, iy, ix).

#### define\_probes

```python
def define_probes(xyz: np.ndarray,
                  fields: List[str],
                  mean_axis: Optional[str] = None) -> Dict
```

Locate the probes once (findpts); later calls of get_probes only interpolate.

#### get\_probes

```python
def get_probes() -> Tuple[float, float, np.ndarray]
```

(time, CFL, values[n_probes, n_fields]).

#### getProbeData

```python
def getProbeData(probe_point_coords,
                 fields=("u", "v", "w", "p"),
                 mean_axis=None) -> np.ndarray
```

m-AIA style: flat coordinates in, flat [fields per probe] out; probes are re-located only if changed.

#### define\_actuators

```python
def define_actuators(
        boundaries: List[int],
        centres: np.ndarray,
        footprint: np.ndarray,
        widths: np.ndarray,
        direction: np.ndarray,
        vectors: np.ndarray,
        angles: np.ndarray,
        znmf: str = "uniform",
        check: str = "error",
        min_nodes_per_width: float = 2.0,
        point_tol: float = 1e-3,
        node_box: Optional[np.ndarray] = None,
        period: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]
```

Install the actuators. Per actuator: centre (3), footprint code, widths (3, sigma or half width,
0 = unbounded), direction code, vector (3), angle [rad]. node_box [xmin, xmax, ymin, ymax, zmin, zmax]
restricts the actuated boundary nodes, period [px, py, pz] (0 = none) makes the footprint distances
periodic. Returns the wall nodes covered, the flux per unit action, nodes per footprint width and the
under-resolution flags.

#### setControlProperties

```python
def setControlProperties(velocities) -> Dict[str, float]
```

Wall velocity amplitude per actuator (m-AIA name); returns the actuation diagnostics.

#### define\_monitors

```python
def define_monitors(monitors: List[Tuple[int, str]]) -> np.ndarray
```

Quantities recorded at every solver step: (bid, 'force' | 'gradient'). Returns the boundary areas.

#### runTimeSteps

```python
def runTimeSteps(n: int) -> np.ndarray
```

Advance n solver steps; rows [tstep, time, dt, CFL, 6 values per monitor] for every step.

#### start\_time\_steps

```python
def start_time_steps(n: int) -> None
```

runTimeSteps without waiting, so that several instances advance concurrently.

#### wall\_data

```python
def wall_data(bid: int,
              geometry: bool = True,
              velocity: bool = False,
              traction: bool = True) -> Dict[str, np.ndarray]
```

Viscous traction mu (grad u + grad u^T) . n and pressure at every GLL node of the boundary faces (one
entry per face copy, so sum(w * f) integrates f over the boundary). With geometry: xyz, the surface
quadrature weight w and the unit normal into the fluid; the node order is the same on every call.
velocity: also the wall velocity imposed by the actuators (`u_wall`, [n, 3], zero off the actuated nodes).
traction=False skips the velocity gradient (pressure and wall velocity only; cheap enough for every step).

#### write\_field

```python
def write_field(name: str) -> float
```

Write u, p (fp64) to the NekRS field file &lt;name>0.f00000 in the case directory; returns the time.

#### save\_snapshot

```python
def save_snapshot(slot: int) -> Dict
```

Copy the complete time-stepper state (all lags, dt history) to solver device memory.

#### reset\_to\_snapshot

```python
def reset_to_snapshot(slot: int) -> Dict
```

Restore a snapshot; the solver clock keeps running and the actuation is zeroed.

#### reset\_from\_file

```python
def reset_from_file(path: str, interpolate: bool = False) -> Dict
```

Load u, p from a field file (relative to the case directory); the lag history is rebuilt (frozen).

#### write\_snapshot

```python
def write_snapshot(slot: int, path: str) -> Dict
```

Write a snapshot slot to path_r&lt;rank> (one file per NekRS rank).

#### read\_snapshot

```python
def read_snapshot(slot: int, path: str) -> Dict
```

Read a snapshot written by write_snapshot (same partition, order and mesh) into a slot.

