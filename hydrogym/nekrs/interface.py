"""
MPI interface for the NekRS CFD solver
======================================

Counterpart of hydrogym/maia/mpmd_interface.py for NekRS. The solver side is
hydrogym/nekrs/udf/hydrogym_nekrs.hpp, included by the case udf. Probe locations and the actuator
layout are sent from Python at run time; the case only marks the boundaries that may be actuated
([CASEDATA] hydrogymBoundaries).

Launch (MPMD, one controller rank, NekRS on the remaining ranks):
    mpirun -n 1 python controller.py : -n N env NEKRS_MPMD_COLOR=1 nekrs-mpmd --setup case.par
"""

from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
from mpi4py import MPI

PROTOCOL_VERSION = 2
TAG_CMD, TAG_INT, TAG_DBL, TAG_RINT, TAG_RDBL, TAG_RMSG = 22, 23, 24, 25, 26, 27

FIELD_CODES = {"u": 0, "v": 1, "w": 2, "p": 3}
FOOTPRINT_CODES = {"point": 0, "gaussian": 1, "box": 2}
DIRECTION_CODES = {"normal": 0, "vector": 1, "angle": 2}
ZNMF_CODES = {"none": 0, "uniform": 1, "actions": 2}
CHECK_CODES = {"off": 0, "warn": 1, "error": 2}
MONITOR_CODES = {"force": 0, "gradient": 1}
BOUNDARY_NODE_MODES = {"all": 0, "facindf": 1, "unique": 2}


class NekRSError(RuntimeError):
    """A request was rejected by the solver (nothing was changed on the solver side)."""


def mpi_split(comm_world: MPI.Comm, nproc: Optional[int] = None) -> MPI.Comm:
    """Inter-communicator between the controller (world rank 0) and the NekRS ranks (as hydrogym.nek.env)."""
    size = comm_world.Get_size()
    if size < 2:
        raise RuntimeError("Launch with MPMD: mpirun -n 1 python ... : -n N nekrs-mpmd ...")
    if nproc is not None and size != 1 + nproc:
        raise RuntimeError(f"MPI world size {size} does not match 1 controller + {nproc} NekRS ranks")
    color = 0 if comm_world.Get_rank() == 0 else 1
    local = comm_world.Split(color, comm_world.Get_rank())
    return local.Create_intercomm(local_leader=0, peer_comm=MPI.COMM_WORLD, remote_leader=1, tag=99)


class NekRSInterface:
    """
    MPI MPMD interface for communication with NekRS.

    Attributes:
        comm: Inter-communicator to the NekRS ranks (requests go to NekRS rank 0).
        info: Solver information from HELLO (ranks, polynomial order, actuatable boundaries, mu, rho).
    """

    def __init__(self, nproc: Optional[int] = None, comm_world: MPI.Comm = MPI.COMM_WORLD):
        self.comm = mpi_split(comm_world, nproc)
        self.nproc = self.comm.Get_remote_size()
        self._probe_key = None
        self.info = self.hello()

    # ------------------------------------------------------------------ transport

    def request(self, cmd: str, ints: Sequence[int] = (), dbls: Sequence[float] = ()) -> Tuple[np.ndarray, np.ndarray]:
        """Send a request and return (ints, doubles) of the reply; raises NekRSError if it was rejected."""
        assert len(cmd) == 5
        self.comm.Send([cmd.encode(), MPI.CHAR], dest=0, tag=TAG_CMD)
        self.comm.Send([np.ascontiguousarray(ints, dtype=np.int32), MPI.INT], dest=0, tag=TAG_INT)
        self.comm.Send([np.ascontiguousarray(dbls, dtype=np.float64), MPI.DOUBLE], dest=0, tag=TAG_DBL)
        rint = self._recv(TAG_RINT, MPI.INT, np.int32)
        rdbl = self._recv(TAG_RDBL, MPI.DOUBLE, np.float64)
        msg = self._recv(TAG_RMSG, MPI.CHAR, np.uint8).tobytes().decode()
        if rint[0] != 0:
            raise NekRSError(msg if msg.startswith(cmd) else f"{cmd}: {msg}")
        if msg:
            warnings.warn(f"[NekRS] {msg}", stacklevel=3)
        return rint[1:], rdbl

    def _recv(self, tag, mpi_type, dtype) -> np.ndarray:
        status = MPI.Status()
        self.comm.Probe(source=0, tag=tag, status=status)
        buf = np.empty(status.Get_count(mpi_type), dtype=dtype)
        self.comm.Recv([buf, mpi_type], source=0, tag=tag)
        return buf

    # ------------------------------------------------------------------ requests

    def hello(self) -> Dict:
        ri, rd = self.request("HELLO")
        if ri[0] != PROTOCOL_VERSION:
            raise NekRSError(f"protocol version {ri[0]} of the udf, expected {PROTOCOL_VERSION}")
        return {
            "nranks": int(ri[1]),
            "N": int(ri[2]),
            "actuatable_boundaries": [int(b) for b in ri[4 : 4 + ri[3]]],
            "time": rd[0],
            "dt": rd[1],
            "mu": rd[2],
            "rho": rd[3],
            "tstep": int(rd[4]),
        }

    def boundary_nodes(self, bid: int, nodes: str = "all") -> Dict[str, np.ndarray]:
        """GLL nodes of a boundary in the Nek5000 HydroGym registration order (rank, element, face, iz, iy, ix)."""
        ri, rd = self.request("BNDRY", [bid, BOUNDARY_NODE_MODES[nodes]])
        n = int(ri[0])
        info = ri[1:].reshape(n, 6)
        geo = rd.reshape(n, 6)
        return {
            "rank": info[:, 0],
            "element": info[:, 1],
            "face": info[:, 2],
            "ix": info[:, 3],
            "iy": info[:, 4],
            "iz": info[:, 5],
            "xyz": geo[:, :3].copy(),
            "normal": geo[:, 3:].copy(),
        }

    def define_probes(self, xyz: np.ndarray, fields: List[str], mean_axis: Optional[str] = None) -> Dict:
        """Locate the probes once (findpts); later calls of get_probes only interpolate."""
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        axis = -1 if mean_axis is None else "xyz".index(mean_axis)
        ints = [len(xyz), len(fields)] + [FIELD_CODES[f] for f in fields] + [axis]
        ri, rd = self.request("PRBDF", ints, xyz.ravel())
        self._probe_key = (xyz.tobytes(), tuple(fields), axis)
        self.n_probes, self.probe_fields = len(xyz), list(fields)
        return {"n_border": int(ri[0]), "max_distance": float(rd[0])}

    def get_probes(self) -> Tuple[float, float, np.ndarray]:
        """(time, CFL, values[n_probes, n_fields])."""
        _, rd = self.request("PROBE")
        vals = rd[2:].reshape(len(self.probe_fields), self.n_probes).T
        return rd[0], rd[1], vals

    def getProbeData(self, probe_point_coords, fields=("u", "v", "w", "p"), mean_axis=None) -> np.ndarray:
        """m-AIA style: flat coordinates in, flat [fields per probe] out; probes are re-located only if changed."""
        xyz = np.asarray(probe_point_coords, dtype=np.float64).reshape(-1, 3)
        axis = -1 if mean_axis is None else "xyz".index(mean_axis)
        if self._probe_key != (xyz.tobytes(), tuple(fields), axis):
            self.define_probes(xyz, list(fields), mean_axis)
        return self.get_probes()[2].ravel()

    def define_actuators(
        self,
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
    ) -> Dict[str, np.ndarray]:
        """
        Install the actuators. Per actuator: centre (3), footprint code, widths (3, sigma or half width,
        0 = unbounded), direction code, vector (3), angle [rad]. Returns the wall nodes covered, the flux per
        unit action, nodes per footprint width and the under-resolution flags.
        """
        n = len(centres)
        ints = [n, len(boundaries)] + list(boundaries) + [ZNMF_CODES[znmf], CHECK_CODES[check]]
        ints += list(np.asarray(footprint, dtype=int)) + list(np.asarray(direction, dtype=int))
        dbls = np.concatenate(
            [
                [min_nodes_per_width, point_tol],
                np.asarray(centres, float).ravel(),
                np.asarray(widths, float).ravel(),
                np.asarray(vectors, float).ravel(),
                np.asarray(angles, float).ravel(),
            ]
        )
        ri, rd = self.request("ACTDF", ints, dbls)
        self.n_actuators = n
        return {
            "nodes": ri[:n].copy(),
            "under_resolved": ri[n : 2 * n].astype(bool),
            "flux_per_unit": rd[:n].copy(),
            "nodes_per_width": rd[n : 2 * n].copy(),
            "area": float(rd[2 * n]),
        }

    def setControlProperties(self, velocities) -> Dict[str, float]:
        """Wall velocity amplitude per actuator (m-AIA name); returns the actuation diagnostics."""
        _, rd = self.request("ACTUA", [], np.asarray(velocities, dtype=np.float64))
        return {"net_flux": rd[0], "max_wall_velocity": rd[1], "znmf_correction": rd[2], "max_command": rd[3]}

    def define_monitors(self, monitors: List[Tuple[int, str]]) -> np.ndarray:
        """Quantities recorded at every solver step: (bid, 'force' | 'gradient'). Returns the boundary areas."""
        ints = [len(monitors)] + [v for bid, kind in monitors for v in (bid, MONITOR_CODES[kind])]
        _, rd = self.request("MONDF", ints)
        self.monitors = list(monitors)
        return rd

    def runTimeSteps(self, n: int) -> np.ndarray:
        """Advance n solver steps; rows [tstep, time, dt, CFL, 6 values per monitor] for every step."""
        ri, rd = self.request("STEPS", [n])
        return rd.reshape(int(ri[0]), int(ri[1]))

    # ------------------------------------------------------------------ episode reset

    def save_snapshot(self, slot: int) -> Dict:
        """Copy the complete time-stepper state (all lags, dt history) to solver device memory."""
        ri, rd = self.request("SNAPS", [slot])
        return {"slots": int(ri[0]), "time": rd[0], "bytes": rd[1]}

    def reset_to_snapshot(self, slot: int) -> Dict:
        """Restore a snapshot; the solver clock keeps running and the actuation is zeroed."""
        _, rd = self.request("RESET", [0, slot])
        return {"time": rd[0], "state_time": rd[1]}

    def reset_from_file(self, path: str, interpolate: bool = False) -> Dict:
        """Load u, p from a field file (relative to the case directory); the lag history is rebuilt (frozen)."""
        spec = path + ("+int" if interpolate else "")
        _, rd = self.request("RESET", [1] + list(spec.encode()))
        return {"time": rd[0], "state_time": rd[1]}

    def write_snapshot(self, slot: int, path: str) -> Dict:
        """Write a snapshot slot to path_r<rank> (one file per NekRS rank)."""
        ri, rd = self.request("SNAPF", [0, slot] + list(path.encode()))
        return {"tstep": int(ri[0]), "time": rd[0]}

    def read_snapshot(self, slot: int, path: str) -> Dict:
        """Read a snapshot written by write_snapshot (same partition, order and mesh) into a slot."""
        ri, rd = self.request("SNAPF", [1, slot] + list(path.encode()))
        return {"tstep": int(ri[0]), "time": rd[0]}

    def finishRun(self) -> None:
        self.comm.Send([b"TERMN", MPI.CHAR], dest=0, tag=TAG_CMD)
        self.comm.Free()
