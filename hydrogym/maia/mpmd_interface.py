"""
MPI Interface for m-AIA CFD Solver
===================================

This module provides the MPI MPMD (Multiple Program Multiple Data) interface
for communication between Python RL controllers and the m-AIA CFD solver.
"""

from typing import List, Optional, Union

import numpy as np
from mpi4py import MPI

# MPI tag mapping for different command types
COMMAND_TAGS = {
    "timeStep": 0,
    "lbRequest": 1,
    "lbContinue": 2,
    "lbControlActions": 3,
    "lbProbePoint": 4,
    "lbReinit": 5,
    "lbForce": 6,
}


class MaiaInterface:
  """
    MPI MPMD interface for communication with the m-AIA CFD solver.

    This class provides an interface to m-AIA via the MPI multiple program
    multiple data (MPMD) execution model. The m-AIA binary and Python controller
    must be launched together, sharing MPI_COMM_WORLD.

    Example launch command:
        mpirun -np <n_controller_ranks> <controller> : -np <n_maia_ranks> <maia>

    Attributes:
        nDim: Number of spatial dimensions (2 or 3).
        worldComm: MPI world communicator.
        appComm: Application-specific communicator.
        appRank: Rank within the application communicator.
        remoteRoot: Root rank of the remote (m-AIA) application.
    """

  def __init__(self, nDim: int):
    """
        Initialize the MaiaInterface.

        Args:
            nDim: Number of spatial dimensions (2 or 3).
        """
    self.worldComm = None
    self.appComm = None
    self.appRank = None
    self.appRoot = 0
    self.appRootInWorld = None
    self.appNoRanks = None
    self.appGroup = None
    self.appnum = None
    self.remoteRoot = None
    self.nDim = nDim

  def init_comm(self, comm_world: MPI.Comm) -> None:
    """
        Initialize MPI communication with m-AIA.

        Sets up the communicators and determines the root ranks for both
        the Python controller and the m-AIA solver.

        Args:
            comm_world: MPI communicator, typically MPI.COMM_WORLD.
        """
    self.appnum = comm_world.Get_attr(MPI.APPNUM)
    rank_world = comm_world.Get_rank()
    self.worldComm = comm_world
    self.appComm = comm_world.Split(self.appnum, rank_world)
    self.appRank = self.appComm.Get_rank()
    self.appNoRanks = self.appComm.Get_size()
    self.appGroup = self.appComm.Get_group()

    # Get root of other application
    group_world = comm_world.Get_group()
    self.appRootInWorld = group_world.Translate_ranks([self.appRoot],
                                                      self.appGroup)[0]

    no_app = 2
    buff_send = np.zeros(no_app, dtype='i')
    app_roots_in_world = np.empty_like(buff_send)
    buff_send.fill(-1)
    buff_send[self.appnum] = self.appRootInWorld
    self.worldComm.Allreduce(buff_send, app_roots_in_world, op=MPI.MAX)
    self.remoteRoot = app_roots_in_world[1 - self.appnum]

  def _comm_send(self,
                 name: str,
                 data: Union[List, np.ndarray],
                 send_tag: bool = True) -> None:
    """
        Send double data to m-AIA root process.

        Args:
            name: Communication command name.
            data: Data to send (list or array of doubles).
            send_tag: Whether to send the command tag first.
        """
    if self.appRank == self.appRoot:
      mpi_tag = COMMAND_TAGS[name]
      if send_tag:
        send_buf = np.zeros(1, dtype='i')
        send_buf[0] = mpi_tag
        self.worldComm.Ssend(
            send_buf, dest=self.remoteRoot, tag=COMMAND_TAGS["lbRequest"])
      send_buf = np.zeros(len(data), dtype=np.float64)
      for i, value in enumerate(data):
        send_buf[i] = value
      self.worldComm.Ssend(send_buf, dest=self.remoteRoot, tag=mpi_tag)

  def _comm_send_int(self, name: str, data: Union[List[int],
                                                  np.ndarray]) -> None:
    """
        Send integer data to m-AIA root process.

        Args:
            name: Communication command name.
            data: Data to send (list or array of integers).
        """
    if self.appRank == self.appRoot:
      mpi_tag = COMMAND_TAGS[name]
      send_buf = np.zeros(1, dtype='i')
      send_buf[0] = mpi_tag
      self.worldComm.Ssend(
          send_buf, dest=self.remoteRoot, tag=COMMAND_TAGS["lbRequest"])
      send_buf = np.zeros(len(data), dtype='i')
      for i, value in enumerate(data):
        send_buf[i] = value
      self.worldComm.Ssend(send_buf, dest=self.remoteRoot, tag=mpi_tag)

  def _comm_recv(self, name: str) -> Optional[np.ndarray]:
    """
        Receive data from m-AIA root process.

        Args:
            name: Communication command name.

        Returns:
            Received data as numpy array of doubles, or None if not root rank.
        """
    if self.appRank == self.appRoot:
      mpi_tag = COMMAND_TAGS[name]
      status = MPI.Status()
      self.worldComm.Probe(tag=mpi_tag, status=status)
      size = status.Get_elements(MPI.DOUBLE)
      recv_buf = np.zeros(size, dtype=np.float64)
      self.worldComm.Recv(recv_buf, source=self.remoteRoot, tag=mpi_tag)
      return recv_buf
    else:
      return None

  def continueRun(self) -> None:
    """
        Signal m-AIA to continue the simulation run.

        This sends a continue command to allow the CFD solver to proceed
        after data exchange.
        """
    if self.appRank == self.appRoot:
      mpi_tag = COMMAND_TAGS["lbContinue"]
      send_buf = np.zeros(1, dtype='i')
      send_buf[0] = mpi_tag
      self.worldComm.Ssend(
          send_buf, dest=self.remoteRoot, tag=COMMAND_TAGS["lbRequest"])

  def runTimeSteps(self, time_steps: int = 1) -> None:
    """
        Advance the m-AIA simulation by a specified number of time steps.

        Args:
            time_steps: Number of time steps to advance. Use 0 to signal finish.
        """
    if self.appRank == self.appRoot:
      send_buf = np.zeros(1, dtype='i')
      send_buf[0] = time_steps
      self.worldComm.Ssend(
          send_buf, dest=self.remoteRoot, tag=COMMAND_TAGS["timeStep"])

  def finishRun(self) -> None:
    """
        Signal m-AIA to finish the simulation run.

        This sends a time step count of 0, indicating the run should terminate.
        """
    self.runTimeSteps(0)

  def setControlProperties(self, control_actions: Union[List,
                                                        np.ndarray]) -> None:
    """
        Set control properties for boundary condition actuation.

        Args:
            control_actions: Control action values. For jets in 2D with n jets,
                format is [u0, v0, u1, v1, ..., un, vn].
        """
    self._comm_send("lbControlActions", control_actions)

  def getProbeData(self, probe_point_coords: Union[List,
                                                   np.ndarray]) -> np.ndarray:
    """
        Get flow field data at specified probe locations.

        Args:
            probe_point_coords: Probe coordinates. For 2D with 3 probes,
                format is [x0, y0, x1, y1, x2, y2].

        Returns:
            Array of probe data. For 2D: [u, v, rho, p] per probe.
            For 3D: [u, v, w, rho, p] per probe.
        """
    no_probes = int(len(probe_point_coords) / self.nDim)
    self._comm_send_int("lbProbePoint", [no_probes])
    self._comm_send("lbProbePoint", probe_point_coords, False)
    probe_states = self._comm_recv("lbProbePoint")
    return probe_states

  def getForce(self, bc_segment_id: int) -> np.ndarray:
    """
        Get the force acting on a boundary segment.

        Args:
            bc_segment_id: Index of the boundary condition segment.

        Returns:
            Force vector array of length nDim.
        """
    self._comm_send_int("lbForce", [bc_segment_id])
    force = self._comm_recv("lbForce")
    return force

  def reinit(self) -> None:
    """
        Reinitialize the m-AIA simulation.

        Triggers the m-AIA solver to recall its initialization routine,
        resetting the simulation to its initial state.
        """
    if self.appRank == self.appRoot:
      mpi_tag = COMMAND_TAGS["lbReinit"]
      send_buf = np.zeros(1, dtype='i')
      send_buf[0] = mpi_tag
      self.worldComm.Ssend(
          send_buf, dest=self.remoteRoot, tag=COMMAND_TAGS["lbRequest"])
