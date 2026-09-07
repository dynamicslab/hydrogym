"""Minimal external-process solver environment (reference skeleton).

This is the copyable starting point for a backend whose solver is a
separate compiled MPI application (the pattern the `maia` and `nek`
backends use). The controller (this Python code) and the solver binary run
as two applications in one MPMD world:

    mpirun -np 1 python controller.py ... : -np N ./your_solver

This template implements *no real wire protocol* -- the send/recv methods
raise NotImplementedError where a real backend would talk MPI -- so it
imports and its tests run in plain Python without mpi4py, a solver binary,
or a GPU. It demonstrates:

  - subclassing `gym.Env` directly (NOT `core.FlowEnv`, which composes
    in-process flow/solver objects that have no meaning across the MPI
    boundary),
  - `ExternalProcessEnvMixin` for the MPMD communicator split and its
    lifecycle contract,
  - the config-schema conventions (canonical key names, pop-before-super,
    unknown-key warning).

See docs/docs/developers/adding-a-solver.md for the full guide, and
`hydrogym/maia/mpmd_interface.py` / `hydrogym/nek/env.py` for production
wire protocols.
"""

import warnings
from typing import Optional

import gymnasium as gym
import numpy as np

from hydrogym.core_external import ExternalProcessEnvMixin


class StubSolverEnv(ExternalProcessEnvMixin, gym.Env):
    """Skeleton controller-side environment for an external solver.

    Class-level protocol knobs (from ExternalProcessEnvMixin) -- override
    if your solver's handshake differs from the defaults:
      CONTROLLER_RANK = 0      # world rank of the RL controller app
      INTERCOMM_TAG = 99       # MPI tag for the intercomm handshake
      MPI_SPLIT_LOG_PREFIX = "[MPI_SPLIT] "
    """

    # Your solver's command-tag map (MAIA and Nek each own theirs -- this
    # is the sanctioned solver-specific escape hatch):
    COMMAND_TAGS = {"timeStep": 0, "termination": 99}

    def __init__(self, env_config: dict = None):
        config = dict(env_config or {})
        self.nproc = int(config.pop("nproc", 4))  # solver worker ranks
        self.hostfile = config.pop("hostfile", None)
        self.mpi_bind_to = config.pop("mpi_bind_to", "none")
        self.num_substeps = int(config.pop("num_substeps", 1))
        self.max_steps = int(config.pop("max_steps", 1000))

        leftover = sorted(config)
        if leftover:
            warnings.warn(
                f"{type(self).__name__} got unsupported env_config key(s) {leftover}; "
                "they have no effect and may indicate a misspelled option.",
                stacklevel=2,
            )

        # Observation/action spaces: sized to your solver's actual layout.
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=float)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

        self.sub_comm = None  # set by launch()
        self.iter = 0

    def launch(self, comm_world=None):
        """Split the MPMD world and establish the controller<->solver comm.

        In production this is called once from the controller process of
        `mpirun -np 1 python ... : -np N ./your_solver`. `_split_mpmd_comm`
        (from ExternalProcessEnvMixin) delegates to
        `hydrogym.core_external.mpi_split`, which validates that
        MPI.COMM_WORLD holds exactly 1 + nproc ranks.
        """
        if comm_world is None:
            from mpi4py import MPI  # imported lazily: only the launch path needs it

            comm_world = MPI.COMM_WORLD
        self.sub_comm = self._split_mpmd_comm(comm_world, nproc=self.nproc)
        return self.sub_comm

    # --- Wire protocol: replace these stubs with your solver's protocol ---
    # (see hydrogym/maia/mpmd_interface.py for a compact real example)

    def _send_command(self, name: str, data: np.ndarray):
        raise NotImplementedError("Replace with your solver's tagged send over self.sub_comm")

    def _recv_observations(self) -> np.ndarray:
        raise NotImplementedError("Replace with your solver's tagged recv over self.sub_comm")

    # --- gymnasium API ---

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.iter = 0
        # A real backend stages config/mesh/restart data here (or in a
        # prepare_<solver>_workspace helper) and hands the solver its
        # initial state. See the MAIA/Nek backends.
        return np.zeros(4, dtype=float), {}

    def step(self, action):
        # A real backend sends the action + a timeStep command, advances
        # the solver num_substeps times, and receives fresh observations.
        obs = self._recv_observations()
        reward = 0.0
        self.iter += self.num_substeps
        # Target semantics (audit Task 5.1): terminated = physics invalid,
        # truncated = step budget reached.
        terminated = False
        truncated = self.iter >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def close(self):
        """REQUIRED lifecycle contract: tell the solver side to terminate
        *before* this process exits, then free the communicator. Skipping
        this deadlocks the rest of the MPMD job. See nek/env.py and
        maia/env_core.py for the two production implementations."""
        if self.sub_comm is not None:
            # e.g. self._send_command("termination", np.zeros(1))
            # then free the communicator:
            self.sub_comm = None
