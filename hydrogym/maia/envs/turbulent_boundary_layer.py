"""
Turbulent Boundary Layer Environments
======================================

This module provides zero-pressure-gradient turbulent boundary layer (ZPG TBL)
CFD environments for reinforcement learning.

Two actuation strategies share the ``ZPGTBLBase`` class:

``ZPGTBLJet``
    Jet-based actuation.  Actions lie in ``[−MAX_CONTROL, +MAX_CONTROL]``
    and are passed directly to the CFD solver.

``ZPGTBLSurfaceWave``
    Actuated traveling surface wave parameterized by
    ``[amplitude, speed, wavelength]``.  All three parameters are strictly
    positive, which requires a per-action asymmetric action space and a
    non-zero reset; see the class docstring for config details.

Both environments receive the same solver force output
``[F_x, F_y, F_z, C_P, area]`` (``nDim + 2`` elements), where ``C_P`` is
the power coefficient (``0.0`` for the jet case) and ``area`` is the wetted
wall area used for normalisation.  Forces and power are pre-normalised by
the solver by dynamic pressure (``q_∞ = ρ_∞ · ½ · U_∞²``) and
``q_∞ · U_∞`` respectively, so dividing by ``area`` yields the final
dimensionless coefficients.  Observation sizing and normalization for this
extended force vector are handled in ``ZPGTBLBase``.
"""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from einops import rearrange

from hydrogym.maia.env_core import ConfigError, MaiaFlowEnv, register_environment

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ZPGTBLBase(MaiaFlowEnv):
    """
    Base class for zero-pressure-gradient turbulent boundary layer environments.

    Uses the structured-grid m-AIA solver (``MAIA_STRCTRD``).

    Both actuation variants (jet and surface wave) receive the solver force
    output ``[F_x, F_y, F_z, C_P, area]`` (``nDim + 2`` elements).
    Forces ``F_x … F_z`` and ``C_P`` are pre-normalised by the solver using
    the dynamic pressure ``q_∞ = ρ_∞ · ½ · U_∞²`` (forces) and
    ``q_∞ · U_∞`` (power), so dividing by the returned ``area`` gives the
    final dimensionless drag and power coefficients.

    This class provides:

    * ``configure_observations()`` – sizes the force observation slot to
      ``nDim + 2`` per boundary.
    * ``setup_normalization()`` – handles the ``C_P`` and ``area`` elements
      in the ``'U_inf'`` normalization strategy.
    * ``get_reward()`` – unified reward for both actuation variants:

      .. math::

          R = -C_D - \\omega \\cdot C_P

      where ``C_D = F_x / area`` and ``C_P = forces[3] / area``.
      For the jet case ``C_P = 0``, so the reward reduces to ``-C_D``.
    """

    SOLVER_TYPE: str = "MAIA_STRCTRD"

    def __init__(self, env_config: Dict):
        super().__init__(env_config)

        # ------------------------------------------------------------------
        # Free-stream conditions from isentropic relations.
        # gamma and angle = [alpha, beta] (degrees) are read from the TOML
        # property file.  All quantities are non-dimensional with stagnation
        # values as references (T_0 = rho_0 = p_0·gamma = 1).
        # ------------------------------------------------------------------
        gamma = float(self._get_property(self.runtime_property_file_data, "gamma"))
        angle = self._get_property(self.runtime_property_file_data, "angle")  # [alpha, beta] in degrees
        alpha = np.deg2rad(float(angle[0]))
        beta = np.deg2rad(float(angle[1]))

        # Isentropic static temperature  T_inf = 1 / (1 + ½·(γ-1)·Ma²)
        T_inf = 1.0 / (1.0 + 0.5 * (gamma - 1.0) * self.Ma**2)

        # Total velocity magnitude  U_T = Ma·√T_inf
        self.U_T = self.Ma * np.sqrt(T_inf)

        # Velocity components aligned with the flow direction
        self.U_inf = self.U_T * np.cos(alpha) * np.cos(beta)
        self.V_inf = self.U_T * np.sin(alpha) * np.cos(beta)
        self.W_inf = self.U_T * np.sin(beta)

        # Isentropic static pressure and density
        self.p_inf = T_inf ** (gamma / (gamma - 1.0)) / gamma
        self.rho_inf = T_inf ** (1.0 / (gamma - 1.0))

    def configure_observations(self) -> None:
        """
        Configure the number of observations.

        Overrides the base class to account for the extended solver force
        output ``[F_x, F_y, F_z, C_P, area]`` (``nDim + 2`` elements per
        boundary instead of ``nDim``).
        """
        self.num_outputs = 0
        self.num_probes = int(len(self.probe_locations) / self.nDim)

        if "forces" in self.observation_type:
            # Solver returns [F_x, F_y, F_z, C_P, area]; only force components used
            self.num_outputs += self.nDim * len(self.bcId)
        if "u" in self.observation_type:
            self.num_outputs += self.num_probes
        if "v" in self.observation_type:
            self.num_outputs += self.num_probes
        if "w" in self.observation_type:
            self.num_outputs += self.num_probes
        if "rho" in self.observation_type:
            self.num_outputs += self.num_probes
        if "p" in self.observation_type:
            self.num_outputs += self.num_probes

    def setup_normalization(self) -> None:
        """
        Set up observation normalization.

        Overrides the base class to handle the extended force vector
        ``[F_x, F_y, F_z, C_P, area]`` (``nDim + 2`` elements per boundary).

        For the ``'U_inf'`` strategy, free-stream values computed from the
        isentropic relations in ``__init__`` are used:

        * velocities scaled by ``U_T`` (total free-stream speed),
        * density by ``rho_inf``,
        * pressure by the dynamic pressure ``q_inf = ½ · rho_inf · U_T²``.

        Force-slot entries use ``loc = 0, scale = 1`` since the solver
        already normalises them by ``q_inf`` (forces) / ``q_inf·U_T`` (power).
        """
        if self.obs_normalization_strategy == "U_inf":
            obs_loc, obs_scale = [], []
            q_inf = 0.5 * self.rho_inf * self.U_T**2

            if "u" in self.observation_type:
                obs_loc.append([0.0] * self.noProbes)
                obs_scale.append([self.U_T] * self.noProbes)
            if "v" in self.observation_type:
                obs_loc.append([0.0] * self.noProbes)
                obs_scale.append([self.U_T] * self.noProbes)

            if self.nDim == 2:
                if "rho" in self.observation_type:
                    obs_loc.append([0.0] * self.noProbes)
                    obs_scale.append([self.rho_inf] * self.noProbes)
                if "p" in self.observation_type:
                    obs_loc.append([0.0] * self.noProbes)
                    obs_scale.append([q_inf] * self.noProbes)
            elif self.nDim == 3:
                if "w" in self.observation_type:
                    obs_loc.append([0.0] * self.noProbes)
                    obs_scale.append([self.U_T] * self.noProbes)
                if "rho" in self.observation_type:
                    obs_loc.append([0.0] * self.noProbes)
                    obs_scale.append([self.rho_inf] * self.noProbes)
                if "p" in self.observation_type:
                    obs_loc.append([0.0] * self.noProbes)
                    obs_scale.append([q_inf] * self.noProbes)
            else:
                print(f"WARNING: nDim = {self.nDim} > 3. Something must be wrong!")

            if "forces" in self.observation_type:
                for _ in range(len(self.bcId)):
                    # [F_x, F_y, F_z, C_P, area]: all already solver-normalised
                    obs_loc.append([0.0] * (self.nDim + 2))
                    obs_scale.append([1.0] * (self.nDim + 2))

            self.obs_loc = np.concatenate(obs_loc)
            self.obs_scale = np.concatenate(obs_scale)

        elif self.obs_normalization_strategy == "none":
            self.obs_loc = np.zeros(self.num_outputs)
            self.obs_scale = np.ones(self.num_outputs)

        elif self.obs_normalization_strategy == "customized":
            if self.obs_loc is None or self.obs_scale is None:
                raise ConfigError(
                    "obs_normalization_strategy='customized' requires both 'obs_loc' and 'obs_scale' in env_config"
                )
            if len(self.obs_loc) != self.num_outputs or len(self.obs_scale) != self.num_outputs:
                raise ConfigError(
                    f"Customized normalization dimension mismatch: "
                    f"obs_loc ({len(self.obs_loc)}), "
                    f"obs_scale ({len(self.obs_scale)}), "
                    f"num_outputs ({self.num_outputs})"
                )

        elif self.obs_normalization_strategy == "probewise_mean_std":
            print("WARNING: Selected Obs Normalization strategy: probewise_mean_std")
            print("Computing normalization factors now...")
            self.compute_normalization_factors()
            print("Computed loc values:", self.obs_loc.tolist(), flush=True)
            print("Computed scale values:", self.obs_scale.tolist(), flush=True)

        else:
            raise ConfigError(f"Invalid obs_normalization_strategy: '{self.obs_normalization_strategy}'.")

    def get_reward(self) -> Tuple[float, Dict]:
        """
        Compute the unified reward: :math:`R = -C_D - \\omega \\cdot C_P`.

        The solver returns ``[F_x, F_y, F_z, C_P, area]``.  Forces and
        power are pre-normalised by ``q_∞`` and ``q_∞ · U_∞`` respectively;
        dividing by ``area`` yields the final dimensionless coefficients::

            C_D  = forces[0] / area
            C_P  = forces[3] / area

        For the jet environment ``C_P = 0`` and the reward reduces to
        ``-C_D``.

        Returns:
            Tuple of ``(reward, info_dict)``.  ``info_dict`` contains
            ``'forces'``, ``'C_D'``, and ``'C_P'``.
        """
        forces = self.maiaInterface.getForce(self.bcId[0])  # [F_x, F_y, F_z, C_P, area]
        print("Forces:", forces)
        area = float(forces[4])
        # avoid scaling by here due to vanishing rewards
        C_D = float(forces[0])
        C_P = float(forces[3])
        # fully non-dimensionalized as follows
        # C_D = float(forces[0]) / area
        # C_P = float(forces[3]) / area
        print("-CD", -C_D, "-CP", -C_P, "area", area)

        reward = -C_D - self.omega * C_P
        return reward, {"forces": forces, "C_D": C_D, "C_P": C_P}


# ---------------------------------------------------------------------------
# Jet actuation
# ---------------------------------------------------------------------------


class ZPGTBLJet(ZPGTBLBase):
    """
    ZPG turbulent boundary layer with jet-based flow control.

    Uses the same actuation setup as the :class:`~hydrogym.maia.envs.Cube`
    environment: actions lie in ``[−MAX_CONTROL, +MAX_CONTROL]`` (scaled by
    the base-class ``step()``) and are passed directly to the CFD solver.
    The number of jet actuators is read from ``lbNoJets`` in the property file.

    The solver returns ``[F_x, F_y, F_z, C_P, area]`` with ``C_P = 0.0``,
    so the reward reduces to ``-C_D``.  Observation sizing, normalization,
    and reward computation are all provided by :class:`ZPGTBLBase`.

    Attributes:
        numJetsInSimulation (int): Number of jet boundary conditions.
    """

    def __init__(self, env_config: Dict):
        super().__init__(env_config)

        self.numJetsInSimulation = self._get_property(self.runtime_property_file_data, "fvNoJets")

        self.configure_observations()
        self.configure_probe_dimensions()
        self.set_observation_action_spaces()
        self.setup_normalization()

    def convert_action(self, action: np.ndarray) -> np.ndarray:
        """
        Pass the (already ``MAX_CONTROL``-scaled) jet actuation to the solver.

        Args:
            action: Action array scaled by ``MAX_CONTROL``.

        Returns:
            Action array for the CFD solver.
        """
        return action


# ---------------------------------------------------------------------------
# Surface-wave actuation
# ---------------------------------------------------------------------------


class ZPGTBLSurfaceWave(ZPGTBLBase):
    """
    ZPG turbulent boundary layer with actuated traveling surface wave.

    The surface is driven by a traveling wave with three strictly positive
    control parameters:

    ============= =============================================
    Parameter     Description
    ============= =============================================
    amplitude     Wave amplitude
    speed         Wave propagation speed
    wavelength    Spatial period of the wave
    ============= =============================================

    **Differences from** :class:`ZPGTBLJet`:

    * **Action space** – Per-action ``[lower_bound, upper_bound]`` read from
      ``maia.action_lower_bounds`` / ``maia.action_upper_bounds`` in the
      config.  The ``MAX_CONTROL`` scaling in ``step()`` is *not* applied.
    * **Reset** – Uses ``[amplitude_init, speed_init, wavelength_init]`` from
      config (zeros would crash the CFD solver).
    * **Reward** – :math:`R = -C_D - \\omega \\cdot C_P` (shared with
      :class:`ZPGTBLJet` via :class:`ZPGTBLBase`; here ``C_P > 0``
      because the surface wave performs work on the fluid).

    Required additional entries in the ``maia`` section of the YAML config:

    .. code-block:: yaml

        maia:
          amplitude_init:      0.05
          speed_init:          1.0
          wavelength_init:     2.0
          action_lower_bounds: [0.01, 0.1, 0.5]
          action_upper_bounds: [0.2,  3.0, 10.0]
          omega:               0.1
    """

    def __init__(self, env_config: Dict):
        super().__init__(env_config)

        # ------------------------------------------------------------------
        # Initial control values for reset (must be strictly positive)
        # ------------------------------------------------------------------
        self.amplitude_init = float(self.cfg.maia.amplitude_init)
        self.speed_init = float(self.cfg.maia.speed_init)
        self.wavelength_init = float(self.cfg.maia.wavelength_init)
        self.init_action = np.array(
            [self.amplitude_init, self.speed_init, self.wavelength_init],
            dtype=float,
        )

        # ------------------------------------------------------------------
        # Per-action asymmetric bounds
        # ------------------------------------------------------------------
        self.action_lower_bounds = np.array(self.cfg.maia.action_lower_bounds, dtype=float)
        self.action_upper_bounds = np.array(self.cfg.maia.action_upper_bounds, dtype=float)

        print("action lower boundary", self.action_lower_bounds)
        print("action upper boundary", self.action_upper_bounds)

        if len(self.action_lower_bounds) != self.num_inputs:
            raise ConfigError(
                f"action_lower_bounds length ({len(self.action_lower_bounds)}) "
                f"must match num_inputs ({self.num_inputs})"
            )
        if len(self.action_upper_bounds) != self.num_inputs:
            raise ConfigError(
                f"action_upper_bounds length ({len(self.action_upper_bounds)}) "
                f"must match num_inputs ({self.num_inputs})"
            )
        if len(self.init_action) != self.num_inputs:
            raise ConfigError(
                f"[amplitude_init, speed_init, wavelength_init] has "
                f"{len(self.init_action)} elements but num_inputs is {self.num_inputs}"
            )

        # ------------------------------------------------------------------
        # Configure observations and spaces
        # (configure_observations / setup_normalization inherited from base)
        # ------------------------------------------------------------------
        self.configure_observations()
        self.configure_probe_dimensions()
        self.set_observation_action_spaces()
        self.setup_normalization()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def set_observation_action_spaces(self) -> None:
        """
        Override to use a normalized ``[0, 1]`` action space per parameter.

        The RL agent always operates in ``[0, 1]``; physical values are
        recovered inside :meth:`convert_action` via the per-parameter affine
        mapping ``lower + action * (upper - lower)``.
        """
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_outputs,),
            dtype=float,
        )

        self.action_space = gym.spaces.Box(
            low=np.zeros(self.num_inputs, dtype=float),
            high=np.ones(self.num_inputs, dtype=float),
            dtype=float,
        )

    def convert_action(self, action: np.ndarray) -> List[float]:
        """
        Scale a normalized ``[0, 1]`` action to physical wave parameters.

        Applies the per-parameter affine map::

            physical[i] = lower[i] + action[i] * (upper[i] - lower[i])

        Args:
            action: Normalized action array with values in ``[0, 1]``,
                corresponding to ``[amplitude, speed, wavelength]``.

        Returns:
            Physical actuation values for the CFD solver.
        """
        physical = self.action_lower_bounds + np.asarray(action, dtype=float) * (
            self.action_upper_bounds - self.action_lower_bounds
        )
        print("converted actions:", physical)
        return list(physical)

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance the environment by one step.

        Overrides the base class to skip ``MAX_CONTROL`` scaling.  The
        incoming ``action`` is in ``[0, 1]`` and is converted to physical
        units by :meth:`convert_action`.

        Args:
            action: Normalized action in ``[0, 1]`` for
                ``[amplitude, speed, wavelength]``.

        Returns:
            Tuple of ``(observation, reward, terminated, truncated, info)``.
        """
        # convert_action maps [0, 1] → physical; no MAX_CONTROL scaling
        self.maiaInterface.runTimeSteps(self.num_substeps_per_iteration)
        self.maiaInterface.setControlProperties(self.convert_action(action=action))

        self.probeData = self.maiaInterface.getProbeData(self.probe_locations)
        self.probeData = rearrange(self.probeData, "(n p) -> n p", n=self.noProbes)

        self.obs = self._collect_observations()
        reward, _ = self.get_reward()
        self.obs = (self.obs - self.obs_loc) / self.obs_scale

        self.iter += 1
        done = self.check_complete()
        self.maiaInterface.continueRun()

        return self.obs, reward, bool(done), bool(done), {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment with physically valid initial wave parameters.

        Overrides the base-class reset to avoid setting zero control actions,
        which would crash the CFD solver for the surface-wave boundary
        condition.  ``init_action`` is passed in physical units directly,
        bypassing the ``[0, 1]`` → physical scaling used during normal steps.

        Args:
            seed: Optional random seed.
            options: Optional reset options (unused).

        Returns:
            Tuple of ``(initial_observation, info)``.
        """
        print("Resetting environment", flush=True)

        self.maiaInterface.runTimeSteps(1)
        self.maiaInterface.reinit()

        # Pass physical init values directly — init_action is in physical
        # units, not the [0, 1] normalized space used by the agent.
        self.maiaInterface.setControlProperties(list(self.init_action))
        print("reset init action", self.init_action)

        self.probeData = self.maiaInterface.getProbeData(self.probe_locations)
        self.probeData = rearrange(self.probeData, "(n p) -> n p", n=self.noProbes)

        self.obs = self._collect_observations()
        self.obs = (self.obs - self.obs_loc) / self.obs_scale

        self.iter = 0
        self.maiaInterface.continueRun()

        return self.obs, {}

    def _collect_observations(self) -> np.ndarray:
        """
        Collect and concatenate all raw observation components.

        Used by the overridden :meth:`step` and :meth:`reset` to share
        observation-gathering logic.  Appends the full ``nDim + 2`` force
        vector ``[F_x, F_y, F_z, C_P, area]`` per boundary.

        Returns:
            Concatenated raw (un-normalized) observation array.
        """
        obs = []

        if "u" in self.observation_type:
            obs.append(self.probeData[:, 0])
        if "v" in self.observation_type:
            obs.append(self.probeData[:, 1])

        if self.nDim == 2:
            if "rho" in self.observation_type:
                obs.append(self.probeData[:, 2])
            if "p" in self.observation_type:
                obs.append(self.probeData[:, 3])
        elif self.nDim == 3:
            if "w" in self.observation_type:
                obs.append(self.probeData[:, 2])
            if "rho" in self.observation_type:
                obs.append(self.probeData[:, 3])
            if "p" in self.observation_type:
                obs.append(self.probeData[:, 4])
        else:
            print(f"WARNING: nDim = {self.nDim} > 3. Something must be wrong!")

        if "forces" in self.observation_type:
            for bc_id in self.bcId:
                obs.append(self.maiaInterface.getForce(bc_id)[: self.nDim])

        return np.concatenate(obs)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_environment("ZPGTBLJet", ZPGTBLJet)
register_environment("ZPGTBLSurfaceWave", ZPGTBLSurfaceWave)
