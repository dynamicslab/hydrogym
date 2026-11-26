"""
NACA 0012 Airfoil Environment
==============================

This module provides NACA 0012 airfoil flow CFD environments for
reinforcement learning, including configurations for steady flow
and gust response control.
"""

import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from hydrogym.maia.env_core import MaiaFlowEnv, register_environment


def gust_field_path(gust_file: str, env_data_path: str, work_dir: Optional[str] = None) -> str:
    """Path of the gust file that m-AIA reads (``lbGustFieldFile``).

    m-AIA opens a relative ``lbGustFieldFile`` in its working directory, i.e. the directory the MPMD job is launched
    from (``work_dir``, default: the current directory of this process; ``prepare_maia_workspace`` links the gust
    files of the case there). If the file is not there, the environment data folder is used (run directory = data
    folder).
    """
    if os.path.isabs(gust_file):
        return gust_file
    in_work = os.path.join(work_dir or os.getcwd(), gust_file)
    if os.path.lexists(in_work):
        return in_work
    return os.path.join(env_data_path, gust_file)


def install_gust_file(src: str, dst: str) -> None:
    """Put the gust file ``src`` at ``dst`` atomically (copy to a temporary file next to ``dst``, then rename).

    m-AIA may read ``dst`` at the same time: it loads the gust file when it starts, while the controller prepares
    the first episode. With the rename it reads either the previous or the new file, never a partly written one. A
    symbolic link at ``dst`` (e.g. into the environment data cache) is replaced; its target is left unchanged.
    """
    folder = os.path.dirname(os.path.abspath(dst))
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".gust_", suffix=".tmp")
    os.close(fd)
    try:
        shutil.copy(src, tmp)  # content and permission bits (the temporary file is created private)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class NACA0012Base(MaiaFlowEnv):
    """
    Base class for NACA 0012 airfoil environments with Hugging Face integration.

    Provides common functionality for NACA airfoil configurations.

    Actuation power (``hydrogym.maia.wall_jets``, one actuator per jet): the three jets of BC 2007 (object class
    1: circular footprint of radius ``lbJetRanges / 2`` about ``lbJetCenter``, fixed direction ``lbJetAngles``) on
    the airfoil wall of the case's geometry file, modelled at the level of the lattice links (the jet centres lie
    inside the airfoil, so which boundary cells carry which velocity matters), including the flow-dependent mass flux
    of the interpolated bounce-back (30 % of the jets' flux in the 2D Re 100 case; with it the model's mass flux
    agrees with a control-volume balance of the solver's flow within 4.4 %); ``C_P = 2 P / (U^3 c [span])`` like
    ``C_D``. The jets are independent (not zero-net-mass-flux); ``p_ref`` is the mean wall pressure over the jets.
    In 3D the section at mid-span is extruded (the in-plane links stand for the D3Q27 links of the extruded wall).
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the NACA 0012 base environment.

        Args:
            env_config: Environment configuration dictionary.
        """
        super().__init__(env_config)

    def drag_coefficient_from_objective(self, obj_dict):
        """Drag coefficient of the airfoil (force coefficient along x, normalised like the reward's)."""
        return float(
            self.compute_nondim_coefficients(
                forces=np.asarray(obj_dict["forces"]),
                density=1.0,
                referenceVelocity=self.Ma / np.sqrt(3),
                projectionLength=self.referenceLength / self.dX * self.zLength / self.dX,
            )[0]
        )


class NACA0012(NACA0012Base):
    """
    NACA 0012 airfoil environment optimizing lift-to-drag ratio.

    This environment controls flow around a NACA 0012 airfoil using
    synthetic jets. The reward maximizes L/D ratio.

    Attributes:
        numJetsInSimulation: Number of jet actuators in the CFD simulation.
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the NACA 0012 environment.

        Args:
            env_config: Environment configuration dictionary.
        """
        super().__init__(env_config)
        self.numJetsInSimulation = self._get_property(self.runtime_property_file_data, "lbNoJets")
        self.configure_observations()
        self.configure_probe_dimensions()
        self.set_observation_action_spaces()

        self.setup_normalization()
        self.setup_actuation_power("wall_jets", geometry="curve")

    def convert_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert RL action to CFD actuation format.

        Args:
            action: Action array from the RL agent.

        Returns:
            Action sequence for the CFD solver.
        """
        return action

    def get_reward(self) -> Tuple[float, Dict]:
        """
        Compute the reward based on lift-to-drag ratio.

        Maximizes L/D ratio. Penalizes heavily for negative lift
        or near-zero drag.

        Returns:
            Tuple containing:
                - reward: L/D ratio (or -100 for invalid conditions)
                - obj_dict: Dictionary with force information
        """
        forces = self.maiaInterface.getForce(self.bcId[0])

        nondim_forces = self.compute_nondim_coefficients(
            forces=forces,
            density=1.0,
            referenceVelocity=self.Ma / np.sqrt(3),
            projectionLength=self.referenceLength / self.dX * self.zLength / self.dX,
        )

        lift = nondim_forces[1]
        drag = nondim_forces[0]

        # Penalize negative lift or near-zero drag
        if drag < 1e-6 or lift < 0:
            reward = -100.0
        else:
            reward = lift / drag

        obj_dict = {"forces": forces}

        return reward, obj_dict


class NACA0012Gust(NACA0012Base):
    """
    NACA 0012 airfoil environment for gust response control.

    This environment controls flow around a NACA 0012 airfoil
    during gust encounters. The reward minimizes deviation from
    unperturbed baseline forces.

    Gust: if the case's ``properties_run.toml`` sets ``lbGustFieldFile`` (m-AIA LB solver), m-AIA superposes the
    gust field of that file onto the (no-gust) restart state every time it loads the restart, i.e. at every
    ``reset()`` (MPMD re-initialisation). The default gust is the file shipped with the case (a streamwise slug).
    Cases without the property contain the gust in the restart state.

    Named gusts shipped with the case as ``gust_<name>.bin``: 'slug' (the default, = the shipped gust file) and
    'transverse' (file ``gust_fig2c.bin``, also selectable as 'fig2c': a one-piece finite transverse gust, the gust
    of Fig. 2c of the HydroGym paper; uncontrolled C_L peak ~4.3 about 3.6 c/U into the episode). Select with
    ``env_config['gust']`` or use the names in ``gust_files`` / ``reset(options={'gust_file': name})``.

    Attributes:
        unperturbed_avg_drag: Baseline drag coefficient without gust.
        unperturbed_avg_lift: Baseline lift coefficient without gust.
        numJetsInSimulation: Number of jet actuators in the CFD simulation.
        gust_field_file: Path of the gust file m-AIA reads (``lbGustFieldFile`` in its working directory, see
            ``gust_field_path``; None for cases without ``lbGustFieldFile``). The selected gust is copied
            there atomically before each reset (``install_gust_file``).
    """

    def __init__(self, env_config: Dict):
        """
        Initialize the NACA 0012 gust environment.

        Args:
            env_config: Environment configuration dictionary. Gust options (cases with ``lbGustFieldFile`` only):
                - gust_files (str or list of str): gust files (MAIAGUST format) used instead of the shipped
                  default; before each reset the next one is copied over the case's gust file (cyclic, or random
                  with ``gust_sampling='random'``).
                - gust_sampling (str): 'cycle' (default) or 'random'.
                - gust_seed (int): seed of the random choice.
                - gust (str): a named gust of the case ('slug' = default, 'transverse' = 'fig2c'), i.e.
                  gust_files=[name].
                Entries of gust_files (and options['gust_file']) may be names: <case>/gust_<name>.bin.
        """
        super().__init__(env_config)
        self.unperturbed_avg_drag = self.cfg.maia.unperturbed_avg_drag
        self.unperturbed_avg_lift = self.cfg.maia.unperturbed_avg_lift

        gf = self.runtime_property_file_data.get("lbGustFieldFile")
        self.gust_field_file: Optional[str] = gust_field_path(gf, self.env_data_path) if gf else None
        files = env_config.get("gust_files")
        if files is None and env_config.get("gust") is not None:
            files = [env_config["gust"]]
        self.gust_files: List[str] = [files] if isinstance(files, str) else list(files or [])
        self.gust_files = [self._gust_path(f) for f in self.gust_files]
        if self.gust_files and self.gust_field_file is None:
            raise ValueError(
                "gust_files given, but the case's properties_run.toml has no lbGustFieldFile "
                "(gust superposition needs a case with a no-gust restart and the m-AIA gust-field option)"
            )
        for f in self.gust_files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"gust file not found: {f}")
        self.gust_sampling = env_config.get("gust_sampling", "cycle")
        if self.gust_sampling not in ("cycle", "random"):
            raise ValueError(f"gust_sampling must be 'cycle' or 'random', not {self.gust_sampling!r}")
        self._gust_rng = np.random.default_rng(env_config.get("gust_seed"))
        self._gust_episode = 0
        self.current_gust_file: Optional[str] = self.gust_field_file

        self.numJetsInSimulation = self._get_property(self.runtime_property_file_data, "lbNoJets")

        self.configure_observations()
        self.configure_probe_dimensions()
        self.set_observation_action_spaces()

        self.setup_normalization()
        self.setup_actuation_power("wall_jets", geometry="curve")

    GUST_ALIASES = {"transverse": "fig2c"}  # descriptive name -> file name of the gust shipped with the case

    def _gust_path(self, f: str) -> str:
        """A gust file path, or a named gust of the case (<case>/gust_<name>.bin, e.g. 'slug', 'transverse')."""
        if os.path.isfile(f):
            return f
        named = os.path.join(self.env_data_path, f"gust_{self.GUST_ALIASES.get(f, f)}.bin")
        if os.sep not in f and os.path.isfile(named):
            return named
        return f

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment; with ``gust_files`` the episode's gust file is put in place first (m-AIA reads it
        when it reloads the restart). ``options={'gust_file': path}`` sets the gust of this episode explicitly.
        """
        path = (options or {}).get("gust_file")
        if path is not None:
            path = self._gust_path(path)
        if path is None and self.gust_files:
            if self.gust_sampling == "random":
                path = self.gust_files[int(self._gust_rng.integers(len(self.gust_files)))]
            else:
                path = self.gust_files[self._gust_episode % len(self.gust_files)]
        if path is not None:
            if self.gust_field_file is None:
                raise ValueError("options['gust_file'] needs a case with lbGustFieldFile")
            if os.path.abspath(path) != os.path.abspath(self.gust_field_file):
                if (
                    self.gust_field_file
                    and os.path.exists(self.gust_field_file)
                    and not os.path.exists(self.gust_field_file + ".default")
                ):
                    shutil.copyfile(self.gust_field_file, self.gust_field_file + ".default")
                install_gust_file(path, self.gust_field_file)
            self.current_gust_file = path
        self._gust_episode += 1
        return super().reset(seed=seed, options=options)

    def convert_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert RL action to CFD actuation format.

        Args:
            action: Action array from the RL agent.

        Returns:
            Action sequence for the CFD solver.
        """
        return action

    def get_reward(self) -> Tuple[float, Dict]:
        """
        Compute the reward based on deviation from baseline forces.

        Minimizes deviation of lift and drag from unperturbed values.

        Returns:
            Tuple containing:
                - reward: Negative weighted deviation from baseline
                - obj_dict: Dictionary with force information
        """
        forces = self.maiaInterface.getForce(self.bcId[0])

        nondim_forces = self.compute_nondim_coefficients(
            forces=forces,
            density=1.0,
            referenceVelocity=self.Ma / np.sqrt(3),
            projectionLength=self.referenceLength / self.dX * self.zLength / self.dX,
        )

        obj_dict = {"forces": forces}

        reward = -np.abs(nondim_forces[1] - self.unperturbed_avg_lift) - self.omega * np.abs(
            nondim_forces[0] - self.unperturbed_avg_drag
        )

        return reward, obj_dict


# Register environment types with the factory
register_environment("NACA0012", NACA0012)
register_environment("NACA0012Gust", NACA0012Gust)
