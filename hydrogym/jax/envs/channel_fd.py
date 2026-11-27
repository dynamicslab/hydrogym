"""Turbulent channel flow control environment on the finite-difference solver (``hydrogym.jax.solvers.channel_fd``).

``ChannelFlowFDEnv`` is the gymnax-style environment of the three-dimensional channel at ``Re_tau = 180``
(domain ``2 pi x pi x 2``, ``72^3`` cells, wall-normal ``tanh`` stretching). ``ChannelFlowEnv(env_config)``
selects between this environment (``env_config["solver"] = "fd"``, the default) and the earlier pseudo-spectral
channel (``"spectral"``, :class:`hydrogym.jax.envs.channel.ChannelFlowSpectralEnv`).

Setup
-----
* Flow: streamwise ``x``, spanwise ``y``, wall-normal ``z``; periodic in ``x`` and ``y``, no slip at ``z = 0`` and
  ``z = 2``; driven by the body force ``8 nu^2 Re_tau^2 / H^3``; dynamic time step for a maximum Courant number of
  ``cfl`` (default 0.9). All arrays have the shape ``(nx, ny, nz)``.
* Jets: ``nx_jets x ny_jets`` (default ``6 x 4``) zero-mass-flux jets on the bottom wall. Each jet is a
  wall-normal velocity with a parabolic profile ``w = gain * a * (1 - r^2 / R^2)`` on a circular footprint of radius
  ``R`` (``jet_radius``), zero at its edge. The centres follow the layout of the pseudo-spectral channel
  (``linspace(3, int(2 nx / 3) - 4, nx_jets)`` cells in ``x``, ``linspace(4, ny - 5, ny_jets)`` cells in ``y``). The
  action ``a`` in ``[-1, 1]`` of each jet is held for ``nsteps`` solver steps (default 100); positive action is
  suction (``w < 0`` at the wall). The net mass flux through the wall is set to zero by subtracting the area-weighted
  mean action from all jets, so the jets keep exactly zero velocity at their edges. ``jet_amplitude`` (default
  0.0624) is the peak jet velocity per unit action in units of the nominal friction velocity ``sqrt(g_x H / 2)``
  (``jet_gain`` sets it in absolute units instead).
* Observation: the streamwise velocity ``u`` at the jet centres on the wall-parallel plane of cell centres closest to
  ``y+ = obs_yplus`` (default 15, using the nominal friction velocity ``sqrt(g_x H / 2)``); ``nx_jets * ny_jets``
  values.
* Reward: ``tau_ref / tau_w`` (``reward="ratio"``, default) or ``-tau_w`` (``reward="neg_tau"``), where ``tau_w =
  nu <u>(z_1) / z_1`` is the plane mean over the controlled (bottom) wall of the wall shear stress from the first cell
  at the end of the action interval (``z_1`` in the viscous sub-layer), and ``tau_ref`` (``tau_ref``, default the value
  of the validated uncontrolled flow) the uncontrolled reference.
* Actuation power: ``P = sum w_jet (p - <p> + w_jet^2 / 2) dA`` (energy flux through the wall; the wall pressure is
  the pressure of the last projection of the step divided by the time step), per jet footprint, normalised by
  ``U_b^3 Lx Ly / 2`` (``u_bulk_ref``); ``power_weight`` (default 0) charges it in the reward (``hydrogym.power``).
* Initial state: ``initial_condition="turbulent"`` (default) loads ``U.npy``, ``V.npy``, ``W.npy`` from
  ``initial_field_dir`` (or from the ``Channel_3D_Retau180_JAXFD`` data of the environment hub); ``"laminar"``
  builds, at every reset, a parabolic laminar profile of bulk velocity ``u_bulk_init`` plus a random perturbation of
  relative amplitude ``perturbation`` (such a flow needs many time units to become turbulent).
* Precision: ``dtype`` is ``"float32"`` (default) or ``"float64"``; the latter needs
  ``jax.config.update("jax_enable_x64", True)`` before the environment is built. The default pressure solver is the
  exact ``"direct"`` one; ``poisson_solver="bicgstab"`` (lineax) needs ``dtype="float64"``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces

from hydrogym.jax.env_core import EnvParams as BaseEnvParams
from hydrogym.jax.env_core import JAXFlowEnvBase
from hydrogym.jax.solvers.channel_fd import ChannelFD, ChannelFDConfig, ChannelFDState
from hydrogym.jax.utils.power import power_info_jax
from hydrogym.power import PowerConfig, power_forms

ENVIRONMENT_NAME = "Channel_3D_Retau180_JAXFD"

# Plane-mean wall shear stress (bottom wall, first cell) of the validated uncontrolled flow and its bulk velocity.
TAU_REF_DEFAULT = 0.1166
U_BULK_REF_DEFAULT = 5.34


@struct.dataclass
class ChannelFDEnvParams(BaseEnvParams):
    """Parameters of the finite-difference channel environment."""

    action_dim: int = 24
    nsteps: int = 100  # solver steps per action
    max_steps_in_episode: int = 5000  # solver steps per episode
    min_action: float = -1.0
    max_action: float = 1.0


@struct.dataclass
class ChannelFDEnvState(environment.EnvState):
    """Environment state.

    Attributes:
        time: Solver steps elapsed in the episode.
        U, V, W: Streamwise, spanwise and wall-normal velocity, shape ``(nx, ny, nz)``.
        P: Pressure of the last projection (warm start of the next Poisson solve).
        dt: Last time step.
        t: Physical time elapsed in the episode.
        terminal: Whether the episode has ended.
    """

    time: int
    U: jnp.ndarray
    V: jnp.ndarray
    W: jnp.ndarray
    P: jnp.ndarray
    dt: jnp.ndarray
    t: jnp.ndarray
    terminal: jnp.bool_


def jet_centres(nx: int, ny: int, nx_jets: int, ny_jets: int):
    """Jet centres in cell units (cell-centre index ``i`` is the coordinate ``i + 0.5``)."""
    x_max = int(nx * (2 / 3))
    xc = np.linspace(3.0, float(x_max - 4), nx_jets)
    yc = np.linspace(4.0, float(ny - 5), ny_jets)
    return xc, yc


class ChannelFlowFDEnv(JAXFlowEnvBase):
    """Turbulent channel flow environment on the finite-difference solver (see the module docstring).

    ``env_config`` keys (defaults in parentheses): ``Nx, Ny, Nz`` (72), ``Lx, Ly, Lz`` (2 pi, pi, 2), ``nu`` (1.9e-3),
    ``Re_tau`` (180), ``ctanh`` (0.05), ``cfl`` (0.9), ``poisson_solver`` (``"direct"``), ``pressure_tol`` (1e-3),
    ``pressure_maxit`` (2000), ``dtype`` (``"float32"``), ``nsteps`` (100), ``max_steps_in_episode`` (5000),
    ``nx_jets``/``ny_jets`` (6/4), ``jet_radius`` (0.25), ``jet_amplitude`` (0.0624 friction velocities; ``jet_gain``:
    absolute),
    ``obs_yplus`` (15), ``reward``
    (``"ratio"``), ``tau_ref`` and ``u_bulk_ref`` (validated uncontrolled values), ``initial_condition``
    (``"turbulent"``), ``initial_field_dir``, ``u_bulk_init``/``perturbation`` (laminar start), and the power options
    ``power_weight``, ``power_cost``, ``power_sampling`` (``"mean"``).
    """

    def __init__(self, env_config: Optional[Dict] = None):
        env_config = dict(env_config or {})
        super().__init__(env_config)
        c = env_config
        cfg = ChannelFDConfig(
            nx=int(c.get("Nx", 72)),
            ny=int(c.get("Ny", 72)),
            nz=int(c.get("Nz", 72)),
            lx=float(c.get("Lx", 2.0 * math.pi)),
            ly=float(c.get("Ly", math.pi)),
            lz=float(c.get("Lz", 2.0)),
            nu=float(c.get("nu", 1.9e-3)),
            re_tau=float(c.get("Re_tau", 180.0)),
            ctanh=float(c.get("ctanh", 5e-2)),
            cfl=float(c.get("cfl", 0.9)),
            poisson_solver=str(c.get("poisson_solver", "direct")),
            preconditioner=str(c.get("preconditioner", "none")),
            pressure_tol=float(c.get("pressure_tol", 1e-3)),
            pressure_maxit=int(c.get("pressure_maxit", 2000)),
            dtype=str(c.get("dtype", "float32")),
        )
        self.cfg = cfg
        self.solver = ChannelFD(cfg)
        self.dtype = self.solver.dtype
        self.Nx, self.Ny, self.Nz = cfg.nx, cfg.ny, cfg.nz
        self.nsteps = int(c.get("nsteps", 100))
        self.max_steps_in_episode = int(c.get("max_steps_in_episode", 5000))
        self.nx_jets = int(c.get("nx_jets", 6))
        self.ny_jets = int(c.get("ny_jets", 4))
        self.n_jets = self.nx_jets * self.ny_jets
        self.jet_radius = float(c.get("jet_radius", 0.25))
        self.jet_amplitude = float(c.get("jet_amplitude", 0.0624))  # peak velocity per unit action / u_tau
        u_tau_nominal = math.sqrt(self.solver.gx * cfg.lz / 2.0)
        self.jet_gain = float(c["jet_gain"]) if "jet_gain" in c else self.jet_amplitude * u_tau_nominal
        self.obs_yplus = float(c.get("obs_yplus", 15.0))
        self.reward_type = str(c.get("reward", "ratio"))
        if self.reward_type not in ("ratio", "neg_tau"):
            raise ValueError(f"reward must be 'ratio' or 'neg_tau', got {self.reward_type!r}")
        self.tau_ref = float(c.get("tau_ref", TAU_REF_DEFAULT))
        self.u_bulk_ref = float(c.get("u_bulk_ref", U_BULK_REF_DEFAULT))
        self.initial_condition = str(c.get("initial_condition", "turbulent"))
        if self.initial_condition not in ("turbulent", "laminar"):
            raise ValueError(f"initial_condition must be 'turbulent' or 'laminar', got {self.initial_condition!r}")
        self.u_bulk_init = float(c.get("u_bulk_init", 5.985))
        self.perturbation = float(c.get("perturbation", 0.4))

        self._setup_jets()
        self._setup_observation()
        self.power_config = PowerConfig.from_config(env_config)
        self.power_sampling = self.power_config.resolve_sampling("mean")
        self._power_ref = 0.5 * self.u_bulk_ref**3 * cfg.lx * cfg.ly
        self._wall_dA = self.solver.dx_cell * self.solver.dy_cell

        self.U0 = self.V0 = self.W0 = None
        if self.initial_condition == "turbulent":
            self._load_initial_field(c)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _setup_jets(self):
        s = self.solver
        cfg = self.cfg
        xc, yc = jet_centres(self.Nx, self.Ny, self.nx_jets, self.ny_jets)
        xc = (xc + 0.5) * s.dx_cell
        yc = (yc + 0.5) * s.dy_cell
        X, Y = np.meshgrid(s.x, s.y, indexing="ij")
        profiles = np.zeros((self.n_jets, self.Nx, self.Ny))
        for i in range(self.nx_jets):
            for j in range(self.ny_jets):
                dxp = np.abs(X - xc[i])
                dyp = np.abs(Y - yc[j])
                dxp = np.minimum(dxp, cfg.lx - dxp)  # periodic distance
                dyp = np.minimum(dyp, cfg.ly - dyp)
                r2 = (dxp**2 + dyp**2) / self.jet_radius**2
                profiles[i * self.ny_jets + j] = np.clip(1.0 - r2, 0.0, None)
        self._profiles = jnp.asarray(profiles, dtype=self.dtype)  # (n_jets, nx, ny)
        integral = profiles.sum(axis=(1, 2))
        if np.any(integral <= 0):
            raise ValueError("a jet footprint contains no cell centre: increase jet_radius or the resolution")
        self._jet_integral = jnp.asarray(integral, dtype=self.dtype)
        self._jet_labels = jnp.asarray(
            (profiles > 0).astype(np.float64), dtype=self.dtype
        )  # footprints, (n_jets, nx, ny)
        self._xc_idx = np.floor(xc / s.dx_cell).astype(int)
        self._yc_idx = np.floor(yc / s.dy_cell).astype(int)

    def _setup_observation(self):
        s = self.solver
        u_tau_nom = math.sqrt(s.gx * self.cfg.lz / 2.0)
        y_obs = self.obs_yplus * self.cfg.nu / u_tau_nom
        self.k_obs = int(np.argmin(np.abs(s.zc - y_obs)))
        self.z_obs = float(s.zc[self.k_obs])
        ix, iy = np.meshgrid(self._xc_idx, self._yc_idx, indexing="ij")
        self._obs_ix = jnp.asarray(ix.ravel())
        self._obs_iy = jnp.asarray(iy.ravel())

    def _load_initial_field(self, c):
        if "initial_field_dir" in c:
            d = Path(c["initial_field_dir"])
        else:
            from hydrogym.data_manager import HFDataManager

            dm = HFDataManager(
                repo_id=c.get("hf_repo_id", "dynamicslab/HydroGym-environments"),
                cache_dir=c.get("cache_dir"),
                local_fallback_dir=c.get("local_fallback_dir"),
                use_clean_cache=c.get("use_clean_cache", True),
                fallback_profile="JAX",
                token=c.get("hf_token"),
                revision=c.get("hf_revision"),
            )
            d = Path(dm.get_environment_path(ENVIRONMENT_NAME)) / "initial_field"
        self.U0 = jnp.asarray(np.load(str(d / "U.npy")), dtype=self.dtype)
        self.V0 = jnp.asarray(np.load(str(d / "V.npy")), dtype=self.dtype)
        self.W0 = jnp.asarray(np.load(str(d / "W.npy")), dtype=self.dtype)
        if self.U0.shape != self.solver.shape:
            raise ValueError(f"initial field has shape {self.U0.shape}, the environment expects {self.solver.shape}")

    # ------------------------------------------------------------------
    # jets, observation, wall shear
    # ------------------------------------------------------------------
    def jet_wall_velocity(self, action):
        """Wall-normal velocity ``(nx, ny)`` imposed on the bottom wall by the jets for ``action`` (``(n_jets,)``).

        Positive action is suction. The area-weighted mean action is removed so that the net mass flux is exactly zero.
        """
        a = jnp.asarray(action, dtype=self.dtype).reshape(self.n_jets)
        a = a - jnp.sum(a * self._jet_integral) / jnp.sum(self._jet_integral)
        return -self.jet_gain * jnp.einsum("j,jxy->xy", a, self._profiles)

    def wall_shear(self, U):
        """Plane-mean shear stress of the bottom wall from the first cell, ``nu <u> / z_1``."""
        return self.solver.wall_shear(U, "bottom")

    def get_obs(self, state: ChannelFDEnvState, params: ChannelFDEnvParams, key=None) -> chex.Array:
        """Streamwise velocity at the jet centres on the plane closest to ``y+ = obs_yplus``, shape ``(n_jets,)``."""
        return state.U[self._obs_ix, self._obs_iy, self.k_obs]

    # ------------------------------------------------------------------
    # gymnax interface
    # ------------------------------------------------------------------
    @property
    def default_params(self) -> ChannelFDEnvParams:
        return ChannelFDEnvParams(
            action_dim=self.n_jets, nsteps=self.nsteps, max_steps_in_episode=self.max_steps_in_episode
        )

    @property
    def name(self) -> str:
        return "ChannelFlowFDEnv"

    def _initial_fields(self, key):
        if self.initial_condition == "turbulent":
            return self.U0, self.V0, self.W0
        s = self.solver
        H = self.cfg.lz
        z = jnp.asarray(s.zc, dtype=self.dtype)
        profile = 1.5 * self.u_bulk_init * (1.0 - (1.0 - 2.0 * z / H) ** 2)
        k1, k2, k3 = jax.random.split(key, 3)
        amp = self.perturbation * jnp.max(profile)
        shape = s.shape

        def noise(k):
            return ((jax.random.uniform(k, shape, dtype=self.dtype) - 0.5) * amp).astype(self.dtype)

        return jnp.broadcast_to(profile, shape) + noise(k1), noise(k2), noise(k3)

    def reset_env(self, key: chex.PRNGKey, params: ChannelFDEnvParams) -> Tuple[chex.Array, ChannelFDEnvState]:
        """Reset to the stored turbulent field, or to a laminar profile plus a random perturbation."""
        u, v, w = self._initial_fields(key)
        st = self.solver.init_state(u, v, w, project=self.initial_condition == "laminar")
        state = ChannelFDEnvState(
            time=0,
            U=st.u,
            V=st.v,
            W=st.w,
            P=st.p,
            dt=st.dt,
            t=jnp.asarray(0.0, dtype=self.dtype),
            terminal=jnp.bool_(False),
        )
        return self.get_obs(state, params, key), state

    def is_terminal(self, state: ChannelFDEnvState, params: ChannelFDEnvParams) -> jnp.ndarray:
        return jnp.logical_or(state.terminal, state.time >= params.max_steps_in_episode)

    def _rollout(self, state: ChannelFDEnvState, action: jnp.ndarray):
        """Advance ``nsteps`` solver steps under the constant jet action; returns the final state and diagnostics."""
        s = self.solver
        w_wall = self.jet_wall_velocity(action)
        st0 = ChannelFDState(state.U, state.V, state.W, state.P, state.dt, state.t)
        area = self._wall_dA

        def body(st, _):
            st, aux = s.step(st, w_bot=w_wall)
            p_wall = st.p[..., 0] / st.dt  # pressure of the projection of the step, per unit time step
            p_wall = p_wall - jnp.mean(p_wall)
            pts = w_wall * (p_wall + 0.5 * w_wall**2) * area / self._power_ref
            per_jet = jnp.einsum("xy,jxy->j", pts, self._jet_labels)
            return st, (per_jet, self.wall_shear(st.u), aux["nits"].sum(), aux["courant"])

        body = jax.checkpoint(body)
        st, (per_jet, tau, nits, co) = jax.lax.scan(body, st0, None, length=self.nsteps)
        return st, per_jet, tau, nits, co

    def step_env(
        self, key: chex.PRNGKey, state: ChannelFDEnvState, action: jnp.ndarray, params: ChannelFDEnvParams
    ) -> Tuple[chex.Array, ChannelFDEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Apply the jet action for ``nsteps`` solver steps and return ``(obs, state, reward, done, info)``."""
        action = jnp.clip(jnp.asarray(action, dtype=self.dtype), params.min_action, params.max_action)
        st, per_jet_steps, tau_steps, nits, co = self._rollout(state, action)
        tau_w = self.wall_shear(st.u)
        if self.reward_type == "ratio":
            reward = self.tau_ref / tau_w
        else:
            reward = -tau_w
        forms_steps = jax.vmap(lambda p: power_forms(p, xp=jnp))(per_jet_steps)
        if self.power_sampling == "mean":
            pick = lambda a: jnp.mean(a, axis=0)  # noqa: E731
        else:
            pick = lambda a: a[-1]  # noqa: E731
        per_jet = pick(per_jet_steps)
        forms = {k: pick(v) for k, v in forms_steps.items()}
        reward = self.power_config.apply(reward, forms)
        cf = pick(tau_steps) / (0.5 * self.u_bulk_ref**2)

        time = state.time + self.nsteps
        next_state = ChannelFDEnvState(
            time=time,
            U=st.u,
            V=st.v,
            W=st.w,
            P=st.p,
            dt=st.dt,
            t=st.t,
            terminal=time >= params.max_steps_in_episode,
        )
        obs = self.get_obs(next_state, params, key)
        done = self.is_terminal(next_state, params)
        u_tau = jnp.sqrt(tau_w)
        info = {
            "discount": self.discount(next_state, params),
            "wall_shear_stress": tau_w,
            "wall_shear_stress_mean": jnp.mean(tau_steps),
            "obs_yplus": u_tau * self.z_obs / self.cfg.nu,
            "poisson_iterations": jnp.where(jnp.any(nits < 0), -1.0, jnp.mean(nits)),  # -1: solver does not report
            "courant": jnp.max(co),
            **power_info_jax(
                per_jet,
                sampling=self.power_sampling,
                forms=forms,
                drag_coefficient=cf,
                objective={"wss": tau_w},
            ),
        }
        return obs, next_state, reward, done, info

    def action_space(self, params: Optional[ChannelFDEnvParams] = None) -> spaces.Box:
        params = params or self.default_params
        return spaces.Box(low=params.min_action, high=params.max_action, shape=(params.action_dim,))

    def observation_space(self, params: Optional[ChannelFDEnvParams] = None) -> spaces.Box:
        params = params or self.default_params
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(self.n_jets,))


class ChannelFlowEnv:
    """Channel flow environment factory: ``env_config["solver"]`` is ``"fd"`` (default) or ``"spectral"``.

    ``"fd"`` returns a :class:`ChannelFlowFDEnv` (the finite-difference channel at ``Re_tau = 180``); ``"spectral"`` the
    earlier pseudo-spectral :class:`hydrogym.jax.envs.channel.ChannelFlowSpectralEnv` (its flow runs at
    ``Re_tau`` of about 310). The remaining ``env_config`` entries are passed on unchanged.
    """

    def __new__(cls, env_config: Optional[Dict] = None):
        env_config = dict(env_config or {})
        solver = str(env_config.get("solver", "fd")).lower()
        if solver == "fd":
            return ChannelFlowFDEnv(env_config)
        if solver == "spectral":
            from hydrogym.jax.envs.channel import ChannelFlowSpectralEnv

            return ChannelFlowSpectralEnv(env_config)
        raise ValueError(f"env_config['solver'] must be 'fd' or 'spectral', got {solver!r}")
