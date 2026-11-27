from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from jaxfluids.callbacks.base_callback import Callback
from jaxfluids.data_types import JaxFluidsBuffers
from jaxfluids.data_types.ml_buffers import (
    CallablesSetup,
    InterfaceFluxCallablesSetup,
    InterfaceFluxParametersSetup,
    LevelSetSetup,
    ParametersSetup,
)
from jaxfluids.domain.helper_functions import (
    reassemble_buffer_np,
    reassemble_cell_centers,
    reassemble_cell_sizes,
)
from jaxfluids_rl.jxf_env import RenderMode
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import OmegaConf

from hydrogym.jaxfluids.env_core import JAXFluidsFlowEnv
from hydrogym.jaxfluids.utils.nozzle import (
    InjectorGeometry,
    ObsData,
    PressureRatios,
    TargetThrustAngleFn,
    TVCSpec,
    build_tvc_env_options,
    build_tvc_runtime_setup,
    compute_thrust,
    initialize_injector_flux_fn,
    initialize_injector_power_fn,
    plot_flowfield_3d,
    primary_reference_power,
)
from hydrogym.power import PowerConfig, power_forms, power_info


class _InjectorPowerCallback(Callback):
    """Samples the injection power after every JAX-Fluids integration step (``power_sampling = "mean"``)."""

    def __init__(self) -> None:
        self.env = None

    def after_step_end(self, jxf_buffers, callback_dict, **kwargs):
        env = self.env
        if env is not None and env.power_sampling == "mean":
            env._power_samples.append(env._evaluate_power(jxf_buffers))
        return jxf_buffers, callback_dict


class NozzleBase(JAXFluidsFlowEnv):
    """Thrust-vector control of a supersonic nozzle by secondary injection (shock vector control).

    Actuation power (one actuator per injector): the injected gas comes from a reservoir at total
    temperature ``T_inf`` and the injector total pressure ``p_t`` set by the action; its power is the exergy
    flow relative to the ambient state, ``P_i = sum_cells mdot R T_inf ln(p_t / p_inf)`` (minimum, isothermal,
    supply work; in the low-Mach limit the incompressible transpiration power ``Q (p_t - p_inf)``), with the
    injected mass flux ``mdot`` of the interface-flux function (forms per interface cell). The thrust-angle
    objective has no force coefficient, so ``C_P = P / P_ref`` with ``P_ref`` the exergy flow rate of the ideal
    choked primary jet, ``mdot* R T0 ln(NPR)`` (2D: per unit span, ``A* = 2 R_throat``; 3D: ``pi R_throat^2``),
    see ``primary_reference_power``. No ``drag_coefficient``.
    Sampling: ``"mean"`` (default, after every JAX-Fluids step, via a callback) or ``"end"``.
    Power options (``power_weight``, ``power_cost``, ``power_sampling``) are read from the ``jaxfluids``
    section of the environment config and can be overridden in ``env_config``.
    """

    SPEC: ClassVar[TVCSpec]
    TARGET_FNS: ClassVar[dict[str, TargetThrustAngleFn]] = {}

    def __init__(self, env_config: dict) -> None:
        self._init_from_hf(env_config)
        # Settings of the ``jaxfluids`` section given in env_config take precedence over the configuration file,
        # e.g. ``num_actuators`` of the 3D nozzle, which the published configuration does not set.
        overrides = env_config.get("jaxfluids")
        if overrides:
            self.conf.jaxfluids = OmegaConf.merge(self.conf.jaxfluids, OmegaConf.create(dict(overrides)))

        self.power_config = PowerConfig.from_config(dict(self.conf.jaxfluids), env_config)
        self.power_sampling = self.power_config.resolve_sampling("mean")
        self._power_samples = []
        self._last_power_forms = None
        power_callback = _InjectorPowerCallback()

        env_options = build_tvc_env_options(
            env_config=self.conf.jaxfluids,
            spec=self.SPEC,
            target_fns=self.TARGET_FNS,
            cls_name=self.__class__.__name__,
        )

        self.num_actuators = env_options.num_actuators
        self.secondary_pressure_ratio = env_options.secondary_pressure_ratio
        self.resolution = env_options.resolution
        self.target_fn = env_options.target_fn
        self.is_pressure_probes = env_options.is_pressure_probes
        self.is_scale_observations = env_options.is_scale_observations

        runtime_setup = build_tvc_runtime_setup(
            base_path=Path(self.env_data_path),
            dim=self.SPEC.dim,
            resolution=self.resolution,
            ngpus=env_options.ngpus,
        )

        self.env_name = runtime_setup.env_name
        self.env_dir = runtime_setup.env_dir
        self.restart_file_path = runtime_setup.restart_file_path
        self.injector_geometry = InjectorGeometry(
            X=self.SPEC.injector_x,
            IW=self.SPEC.injector_width,
            N=self.num_actuators,
        )
        self.pressure_ratios = PressureRatios(
            NPR=self.SPEC.nozzle_pressure_ratio,
            SPR=self.secondary_pressure_ratio,
        )

        super().__init__(
            self.conf.jaxfluids,
            runtime_setup.case_setup_dict,
            runtime_setup.numerical_setup_dict,
            callbacks=[power_callback],
        )

        self.default_action_reset = np.zeros(self.num_actuators)

        if self.is_pressure_probes:
            self.probe_locations = self._compute_probe_locations()
            self.num_probes = self.probe_locations.shape[0]
        else:
            self.probe_locations = None
            self.num_probes = 0

        self.action_callable_setup = self._build_action_callable_setup()
        self._injector_power_fn = initialize_injector_power_fn(
            injector_geometry=self.injector_geometry,
            pressure_ratios=self.pressure_ratios,
            p_infty=self.SPEC.ambient_pressure,
            T_infty=self.SPEC.ambient_temperature,
            specific_heat_ratio=self.SPEC.specific_heat_ratio,
            specific_gas_constant=self.SPEC.specific_gas_constant,
            sim_manager=self.sim_manager,
        )
        self.power_reference = primary_reference_power(self.SPEC)
        self._power_jit = None
        power_callback.env = self

        self._set_spaces(
            action_space=self._build_action_space(),
            observation_space=self._build_observation_space(),
        )

    def _is_terminated(self, action: np.ndarray, jxf_buffers: JaxFluidsBuffers, info: dict) -> bool:
        physical_simulation_time = jxf_buffers.time_control_variables.physical_simulation_time
        return bool(physical_simulation_time >= self.SPEC.t_end)

    def _is_truncated(self, jxf_buffers: JaxFluidsBuffers, info: dict) -> bool:
        return False

    @abstractmethod
    def _get_reward(self, action: np.ndarray) -> float:
        pass

    def _build_action_callable_setup(self) -> CallablesSetup:
        interface_flux_fn = initialize_injector_flux_fn(
            injector_geometry=self.injector_geometry,
            pressure_ratios=self.pressure_ratios,
            p_infty=self.SPEC.ambient_pressure,
            T_infty=self.SPEC.ambient_temperature,
            specific_heat_ratio=self.SPEC.specific_heat_ratio,
            specific_gas_constant=self.SPEC.specific_gas_constant,
            sim_manager=self.sim_manager,
        )
        levelset_setup = LevelSetSetup(fluid_solid=InterfaceFluxCallablesSetup(interface_flux_fn))
        return CallablesSetup(levelset=levelset_setup)

    def _build_action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_actuators,),
        )

    def _get_observation_shape(self) -> tuple[int, ...]:
        num_angles = self.SPEC.dim - 1
        num_obs = 2 * num_angles + self.num_probes
        return (num_obs,)

    def _build_observation_space(self) -> gym.spaces.Box:
        num_angles = self.SPEC.dim - 1

        if self.is_scale_observations:
            low = np.array([-1.0] * (2 * num_angles) + [0.0] * self.num_probes, dtype=np.float32)
            # the scaled pressures p / p0 are not bounded by 1 (the probes can see more than the reservoir pressure)
            high = np.array([1.0] * (2 * num_angles) + [np.inf] * self.num_probes, dtype=np.float32)
        else:
            low = np.array([-np.pi] * (2 * num_angles) + [0.0] * self.num_probes, dtype=np.float32)
            high = np.array(
                [np.pi] * (2 * num_angles) + [np.inf] * self.num_probes,
                dtype=np.float32,
            )

        return gym.spaces.Box(
            low=low,
            high=high,
            shape=self._get_observation_shape(),
        )

    def _convert_action_for_jxf(self, action: np.ndarray) -> ParametersSetup:
        levelset_setup = LevelSetSetup(fluid_solid=InterfaceFluxParametersSetup(jnp.array(action)))
        return ParametersSetup(levelset=levelset_setup)

    def _get_obs(self) -> np.ndarray:
        jxf_buffers, _ = self._require_state()
        obs_data = self.compute_obs(jxf_buffers)

        self.thrust_angle = obs_data.thrust_angle
        self.target_angle = obs_data.target_angle
        self.pressure_probes = obs_data.pressure_probes

        thrust_angle = jnp.atleast_1d(obs_data.thrust_angle)
        target_angle = jnp.atleast_1d(obs_data.target_angle)
        if self.is_scale_observations:
            thrust_angle /= jnp.pi
            target_angle /= jnp.pi

        obs = [thrust_angle, target_angle]

        if self.is_pressure_probes:
            pressure_probes = jnp.atleast_1d(obs_data.pressure_probes)
            if self.is_scale_observations:
                pressure_probes /= self.SPEC.p0
            obs.append(pressure_probes)

        obs = jnp.concatenate(obs)

        if obs.shape != self.observation_space.shape:
            raise ValueError(f"Observation shape mismatch: got {obs.shape}, expected {self.observation_space.shape}")

        # the solver runs in float64; the observation space is float32 (the dtype RL libraries expect)
        return np.asarray(obs, dtype=self.observation_space.dtype)

    # ------------------------------------------------------------------
    # Actuation power
    # ------------------------------------------------------------------
    def _power_terms(self, jxf_buffers: JaxFluidsBuffers, actuator: Array):
        """Per-injector power, cost forms (per interface cell) and injected mass flow; ``C_P`` units."""
        domain_information = self.sim_manager.domain_information
        nhx, nhy, nhz = domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = domain_information.domain_slices_geometry
        simulation_buffers = jxf_buffers.simulation_buffers
        primitives = simulation_buffers.material_fields.primitives[..., nhx, nhy, nhz]
        apertures = simulation_buffers.levelset_fields.apertures
        geometry = self.sim_manager.levelset_handler.fluid_solid_handler.geometry_calculator
        normal = geometry.compute_normal_apertures_based(apertures)[..., nhx_, nhy_, nhz_]
        interface_length = geometry.compute_interface_length(apertures)[..., nhx_, nhy_, nhz_]
        mesh_grid = jnp.array(domain_information.compute_device_mesh_grid())
        n_cells = primitives[0].size
        mdot, power = self._injector_power_fn(
            primitives.reshape(primitives.shape[0], n_cells),
            interface_length.reshape(n_cells),
            normal.reshape(normal.shape[0], n_cells),
            mesh_grid.reshape(mesh_grid.shape[0], n_cells),
            actuator,
        )
        power = power / self.power_reference
        forms = power_forms(power, xp=jnp)
        out = (jnp.sum(power, axis=1), forms, jnp.sum(mdot))
        if domain_information.is_parallel:
            out = jax.lax.psum(out, axis_name="i")
        return out

    def _evaluate_power(self, jxf_buffers: JaxFluidsBuffers):
        """``(per_injector, forms, mdot)`` as numpy values for the current buffers and action."""
        if self._power_jit is None:
            if self.sim_manager.domain_information.is_parallel:
                self._power_jit = jax.pmap(
                    self._power_terms,
                    axis_name="i",
                    in_axes=(JaxFluidsBuffers(0, None, None, None), None),
                    out_axes=None,
                )
            else:
                self._power_jit = jax.jit(self._power_terms)
        actuator = jnp.asarray(self.action_params_setup.levelset.fluid_solid.actuator)
        per, forms, mdot = self._power_jit(jxf_buffers, actuator)
        return np.asarray(per), {k: float(v) for k, v in forms.items()}, float(mdot)

    def _power_info(self) -> dict[str, Any]:
        """Power ``info`` entries ("mean" over the samples of the last action, else instantaneous)."""
        samples = self._power_samples
        if self.power_sampling != "mean" or not samples:
            jxf_buffers, _ = self._require_state()
            samples = [self._evaluate_power(jxf_buffers)]
        self._power_samples = []
        per = np.mean([sm[0] for sm in samples], axis=0)
        forms = {k: float(np.mean([sm[1][k] for sm in samples])) for k in ("signed", "positive", "abs")}
        mdot = float(np.mean([sm[2] for sm in samples]))
        self._last_power_forms = forms
        objective = {
            "thrust_angle": np.array(self.thrust_angle),
            "target_angle": np.array(self.target_angle),
            "injector_mass_flow": mdot,
        }
        return power_info(per, sampling=self.power_sampling, objective=objective, forms=forms)

    def step(self, action: np.ndarray):
        self._power_samples = []
        observation, reward, terminated, truncated, info = super().step(action)
        reward = self.power_config.apply(reward, self._last_power_forms)
        return observation, reward, terminated, truncated, info

    def _get_info(self) -> dict[str, Any]:
        return {
            "thrust_angle": np.array(self.thrust_angle),
            "target_angle": np.array(self.target_angle),
            **self._power_info(),
        }

    def _after_step(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        jxf_buffers: JaxFluidsBuffers,
    ) -> None:
        t = float(jxf_buffers.time_control_variables.physical_simulation_time)
        self._append_history(
            time=t,
            thrust_angle=self.thrust_angle,
            pressure_probes=self.pressure_probes,
            action=action,
        )

    def compute_obs(self, jxf_buffers: JaxFluidsBuffers) -> ObsData:
        if self.sim_manager.domain_information.is_parallel:
            return jax.pmap(
                self._compute_obs,
                axis_name="i",
                in_axes=(JaxFluidsBuffers(0, None, None, None),),
                out_axes=None,
            )(jxf_buffers)

        else:
            return jax.jit(self._compute_obs)(jxf_buffers)

    def _compute_obs(self, jxf_buffers: JaxFluidsBuffers) -> ObsData:
        current_angle = self.compute_thrust_angle(jxf_buffers)

        sim_time = jxf_buffers.time_control_variables.physical_simulation_time
        target_angle = jnp.asarray(self.target_fn(sim_time))
        if self.SPEC.dim == 2 and target_angle.ndim != 0:
            raise ValueError(f"2D target_angle must be scalar, got {target_angle.shape}")
        if self.SPEC.dim == 3 and target_angle.shape != (2,):
            raise ValueError(f"3D target_angle must have shape (2,), got {target_angle.shape}")

        if self.is_pressure_probes:
            pressure_probes = self.compute_pressure_probes(jxf_buffers)
        else:
            pressure_probes = None

        return ObsData(current_angle, target_angle, pressure_probes)

    def compute_thrust_angle(self, jxf_buffers: JaxFluidsBuffers) -> Array:
        domain_information = self.sim_manager.domain_information
        nhx, nhy, nhz = domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = domain_information.domain_slices_geometry
        cell_centers = domain_information.get_device_cell_centers()
        cell_sizes = domain_information.get_device_cell_sizes()
        is_parallel = domain_information.is_parallel

        simulation_buffers = jxf_buffers.simulation_buffers

        primitives = simulation_buffers.material_fields.primitives[..., nhx, nhy, nhz]
        apertures_x = simulation_buffers.levelset_fields.apertures[0][..., nhx_, nhy_, nhz_]
        thrust, _, _ = compute_thrust(
            primitives,
            self.SPEC.ambient_pressure,
            apertures_x,
            cell_centers,
            cell_sizes,
        )

        if is_parallel:
            thrust = jax.lax.psum(thrust, axis_name="i")

        if self.SPEC.dim == 2:
            current_angle = jnp.atan2(thrust[1], thrust[0])
        else:
            current_angle = jnp.stack(
                [
                    jnp.atan2(thrust[1], thrust[0]),  # Pitch
                    jnp.atan2(thrust[2], thrust[0]),  # Yaw
                ]
            )

        return current_angle

    def compute_pressure_probes(self, jxf_buffers: JaxFluidsBuffers) -> Array:
        x_p = self.probe_locations[:, 0]
        y_p = self.probe_locations[:, 1]
        z_p = self.probe_locations[:, 2]

        domain_information = self.sim_manager.domain_information

        cell_centers = domain_information.get_device_cell_centers()
        x, y, z = [xi.flatten() for xi in cell_centers]

        x_id = jnp.searchsorted(x, x_p, side="left", method="scan_unrolled")
        y_id = jnp.searchsorted(y, y_p, side="left", method="scan_unrolled")

        if self.SPEC.dim == 3:
            z_id = jnp.searchsorted(z, z_p, side="left", method="scan_unrolled")
        else:
            z_id = 0

        nhx, nhy, nhz = domain_information.domain_slices_conservatives
        pressure = jxf_buffers.simulation_buffers.material_fields.primitives[4, nhx, nhy, nhz]
        pressure_probes = pressure[x_id, y_id, z_id]

        if domain_information.is_parallel:
            device_domain_size = domain_information.get_device_domain_size()

            mask = 1
            for i in range(domain_information.dim):
                xi = self.probe_locations[:, i]
                mask *= (device_domain_size[i][0] <= xi) & (xi < device_domain_size[i][1])

            pressure_probes = jax.lax.psum(mask * pressure_probes, axis_name="i")

        return pressure_probes

    def _get_fields_for_plotting(self, jxf_buffers: JaxFluidsBuffers) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        domain_information = self.sim_manager.domain_information
        nhx, nhy, nhz = domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = domain_information.domain_slices_geometry

        levelset_fields = jxf_buffers.simulation_buffers.levelset_fields
        primitives = jxf_buffers.simulation_buffers.material_fields.primitives

        fields = [
            primitives[..., nhx, nhy, nhz],
            levelset_fields.levelset[..., nhx, nhy, nhz],
            levelset_fields.volume_fraction[..., nhx_, nhy_, nhz_],
        ]

        if domain_information.is_parallel:
            fields = [reassemble_buffer_np(field, domain_information.split_factors) for field in fields]

        fields = [field.squeeze() for field in fields]
        return tuple(fields)

    def _get_meshgrid_for_plotting(self) -> tuple[np.ndarray, np.ndarray]:
        domain_information = self.sim_manager.domain_information
        cell_centers = domain_information.get_global_cell_centers()
        if domain_information.is_parallel:
            cell_centers = reassemble_cell_centers(cell_centers, domain_information.split_factors)

        x, y, _ = [xi.flatten() for xi in cell_centers]
        return np.meshgrid(x, y, indexing="ij")

    @abstractmethod
    def _compute_probe_locations(self) -> np.ndarray:
        pass

    def render(self) -> None:
        if self.render_mode is None:
            return

        self._plot_flow_field()
        self._plot_observations()

    @abstractmethod
    def _plot_flow_field(self) -> None:
        pass

    @abstractmethod
    def _plot_observations(self) -> None:
        pass


class Nozzle2D(NozzleBase):
    """Nozzle2D environment for shock vector control.

    For the 2D nozzle, two actuators are present: one located on the upper
    side of the nozzle, and one located at the lower side of the nozzle.

    Arguments:
        - secondary_pressure_ratio: Optional. Float, must be between 0.7 and 0.9.
            Defaults to 0.7.
        - resolution: Optional. Spatial resolution of the environment. Choose either 'coarse' or 'fine'.
        - ngpus: Optional. Number of GPUs for running the environment.
        - is_pressure_probes: Optional. Boolean indicating whether pressure probes
            are part of the observation
        - is_scale_observations: Optional. Boolean indicating whether observations are scaled to [0, 1].
        - target_fn: Optional. Target thrust vector function. Choose either 'sine' or 'step'.
    """

    SPEC = TVCSpec(
        dim=2,
        fixed_num_actuators=2,
    )

    TARGET_FNS = {
        "sine": lambda t: (t > 5e-4) * (10.0 / 180.0 * jnp.pi) * jnp.sin(2 * jnp.pi * (t - 5e-4) / 4e-3),
        "step": lambda t: (t > 5e-4) * (5.0 / 180.0 * jnp.pi),
    }

    def _compute_probe_locations(self) -> np.ndarray:
        G = self.SPEC.nozzle_geometry.G
        H = self.SPEC.nozzle_geometry.H

        probe_locations = []
        for i in (1, 2):
            x = G[0] + i / 3 * (H[0] - G[0])
            y = G[1] + i / 3 * (H[1] - G[1])

            x_probes = np.array([x, x])
            y_probes = np.array([y, -y])
            z_probes = np.zeros_like(x_probes)
            probe_locations.append(np.stack([x_probes, y_probes, z_probes], axis=1))

        return np.concatenate(probe_locations, axis=0)

    def _get_reward(self, action: np.ndarray) -> float:
        error = np.abs(self.target_angle - self.thrust_angle)
        return float(-error)

    def _plot_flow_field(self) -> None:
        jxf_buffers, _ = self._require_state()
        primitives, _, volume_fraction = self._get_fields_for_plotting(jxf_buffers)
        X, Y = self._get_meshgrid_for_plotting()

        D_throat = self.SPEC.nozzle_geometry.D_throat
        X = X / D_throat
        Y = Y / D_throat

        physical_simulation_time = jxf_buffers.time_control_variables.physical_simulation_time

        rho = primitives[0]
        p = primitives[-1]
        v = primitives[1:3]
        c = np.sqrt(self.SPEC.specific_heat_ratio * p / rho)
        M = np.linalg.norm(v, axis=0, ord=2) / c

        mask = volume_fraction == 0.0
        p = np.ma.masked_where(mask, p)
        M = np.ma.masked_where(mask, M)

        fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 4))
        fig.suptitle(f"Env Step: {self.env_step}, Time: {physical_simulation_time * 1e3:.3f} ms")

        quants = (M, p / self.SPEC.p0)
        vmins = (0.0, 0.0)
        vmaxs = (3.0, 1.0)
        for axi, quant, vmin, vmax in zip(ax, quants, vmins, vmaxs):
            pci = axi.pcolormesh(X, Y, quant, cmap="Spectral_r", vmin=vmin, vmax=vmax, shading="auto")
            axi.contour(X, Y, M, levels=[1.0], linewidths=0.5, colors="k", linestyles="-")

            divider = make_axes_locatable(axi)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pci, cax=cax, orientation="vertical")

        titles = ("Mach number", "Pressure p / p_0")
        for axi, title in zip(ax, titles):
            axi.set_aspect("equal")
            axi.set_xlabel(r"x / D")
            axi.set_ylabel(r"y / D")
            axi.set_title(title)
            axi.set_xlim(0, 8)
            axi.set_ylim(-2, 2)

        if self.is_pressure_probes:
            for axi in ax:
                axi.scatter(
                    self.probe_locations[:, 0] / D_throat,
                    self.probe_locations[:, 1] / D_throat,
                    s=2,
                    c="black",
                )

        if self.render_mode is RenderMode.SHOW:
            plt.show()
        elif self.render_mode is RenderMode.SAVE:
            self._save_render_figure(fig, "flowfield")
        else:
            raise ValueError(f"RenderMode {self.render_mode} is not valid.")
        plt.close(fig)

    def _plot_observations(self) -> None:
        jxf_buffers, _ = self._require_state()
        physical_simulation_time = jxf_buffers.time_control_variables.physical_simulation_time

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax = ax.flatten()

        fig.suptitle(f"Env Step: {self.env_step}, Time: {physical_simulation_time * 1e3:.3f} ms")

        times_full = np.linspace(0.0, self.SPEC.t_end, 100)
        target_full = self.target_fn(times_full)
        ax[0].plot(times_full * 1e3, np.rad2deg(target_full), "b--", label="target")

        t = np.array(self.history["time"])
        thrust_angle = np.array(self.history["thrust_angle"])
        action = np.array(self.history["action"])

        if len(t) > 0:
            ax[0].plot(t * 1e3, np.rad2deg(thrust_angle), "k", label="current")
            for actuator_i in range(self.num_actuators):
                ax[1].plot(t * 1e3, action[:, actuator_i], label=f"Inj. {actuator_i:02d}")

        ax[1].set_ylim(-0.05, 1.05)

        if self.is_pressure_probes and len(t) > 0:
            pressure_probes = np.array(self.history["pressure_probes"])
            pressure_probes /= self.SPEC.p0
            for probe_i in range(self.num_probes // 2):
                ax[2].plot(t * 1e3, pressure_probes[:, probe_i], label=f"Probe 0{probe_i:d}")
                ax[3].plot(
                    t * 1e3,
                    pressure_probes[:, probe_i + self.num_probes // 2],
                    label=f"Probe 1{probe_i:d}",
                )
            ax[2].set_ylim(0.0, 0.5)
            ax[3].set_ylim(0.0, 0.5)

        titles = (
            "Thrust angle [deg]",
            "Actuators",
            "Pressure probes downstream loc 0",
            "Pressure probes downstream loc 1",
        )
        for axi, title in zip(ax, titles):
            axi.set_box_aspect(1.0)
            axi.set_xlabel("t [ms]")
            axi.set_title(title)
            axi.legend()

        if self.render_mode is RenderMode.SHOW:
            plt.show()
        elif self.render_mode is RenderMode.SAVE:
            self._save_render_figure(fig, "observations")
        else:
            raise ValueError(f"RenderMode {self.render_mode} is not valid.")
        plt.close(fig)


class Nozzle3D(NozzleBase):
    """Nozzle3D environment for shock vector control.

    For the 3D nozzle, the number of actuators can be set by the user. The minimum number of
    actuators is 4, and the maximum number is 12. The actuators are distributed uniformly over
    the diameter.

    Arguments:
        - num_actuators: Required. Integer number of actuators. Must be between 4 and 12.
        - secondary_pressure_ratio: Optional. Float, must be between 0.7 and 0.9.
            Defaults to 0.7.
        - resolution: Optional. Spatial resolution of the environment. Choose either 'coarse' or 'fine'.
        - ngpus: Optional. Number of GPUs for running the environment.
        - is_pressure_probes: Optional. Boolean indicating whether pressure probes
            are part of the observation
        - is_scale_observations: Optional. Boolean indicating whether observations are scaled to [0, 1].
        - target_fn: Optional. Target thrust vector function. Choose either 'sine' or 'step'.
    """

    SPEC: ClassVar[TVCSpec] = TVCSpec(
        dim=3,
        min_num_actuators=4,
        max_num_actuators=12,
    )

    TARGET_FNS = {
        "sine": lambda t: jnp.array(
            [
                (t > 1e-3) * (10.0 / 180.0 * jnp.pi) * jnp.sin(2 * jnp.pi * (t - 1e-3) / 4e-3),
                jnp.zeros_like(t),
            ]
        ),
        "step": lambda t: jnp.array(
            [
                (t > 1e-3) * (5.0 / 180.0 * jnp.pi),
                jnp.zeros_like(t),
            ]
        ),
    }

    def _compute_probe_locations(self) -> np.ndarray:
        G = self.SPEC.nozzle_geometry.G
        H = self.SPEC.nozzle_geometry.H

        num_probes_per_diameter = 6
        theta = np.linspace(0, 2 * np.pi, num_probes_per_diameter, endpoint=False)

        probe_locations = []
        for i in (1, 2):
            x = G[0] + i / 3 * (H[0] - G[0])
            R = G[1] + i / 3 * (H[1] - G[1])
            x_probes = np.full_like(theta, x)
            y_probes = R * np.cos(theta)
            z_probes = -R * np.sin(theta)
            probe_locations.append(np.stack([x_probes, y_probes, z_probes], axis=1))
        return np.concatenate(probe_locations, axis=0)

    def _get_reward(self, action: np.ndarray) -> float:
        error = jnp.sqrt(jnp.sum((self.target_angle - self.thrust_angle) ** 2))
        return float(-error)

    def _plot_flow_field(self) -> None:
        jxf_buffers, _ = self._require_state()
        primitives, levelset, _ = self._get_fields_for_plotting(jxf_buffers)

        domain_information = self.sim_manager.domain_information
        cell_centers = domain_information.get_global_cell_centers()
        cell_sizes = domain_information.get_global_cell_sizes()
        if domain_information.is_parallel:
            cell_centers = reassemble_cell_centers(cell_centers, domain_information.split_factors)
            cell_sizes = reassemble_cell_sizes(cell_sizes, domain_information.split_factors)

        cell_centers = tuple(x.squeeze() for x in cell_centers)
        cell_sizes = tuple(x.squeeze() for x in cell_sizes)

        self.render_dir.mkdir(parents=True, exist_ok=True)

        plotter = plot_flowfield_3d(
            primitives,
            levelset,
            cell_centers,
            cell_sizes,
            self.SPEC.p0,
            self.injector_geometry,
            self.SPEC.specific_heat_ratio,
        )

        filename = self.render_dir / f"flowfield_{self.env_step:04d}.png"
        if self.render_mode is RenderMode.SHOW:
            plotter.show(auto_close=False)
        elif self.render_mode is RenderMode.SAVE:
            plotter.show(screenshot=str(filename), auto_close=False)
        else:
            raise ValueError(f"RenderMode {self.render_mode} is not valid.")
        plotter.clear()

    def _plot_observations(self) -> None:
        jxf_buffers, _ = self._require_state()
        physical_simulation_time = jxf_buffers.time_control_variables.physical_simulation_time

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax = ax.flatten()

        fig.suptitle(f"Env Step: {self.env_step}, Time: {physical_simulation_time * 1e3:.3f} ms")

        times_full = np.linspace(0.0, self.SPEC.t_end, 100)
        target_full = self.target_fn(times_full)

        ax[0].plot(times_full * 1e3, np.rad2deg(target_full[0]), "b--", label="target")
        ax[1].plot(times_full * 1e3, np.rad2deg(target_full[1]), "b--", label="target")

        t = np.array(self.history["time"])
        thrust_angle = np.array(self.history["thrust_angle"])

        if len(t) > 0:
            ax[0].plot(t * 1e3, np.rad2deg(thrust_angle[:, 0]), "k", label="current")
            ax[1].plot(t * 1e3, np.rad2deg(thrust_angle[:, 1]), "k", label="current")

            ax[0].set_ylim(-10.0, 10.0)
            ax[1].set_ylim(-10.0, 10.0)

        if self.is_pressure_probes and len(t) > 0:
            pressure_probes = np.array(self.history["pressure_probes"])
            pressure_probes /= self.SPEC.p0
            for probe_i in range(self.num_probes // 2):
                ax[2].plot(t * 1e3, pressure_probes[:, probe_i], label=f"Probe 0{probe_i:d}")
                ax[3].plot(
                    t * 1e3,
                    pressure_probes[:, probe_i + self.num_probes // 2],
                    label=f"Probe 1{probe_i:d}",
                )
            ax[2].set_ylim(0.0, 0.5)
            ax[3].set_ylim(0.0, 0.5)

        titles = (
            "Thrust angle" + r"$\delta_0$" + "[deg]",
            "Thrust angle" + r"$\delta_1$" + "[deg]",
            "Pressure probes downstream loc 0",
            "Pressure probes downstream loc 1",
        )
        for axi, title in zip(ax, titles):
            axi.set_box_aspect(1.0)
            axi.set_xlabel("t [ms]")
            axi.set_title(title)
            axi.legend()

        if self.render_mode is RenderMode.SHOW:
            plt.show()
        elif self.render_mode is RenderMode.SAVE:
            self._save_render_figure(fig, "observations")
        else:
            raise ValueError(f"RenderMode {self.render_mode} is not valid.")
        plt.close(fig)
