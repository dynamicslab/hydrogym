import os
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax

from hydrogym.data_manager import HFDataManager
from hydrogym.jax.env_core import EnvParams as BaseEnvParams
from hydrogym.jax.env_core import JAXFlowEnvBase
from hydrogym.jax.equation import SplitEquation
from hydrogym.jax.solvers.base import RungeKutta4, VelocityState
from hydrogym.jax.utils.utils import cheb_D_matrices, dealias_mask_2_3

jax.config.update("jax_enable_x64", True)

#######################################################################################
#                                                                                     #
#                             FLOW CONFIGURATION                                      #
#                                                                                     #
#######################################################################################


@struct.dataclass
class ChannelEnvParams(BaseEnvParams):
    """Extends base EnvParams with channel-specific settings."""

    action_dim: int = 24
    obs_subsample: int = 8
    obs_include_components: int = 2
    k_det: int = 9

    Nx: int = 72
    Ny: int = 72
    Nz: int = 72

    # DNS stepping
    nsteps: int = 50  # DNS substeps per RL step
    dt: float = 2e-4

    # Episode horizon in RL steps
    max_steps_in_episode: int = 5000

    # Action bounds
    min_action: float = -1
    max_action: float = 1


@struct.dataclass
class ChannelEnvState(environment.EnvState):
    time: int
    U: jnp.ndarray
    V: jnp.ndarray
    W: jnp.ndarray
    dt: jnp.ndarray
    terminal: jnp.bool_


class SpectralState(NamedTuple):
    u_hat: jnp.ndarray  # (Nx,Ny,Nz), complex
    v_hat: jnp.ndarray
    w_hat: jnp.ndarray


def make_obs_grid_indices(Nx: int, Ny: int, n: int):
    xs = jnp.linspace(0, Nx - 1, n).astype(jnp.int64)
    ys = jnp.linspace(0, Ny - 1, n).astype(jnp.int64)
    Xi, Yi = jnp.meshgrid(xs, ys, indexing="ij")
    return Xi, Yi


def get_obs_spectral_channel(
    state: ChannelEnvState,
    params: ChannelEnvParams,
    Xi: jnp.ndarray,
    Yi: jnp.ndarray,
) -> chex.Array:
    k = params.k_det
    Usl = state.U[:, :, k]
    Wsl = state.W[:, :, k]
    return jnp.stack([Usl[Xi, Yi], Wsl[Xi, Yi]], axis=0).reshape(-1)


def wss_compute(k, U, nu=1.9e-3, z=None):
    dz = z[k] - z[0]
    du_dz_wall = (U[:, :, k] - U[:, :, 0]) / dz
    tau_w = nu * du_dz_wall
    return jnp.mean(tau_w)


#######################################################################################
#                                                                                     #
#                             PSEUDOSPECTRAL EQUATION                                 #
#                                                                                     #
#######################################################################################


class PseudoSpectralNavierStokes3D(SplitEquation):
    def __init__(self, Nx, Ny, Nz, Lx, Ly, Lz, nu):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nu = nu
        self.dtype = jnp.float32

        kx = jnp.fft.fftfreq(self.Nx, d=self.Lx / self.Nx) * 2.0 * jnp.pi
        ky = jnp.fft.fftfreq(self.Ny, d=self.Ly / self.Ny) * 2.0 * jnp.pi

        self.kx = kx
        self.ky = ky
        self.ikx = (1j * kx)[:, None, None]
        self.iky = (1j * ky)[None, :, None]
        self.k2 = kx[:, None] ** 2 + ky[None, :] ** 2  # .astype(dtype)

        self.z, self.Dz, self.Dzz = cheb_D_matrices(self.Nz, self.Lz)
        self.dealias = dealias_mask_2_3(self.Nx, self.Ny)[:, :, None]  # .astype(dtype)

    def fft_xy(self, f):
        return jnp.fft.fftn(f, axes=(0, 1))

    def ifft_xy(self, f_hat):
        return jnp.fft.ifftn(f_hat, axes=(0, 1)).real

    def to_spectral(self, state: "VelocityState") -> "VelocityState":
        return VelocityState(
            self.fft_xy(state.u),
            self.fft_xy(state.v),
            self.fft_xy(state.w),
        )

    def to_physical(self, state_hat: "VelocityState") -> "VelocityState":
        return VelocityState(
            self.ifft_xy(state_hat.u),
            self.ifft_xy(state_hat.v),
            self.ifft_xy(state_hat.w),
        )

    def dx_hat(self, f_hat):
        return self.ikx * f_hat

    def dy_hat(self, f_hat):
        return self.iky * f_hat

    def dz_phys(self, f_phys):
        return jnp.einsum("ij,xyj->xyi", self.Dz, f_phys)

    def dzz_phys(self, f_phys):
        return jnp.einsum("ij,xyj->xyi", self.Dzz, f_phys)

    def apply_jets_v(
        self,
        v,
        Vjets,
        z0=1,
        jet_thickness=5,
        slit_length_x=3.0,
        slit_width_y=2.0,
        x_span_frac=2 / 3,
        nx_jets=6,
        ny_jets=4,
    ):
        Nx, Ny, Nz = v.shape
        v = v.at[:, :, 0].set(0.0)
        v = v.at[:, :, -1].set(0.0)

        x_max = int(Nx * x_span_frac)
        x_centers = jnp.linspace(3.0, float(x_max - 4), nx_jets)
        y_centers = jnp.linspace(4.0, float(Ny - 5), ny_jets)

        X = jnp.arange(Nx)[:, None]
        Y = jnp.arange(Ny)[None, :]

        xc = x_centers[:, None, None, None]
        yc = y_centers[None, :, None, None]
        Xg = X[None, None, :, :]
        Yg = Y[None, None, :, :]

        dx2 = (Xg - xc) ** 2
        dy2 = (Yg - yc) ** 2
        jets = jnp.exp(-0.5 * (dx2 / (slit_length_x**2) + dy2 / (slit_width_y**2)))

        V = Vjets.reshape(nx_jets, ny_jets, 1, 1)
        mask = jnp.sum(V * jets, axis=(0, 1))
        mask = mask - jnp.mean(mask)

        z1 = min(z0 + jet_thickness, Nz - 1)
        for z in range(z0, z1):
            v = v.at[:, :, z].set(mask)

        return v

    def enforce_noslip(self, u, v, w, action=None, action_time=50.0, t=0.0):
        u = u.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        v = v.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

        t = jnp.asarray(t)
        gain = jnp.asarray(0.3)
        gain = jnp.where(t < 10, gain * (t * 0.1), gain)
        gain = jnp.where(t >= (action_time - 10), gain * ((action_time - t) * 0.1), gain)

        if action is not None:
            w = self.apply_jets_v(w, Vjets=-gain * action)
        else:
            w = w.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

        return u, v, w

    def nonlinear_terms(self, state_hat, action=None, t=0.0, fx=0.0, fy=0.0, fz=0.0):
        state = self.to_physical(state_hat)
        u, v, w = self.enforce_noslip(state.u, state.v, state.w, action=action, t=t)

        u_hat_bc = self.fft_xy(u)
        v_hat_bc = self.fft_xy(v)
        w_hat_bc = self.fft_xy(w)

        du_dx = self.ifft_xy(self.dx_hat(u_hat_bc))
        du_dy = self.ifft_xy(self.dy_hat(u_hat_bc))
        dv_dx = self.ifft_xy(self.dx_hat(v_hat_bc))
        dv_dy = self.ifft_xy(self.dy_hat(v_hat_bc))
        dw_dx = self.ifft_xy(self.dx_hat(w_hat_bc))
        dw_dy = self.ifft_xy(self.dy_hat(w_hat_bc))

        du_dz = self.dz_phys(u)
        dv_dz = self.dz_phys(v)
        dw_dz = self.dz_phys(w)

        Nu = u * du_dx + v * du_dy + w * du_dz
        Nv = u * dv_dx + v * dv_dy + w * dv_dz
        Nw = u * dw_dx + v * dw_dy + w * dw_dz

        def forcing_hat(f):
            if jnp.ndim(f) == 0:
                f_phys = jnp.ones((self.Nx, self.Ny, self.Nz), dtype=self.dtype) * f
            else:
                f_phys = f
            return self.fft_xy(f_phys)

        fx_hat = forcing_hat(fx)
        fy_hat = forcing_hat(fy)
        fz_hat = forcing_hat(fz)

        return VelocityState(
            -self.fft_xy(Nu) * self.dealias + fx_hat,
            -self.fft_xy(Nv) * self.dealias + fy_hat,
            -self.fft_xy(Nw) * self.dealias + fz_hat,
        )

    def linear_terms(self, state_hat, action=None, t=0.0):
        state = self.to_physical(state_hat)
        u, v, w = self.enforce_noslip(state.u, state.v, state.w, action=action, t=t)

        uzz_hat = self.fft_xy(self.dzz_phys(u))
        vzz_hat = self.fft_xy(self.dzz_phys(v))
        wzz_hat = self.fft_xy(self.dzz_phys(w))

        k2 = self.k2[:, :, None]
        return VelocityState(
            self.nu * (-k2 * state_hat.u + uzz_hat),
            self.nu * (-k2 * state_hat.v + vzz_hat),
            self.nu * (-k2 * state_hat.w + wzz_hat),
        )

    def rhs(self, state_hat, action=None, t=0.0, fx=0.0, fy=0.0, fz=0.0):
        N = self.nonlinear_terms(state_hat, action=action, t=t, fx=fx, fy=fy, fz=fz)
        L = self.linear_terms(state_hat, action=action, t=t)
        return VelocityState(
            N.u + L.u,
            N.v + L.v,
            N.w + L.w,
        )

    def project(self, state_hat, dt, action=None, t=0.0):
        u_hat = state_hat.u
        v_hat = state_hat.v
        w_hat = state_hat.w

        du_dx = self.ifft_xy(self.dx_hat(u_hat))
        dv_dy = self.ifft_xy(self.dy_hat(v_hat))
        w = self.ifft_xy(w_hat)
        dw_dz = self.dz_phys(w)

        div_phys = du_dx + dv_dy + dw_dz
        rhs_hat = self.fft_xy(div_phys) / dt

        w_bot = w[:, :, 0] / dt
        w_top = w[:, :, -1] / dt
        dpdz_bot_hat = self.fft_xy(w_bot)
        dpdz_top_hat = self.fft_xy(w_top)

        Nz = self.Nz
        Nz_identity = jnp.eye(Nz, dtype=self.dtype)
        Dzz = self.Dzz.astype(self.dtype)
        Dz = self.Dz.astype(self.dtype)

        Nm = self.Nx * self.Ny
        k2_flat = self.k2.reshape(-1)
        rhs_flat = rhs_hat.reshape(Nm, Nz)

        A = Dzz[None, :, :] - k2_flat[:, None, None] * Nz_identity[None, :, :]
        A = A.at[:, 0, :].set(Dz[0, :][None, :])
        A = A.at[:, -1, :].set(Dz[-1, :][None, :])

        rhs = rhs_flat
        rhs = rhs.at[:, 0].set(dpdz_bot_hat.reshape(-1))
        rhs = rhs.at[:, -1].set(dpdz_top_hat.reshape(-1))

        mid = Nz // 2
        A0 = A[0].at[mid, :].set(0.0).at[mid, mid].set(1.0)
        b0 = rhs[0].at[mid].set(0.0)
        A = A.at[0].set(A0)
        rhs = rhs.at[0].set(b0)

        p_flat = jax.vmap(jnp.linalg.solve, in_axes=(0, 0))(A, rhs)
        p_hat = p_flat.reshape(self.Nx, self.Ny, Nz)

        u_hat_new = u_hat - dt * (self.ikx * p_hat)
        v_hat_new = v_hat - dt * (self.iky * p_hat)

        p_phys = self.ifft_xy(p_hat)
        dp_dz = self.dz_phys(p_phys)
        dp_dz_hat = self.fft_xy(dp_dz)
        w_hat_new = w_hat - dt * dp_dz_hat

        state_phys = VelocityState(
            self.ifft_xy(u_hat_new),
            self.ifft_xy(v_hat_new),
            self.ifft_xy(w_hat_new),
        )
        u_new, v_new, w_new = self.enforce_noslip(state_phys.u, state_phys.v, state_phys.w, action=action, t=t)

        return VelocityState(
            self.fft_xy(u_new),
            self.fft_xy(v_new),
            self.fft_xy(w_new),
        )


def run_channel_pseudospectral(
    U0,
    V0,
    W0,
    action,
    dt: float,
    equation: PseudoSpectralNavierStokes3D,
    integrator: RungeKutta4,
    nsteps: int = 50,
    return_trajectory: bool = False,
    checkpoint_steps: bool = True,
):
    state0 = equation.to_spectral(VelocityState(U0, V0, W0))

    def step_fn(state, n):
        t = n
        fx = 2.0
        new_state = integrator.rk4_step(
            state,
            dt=dt,
            action=action,
            t=t,
            fx=fx,
            target_bulk_u=8.0,
        )
        if return_trajectory:
            state_phys = equation.to_physical(new_state)
            return new_state, (state_phys.u, state_phys.v, state_phys.w)
        else:
            return new_state, None

    if checkpoint_steps:
        step_fn = jax.checkpoint(step_fn)

    stateT, traj = lax.scan(step_fn, state0, xs=jnp.arange(nsteps))

    if return_trajectory:
        Utraj, Vtraj, Wtraj = traj
        return Utraj, Vtraj, Wtraj
    else:
        state_phys = equation.to_physical(stateT)
        return state_phys.u, state_phys.v, state_phys.w


#######################################################################################
#                                                                                     #
#                             GYMNAX ENVIRONMENT                                      #
#                                                                                     #
#######################################################################################


class ChannelFlowSpectralEnv(JAXFlowEnvBase):
    """
    3D turbulent channel flow environment using a pseudo-spectral DNS solver.
    """

    def __init__(self, env_config: Dict):
        super().__init__(env_config)
        self.Lx = 2.0 * jnp.pi
        self.Ly = 1.0 * jnp.pi
        self.Lz = 2.0
        self.nu = 1.9e-3
        self.Nx = 72
        self.Ny = 72
        self.Nz = 72

        self.equation = PseudoSpectralNavierStokes3D(
            Nx=self.Nx,
            Ny=self.Ny,
            Nz=self.Nz,
            Lx=self.Lx,
            Ly=self.Ly,
            Lz=self.Lz,
            nu=self.nu,
        )
        self.integrator = RungeKutta4(
            equation=self.equation,
            dt=float(self.default_params.dt),
            save_n=1,
        )

        # Load initial fields from HuggingFace (downloaded/cached via HFDataManager).
        # env_config may override with:
        #   - "initial_field_dir": path to a directory containing U/V/W_nocontrol.npy
        #   - "local_fallback_dir": passed to HFDataManager for offline use
        if "initial_field_dir" in env_config:
            initial_field_dir = Path(env_config["initial_field_dir"])
        else:
            dm = HFDataManager(
                local_fallback_dir=env_config.get("local_fallback_dir"),
                fallback_profile="JAX",
            )
            env_path = dm.get_environment_path("Channel_3D_Retau180")
            initial_field_dir = Path(env_path) / "initial_field"

        self.U0 = jnp.load(str(initial_field_dir / "U.npy"))
        self.V0 = jnp.load(str(initial_field_dir / "V.npy"))
        self.W0 = jnp.load(str(initial_field_dir / "W.npy"))

        p = self.default_params
        self.Xi_obs, self.Yi_obs = make_obs_grid_indices(p.Nx, p.Ny, p.obs_subsample)

    @property
    def default_params(self) -> ChannelEnvParams:
        return ChannelEnvParams()

    @property
    def name(self) -> str:
        return "ChannelFlowSpectralEnv"

    def reset_env(self, key: chex.PRNGKey, params: ChannelEnvParams) -> Tuple[chex.Array, ChannelEnvState]:
        state = ChannelEnvState(
            time=0,
            U=self.U0,
            V=self.V0,
            W=self.W0,
            dt=jnp.asarray(params.dt),
            terminal=jnp.bool_(False),
        )
        obs = self.get_obs(state, params, key)
        return obs, state

    def get_obs(self, state: ChannelEnvState, params: ChannelEnvParams, key=None) -> chex.Array:
        return get_obs_spectral_channel(state, params, self.Xi_obs, self.Yi_obs)

    def is_terminal(self, state: ChannelEnvState, params: ChannelEnvParams) -> jnp.ndarray:
        return jnp.logical_or(state.terminal, state.time >= params.max_steps_in_episode)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: ChannelEnvState,
        action: jnp.ndarray,
        params: ChannelEnvParams,
    ) -> Tuple[chex.Array, ChannelEnvState, jnp.ndarray, jnp.ndarray, Dict]:
        action = jnp.clip(action, params.min_action, params.max_action)

        (wss, (U1, V1, W1)), grad_wss = jax.value_and_grad(
            lambda a: self._wss_with_aux(state, a, params),
            has_aux=True,
        )(action)

        reward = -wss

        time = state.time + params.nsteps
        terminal = time >= params.max_steps_in_episode

        next_state = ChannelEnvState(
            time=time,
            U=U1,
            V=V1,
            W=W1,
            dt=state.dt,
            terminal=terminal,
        )

        obs = self.get_obs(next_state, params, key)
        done = self.is_terminal(next_state, params)

        return obs, next_state, reward, done, {"discount": self.discount(next_state, params)}

    def action_space(self, params: Optional[ChannelEnvParams] = None) -> spaces.Box:
        params = params or self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(params.action_dim,),
        )

    def observation_space(self, params: Optional[ChannelEnvParams] = None) -> spaces.Box:
        params = params or self.default_params
        obs_dim = params.obs_subsample**2 * params.obs_include_components
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(obs_dim,))

    def _wss_with_aux(
        self,
        state: ChannelEnvState,
        action: jnp.ndarray,
        params: ChannelEnvParams,
    ):
        state0 = self.equation.to_spectral(VelocityState(state.U, state.V, state.W))

        def step_fn(state_hat, n):
            t = state.time + n
            fx = 2.0

            new_state = self.integrator.rk4_step(
                state_hat,
                dt=state.dt,
                action=action,
                t=t,
                fx=fx,
                target_bulk_u=8.0,
            )
            return new_state, None

        step_fn = jax.checkpoint(step_fn)

        nsteps = int(self.default_params.nsteps)
        stateT, _ = jax.lax.scan(
            step_fn,
            state0,
            xs=jnp.arange(nsteps),
        )

        state_phys = self.equation.to_physical(stateT)
        U1, V1, W1 = state_phys.u, state_phys.v, state_phys.w

        wss = wss_compute(4, U1, z=self.equation.z)
        return wss, (U1, V1, W1)
