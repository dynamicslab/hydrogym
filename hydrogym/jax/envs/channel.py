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
from hydrogym.jax.utils.power import power_info_jax
from hydrogym.jax.utils.utils import cheb_D_matrices, dealias_mask_2_3
from hydrogym.power import PowerConfig, power_forms

jax.config.update("jax_enable_x64", True)

# The wall-normal derivatives and the pressure solve are matrix products with ill-conditioned Chebyshev matrices. On
# GPUs, float32 matrix products use reduced-precision tensor-core arithmetic by default (TF32), and in batched form
# (jax.vmap over environments) the pressure solve then loses about 3 decimal digits and the flow blows up within a
# few actions. These products therefore run at full float32/float64 precision.
_EXACT = lax.Precision.HIGHEST

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
    """Channel-flow environment state (physical-space velocity fields).

    Attributes:
        time: Simulation time (in DNS steps) elapsed in the episode.
        U: Streamwise (x) velocity field, shape (Nx, Ny, Nz).
        V: Spanwise (y) velocity field, shape (Nx, Ny, Nz).
        W: Wall-normal (z) velocity field, shape (Nx, Ny, Nz).
        dt: Current DNS timestep.
        terminal: Whether the episode has terminated.
    """

    time: int
    U: jnp.ndarray
    V: jnp.ndarray
    W: jnp.ndarray
    dt: jnp.ndarray
    terminal: jnp.bool_


class SpectralState(NamedTuple):
    """Three-component spectral (FFT'd) velocity state.

    Attributes:
        u_hat: x-component spectrum, shape (Nx, Ny, Nz), complex.
        v_hat: y-component spectrum.
        w_hat: z-component spectrum.
    """

    u_hat: jnp.ndarray  # (Nx,Ny,Nz), complex
    v_hat: jnp.ndarray
    w_hat: jnp.ndarray


def make_obs_grid_indices(Nx: int, Ny: int, n: int):
    """Build the (x, y) index grid used to subsample the observation plane.

    Args:
        Nx: Number of grid points in x.
        Ny: Number of grid points in y.
        n: Number of subsample points along each axis.

    Returns:
        Tuple ``(Xi, Yi)`` of index arrays of shape (n, n) spanning the full
        x/y ranges, suitable for advanced indexing of a slice of the state.
    """
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
    """Extract the channel-flow observation vector from the current state.

    Samples the streamwise (U) and wall-normal (W) velocity at the subsampled
    grid points on the wall-parallel plane ``z = params.k_det``.

    Args:
        state: Current channel environment state.
        params: Channel environment parameters.
        Xi: Observation x-index grid (from ``make_obs_grid_indices``).
        Yi: Observation y-index grid.

    Returns:
        1D array of the sampled U and W values stacked and flattened.
    """
    k = params.k_det
    Usl = state.U[:, :, k]
    Wsl = state.W[:, :, k]
    return jnp.stack([Usl[Xi, Yi], Wsl[Xi, Yi]], axis=0).reshape(-1)


def clenshaw_curtis_weights(n_points: int, length: float) -> np.ndarray:
    """Clenshaw-Curtis weights of the Chebyshev-Gauss-Lobatto points of ``cheb_D_matrices`` on ``length``."""
    n = n_points - 1
    theta = np.pi * np.arange(n_points) / n
    w = np.zeros(n_points)
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0 / (n**2 - 1)
        for k in range(1, n // 2):
            v -= 2.0 * np.cos(2 * k * theta[1:-1]) / (4 * k**2 - 1)
        v -= np.cos(n * theta[1:-1]) / (n**2 - 1)
    else:
        w[0] = w[n] = 1.0 / n**2
        for k in range(1, (n - 1) // 2 + 1):
            v -= 2.0 * np.cos(2 * k * theta[1:-1]) / (4 * k**2 - 1)
    w[1:-1] = 2.0 * v / n
    return w * length / 2.0


def wss_compute(k, U, nu=1.9e-3, z=None):
    """Compute the domain-mean wall shear stress from a finite-difference gradient.

    Approximates the wall-normal velocity gradient at the wall by a one-sided
    difference between wall-parallel slices ``k`` and ``0``, using the physical
    z-coordinates in ``z``.

    Args:
        k: Index of the near-wall slice used for the finite difference.
        U: Streamwise velocity field, shape (Nx, Ny, Nz).
        nu: Kinematic viscosity used to convert the gradient to stress.
        z: Physical z-coordinates of the grid points (1D array).

    Returns:
        Mean wall shear stress over the (Nx, Ny) plane.
    """
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
    """Pseudo-spectral 3D Navier-Stokes equation for a channel flow.

    Horizontal (x, y) directions are treated spectrally with FFTs and the
    wall-normal (z) direction with Chebyshev collocation. The state is the
    triple of spectral velocity components on the (Nx, Ny, Nz) grid. The
    incompressibility constraint is enforced by a pressure Poisson solve whose
    per-horizontal-wavenumber Chebyshev matrices are pre-inverted at
    construction, with the no-penetration wall condition folded in.

    The pressure solver imposes the mean-pressure gauge by pinning one grid
    point (the zero-wavenumber row), and the wall-normal pressure-gradient
    boundary conditions (from the wall values of w) are applied in
    ``project``.

    Jets (``jet_model``):

    - ``"wall"`` (default): the 24 jets are a wall-normal transpiration
      velocity ``w(x, y, z_wall) = w_jet(x, y)`` on the bottom wall (zero net
      mass flux), imposed consistently in the pressure projection. At zero
      action ``w_jet = 0`` and the equation is the uncontrolled channel.
    - ``"legacy"``: reproduces the jets of HydroGym <= 1.0, which overwrite
      ``w`` in the grid layers ``z[1:6]`` next to the bottom wall after every
      stage (also at zero action) with the gain of ``jet_gain``.
    """

    JET_GAIN = 0.3  # peak jet velocity per unit action

    def __init__(
        self,
        Nx,
        Ny,
        Nz,
        Lx,
        Ly,
        Lz,
        nu,
        dtype=jnp.float32,
        jet_model="wall",
        action_time=50.0,
        ramp_time=10.0,
    ):
        """Precompute wavenumbers, derivative operators, dealiasing mask, and pressure solver.

        Args:
            Nx: Number of grid points in x.
            Ny: Number of grid points in y.
            Nz: Number of Chebyshev collocation points in z.
            Lx: Domain length in x.
            Ly: Domain length in y.
            Lz: Domain height in z.
            nu: Kinematic viscosity.
            dtype: Floating dtype for the precomputed operators.
            jet_model: ``"wall"`` (transpiration on the bottom wall, default)
                or ``"legacy"`` (overwrite of the layers ``z[1:6]``, HydroGym
                <= 1.0); see the class docstring.
            action_time: Length of one action window in the time unit of the
                ``t`` passed to the boundary conditions (``"wall"`` model; the
                legacy gain always uses 50).
            ramp_time: Duration of the linear up/down ramp of the jet gain at
                the start/end of each action window (``"wall"`` model).
        """
        if jet_model not in ("wall", "legacy"):
            raise ValueError(f"jet_model must be 'wall' or 'legacy', got {jet_model!r}")
        self.jet_model = jet_model
        self.action_time = float(action_time)
        self.ramp_time = float(ramp_time)
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nu = nu
        self.dtype = dtype

        kx = jnp.fft.fftfreq(self.Nx, d=self.Lx / self.Nx) * 2.0 * jnp.pi
        ky = jnp.fft.fftfreq(self.Ny, d=self.Ly / self.Ny) * 2.0 * jnp.pi

        self.kx = kx.astype(self.dtype)
        self.ky = ky.astype(self.dtype)
        self.ikx = (1j * self.kx)[:, None, None]
        self.iky = (1j * self.ky)[None, :, None]
        self.k2 = self.kx[:, None] ** 2 + self.ky[None, :] ** 2

        self.z, self.Dz, self.Dzz = cheb_D_matrices(self.Nz, self.Lz)
        self.Dz = self.Dz.astype(self.dtype)
        self.Dzz = self.Dzz.astype(self.dtype)
        self.dealias = dealias_mask_2_3(self.Nx, self.Ny)[:, :, None].astype(self.dtype)

        # ===================================
        # Pre-compute Pressure Inverse Matrix
        # ===================================
        Nz_identity = jnp.eye(self.Nz, dtype=self.dtype)
        Dzz = self.Dzz
        Dz = self.Dz
        k2_flat = self.k2.reshape(-1)

        # Assemble constant matrix A
        A = Dzz[None, :, :] - k2_flat[:, None, None] * Nz_identity[None, :, :]
        A = A.at[:, 0, :].set(Dz[0, :][None, :])
        A = A.at[:, -1, :].set(Dz[-1, :][None, :])

        # Handle the zero-wavenumber singularity (mean pressure)
        mid = self.Nz // 2
        A0 = A[0].at[mid, :].set(0.0).at[mid, mid].set(1.0)
        A = A.at[0].set(A0)

        # Precompute the inverse
        self.A_inv = jax.vmap(jnp.linalg.inv)(A)

    def fft_xy(self, f):
        """Forward FFT of a field over the horizontal (x, y) axes.

        Args:
            f: Physical-space field of shape (Nx, Ny, Nz).

        Returns:
            Spectral field of the same shape.
        """
        return jnp.fft.fftn(f, axes=(0, 1))

    def ifft_xy(self, f_hat):
        """Inverse FFT of a field over the horizontal (x, y) axes, taking the real part.

        Args:
            f_hat: Spectral field of shape (Nx, Ny, Nz).

        Returns:
            Physical-space (real) field of the same shape.
        """
        return jnp.fft.ifftn(f_hat, axes=(0, 1)).real

    def to_spectral(self, state: "VelocityState") -> "VelocityState":
        """Transform a physical-space velocity state to spectral space.

        Args:
            state: Physical-space ``VelocityState``.

        Returns:
            Spectral-space ``VelocityState``.
        """
        return VelocityState(
            self.fft_xy(state.u),
            self.fft_xy(state.v),
            self.fft_xy(state.w),
        )

    def to_physical(self, state_hat: "VelocityState") -> "VelocityState":
        """Transform a spectral velocity state to physical space.

        Args:
            state_hat: Spectral-space ``VelocityState``.

        Returns:
            Physical-space ``VelocityState``.
        """
        return VelocityState(
            self.ifft_xy(state_hat.u),
            self.ifft_xy(state_hat.v),
            self.ifft_xy(state_hat.w),
        )

    def dx_hat(self, f_hat):
        """x-derivative of a spectral field (multiplication by i*kx).

        Args:
            f_hat: Spectral field.

        Returns:
            Spectral x-derivative.
        """
        return self.ikx * f_hat

    def dy_hat(self, f_hat):
        """y-derivative of a spectral field (multiplication by i*ky).

        Args:
            f_hat: Spectral field.

        Returns:
            Spectral y-derivative.
        """
        return self.iky * f_hat

    def dz_phys(self, f_phys):
        """First z-derivative of a physical field via the Chebyshev matrix.

        Args:
            f_phys: Physical-space field of shape (Nx, Ny, Nz).

        Returns:
            z-derivative of the same shape.
        """
        return jnp.einsum("ij,xyj->xyi", self.Dz, f_phys, precision=_EXACT)

    def dzz_phys(self, f_phys):
        """Second z-derivative of a physical field via the Chebyshev matrix.

        Args:
            f_phys: Physical-space field of shape (Nx, Ny, Nz).

        Returns:
            Second z-derivative of the same shape.
        """
        return jnp.einsum("ij,xyj->xyi", self.Dzz, f_phys, precision=_EXACT)

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
        """Imprint a grid of wall-normal jet velocity profiles onto the wall-normal field ``w``.

        A periodic array of ``nx_jets x ny_jets`` Gaussian jets centered on the
        bottom wall builds a velocity mask from the per-jet amplitudes in
        ``Vjets``; the mask is mean-subtracted (zero net blowing) and written
        into the wall-normal component over ``jet_thickness`` layers
        starting at wall layer ``z0``. The wall layers are first set
        to zero.

        Args:
            v: Wall-normal velocity field ``w`` (positional argument), shape (Nx, Ny, Nz).
            Vjets: Jet amplitudes, shape (nx_jets, ny_jets).
            z0: First wall-normal layer the jets penetrate.
            jet_thickness: Number of layers the jets span.
            slit_length_x: Gaussian width of each jet in x.
            slit_width_y: Gaussian width of each jet in y.
            x_span_frac: Fraction of the x domain covered by the jet array.
            nx_jets: Number of jets along x.
            ny_jets: Number of jets along y.

        Returns:
            The modified wall-normal velocity field.
        """
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

    def jet_mask(
        self,
        Vjets,
        slit_length_x=3.0,
        slit_width_y=2.0,
        x_span_frac=2 / 3,
        nx_jets=6,
        ny_jets=4,
    ):
        """Zero-mean wall-normal velocity profile of the jet array, shape (Nx, Ny).

        The same Gaussian jets as ``apply_jets_v``: ``nx_jets x ny_jets``
        Gaussians with amplitudes ``Vjets``, minus their plane mean (zero net
        mass flux).
        """
        x_max = int(self.Nx * x_span_frac)
        x_centers = jnp.linspace(3.0, float(x_max - 4), nx_jets)
        y_centers = jnp.linspace(4.0, float(self.Ny - 5), ny_jets)
        X = jnp.arange(self.Nx)[None, None, :, None]
        Y = jnp.arange(self.Ny)[None, None, None, :]
        dx2 = (X - x_centers[:, None, None, None]) ** 2
        dy2 = (Y - y_centers[None, :, None, None]) ** 2
        jets = jnp.exp(-0.5 * (dx2 / (slit_length_x**2) + dy2 / (slit_width_y**2)))
        mask = jnp.sum(Vjets.reshape(nx_jets, ny_jets, 1, 1) * jets, axis=(0, 1))
        return mask - jnp.mean(mask)

    def jet_gain(self, t, action_time=50.0):
        """Jet gain (peak jet velocity per unit action) at time ``t`` of the action window.

        ``"wall"`` model: ``JET_GAIN * clip(min(t, T - t) / ramp_time, 0, 1)``
        with ``T = self.action_time``: a linear ramp up over the first
        ``ramp_time`` of each action window, constant, and a linear ramp down
        to 0 over the last ``ramp_time``. ``t`` is the time since the start
        of the current action.

        ``"legacy"`` model (reproduces HydroGym <= 1.0): the same formula
        without the clip, with ``T = action_time`` (default 50) and
        ``ramp_time = 10`` in units of DNS steps, evaluated at the episode DNS
        step count ``state.time + n``.
        """
        t = jnp.asarray(t)
        if self.jet_model == "legacy":
            gain = jnp.asarray(0.3)
            gain = jnp.where(t < 10, gain * (t * 0.1), gain)
            gain = jnp.where(t >= (action_time - 10), gain * ((action_time - t) * 0.1), gain)
            return gain
        ramp = jnp.minimum(t, self.action_time - t) / self.ramp_time
        return self.JET_GAIN * jnp.clip(ramp, 0.0, 1.0)

    def jet_wall_velocity(self, action, t):
        """Transpiration velocity ``w_jet(x, y)`` the jets impose on the bottom wall at time ``t`` (``"wall"`` model).

        Positive action = suction (``w < 0`` at the bottom wall, as in the
        legacy model). Returns zeros for ``action=None``.
        """
        if action is None:
            return jnp.zeros((self.Nx, self.Ny), dtype=self.dtype)
        return self.jet_mask(-self.jet_gain(t) * action).astype(self.dtype)

    def enforce_noslip(self, u, v, w, action=None, action_time=50.0, t=0.0):
        """Apply the wall boundary conditions to a physical-space velocity state.

        u and v are forced to zero at both walls, and w at the top wall. On
        the bottom wall:

        - ``"wall"`` model: ``w = w_jet`` (``jet_wall_velocity``, zero for
          zero action or ``action=None``); the interior is not touched.
        - ``"legacy"`` model: with an ``action``, the jets
          (``apply_jets_v``) overwrite w in the layers ``z[1:6]`` (also at
          zero action) with the legacy gain; without an action, w is set to
          zero at the walls.

        Args:
            u: Streamwise velocity field, shape (Nx, Ny, Nz).
            v: Spanwise velocity field.
            w: Wall-normal velocity field.
            action: Jet control amplitudes, or None for the unactuated case.
            action_time: Legacy model only: actuation window of the gain ramp.
            t: Current time, used for the gain ramp (see ``jet_gain``).

        Returns:
            Tuple ``(u, v, w)`` with the boundary conditions applied.
        """
        u = u.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        v = v.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

        if self.jet_model == "legacy":
            gain = self.jet_gain(t, action_time)
            if action is not None:
                w = self.apply_jets_v(w, Vjets=-gain * action)
            else:
                w = w.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
            return u, v, w

        w = w.at[:, :, -1].set(0.0)
        if action is None:
            w = w.at[:, :, 0].set(0.0)
        else:
            w = w.at[:, :, 0].set(self.jet_wall_velocity(action, t))
        return u, v, w

    def nonlinear_terms(self, state_hat, action=None, t=0.0, fx=0.0, fy=0.0, fz=0.0):
        """Evaluate the explicitly treated (nonlinear advection + forcing) terms.

        The state is transformed to physical space, wall boundary conditions
        (and jets, if actuated) are applied, and the advective terms
        ``u · ∇u`` are computed with spectral x/y derivatives and Chebyshev
        z-derivatives. The result is returned in spectral space with the 2/3
        dealiasing mask applied and the body forcing added.

        Args:
            state_hat: Spectral velocity state.
            action: Jet control amplitudes, or None.
            t: Current time (used for the actuation gain ramp).
            fx: x-direction body forcing (scalar or physical-space field).
            fy: y-direction body forcing.
            fz: z-direction body forcing.

        Returns:
            Spectral ``VelocityState`` of the nonlinear terms.
        """
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
            """Broadcast a scalar forcing to the full grid (or pass a field through) and FFT it in x/y."""
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
        """Evaluate the implicitly treated (viscous diffusion) term.

        Computes ``nu * (∂²u/∂z² - k²u)`` for each component, combining the
        Chebyshev second z-derivative in physical space with the horizontal
        Laplacian applied spectrally.

        Args:
            state_hat: Spectral velocity state.
            action: Unused; present for interface compatibility.
            t: Unused; present for interface compatibility.

        Returns:
            Spectral ``VelocityState`` of the linear terms.
        """
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
        """Evaluate the full right-hand side as nonlinear + linear terms.

        Args:
            state_hat: Spectral velocity state.
            action: Jet control amplitudes, or None.
            t: Current time.
            fx: x-direction body forcing.
            fy: y-direction body forcing.
            fz: z-direction body forcing.

        Returns:
            Spectral ``VelocityState`` of the full right-hand side.
        """
        N = self.nonlinear_terms(state_hat, action=action, t=t, fx=fx, fy=fy, fz=fz)
        L = self.linear_terms(state_hat, action=action, t=t)
        return VelocityState(
            N.u + L.u,
            N.v + L.v,
            N.w + L.w,
        )

    def project(self, state_hat, dt, action=None, t=0.0, return_pre_bc=False):
        """Project the state onto the divergence-free (incompressibility) constraint.

        Solves the pressure Poisson problem for the divergence error over one
        timestep using the precomputed per-wavenumber inverse Chebyshev
        matrices, with the wall-normal pressure-gradient conditions derived
        from the wall values of w, and the mean-pressure gauge pinned at the
        zero-wavenumber mode. The velocity is corrected by the pressure
        gradient, the boundary conditions (and jets, if actuated) are
        re-applied, and the result is returned in spectral space.

        In the ``"wall"`` jet model the Neumann condition on the bottom wall is
        ``dp/dz = (w - w_jet) / dt``, so that the corrected velocity has the
        jet transpiration velocity at the wall (and the mass-conserving
        divergence-free projection includes it).

        Args:
            state_hat: Spectral velocity state to project.
            dt: Timestep used in the pressure correction.
            action: Jet control amplitudes, or None.
            t: Current time (used for the actuation gain ramp).
            return_pre_bc: If True, also return the power diagnostic: for the
                ``"legacy"`` model the physical wall-normal velocity after the
                pressure correction and before the boundary/jet overwrite; for
                the ``"wall"`` model the physical pressure on the bottom wall,
                shape (Nx, Ny).

        Returns:
            Projected spectral ``VelocityState`` (and, with
            ``return_pre_bc``, the power diagnostic).
        """
        u_hat = state_hat.u
        v_hat = state_hat.v
        w_hat = state_hat.w

        du_dx = self.ifft_xy(self.dx_hat(u_hat))
        dv_dy = self.ifft_xy(self.dy_hat(v_hat))
        w = self.ifft_xy(w_hat)
        dw_dz = self.dz_phys(w)

        div_phys = du_dx + dv_dy + dw_dz
        rhs_hat = self.fft_xy(div_phys) / dt

        if self.jet_model == "legacy":
            w_bot = w[:, :, 0] / dt
        else:
            w_bot = (w[:, :, 0] - self.jet_wall_velocity(action, t)) / dt
        w_top = w[:, :, -1] / dt
        dpdz_bot_hat = self.fft_xy(w_bot)
        dpdz_top_hat = self.fft_xy(w_top)

        Nm = self.Nx * self.Ny
        rhs_flat = rhs_hat.reshape(Nm, self.Nz)

        rhs = rhs_flat
        rhs = rhs.at[:, 0].set(dpdz_bot_hat.reshape(-1))
        rhs = rhs.at[:, -1].set(dpdz_top_hat.reshape(-1))

        mid = self.Nz // 2
        b0 = rhs[0].at[mid].set(0.0)
        rhs = rhs.at[0].set(b0)

        # Use the pre-computed inverse!
        p_flat = jnp.einsum("mij,mj->mi", self.A_inv, rhs, precision=_EXACT)
        p_hat = p_flat.reshape(self.Nx, self.Ny, self.Nz)

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

        projected = VelocityState(
            self.fft_xy(u_new),
            self.fft_xy(v_new),
            self.fft_xy(w_new),
        )
        if return_pre_bc:
            if self.jet_model == "legacy":
                return projected, state_phys.w
            return projected, p_phys[:, :, 0]
        return projected


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
    """Run a channel-flow DNS rollout of ``nsteps`` RK4 steps.

    The initial physical fields are transformed to spectral space, advanced
    with ``integrator.rk4_step`` for ``nsteps`` substeps (constant body forcing
    ``fx = 2.0``, constant-mass-flux correction toward a bulk velocity of 8.0),
    and returned either as a trajectory of physical-space snapshots or as the
    final state only.

    Args:
        U0: Initial streamwise velocity field, shape (Nx, Ny, Nz).
        V0: Initial spanwise velocity field.
        W0: Initial wall-normal velocity field.
        action: Jet control amplitudes applied at every substep, or None.
        dt: DNS timestep.
        equation: The pseudo-spectral equation (provides transforms).
        integrator: The RK4 integrator to step with.
        nsteps: Number of DNS substeps to run.
        return_trajectory: If True, return the full trajectory of physical
            velocity fields instead of just the final state.
        checkpoint_steps: If True, wrap the step function in ``jax.checkpoint``
            to trade compute for reduced memory during the scan.

    Returns:
        If ``return_trajectory``: arrays ``(U, V, W)`` of trajectories with a
        leading time axis of length ``nsteps``; otherwise the final state's
        physical ``(u, v, w)`` fields.
    """
    state0 = equation.to_spectral(VelocityState(U0, V0, W0))

    def step_fn(state, n):
        """One RK4 substep: constant body forcing ``fx = 2.0``, time = substep index."""
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

    x streamwise, y spanwise (periodic), z wall-normal (Chebyshev, walls at
    ``z = -Lz/2, Lz/2``); constant mass flux at bulk velocity ``U_b = 8``.
    The reward is ``-tau_w``, the mean shear stress on the bottom wall.

    Jets. The 24 jets (6 x 4 Gaussian patches, zero net mass flux) impose a
    wall-normal transpiration velocity ``w_jet(x, y)`` on the bottom wall
    (positive action = suction). Each action window of ``nsteps`` DNS steps
    starts and ends with a linear ramp of the jet gain over 10 DNS steps
    (peak gain 0.3 per unit action), the same for every action. At zero
    action the environment is the uncontrolled channel.

    Actuation power. The power the jets put into the fluid is the energy flux
    through the bottom wall, ``P = sum w_jet (p - p_ref + w_jet^2 / 2) dA``
    (``p`` = wall pressure of the projection step, ``p_ref`` = its plane mean,
    ``dA`` = cell area; the viscous normal stress vanishes at a no-slip wall),
    the transpiration form also used for the Firedrake jets. It is formed
    point-wise (``positive``/``abs`` forms) and attributed per jet to the
    nearest jet centre.

    Legacy (``env_config["legacy_jets"] = True``) reproduces the jets of
    HydroGym <= 1.0: the jet profile overwrites ``w`` in the layers ``z[1:6]``
    next to the bottom wall after every stage, also at zero action, with the
    gain of ``PseudoSpectralNavierStokes3D.jet_gain``. The power is then
    the rate of the kinetic-energy change of that impulsive constraint,
    ``P = sum (w_jet^2 - w_free^2) / 2 dV / dt`` (``w_free`` = velocity after
    the pressure correction, before the overwrite; ``dV`` = cell area x
    Clenshaw-Curtis weight).

    Normalisation: ``C_P = P / (rho U_b^3 Lx Ly / 2)`` and
    ``drag_coefficient = C_f = tau_w / (rho U_b^2 / 2)`` (density 1), so that
    ``C_f`` is the drag power per bottom-wall area in the same units. Sampling:
    ``"mean"`` (default, all DNS substeps of the action) or ``"end"``.
    Power options in ``env_config``: ``power_weight`` (default 0: reward
    unchanged), ``power_cost``, ``power_sampling``.
    """

    JET_Z0 = 1  # legacy jets: first layer overwritten (apply_jets_v default)
    JET_THICKNESS = 5  # legacy jets: number of overwritten layers (apply_jets_v default)
    JET_RAMP_STEPS = 10  # DNS steps of the gain ramp at the start and end of each action
    U_BULK = 8.0  # target bulk velocity of the constant-mass-flux correction

    def __init__(self, env_config: Dict):
        """Build the equation, integrator, and initial fields for the channel.

        Physical/spectral grid parameters are taken from ``env_config`` (with
        the defaults of the Re_tau 180 channel). Initial velocity
        fields are loaded either from a local directory or from Hugging Face
        (see the source comments for the accepted overrides).

        Args:
            env_config: Configuration dictionary; recognized keys include
                ``Lx``/``Ly``/``Lz`` (domain extents), ``nu`` (viscosity),
                ``Nx``/``Ny``/``Nz`` (grid resolution, note that non-default
                values change the JIT-compiled operator shapes and require
                matching initial-field shapes), ``dtype`` ("float32" or
                "float64"), ``initial_field_dir`` (local directory containing
                U/V/W.npy), and the HFDataManager forwarding options
                ``hf_repo_id``/``cache_dir``/``use_clean_cache``/
                ``local_fallback_dir``/``hf_token``/``hf_revision``,
                ``legacy_jets`` (default False: wall transpiration jets with a
                per-action gain ramp; True: the HydroGym <= 1.0 layer overwrite
                and gain, see the class docstring).
        """
        super().__init__(env_config)
        # Physical/spectral grid parameters. The defaults are the Re_tau 180
        # channel; env_config overrides exist for running other channel
        # configurations (NOTE: non-default Nx/Ny/Nz change the
        # JIT-compiled spectral operator shapes and require matching
        # initial-field array shapes at reset()).
        self.Lx = env_config.get("Lx", 2.0 * jnp.pi)
        self.Ly = env_config.get("Ly", 1.0 * jnp.pi)
        self.Lz = env_config.get("Lz", 2.0)
        self.nu = env_config.get("nu", 1.9e-3)
        self.Nx = int(env_config.get("Nx", 72))
        self.Ny = int(env_config.get("Ny", 72))
        self.Nz = int(env_config.get("Nz", 72))

        dtype_str = env_config.get("dtype", "float32")
        self.dtype = jnp.float64 if dtype_str == "float64" else jnp.float32
        self.legacy_jets = bool(env_config.get("legacy_jets", False))

        # Gain ramp of the wall jets in physical time since the start of the action:
        # the action window is nsteps DNS steps, the ramp 10 DNS steps (default dt).
        p0 = self.default_params
        self.equation = PseudoSpectralNavierStokes3D(
            Nx=self.Nx,
            Ny=self.Ny,
            Nz=self.Nz,
            Lx=self.Lx,
            Ly=self.Ly,
            Lz=self.Lz,
            nu=self.nu,
            dtype=self.dtype,
            jet_model="legacy" if self.legacy_jets else "wall",
            action_time=float(p0.nsteps) * float(p0.dt),
            ramp_time=self.JET_RAMP_STEPS * float(p0.dt),
        )
        self.integrator = RungeKutta4(
            equation=self.equation,
            dt=float(self.default_params.dt),
            save_n=1,
        )

        # Load initial fields from HuggingFace (downloaded/cached via HFDataManager).
        # env_config may override with:
        #   - "initial_field_dir": path to a directory containing U.npy, V.npy, W.npy
        #   - "hf_repo_id" / "cache_dir" / "use_clean_cache" / "local_fallback_dir":
        #     forwarded to HFDataManager (defaults match JAXFlowEnv's).
        if "initial_field_dir" in env_config:
            initial_field_dir = Path(env_config["initial_field_dir"])
        else:
            dm = HFDataManager(
                repo_id=env_config.get("hf_repo_id", "dynamicslab/HydroGym-environments"),
                cache_dir=env_config.get("cache_dir"),
                local_fallback_dir=env_config.get("local_fallback_dir"),
                use_clean_cache=env_config.get("use_clean_cache", True),
                fallback_profile="JAX",
                token=env_config.get("hf_token"),
                revision=env_config.get("hf_revision"),
            )
            env_path = dm.get_environment_path("Channel_3D_Retau180")
            initial_field_dir = Path(env_path) / "initial_field"

        self.U0 = jnp.load(str(initial_field_dir / "U.npy")).astype(self.dtype)
        self.V0 = jnp.load(str(initial_field_dir / "V.npy")).astype(self.dtype)
        self.W0 = jnp.load(str(initial_field_dir / "W.npy")).astype(self.dtype)

        p = self.default_params
        self.Xi_obs, self.Yi_obs = make_obs_grid_indices(p.Nx, p.Ny, p.obs_subsample)

        # Actuation power
        self.power_config = PowerConfig.from_config(env_config)
        self.power_sampling = self.power_config.resolve_sampling("mean")
        self._setup_power()

    def _setup_power(self):
        """Quadrature weights of the jet layers, jet labels and the power normalisation."""
        z0 = self.JET_Z0
        z1 = min(z0 + self.JET_THICKNESS, self.Nz - 1)
        self._jet_layers = (z0, z1)
        wz = clenshaw_curtis_weights(self.Nz, self.Lz)[z0:z1]
        self._jet_dV = jnp.asarray(wz * (self.Lx / self.Nx) * (self.Ly / self.Ny), dtype=self.dtype)
        # nearest jet centre (in the Gaussian metric of apply_jets_v), jets ordered like action.reshape(6, 4)
        nx_jets, ny_jets = 6, 4
        x_centers = np.linspace(3.0, float(int(self.Nx * 2 / 3) - 4), nx_jets)
        y_centers = np.linspace(4.0, float(self.Ny - 5), ny_jets)
        X, Y = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny), indexing="ij")
        d2 = (X[None, None] - x_centers[:, None, None, None]) ** 2 / 3.0**2 + (
            Y[None, None] - y_centers[None, :, None, None]
        ) ** 2 / 2.0**2
        label = np.argmin(d2.reshape(nx_jets * ny_jets, self.Nx, self.Ny), axis=0)
        self._jet_onehot = jnp.asarray(np.eye(nx_jets * ny_jets)[label], dtype=self.dtype)  # (Nx, Ny, 24)
        self._power_ref = 0.5 * self.U_BULK**3 * self.Lx * self.Ly
        self._wall_dA = (self.Lx / self.Nx) * (self.Ly / self.Ny)

    def _substep_time(self, state: ChannelEnvState, n):
        """Time passed to the integrator for DNS substep ``n`` of the current action.

        Wall jets: physical time since the start of the action, ``n * dt``.
        Legacy jets: the episode DNS step count ``state.time + n`` (HydroGym <= 1.0).
        """
        if self.legacy_jets:
            return state.time + n
        return n * state.dt

    def _jet_velocity(self, action: jnp.ndarray, t) -> jnp.ndarray:
        """Wall-normal velocity the jets impose in the jet layers at time ``t``, shape (Nx, Ny, layers)."""
        z0, z1 = self._jet_layers
        nz = min(self.Nz, z1 + 1)  # enough layers for apply_jets_v to address the same slab
        zeros = jnp.zeros((self.Nx, self.Ny, nz), dtype=self.dtype)
        _, _, w = self.equation.enforce_noslip(zeros, zeros, zeros, action=action, t=t)
        return w[:, :, z0:z1]

    def _power_from_rollout(self, state: ChannelEnvState, action: jnp.ndarray, diag: jnp.ndarray, wss_steps):
        """Power coefficients and drag from the per-substep power diagnostic of the rollout.

        ``diag``: wall jets: the bottom-wall pressure of every substep, shape (n, Nx, Ny);
        legacy jets: the pre-overwrite jet-layer velocities, shape (n, Nx, Ny, layers).
        """
        nsteps = diag.shape[0]
        t_proj = self._substep_time(state, jnp.arange(nsteps)) + state.dt  # time passed to project() in rk4_step
        if self.legacy_jets:
            w_jet = jax.vmap(lambda t: self._jet_velocity(action, t))(t_proj)
            p_pts = 0.5 * (w_jet**2 - diag**2) * self._jet_dV / state.dt / self._power_ref  # (n, Nx, Ny, layers)
            per_jet = jnp.einsum("nxyl,xyj->nj", p_pts, self._jet_onehot)
        else:
            w_jet = jax.vmap(lambda t: self.equation.jet_wall_velocity(action, t))(t_proj)  # (n, Nx, Ny)
            p_wall = diag - jnp.mean(diag, axis=(1, 2), keepdims=True)
            p_pts = w_jet * (p_wall + 0.5 * w_jet**2) * self._wall_dA / self._power_ref
            per_jet = jnp.einsum("nxy,xyj->nj", p_pts, self._jet_onehot)
        forms = jax.vmap(lambda p: power_forms(p, xp=jnp))(p_pts)
        cf = wss_steps / (0.5 * self.U_BULK**2)
        if self.power_sampling == "mean":
            pick = lambda a: jnp.mean(a, axis=0)  # noqa: E731
        else:
            pick = lambda a: a[-1]  # noqa: E731
        return pick(per_jet), {k: pick(v) for k, v in forms.items()}, pick(cf)

    @property
    def default_params(self) -> ChannelEnvParams:
        """Default ``ChannelEnvParams`` for this environment."""
        return ChannelEnvParams()

    @property
    def name(self) -> str:
        """Name of this environment."""
        return "ChannelFlowSpectralEnv"

    def reset_env(self, key: chex.PRNGKey, params: ChannelEnvParams) -> Tuple[chex.Array, ChannelEnvState]:
        """Reset the channel to the stored initial fields.

        Args:
            key: PRNG key (forwarded to the observation function; the reset
                itself is deterministic).
            params: Channel environment parameters.

        Returns:
            Tuple ``(obs, ChannelEnvState)`` at time 0 with ``terminal=False``.
        """
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
        """Sample the subsampled U/W observation plane at ``k_det``.

        Args:
            state: Current channel environment state.
            params: Channel environment parameters.
            key: Unused; accepted for API compatibility.

        Returns:
            1D observation array (see ``get_obs_spectral_channel``).
        """
        return get_obs_spectral_channel(state, params, self.Xi_obs, self.Yi_obs)

    def is_terminal(self, state: ChannelEnvState, params: ChannelEnvParams) -> jnp.ndarray:
        """Whether the episode has ended (``terminal`` flag or step limit reached).

        Args:
            state: Current channel environment state.
            params: Channel environment parameters.

        Returns:
            Boolean array indicating termination.
        """
        return jnp.logical_or(state.terminal, state.time >= params.max_steps_in_episode)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: ChannelEnvState,
        action: jnp.ndarray,
        params: ChannelEnvParams,
    ) -> Tuple[chex.Array, ChannelEnvState, jnp.ndarray, jnp.ndarray, Dict]:
        """Advance the channel flow by ``params.nsteps`` DNS substeps under the given jet action.

        The action is clipped to the parameter bounds; the reward is ``-wss``,
        the mean wall shear stress of the rollout (with the power term of
        ``power_config``). The step is differentiable with respect to the
        action, e.g. ``jax.grad`` of the reward through ``step_env``.

        Args:
            key: PRNG key (forwarded to the observation function; the dynamics
                are deterministic).
            state: Current channel environment state.
            action: Jet control input (clipped to ``[min_action, max_action]``).
            params: Channel environment parameters.

        Returns:
            Tuple ``(obs, next_state, reward, done, info)`` where ``info``
            contains ``"discount"``.
        """
        action = jnp.clip(action, params.min_action, params.max_action)

        wss, (U1, V1, W1, diag, wss_steps) = self._wss_with_aux_power(state, action, params)

        reward = -wss
        per_jet, forms, cf = self._power_from_rollout(state, action, diag, wss_steps)
        reward = self.power_config.apply(reward, forms)

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

        info = {
            "discount": self.discount(next_state, params),
            **power_info_jax(
                per_jet,
                sampling=self.power_sampling,
                forms=forms,
                drag_coefficient=cf,
                objective={"wss": wss},
            ),
        }
        return obs, next_state, reward, done, info

    def action_space(self, params: Optional[ChannelEnvParams] = None) -> spaces.Box:
        """Return the jet-control action space.

        Args:
            params: Channel environment parameters; defaults to
                ``self.default_params``.

        Returns:
            Gymnax Box of shape ``(params.action_dim,)`` with bounds
            ``[params.min_action, params.max_action]``.
        """
        params = params or self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(params.action_dim,),
        )

    def observation_space(self, params: Optional[ChannelEnvParams] = None) -> spaces.Box:
        """Return the (unbounded) observation space.

        Args:
            params: Channel environment parameters; defaults to
                ``self.default_params``.

        Returns:
            Gymnax Box of shape
            ``(params.obs_subsample**2 * params.obs_include_components,)`` with
            infinite bounds.
        """
        params = params or self.default_params
        obs_dim = params.obs_subsample**2 * params.obs_include_components
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(obs_dim,))

    def _wss_with_aux(
        self,
        state: ChannelEnvState,
        action: jnp.ndarray,
        params: ChannelEnvParams,
    ):
        """Roll the flow forward ``nsteps`` DNS substeps and compute the mean wall shear stress.

        Differentiable through the action: the rollout runs under
        ``jax.checkpoint``, so gradients of the wall shear stress with respect
        to the jet action (e.g. ``jax.grad`` through ``step_env``) need little memory.

        Args:
            state: Current channel environment state (physical-space fields).
            action: Jet control amplitudes applied at every substep.
            params: Channel environment parameters (only ``nsteps`` is read).

        Returns:
            Tuple ``(wss, aux)`` where ``wss`` is the mean wall shear stress of
            the final state and ``aux`` is the final physical-space velocity
            tuple ``(U1, V1, W1)``.
        """
        state0 = self.equation.to_spectral(VelocityState(state.U, state.V, state.W))

        def step_fn(state_hat, n):
            """One checkpointed RK4 substep at the time of ``_substep_time`` (constant forcing ``fx = 2.0``)."""
            t = self._substep_time(state, n)
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

    def _wss_with_aux_power(
        self,
        state: ChannelEnvState,
        action: jnp.ndarray,
        params: ChannelEnvParams,
    ):
        """``_wss_with_aux`` plus, per substep, the power diagnostic (see Returns) and ``tau_w``.

        The state update is the one of ``_wss_with_aux``.

        Returns:
            Tuple ``(wss, (U1, V1, W1, diag, wss_steps))`` with ``wss_steps`` of shape (nsteps,) and the power
            diagnostic ``diag``: wall jets: the bottom-wall pressure, shape (nsteps, Nx, Ny); legacy jets: the
            jet-layer ``w`` before the overwrite, shape (nsteps, Nx, Ny, layers).
        """
        state0 = self.equation.to_spectral(VelocityState(state.U, state.V, state.W))
        z0, z1 = self._jet_layers

        def step_fn(state_hat, n):
            """One checkpointed RK4 substep (as in ``_wss_with_aux``) plus the power diagnostics."""
            t = self._substep_time(state, n)
            fx = 2.0

            new_state, w_pre = self.integrator.rk4_step(
                state_hat,
                dt=state.dt,
                action=action,
                t=t,
                fx=fx,
                target_bulk_u=8.0,
                return_pre_bc=True,
            )
            wss_n = wss_compute(4, self.equation.ifft_xy(new_state.u), z=self.equation.z)
            diag = w_pre[:, :, z0:z1] if self.legacy_jets else w_pre  # legacy: layer w; wall jets: wall pressure
            return new_state, (diag, wss_n)

        step_fn = jax.checkpoint(step_fn)

        nsteps = int(self.default_params.nsteps)
        stateT, (diag, wss_steps) = jax.lax.scan(
            step_fn,
            state0,
            xs=jnp.arange(nsteps),
        )

        state_phys = self.equation.to_physical(stateT)
        U1, V1, W1 = state_phys.u, state_phys.v, state_phys.w

        wss = wss_compute(4, U1, z=self.equation.z)
        return wss, (U1, V1, W1, diag, wss_steps)
